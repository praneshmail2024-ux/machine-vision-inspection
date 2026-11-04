"""Mission Vision web server with an isolated Orbbec/OpenCV worker process."""

from __future__ import annotations

import json
import msvcrt
import multiprocessing as mp
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request

from orbbec_camera import OrbbecCamera, OrbbecCameraError, is_orbbec_physically_connected
from ring_detector import RingDetector


app = Flask(__name__)
runtime: dict[str, Any] | None = None
worker_process: mp.Process | None = None
worker_started_at = 0.0
MAX_JPEG_BYTES = 1_000_000
MAX_DEBUG_JPEG_BYTES = 1_500_000
PROJECT_ROOT = Path(__file__).resolve().parent
BACKGROUND_PATH = PROJECT_ROOT / "data" / "calibration" / "empty_belt_background.jpg"
SETTINGS_PATH = PROJECT_ROOT / "config" / "vision_settings.json"
INSTANCE_LOCK_PATH = PROJECT_ROOT / ".mission_vision.lock"
DEFAULT_SETTINGS = (0.0, 0.03, 0.70, 0.97, 0.50)
STALE_WORKER_SECONDS = 8.0
WORKER_START_GRACE_SECONDS = 12.0
BACKGROUND_MISMATCH_RATIO = 0.55
BACKGROUND_REFRESH_FRAMES = 8


def _acquire_instance_lock():
    """Prevent a second Flask process from opening the same Orbbec device."""
    handle = INSTANCE_LOCK_PATH.open("a+b")
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        return None
    return handle


def _load_saved_settings() -> tuple[float, float, float, float, float]:
    try:
        payload = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        values = tuple(float(payload[key]) for key in ("left", "top", "right", "bottom", "counting_line"))
        left, top, right, bottom, line = values
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("saved field of view is outside the image")
        if right - left < 0.10 or bottom - top < 0.10 or not top <= line <= bottom:
            raise ValueError("saved field of view is invalid")
        return values  # type: ignore[return-value]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return DEFAULT_SETTINGS


def _save_settings(values: tuple[float, float, float, float, float]) -> None:
    payload = dict(zip(("left", "top", "right", "bottom", "counting_line"), values))
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = SETTINGS_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(SETTINGS_PATH)


def _write_jpeg(
    shared: dict[str, Any],
    frame: np.ndarray,
    buffer_key: str = "jpeg",
    length_key: str = "jpeg_length",
) -> None:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return
    data = encoded.tobytes()
    if len(data) > len(shared[buffer_key]):
        return
    with shared[buffer_key].get_lock():
        shared[buffer_key][: len(data)] = data
        shared[length_key].value = len(data)


def _placeholder(message: str) -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, message, (28, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _calibration_frame_contains_rings(
    frame: np.ndarray, settings: tuple[float, ...]
) -> bool:
    """Reject an empty-belt calibration while cardboard remains in the ROI."""
    left, top, right, bottom, _ = settings
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cardboard = cv2.inRange(
        hsv, np.array((4, 8, 70)), np.array((42, 240, 255))
    )
    roi = np.zeros(cardboard.shape, dtype=np.uint8)
    x1, x2 = int(width * left), int(width * right)
    y1, y2 = int(height * top), int(height * bottom)
    roi[y1:y2, x1:x2] = 255
    cardboard = cv2.bitwise_and(cardboard, roi)
    cardboard = cv2.morphologyEx(
        cardboard,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    roi_area = max((x2 - x1) * (y2 - y1), 1)
    cardboard_pixels = cv2.countNonZero(cardboard)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(cardboard)
    largest_component = (
        int(np.max(stats[1:, cv2.CC_STAT_AREA])) if component_count > 1 else 0
    )
    return (
        cardboard_pixels > max(1800, round(roi_area * 0.014))
        or largest_component > max(750, round(roi_area * 0.006))
    )


def camera_worker(shared: dict[str, Any]) -> None:
    """Own all OpenNI objects in a process that receives no web requests."""
    # This 640x480 workload benchmarks faster with eight OpenCV workers than
    # with the twelve logical CPU threads exposed by this laptop.
    cv2.setUseOptimized(True)
    cv2.setNumThreads(8)
    camera = OrbbecCamera()
    # Use almost the full conveyor. The previous 62% right edge could silently
    # discard otherwise-valid rings on the right side of the camera image.
    detector = RingDetector(
        roi_left=0.0,
        roi_top=0.03,
        roi_right=0.70,
        roi_bottom=0.97,
        count_mode="exit",
    )
    saved_background = cv2.imread(str(BACKGROUND_PATH)) if BACKGROUND_PATH.is_file() else None
    if saved_background is not None:
        detector.calibrate_background(saved_background)
        shared["background"].value = 1
    # Preserve the production total if only the native capture worker is replaced.
    detector.total_count = int(shared["total"].value)
    detector.track_crossing_count = detector.total_count
    applied_settings: tuple[float, ...] | None = None
    startup_settings_pending = True
    background_mismatch_frames = 0
    next_start_attempt = 0.0
    _write_jpeg(shared, _placeholder("Orbbec Astra is ready. Click Start Orbbec camera."))

    try:
        while not bool(shared["shutdown"].value):
            wanted = bool(shared["want_camera"].value)
            stream_open = bool(camera.status()["running"])

            if wanted and not stream_open and time.monotonic() >= next_start_attempt:
                try:
                    camera.start()
                    # Opening the OpenNI handle is not proof that the RGB stream works.
                    # The status becomes LIVE only after read_rgb_frame succeeds below.
                    shared["running"].value = 0
                    shared["camera_error"].value = 0
                except OrbbecCameraError:
                    # OpenNI caches its device list for the lifetime of an initialized
                    # runtime. If initialization happened while the USB cable was out,
                    # reusing this object can miss a later hot-plug forever. Fully unload
                    # OpenNI before the next attempt so reconnect works without restarting
                    # the Flask server.
                    camera.close()
                    camera = OrbbecCamera()
                    shared["running"].value = 0
                    shared["camera_error"].value = 1
                    next_start_attempt = time.monotonic() + 1.5
            elif not wanted and stream_open:
                camera.stop()
                shared["running"].value = 0
                shared["last_fresh_frame_at"].value = 0.0

            if stream_open and not camera.is_physically_connected():
                camera.stop()
                shared["running"].value = 0
                shared["camera_error"].value = 2
                shared["last_fresh_frame_at"].value = 0.0
                next_start_attempt = time.monotonic() + 1.5

            with shared["settings"].get_lock():
                current_settings = tuple(shared["settings"][:])
            if current_settings != applied_settings:
                detector.update_settings(*current_settings)
                applied_settings = current_settings
                background_mismatch_frames = 0
                # A saved background is already proof of an empty conveyor.
                # Do not leave production counting locked until the live view
                # happens to become empty again after every app restart.
                if startup_settings_pending and saved_background is not None:
                    detector.arm_counting()
                startup_settings_pending = False

            if shared["reset"].value:
                detector.reset()
                shared["reset"].value = 0

            # Read while the native stream is open even before it is considered LIVE.
            # This first successful read is the startup handshake for the website.
            if bool(camera.status()["running"]) and wanted:
                try:
                    raw = camera.read_rgb_frame()
                    timestamp = camera.last_frame_timestamp
                    # This Astra driver can keep one timestamp across valid frames. A
                    # successful OpenNI read is the reliable heartbeat; USB presence is
                    # checked separately before the webpage reports LIVE.
                    shared["source_timestamp"].value = -1 if timestamp is None else int(timestamp)
                    shared["last_fresh_frame_at"].value = time.monotonic()
                    shared["running"].value = 1
                    shared["camera_error"].value = 0
                    if shared["raw_request"].value:
                        _write_jpeg(shared, raw, "raw_jpeg", "raw_jpeg_length")
                        shared["raw_request"].value = 0
                    if shared["calibrate"].value == 1:
                        if _calibration_frame_contains_rings(raw, current_settings):
                            shared["calibrate"].value = -1
                        else:
                            detector.calibrate_background(raw)
                            saved_background = raw.copy()
                            BACKGROUND_PATH.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(BACKGROUND_PATH), raw)
                            detector.reset()
                            detector.arm_counting()
                            background_mismatch_frames = 0
                            shared["background"].value = 1
                            shared["visible"].value = 0
                            shared["total"].value = 0
                            shared["calibrate"].value = 0
                    output, status = detector.process(raw)
                    # A saved empty-belt image becomes invalid after a major,
                    # persistent lighting or camera-position change. Leaving
                    # that stale reference active marks most of the belt as
                    # foreground and needlessly runs the expensive collapsed-
                    # ring recovery on every frame. Refresh it automatically
                    # only after several consecutive frames are both severely
                    # mismatched and independently proven cardboard-free.
                    if (
                        saved_background is not None
                        and detector.background_change_ratio()
                        >= BACKGROUND_MISMATCH_RATIO
                        and not _calibration_frame_contains_rings(
                            raw, current_settings
                        )
                    ):
                        background_mismatch_frames += 1
                    else:
                        background_mismatch_frames = 0
                    if background_mismatch_frames >= BACKGROUND_REFRESH_FRAMES:
                        detector.calibrate_background(raw)
                        saved_background = raw.copy()
                        BACKGROUND_PATH.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(BACKGROUND_PATH), raw)
                        background_mismatch_frames = 0
                        shared["background"].value = 1
                        # Reprocess the refresh frame so the dashboard switches
                        # to the fast, clean-background result immediately.
                        output, status = detector.process(raw)
                    shared["visible"].value = int(status["visible"])
                    shared["total"].value = int(status["total"])
                    shared["processing_fps"].value = float(status["processing_fps"])
                    _write_jpeg(shared, output)
                    # Build the six-panel diagnostic only while the browser
                    # requests it. The production view should not spend CPU on
                    # a hidden settings panel.
                    if shared["debug_request"].value:
                        debug_mosaic = detector.debug_mosaic(raw, output)
                        _write_jpeg(shared, debug_mosaic, "debug_jpeg", "debug_jpeg_length")
                        shared["debug_request"].value = 0
                except OrbbecCameraError:
                    # A single USB/OpenNI hiccup must not require a page reload or another
                    # button click. Release the native handle and retry while Start is wanted.
                    camera.close()
                    camera = OrbbecCamera()
                    shared["running"].value = 0
                    shared["camera_error"].value = 1
                    shared["last_fresh_frame_at"].value = 0.0
                    next_start_attempt = time.monotonic() + 1.5
            else:
                shared["visible"].value = 0
                message = (
                    "Starting Orbbec Astra: waiting for a real RGB frame."
                    if wanted else "Orbbec Astra is offline. Click Start Orbbec camera."
                )
                _write_jpeg(shared, _placeholder(message))
                time.sleep(0.08)
    finally:
        camera.close()


def _new_runtime() -> dict[str, Any]:
    return {
        "jpeg": mp.Array("B", MAX_JPEG_BYTES, lock=True),
        "jpeg_length": mp.Value("i", 0),
        "debug_jpeg": mp.Array("B", MAX_DEBUG_JPEG_BYTES, lock=True),
        "debug_jpeg_length": mp.Value("i", 0),
        "debug_request": mp.Value("i", 0),
        "raw_jpeg": mp.Array("B", MAX_JPEG_BYTES, lock=True),
        "raw_jpeg_length": mp.Value("i", 0),
        "raw_request": mp.Value("i", 0),
        # Start automatically when the known Orbbec USB device is present. If it
        # is unplugged, the worker keeps retrying safely after reconnection.
        "want_camera": mp.Value("i", 1),
        "running": mp.Value("i", 0),
        "camera_error": mp.Value("i", 0),
        "visible": mp.Value("i", 0),
        "total": mp.Value("i", 0),
        "processing_fps": mp.Value("d", 0.0),
        "last_fresh_frame_at": mp.Value("d", 0.0),
        "source_timestamp": mp.Value("q", -1),
        "background": mp.Value("i", int(BACKGROUND_PATH.is_file())),
        "worker_restarts": mp.Value("i", 0),
        "force_restart": mp.Value("i", 0),
        "settings": mp.Array("d", _load_saved_settings(), lock=True),
        # Shared integer flags are reliable with Windows' spawned worker process.
        # multiprocessing.Event caused the worker to crash intermittently here.
        "reset": mp.Value("i", 0),
        "calibrate": mp.Value("i", 0),
        "shutdown": mp.Value("i", 0),
    }


def _shared() -> dict[str, Any]:
    if runtime is None:
        raise RuntimeError("Camera runtime has not been initialized.")
    return runtime


def _start_camera_worker(shared: dict[str, Any]) -> None:
    """Start a fresh OpenNI process; only this process ever touches the camera SDK."""
    global worker_process, worker_started_at
    worker_process = mp.Process(target=camera_worker, args=(shared,), name="orbbec-worker", daemon=True)
    worker_process.start()
    worker_started_at = time.monotonic()


def _camera_watchdog(shared: dict[str, Any]) -> None:
    """Recover dead workers and native frame reads that remain blocked."""
    was_connected = is_orbbec_physically_connected()
    while not bool(shared["shutdown"].value):
        process = worker_process
        now = time.monotonic()
        requested = bool(shared["want_camera"].value)
        connected = is_orbbec_physically_connected()
        reconnected = connected and not was_connected
        forced = bool(shared["force_restart"].value)
        shared["force_restart"].value = 0
        fresh_at = float(shared["last_fresh_frame_at"].value)
        past_start_grace = now - worker_started_at > WORKER_START_GRACE_SECONDS
        frame_stalled = (
            process is not None
            and process.is_alive()
            and requested
            and connected
            and past_start_grace
            and (fresh_at <= 0 or now - fresh_at > STALE_WORKER_SECONDS)
        )
        clean_restart = requested and connected and (forced or reconnected)
        if frame_stalled or clean_restart:
            # OpenNI read_frame can block forever without raising. Because OpenNI
            # is isolated, terminating this worker safely releases that handle.
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            shared["worker_restarts"].value += 1
            process = None
        if not connected:
            # An unplug is expected operational state, not a stalled worker. Keep
            # one idle worker alive and stop the endless 12-second restart cycle.
            shared["running"].value = 0
            shared["last_fresh_frame_at"].value = 0.0
            shared["camera_error"].value = 2
            shared["visible"].value = 0
        if process is None or not process.is_alive():
            shared["running"].value = 0
            shared["last_fresh_frame_at"].value = 0.0
            shared["camera_error"].value = 1
            shared["visible"].value = 0
            _start_camera_worker(shared)
        was_connected = connected
        time.sleep(2.0)


def _jpeg_response(buffer_key: str = "jpeg", length_key: str = "jpeg_length") -> Response:
    shared = _shared()
    fresh = shared["last_fresh_frame_at"].value > 0 and time.monotonic() - shared["last_fresh_frame_at"].value < 2.0
    if not is_orbbec_physically_connected() or not fresh:
        frame = _placeholder("Orbbec Astra offline or video stream stalled. Reconnect, then click Start Camera.")
        ok, encoded = cv2.imencode(".jpg", frame)
        return Response(encoded.tobytes() if ok else b"", mimetype="image/jpeg", headers={"Cache-Control": "no-store, max-age=0"})
    with shared[buffer_key].get_lock():
        length = shared[length_key].value
        data = bytes(shared[buffer_key][:length])
    if not data:
        return jsonify({"ok": False, "message": "No camera frame is ready."}), 503
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video_feed")
def video_feed():
    return _jpeg_response()


@app.get("/api/frame.jpg")
def current_frame():
    return _jpeg_response()


@app.get("/api/debug.jpg")
def debug_frame():
    shared = _shared()
    if shared["running"].value:
        shared["debug_request"].value = 1
        deadline = time.monotonic() + 2.0
        while shared["debug_request"].value and time.monotonic() < deadline:
            time.sleep(0.01)
    return _jpeg_response("debug_jpeg", "debug_jpeg_length")


@app.get("/api/raw.jpg")
def raw_frame():
    """Return one stabilized, unannotated worker frame for detector diagnostics."""
    shared = _shared()
    if not shared["running"].value:
        return jsonify({"ok": False, "message": "Camera is not running."}), 503
    shared["raw_request"].value = 1
    deadline = time.monotonic() + 2.0
    while shared["raw_request"].value and time.monotonic() < deadline:
        time.sleep(0.01)
    return _jpeg_response("raw_jpeg", "raw_jpeg_length")


@app.get("/api/status")
def status():
    shared = _shared()
    connected = is_orbbec_physically_connected()
    requested = bool(shared["want_camera"].value)
    fresh = shared["last_fresh_frame_at"].value > 0 and time.monotonic() - shared["last_fresh_frame_at"].value < 2.0
    running = bool(shared["running"].value) and connected and fresh
    if not connected:
        message = "Orbbec Astra was disconnected. Reconnect it, then click Start Orbbec camera."
    elif not requested:
        message = "Orbbec Astra stream is stopped."
    elif requested and not running:
        message = "Orbbec Astra is starting or reconnecting; waiting for a real RGB frame."
    elif not fresh:
        message = "Orbbec Astra video stream is stalled. Reconnect it, then click Start Orbbec camera."
    elif running:
        message = "Orbbec Astra connected: Astra (640x480 @ 30 FPS)."
    elif shared["camera_error"].value == 2:
        message = "Orbbec Astra was disconnected. Reconnect it, then click Start Orbbec camera."
    elif shared["camera_error"].value:
        message = "Orbbec Astra could not start. Check its USB connection."
    else:
        message = "Orbbec Astra stream is stopped."
    return jsonify({
        "running": running,
        "device_present": connected,
        "frame_fresh": fresh,
        "camera": "Orbbec Astra",
        "stream_mode": "640x480 @ 30 FPS" if running else "not running",
        "message": message,
        "visible": shared["visible"].value,
        "total": shared["total"].value,
        "processing_fps": round(shared["processing_fps"].value, 1),
        "background_calibrated": bool(shared["background"].value),
        "camera_restarts": int(shared["worker_restarts"].value),
        "count_mode": "verified_line_with_exit_fallback",
    })


@app.get("/api/stats")
def stats():
    return status()


@app.get("/api/cameras")
def cameras():
    return jsonify([status().get_json()])


@app.post("/api/camera/start")
def start_camera():
    shared = _shared()
    if not is_orbbec_physically_connected():
        return jsonify({"ok": False, "message": "Orbbec Astra is disconnected. Reconnect it before starting."}), 503
    fresh = shared["last_fresh_frame_at"].value > 0 and time.monotonic() - shared["last_fresh_frame_at"].value < 2.0
    if shared["running"].value and fresh:
        return jsonify({"ok": True, "message": "Orbbec Astra connected: Astra (640x480 @ 30 FPS)."})
    was_requested = bool(shared["want_camera"].value)
    shared["camera_error"].value = 0
    shared["last_fresh_frame_at"].value = 0.0
    # A normal Stop -> Start can reuse the healthy idle worker. Force replacement
    # only when streaming was already requested but no frames were arriving.
    if was_requested:
        shared["force_restart"].value = 1
    shared["want_camera"].value = 1
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        fresh = shared["last_fresh_frame_at"].value > 0 and time.monotonic() - shared["last_fresh_frame_at"].value < 2.0
        if shared["running"].value and fresh:
            return jsonify({"ok": True, "message": "Orbbec Astra connected: Astra (640x480 @ 30 FPS)."})
        time.sleep(0.05)
    return jsonify({"ok": False, "message": "No RGB frame arrived yet. The camera will keep retrying automatically."}), 503


@app.post("/api/camera/stop")
def stop_camera():
    shared = _shared()
    shared["want_camera"].value = 0
    return jsonify({"ok": True, "message": "Orbbec Astra stream stopped."})


@app.post("/api/count/reset")
def reset_count():
    shared = _shared()
    shared["reset"].value = 1
    shared["total"].value = 0
    return jsonify({"ok": True, "message": "Ring count reset.", "total": 0})


@app.post("/api/background/calibrate")
def calibrate_background():
    shared = _shared()
    if not shared["running"].value:
        return jsonify({"ok": False, "message": "Start the camera before calibrating."}), 503
    shared["calibrate"].value = 1
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and shared["calibrate"].value == 1:
        time.sleep(0.03)
    result = int(shared["calibrate"].value)
    if result == -1:
        shared["calibrate"].value = 0
        return jsonify({
            "ok": False,
            "message": (
                "Calibration refused: rings or cardboard are still inside the "
                "inspection area. Clear the belt, then calibrate again."
            ),
        }), 409
    if result == 1:
        return jsonify({
            "ok": False,
            "message": "Calibration timed out before a fresh camera frame arrived.",
        }), 504
    return jsonify({"ok": True, "message": "Empty-belt background saved; tracks and count reset to zero."})


@app.get("/api/settings")
def get_settings():
    shared = _shared()
    with shared["settings"].get_lock():
        left, top, right, bottom, line = shared["settings"][:]
    return jsonify({"left": left * 100, "top": top * 100, "right": right * 100, "bottom": bottom * 100, "counting_line": line * 100})


@app.post("/api/settings")
def update_settings():
    payload = request.get_json(silent=True) or {}
    try:
        values = tuple(float(payload[key]) / 100 for key in ("left", "top", "right", "bottom", "counting_line"))
        left, top, right, bottom, line = values
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("Field-of-view values must be between 0 and 100 percent.")
        if right - left < 0.10 or bottom - top < 0.10 or not top <= line <= bottom:
            raise ValueError("Invalid detection area or counting-line position.")
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"ok": False, "message": str(error)}), 400
    shared = _shared()
    with shared["settings"].get_lock():
        shared["settings"][:] = values
    try:
        _save_settings(values)
    except OSError:
        return jsonify({"ok": False, "message": "Settings were applied but could not be saved to disk."}), 500
    return jsonify({"ok": True, "message": "Detection area and counting line updated."})


@app.get("/<path:unknown_path>")
def dashboard_fallback(unknown_path: str):
    if unknown_path.startswith("api/"):
        return jsonify({"ok": False, "message": "Unknown API endpoint."}), 404
    return redirect("/", code=302)


if __name__ == "__main__":
    mp.freeze_support()
    instance_lock = _acquire_instance_lock()
    if instance_lock is None:
        raise SystemExit(
            "Mission Vision is already running at http://127.0.0.1:5000/"
        )
    runtime = _new_runtime()
    # Keep the native OpenNI driver separate from Flask. A driver fault can then
    # never terminate the website or leave a stale LIVE indicator behind.
    _start_camera_worker(runtime)
    threading.Thread(target=_camera_watchdog, args=(runtime,), name="camera-watchdog", daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
