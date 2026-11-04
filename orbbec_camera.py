"""OpenNI2 adapter for the Orbbec Astra RGB stream."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from openni import openni2

RUNTIME_DIR = Path(__file__).resolve().parent / "vendor" / "orbbec_runtime"
_presence_lock = threading.Lock()
_presence_checked_at = 0.0
_presence_value = False


def is_orbbec_physically_connected() -> bool:
    """Return Windows' current USB state, cached briefly for the live webpage."""
    global _presence_checked_at, _presence_value
    with _presence_lock:
        now = time.monotonic()
        if now - _presence_checked_at < 0.75:
            return _presence_value
        _presence_checked_at = now
        try:
            result = subprocess.run(
                ["pnputil", "/enum-devices", "/connected"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            devices = (result.stdout + result.stderr).lower()
            _presence_value = "vid_2bc5" in devices or "orbbec" in devices
        except (OSError, subprocess.SubprocessError):
            # Do not declare an unplug merely because the Windows command is unavailable.
            pass
        return _presence_value


class OrbbecCameraError(RuntimeError):
    """A friendly error for camera setup and frame-capture failures."""


class OrbbecCamera:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._device = None
        self._color_stream = None
        self._initialized = False
        self._dll_directory = None
        self._message = "Orbbec Astra is not connected."
        self._stream_mode = "not running"
        self.last_frame_timestamp: int | None = None
        self._last_presence_check = 0.0
        self._physical_present = False

    def _enable_automatic_colour_controls(self) -> None:
        """Enable Astra colour controls both before and after stream start."""
        if self._color_stream is None or self._color_stream.camera is None:
            return
        try:
            self._color_stream.camera.auto_exposure = True
        except Exception:
            pass
        try:
            self._color_stream.camera.auto_white_balance = True
        except Exception:
            pass

    def _warm_up_exposure(self) -> None:
        """Discard startup frames until auto exposure is visibly usable."""
        if self._color_stream is None:
            return
        deadline = time.monotonic() + 2.5
        bright_frames = 0
        frame_attempts = 0
        while time.monotonic() < deadline and frame_attempts < 60:
            frame = self._color_stream.read_frame()
            frame_attempts += 1
            pixels = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8)
            pixel_count = frame.width * frame.height
            if len(pixels) == pixel_count * 3:
                brightness = float(pixels[::12].mean())
            elif len(pixels) == pixel_count * 2:
                # YUYV stores luminance in every even byte.
                brightness = float(pixels[0::8].mean())
            else:
                brightness = 0.0
            if brightness >= 35.0:
                bright_frames += 1
                if bright_frames >= 3:
                    return
            else:
                bright_frames = 0
                # Some drivers retain the flag but do not restart their control
                # loop until it is explicitly written again.
                if frame_attempts in (8, 20):
                    self._enable_automatic_colour_controls()

    def is_physically_connected(self) -> bool:
        """Check Windows' connected-device list, not only OpenNI's cached stream handle."""
        self._physical_present = is_orbbec_physically_connected()
        return self._physical_present

    def _initialize(self) -> None:
        if self._initialized:
            return
        if not (RUNTIME_DIR / "OpenNI2.dll").is_file():
            raise OrbbecCameraError("OpenNI2 runtime is missing from vendor/orbbec_runtime.")

        # Allows Windows to resolve OpenNI2.dll and the Orbbec driver DLL.
        self._dll_directory = os.add_dll_directory(str(RUNTIME_DIR))
        openni2.initialize(str(RUNTIME_DIR))
        self._initialized = True

    def start(self) -> str:
        with open('worker_debug.log', 'a') as f: f.write("Entering OrbbecCamera.start()\n")
        with self._lock:
            if self._color_stream is not None:
                return "Orbbec Astra RGB stream is already running."
            try:
                with open('worker_debug.log', 'a') as f: f.write("Calling _initialize()\n")
                self._initialize()
                with open('worker_debug.log', 'a') as f: f.write("Calling Device.open_any()\n")
                self._device = openni2.Device.open_any()
                with open('worker_debug.log', 'a') as f: f.write("Calling create_color_stream()\n")
                self._color_stream = self._device.create_color_stream()
                try:
                    self._color_stream.set_video_mode(
                        openni2.VideoMode(
                            pixelFormat=openni2.PIXEL_FORMAT_RGB888,
                            resolutionX=640,
                            resolutionY=480,
                            fps=30,
                        )
                    )
                except Exception:
                    pass
                with open('worker_debug.log', 'a') as f: f.write("Starting color stream\n")
                self._enable_automatic_colour_controls()
                self._color_stream.start()
                self._enable_automatic_colour_controls()
                with open('worker_debug.log', 'a') as f: f.write("Warming up exposure\n")
                self._warm_up_exposure()
                with open('worker_debug.log', 'a') as f: f.write("Warm up done\n")
            except Exception as error:
                with open('worker_debug.log', 'a') as f: f.write(f"Exception in start: {error}\n")
                if self._color_stream is not None:
                    try:
                        self._color_stream.stop()
                    except Exception:
                        pass
                self._color_stream = None
                self._device = None
                self._message = "No Orbbec Astra detected. Connect it by USB, then start the stream."
                raise OrbbecCameraError(self._message) from error

            info = self._device.get_device_info()
            mode = self._color_stream.get_video_mode()
            self._stream_mode = f"{mode.resolutionX}x{mode.resolutionY} @ {mode.fps} FPS"
            name = info.name.decode() if isinstance(info.name, bytes) else info.name
            self._message = f"Orbbec Astra connected: {name} ({self._stream_mode})."
            with open('worker_debug.log', 'a') as f: f.write("Start success!\n")
            return self._message

    def read_rgb_frame(self) -> np.ndarray:
        with self._lock:
            if self._color_stream is None:
                raise OrbbecCameraError("Orbbec Astra stream is offline. Click Start Orbbec camera.")
            try:
                frame = self._color_stream.read_frame()
                self.last_frame_timestamp = getattr(frame, "timestamp", None)
                data = frame.get_buffer_as_uint8()
                pixels = np.frombuffer(data, dtype=np.uint8)
                if len(pixels) == frame.width * frame.height * 3:
                    rgb = pixels.reshape(frame.height, frame.width, 3)
                    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if len(pixels) == frame.width * frame.height * 2:
                    yuyv = pixels.reshape(frame.height, frame.width, 2)
                    return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
                raise ValueError(f"Unexpected colour-frame size: {len(pixels)} bytes")
            except Exception as error:
                self._message = "Orbbec Astra frame could not be read. Check the USB cable."
                raise OrbbecCameraError(self._message) from error

    def stop(self) -> None:
        with self._lock:
            if self._color_stream is not None:
                self._color_stream.stop()
            self._color_stream = None
            self._device = None
            self._message = "Orbbec Astra stream is stopped."
            self._stream_mode = "not running"
            self.last_frame_timestamp = None

    def close(self) -> None:
        self.stop()
        if self._initialized:
            try:
                openni2.unload()
            finally:
                self._initialized = False

    def status(self) -> dict[str, str | bool]:
        # Status is requested frequently by several browser tabs. It must use the same lock
        # as OpenNI start/read/stop so Python never touches the native stream wrapper while
        # a frame call is active on the capture worker.
        with self._lock:
            return {
                "running": self._color_stream is not None,
                "camera": "Orbbec Astra",
                "stream_mode": self._stream_mode,
                "message": self._message,
            }
