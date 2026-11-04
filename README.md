# Mission Vision

Local Orbbec Astra inspection and ring-counting system built with Flask, OpenNI2, and OpenCV. It is deliberately configured not to use the laptop webcam.

## Project structure

```text
mision vision/
├── app.py                     Flask server and camera worker
├── ring_detector.py           Detection, tracking, and counting logic
├── orbbec_camera.py           Orbbec/OpenNI2 camera adapter
├── config/                    Saved field-of-view settings
├── data/
│   ├── calibration/           Empty-belt background image
│   ├── frames/                Extracted recording frames
│   └── samples/               Detector sample images
├── scripts/                   Recording-analysis utilities
├── tests/                     Detector regression tests
├── templates/                 Dashboard HTML
├── static/                    Dashboard CSS and JavaScript
├── vendor/orbbec_runtime/     Bundled OpenNI2 runtime
├── artifacts/                 Generated diagnostics, replays, and contact sheets
├── docs/images/               Documentation screenshots
└── logs/                      Local server logs
```

## Run the dashboard

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) and click **Start Camera** if the stream is stopped. Install the Orbbec OpenNI2 runtime separately under `vendor/orbbec_runtime/` using the vendor-provided package. Vendor driver binaries are not included in this source export.

If the page reports that the camera is missing, connect the Astra directly to a USB 2.0/3.0 port and wait for Windows Device Manager to show an Orbbec device. The application never falls back to the laptop webcam.

## Detection safeguards

The production path combines background/material masking, contour hierarchy,
ellipse-shape validation, circular recovery for partially overlapping rings,
temporal confirmation, and direction-aware tracking. Watershed is kept as a
diagnostic view and cannot independently add to the count. Belt joints, short
edge fragments, static objects on the counting line, and reverse-moving objects
are covered by regression tests.

Use **Calibrate Empty Belt** only while the belt is empty. A completely hidden
ring cannot be distinguished from one ring in a single RGB image; that case
requires Orbbec depth/thickness evidence or another sensor. Partially visible,
touching, collapsed, irregular, and overlapping rings are handled by the current
detector.

## Run detector tests

```powershell
python -m tests.verify_detector_cases
```

## Analyze a recording

```powershell
python scripts/analyze_recording.py path\to\recording.mp4 --replay
```

Generated contact sheets and replay videos are written under `artifacts/` by default.

## Source export

Prepared from the existing Codex project on 4 November 2026. Private camera settings, calibration backgrounds, server logs, raw recordings, and vendor driver binaries are excluded. Camera capture requires compatible Orbbec hardware; no laptop webcam fallback is used by this version. Saved example images document prior local tests and do not establish general accuracy.
