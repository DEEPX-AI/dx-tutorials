# Tutorial 23: Hand Landmarks Demo

This tutorial builds and runs a Qt-based C++ application that tracks 21 landmarks on each detected hand. It accepts camera or video input and runs palm detection followed by hand landmark inference on a DEEPX NPU.

![hand landmarks demo](assets/hands-landmarks-sc.png)

## Model provenance

The palm-detection and hand-landmark models used by this demo originate from the [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) model bundle provided by Google AI Edge. The models were converted to DXNN format for DEEPX NPU inference and are used as `hand-detector_192x192.dxnn` and `HandLandmarkLite.dxnn`.

## How it works

```text
Camera or video frame
        |
        v
Palm detector (192 x 192) -> palm boxes and rotated hand regions
        |
        v
Hand crop (224 x 224) -> landmark model -> 21 points and handedness
        |
        v
Rendered landmarks, labels, and performance metrics in a Qt window
```

Only the palm and hand-landmark stages are included in this application.

## Project layout

```text
T23-demo-hand-landmarks/
├── README.md
├── hand_landmarks.ipynb
├── get_resources.sh          # Empty resource-download placeholder
├── assets/
│   ├── models/               # Palm and hand-landmark models
│   └── videos/               # Video input files
└── app/
    ├── CMakeLists.txt
    ├── build.sh
    ├── run_camera.sh
    ├── run_video.sh
    └── hand_landmarks.cpp
```

## Prerequisites

- A supported DEEPX NPU
- The DEEPX device driver and DXRT SDK
- A graphical desktop session for the Qt window
- A V4L2 camera for the camera demo

Install the required Debian or Ubuntu packages:

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    qtbase5-dev \
    ffmpeg \
    v4l-utils
```

Verify that DXRT can see the NPU:

```bash
dxrt-cli -s
```

## 1. Prepare resources

`get_resources.sh` is intentionally empty. Add its download logic later or place the files manually at these paths:

```text
assets/
├── models/
│   ├── hand-detector_192x192.dxnn
│   └── HandLandmarkLite.dxnn
└── videos/
    └── hands.mp4
```

The palm model must accept a UINT8 `[1, 192, 192, 3]` tensor. The landmark model must accept a UINT8 `[1, 224, 224, 3]` tensor.

## 2. Build the application

```bash
cd notebooks/T23-demo-hand-landmarks/app
./build.sh
```

The script configures a Release build and uses all available CPU cores. The executable is created at `app/build/hand_landmarks`.

Use a clean build when needed:

```bash
./build.sh --clean
```

## 3. Run with a camera

```bash
cd notebooks/T23-demo-hand-landmarks/app
./run_camera.sh
```

The default camera index is `0`, with a requested size of 1280 x 720 at 30 FPS. Override these values as needed:

```bash
./run_camera.sh --camera 2 --width 1920 --height 1080 --fps 30
```

## 4. Run with a video

```bash
cd notebooks/T23-demo-hand-landmarks/app
./run_video.sh
```

The script uses `assets/videos/hands.mp4` and loops the video. To use another file, run the executable directly:

```bash
./build/hand_landmarks --video /path/to/input.mp4 --loop --landmark-only
```

## Useful options

- `--max-hands N`: set the maximum number of hands per frame; the default is 4.
- `--palm-conf VALUE`: set the palm confidence threshold; the default is 0.2.
- `--landmark-conf VALUE`: set the landmark presence threshold; the default is 0.5.
- `--show-palm`: also draw palm boxes and rotated regions.
- `--landmark-only`: draw only the hand landmarks.
- `--windowed`: use a 1280 x 720 window instead of full screen.
- `--save`: save the rendered result as the next available `output-XX.mp4` file in `app/`.

Run `./build/hand_landmarks --help` for the complete option list. Press `Esc` or `Q` to quit and `F` to toggle full-screen mode.

## Jupyter tutorial

Start JupyterLab from the repository root:

```bash
./run-jupyter-lab.sh
```

Open `notebooks/T23-demo-hand-landmarks/hand_landmarks.ipynb` and run the cells in order.

## Troubleshooting

- **A model file is not found:** Check both filenames and their locations under `assets/models/`.
- **The model shape is rejected:** Confirm the input tensor shapes and UINT8 data type shown above.
- **The camera cannot be opened:** Run `v4l2-ctl --list-devices` and select another camera index.
- **The Qt window does not appear:** Use a graphical desktop session and check the `DISPLAY` environment variable.
- **CMake cannot find DXRT:** Set `DXRT_INSTALLED_DIR` to the DXRT installation prefix and run `./build.sh --clean`.
