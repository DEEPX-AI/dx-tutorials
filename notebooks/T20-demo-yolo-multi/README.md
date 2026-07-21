# Tutorial 20: YOLO Multi-Channel C++ Demo

This tutorial explains how to build and run a multi-channel YOLO object detection demo on a DEEPX NPU.

## Project layout

```text
T20-demo-yolo-multi/
├── yolo_multi.ipynb          # Tutorial and executable examples
├── README.md                 # Quick-start guide
├── get_resources.sh         # Model and video downloader
├── assets/
│   ├── models/              # DXNN model files
│   └── videos/              # Sample video files
└── app/
    ├── build.sh             # Release build script
    ├── run_camera.sh        # 33-channel camera demo
    ├── run_video.sh         # 36-channel video demo
    ├── CMakeLists.txt
    ├── config/              # Demo configuration files
    ├── include/             # Application headers
    ├── src/                 # C++ sources
    ├── lib/                 # Shared demo utilities
    ├── extern/              # Header-only third-party libraries
    └── sample/              # Fonts and UI images
```

## Prerequisites

- A supported DEEPX NPU
- The DEEPX device driver and DXRT SDK
- A graphical desktop session for the OpenCV output window
- A V4L2 camera at `/dev/video0` for the camera demo

Install the required Debian or Ubuntu packages:

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    ca-certificates \
    curl \
    tar \
    libopencv-dev \
    libopencv-contrib-dev \
    libfreetype-dev \
    ffmpeg \
    v4l-utils \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav
```

Verify that DXRT can see the NPU:

```bash
dxrt-cli -s
```

## 1. Download the model and videos

Run the resource script from the tutorial directory:

```bash
cd notebooks/T20-demo-yolo-multi
./get_resources.sh
```

The script downloads `yolo-multi.tar.gz`, extracts it into `assets/`, and removes the archive after successful extraction.

Expected files include:

```text
assets/
├── models/YOLOV5S_PPU.dxnn
└── videos/*.mp4
```

## 2. Build the C++ application

```bash
cd app
./build.sh
```

The script configures a Release build and uses all available CPU cores. The executable is created at:

```text
app/build/yolo_multi_demo
```

Use a clean build when needed:

```bash
./build.sh --clean
```

## 3. Run the video demo

Run all application scripts from `app/` because the configuration files use paths relative to that directory.

```bash
cd notebooks/T20-demo-yolo-multi/app
./run_video.sh
```

This starts 36 video channels in a 6 x 6 grid using `config/ppu_yolo_multi_36_video.json`.

## 4. Run the camera demo

Check the camera first:

```bash
v4l2-ctl --list-devices
ls -l /dev/video0
```

Then run:

```bash
cd notebooks/T20-demo-yolo-multi/app
./run_camera.sh
```

This starts one camera channel and 32 video channels using `config/ppu_yolo_multi_33_camera.json`. The camera channel is highlighted in the grid.

## Controls

- `Esc` or `q`: exit the demo
- `t`: show or hide detection boxes
- `EXIT` button: exit when the button overlay is available

## Jupyter tutorial

Start JupyterLab from the repository root:

```bash
./run-jupyter-lab.sh
```

Open `notebooks/T20-demo-yolo-multi/yolo_multi.ipynb` and run the cells in order.

## Troubleshooting

### CMake cannot find DXRT

Confirm that the DXRT headers, CMake package files, and shared libraries are installed and visible to CMake.

### The model or a video cannot be opened

Run `get_resources.sh` again and confirm that the paths under `assets/` match the selected JSON configuration.

### The camera cannot be opened

Confirm that `/dev/video0` exists and that the current user has permission to access it. The user may need to be added to the `video` group.

### The output window does not appear

The demo uses an OpenCV GUI window. A local desktop, remote desktop, or correctly configured X11 forwarding session is required.

