# Hand Landmarks C++ Demo

![Hand Landmarks C++ Demo](assets/hand-landmarks-sc.png)

Tracks 21 hand landmarks per hand from a camera or a video. A palm detector finds hands and a landmark model predicts the keypoints; both run asynchronously on the NPU and a Qt window draws the result.

## What you need

- A supported DEEPX NPU with DX-RT installed (Tutorial 01)
- A graphical desktop session for the Qt window
- A V4L2 camera for the camera demo (optional)

## Run the tutorial

Everything else (package installation, resource download, build, run commands, options, and troubleshooting) is in the notebook. Start JupyterLab from the repository root and open it:

```bash
./run-jupyter-lab.sh
```

Then open `notebooks/T23-demo-hand-landmarks/hand_landmarks.ipynb` in JupyterLab and run the cells from the top.

## Project layout

```text
T23-demo-hand-landmarks/
├── hand_landmarks.ipynb      # the tutorial
├── README.md
├── get_resources.sh    # downloads the models and sample videos into assets/
├── app/                # C++ sources, CMakeLists.txt, build.sh, run_camera.sh, run_video.sh
└── assets/             # models/ and videos/ (downloaded, ignored by git)
```
