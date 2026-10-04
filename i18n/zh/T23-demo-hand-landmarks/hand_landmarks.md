<!-- i18n source: notebooks/T23-demo-hand-landmarks/hand_landmarks.ipynb -->
<!-- i18n lang: zh -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: title src: 08f7a63301 -->
# DEEPX Tutorial 23 - 手部关键点 C++ 演示

本笔记本讲解、构建并运行一个 C++ 应用程序,它为每只检测到的手跟踪 21 个关键点。该应用接受摄像头或视频输入,并在两个推理阶段都使用 DEEPX NPU。

<!-- cell: 169e124a src: c366966d31 -->
![手部关键点演示](assets/hand-landmarks-sc.png)

<!-- cell: 99f74090-1591-4f3c-83d3-c7911115ef61 src: e791a70079 -->
## 学习目标

完成本教程后,您将能够:

- 说明手掌检测与手部关键点两阶段流水线;
- 下载演示所需的模型和示例视频;
- 使用 `dxparse` 检查两个 DXNN 模型;
- 构建 Qt 应用程序并在摄像头或视频上运行;
- 通过命令行调整检测阈值和显示选项。

<!-- cell: t23-npu-pattern src: 1878462df3 -->
## 本应用如何使用 NPU

| 项目 | 在本教程中 | 学习来源 |
|---|---|---|
| 引擎 | 每帧依次运行两个 `InferenceEngine` 对象:先运行一次手掌检测器,然后对每只检测到的手运行一次关键点模型 | T06-2 §4 |
| 执行 | 手掌检测同步运行;每只手的关键点请求通过 `RunAsync()` 提交,并在回调中映射回帧坐标,因此多只手的推理会在 NPU 上重叠执行 | T06-2 §5 |
| 任务图 | 手掌检测器以一个 `cpu_0` 任务结尾(通过 ONNX Runtime 进行锚框解码);关键点模型为仅 NPU | T06-1 §6 |
| 每次请求的输入 | `[1, 192, 192, 3]` UINT8(110 KB)和 `[1, 224, 224, 3]` UINT8(150 KB):输入很小,因此主机侧的裁剪与旋转占主导 | T06-3 §4 |
| 测量内容 | 将 4.1 节中的两个 `dxrun` 基线与窗口中显示的 FPS 进行比较;使用 `--max-hands` 可限制每帧的关键点请求数量 | T06-1 §5 |

<!-- cell: pipeline src: 148af37d15 -->
## 1. 处理流水线

该应用程序仅包含手部流水线:

```text
Camera or video frame
        |
        v
Palm detector, 192 x 192
        |
        v
Palm box and rotated hand region
        |
        v
Hand crop, 224 x 224 -> landmark model
        |
        v
21 landmarks and handedness -> Qt GUI
```

手掌检测首先找到手部区域。每个区域经过旋转、裁剪后传给关键点模型。结果包括 21 个图像空间点、世界坐标关键点、手部存在置信度以及左右手判别。

<!-- cell: prerequisites src: 21ce181bb7 -->
## 2. 前提条件

- 已完成教程 01:已安装 DX-RT,且 DEEPX NPU 以 `/dev/dxrt0` 的形式可见(用 `dxcli -s` 检查);Qt 窗口需要 `cmake`、`g++` 和 `qtbase5-dev`。
- 图形桌面会话;摄像头演示需要一个 V4L2 摄像头(可选)。
- 下载量:一个约 10 MB 的压缩包(两个模型和示例视频),解压到被 git 忽略的 `assets/` 中。
- 耗时:约 15 分钟;构建约需一分钟。除非缺少软件包需要运行下面的 `apt` 命令,否则不需要 `sudo`。
- 已在 DX-RT 3.4.2 上验证。模型由 DX-COM 2.3.0 编译。

使用以下命令安装 C++ 构建和 GUI 依赖:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev qtbase5-dev ffmpeg v4l-utils
```

下一个单元格会定位 SDK、检查这些要求并打印状态表。任何标记为 `MISSING` 的项目都会附上提供它的步骤。

<!-- cell: layout-heading src: e62a57c223 -->
## 3. 项目结构

应用程序代码和脚本位于 `app/` 下。模型和视频保存在构建树之外的 `assets/` 下。

<!-- cell: resources src: 46ee82775e -->
## 4. 下载并检查资源

两个推理模型来源于 Google AI Edge 提供的 [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) 模型包。其手掌检测模型和手部关键点模型已转换为 DXNN 格式,用于 DEEPX NPU 推理。

`get_resources.sh` 下载资源压缩包,将模型和示例视频解压到 `assets/`,并在成功解压后删除下载的压缩包:

```text
assets/
├── models/
│   ├── hand-detector_192x192.dxnn
│   └── HandLandmarkLite.dxnn
└── videos/
    └── hands.mp4
```

手掌模型的输入必须是 UINT8 `[1, 192, 192, 3]`。关键点模型的输入必须是 UINT8 `[1, 224, 224, 3]`。

<!-- cell: resource-download-note src: 670c65bcaf -->
仅当资源缺失时才运行下一个单元格。解压过程中,同名的现有文件可能会被替换。

<!-- cell: inspect-heading src: c632a33a79 -->
### 4.1 可选的模型检查

如果 `dxparse` 和模型都可用,下一个单元格会打印它们的张量信息以及各自的 `dxrun` 基线。手掌检测器以一个 `cpu_0` 任务结尾(其锚框由 DX-RT 内部的 ONNX Runtime 解码),因此需要 `--use-ort`;关键点模型为仅 NPU。否则,该单元格会跳过检查而不报错。

<!-- cell: code-heading src: 14c08de877 -->
## 5. C++ 代码导读

实现位于 `app/hand_landmarks.cpp`。下一个单元格直接从源文件中显示主要的处理部分,与教程 20 到 22 的做法相同,因此笔记本不会保留第二份代码副本。

<!-- cell: code-explanation src: 1a18d06026 -->
### 5.1 主要实现阶段

1. `Options` 和 `parse_args` 选择输入、模型、阈值、显示模式和摄像头设置。
2. `preprocess_palm_frame` 将 BGR 转换为 RGB,并生成 192 x 192 的手掌检测器输入。默认使用 letterbox 填充。
3. `decode_palm_detections` 将原始张量转换为手掌框和七个关键点,然后应用加权非极大值抑制(NMS)。
4. 手腕和中指关键点定义一个旋转的手部区域。`make_landmark_input` 将其变换为 224 x 224。
5. `run_landmark_async` 为每个检测到的手掌提交一次关键点推理。其回调将结果转换回帧坐标。
6. `draw_hand_landmarks` 连接并渲染 21 个点。`FrameView` 显示帧和性能指标。
7. `run_detection_loop` 将采集、两个推理阶段、渲染、可选的视频保存以及播放节奏控制组合在一起。

<!-- cell: build-heading src: b00b1b09c7 -->
## 6. 构建

`build.sh` 以 Release 模式配置 CMake,并使用所有可用的 CPU 核心运行 `make`。使用 `--clean` 可先删除之前的构建目录。

<!-- cell: camera-heading src: 176345be53 -->
## 7. 使用摄像头运行

脚本使用摄像头索引 0,请求 1280 x 720 分辨率和 30 FPS。仅当处于图形会话且 NPU、摄像头和模型均已就绪时,才将 `RUN_CAMERA` 设为 `True`。按 `Esc` 或 `Q` 关闭应用程序。

<!-- cell: video-heading src: 51c4ab12d3 -->
## 8. 使用视频运行

`run_video.sh` 读取 `assets/videos/hands.mp4` 并循环播放。将模型和视频放到预期位置后,再把 `RUN_VIDEO` 设为 `True`。

<!-- cell: custom-run src: 1a1d864d58 -->
## 9. 自定义命令

选择另一个摄像头和采集模式:

```bash
cd app
./run_camera.sh --camera 2 --width 1920 --height 1080 --fps 30
```

使用自定义视频并显示手掌区域以便调试:

```bash
./build/hand_landmarks --video /path/to/input.mp4 --loop --show-palm
```

保存渲染输出。文件 `output-XX.mp4` 会写入当前目录,因此请在教程的 `workspace/` 中运行,以保持 `app/` 整洁:

```bash
mkdir -p ../workspace && cd ../workspace
../app/build/hand_landmarks --video /path/to/input.mp4 --save --landmark-only
```

使用 `Esc` 或 `Q` 退出,使用 `F` 切换全屏模式。

<!-- cell: c1eb325f-0e0f-441f-979a-f3b1a4643a01 src: ecf000b5c7 -->
## 10. 模型来源

本演示使用的手掌检测模型和手部关键点模型来源于 Google AI Edge 提供的 [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) 模型包。这些模型已转换为 DXNN 格式用于 DEEPX NPU 推理,分别以 `hand-detector_192x192.dxnn` 和 `HandLandmarkLite.dxnn` 的形式使用。

<!-- cell: 86932e85-d1bd-4636-8069-5dd13cbb56ac src: 59326af863 -->
## 11. 实用选项

- `--max-hands N`:设置每帧的最大手数;默认值为 4。
- `--palm-conf VALUE`:设置手掌置信度阈值;默认值为 0.2。
- `--landmark-conf VALUE`:设置关键点存在阈值;默认值为 0.5。
- `--show-palm`:同时绘制手掌框和旋转区域。
- `--landmark-only`:仅绘制手部关键点。
- `--windowed`:使用 1280 x 720 窗口而不是全屏。
- `--save`:将渲染结果保存为当前目录中下一个可用的 `output-XX.mp4` 文件(请在 `workspace/` 中运行,参见第 9 节)。

运行 `./build/hand_landmarks --help` 查看完整选项列表。按 `Esc` 或 `Q` 退出,按 `F` 切换全屏模式。

<!-- cell: troubleshooting src: 22102be3a3 -->
## 12. 故障排除

| 症状 | 看到的内容 | 可能原因 | 解决方法 |
|---|---|---|---|
| 缺少模型 | 资源检查将某个文件列为 `missing` | `get_resources.sh` 未运行或压缩包已变更 | 运行资源单元格;确认 `assets/models/` 下的两个文件名 |
| 模型被拒绝 | 加载时出现形状或 dtype 错误 | 模型文件错误 | 用 `dxparse` 检查输入形状和 UINT8 dtype |
| 无法打开摄像头 | `cannot open camera 0` | 没有摄像头或没有权限 | 运行 `v4l2-ctl --list-devices`,并用 `--camera` 尝试其他索引 |
| 没有窗口出现 | `could not connect to display` | 没有图形会话 | 在桌面会话中运行 |
| CMake 找不到 DXRT | `Could not find ... dxrt` | 未安装 DX-RT,或安装在自定义前缀下 | 安装 DX-Runtime(教程 01 第 3 节),或运行 `./build.sh -DCMAKE_PREFIX_PATH=<prefix>` |

<!-- cell: 24061ba3-6ced-4492-93d3-3508af97fdf0 src: 99d0cc5118 -->
## 13. 总结

两个 DXRT 模型依次运行:手掌检测器找到手,关键点模型为每只手预测 21 个关键点。Qt 应用程序采集帧,以异步方式驱动两个模型,并在实时画面上绘制关键点。

### 13.1 完成清单

- [ ] 使用 `get_resources.sh` 下载模型和示例视频
- [ ] 使用 `dxparse` 检查两个模型
- [ ] 使用 `build.sh` 构建应用程序
- [ ] 在视频或摄像头上运行应用程序

> **下一步:** 继续学习教程 24,运行带有实时后处理控制的 PIDNet 语义分割。
