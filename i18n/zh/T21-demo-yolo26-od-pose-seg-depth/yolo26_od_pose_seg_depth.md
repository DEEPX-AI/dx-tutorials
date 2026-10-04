<!-- i18n source: notebooks/T21-demo-yolo26-od-pose-seg-depth/yolo26_od_pose_seg_depth.ipynb -->
<!-- i18n lang: zh -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: fa62a5df src: a9203579d1 -->
# DEEPX Tutorial 21 - YOLO26 目标检测、姿态、分割与深度演示

本教程介绍一个基于 Qt 的 C++ 应用程序,它使用 DEEPX NPU 在同一路摄像头或视频流上运行以下四个 YOLO26s 模型。
- YOLO26s 目标检测
- YOLO26s 姿态估计
- YOLO26s 实例分割
- YOLO26s 单目深度估计

![YOLO 多通道演示](assets/yolo26-od-pos-seg-depth.png)

<!-- cell: 7b3ff195 src: d25198ce78 -->
## 学习目标

完成本教程后,您将能够:

- 理解自包含的 C++ 项目结构;
- 识别检测、姿态、分割和深度四条流水线;
- 理解一帧输入如何分发给四个异步工作线程;
- 以 Release 模式构建 Qt 应用程序;以及
- 使用摄像头或视频文件运行演示。

<!-- cell: t21-npu-pattern src: 0c9b4db3ff -->
## 本应用如何使用 NPU

| 项目 | 本教程中 | 学习来源 |
|---|---|---|
| 引擎 | **四个** `InferenceEngine` 对象,每个任务一个,全部位于同一设备上;DX-RT 在 NPU 核心上交错处理它们的请求 | T06-3 §2 |
| 执行 | 每个工作线程调用 `RunAsync()`,`bufferCount` 设为其在途请求上限,并在完成回调中渲染;最新帧队列丢弃过期帧,而不是累积延迟 | T06-2 §5, §7 |
| 任务图 | 四个模型都以一个 `cpu_0` 任务结尾(检测头通过 ONNX Runtime 在 CPU 上解码),因此每多一个模型,主机 CPU 负载就会随之增加 | T06-1 §6 |
| 每次请求的输入 | 三个模型接受 `[1, 640, 640, 3]` UINT8(1.2 MB),深度模型接受 `[1, 768, 768, 3]`(1.8 MB);一帧摄像头画面变成四个经 letterbox 处理的输入 | T06-3 §4 |
| 需要测量的内容 | 3.1 节中各模型的 `dxrun` 基准值与窗口中的四个 FPS 标签:四个面板速率之和显示了 NPU 是如何被共享的 | T06-1 §5 |

<!-- cell: t21-prerequisites src: 2ab76b757a -->
## 前提条件

- 已完成教程 01:已安装 DX-RT,DEEPX NPU 以 `/dev/dxrt0` 的形式可见;Qt 窗口需要 `cmake`、`g++` 和 `qtbase5-dev`。
- 图形桌面会话;摄像头演示需要一个 V4L2 摄像头(可选)。
- 下载量:一个约 80 MB 的归档(深度模型和示例视频,外加教程 01 已放入共享工作区的三个 YOLO26s 模型的副本)。
- 耗时:约 20 分钟;构建需要一到两分钟。除第 2 节中 `apt-get` 行(如果缺少软件包)外不需要 `sudo`。
- 已在 DX-RT 3.4.2 上验证。模型由 DX-COM 2.4.0 编译。

下一个单元格会定位 SDK、检查这些要求并打印状态表。任何标记为 `MISSING` 的项目都会附上提供它的步骤。

<!-- cell: f8c4f71c src: bafe0141d6 -->
## 1. 定位教程文件

<!-- cell: 9c6f9a02 src: fa676f3bae -->
### 1.1 项目布局

```text
T21-demo-yolo26-od-pose-seg-depth/
├── get_resources.sh
├── assets/
│   ├── models/
│   └── videos/
├── app/
│   ├── build.sh
│   ├── run_camera.sh
│   ├── run_video.sh
│   ├── CMakeLists.txt
│   ├── yolo26s_4.cpp
│   ├── common/
│   │   ├── base/
│   │   ├── processors/
│   │   └── utility/
│   ├── factory/
│   └── extern/
└── yolo26_od_pose_seg_depth.ipynb
```

<!-- cell: afe9307e src: b0dd491e5c -->
## 2. 检查环境

安装 Debian 或 Ubuntu 软件包:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config qtbase5-dev libopencv-dev ffmpeg v4l-utils
```

资源归档约 84 MB(四个模型和一个示例视频)。

<!-- cell: c457d2df src: 9ff0a2e16c -->
## 3. 下载并检查资源

应用程序期望在 `assets/` 下有四个模型和一个示例视频:

```text
assets/
├── models/
│   ├── yolo26-s_640x640.dxnn
│   ├── yolo26-s-pose_640x640.dxnn
│   ├── yolo26-s-seg_640x640.dxnn
│   └── yolo26-depth-s_768x768_q-lite.dxnn
└── videos/
    └── dance-960-540.mp4
```

检测、姿态和分割这三个模型与教程 01 第 3.3 节下载到共享工作区(`<DX_ALL_SUITE_DIR>/workspace/res/models`)的文件相同,因此当它们存在时,下一个单元格会从那里创建链接。深度模型和视频只包含在本教程的归档中(约 80 MB);`get_resources.sh` 会下载它、解压到 `assets/`,并在成功解压后删除归档。

<!-- cell: bbbdc0e0 src: 670c65bcaf -->
仅当资源缺失时才运行下一个单元格。解压过程中,同名的现有文件可能会被替换。

<!-- cell: t21-baseline-md src: e5aeb7a350 -->
### 3.1 检查并基准测试四个模型

在阅读应用程序之前,先看看四个引擎将要运行的内容。`dxparse -v` 显示每个模型都有一个 NPU 任务,后跟一个 `cpu_0` 任务:检测头(以及深度模型的输出阶段)由 DX-RT 内部的 ONNX Runtime 解码,因此要获得有意义的基准测试必须加上 `--use-ort`。`dxrun` 的数值是单模型基准值;应用程序在同一个 NPU 上同时运行四个模型,因此每个面板的 FPS 都会低于其基准值。

```bash
cd <T21>/assets/models
dxparse -m yolo26-s_640x640.dxnn -v
dxrun -m yolo26-s_640x640.dxnn --use-ort -t 3 -v
```

<!-- cell: 2deb9884 src: 1d367207ac -->
## 4. 应用程序架构

```text
Camera or video
       |
       v
CaptureThread
       |
       +--> Detection worker    --> Object Detection panel
       +--> Pose worker         --> Pose Estimation panel
       +--> Segmentation worker --> Instance Segmentation panel
       +--> Depth worker        --> Depth Estimation panel
```

采集线程把每一帧 BGR 画面发布到四个最新帧队列。检测、姿态和分割使用各自任务专用的工厂。`DepthWorker` 执行 768 x 768 的 letterbox(信箱式填充)预处理、异步 DXRT 推理、去除 letterbox、缩放回源帧尺寸以及 Turbo 颜色映射。Qt 以全屏 2 x 2 网格显示全部四个实时结果。

<!-- cell: 51d08357 src: 44c389b817 -->
## 5. 阅读 C++ 代码

以下辅助函数显示当前源文件中选定的片段。

<!-- cell: 5e81e61e src: aa11e7c86a -->
### 5.1 构建配置

CMake 构建一个 C++17 可执行文件,并链接 Qt5 Widgets、OpenCV 和 DXRT。`PROJECT_ROOT_DIR` 指向教程目录,因此默认模型路径会解析到 `assets/models/` 下。

<!-- cell: 02a0873a src: 4e66e3d6da -->
### 5.2 命令行选项与默认资源

`AppArgs` 定义了四个模型路径、摄像头设置、可选的视频路径以及调试选项。`--model-depth` 覆盖默认的深度模型。省略 `--video` 即选择摄像头模式。使用 `-c` 或 `--camera` 选择 V4L2 设备,使用 `--width`、`--height` 和 `--fps` 请求采集设置。如果省略这些选项,默认值为 `/dev/video0`、1280 x 720 和 30 FPS。

<!-- cell: 3a6d2550 src: 5104ab68bd -->
### 5.3 任务工厂

每个工厂为一个任务创建正确的预处理、后处理和可视化组件。

<!-- cell: 70995906 src: e8239c4efd -->
### 5.4 异步结果工作线程

每个工作线程创建自己的 `InferenceEngine` 并注册一个异步回调。检测、姿态和分割使用各自任务专用的工厂。专用的深度工作线程验证 `[1, 768, 768, 3]` UINT8 输入和 `[1, 1, 768, 768]` FLOAT 输出契约,在回调完成前保留输入缓冲区,去除 letterbox 填充,恢复源帧几何尺寸,并应用 OpenCV 的 Turbo 颜色映射。当某个模型慢于输入流时,每个最新帧队列都会替换过期帧,而不是累积延迟。

<!-- cell: 01af8912 src: 6bab0dfd1b -->
### 5.5 摄像头与视频采集

`CaptureThread` 使用 OpenCV 的 `VideoCapture`。除非设置了 `--no-loop-video`,否则视频输入会循环播放。摄像头输入使用 V4L2 后端并请求 MJPG 格式。

<!-- cell: 6ec88213 src: f00de1deb0 -->
### 5.6 Qt 2 x 2 窗口

`QuadWindow` 创建四个实时面板,启动四个推理工作线程和采集线程,更新 FPS 标签,并执行有序的关闭流程。

<!-- cell: fef60649 src: 2836bd7037 -->
## 6. 构建应用程序

`build.sh` 创建 `app/build/`,以 Release 模式配置 CMake,并使用 `nproc` 报告的全部 CPU 核心运行 `make`。使用 `./build.sh --clean` 进行完整重新构建;当 DX-RT 安装在自定义前缀下时,使用 `./build.sh -DCMAKE_PREFIX_PATH=<prefix>`。

<!-- cell: 6e6f1fe2 src: b5bfeb4275 -->
## 7. 运行摄像头演示

默认设备为 `/dev/video0`,请求的分辨率为 1280 x 720、帧率为 30 FPS。修改下面的变量可选择其他摄像头或采集设置。运行单元格会打开一个全屏 Qt 窗口,并阻塞直到应用程序退出。

<!-- cell: c267e0de src: db9fcae673 -->
## 8. 运行视频演示

将 `VIDEO_PATH` 设置为 `assets/videos/` 下的某个文件。`run_video.sh` 接受视频路径作为其第一个参数。

<!-- cell: 9dee3c0f src: ca736df157 -->
## 9. 操作控制

- `Esc` 或 `q`:退出
- `EXIT` 按钮:用鼠标退出

<!-- cell: 05fc0113-d263-4d67-a4af-92c174b45b94 src: 6093c888ab -->
## 10. 故障排除

| 症状 | 看到的内容 | 可能原因 | 解决方法 |
|---|---|---|---|
| CMake 找不到 Qt5 | `Could not find a package configuration file provided by "Qt5"` | 未安装 `qtbase5-dev` | `sudo apt install qtbase5-dev` |
| CMake 找不到 DXRT | `Could not find a package configuration file provided by "dxrt"` | 未安装 DX-RT,或它安装在自定义前缀下 | 安装 DX-Runtime(教程 01 第 3 节)或运行 `./build.sh -DCMAKE_PREFIX_PATH=<prefix>` |
| 模型在加载时被拒绝 | DX-RT 版本或格式错误 | DXNN 是为其他 SDK 版本编译的(这些模型:DX-COM 2.4.0) | 使用本教程验证过的 DX-RT 版本(3.4.2),或用您的 DX-COM 重新编译 |
| 模型无法打开 | 指向 `assets/models` 下某个文件的加载错误 | 资源缺失 | 运行第 3 节中的资源单元格 |
| 摄像头无法打开 | `cannot open /dev/video0` | 没有摄像头或没有权限 | `v4l2-ctl --list-devices`;将用户加入 `video` 组;用 `--camera` 传入其他设备 |
| Qt 窗口没有出现 | `could not connect to display` | 没有图形会话 | 在桌面会话中运行 |

<!-- cell: 83f9ac2c src: 18f95b8d4f -->
## 11. 总结

一个采集线程把每一帧发送到四条独立的异步 DXRT 流水线。任务专用的工厂处理目标检测、姿态估计和实例分割。专用的深度工作线程对帧做 letterbox 处理、运行 768 x 768 的深度模型、将深度图恢复到源帧几何尺寸,并应用 Turbo 颜色映射。Qt 以全屏 2 x 2 布局显示全部四个实时结果。

### 11.1 完成清单

- [ ] 下载四个模型和示例视频
- [ ] 阅读一个采集线程如何为四条异步流水线供帧
- [ ] 构建 Qt 应用程序
- [ ] 运行视频演示和摄像头演示

> **下一步:** 继续学习教程 22,使用 CLIP 将摄像头帧与文本查询进行匹配。
