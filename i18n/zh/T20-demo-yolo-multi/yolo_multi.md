<!-- i18n source: notebooks/T20-demo-yolo-multi/yolo_multi.ipynb -->
<!-- i18n lang: zh -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 7bc75515 src: 7038880fdd -->
# DEEPX Tutorial 20 - YOLO 多通道 C++ 演示

本教程介绍一个 C++ 应用程序如何使用 DEEPX NPU 在多个视频通道上运行 YOLO 目标检测。同时还展示如何下载资源、构建应用程序,以及运行视频或摄像头演示。

![YOLO 多通道演示](assets/yolo-multi-sc.png)

<!-- cell: 8f86f5a0 src: 6807896955 -->
## 学习目标

完成本教程后,您将能够:

- 理解 C++ 项目结构;
- 读懂 JSON 文件中的模型、输入和显示设置;
- 跟踪异步多通道推理流程;
- 以 Release 模式构建应用程序;以及
- 运行视频和摄像头示例。

<!-- cell: t20-npu-pattern src: 80e34194de -->
## 本应用如何使用 NPU

| 项目 | 在本教程中 | 学习来源 |
|---|---|---|
| 引擎 | **一个** `InferenceEngine` 由全部 36 个通道线程共享;DX-RT 将它们的请求排队并调度到 NPU 上 | T06-3 §2 |
| 执行方式 | 每个通道调用 `RunAsync()`,并在回调中接收检测框,同时已经开始准备下一帧 | T06-2 §5 |
| 任务图 | `YOLOV5S_PPU` 是单个 NPU 任务:PPU 返回已解码的检测框,因此主机只需运行 NMS。在 36 个通道的场景下,PPU 的价值在这里得以体现(对比 T05-2 §4) | T05-2 §3 |
| 每次请求的输入 | `[1, 512, 512, 3]` UINT8 = 786 KB;36 个通道以 30 FPS 运行时 H2D 流量将达到 850 MB/s,这就是显示图块较小的原因 | T06-3 §4 |
| 测量内容 | 将 3.2 节的 `dxrun` 基准(单路流、无解码和显示)与窗口标题栏中显示的 FPS 进行对比 | T06-1 §5 |

<!-- cell: t20-prerequisites src: c5253820d0 -->
## 前提条件

- 已完成教程 01:已安装 DX-RT,且 DEEPX NPU 以 `/dev/dxrt0` 的形式可见;已安装 `cmake` 和 `g++`(`build-essential`)。
- OpenCV 窗口需要图形桌面会话;摄像头演示需要位于 `/dev/video0` 的 V4L2 摄像头(可选)。
- 下载量:一个约 134 MB 的归档文件(YOLOv5s PPU 模型和 45 个示例视频),解压到 `assets/` 中,该目录被 git 忽略。
- 耗时:约 20 分钟;构建需要一到两分钟。除第 2 节中在缺少软件包时需要运行的 `apt-get` 一行外,不需要 `sudo`。
- 已在 DX-RT 3.4.2 上验证。模型由 DX-COM 2.2.0 编译,可在此运行时上加载。

下一个单元格会定位 SDK、检查这些要求并打印状态表。任何标记为 `MISSING` 的项目都会附上提供它的步骤。

<!-- cell: 8ea9a7fc src: a4b88669a3 -->
## 1. 定位教程文件

无论 JupyterLab 是从仓库根目录还是从本笔记本目录启动,以下单元格都能找到教程目录。

<!-- cell: 77ab9dbf src: 9a378a9fc7 -->
### 1.1 项目布局

```text
T20-demo-yolo-multi/
├── get_resources.sh
├── assets/
│   ├── models/
│   └── videos/
├── app/
│   ├── build.sh
│   ├── run_camera.sh
│   ├── run_video.sh
│   ├── config/
│   ├── include/
│   ├── src/
│   ├── lib/
│   ├── extern/
│   └── sample/
└── yolo_multi.ipynb
```

`extern/` 包含仅头文件的 cxxopts 和 RapidJSON 依赖。`sample/` 包含显示 UI 使用的字体和图片。

<!-- cell: 4173a889 src: 76d6941463 -->
## 2. 检查环境

安装 Debian 或 Ubuntu 软件包:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config ca-certificates curl tar \
    libopencv-dev libopencv-contrib-dev libfreetype-dev ffmpeg v4l-utils \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
```

资源归档文件约 140 MB(一个 PPU 模型和 45 个示例视频)。

<!-- cell: e29c2c86 src: 4446f1b209 -->
## 3. 下载并检查资源

两个演示都使用 `assets/models/YOLOV5S_PPU.dxnn` 以及 `assets/videos/` 下的示例 MP4 文件。`get_resources.sh` 下载一个归档文件,将其解压到 `assets/` 中,并在成功解压后删除归档文件。

<!-- cell: d87f310e src: 670c65bcaf -->
仅当资源缺失时才运行下一个单元格。解压过程中同名的现有文件可能会被替换。

<!-- cell: 49a25e54 src: 2ba54cf8ba -->
### 3.1 使用 `dxparse` 检查模型

使用 `dxparse` 检查模型结构、任务图、张量、内存使用和依赖关系:

```bash
dxparse -m assets/models/YOLOV5S_PPU.dxnn -v
```

详细输出会像这样列出模型输入:

```text
  Inputs
     -  images, UINT8, [1, 512, 512, 3 ]
```

各维度依次为批次、高度、宽度和通道。因此,该模型期望的是 512 x 512 的三通道输入,而不是 640 x 640 的输入。

<!-- cell: 16d5dc87 src: 339ac4ce39 -->
### 3.2 使用 `dxrun` 对模型进行基准测试

使用自动生成的虚拟输入运行一次五秒钟的 CLI 基准测试:

```bash
dxrun -m assets/models/YOLOV5S_PPU.dxnn --use-ort -t 5
```

当既未指定 `--single` 也未指定 `--fps` 时,`dxrun` 默认使用基准测试模式。`-t 5` 将测量时长设为五秒,`--use-ort` 为模型图中的 CPU 任务启用 ONNX Runtime。报告的结果提供了一个不含多通道视频解码和显示开销的命令行性能基准。

<!-- cell: edb6cae2 src: 47f55de289 -->
## 4. 理解演示配置

应用程序从一个 JSON 文件读取全部运行时设置。重要字段包括:

- `model_path`:DXNN 模型的路径,相对于 `app/`;
- `model_name`:选择匹配的 YOLO 后处理参数;
- `video_sources`:输入路径、输入类型以及可选的保存帧数;
- `display_config`:输出尺寸、网格、FPS 和布局设置;以及
- `num_devices`:标题栏中显示的 NPU 设备数量。

<!-- cell: 3313e802 src: 0facc52619 -->
摄像头配置包含一个 `/dev/video0` 输入和 32 个视频输入。摄像头高亮显示与 `expand_mode` 无关,因此摄像头会被放置在放大的中央区域并带有黄色边框。

视频配置包含 36 个视频输入,排列为 6 x 6 网格,`expand_mode` 设为 `false`。应用程序的扩展布局(一个放大的中央图块)仅适用于 33、41、61 或 73 个输入源,这就是 33 源的摄像头配置使用它而 36 源的视频配置不使用它的原因。

<!-- cell: 44889672 src: 4575c7a014 -->
## 5. 阅读 C++ 代码

笔记本直接从当前源文件中显示选定的片段。这样可以避免在笔记本中保存第二份 C++ 代码副本。

<!-- cell: 940a9b9a src: c69dca4723 -->
### 5.1 构建配置

CMake 构建一个 C++14 可执行文件,并链接 DXRT、OpenCV、pthread、OpenMP 以及 C++14 所需的 filesystem 库。在可用时会使用 OpenCV 的 FreeType 支持。`cmake/dxdemo.function.cmake` 中的 `add_dxrt_lib()` 辅助函数封装了与教程 06-2 和其他演示相同的 `find_package(dxrt)` 查找,并链接导入目标 `dxrt::dxrt`;其额外的分支仅用于交叉编译和 Windows 构建。

<!-- cell: cfc22a04 src: f378c3defe -->
### 5.2 配置解析

`ApplicationJsonParser()` 验证 JSON 字段并填充 `AppConfig`。解析器还会为可选的显示设置提供默认值。

<!-- cell: b299367c src: 4f1996b60f -->
### 5.3 多通道推理流水线

```text
JSON configuration
        |
        v
DXRT InferenceEngine (shared)
        |
        +-- ObjectDetection: channel 1 --+
        +-- ObjectDetection: channel 2 --+--> output grid --> OpenCV window
        +-- ObjectDetection: channel N --+
```

`main()` 创建一个 DXRT `InferenceEngine`,并为每个输入源创建一个 `ObjectDetection` 对象。每个通道在自己的工作线程中运行。

<!-- cell: b25e522d src: 0cf1ca9c29 -->
### 5.4 每通道异步推理

`ObjectDetection::threadFunc()` 获取一帧预处理后的输入并调用 `RunAsync()`。回调更新最新的边界框,同时通道线程准备显示帧。互斥锁保护工作线程与回调之间共享的数据。

<!-- cell: 966230b7 src: 614c770d7d -->
### 5.5 YOLO 后处理

`Yolo::PostProc()` 根据模型输出类型选择解码器。这些演示使用 PPU 输出路径。解码后的候选框按类别排序,然后传入非极大值抑制以去除重叠的检测框。

<!-- cell: bf10922b src: 576016cf2b -->
## 6. 构建应用程序

`build.sh` 创建 `app/build/`,以 Release 模式配置 CMake,并使用 `nproc` 报告的全部 CPU 核心运行 `make`。需要完全重新构建时使用 `./build.sh --clean`;CMake 选项可以直接透传,例如当 DX-RT 安装在自定义前缀下时使用 `./build.sh -DCMAKE_PREFIX_PATH=<prefix>`。

<!-- cell: a3cc6532 src: 1521315f86 -->
## 7. 运行 36 通道视频演示

下一个单元格会打开一个 OpenCV 窗口,并等待直到演示退出。按 `Esc` 或 `q`,或点击 `EXIT` 按钮。

<!-- cell: 34f09e43 src: 1e56cf0884 -->
## 8. 运行 33 通道摄像头演示

默认配置期望在 `/dev/video0` 有一个 V4L2 摄像头。应用程序会自动选择受支持的摄像头模式,并将摄像头帧率限制为最多 30 FPS。

<!-- cell: cec0fe9e src: 0cee838170 -->
## 9. 控制方式

- `Esc` 或 `q`:退出
- `t`:显示或隐藏检测框
- `EXIT` 按钮:用鼠标退出

<!-- cell: 0c8262f3-3b63-458d-a93a-f9bb5f21dc27 src: c73952716d -->
## 10. 故障排除

| 症状 | 看到的内容 | 可能原因 | 解决方法 |
|---|---|---|---|
| CMake 找不到 DXRT | `Could not find a package configuration file provided by "dxrt"` | 未安装 DX-RT,或它位于自定义前缀下 | 安装 DX-Runtime(教程 01 第 3 节)或运行 `./build.sh -DCMAKE_PREFIX_PATH=<prefix>` |
| 无法打开模型或视频 | 应用程序日志中出现 `[ER] ... cannot open` | 资源缺失 | 运行第 3 节中的资源单元格 |
| 模型与运行时版本不匹配 | `The version of the compiled model is not compatible with the version of the runtime` | DXNN 是为另一个 SDK 版本编译的(本模型:DX-COM 2.2.0) | 使用本教程验证过的 DX-RT 版本(3.4.2),或用您的 DX-COM 重新编译模型 |
| 无法打开摄像头 | `cannot open /dev/video0` | 没有摄像头,或用户不在 `video` 组中 | `v4l2-ctl --list-devices`;`sudo usermod -aG video $USER` 然后重新登录;编辑摄像头 JSON 以使用其他设备 |
| 没有出现窗口 | `cannot open display` | 没有图形会话 | 在桌面会话中运行 |

<!-- cell: 1da13663 src: 0ee7a9d3d1 -->
## 11. 总结

应用程序在多个通道工作线程之间共享一个 DXRT 推理引擎。每个工作线程执行异步推理、YOLO PPU 后处理和帧渲染。主循环将各通道的帧合成为一个可配置的网格,并在 OpenCV 窗口中显示运行时信息。

### 11.1 完成清单

- [ ] 使用 `get_resources.sh` 下载模型和视频
- [ ] 使用 `dxparse` 检查 PPU 模型,并使用 `dxrun` 进行基准测试
- [ ] 阅读通道工作线程如何共享一个 `InferenceEngine`
- [ ] 构建应用程序并运行视频演示
- [ ] 在 JSON 配置中修改网格或输入列表,然后再次运行

> **下一步:** 继续学习教程 21,在一个 Qt 应用程序中同时运行目标检测、姿态估计、分割和深度估计。
