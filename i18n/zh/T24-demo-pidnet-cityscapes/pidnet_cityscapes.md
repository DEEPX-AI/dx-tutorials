<!-- i18n source: notebooks/T24-demo-pidnet-cityscapes/pidnet_cityscapes.ipynb -->
<!-- i18n lang: zh -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: title src: a79a64189a -->
# DEEPX Tutorial 24 - PIDNet Cityscapes C++ 演示

本笔记本讲解、构建并运行一个 Qt5 C++ 语义分割应用。它在 DEEPX NPU 上使用 PIDNet 处理摄像头或视频帧,在底部控制栏中突出显示较大的输入张量,并提供一个紧凑的实时滑块用于调节 PIDNet argmax 缩放比例。

<!-- cell: 062ad116-becd-4b26-a4b3-475220ad077f src: cb4f28770c -->
## 学习目标

完成本教程后,您将能够:

- 说明 PIDNet 的 logits(类别得分)如何变成带颜色的 Cityscapes 叠加层;
- 下载演示所需的模型和示例视频;
- 构建 Qt 应用并在摄像头或视频上运行;
- 使用 Argmax 缩放滑块在后处理分辨率与 CPU 时间之间进行权衡;
- 为您的主机选择合适的在途(in-flight)请求数量。

<!-- cell: t24-npu-pattern src: 0b3416de9c -->
## 本应用如何使用 NPU

| 项目 | 在本教程中 | 学习来源 |
|---|---|---|
| 引擎 | 一个 `InferenceEngine`;模型为仅 NPU(没有 CPU 任务),因此 DX-RT 返回原始类别 logits,应用负责全部后处理 | T06-1 §6 |
| 执行 | 采集线程最多保持 `--inflight`(默认 4)个 `RunAsync()` 请求排队;argmax 和渲染在完成回调中运行,同时 NPU 已在处理后续帧 | T06-2 §5, T06-3 §3 |
| 每个请求的输入 | `[1, 1024, 2048, 3]` UINT8 = 6.3 MB,是这些教程中最大的输入;输出为 `[1, 19, 128, 256]` FLOAT = 2.5 MB。主机到设备和设备到主机的传输在每个请求中占有可见的比例 | T06-3 §4 |
| CPU 侧 | 对 19 个类别做逐像素 argmax 是 CPU 工作;Argmax 缩放滑块以其分辨率换取 CPU 时间,而 NPU 速率保持不变 | T06-3 §10 |
| 测量内容 | 4.1 节中的 `dxrun -v`(NPU 处理时间与延迟)对比不同滑块位置和 `--inflight` 值下的 GUI FPS | T06-1 §5 |

<!-- cell: pipeline src: 2dfd7495e4 -->
## 1. 处理流水线

```text
Camera or video frame
        |
        v
Resize to model input and convert BGR to RGB
        |
        v
Asynchronous PIDNet inference -> class logits
        |
        v
Bilinear interpolation at the selected argmax scale
        |
        v
Per-pixel argmax -> Cityscapes color mask -> Qt5 GUI
```

默认模型接受 UINT8 `[1, 1024, 2048, 3]` 输入,并返回 FLOAT32 `[1, 19, 128, 256]` logits。

<!-- cell: prerequisites src: 9341de1df6 -->
## 2. 前提条件

- 已完成教程 01:已安装 DX-RT,且 DEEPX NPU 以 `/dev/dxrt0` 的形式可见(用 `dxcli -s` 检查);Qt 窗口需要 `cmake`、`g++` 和 `qtbase5-dev`。
- 图形桌面会话;摄像头演示需要一台 V4L2 摄像头(可选)。
- 下载量:一个约 20 MB 的压缩包(模型和示例视频),解压到被 git 忽略的 `assets/` 目录。
- 耗时:约 15 分钟;构建约需一分钟。除非缺少软件包需要运行下面的 `apt` 命令,否则不需要 `sudo`。
- 已在 DX-RT 3.4.2 上验证。模型使用 DX-COM 2.3.0 编译。

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev qtbase5-dev ffmpeg v4l-utils
```

下一个单元格会定位 SDK、检查这些要求并打印状态表。任何标记为 `MISSING` 的项目都会附上提供它的步骤。

<!-- cell: layout-heading src: 572e07788b -->
## 3. 项目布局

C++ 代码和启动脚本位于 `app/` 下。下载的模型和视频应放在 `assets/` 下。

<!-- cell: resources src: 163f686043 -->
## 4. 下载并检查资源

本演示参考官方 [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet) 项目。其 `PIDNet_S_Cityscapes_val.pt` PyTorch 检查点已转换为 DXNN 格式,用于 DEEPX NPU 推理。应用使用转换得到的 `pidnet_s_cityscapes_val_fixed.dxnn` 模型。

`get_resources.sh` 会下载资源压缩包,将模型和示例视频解压到 `assets/`,并在成功解压后删除下载的压缩包:

```text
assets/
├── models/
│   └── pidnet_s_cityscapes_val_fixed.dxnn
└── videos/
    └── pidnet.mp4
```

<!-- cell: resource-download-note src: 670c65bcaf -->
仅当资源缺失时才运行下一个单元格。解压过程中,已有的同名文件可能会被替换。

<!-- cell: inspect-heading src: bf1c859947 -->
### 4.1 可选的模型检查

如果 `dxparse` 和模型都可用,下一个单元格会打印输入和输出张量以及 `dxrun -v` 基准。请比较 NPU 处理时间与端到端延迟:6.3 MB 的输入使主机到设备的传输成为每个请求中可见的一部分,而这正是异步在途(in-flight)队列所隐藏的部分。否则,该单元格会跳过检查而不报错。

<!-- cell: scale-heading src: affd9c896d -->
## 5. 理解 `pidnet_argmax_scale`

PIDNet 生成较低分辨率的类别 logits。在为每个像素选择最可能的类别之前,应用会对这些 logits 进行双线性插值。缩放比例控制中间插值的尺寸:

- `0.1`:CPU 开销最低,边界较粗糙。
- `0.4`:默认值,在速度与细节之间取得平衡。
- `1.0`:全帧 argmax 分辨率,CPU 开销最高。

GUI 滑块的取值范围为 0.10 到 1.00,步长 0.05。它写入一个原子值,每个异步完成回调在处理一帧之前读取该值一次。这样每一帧都使用一个一致的缩放比例。

<!-- cell: code-heading src: ed5809056c -->
## 6. C++ 代码指南

实现位于 `app/pidnet_cityscapes.cpp`。下一个单元格会直接从源文件显示其主要部分,与教程 20 到 23 的做法相同。

<!-- cell: code-explanation src: f2781bc990 -->
### 6.1 主要实现阶段

1. `Options` 和 `parse_args` 选择摄像头或视频、模型、摄像头设置、初始缩放比例以及叠加层不透明度。
2. `preprocess_frame` 将 BGR 转换为 RGB,并直接将帧缩放到模型输入张量。
3. `compute_argmax_mask` 按当前缩放比例对每个类别通道进行插值,并选出得分最高的类别。
4. `render_segmentation` 将类别 ID 映射为 Cityscapes 颜色,并将掩码与输入帧混合。
5. `MainWindow` 将视频置于底部控制栏之上,控制栏包含运行时输入形状、紧凑的缩放滑块和 Exit 按钮。
6. 采集线程通过 `RunAsync` 最多提交四个请求,使 NPU 推理与 CPU 后处理得以重叠。
7. 完成回调只把最新的渲染结果发布到一个单帧邮箱。一个 Qt 定时器消费该结果,而不会堆积过时的 GUI 事件。
8. OpenCV 操作只使用一个内部线程,因为帧级别的回调并行已经用满了可用的 CPU 核心。

<!-- cell: build-heading src: 6910ddc7ad -->
## 7. 构建

`build.sh` 配置一个 Release 构建,并使用所有可用的 CPU 核心运行 `make`。使用 `--clean` 可先删除旧的构建目录。

<!-- cell: camera-heading src: 6ce2ff2946 -->
## 8. 使用摄像头运行

该脚本请求索引为 0 的摄像头,分辨率 1280 x 720,30 FPS。仅当 NPU、模型、摄像头和图形会话都已就绪时,才将 `RUN_CAMERA` 设为 `True`。

<!-- cell: video-heading src: 85d23010af -->
## 9. 使用视频运行

`run_video.sh` 读取 `assets/videos/pidnet.mp4` 并循环播放。将模型和视频放到预期位置后,再将 `RUN_VIDEO` 设为 `True`。

<!-- cell: custom-run src: 3eda8227c7 -->
## 10. 自定义命令

以另一个滑块值启动:

```bash
cd app
./run_video.sh --pidnet-argmax-scale 0.7
```

以窗口模式使用另一个视频:

```bash
./build/pidnet_cityscapes --video /path/to/input.mp4 --loop --windowed
```

选择另一台摄像头并降低掩码不透明度:

```bash
./run_camera.sh --camera 2 --width 1920 --height 1080 --fps 30 --alpha 0.45
```

使用 `Esc`、`Q` 或 Exit 按钮退出。按 `F` 切换全屏模式。默认的 `--inflight 4` 面向四核 Raspberry Pi 5;如果需要更高吞吐量,请测试 `--inflight 6`。

<!-- cell: d4dfe950-47a4-4a4e-adb8-c9defa2b6703 src: 5c302aa913 -->
## 11. 模型来源

本演示参考官方 [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet) 项目。`PIDNet_S_Cityscapes_val.pt` PyTorch 检查点取自该项目,并已转换为 DXNN 格式,用于 DEEPX NPU 推理。本应用使用转换得到的 `pidnet_s_cityscapes_val_fixed.dxnn` 模型。

<!-- cell: 508b6045-6a18-4e39-8844-ad638d890d04 src: 08a0e7b298 -->
## 12. 其他选项与控制

- `--alpha VALUE`:设置分割叠加层的不透明度,范围 0.0 到 1.0;默认值为 0.6。
- `--inflight N`:设置最大异步请求数,范围 1 到 6。默认值为 4,面向四核 Raspberry Pi 5。
- `--no-pace`:以尽可能快的速度处理视频,而不是匹配其源 FPS。
- `--windowed`:以 1280 x 800 的窗口启动。
- `--full-screen`:以全屏模式启动;这是脚本的默认设置。
- `Esc`、`Q` 或 **Exit** 按钮:关闭应用。
- `F`:切换全屏模式。

运行 `./build/pidnet_cityscapes --help` 查看完整的选项列表。

<!-- cell: 81d36d4a-2c61-4c7b-aa91-004b609ab880 src: 12bee04ca1 -->
## 13. 性能设计

采集线程通过 `RunAsync` 提交帧。完成回调执行 argmax 和掩码渲染,同时 NPU 开始处理其他帧。有界的在途(in-flight)限制器防止内存无限增长。OpenCV 的内部工作线程池被限制为一个线程,因为并行已经在各回调之间发生;这避免了在四核系统上出现嵌套的过度订阅。

完成的帧被写入一个单帧邮箱。Qt 线程始终取最新的结果,因此延迟的帧不会在 GUI 队列中堆积。当 CPU 后处理跟不上输入速率时,这能保持交互响应灵敏。

默认的 `--inflight 4` 与 Raspberry Pi 5 的 CPU 核心数相匹配。如果 CPU 使用率和温度仍在可接受范围内,请测试 `--inflight 6` 以用满整个 DXRT 缓冲池:

```bash
./run_video.sh --inflight 6
```

<!-- cell: troubleshooting src: 2b346be6d4 -->
## 14. 故障排除

| 症状 | 看到的内容 | 可能原因 | 解决方法 |
|---|---|---|---|
| 模型缺失 | 资源检查将其列为 `missing` | `get_resources.sh` 未运行 | 运行资源单元格;核对 `assets/models/` 下的文件名 |
| 视频缺失 | 找不到 `pidnet.mp4` | 资源不完整 | 将 `pidnet.mp4` 放到 `assets/videos/` 下,或向 `run_video.sh` 传入其他路径 |
| 无法打开摄像头 | `cannot open camera 0` | 没有摄像头或没有权限 | 运行 `v4l2-ctl --list-devices` 并尝试其他索引 |
| 没有窗口出现 | `could not connect to display` | 没有图形会话 | 在桌面会话中运行 |
| CMake 找不到 DXRT | `Could not find a package configuration file provided by "dxrt"` | 未安装 DX-RT,或它安装在自定义前缀下 | 安装 DX-Runtime(教程 01 第 3 节)或运行 `./build.sh -DCMAKE_PREFIX_PATH=<prefix>` |
| 模型在加载时被拒绝 | DX-RT 版本或格式错误 | DXNN 由 DX-COM 2.3.0 为较旧的运行时编译 | 使用本教程验证过的 DX-RT 版本(3.4.2) |
| 后处理缓慢 | NPU 空闲时 FPS 很低 | CPU 以全分辨率执行 argmax | 将 Argmax 缩放滑块向 0.1 方向移动,或降低 `--inflight` |

<!-- cell: 06f03620-4bef-40dd-84a5-9fcfb0ea5fc8 src: eae16a4f91 -->
## 15. 总结

PIDNet 在 NPU 上运行,应用在 CPU 上将其类别 logits 转换为带颜色的 Cityscapes 分割叠加层。Argmax 缩放滑块以后处理分辨率换取 CPU 时间,而模型始终以全速运行。

### 15.1 完成清单

- [ ] 使用 `get_resources.sh` 下载模型和示例视频
- [ ] 使用 `dxparse` 检查模型
- [ ] 使用 `build.sh` 构建应用
- [ ] 运行应用并调节 Argmax 缩放比例

> **下一步:** 您已完成演示系列。请返回仓库 README 查看完整的教程列表。
