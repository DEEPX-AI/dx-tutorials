<!-- i18n source: notebooks/T22-demo-clip-single/clip_single.ipynb -->
<!-- i18n lang: zh -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: t22-01 src: e41c8db501 -->
# DEEPX Tutorial 22 - CLIP 单路 C++ 演示

本教程介绍如何构建并运行一个使用摄像头或视频输入的 C++ CLIP 应用程序。文本查询由 ONNX Runtime 编码,图像由 DEEPX NPU 异步编码,Qt GUI 显示相似度分数。

<!-- cell: t22-02 src: 4b7588c5da -->
![CLIP 单路演示](assets/clip-single-sc.png)

<!-- cell: t22-03 src: 0d359c5f65 -->
## 学习目标

完成本教程后,您将理解:

- CLIP 如何比较图像嵌入向量与文本嵌入向量,
- C++ 应用程序如何结合 ONNX Runtime 与 DXRT,
- 摄像头和视频帧如何为图像编码器做准备,
- 异步 NPU 推理和跳帧如何工作,以及
- 如何构建并运行该应用程序。

<!-- cell: t22-npu-pattern src: f43836dc91 -->
## 本应用如何使用 NPU

| 项目 | 本教程中 | 学习来源 |
|---|---|---|
| 引擎 | 一个用于 ViT-L/14 图像编码器的 `InferenceEngine`;文本编码器在 DX-RT 之外由 ONNX Runtime 在 CPU 上运行,每个查询只运行一次,其嵌入向量会被缓存 | T06-2 §4 |
| 执行方式 | `RunAsync()`,`bufferCount` 等于同时在途的帧数;每个已提交的帧在回调复制出 768 维嵌入向量之前都保持其输入缓冲区有效 | T06-2 §5, §7 |
| 任务图 | DXNN 以 `cpu_0` 任务开始、在 NPU 上结束,因此 CPU 部分位于关键路径上,且必须使用 `--use-ort` | T06-1 §6 |
| 每次请求的输入 | `[1, 3, 224, 224]` FLOAT = 602 KB,在主机上完成缩放、中心裁剪和归一化之后 | T06-3 §4 |
| 测量内容 | 第 4 节的 `dxrun` 基线与 `--skip-frames 0` 时的 GUI 速率对比;`--skip-frames` 是在排名新鲜度与 NPU 负载之间取舍的旋钮 | T06-1 §5 |

<!-- cell: 307b3492-06ab-4f38-8103-9d48f85e1fd8 src: 5091fc3c20 -->
## 前提条件

- 已安装 DX-RT 的 DEEPX NPU(教程 01 第 3 节)以及 `cmake`。
- 用于文本编码器的 ONNX Runtime C++ 头文件和共享库。DX-Runtime 会将它们安装在 `/usr/local` 下;若使用其他前缀,请运行 `./build.sh -DONNXRUNTIME_ROOT=<prefix>`。
- 用于显示 Qt 窗口的显示器;摄像头演示还需要一个 V4L2 摄像头。
- 下载量: 资源归档约 1.4 GB(图像编码器 DXNN、文本编码器 ONNX、词表、示例视频),解压到 `assets/` 中,该目录被 git 忽略。
- 耗时: 约 20 分钟加上下载时间;构建约需一分钟。
- 已在 DX-RT 3.4.2 上验证。图像编码器由 DX-COM 2.2.1 编译。

安装 Debian 或 Ubuntu 软件包:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev qtbase5-dev zlib1g-dev
```

下一个单元格会定位 SDK、检查这些要求并打印状态表。任何标记为 `MISSING` 的项目都会附上提供它的步骤。

<!-- cell: t22-05 src: 22cfa55977 -->
## 1. 项目结构

```text
T22-demo-clip-single/
├── README.md
├── clip_single.ipynb
├── get_resources.sh
├── assets/
│   ├── models/
│   ├── videos/
│   └── images/
└── app/
    ├── CMakeLists.txt
    ├── build.sh
    ├── run_camera.sh
    ├── run_video.sh
    ├── main.cpp
    ├── clip_tokenizer.cpp
    └── clip_tokenizer.hpp
```

<!-- cell: t22-07 src: 25298446ba -->
## 2. 下载并检查资源

`get_resources.sh` 会下载资源归档,将图像编码器、文本编码器、BPE 词表和示例视频解压到 `assets/` 中,并在解压成功后删除下载的归档文件。由于文本编码器将权重存储为外部数据,因此 ONNX `.data` 文件是必需的。

<!-- cell: t22-08-download-note src: 670c65bcaf -->
仅当资源缺失时才运行下一个单元格。解压过程中,同名的现有文件可能会被替换。

<!-- cell: t22-09 src: 49c8ad89f7 -->
## 3. 推理流水线

```text
Text queries -> BPE tokens -> ONNX Runtime text encoder -> text embeddings
                                                              |
Camera/video -> resize and center crop -> DXRT image encoder -> image embedding
                                                              |
                                                              v
                                      L2 normalization -> dot products -> ranked GUI scores
```

文本嵌入向量在启动时计算并缓存。每个被选中的视频帧都会转换为 224 x 224 的 CHW 浮点张量。DXRT 以异步方式提交图像推理,因此采集和 GUI 更新无需等待每个 NPU 请求完成。

<!-- cell: t22-11 src: 2cc1224d10 -->
### 3.1 命令行选项

`AppOptions` 定义了模型路径、输入源、摄像头设置、跳帧间隔和 GUI 标志。如果省略 `--input`,应用程序将使用由 `--camera` 选择的摄像头。

<!-- cell: t22-13 src: d79b113959 -->
### 3.2 图像预处理

图像在保持宽高比的前提下缩放,然后中心裁剪为 224 x 224,从 BGR 顺序转换为 RGB 顺序,进行归一化,并以 CHW 布局存储。

<!-- cell: t22-15 src: 160edac714 -->
### 3.3 文本编码与缓存

`TextEncoder` 对每个查询进行分词并运行 ONNX 文本编码器。`TextFeatureStore` 将归一化后的嵌入向量保存在 `.cache/text_features_cpp/` 下,以模型和文本列表作为键。

<!-- cell: t22-17 src: dd4b9fd45c -->
### 3.4 异步图像编码

`ImageEncoderAsync` 创建一个带有多个缓冲区的 DXRT 推理引擎。每个已提交的帧都拥有自己的输入内存,直到完成回调复制出 768 维图像嵌入向量为止。

<!-- cell: t22-19 src: f266352d14 -->
### 3.5 相似度排名

在可选的 L2 归一化之后,应用程序为每个文本查询计算一个点积。对于归一化后的嵌入向量,这就是余弦相似度。GUI 会高亮显示超过所配置分数阈值的最强匹配项。

<!-- cell: t22-21 src: 39b6816859 -->
## 4. 检查 DXNN 图像编码器

当模型可用时,`dxparse` 会显示模型的输入和输出张量。应用程序期望形状为 `[1, 3, 224, 224]` 的浮点输入和一个 768 维图像嵌入向量。请注意任务顺序: 该模型以 `cpu_0` 任务开始(patch 嵌入在 CPU 上运行)并在 NPU 上结束,因此必须使用 `--use-ort`,且 CPU 部分位于每次请求的关键路径上。随后 `dxrun` 给出单模型基线,用于与 GUI 进行比较。

<!-- cell: t22-23 src: b823314d41 -->
## 5. 构建应用程序

`build.sh` 以 Release 模式配置 CMake,并使用 `nproc` 报告的全部 CPU 核心运行 `make`。使用 `./build.sh --clean` 可先删除现有的构建目录;对于位于自定义前缀下的库,使用 `./build.sh -DONNXRUNTIME_ROOT=<prefix>` 或 `-DCMAKE_PREFIX_PATH=<prefix>`。构建不需要模型或视频文件。

<!-- cell: t22-26 src: 54770cb65b -->
## 6. 运行摄像头演示

默认输入源是 `/dev/video0`,请求分辨率为 1280 x 720、帧率为 30 FPS。在资源和摄像头准备就绪后,设置 `RUN_CAMERA = True`。GUI 会阻塞单元格,直到窗口关闭。

<!-- cell: t22-28 src: 2048774b7b -->
## 7. 运行视频演示

`run_video.sh` 使用 `assets/videos/CLIP-demo.mp4`,并在文件结束时循环播放。在资源准备就绪后,设置 `RUN_VIDEO = True`。

<!-- cell: t22-30 src: 021a013221 -->
## 8. 实用选项

- `--skip-frames N` 每 `N + 1` 帧运行一次图像推理: `0` 表示对每一帧编码,`2`(默认值)表示每三帧编码一次。值越大,NPU 和 CPU 负载越低;排名的更新频率也随之降低。
- `--no-normalize` 禁用相似度计算前的 L2 归一化。
- `--camera SOURCE` 在省略 `--input` 时选择摄像头。
- `--input SOURCE` 接受摄像头索引、摄像头设备或视频路径。
- `--full-screen` 以全屏模式打开 Qt 窗口。
- `--exit-btn` 添加一个 Exit 按钮。

按 `Esc` 或 `Q` 关闭 GUI。运行脚本包含默认的文本查询;如果需要完全自定义的查询列表,请直接运行 `build/clip_single`。

<!-- cell: 8d0c598f-eb24-4c6c-8238-f1ea3b1878b0 src: 73c3cfa45c -->
## 9. 使用自定义文本查询

运行脚本提供了适合各自任务的默认文本查询。若只想使用自己的查询,请直接运行可执行文件:

```bash
cd notebooks/T22-demo-clip-single/app
./build/clip_single \
    --texts "A person" "A bicycle" "A cup" \
    --camera /dev/video0 \
    --skip-frames 2 \
    --full-screen \
    --exit-btn
```

使用自定义视频:

```bash
./build/clip_single \
    --texts "Cars are driving" "An empty road" \
    --input /path/to/video.mp4 \
    --exit-btn
```

运行 `./build/clip_single --help` 查看所有选项。按 `Esc` 或 `Q`,或点击 Exit 按钮,即可关闭应用程序。

<!-- cell: 666c080f-6927-4134-9841-3a3cb4999793 src: 3fcbd3e27f -->
## 10. 故障排除

| 症状 | 看到的内容 | 可能原因 | 解决方法 |
|---|---|---|---|
| CMake 找不到 ONNX Runtime | `Could not find ONNXRUNTIME` | 未安装 ONNX Runtime C++ 包,或它不在默认前缀下 | 安装它(DX-Runtime 提供)或运行 `./build.sh -DONNXRUNTIME_ROOT=<prefix>` |
| CMake 找不到 DXRT | `Could not find a package configuration file provided by "dxrt"` | 未安装 DX-RT,或它位于自定义前缀下 | 安装 DX-Runtime(教程 01 第 3 节)或运行 `./build.sh -DCMAKE_PREFIX_PATH=<prefix>` |
| 图像编码器在加载时被拒绝 | DX-RT 版本或格式错误 | 该 DXNN 由 DX-COM 2.2.1 针对较旧的运行时编译 | 使用本教程验证过的 DX-RT 版本(3.4.2) |
| 下载缓慢或中断 | `get_resources.sh` 运行很长时间 | 归档约 1.4 GB | 等待其完成;重新运行单元格会从 `.part` 文件继续下载 |
| 无法打开摄像头 | `cannot open /dev/video0` | 没有摄像头或没有权限 | `v4l2-ctl --list-devices`;用 `--camera` 传入其他设备 |
| Qt 窗口未出现 | `could not connect to display` | 没有图形会话 | 在桌面会话中运行 |

<!-- cell: t22-31 src: e079ee85d4 -->
## 11. 总结

本应用程序在一个 C++ GUI 中结合了 CPU 文本编码器和异步 NPU 图像编码器。文本特征可重复使用,图像特征持续生成,归一化后的点积无需任务专用分类器即可实现零样本的文本-图像匹配。

### 11.1 完成清单

- [ ] 下载图像编码器、文本编码器和词表
- [ ] 使用 `dxparse` 检查图像编码器
- [ ] 构建应用程序并运行 `--help`
- [ ] 使用视频或摄像头运行演示
- [ ] 使用自己的文本查询运行 `clip_single`

> **下一步:** 继续学习教程 23,在 C++ 应用程序中从摄像头跟踪手部关键点。
