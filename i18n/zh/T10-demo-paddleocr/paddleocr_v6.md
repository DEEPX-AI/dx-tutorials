<!-- i18n source: notebooks/T10-demo-paddleocr/paddleocr_v6.ipynb -->
<!-- i18n lang: zh -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: t10-v6-001 src: b22c49c292 -->
# DEEPX Tutorial 10 - 在 DEEPX NPU 上运行 PP-OCRv6

本教程介绍 PP-OCRv6,并在 DEEPX NPU 上演示从 ONNX 到 DXNN 的完整 OCR 工作流程。选择 PP-OCRv6_tiny 作为目标层级,是因为它能缩短编译时间和课堂上的周转时间。

工作流程分为若干可独立检查的阶段:

1. 下载原始 ONNX 模型;
2. 将动态输入维度替换为固定形状;
3. 验证固定形状模型保留了 ONNX 的结果;
4. 将模型编译为 DXNN;
5. 将检测和文本识别作为一个应用运行。

<!-- cell: t10-v6-002 src: f4377ce97d -->
## 学习目标

完成本教程后,您将能够:

- 解释 PP-OCRv6 架构,并在 tiny、small 和 medium 层级之间做出选择;
- 解释检测 → 识别的 OCR 流水线;
- 解释为什么一个动态识别模型会变成六个固定形状的 DXNN 模型;
- 安全地下载并验证源 ONNX 文件;
- 创建预处理与应用一致的校准配置;
- 使用 <code>dxcom</code> 编译模型,且不隐藏失败;
- 检查生成的 DXNN 产物;
- 运行经过审查的摄像头应用,并识别 OCR 精度的实际限制。

<!-- cell: t10-npu-pattern src: deed40455a -->
## 本应用如何使用 NPU

| 项目 | 在本教程中 | 学习来源 |
|---|---|---|
| 引擎 | 六个 `InferenceEngine` 对象常驻内存:一个检测器和五个识别器,按每个文本裁剪区域的宽高比选择 | T06-2 §3 |
| 执行 | 检测同步运行(`run()`);识别器用 `run_async()` 处理一帧中的每个裁剪区域,并用 `wait()` 收集结果 | T06-2 §4, §5 |
| 任务图 | 每个编译后的模型都有一个 NPU 任务,后接一个 `cpu_0` 任务(ONNX Runtime 执行 DX-COM 留在 CPU 上的算子) | T06-1 §6 |
| 每次请求的输入 | DET `[1, 640, 640, 3]` UINT8 = 1.2 MB;REC 裁剪区域高 48 像素,宽 120 到 1200 像素 | T06-3 §4 |
| 测量内容 | 检测器和一个识别器的 `dxrun` 吞吐量(6.3 节)与应用的每帧时间对比:一帧中单词越多,REC 请求就越多,因此应按文本数量报告延迟 | T06-1 §5 |

<!-- cell: 659aa80e-30b9-4cf8-ae33-1ed4bf0cbd53 src: cf3775d539 -->
## 前提条件

- 已完成教程 01: DX-COM、配有 DEEPX NPU 的 DX-RT,以及 `uv`。
- 下载量: 来自 Hugging Face 的两个 ONNX 模型(1.8 MB 和 4.5 MB),以及来自 `cs.deepx.ai` 的一个校准数据压缩包(110 MB)。
- 耗时: 约 15 分钟。第 6 节的六次 `dxcom` 编译每次只需几秒(检测器约 10 秒)。7.1 节会在 `workspace/.venv-ocr` 下创建一个 90 MB 的 Python 环境。
- 已在 DX-RT 3.4.2 和 DX-COM 2.4.1 上验证;模型在本教程中编译,因此始终与您的编译器匹配。
- 第 8 节的摄像头应用需要一个 USB 摄像头和一个显示器;`--check` 冒烟测试无需二者即可运行。

下一个单元格会定位 SDK、检查这些要求并打印状态表。任何标记为 `MISSING` 的项目都会附上提供它的步骤。

<!-- cell: t10-v6-overview src: fd7df8aac3 -->
## 1. PP-OCRv6 概述

### 1.1 什么是 OCR?

**光学字符识别**(OCR)是一种将不同类型的文档(扫描的纸质文档、PDF 文件或数码相机拍摄的图像)转换为可编辑、可搜索数据的技术。

可以把它理解为给您的 AI 装上"眼睛"。它通常以两步流水线的方式工作:
1. 文本检测: 定位图像中文本的位置(在其周围画出一个框)。
2. 文本识别: 辨认该框内的字符是什么。

<!-- cell: 5b35baad-56a0-4995-aaa9-9b39b7510f74 src: e4261a9a39 -->
### 1.2 什么是 PP-OCRv6?

PaddleOCR 是百度基于 PaddlePaddle 框架开发的超轻量级开源 OCR 系统。

[PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR) 是 PaddleOCR 通用 OCR 模型系列的最新一代。它在文本检测和文本识别中都使用新的 **PPLCNetV4** 骨干网络,并提供从边缘设备到服务器的三个部署层级。

[官方 PP-OCRv6 技术文档](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)中描述的主要改进包括:

- **一个可扩展的模型系列:** tiny、small 和 medium 层级的参数量从 1.5M 到 34.5M 不等;
- **统一的多语言识别:** small 和 medium 在一个模型中支持 50 种语言,而 tiny 支持 49 种语言,不含日语;
- **改进的文本检测:** RepLKFPN 在减少 neck 参数的同时扩大了感受野;
- **改进的文本识别:** EncoderWithLightSVTR 结合了局部上下文和全局注意力,而 CTC 提供高效的并行解码;以及
- **特殊场景覆盖:** 官方评估包含手写体、旋转和艺术字、数码显示屏、点阵字符、轮胎印字以及其他工业文本。

<img src="assets/ppocrv6-backbone.jpg" style="max-width: 1000px; width: 100%;" alt="用于 PP-OCRv6 识别和检测的 PPLCNetV4 骨干网络设计">

*PPLCNetV4 使用任务自适应下采样:识别保留水平序列信息,而检测生成多尺度特征图。来源:[PaddleOCR PP-OCRv6 官方文档](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.md)。*

<!-- cell: t10-v6-tier-comparison src: 998aa1c53c -->
### 1.3 tiny、small 和 medium

各层级采用相同的 PP-OCRv6 设计,只是规模不同。更大的层级通常能提高困难样例的精度,但会增加模型大小和计算成本。

| 层级 | 参数量 | 目标场景 | 检测 Hmean (%) | 识别精度 (%) | Intel Xeon OpenVINO(秒/图) | NVIDIA A100 PaddlePaddle(秒/图) |
|---|---:|---|---:|---:|---:|---:|
| **tiny** | 1.5M | 边缘 / IoT | 80.6 | 73.5 | **0.20** | **0.13** |
| **small** | 7.7M | 移动端 / 桌面 | 84.1 | 81.3 | 0.59 | 0.25 |
| **medium** | 34.5M | 服务器 / 最高精度 | **86.2** | **83.2** | 1.40 | 0.29 |

> **如何阅读此表:** 精度值来自 PaddleOCR 内部的多场景基准测试。速度是在 200 张通用和文档图像上端到端的每图秒数,包含图像 I/O、预处理、后处理和推理。这些 CPU/GPU 数值用于说明各层级之间的相对取舍;它们**不是 DEEPX NPU 的测量结果**,也不能预测 DX-COM 的编译时间。

<img src="assets/ppocrv6-performance.png" style="max-width: 1100px; width: 100%;" alt="PP-OCRv6 官方检测与识别精度对比">

*左: 平均文本检测 Hmean。右: 加权平均文本识别精度。来源:[PaddleOCR PP-OCRv6 官方文档](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.md)。*

<!-- cell: t10-v6-tiny-selection src: c7b6b6fe09 -->
### 1.4 本教程为何选择 PP-OCRv6_tiny

本教程选择 **tiny** 层级,以尽量缩短课堂上模型准备和编译的时间。其 1.5M 的参数规模也使它成为边缘和 IoT 部署的自然起点。这是一个基于教程时间的决定,并不意味着 tiny 是最佳的生产模型。当实测的精度提升足以抵消额外的资源消耗时,请选择 small 或 medium。

tiny 做了两个重要的取舍:

1. 它的官方检测和识别精度低于 small 和 medium;以及
2. 它支持 49 种语言,不含日语,而 small 和 medium 支持 50 种。

> 第 4 节下载官方 PP-OCRv6_tiny 检测器和识别器 ONNX 模型。随后 7.3 节会在现场演示之前检查应用的识别字典、预处理和张量契约是否与这些模型匹配。

<img src="assets/ppocrv6-detection-comparison.jpg" style="max-width: 1000px; width: 100%;" alt="PP-OCRv6 medium 在工业和困难文本上的官方文本检测对比">

*这张官方定性对比图使用 PP-OCRv6_medium,展示了困难的检测场景;它不是 tiny 层级的精度结果。来源:[PaddleOCR PP-OCRv6 官方文档](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.md)。*

<!-- cell: t10-v6-006 src: 243e5834d1 -->
## 2. 准备教程环境

本笔记本通过共享的教程路径助手读取 <code>config.json</code>。它不假设 SDK 安装在教程仓库内部。

<!-- cell: t10-v6-008 src: a5b14575cb -->
### 2.1 将 Python 包安装到当前的 uv 环境中

Jupyter 环境由 uv 创建,可能不包含 <code>pip</code> 模块。因此下一个单元格使用 <code>uv pip install --python ...</code> 而不是 <code>%pip</code>。

重复运行该单元格是安全的:uv 会复用已经满足要求的包。

<!-- cell: t10-v6-004 src: 33b7724658 -->
## 3. 理解 OCR 流水线

PP-OCRv6 分两个模型阶段运行。同样的"先检测后识别"流程适用于每个层级(tiny、small、medium)。

| 阶段 | 输入 | 输出 | 用途 |
|---|---|---|---|
| 文本检测 | 完整图像,640 × 640 | 文本多边形 | 找到文本区域 |
| 文本识别 | 一个文本裁剪区域,高 48 | 字符序列 | 将像素转换为文本 |

<img src="assets/ocr-workflow.jpg" style="max-width: 980px;" alt="PP-OCR 工作流程">

要将 PaddleOCR 应用到 DX NPU,需要以下 4 个步骤:

1. 下载 PaddleOCR ONNX 模型

2. 固定动态输入形状

3. 将 ONNX 编译为面向 DX NPU 的 *.dxnn

4. 使用 DEEPX-SDK 实现 OCR 应用

<!-- cell: t10-v6-010 src: 1523e85531 -->
## 4. 下载并检查资源

### 4.1 下载 ONNX 模型

本教程直接从 Hugging Face 上的 PaddlePaddle 官方 ONNX 仓库下载 **tiny** 检测器和识别器:

- [PP-OCRv6_tiny_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx)
- [PP-OCRv6_tiny_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx)

只下载 **DET** 和 **REC** 两个 ONNX 模型。可选的文本行方向分类器在本教程中不使用。

为了使教程结果可复现,每个 URL 都固定到官方仓库的特定修订版本。下载使用 HTTPS 证书验证,先写入临时的 <code>.part</code> 文件,并在替换目标文件之前验证 SHA-256 校验和。已存在且匹配的文件会被复用;过期或不同的文件会重新下载。

<!-- cell: t10-v6-calibration-download src: fcfdb536d1 -->
### 4.2 下载校准数据集

将 OCR 校准图片下载并解压到本教程目录中。该压缩包会创建以下目录:

```text
models/
├── det_dataset/
└── rec_dataset/
```

压缩包文件充当完成标记。如果 <code>ocr-dataset.tar.gz</code> 已经存在,下载和解压都会被跳过。

<!-- cell: t10-v6-013 src: d18bf1f9fc -->
源模型中的 batch size 和图像尺寸都是动态的。

动态形状对通用的 ONNX Runtime 应用很有用,但每次 DEEPX 编译都需要一个具体的输入形状。

<!-- cell: t10-v6-014 src: f5160e4612 -->
## 5. 固定动态输入形状

下表是固定形状模型的唯一权威来源。形状采用 NCHW 顺序:batch、通道、高度、宽度。

<!-- cell: b7060f4d-1c5c-4ac6-b461-2eb3a5598a95 src: e31dd6c875 -->
### 5.1 固定文本识别模型的输入形状

**为什么需要 5 个不同的模型?**

由于 NPU 需要固定的输入形状,如果把一个短单词(如 'Hi')和一个长句子放进同一个 1200 宽的框中处理,会导致过多的填充和细节丢失。

通过创建不同宽高比的"分桶",我们可以更高效、更准确地处理文本。这就是本教程使用五个不同宽高比的独立识别模型(外加一个检测模型,共六个 DXNN 文件)的原因。

对于每种情况,我们选择并应用与检测到的文本比例最匹配的模型。

<img src="assets/ocr-ratio.png" style="max-width: 800px;">

下面的 gif 动画展示了根据实际检测到的文本比例匹配到哪个文本识别模型。

<img src="assets/ocr-ratio.gif" style="max-width: 800px;">

<!-- cell: t10-v6-016 src: 231600d020 -->
### 5.2 使用 `onnxsim` 固定动态输入形状

笔记本通过当前内核的 Python 解释器调用 ONNX Simplifier。例如,第一次转换等价于:

~~~bash
python -m onnxsim models/det.onnx models/det_fixed.onnx           --overwrite-input-shape x:1,3,640,640
python -m onnxsim models/rec.onnx models/rec_fixed_ratio_2_5.onnx --overwrite-input-shape x:1,3,48,120
python -m onnxsim models/rec.onnx models/rec_fixed_ratio_5.onnx   --overwrite-input-shape x:1,3,48,240
...
~~~

已存在且非空的固定形状模型会被检查并复用。

<!-- cell: t10-v6-018 src: 5fa1f3bd19 -->
### 5.3 验证形状和数值等价性

仅做结构验证是不够的。接下来的单元格:

1. 检查每个固定形状的 ONNX 模型;
2. 确认其精确的输入形状;
3. 比较检测模型和两个有代表性的识别分桶的 ONNX Runtime 输出。

所有识别分桶都来自同一个源图。数值识别检查使用 ratio-2.5 和 ratio-5 模型,以保持教程验证的快速。

<!-- cell: t10-v6-021 src: 803f189d08 -->
## 6. 将 ONNX 编译为面向 DX NPU 的 *.dxnn

### 6.1 创建 DX-COM 校准配置

校准预处理必须与应用预处理一致。颜色顺序、缩放、归一化或布局的不匹配即使在编译成功的情况下也会降低精度。

| 模型 | 校准图片 | 缩放尺寸 | 均值 / 标准差 |
|---|---|---:|---|
| 检测 | <code>det_dataset</code> | 640 × 640 | ImageNet 值 |
| 识别 | 对应的比例分桶 | 固定宽度 × 48 | [0.5, 0.5, 0.5] |

<!-- cell: t10-v6-024 src: b5cb9cbaae -->
编译前请至少检查一个配置。特别要检查 NCHW 输入形状、校准数据集、缩放尺寸、通道顺序、归一化和转置顺序。

<!-- cell: t10-v6-026 src: 7df5a7156c -->
### 6.2 将 ONNX 模型编译为 DXNN

每个模型写入各自的目录。已存在的有效 DXNN 文件会被跳过,这样重复上课时不必花时间重新编译。

单元格会打印模型名称,然后对每个模型精确地运行以下命令:

~~~bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
dxcom -m <fixed-model.onnx> \
      -c <calibration-config.json> \
      -o <output-directory> \
      --gen_log \
      --export_html
~~~

编译器输出保持可见,编译失败时单元格会报错停止,而不是静默地继续。

<!-- cell: t10-v6-029 src: 9a976ceadb -->
### 6.3 验证编译产物

一个完整的 OCR 应用恰好需要六个 DXNN 文件:检测模型和五个识别分桶。如果缺少任何必需文件,单元格会提前失败。

<!-- cell: t10-v6-031 src: f143ae02c1 -->
使用 <code>dxparse</code> 检查两种模型角色。识别 ratio 2.5 具有代表性;其他识别文件的差异主要在于固定输入宽度。

<!-- cell: t10-dxrun-baseline-md src: 97352c10c3 -->
为同样的两个模型测量一个 CLI 基线。由于两个图都以 CPU 任务结尾,因此必须加 `--use-ort`;`-v` 会增加延迟分解。稍后把这些数值与应用进行比较:摄像头循环每帧运行一次 DET 请求,并为每个检测到的单词运行一次 REC 请求,因此其帧时间会随文本数量增长。

```bash
dxrun -m <workspace>/outputs/paddleocr_v6/det_fixed/det_fixed.dxnn --use-ort -t 3 -v
dxrun -m <workspace>/outputs/paddleocr_v6/rec_fixed_ratio_2_5/rec_fixed_ratio_2_5.dxnn --use-ort -t 3 -v
```

<!-- cell: t10-v6-033 src: 877fbbedf2 -->
## 7. 使用 DEEPX-SDK 测试 OCR 应用

可复用的 Python 应用位于 <code>app/</code>。它使用 <code>outputs/paddleocr_v6/</code> 中编译好的 PP-OCRv6 tiny 模型,并按如下方式处理每一帧:

<img src="assets/ppocrv6-camera-pipeline.png" alt="PP-OCRv6 摄像头推理流水线" style="max-width: 1000px; width: 100%; height: auto;">

检测器和五个识别引擎只加载一次并保持常驻。

<!-- cell: t10-v6-034 src: d8ac433600 -->
### 7.1 安装应用依赖

该应用需要 <code>dx_engine</code>、OpenCV、NumPy 和 Pillow。与教程 06-2 一样,它们被安装到 <code>workspace/.venv-ocr</code> 下的一个小型环境中,而不是共享的 Jupyter 环境:DX-RT 包在 <code>/usr/share/libdxrt-bin/python</code> 下提供预构建的 <code>dx_engine</code> wheel,单元格会选择 <code>cpXY</code> 标签与内核 Python 匹配的那个。<code>app/run_camera.sh</code> 使用同一个环境。重复运行该单元格是安全的;它会跳过已存在的内容。

```bash
uv venv --python <jupyter-python> <T10>/workspace/.venv-ocr
uv pip install --python <T10>/workspace/.venv-ocr/bin/python -r app/requirements.txt \
    /usr/share/libdxrt-bin/python/dx_engine-<version>-cp<XY>-*.whl
```

<!-- cell: t10-v6-036 src: 507594f64b -->
### 7.2 应用结构与安全保障

笔记本现在使用 <code>app/</code> 下维护的文件,而不是再生成一个运行器。

| 文件 | 职责 |
|---|---|
| <code>app/ocr_engine.py</code> | DXNN 加载、DET 后处理、裁剪区域提取、五分桶 REC 路由和 CTC 解码 |
| <code>app/camera_app.py</code> | CLI 校验、640×480 摄像头采集、OpenCV 预览、BBOX 和多语言文本叠加 |
| <code>app/run_camera.sh</code> | 使用 <code>workspace/.venv-ocr</code> 启动应用(可用 <code>PYTHON_BIN</code> 覆盖) |
| <code>app/assets/ppocrv6_tiny_dict.txt</code> | 与 6906 类 REC 输出匹配的官方 tiny 字典 |

对于缺失的模型、无效的张量形状、不匹配的字典或摄像头打开/读取失败,应用会明确报错。已有的 DXNN 文件绝不会被修改。

<!-- cell: t10-v6-037 src: 90057954f1 -->
### 7.3 运行无需摄像头的应用冒烟测试

<code>--check</code> 模式不会打开摄像头,也不会创建 GUI 窗口。它加载打包好的应用,并通过检测器和全部五个识别器运行一次合成输入推理。这会在现场演示之前验证 NPU 运行时、输入/输出张量契约以及 tiny 字典的类别数。

下面执行的命令等价于带显式模型和字典路径运行 <code>workspace/.venv-ocr/bin/python app/camera_app.py --check</code>。

<!-- cell: t10-v6-039 src: f2bee9052c -->
### 7.4 测试 640×480 摄像头应用

实时应用会打开一个交互式 OpenCV 窗口。请在 **JupyterLab 终端**中运行它,这样笔记本内核不会被阻塞,摄像头也能被干净地释放。`run_camera.sh` 使用 7.1 中创建的 `workspace/.venv-ocr` 环境启动它。

默认输入为 <code>/dev/video0</code>,640×480,15 FPS。在预览窗口中按 **q** 或 **Esc** 退出。使用 <code>--camera /dev/videoN</code> 选择其他设备。

<!-- cell: t10-v6-041 src: 4c25749e1a -->
<img src="assets/sc-ocr-app.png" style="max-width: 800px;" alt="预期的 PP-OCR 结果">

<!-- cell: t10-v6-042 src: 4c5e64d2e3 -->
## 8. 精度与性能检查清单

编译成功并不保证 OCR 精度可以接受。请使用有代表性的图像验证整个流水线。

| 检查项 | 重要性 | 实际措施 |
|---|---|---|
| 检测分辨率 | 小文本在 640 × 640 下可能消失 | 在预期的摄像头距离下比较召回率 |
| 校准覆盖范围 | 量化遵循校准统计量 | 包含真实的光照、字体、模糊和背景 |
| 预处理一致性 | 颜色或归一化不匹配会使模型输入偏移 | 保持编译和运行时预处理完全一致 |
| 识别分桶 | 过度缩放或填充会损害字符 | 测量文本宽高比分布 |
| 识别置信度 | 阈值过低会显示错误文本;阈值过高会丢失文本 | 在带标签的验证集上调优 |
| 透视与方向 | 摄像头拍摄的文档并不总是平整或正立的 | 必要时添加文档方向和去畸变处理 |
| 端到端延迟 | 一帧可能包含许多识别裁剪区域 | 按文本数量报告延迟,而不只是 FPS |

不要针对单个好看的摄像头样本进行调优。请保留一个固定的验证集,并为每次配置更改记录检测召回率、识别精度和端到端延迟。

<!-- cell: t10-v6-043 src: bf21a3d708 -->
## 9. 故障排除

| 症状 | 看到的内容 | 可能原因 | 解决方法 |
|---|---|---|---|
| `dxcom` 缺失 | 状态表中显示 `[MISSING] dx_com` | DX-COM 未安装,或 `config.json` 指向其他位置 | 完成教程 01 第 2 节,并检查 `python tutorial_paths.py --show` |
| 下载被拒绝 | `Checksum mismatch` 或 `Incomplete download` | 传输中断或上游文件已更改 | 只删除 `workspace/models` 下指定的模型,然后重新运行 4.1 节 |
| 固定形状模型的形状错误 | `AssertionError: ... expected [1, 3, 48, 240]` | 过期的固定形状 ONNX 文件 | 删除该固定形状 ONNX 文件并重新运行第 5 节 |
| 编译失败 | `Error Type: DataNotFoundError` 或 `dxcom` 的其他 `[ERROR]` 行 | `dataset_path` 错误,或没有列出扩展名的图片 | 将 `configs_v6/<model>.json` 与 6.1 中打印的目录进行比较;完整日志位于 `outputs/paddleocr_v6/<model>/compiler.log` |
| 7.1 中 `dx_engine` 导入失败 | `ModuleNotFoundError: No module named 'dx_engine'` 或 wheel/ABI 不匹配 | 环境是用其他 Python 创建的,或 DX-RT 已升级 | 删除 `workspace/.venv-ocr` 并重新运行 7.1;wheel 必须与已安装的 `libdxrt-bin` 匹配 |
| `--check` 报告缺少模型 | `Application test files are missing` | 六个 DXNN 文件没有全部编译 | 在 `SELECTED_COMPILES` 中包含所有模型并重新运行 6.2 |
| 摄像头无法打开 | `Could not open camera /dev/video0` | 没有摄像头、设备路径不同,或用户不在 `video` 组中 | `v4l2-ctl --list-devices`;传入 `--camera /dev/videoN`;执行 `sudo usermod -aG video $USER` 后重新登录 |
| OpenCV 显示错误 | `cannot open display` | 没有图形会话 | 在桌面会话中运行(具有有效 `DISPLAY` 的 JupyterLab 终端) |
| 未显示文本 | 有框但无文本,或没有框 | 光照、对焦或阈值 | 在有代表性的图像上调整 `--det-threshold`、`--box-threshold`、`--rec-threshold` |

<!-- cell: t10-v6-044 src: 7776b639fb -->
## 10. 总结

### 10.1 已完成的工作流程

```text
Official PP-OCRv6 tiny DET + REC ONNX
                    │
                    ▼
       Fix dynamic input dimensions
          ┌─────────┴─────────┐
          │                   │
   DET 640×640       REC width buckets ×5
          └─────────┬─────────┘
                    ▼
     Calibration configuration + DX-COM
                    │
                    ▼
           Six verified DXNN models
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 camera_app.py --check    640×480 camera
                              │
                              ▼
                   BBOX + recognized text
```

### 10.2 产物与验证一览

| 阶段 | 产物或命令 | 验证内容 |
|---|---|---|
| 源模型 | <code>det.onnx</code>、<code>rec.onnx</code> | 官方修订版本和 SHA-256 校验和 |
| 固定形状 ONNX | 一个 DET 和五个 REC 模型 | 静态输入形状和 ONNX 有效性 |
| 数值检查 | ONNX Runtime 比较 | 固定形状转换保留了有代表性的输出 |
| 校准 | <code>configs_v6/*.json</code> | 数据集、缩放尺寸、颜色顺序、归一化和布局 |
| 编译 | <code>dxcom</code> | 带可见日志和报告的 ONNX 到 DXNN 转换 |
| 运行时契约 | <code>camera_app.py --check</code> | DET 和全部五个 REC 模型按预期的张量契约执行 |
| 实时应用 | <code>app/run_camera.sh</code> | 640×480 采集、比例路由、CTC 解码、BBOX 和文本叠加 |

### 10.3 运行时模型映射

下面的形状是 DX-RT 在运行时报告的内容(`dxparse`、`get_input_tensors_info()`):batch、高度、宽度、通道(NHWC,UINT8)。第 5 节以 NCHW 浮点顺序固定了 ONNX 输入;DX-COM 把布局变换和归一化折叠进编译后的模型,这就是应用直接输入原始 HWC 像素的原因。

| 角色 | 固定输入 | 选择规则 |
|---|---:|---|
| DET | `[1, 640, 640, 3]` | 每个摄像头帧一次 |
| REC 2.5 | `[1, 48, 120, 3]` | 裁剪区域宽高比 ≤ 2.5 |
| REC 5 | `[1, 48, 240, 3]` | 裁剪区域宽高比 ≤ 5 |
| REC 10 | `[1, 48, 480, 3]` | 裁剪区域宽高比 ≤ 10 |
| REC 15 | `[1, 48, 720, 3]` | 裁剪区域宽高比 ≤ 15 |
| REC 25 | `[1, 48, 1200, 3]` | 裁剪区域宽高比 ≤ 25,或作为更宽裁剪区域的回退 |

### 10.4 完成清单

- [ ] 下载官方 PP-OCRv6 tiny DET 和 REC ONNX 模型
- [ ] 验证源模型校验和与 ONNX 结构
- [ ] 创建一个固定形状 DET 模型和五个固定形状 REC 模型
- [ ] 比较有代表性的源 ONNX 与固定形状 ONNX 的输出
- [ ] 创建校准配置并编译 DXNN 模型
- [ ] 检查编译产物并在 NPU 上测试全部六个模型
- [ ] 准备 DET 到 REC 的 OpenCV 摄像头应用
- [ ] 在带标签的验证集上测量检测和识别精度
- [ ] 使用有代表性的摄像头图像调整阈值
- [ ] 记录端到端延迟、吞吐量和主机 CPU 占用

### 10.5 尝试 small 或 medium 层级

要用更大的 PP-OCRv6 层级重复此工作流程,请从对应的 PaddlePaddle 官方 ONNX 仓库下载检测器和识别器:

| 层级 | 检测 ONNX 仓库 | 识别 ONNX 仓库 |
|---|---|---|
| **small** | [PP-OCRv6_small_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx) | [PP-OCRv6_small_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx) |
| **medium** | [PP-OCRv6_medium_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx) | [PP-OCRv6_medium_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx) |

<!-- cell: 87889eeb-8768-43a7-97e9-5c0805d6cd64 src: 78b7602a47 -->
### 10.6 如何提升 OCR 性能?

本教程没有实现 `Document Image Orientation Classification`、`Text Image Unwarping` 和 `Text Line Orientation Classification` 等预处理模块。

如果您希望获得更高的 OCR 性能,请实现这些缺失的 AI 模型并将其应用到 OCR AI 流水线中。

<img src="assets/ppocrv6-full-pipeline.png" style="max-width: 1200px;">

为了获得更高的精度:

- 使用来自**实际部署环境**的文本图像构建校准数据集。
- 覆盖真实的摄像头、距离、文本大小、宽高比、字体、语言、光照、模糊和透视。
- 当前的**校准数据集**有三个粗粒度的 REC 分桶:ratio 5、15 和 25。当部署场景中的文本形状差异很大时,请把这些数据集分桶划分得更细。
- 这些校准分桶不同于运行时的五个 REC 模型路由:ratio 2.5、5、10、15 和 25。
- 根据实测的部署数据选择固定形状,将动态的 DET 和 REC 输入转换为固定形状。
- 保持校准和运行时预处理完全一致。

| 部署数据 | 调整 |
|---|---|
| 介于当前分桶之间的比例 | 添加中间的 REC 分桶 |
| 高、窄或很长的文本 | 使用匹配的固定 REC 形状 |
| 非常小的文本 | 评估更大的 DET 输入 |
| 不常见的摄像头宽高比 | 匹配 DET 输入的几何尺寸 |
| 缩放后出现畸变 | 调整形状和预处理 |

发布前:

- 在同一个带标签的数据集上比较 ONNX 和 DXNN 的精度。
- 测量检测召回率、识别精度、延迟、内存和 CPU 负载。
- 重新检查张量契约、预处理、校准和识别字典。
- 只有当新形状能带来可测量的价值时才添加;更多的形状会增加构建时间、存储和路由复杂度。

> **下一步:** 继续学习教程 20,在同一运行时上用 C++ 构建并运行多通道 YOLO 应用。
