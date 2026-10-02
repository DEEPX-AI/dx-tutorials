<!-- i18n source: notebooks/T03-E2E-AI-Workflow/e2e_ai_workflow.ipynb -->
<!-- i18n lang: zh -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 2dbf2d92-815e-4a2e-b468-26f71cd8dbe3 src: e823f81eaf -->
# DEEPX Tutorial 03 - 基于 DEEPX NPU 的 AI 项目工作流

<!-- cell: 74c3b426-7d6e-4844-bf75-737a94d1055f src: 7da7d09c6b -->
第三篇教程演示在 DEEPX 硬件上部署 AI 模型的完整端到端工作流。

我们将训练一个叉车与工人检测模型,使用 DX-Compiler 工具将其转换为 DXNN 格式,并在 DEEPX NPU 上运行最终的 AI 应用。通过这一过程,您将对 DEEPX NPU 开发流水线有一个全面的了解。 

<!-- cell: 38bde5ed-95d6-4750-a6fb-174273a70167 src: 7bfac3c725 -->
## 学习目标

完成本教程后,您将能够:

- 跟随一个完整的 AI 项目,从模型选择一直到 NPU 部署;
- 使用 `dxcom` 和 Model Zoo 的 JSON 配置编译自定义训练的 ONNX 模型;
- 说明所涉及的两个 JSON 文件(编译器配置和 DX-APP 运行时配置);
- 使用标准的 DX-APP 可执行文件,在图像和视频上运行自定义的双类别检测器。

<!-- cell: 445ad29e-2955-4f2e-beac-2e2a34bbe81f src: 6275be0099 -->
## 实践项目概览

<!-- cell: e37c1d96-a612-4246-af9c-ebf798355fba src: 08e68a9f0e -->
- **检测类别**: Forklift、Worker
- **基础 AI 模型**: YOLOv7
- **数据集**: 来自 [Kaggle](https://www.kaggle.com/datasets/hakantaskiner/personforklift-dataset/data) 的 1448 张叉车与工人图像
- **训练**: 一块显存至少 24 GB 的 NVIDIA GPU(训练需要高性能 GPU,但部署只需要 DEEPX NPU)
- **推理 NPU**: `DX-M1`
- **AI 应用**: 修改并复用 DX-APP 的 yolo 示例
- **预期输出**:
<img src="assets/detection-goal.jpg" style="max-width: 1200px;">

<!-- cell: e8217ad7-fc19-421b-967d-98ff2dfbc762 src: 73e0f10840 -->
## AI 工作流概览

<!-- cell: 955d81f9-0699-45e6-ba37-e27a481ad370 src: 2d9b410433 -->
此图说明了 AI 项目的常见工作流。

我们先定义目标,收集并标注数据,然后训练模型。
DX-Compiler 帮助模型变得更快、更轻量(INT8),以适配 DX NPU。
最后一步是使用 DX-APP 或 DX-STREAM 将模型部署到 DEEPX NPU。

每一步都朝着实际的 AI 解决方案推进,例如工人与叉车检测。

  <img src="assets/workflow2.jpg" style="max-width: 1200px;">

<!-- cell: 1b3d8491-f89b-4fbf-b601-2cfc061e96a5 src: 22621a4e9d -->
## 前提条件

- 已完成教程 01:已安装 DX-COM、DX-RT 和 DX-APP,且 NPU 可见。
- 第 4 节(编译)需要 DX-COM 和校准数据集,或者使用选项 B 下载预编译模型。
- 第 5 节需要一块 DEEPX NPU 以及用于显示视频结果的显示器。
- 训练(第 3 节)为可选项,需要一块 24 GB 显存的 NVIDIA GPU;Colab 笔记本的链接在该节给出。
- 下载量:从 `cs.deepx.ai` 下载 ONNX 模型(139 MB)、预编译的 DXNN(71 MB)和一段测试视频(19 MB)。可选的 YOLOv7 编译耗时超过 10 分钟。
- 本教程下载或生成的所有内容都位于 `notebooks/T03-E2E-AI-Workflow/workspace/`,该目录被 git 忽略;SDK 检出保持干净。

下一个单元格会定位 SDK、检查这些要求并打印状态表。任何标记为 `MISSING` 的项目都会附带提供它的步骤。

<!-- cell: 691e8f9d-0115-48d7-a02a-1b64c84c3b72 src: 92bb42d767 -->
## 1. AI 工作流 - 根据用例选择模型

<!-- cell: afa66140-dbee-4bf2-a9ce-82f4316a8512 src: 9e254de77f -->
要启动一个 AI 项目,我们需要选择一个适合用例的 AI 模型。

在本教程中,我们的目标是检测叉车和工人。
我们将使用 YOLOv7,一个广为人知的目标检测模型。

- 选择 YOLOv7 来检测 Forklift 与 Worker
- 有关 YOLOv7 的更多细节:[链接](https://docs.ultralytics.com/models/yolov7/)
- 如何使用 YOLOv7:[链接](https://github.com/WongKinYiu/yolov7)

<!-- cell: 0a81b7c5-ec05-412b-bc79-338f77d02b18 src: dbe5a538f7 -->
## 2. AI 工作流 - 数据准备与标注

<!-- cell: ac53d3cc-8142-4d59-ac44-fb7f2f836e74 src: 0be5dc9c10 -->
从 Kaggle 下载已标注的叉车-人员数据集:
 - 参考:[Kaggle 链接](https://www.kaggle.com/datasets/hakantaskiner/personforklift-dataset)

<!-- cell: e071e952-82bb-4cf9-8c88-6a007b150d3b src: 0aadb5e072 -->
## 3. AI 工作流 - 训练

<!-- cell: cbd31d3c-e849-49d3-b7f4-10efefca0861 src: 88b6d45076 -->
为了高效地训练模型,应使用显存为 24GB 或更多的 GPU。

 - 如何训练 YOLOv7:[链接](https://colab.research.google.com/drive/1dAdjJuhXqFM_Qcd0QqAn7_AGx7abA5aX?usp=sharing)

训练在该 Colab 笔记本中完成,而不是在这里。其结果,即导出的 ONNX 文件 `yolov7-forklift-person.onnx`,在第 4 节提供下载,因此没有 GPU 也可以继续。训练中有一个细节会延续下来:数据集标签中的类别顺序为 `0 = Forklift`、`1 = Worker`,5.1 节的 `class_names` 必须严格按此顺序列出。

<!-- cell: f7dd943a-796f-4aec-8341-a524c71c4e2f src: 9802354bde -->
## 4. AI 工作流 - 使用 DX-Compiler 进行优化

<!-- cell: 6347d2b0-3fd8-4a96-bc0e-84c5885586eb src: 70970cc292 -->
让我们把预训练的 AI 模型编译为 DXNN 格式。

总体流程如下:
1. 获取一个基于 pytorch 框架的预训练模型
2. 将其转换为 ONNX 格式
3. 将 ONNX 编译为 DXNN(有关 DX-Compiler 的更多细节,请参阅[此处](https://developer.deepx.ai/download/?id=581)的用户指南
                                                                                             
> **注意:** 下载用户指南前,必须先登录 https://developer.deepx.ai/。

<img src="assets/dx-com-workflow.jpg" style="max-width: 1200px;">

<!-- cell: bdca7205-5cef-453c-863a-794925e61820 src: 0df77eb438 -->
DX-Compiler 的源码结构组织如下:
```bash
dx_com
 ├── calibration_dataset   # Dataset used to optimize model accuracy
 └── sample_models         # Sample configuration file and ONNX files 
```

<!-- cell: 6655c5ab-7718-4a75-a305-54516a723d1d src: eb8bb286ef -->
### 4.1 准备导出的 ONNX 模型和 YOLOv7 DX-Compiler 配置

<!-- cell: a2886542-8a78-4e17-801b-121deb3b41e0 src: 9e9aa01488 -->
自定义的 YOLOv7 ONNX 模型会直接下载。本教程还附带了一份 YOLOv7 Q-Lite JSON 配置的副本,这样笔记本就可以从头到尾运行,而无需假定浏览器的下载位置。

原始配置来自 [DEEPX Model Zoo](https://developer.deepx.ai/modelzoo/):Object Detection、YOLOv7、Q-Lite JSON 下载。

<img src="assets/sc-modelzoo-yolov7.png" style="max-width: 1000px;">

  
  **注意:** DEEPX Model Zoo 提供经 DEEPX 验证的 AI 模型和编译器配置。

<!-- cell: acb10634-544d-4ace-ba8a-0fd870fbf87e src: e1d203991b -->
### 4.2 针对您的自定义环境修改 YOLOv7 json 文件

<!-- cell: 97b44e9e-20ad-4a1a-be68-321b3e981ec2 src: 618757da53 -->
### 4.3 参照 json 配置将 ONNX 编译为 DXNN

<!-- cell: t03-dxcom-environment-guide src: 8e71d364a7 -->
DX-Compiler 使用在 SDK 安装期间创建的专用 Python 虚拟环境。JupyterLab 运行在独立的 `dx-tutorials` 环境中,因此 `dxcom` 通常不在笔记本的 `PATH` 中。

下一个代码单元格从 `config.json` 推导出 DX-Compiler 的路径,激活 DX-Compiler 环境,进入 `dx_com` 工作区,并在同一个 shell 中运行 `dxcom`。每个 `!` 代码单元格都会启动一个新的 shell,因此激活不会在单元格之间保持。因此,每个运行 `dxcom` 的笔记本单元格都会重复激活命令。

在本教程之外,请打开终端并激活一次环境:

```bash
cd <DX_ALL_SUITE_DIR>/dx-compiler
source venv-dx-compiler-local/bin/activate
cd dx_com
dxcom -h
```

激活后的环境在该终端中会一直保持激活状态,直到您运行 `deactivate` 或关闭终端。

<!-- cell: 87678442-bbd9-47c0-a4cd-631987888e24 src: aed9a3d45a -->
从两种获取 DXNN 的方式中选择一种。在下一个单元格中设置 `COMPILE_OPTION`:`"A"` 表示自行编译 ONNX(超过 10 分钟),`"B"` 表示下载 DEEPX 用同一命令编译好的 DXNN(71 MB)。当工作区中已存在 DXNN 时,两个单元格都会自动跳过。

<!-- cell: 39c4982e-a87b-48c0-8a3f-1ed63295c4cd src: cfeb99e9ac -->
#### 4.3.1 选项 A:使用 `dxcom` 编译 ONNX

这与教程 01 的 2.2.2 节中的命令相同,只是换成了自定义 ONNX 和编辑后的 JSON。`-o` 指向工作区,因此不会向 SDK 目录树写入任何内容。

```bash
cd <DX_ALL_SUITE_DIR>/dx-compiler
source venv-dx-compiler-local/bin/activate
cd dx_com
dxcom -m <workspace>/yolov7-forklift-person.onnx -c <workspace>/yolov7-forklift-person.json -o <workspace>/output --gen_log
```

> **注意:** 根据主机不同,编译耗时**超过 10 分钟**。该单元格仅当 `COMPILE_OPTION` 为 `"A"` 时才会运行。

<!-- cell: ddb0f169-582f-4a96-9c49-d8325df7e616 src: 6d1d5efeaf -->
#### 4.3.2 选项 B:下载预编译的 DXNN

DEEPX 用同一份 JSON 编译了同一个 ONNX 并发布了结果。下载它可在几秒内得到完全相同的文件。

```bash
cd <workspace>/output
wget -nc https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/yolov7-forklift-person.dxnn
```

<!-- cell: 6daf1b20-9182-4a77-82d0-8188d9c6e257 src: c0768f5b3c -->
## 5. AI 工作流 - 在 DEEPX NPU 上部署

<!-- cell: 4d54f93a-31ff-4f9c-8fd5-1de4776cc622 src: f2166618d1 -->
### 5.1 为自定义 YOLOv7 模型配置 DX-APP

当前的 DX-APP 不需要为自定义类别数修改 C++ 源码。YOLOv7 工厂从运行时配置文件读取 `num_classes` 和 `class_names`,后处理器在解码检测结果时使用这些值。

> **参考:** 本节遵循 [DX-APP YOLO Customizing Guide](https://github.com/DEEPX-AI/dx_app/blob/main/docs/source/docs/12_DX-APP_YOLO_Customizing_Guide.md#step-3-tune-configjson) 的 **Step 3, "Tune `config.json`"**(位于您检出的 `dx_app/docs/source/docs/12_DX-APP_YOLO_Customizing_Guide.md`)。该指南列出了 YOLO 后处理器读取的每个键(`obj_threshold`、`score_threshold`、`nms_threshold`、`num_classes`、`class_names`、`anchors`、`strides`),并展示了一个三类别模型的同类文件。随附的 YOLOv7 示例在 `src/cpp_example/object_detection/yolov7/config.json` 中提供默认值;下面写入的文件会针对双类别的 Forklift 和 Worker 模型覆盖这些默认值。

请勿混淆本教程中使用的两个 JSON 配置文件:

| 配置 | 使用者 | 用途 |
|---|---|---|
| `yolov7-forklift-person.json` | `dxcom -c` | 描述模型编译、输入形状、预处理和校准数据。 |
| `yolov7-forklift-runtime.json` | DX-APP `--config` | 描述运行时的后处理阈值、类别数和显示标签。 |

编译后的模型解码输出形状为 `[1, 25200, 7]`。每一行包含四个边界框值、一个目标置信度值和两个类别得分:`4 + 1 + 2 = 7`。原始检测头包含 21 个通道,因为每个检测头使用三个锚框:`3 × (5 + 2) = 21`。

<!-- cell: c8598424-1ee1-4413-9ecd-ab7f6fd6e2d6 src: 37fd0c267c -->
#### 运行时配置字段

| 字段 | 说明 |
|---|---|
| `obj_threshold` | 拒绝目标置信度得分低于此值的候选框。 |
| `score_threshold` | 拒绝最终类别置信度低于此值的检测结果。对于 YOLOv7,最终置信度基于目标置信度和类别得分。 |
| `nms_threshold` | 非极大值抑制(NMS)用于移除重叠检测结果的 IoU 阈值。值越低,移除重叠框越激进。 |
| `num_classes` | 模型输出的类别数。此模型输出两个类别,因此该值必须为 `2`。 |
| `class_names` | 每个类别 ID 显示的标签。数组顺序必须与训练时使用的类别顺序完全一致。 |

对于此模型,类别 ID `0` 为 `Forklift`,类别 ID `1` 为 `Worker`。仅修改标签文本不会改变模型行为;它只会改变检测到的类别 ID 的显示方式。`num_classes`、`class_names` 与模型输出之间的不匹配可能导致解码错误或标签错误。

> **注意:** 正如该指南所警告的,`num_classes` 必须与编译后模型的输出一致。类别数在编译时就已固定;仅修改运行时 JSON 无法修复张量形状不匹配的问题。

<!-- cell: 12ba0bdd-1df3-47d2-a454-a1bc39c8768c src: d725d9b9f9 -->
### 5.2 验证 YOLOv7 可执行文件

运行时配置通过 `--config` 在 DX-APP 创建 YOLOv7 后处理器之前加载。由于没有改动任何 C++ 源码,当 `bin/yolov7_async` 已存在时无需重新构建 DX-APP。

模型运行时,DX-APP 会记录 `Config loaded: yolov7-forklift-runtime.json (4 keys)`。该计数仅涵盖标量值;`class_names` 是一个列表,单独存储,因此全部五项设置都会生效。

如果可执行文件缺失,请在终端(File > New > Terminal)中只构建此目标:

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_app
./build.sh --target yolov7_async
```

<!-- cell: aef5fbb5-57c5-425b-bb18-be6c09f84008 src: 548c63ca47 -->
### 5.3 使用图像运行自定义模型

将自定义 DXNN 及其运行时配置传给标准的 YOLOv7 可执行文件;共享的 DX-APP 源码保持不变。该单元格以无头模式运行(`--no-display --save`),因此无需显示器也能工作,并在下方显示保存的结果。去掉 `--no-display` 则会改为打开 DX-APP 窗口;此时单元格会一直处于运行状态,直到窗口关闭。

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_app
./bin/yolov7_async -m <workspace>/output/yolov7-forklift-person.dxnn \
    --config <workspace>/yolov7-forklift-runtime.json \
    -i <tutorial>/assets/forklift-worker.png --no-display --save --save-dir <workspace>/outputs
```

<!-- cell: c6d86bb5-c77a-4144-8072-4c30ac0e0386 src: 2dbebc5c03 -->
### 5.4 使用视频运行自定义模型

视频运行使用同一命令,只是改用 `-v`。视频片段长 80 秒(2001 帧)。当 `SHOW_WINDOW = False`(默认值)时,应用以无头模式运行,将渲染后的视频写入工作区,并且**在完成之前不打印任何内容**;在 DX-M1 上大约需要 20 秒,因此单元格并没有卡住。保存的视频随后会在笔记本中显示。设置 `SHOW_WINDOW = True` 则改为在 DX-APP 窗口中实时观看(单元格会一直处于运行状态,直到窗口关闭)。

```bash
./bin/yolov7_async -m <workspace>/output/yolov7-forklift-person.dxnn \
    --config <workspace>/yolov7-forklift-runtime.json -v <workspace>/forklift-worker.mp4 \
    --no-display --save --save-dir <workspace>/outputs
```

<!-- cell: 0fefce5e-afd0-44ab-a03f-ed3a18e92272 src: 015011ee11 -->
## 6. 故障排除

| 症状 | 看到的内容 | 可能原因 | 解决方法 |
|---|---|---|---|
| 下载失败 | `wget: unable to resolve host` 或 `404` | 没有网络连接,或归档文件已移动 | 检查网络连接并重试;模型请使用 4.3.2 的选项 B |
| `dxcom` 找不到校准图像 | `No images found` 或校准错误 | JSON 中的 `dataset_path` 不正确 | 重新运行 4.2 的单元格;它会写入 `dx_com/calibration_dataset` 的绝对路径 |
| 编译耗时很长 | 超过 10 分钟且没有错误 | 640x640 的 YOLOv7 受 CPU 性能限制 | 等待,或使用选项 B(预编译的 DXNN) |
| `No DXNN at .../workspace/output/...` | 4.3.2 之后出现 `FileNotFoundError` | 两个选项都没有生成该文件 | 将 `COMPILE_OPTION` 设为 `"A"` 或 `"B"`,然后再次运行对应的单元格 |
| 5.3 或 5.4 没有出现窗口 | 单元格仅以性能摘要结束 | 单元格默认以无头模式运行 | 打开 `workspace/outputs/` 下保存的结果,或在桌面会话中去掉 `--no-display` / 设置 `SHOW_WINDOW = True` |
| 边界框标签错误 | 叉车被标为 Worker,或者相反 | `class_names` 的顺序与训练顺序不一致 | 修正运行时 JSON 中的顺序并重新运行;`num_classes` 必须保持为 2 |

<!-- cell: 2df122e0-2ea3-4c2e-b294-6fee2d7c5b24 src: 4848d5487c -->
## 7. 总结

### 7.1 已完成的工作流

```text
Choose a model (YOLOv7) and a labeled dataset
       │
       ▼
Train on a GPU (Colab notebook)
       │
       ▼
Export ONNX and compile to DXNN with dxcom
       │
       ▼
Describe classes and thresholds in a DX-APP runtime config
       │
       ▼
Run the custom model on the NPU with yolov7_async
```

### 7.2 完成清单

- [ ] 为编译器准备 ONNX 模型和 Model Zoo JSON 配置
- [ ] 将 `dataset_path` 指向本地校准数据集
- [ ] 使用 `dxcom` 编译模型或下载预编译的 DXNN
- [ ] 编写包含 `num_classes` 和 `class_names` 的 DX-APP 运行时配置
- [ ] 在图像和视频上运行自定义模型

> **下一步:** 继续学习教程 04,使用 DX-STREAM 将同一个 Forklift 和 Worker 检测器构建到 GStreamer 流水线中。
