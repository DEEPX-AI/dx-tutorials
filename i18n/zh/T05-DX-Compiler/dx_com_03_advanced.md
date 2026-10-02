<!-- i18n source: notebooks/T05-DX-Compiler/dx_com_03_advanced.ipynb -->
<!-- i18n lang: zh -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: advanced-title src: ca77fa067f -->
# DEEPX Tutorial 05-3 - DX-COM 高级

本教程涵盖精度恢复、量化诊断、可复用的 QXNN 检查点、量化感知训练(QAT),以及使用 `dx_com` Python API 进行编程式编译。

前几个实验使用 SqueezeNet,以便您专注于工作流本身。最后一个实验使用一个紧凑的双输入立体融合模型,演示一个必须使用 Python API 的场景。

> **开始之前:** 请先完成入门和中级教程。编译可能需要几分钟。每个编译单元格都会检查是否已存在 DXNN,并跳过重复的工作。

<!-- cell: ac85dc95-b527-4b0a-8cad-e82a0d512e16 src: 215e801d3c -->
> 本教程共 3 部分,这是第 3 部分。课程地图和完整的部分列表见入门笔记本。

<!-- cell: learning-objectives src: c53dcc566f -->
## 学习目标

完成本教程后,您将能够:

- 比较 Q-Lite PTQ、Q-PRO 增强 PTQ 和 Q-Master QAT,并选择合适的工作流,
- 基于同一模型、数据和预处理构建受控的 Q-Lite 实验与自动 Q-PRO 实验,
- 使用量化诊断报告确定一次只改一个变量的精度恢复实验,
- 复用 QXNN 检查点进行 IQR 重新校准和自动 Q-PRO,而无需重复完整的 ONNX 编译,
- 准备 Q-Master/QAT 配置,并区分流水线冒烟测试与有意义的精度训练,
- 根据模型和工作流需求在 `dxcom` CLI 与 `dx_com.compile()` Python API 之间做出选择,
- 实现以名称为键的 PyTorch `DataLoader`,并用 Python API 编译双输入模型,
- 仅在需要回答特定的图或性能问题时才使用高级编译器控制选项,以及
- 收集模型、校准、编译器、精度、性能和可复现性方面的证据,供发布评审使用。

本笔记本不会修改 SDK 源码树。所有生成的文件都保存在 `<dx-tutorials>/notebooks/T05-DX-Compiler/workspace` 下,该目录被 git 忽略。

<!-- cell: 2120bfff-8ac4-408b-ac67-6dfb87c7f8ab src: 709b2f76c9 -->
## 前提条件

- 已完成教程 05-1 和 05-2。
- DX-COM、校准数据集,以及 DX-RT 提供的 `dxparse`。
- 下载量: 来自 Model Zoo 的 SqueezeNet ONNX 和 JSON(约 5 MB)。第 8 节的双输入模型在本地生成。
- 耗时: 本笔记本以不同的量化设置编译 SqueezeNet 五次,外加一次小型的 Python API 编译。Q-Lite 基线、诊断和 Python API 编译只需几秒;两次自动 Q-PRO 编译(第 4 节和 6.2 节)各需约 10 分钟或更久。

下一个单元格会定位 SDK、检查这些要求并打印状态表。任何标记为 `MISSING` 的项目都会附带提供它的步骤。

<!-- cell: workspace-heading src: be980c7ccd -->
## 1. 初始化教程工作区

设置单元格会从 `config.json` 解析所有 SDK 路径,创建一个独立的工作区,并通过符号链接复用 SDK 的校准图像。

生成的教程文件保存在:

```text
<dx-tutorials>/notebooks/T05-DX-Compiler/workspace/      (ignored by git)
├── models/                 downloaded and generated ONNX files
├── configs/                compiler JSON files
├── outputs/                one directory per compile
└── calibration_dataset -> <DX_COM_DIR>/calibration_dataset
```

<!-- cell: notebook-dependencies-heading src: 8bf0df0e3a -->
Jupyter 内核与 DX-COM 编译器使用各自独立的 Python 环境。这是有意为之:

```text
Jupyter code       -> <dx-tutorials>/.venv/bin/python
DX-COM Python API  -> <DX_COMPILER_DIR>/venv-dx-compiler-local/bin/python
```

请用 `uv` 把仅供笔记本使用的检查类软件包安装到 Jupyter 环境中。在这个由 uv 管理的环境中不要使用 `%pip`。

<!-- cell: prepare-model-heading src: 2002bc78a7 -->
## 2. 准备一个紧凑的参考模型

我们使用 Model Zoo 的 SqueezeNet ONNX 及其配套的 Q-Lite JSON。下载得到的预处理保持不变;只替换本地的校准数据集路径。

下一个单元格从 `tutorial_paths.py` 导入 `download_file`(教程 05-2 也使用的共享辅助函数)。它遵循两条规则:

1. 如果目标文件已存在且非空,则跳过下载,不发出网络请求。
2. 对于新文件,先下载到 `.part`,只有在大小检查通过后才替换目标文件。

如果您确实需要一份新的副本,请先删除本地文件。

<!-- cell: quantization-strategy src: 9e98660596 -->
## 3. 选择量化策略

这三条路径都会生成 INT8 DXNN,但它们使用不同的方法,所需的数据量和计算量也不同。

<img src="assets/q-lite-q-pro-q-master.png" alt="Q-Lite、Q-PRO 与 Q-Master 的比较" style="max-width: 600px;">

| 项目 | Q-Lite | Q-PRO | Q-Master |
|---|---|---|---|
| 量化方法 | 标准 PTQ | 增强 PTQ | QAT(量化感知训练) |
| 主要机制 | 校准浮点范围 | 应用自动或手动的 DXQ 增强阶段 | 微调量化后的学生模型 |
| 所需数据 | 有代表性的校准样本 | 同样有代表性的校准样本 | 训练和验证样本;使用任务损失时需要标签 |
| 训练 | 否 | 否 | 是 |
| 典型计算成本 | 最低 | 高于 Q-Lite | 最高;通常使用 CUDA GPU |
| DX-COM 入口 | 普通编译 | `--use_q_pro` 或手动 DXQ 方案 | 在配置中添加 `qmaster` 块 |
| 推荐用途 | 受控基线 | 第一个 PTQ 精度恢复实验 | 当 PTQ 仍未达到精度目标时使用 |

在本教程中,**Q-Master** 指由 JSON `qmaster` 块启用的 DX-COM QAT 工作流。不要仅凭名称选择方法:

<img src="assets/quantization-strategy-decision-flow.png" alt="从 Q-Lite 到 Q-PRO 再到 Q-Master 的决策流程" style="max-width: 800px; width: 100%;">

<!-- cell: baseline-heading src: 0c8f1bdab9 -->
### 3.1 建立 Q-Lite 基线

公平的基线应与后续每个实验使用相同的 ONNX、校准数据、预处理、编译器版本和验证协议。

下一个单元格等价于在单独的终端中输入以下命令:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/squeezenet-1.0_224x224.onnx \
      -c configs/squeezenet.local.json \
      -o outputs/squeezenet_q_lite_baseline \
      --gen_log \
      --export_html
```

笔记本会自动解析真实路径。如果输出目录中已经包含 DXNN,则跳过编译。

> **关于代码单元格末尾的 `2>&1 | sed ... | grep ...`:** `dxcom` 使用光标移动转义码绘制进度条,而 Jupyter 无法渲染这些转义码,否则会显示为数百行空行。该过滤器只丢弃进度条行;所有 `[INFO]`、`[WARNING]` 和 `[ERROR]` 消息都会保留。在终端中可以省略该过滤器。`--gen_log` 会把完整输出保存在 `compiler.log` 中。本笔记本的每个编译单元格都使用同样的过滤器。

<!-- cell: baseline-artifacts-heading src: 2bc6375342 -->
HTML 摘要和 `compiler.log` 是实验证据的一部分。它们不能替代任务精度验证,但能让构建过程可评审、可复现。

<!-- cell: qpro-heading src: c75215ec57 -->
## 4. 运行自动 Q-PRO

Q-PRO 会尝试从 DXQ 系列中选出的量化增强阶段。`--use_q_pro` 是最简单的起点,因为 DX-COM 会自动选择组合。

重要规则:

- 保持 Q-Lite 与 Q-PRO 的输入完全一致。
- `--use_q_pro` 与手动选择的 `enhanced_scheme` 互斥。
- 只有当实测任务精度有所提升时,更高的编译成本才是合理的。

等价的终端命令:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/squeezenet-1.0_224x224.onnx \
      -c configs/squeezenet.local.json \
      -o outputs/squeezenet_q_pro \
      --use_q_pro \
      --opt_level 1 \
      --gen_log \
      --export_html
```

<!-- cell: compare-qpro src: a0cd8c050e -->
### 4.1 正确比较实验

对两个模型使用同一个保留验证数据集。记录的指标不要只有一个:

| 证据 | 回答的问题 |
|---|---|
| 任务指标 | Q-PRO 是否恢复了有用的精度? |
| 编译时间 | 增加了多少开发成本? |
| DXNN 大小 | 部署产物是否发生了实质性变化? |
| `dxrun` 吞吐量 | 独立的运行时吞吐量是否变化? |
| 端到端延迟和 CPU | 完整应用是否有所改善? |

当 Q-Lite 已经满足产品目标时,Q-PRO 并不自动更优。

<!-- cell: diagnosis-heading src: 7923517f6c -->
## 5. 诊断量化损失

在能够复现 ONNX 与 DXNN 之间的精度差距之后,再使用诊断。`--quant_diagnosis` 会生成:

- `quant_diagnosis/diagnosis_report.html`: 按区域划分的质量证据和重试指导,
- `quant_diagnosis/<model>.qxnn`: 可复用于重新量化的检查点。

等价的终端命令:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/squeezenet-1.0_224x224.onnx \
      -c configs/squeezenet.local.json \
      -o outputs/squeezenet_diagnosis \
      --quant_diagnosis \
      --gen_log \
      --export_html
```

只有当 DXNN、诊断 HTML 和 QXNN 检查点都存在时才会跳过编译。

<!-- cell: read-diagnosis src: bcd73e08c0 -->
### 5.1 把报告当作实验计划来读

1. 找到标记为 **Warning** 或 **Critical** 的区域。
2. 检查浮点行为与量化行为最先出现分歧的位置。
3. 阅读证据和推荐的重新编译意图。
4. 选择一项更改,例如校准方法或 Q-PRO。
5. 从 QXNN 检查点恢复。
6. 再次测量同一个保留任务指标。

一次只改一个变量。如果数据集、预处理、观察器和编译器选项同时改变,您就无法判断是哪项更改起了作用。

<!-- cell: resume-heading src: c604a33ce2 -->
## 6. 使用 QXNN 恢复重新量化

QXNN 恢复会跳过前面的 ONNX 编译工作,只重新运行依赖量化的阶段:

```text
Normal compile: ONNX -> optimize/partition -> quantize -> codegen -> DXNN
                                |
                                +-> diagnosis QXNN checkpoint
                                             |
Resume:                             new quantization setting -> DXNN
```

规则:

- `--checkpoint` 与 `--model_path` 互斥。
- 原始配置已嵌入检查点中,因此不需要 `-c`。
- 仅当需要有意覆盖嵌入的位置时才使用 `--dataset_path`。

<!-- cell: resume-iqr-heading src: 8455073c65 -->
### 6.1 使用 IQR 重新校准恢复

等价的终端命令:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom --checkpoint outputs/squeezenet_diagnosis/quant_diagnosis/<model>.qxnn \
      -o outputs/squeezenet_resume_iqr \
      --recalibration_method iqr \
      --dataset_path calibration_dataset
```

<!-- cell: resume-qpro-heading src: 3df09ba500 -->
### 6.2 使用自动 Q-PRO 恢复

本实验复用同一个检查点,但应用自动 Q-PRO 而不是 IQR。

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom --checkpoint outputs/squeezenet_diagnosis/quant_diagnosis/<model>.qxnn \
      -o outputs/squeezenet_resume_q_pro \
      --use_q_pro \
      --dataset_path calibration_dataset
```

<!-- cell: qat-heading src: d9acd3286b -->
## 7. 准备 Q-Master(QAT)实验

当 PTQ 方法无法达到精度目标时,Q-Master 使用量化感知训练。在普通的 JSON 配置中添加 `qmaster` 块即可;不需要单独的 CLI 标志。

```text
Calibration -> QAT training -> best QXNN checkpoint -> final DXNN
```

一次有意义的 QAT 运行需要有代表性的训练和验证数据,并且通常需要 CUDA GPU。教程的校准图像不是训练数据集,因此本节只创建一个经过审阅的模板,而不是启动一次昂贵且会产生误导的训练。

要点:

- `default_loader` 是校准和训练共用的预处理来源。
- `qmaster` 包含训练超参数。
- `fast_run: true` 用一个 epoch 和一个 batch 验证流水线;它**不**验证精度。
- 保留数据集版本、配置、训练日志、最佳检查点、编译器版本和验证结果。

<!-- cell: run-qat-guidance src: 6a412f8ffc -->
替换数据集路径并审阅超参数之后,在单独的终端中运行 QAT:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/squeezenet-1.0_224x224.onnx \
      -c configs/squeezenet.qat.template.json \
      -o outputs/squeezenet_qat \
      --gen_log \
      --export_html
```

重要输出:

| 产物 | 用途 |
|---|---|
| `qat_checkpoint/qat_checkpoint.qxnn` | 最佳训练检查点;可复用于仅编译的恢复 |
| `*.dxnn` | 最终部署产物 |
| `compiler.log` 和训练日志 | 用于调试和复现的证据 |

如果只做流水线冒烟测试,请在 `qmaster` 内添加 `"fast_run": true`。`fast_run` 是一个 JSON 键,不是 `dxcom` 选项。

<!-- cell: python-api-overview src: 666d46f4f0 -->
## 8. 使用 Python API 进行编程式编译

CLI 和 Python API 使用同一个 DX-COM 编译器引擎。Python API 本身不会让标准模型更快或更精确;它改变的是应用程序提供模型、校准数据和编译器选项的方式。

### 8.1 何时使用哪种接口?

| 需求 | `dxcom` CLI | `dx_com.compile()` Python API |
|---|---:|---:|
| 标准单图像输入并使用 JSON 预处理 | 推荐 | 支持 |
| 可复现的 shell 命令或 CI 步骤 | 推荐 | 支持 |
| 内存中的 `onnx.ModelProto` | 否 | 是 |
| 自定义 PyTorch 预处理/数据流水线 | 否 | 是 |
| 非图像校准数据 | 否 | 是 |
| 多输入模型 | 否 | **必需** |
| 编程式实验循环和异常处理 | 有限 | 推荐 |
| 从应用程序直接控制 QAT/检查点 | 有限 | 推荐 |

Python API 的优势:

- 可传入 ONNX 路径或内存中的 `onnx.ModelProto`,
- 通过 PyTorch `DataLoader` 提供精确的校准张量,
- 支持多输入模型和非图像模型,
- 无需解析 shell 输出即可构建可重复的实验循环,
- 处理 Python 异常并记录结构化元数据,以及
- 把编译集成到模型导出、验证和产物管理代码中。

两条约束可以避免常见错误:

1. `config=` 和 `dataloader=` 只能提供其中一个;两者互斥。
2. 使用 `dataloader=` 时,预处理必须在 `Dataset.__getitem__()` 内完成,并且必须与部署时的预处理一致。

<!-- cell: inspect-python-api-heading src: c31ad53df9 -->
### 8.2 检查已安装的 API

笔记本内核不会直接导入 `dx_com`,因为 DX-COM 安装在它自己的编译器环境中。下一个单元格精确地运行以下命令,并合并在一个 `!` 行中:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
python -c 'import inspect, dx_com; print(inspect.signature(dx_com.compile))'
```

<!-- cell: python-api-parameters src: ab00c54c22 -->
最重要的参数如下:

| 参数 | 含义 |
|---|---|
| `model` | ONNX 文件路径或 `onnx.ModelProto` |
| `output_dir` | 接收 DXNN 和报告的目录 |
| `config` | 面向标准单输入图像模型的 JSON 驱动校准 |
| `dataloader` | 编程式校准张量;本多输入实验必需 |
| `calibration_method` | 普通编译使用 `ema` 或 `minmax` |
| `calibration_num` | 从 DataLoader 中消费的样本数 |
| `opt_level` | `0` 用于更快的实验,`1` 用于完整优化 |
| `use_q_pro` | 启用自动 Q-PRO;不能与手动 `enhanced_scheme` 同时使用 |
| `quant_diagnosis` | 生成诊断 HTML 和可恢复的 QXNN |
| `gen_log`、`export_html` | 保留评审和调试证据 |

标准的单输入调用如下所示:

```python
import dx_com

dx_com.compile(
    model="models/model.onnx",
    config="configs/model.json",
    output_dir="outputs/model",
    gen_log=True,
    export_html=True,
)
```

对于自定义 DataLoader,请把 `config=` 替换为 `dataloader=`;切勿同时传入两者。

<!-- cell: multi-input-model-heading src: c96a18bc8d -->
### 8.3 一个必须使用 Python API 的模型

立体深度、光流和特征融合网络通常需要消费两个同步的张量。本实验选择一个紧凑的立体特征融合模型,因为它无需漫长的模型下载或编译即可演示多输入需求。`dxcom` CLI 的 JSON 加载器只支持一个模型输入,因此无法提供两个张量。而 Python API 可以返回一个以精确的 ONNX 输入名称为键的字典:

```text
left image  -- preprocessing -- tensor "left_image"  --+
                                                        +--> stereo-fusion ONNX --> DXNN
right image -- preprocessing -- tensor "right_image" --+

Dataset.__getitem__()
    -> {"left_image": Tensor[C,H,W], "right_image": Tensor[C,H,W]}
DataLoader(batch_size=1)
    -> {"left_image": Tensor[1,C,H,W], "right_image": Tensor[1,C,H,W]}
```

这个代表性的图在特征融合之前分别对两个输入进行编码。这样可以避免把两张图像当作一个输入,并保留真正的双输入 DXNN 契约:

```text
left_image  -> Conv -> left_features  --+
                                          Add -> ReLU -> fused_features
right_image -> Conv -> right_features --+
```

本实验构建一个紧凑的代表性立体融合图。它被有意设计得很小,以便练习聚焦在 API 契约上。其合成校准数据只用于验证编译机制;生产环境的立体模型需要同步且有代表性的左/右样本。

<!-- cell: multi-input-dataloader-heading src: 80154869ac -->
### 8.4 实现按名称命名的校准 DataLoader

每个数据集项都不包含 batch 维度。`DataLoader(batch_size=1)` 会自动添加它。

字典键比元组更安全:

- 字典: 输入按 ONNX 名称匹配,
- 元组/列表: 输入按模型内部的输入顺序匹配。

脚本会在调用编译器之前验证第一个 batch,然后在 DXNN 已存在时跳过编译。

<!-- cell: run-multi-input-heading src: d764daa244 -->
### 8.5 使用 DX-COM Python 环境编译

下一个笔记本单元格使用编译器环境的 Python 解释器运行该脚本。它等价于:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
python compile_stereo_fusion.py
```

这个模型有意没有提供等价的 `dxcom` CLI 命令:编译器需要由自定义 DataLoader 提供的两个张量。

<!-- cell: inspect-multi-input-heading src: f6368021a6 -->
### 8.6 验证结果

`dxparse -v` 应显示两个模型输入:`left_image` 和 `right_image`。这确认了编译产物保留了多输入接口。

校准 DataLoader 符合 ONNX 契约(`float32`,NCHW)。编译后的 NPU 接口可能报告为 `uint8`、NHWC,因为 DX-COM 会把受支持的输入转换移到 NPU 路径中。部署时,请遵循为生成的 DXNN 报告的输入形状、数据类型和预处理指导,而不要假设 ONNX 布局保持不变。

<!-- cell: advanced-controls-heading src: 7bf0d30ccb -->
## 9. 审慎使用高级编译器控制选项

这些选项应当用于回答特定的图、性能或可复现性问题。不要把它们当作一个笼统的“更多优化”套餐来启用。

| CLI 选项 | Python API 参数 | 用途 | 主要注意事项 |
|---|---|---|---|
| `--opt_level {0,1}` | `opt_level` | 以编译时间换取完整优化 | 与受控基线进行比较 |
| `--aggressive_partitioning` | `aggressive_partitioning` | 探索更多的 NPU 分区方式 | 实验性功能;请验证输出和端到端延迟 |
| `--compile_input_nodes` | `input_nodes` | 从选定的 ONNX 算子节点开始 | 会改变编译后的接口 |
| `--compile_output_nodes` | `output_nodes` | 在选定的 ONNX 算子节点结束 | 剩余工作转移到主机 |
| `--float64_calibration` | `float64_calibration` | 提高跨 CPU 的校准确定性 | 更高的计算和内存开销 |
| `--gen_log` | `gen_log` | 保留详细的编译器日志 | 与发布产物一起保存 |
| `--export_html` | `export_html` | 生成自包含的摘要 | 不能替代任务验证 |

进行子图编译时,请使用 **ONNX 算子节点名称**,而不是张量名称。请记录由此产生的输入/输出契约以及仍保留在主机端的操作。

<!-- cell: release-validation-heading src: 82c9e17d3c -->
## 10. 建立发布验证记录

编译器成功退出是必要条件,但不足以作为发布的充分证据。

| 领域 | 最低限度的证据 |
|---|---|
| 源模型 | 导出命令、框架/版本、ONNX 检查器结果 |
| 校准 | 数据集版本、样本数量、预处理、方法 |
| 编译器 | DX-COM 版本、完整的命令/API 参数、日志、HTML 报告 |
| 精度 | ONNX 和每个 DXNN 候选使用同一任务指标 |
| 性能 | 预热策略、持续时间、设备、吞吐量、延迟、CPU 负载 |
| 集成 | 输入/输出形状、预处理归属、解码器契约 |
| 可复现性 | 产物哈希和相互隔离的输出目录 |

使用虚拟输入的 `dxrun` 适合检查吞吐量,但不能替代任务精度或完整应用的延迟测量。

<!-- cell: 20d9ebdb-92ef-4b3e-9c73-c4bf02326bd8 src: 175675be1b -->
## 11. 故障排除

| 症状 | 看到的内容 | 可能原因 | 解决方法 |
|---|---|---|---|
| `--use_q_pro` 与 `--enhanced_scheme` 冲突 | 关于选项冲突的编译器错误 | 同时请求了自动和手动 Q-PRO 选择 | 每次编译只使用两者之一 |
| 找不到 `--checkpoint` 文件 | 针对 `.qxnn` 文件的 `FileNotFoundError` | 第 5 节中写入 `quant_diagnosis/*.qxnn` 的诊断编译被跳过或失败 | 先运行第 5 节,并确认它打印了检查点路径 |
| 第 7 节没有运行 QAT | 只写入了 `configs/squeezenet.qat.template.json` | 有意为之:示例图像无法训练出有用的模型 | 把数据集路径替换为有代表性的训练数据,并在终端中运行第 7 节的命令 |
| JSON 更改不起作用 | `Skip compilation: found ...` | 输出目录中已经包含 DXNN | 删除 `workspace/outputs/<dir>` 后重新运行 |
| 缺少 `dxparse` | `[MISSING] dx_rt` | 未安装 DX-RT | 安装 DX-Runtime(教程 01 第 3 节) |

<!-- cell: summary src: 0316911a19 -->
## 12. 总结

### 12.1 高级精度恢复路线图

```text
Build a controlled Q-Lite baseline
                │
                ▼
       Accuracy target met?
          ┌─────┴─────┐
         Yes          No
          │            │
          │            ▼
          │      Try automatic Q-PRO
          │            │
          │            ▼
          │   Accuracy target met?
          │      ┌─────┴─────┐
          │     Yes          No
          │      │            │
          │      │            ▼
          │      │    Diagnose quantization loss
          │      │            │
          │      │            ▼
          │      │    Resume from QXNN and test
          │      │    one change at a time
          │      │            │
          │      │            ▼
          │      │    Prepare Q-Master QAT
          └──────┴────────────┘
                │
                ▼
 Validate accuracy, performance,
 integration, and reproducibility
```

### 12.2 高级工作流一览

| 工作流 | 适用时机 | 主要输入 | 主要结果 |
|---|---|---|---|
| Q-Lite 基线 | 第一个受控 PTQ 实验 | ONNX、JSON、有代表性的校准数据 | 基线 DXNN 和证据 |
| 自动 Q-PRO | Q-Lite 未达到精度目标 | 相同的模型、数据、预处理和验证协议 | 增强 PTQ 候选 |
| 诊断与 QXNN 恢复 | ONNX 到 DXNN 的精度差距可复现 | 诊断报告和 QXNN 检查点 | 更快的一次只改一个变量的试验 |
| Q-Master QAT | PTQ 方法仍未达到目标 | 有代表性的训练和验证数据 | 经过训练的量化候选 |
| Python API | 校准需要自定义张量、非图像数据或多个输入 | ONNX 加上以名称为键的 PyTorch DataLoader | 以编程方式编译的 DXNN |

### 12.3 已完成的实验

| 实验 | 受控比较或契约 | 产生的证据 |
|---|---|---|
| SqueezeNet Q-Lite 与 Q-PRO 对比 | 相同的 ONNX、校准、预处理和编译器设置 | 独立的 DXNN 文件、日志和 HTML 报告 |
| 量化诊断 | 在更改设置之前复现精度损失候选 | 诊断 HTML 和可复用的 QXNN 检查点 |
| QXNN 恢复 | 基于同一检查点的 IQR 重新校准与自动 Q-PRO 对比 | 更快的量化实验 |
| Q-Master 准备 | 冒烟测试配置与有意义的 QAT 训练对比 | 经过审阅的 QAT 模板和验证要求 |
| 立体融合 Python API | 由一个 DataLoader 提供的两个命名输入 | 经 `dxparse` 验证的双输入 DXNN |

### 12.4 完成清单

- [ ] 构建可复用的 Q-Lite 基线
- [ ] 把自动 Q-PRO 作为受控 PTQ 实验运行
- [ ] 生成并解读量化诊断报告
- [ ] 复用 QXNN 检查点进行 IQR 和 Q-PRO 试验
- [ ] 区分 Q-Master 冒烟测试与有意义的 QAT
- [ ] 根据工作流需求选择 CLI 或 Python API
- [ ] 实现以名称为键的多输入校准 DataLoader
- [ ] 编译并检查双输入 DXNN
- [ ] 确定发布评审所需的证据
- [ ] 用有代表性的产品数据替换教程样本
- [ ] 使用同一个保留任务指标比较 ONNX 和每个 DXNN 候选
- [ ] 测量完整应用的延迟、吞吐量和主机 CPU 使用率

### 12.5 如何选择

| 如果您需要... | 从这里开始... |
|---|---|
| 建立可复现的 INT8 参考 | Q-Lite |
| 在不训练的情况下恢复 PTQ 精度 | 自动 Q-PRO |
| 找出量化损失从何处开始 | 量化诊断 |
| 在不重复完整 ONNX 编译的情况下测试另一种量化设置 | QXNN 恢复 |
| 在 PTQ 选项用尽后恢复精度 | Q-Master QAT |
| 提供自定义、非图像、内存中或多个命名输入 | `dx_com.compile()` Python API |
| 为标准模型构建可复现的 shell 或 CI 编译步骤 | `dxcom` CLI |

> **请记住:** 编译器成功消息、生成的 DXNN 和快速的虚拟输入基准测试都只是部分证据。发布决策需要有代表性的数据、ONNX 与 DXNN 使用同一个保留任务指标、完整流水线的性能测量,以及可复现的产物。

### 12.6 将此工作流应用到您的模型

1. 固定源 ONNX、预处理、数据集版本和验证指标。
2. 在更改量化设置之前先建立 Q-Lite 基线。
3. 一次只改一个变量,并把每个结果保存在相互隔离的输出目录中。
4. 仅当实测精度确有需要时,才从 Q-Lite 逐级升级到 Q-PRO、诊断/QXNN 恢复,最后到 Q-Master。
5. 把命令或 API 参数、编译器日志、报告、哈希、精度和性能结果与发布候选一起保存。

> **下一步:** 继续学习教程 06,使用 DX-Runtime 运行编译后的模型:CLI 工具、Python 和 C++ API、性能分析以及发布验证。
