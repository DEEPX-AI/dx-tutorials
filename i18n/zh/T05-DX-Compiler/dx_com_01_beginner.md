<!-- i18n source: notebooks/T05-DX-Compiler/dx_com_01_beginner.ipynb -->
<!-- i18n lang: zh -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 263bea17 src: 05ca91a778 -->
# DEEPX Tutorial 05-1 - DX-COM 入门

本笔记本通过小型、可复现的示例介绍从 ONNX 到 DXNN 的完整工作流程。

## 学习目标

完成本教程后,您将能够:

1. 理解 DX-COM 的整体流程,
2. 验证 DX-COM 安装,
3. 导出并验证 MobileNetV2 ONNX 模型,
4. 创建校准配置,
5. 编译并检查 DXNN 模型,以及
6. 从 DEEPX Model Zoo 下载 ONNX 模型及其 JSON 配置并进行编译。

本笔记本不会修改 SDK 源码树。所有生成的文件都保存在 `<dx-tutorials>/notebooks/T05-DX-Compiler/workspace` 下,该目录被 git 忽略。

<!-- cell: 09404f54 src: 15d2514cfa -->
## 课程地图

| 笔记本 | 主要内容 |
|---|---|
| 入门 | 安装、ONNX 验证、JSON 基础、首次编译、Model Zoo |
| 中级 | 校准质量、硬件 PPU、YOLO26 TopK 优化 |
| 高级 | Q-PRO、诊断、QXNN 恢复、QAT、Python API、高级编译器控制 |

除非您已经熟悉 DX-COM 的配置与校准,否则请按顺序完成这些笔记本。

<!-- cell: dx-com-01-prerequisites src: 70ef62da25 -->
## 前提条件

- 已完成教程 01: DX-COM 已安装在 `dx-compiler/venv-dx-compiler-local` 中,并带有示例校准数据集;第 5 节和第 6 节的 `dxparse` 与 `dxrun` 单元格需要 DX-RT 和一块 DEEPX NPU。
- 下载量: 第 3 节导出所需的仅 CPU 版 PyTorch wheel 和 ONNX 工具(首次安装约几百 MB),以及来自 Model Zoo 的 ResNet50 ONNX(97 MB)。
- 无需 `sudo`。预计约 20 分钟;两次编译各不到一分钟。
- 本教程创建的所有内容都放在 `notebooks/T05-DX-Compiler/workspace/` 下,该目录被 git 忽略。

第 2 节的设置单元格会定位 SDK、检查这些要求并打印状态表。任何标记为 `MISSING` 的项目都会附上提供它的步骤。

<!-- cell: dxcom-workflow-overview src: 44853a9f8f -->
## 1. 编译工作流程

整体编译过程包括以下三个步骤:

1. **获取预训练模型** — 训练一个模型,或从 PyTorch 等框架获取兼容的预训练模型。
2. **将模型转换为 ONNX** — 将框架模型导出为 ONNX,并验证其输入名称、输入形状、算子和输出。
3. **将 ONNX 编译为 DXNN** — 使用 DX-COM,配合 JSON 配置和有代表性的校准数据,生成面向 DEEPX NPU 的模型。

<img src="assets/dx-com-workflow.jpg" style="max-width: 800px; width: 100%;" alt="从预训练模型到 ONNX 再到 DXNN 的 DX-COM 编译工作流程">

本入门教程以 MobileNetV2 为例遵循同一工作流程:PyTorch → ONNX → DXNN。

更多细节请参阅 DX-Compiler 用户指南 [下载](https://developer.deepx.ai/download/?id=581)

> **注意:** 下载用户指南前,必须先登录 https://developer.deepx.ai/。

<!-- cell: dxcom-artifact-glossary src: 6e807062e7 -->
### 1.1 编译过程中使用和生成的文件

请在脑中区分这些产物。每一项在工作流程中的角色都不同。

| 产物 | 角色 | 使用方或生成方 |
|---|---|---|
| 框架模型,例如 `.pt` | 来自原始框架的训练权重和网络定义 | 导出 ONNX 时使用 |
| `.onnx` | DX-COM 读取的与框架无关的模型图 | DX-COM 的输入 |
| 编译器 `.json` | 输入形状、校准、预处理、量化以及可选的编译器设置 | DX-COM 的输入 |
| 校准数据集 | 用于估计量化范围的代表性样本 | DX-COM 在校准期间读取 |
| `.dxnn` | 由 DEEPX 运行时执行的编译后模型 | DX-COM 的主要输出 |
| `compiler.log` | 详细的编译消息、警告和错误 | 由 `--gen_log` 生成 |
| `*_summary.html` | 包含图和编译器信息的可视化编译报告 | 由 `--export_html` 生成 |

`.onnx`、编译器 `.json` 和校准数据集是编译输入。`.dxnn`、日志和 HTML 报告是输出。

<img src="assets/dx-compile-progress.png" style="max-width: 1000px; width: 100%;" alt="DX-COM 编译及其所需文件">

<!-- cell: f668eb12 src: 50dcc961e3 -->
## 2. 要求与工作区

DX-COM 支持 **x86-64 Linux**。请使用 **batch size 为 1** 的静态 **ONNX 输入**形状。实用的主机至少需要 **16 GB 内存**和 8 GB 可用存储空间。

本教程需要 SDK 安装中的两项内容:

| 要求 | 提供方 | 如果报告为 `MISSING` |
|---|---|---|
| `dx_com`(`venv-dx-compiler-local` 中的 `dxcom` 编译器) | `./dx-compiler/install.sh` | 按照教程 01 第 2 节或下面的 2.1 节操作 |
| `calibration_dataset`(`dx_com/` 下的示例图片) | `dx-compiler/example/2-download_sample_calibration_dataset.sh` | 在终端中运行该脚本一次 |

下一个单元格通过 `tutorial_paths.py` 定位 DX-All Suite(依次为环境变量、`config.json`、自动检测),打印找到的内容,并准备本教程的工作区。缺少某项要求时它不会停止;会停止的检查位于 2.2 节,在介绍完安装选项之后。

<!-- cell: 16e77545 src: 141db488f9 -->
### 2.1 DX-COM 安装选项

DX-COM 是一个 Python 包,**强烈建议将其安装在专用的 Python 虚拟环境中**。虚拟环境将 DX-COM 及其依赖与系统 Python 和 Jupyter 内核隔离,从而减少版本冲突,并使编译器环境更易于复现或移除。

最简单的独立安装方式是使用发布在 [PyPI](https://pypi.org/project/dx-com/) 上的包。请在单独的终端中运行以下命令:

```bash
# 1. Create a dedicated environment.
python3 -m venv ~/venv-dx-com

# 2. Activate it in the current terminal.
source ~/venv-dx-com/bin/activate

# 3. Install DX-COM from PyPI.
python -m pip install --upgrade pip
pip install dx-com

# 4. Confirm that the virtual environment owns the commands.
which python
which dxcom
dxcom --version
```

两条 `which` 命令应打印 `~/venv-dx-com/` 下的路径。每次打开新终端时请再次运行 `source ~/venv-dx-com/bin/activate`,结束后运行 `deactivate`。

本笔记本使用由 `dx-all-suite/dx-compiler/install.sh` 在 `venv-dx-compiler-local` 创建的专用环境,因此其编译器不依赖 Jupyter 内核的 Python 环境。上述 PyPI 流程是单独安装 DX-COM 时推荐的替代方案。

> 安装 `dx-com` 会提供编译器包和 `dxcom` 命令。完整的编译与部署工作流程仍然需要 DEEPX SDK 的其余部分以及受支持的 Linux x86-64 主机。

<!-- cell: c7a528a2 src: dd9c03046a -->
### 2.2 验证 DX-COM 安装

下一个单元格首先确认第 2 节的两项要求都已满足。如果缺少其中一项,它会停止并给出需要运行的确切步骤;完成该步骤后,再次运行该单元格。

单元格的其余部分等价于以下终端命令:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
dxcom --version
dxcom -h
```

`<DX_ALL_SUITE_DIR>` 是上面设置单元格打印的位置。

<!-- cell: f8556d44 src: 6e8fd67cf4 -->
## 3. 导出一个小型 ONNX 模型

本项目使用由 `uv` 管理的 Jupyter 环境,其中可能不包含 `pip` 模块。

`uv pip install --python "{sys.executable}"` 会显式地将包安装到当前 Jupyter 内核中。 

这些包用于导出和检查 ONNX。DX-COM 本身继续从 `DX_COMPILER_VENV` 运行。

较新版本的 torch(2.9 及以后)的 ONNX 导出器需要 `onnxscript`。`--extra-index-url` 让 `uv` 指向仅 CPU 版的 PyTorch wheel:导出不需要 GPU,而 CPU wheel 只有几百 MB,而不是数 GB。已安装的 torch 会保持原样。

<!-- cell: 206f9840 src: a88480414f -->
### 3.1 验证 ONNX 契约

JSON 文件中的输入名称必须与 ONNX 图的输入名称完全一致。形状推断和 `onnx.checker` 可以在编译前捕获许多导出错误。

<!-- cell: 96ab8536 src: 372ae5a1e1 -->
您也可以在 [Netron](https://netron.app/) 中打开 ONNX 文件,检查张量名称、形状和算子。对于编译后的 `.dxnn`,请使用 `--export_html` 生成的 DX-COM HTML 摘要,或使用 DX-TRON(教程 01 第 2.3 节)。DX-TRON 仍随 SDK 一起发布,但已不再积极维护,因此推荐以 HTML 摘要作为报告。

<!-- cell: 0898f061 src: 1303aaeedb -->
## 4. 创建校准配置

校准图片对于在量化过程中保持高精度非常重要。

<img src="assets/calibration.jpeg" style="max-width: 1000px;" alt="量化期间使用的代表性校准图片">

该配置既描述模型输入契约,也描述源图片如何变为输入张量。

| 字段 | 用途 |
|---|---|
| `inputs` | 精确的 ONNX 输入名称和静态形状 |
| `calibration_method` | 用于估计量化范围的观察器 |
| `calibration_num` | 使用的代表性样本数量 |
| `default_loader.dataset_path` | 包含校准输入的目录 |
| `preprocessings` | 有序的图片到张量变换 |

预处理的顺序很重要。这些值必须与原始模型训练和评估时使用的预处理一致。

<!-- cell: mobilenet-preflight-checklist src: 1110d1b61b -->
### 4.1 编译前的预检清单

在开始可能耗时较长的编译之前,请运行以下检查:

- ONNX 文件通过 `onnx.checker`;
- 每个输入维度都是静态的;
- ONNX 输入名称和形状与编译器 JSON 完全一致;
- 校准目录存在且包含受支持的文件;
- 校准预处理与模型训练和评估时的预处理一致;以及
- 所选的 DX-COM 可执行文件存在。

> **关键要求 — batch size 必须为 `1`。** ONNX 输入和 JSON 中的 `inputs` 条目都必须以 `1` 开头,例如 `[1, 3, 224, 224]`。
>
> DXNN 模型面向 DEEPX NPU 的单样本执行契约,因此不同的或动态的 batch 维度不符合编译和运行时契约。若要提高吞吐量,请提交多个推理请求或使用异步流水线;不要增大模型的 batch 维度。

下一个单元格将该清单转化为可执行的检查,并在某项要求不满足时在运行 DX-COM 之前停止。

<!-- cell: be4871a6 src: 4880623ba4 -->
## 5. 编译为 DXNN

### 5.1 编译

`--gen_log` 保留编译器日志,`--export_html` 生成模型摘要。该命令保持错误可见,并使用专用的输出目录。如果输出目录中已经存在 DXNN 文件,单元格会打印 `Skip compilation: found ...` 并且不再运行 `dxcom`;删除该目录即可强制重新编译。

下一个单元格等价于以下终端命令:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
dxcom -m models/mobilenet_v2.onnx \
      -c configs/mobilenet_v2.json \
      -o outputs/mobilenet_v2_q_lite \
      --gen_log \
      --export_html
```

> **关于代码单元格末尾的 `2>&1 | sed ... | grep ...`:** `dxcom` 使用 Jupyter 无法渲染的光标移动转义码绘制进度条,否则会显示为数百行空行。该过滤器只丢弃进度条行;每条 `[INFO]`、`[WARNING]` 和 `[ERROR]` 消息都保持可见。在终端中可以省略该过滤器。`--gen_log` 会把完整输出保存在 `compiler.log` 中。本笔记本的每个编译单元格都使用同样的过滤器。

<!-- cell: c56da884 src: 824e864a9a -->
### 5.2 检查与基准测试

`dxparse -v` 报告编译后模型的结构和张量元数据。随后 `dxrun --use-ort -t 5` 执行一次五秒钟的合成输入基准测试。

<!-- cell: 85f6c539 src: 70726cba9e -->
如果编译器报告 `[INFO] Added nodes`,请检查哪些预处理操作被插入到了图中。不要在运行时应用中再次应用相同的归一化、颜色转换或转置。

<!-- cell: 8bc0a25d-367f-411d-96b4-5a0ca6d02ad4 src: cfccad8843 -->
### 5.3 性能基准测试与精度评估是两回事

| | `dxrun` 合成基准测试 | 数据集精度评估 |
|---|---|---|
| 输入 | 生成的虚拟输入 | 真实的带标签验证数据集 |
| 主要结果 | 运行时吞吐量和延迟 | Top-1、Top-5、mAP 或 mIoU 等任务指标 |
| 检查内容 | 编译后的模型能够执行及其运行时性能 | 预处理、推理、后处理和预测质量 |
| **不能**证明的内容 | 模型精度 | 最终应用负载下的部署延迟 |

> **编译成功和 `dxrun` 结果很快并不能证明模型精度。**

关于精度测量,请参阅 [DEEPX-AI/dx-modelzoo](https://github.com/DEEPX-AI/dx-modelzoo) 及其[模型评估指南](https://github.com/DEEPX-AI/dx-modelzoo/blob/main/docs/source/guides/evaluation.md)。DX-ModelZoo 通过模型配置把数据集、预处理、运行时配置文件、后处理和评估器连接起来,并报告特定任务的指标。比较 ONNX 和 DXNN 结果时,请使用同一个带标签的验证集。

<!-- cell: 74b9a787 src: d8d71d6109 -->
## 6. 编译来自 DEEPX Model Zoo 的模型

[DEEPX Model Zoo](https://developer.deepx.ai/modelzoo/) 提供可搜索的模型元数据和可下载的产物,包括 ONNX 模型、DXNN 模型以及受支持量化变体的编译器 JSON 文件。

本练习使用 **Resnet50**,因为它的 ONNX 文件较小。对于更大的模型,工作流程相同:

1. 选择模型和量化变体,
2. 下载其 ONNX 和匹配的 JSON,
3. 检查 ONNX 输入契约,
4. 调整与环境相关的 JSON 值,以及
5. 编译到新的输出目录。

<!-- cell: 0d7e5ebb-7b55-4331-ad08-ef736c05424c src: 683283911f -->
### 6.1 下载 Resnet50 ONNX 文件和 dxcom 配置文件(json)

<!-- cell: 250b29cb src: a1a1b7cc23 -->
### 6.2 更新 dxcom 配置文件(json)中的校准路径

Model Zoo 的 JSON 文件包含用于生成已发布模型的校准设置。其中的数据集路径属于构建环境,通常不会存在于您的计算机上。请保留模型特定的预处理,但把数据集路径替换为您本地的代表性数据集。

这里使用 SDK 示例图片只是为了让编译器工作流程可复现。若要做精度方面的决策,请使用来自真实部署领域的样本。

下载的文件保持不变,保存为 `configs/resnet50_224x224.modelzoo.json`;调整后的副本写入 `configs/resnet50_224x224.local.json`。将两者分开意味着重新运行下载单元格永远不会覆盖您的修改,`wget --continue` 也永远不会向已编辑的文件追加内容。

<!-- cell: b45307f5-9e35-4bcb-9e33-7814434037be src: 3464f02b41 -->
### 6.3 编译

下一个单元格等价于以下终端命令:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
dxcom -m models/resnet50_224x224.onnx \
      -c configs/resnet50_224x224.local.json \
      -o outputs/resnet50_224x224_q_lite \
      --gen_log \
      --export_html
```

与 5.1 一样,当输出目录中已经存在 DXNN 文件时,单元格会跳过 `dxcom`,并且进度条会被过滤掉。

<!-- cell: dbd331d3-99c3-4615-acb9-4ea826e758fd src: 29085b4480 -->
### 6.4 检查与基准测试

<!-- cell: 9236bfd1 src: 5dfd2e5d63 -->
### 6.5 打开最新的 HTML 编译报告

启用 `--export_html` 时,DX-COM 会生成 HTML 模型摘要。下一个单元格在本教程的输出目录下查找最近生成的报告,并显示一个按钮,用于在新的浏览器标签页中打开它。

<!-- cell: eb56f5d4 src: 9857c6dd8b -->
## 7. 故障排除

| 症状 | 看到的内容 | 可能原因 | 解决方法 |
|---|---|---|---|
| `dxcom: command not found` | 在终端中 | DX-COM 环境未激活 | 先执行 `source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate`;单元格在每个 `!` 行都会这样做 |
| 输入名称错误 | `KeyError` 或 `input ... not found in model` | JSON 中的 `inputs` 与 `model.graph.input` 不匹配 | 对照第 4 节的 PASS 检查比较名称并修正 JSON |
| 动态形状或 batch 错误 | `Dynamic shape is not supported` 或 batch size 错误 | ONNX 导出使用了动态轴或 batch size > 1 | 以 batch size 1 导出静态形状 |
| 没有校准文件 | `dataset_path` 下 `No images found` | 路径或文件扩展名错误 | 检查 JSON 中的 `dataset_path` 和 `file_extensions` |
| 修改 JSON 不起作用 | `Skip compilation: found ...` | 输出目录中已经存在 DXNN | 删除 `workspace/outputs/<dir>` 并重新运行编译单元格 |
| ONNX 导出失败 | `ModuleNotFoundError: No module named 'onnxscript'` | torch 2.9+ 的 ONNX 导出器需要 `onnxscript` | 再次运行第 3 节的安装单元格;它会安装 `onnxscript`。或者向 `torch.onnx.export` 传入 `dynamo=False` |
| 编译单元格找不到 venv | `{DX_COMPILER_VENV}/bin/activate: No such file or directory` | 由于之前的单元格失败,命令中的变量未定义;IPython 因此不展开整条命令 | 向上滚动到第一个失败的单元格,修复并重新运行它,然后重新运行编译单元格 |
| `wget` 报告 `416 Requested Range Not Satisfiable` | 重新运行下载单元格时 | `--continue` 发现文件已经完整 | 无需处理;文件已完整 |
| 运行时预测错误 | 尽管 `dxcom` 成功但精度很低 | 应用中的预处理与 JSON 不同 | 验证通道顺序、缩放、归一化和缩放策略;使用带标签的数据测量精度 |

请把每次实验放在各自的输出目录中。这样编译器报告和二进制文件都可以追溯。

<!-- cell: de985ffc-1b44-4182-94cd-8a16c7dfb545 src: 335b61b8ab -->
## 8. 总结

### 8.1 您完成的工作流程

**PyTorch 模型**  
→ **ONNX 导出与验证**  
→ **JSON 配置与校准数据**  
→ **DX-COM 编译**  
→ **DXNN 检查与基准测试**

<img src="assets/dx-compile-progress.png"
     style="max-width: 1000px; width: 100%;"
     alt="DX-COM 编译工作流程">

### 8.2 输入与输出

| 阶段 | 主要产物 | 验证方式 |
|---|---|---|
| 模型准备 | `mobilenet_v2.onnx` | `onnx.checker` |
| 编译器配置 | `mobilenet_v2.json` | 预检清单 |
| 编译 | `mobilenet_v2.dxnn` | 输出文件检查 |
| 结构检查 | DXNN 元数据 | `dxparse -v` |
| 性能检查 | 虚拟输入基准测试 | `dxrun` |
| 精度评估 | 带标签的验证数据集 | DX-ModelZoo |

### 8.3 完成清单

- [ ] 将 PyTorch 模型导出为 ONNX
- [ ] 验证 ONNX 输入名称和静态形状
- [ ] 确认 batch size 为 `1`
- [ ] 创建校准配置
- [ ] 将 ONNX 编译为 DXNN
- [ ] 使用 `dxparse` 检查编译后的模型
- [ ] 使用 `dxrun` 测量运行时性能
- [ ] 使用带标签的数据集评估模型精度

> **请记住:** 编译成功和 `dxrun` 结果很快并不能
> 证明模型精度。精度必须另行使用
> 有代表性的带标签数据进行测量。

### 8.4 下一步

继续学习**中级教程**,了解:

- 校准数据集设计;
- Type 0 和 Type 1 硬件 PPU;
- PPU 与非 PPU 的性能比较;以及
- 基于 YOLO26 TopK 的优化。
