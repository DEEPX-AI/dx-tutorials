<!-- i18n source: notebooks/T05-DX-Compiler/dx_com_02_intermediate.ipynb -->
<!-- i18n lang: zh -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: intro src: bb61e22e24 -->
# DEEPX Tutorial 05-2 - DX-COM 中级

本笔记本从一次成功的编译出发,进一步介绍可控且可解释的编译工作流程。

您将改进校准决策,使用 PPU Type 0 和 Type 1 构建真实的 Model Zoo 模型,并测量 TopK 优先优化对 YOLO26 的影响(默认为 n;只需修改一个变量即可切换到 s、m、l 和 x)。

<!-- cell: 6dffc67d-0a31-4b77-b975-65a83973f314 src: caf8fba665 -->
> 本教程共 3 部分,这是第 2 部分。课程地图和完整的部分列表见入门笔记本。

<!-- cell: objectives src: 46a3025a59 -->
## 学习目标

完成本教程后,您将能够:

- 选择具有代表性的校准数据并复现模型预处理,
- 说明哪些检测操作在 NPU 的 PPU 上运行、哪些仍留在主机 CPU 上,
- 根据检测头架构选择 PPU Type 0 或 Type 1,
- 对照 ONNX 图验证 PPU 层映射,
- 使用硬件 PPU 编译 Model Zoo 的 YOLOv7 和 YOLOX-S 模型,
- 比较 PPU 与非 PPU 的 Model Zoo DXNN 运行时行为,
- 对任意尺寸(n、s、m、l、x)的 YOLO26 模型应用 TopK 优先优化,以及
- 使用 `dxrun` 比较基线 DXNN 与优化后 DXNN 的性能。

本笔记本不会修改 SDK 源码树。所有生成的文件都存放在 `<dx-tutorials>/notebooks/T05-DX-Compiler/workspace` 下,该目录被 git 忽略。

<!-- cell: 64617a25-7ca0-407c-97d2-80bcd28d505c src: f0911d99ba -->
## 前提条件

- 已完成教程 05-1:`venv-dx-compiler-local` 中的 `dxcom` 以及示例校准数据集。
- 已安装 DX-RT 并配有 DEEPX NPU,用于第 4 节和第 6 节的 `dxrun` 比较。
- 下载量:来自 Model Zoo 的 YOLOv7 和 YOLOX 的 ONNX 与 DXNN 文件(约 290 MB)。
- 耗时:YOLOv7 640x640 的编译是本系列中最长的;在笔记本电脑上预计需要数十分钟。

下一个单元格会定位 SDK、检查这些要求并打印状态表。任何标记为 `MISSING` 的项目都会附带提供它的步骤。

<!-- cell: workspace-heading src: 98620d01b8 -->
## 1. 初始化教程工作区

<!-- cell: calibration src: d60c6fc7ed -->
## 2. 校准是模型定义的一部分

量化会观测由校准输入产生的浮点张量。一组方便获取但与任务无关的图像也能生成 DXNN 文件,却会降低精度。

| 决策 | 推荐做法 |
|---|---|
| 样本 | 覆盖真实的光照、尺度、背景和类别分布 |
| 预处理 | 与训练和评估完全一致 |
| 数量 | 从模型配方开始;只有在测量稳定性之后才增加 |
| 方法 | 从已发布的配置开始,然后用指标比较各种方法 |
| 验证 | 在留出数据集上比较 ONNX 与 DXNN 的输出 |

SDK 的 `calibration_dataset` 链接可以让本教程正常运行,但它不能替代针对具体任务的数据集。

<!-- cell: preprocessing-order src: 1a5af2c2fa -->
### 2.1 预处理顺序

典型的检测流水线如下:

```text
image → letterbox/pad → BGR-to-RGB → divide by 255 → HWC-to-CHW → add batch axis
```

不要盲目照搬这个顺序。请检查模型导出器和训练代码。不同的填充位置、填充值、颜色顺序或归一化方式都会改变校准张量的分布。

<!-- cell: ppu-overview src: 906986dee8 -->
## 3. PPU 概述

**后处理单元(Post-Processing Unit,PPU)** 是 DEEPX NPU 内部的硬件。对于受支持的 YOLO 检测头,它可以减少主机 CPU 必须处理的原始候选框数量。

PPU 执行两项选定的操作:

1. **置信度过滤**:移除低于编译时阈值的候选框,以及
2. **类别预测**:为每个剩余的候选框选出得分最高的类别。

PPU **不**执行非极大值抑制(NMS)。NMS 和应用层面的结果解析仍在主机 CPU 上运行。PPU 也不会取代图像预处理或 NPU 推理。

<!-- cell: ppu-flow src: 2b907dff84 -->
### 3.1 PPU 在检测流水线中的位置

```text
Without hardware PPU
┌──────────┐   ┌───────────────┐   ┌──────────────────────────────┐   ┌─────────┐
│ Input    │ → │ NPU inference │ → │ CPU filtering/class selection│ → │ CPU NMS │
└──────────┘   └───────────────┘   └──────────────────────────────┘   └─────────┘

With PPU Type 0 or Type 1
┌──────────┐   ┌───────────────┐   ┌────────────────────────────┐   ┌─────────┐
│ Input    │ → │ NPU inference │ → │ PPU filter/class prediction│ → │ CPU NMS │
└──────────┘   └───────────────┘   └────────────────────────────┘   └─────────┘
                                           hardware                    host
```

| 阶段 | 无 PPU | 使用 PPU Type 0/1 |
|---|---|---|
| 神经网络推理 | NPU | NPU |
| 置信度过滤 | 主机 CPU | PPU 硬件 |
| 最佳类别选择 | 主机 CPU | PPU 硬件 |
| 边界框解码 | 取决于模型/类型 | 取决于模型/类型 |
| NMS | 主机 CPU | 主机 CPU |

主要收益是降低主机 CPU 的工作量并减少中间检测数据。具体的端到端收益取决于模型、阈值、场景、主机 CPU 以及应用的后处理。

<!-- cell: ppu-types src: 7508872dcf -->
### 3.2 根据模型架构选择模式

| 模式 | 架构 | 典型模型 | 层映射 | 执行位置 |
|---|---|---|---|---|
| PPU Type 0 | 基于锚框(anchor-based) | YOLOv3、YOLOv4、YOLOv5、YOLOv7 | 检测 Conv 节点 → 锚框数量 | PPU 硬件 |
| PPU Type 1 | 无锚框(anchor-free) | YOLOX、YOLOv8–YOLOv12 | `bbox`、`obj_conf`(如存在)和 `cls_conf` 节点 | PPU 硬件 |
| `pre_optimize()` | TopK 优先的 ONNX 重写 | YOLOv8 系列、YOLOv10、YOLO26 | 每个尺度的 bbox/class 输出张量 | NPU + 主机 CPU 图 |

不要仅凭模型名称选择类型。请确认导出的 ONNX 检测头结构,并使用该文件中的实际节点名。

<!-- cell: ppu-config-anatomy src: 7ae6e1120b -->
### 3.3 读懂 PPU 配置

```text
PPU configuration
├── type           selects the supported head architecture
├── conf_thres     fixed confidence threshold compiled into the DXNN
├── num_classes    class count of this exported model
├── activation     Type 0 activation, usually Sigmoid
└── layer          exact ONNX head-node mapping
```

| 字段 | Type 0 | Type 1 | 重要性 |
|---|:---:|:---:|---|
| `type` | 必需 | 必需 | 选择硬件数据通路 |
| `conf_thres` | 必需 | 必需 | 控制有多少候选框离开 PPU |
| `num_classes` | 必需 | 必需 | 必须与检测头的通道布局一致 |
| `activation` | 必需 | 不使用 | 应用基于锚框的得分激活函数 |
| `layer` | 字典 | 列表 | 将 PPU 输入连接到确切的 ONNX 节点 |
| `num_anchors` | 每层设置 | 不使用 | 必须与每个尺度的锚框数量一致 |
| `obj_conf` | 不使用 | 取决于模型 | YOLOX 有独立的 objectness 分支 |

**重要:** `conf_thres` 在编译时固定。之后更改需要重新生成 DXNN。较高的值可以减少主机工作量,但也可能过滤掉有效的检测结果。请在部署前测量精度。

<!-- cell: type0-heading src: 269378c531 -->
## 4. PPU Type 0 实验:来自 Model Zoo 的 YOLOv7

YOLOv7 使用基于锚框的检测头,因此本实验使用 PPU Type 0。您将下载实际的 Model Zoo ONNX 及其配套的 PPU JSON,验证映射的 Conv 节点,调整数据集路径,并编译出 DXNN。

<!-- cell: download-helper-heading src: 110755aed3 -->
### 4.1 安全地下载 Model Zoo 文件

下一个单元格从 `tutorial_paths.py` 导入 `download_file`(教程 05-2 和 05-3 共用的辅助函数;可打开该文件阅读)。如果目标文件已存在且非空,辅助函数会跳过下载,不发起网络请求。需要重新下载时,请先删除该本地文件。对于新的下载,辅助函数会检查远程文件大小,写入临时 `.part` 文件,并仅在传输完整后才替换目标文件。DNS 或网络故障会抛出异常,不会留下空的最终文件。

本实验下载用于编译的 ONNX 和 PPU JSON,同时下载已发布的非 PPU DXNN 作为基准参考。

<!-- cell: inspect-type0-heading src: cb55365789 -->
### 4.2 检查 ONNX 与 Type 0 映射

对于 Type 0,`layer` 是一个字典。每个键必须是检测头 Conv 节点的名称,且 `num_anchors` 必须与该尺度一致。输出通道数满足:

```text
channels = num_anchors × (5 + num_classes)
         = 3 × (5 + 80)
         = 255
```

<!-- cell: yolov7-diagram src: d3c1a04b7a -->
高亮的 Conv 节点是 Model Zoo PPU 配置所使用的三个尺度各自的检测头。

![YOLOv7 Type 0 PPU 检测头映射](assets/yolov7-class-n80-ppu.png)

<!-- cell: adapt-type0-heading src: e20da5e57c -->
### 4.3 仅调整与环境相关的路径

Model Zoo JSON 包含该模型使用的预处理配方。实验中请保持其不变,只替换不可用的校准数据集路径。对于产品模型,请使用来自部署领域的代表性数据集。

<!-- cell: compile-helper-heading src: 320ebf992e -->
### 4.4 编译并检查 Type 0 DXNN

下一个代码单元格会激活 DX-COM 环境并直接运行 `dxcom`。它等价于在单独的终端中输入以下命令:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/yolov7_640x640.onnx \
      -c configs/yolov7_640x640_ppu.local.json \
      -o outputs/yolov7_type0_ppu \
      --gen_log \
      --export_html
```

实际的 SDK 和教程路径可能不同。代码单元格使用从 `config.json` 加载的路径。如果输出目录中已经存在 DXNN 文件,它会打印跳过消息,不会再次运行 `dxcom`。

> **关于代码单元格末尾的 `2>&1 | sed ... | grep ...`:** `dxcom` 使用 Jupyter 无法渲染的光标移动转义码绘制进度条,否则会显示为数百行空行。该过滤器只丢弃这些进度条行;所有 `[INFO]`、`[WARNING]` 和 `[ERROR]` 消息都保持可见。在终端中可以省略该过滤器。`--gen_log` 会把完整输出保存在 `compiler.log` 中。本笔记本的每个编译单元格都使用同样的过滤器。

<!-- cell: benchmark-yolov7-heading src: 50a17e9138 -->
### 4.5 使用 `dxrun` 比较 PPU 与非 PPU

非 PPU DXNN 直接从 Model Zoo 下载,而 PPU DXNN 是 4.4 节编译的结果。下一个单元格使用合成输入将两个模型各运行五秒:

```bash
dxrun -m models/yolov7_640x640_non_ppu.dxnn --use-ort -t 5
dxrun -m outputs/yolov7_type0_ppu/<compiled-model>.dxnn --use-ort -t 5
```

非 PPU 模型包含 NPU 和 CPU 任务,返回原始检测值。PPU 模型返回由硬件生成的 `BBOX` 数据,减少了主机需要处理的输出。PPU **不**保证更高的合成输入 FPS:模型调度、输出传输和设备状态都可能使非 PPU 结果在这个孤立测试中更快。做产品决策时,还应使用真实输入测量完整流水线延迟和主机 CPU 占用率。

<!-- cell: type1-heading src: 103ddfd632 -->
## 5. PPU Type 1 实验:来自 Model Zoo 的 YOLOX-S

YOLOX 使用解耦的无锚框检测头。每个尺度都有独立的边界框、objectness 和类别置信度分支,因此本实验使用 PPU Type 1。

<!-- cell: download-type1-heading src: dcfccc5610 -->
### 5.1 下载 ONNX 和 Type 1 JSON

复用同一个安全下载器。ONNX 和 PPU JSON 用于编译,已发布的非 PPU DXNN 用作基准参考。

<!-- cell: inspect-type1-heading src: 80d640adfd -->
### 5.2 检查 ONNX 与 Type 1 映射

对于 YOLOX,每个尺度映射三个具名节点:

```text
feature map ─┬─ bbox branch ─────→ bbox
             ├─ object branch ───→ obj_conf
             └─ class branch ────→ cls_conf
```

三个条目分别对应 80×80、40×40 和 20×20 的检测尺度。

<!-- cell: yolox-diagram src: 8a4b542192 -->
颜色标出了每个尺度上必须配对的 `bbox`、`obj_conf` 和 `cls_conf` 分支。

![YOLOX Type 1 PPU 检测头映射](assets/yolox-class-n80-ppu.png)

<!-- cell: adapt-type1-heading src: 7559b7cea7 -->
### 5.3 调整校准数据集路径

保留下载的 Type 1 映射和 YOLOX 预处理。本练习只替换 Model Zoo 构建机器上的数据集路径。

<!-- cell: compile-type1-heading src: 818827cba8 -->
### 5.4 编译并检查 Type 1 DXNN

下一个代码单元格等价于在单独的终端中输入以下命令:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/yolox-s_640x640.onnx \
      -c configs/yolox-s_640x640_ppu.local.json \
      -o outputs/yolox_s_type1_ppu \
      --gen_log \
      --export_html
```

代码单元格会自动解析实际路径。如果输出目录中已经存在 DXNN 文件,它会跳过 `dxcom`。请将其 `dxparse -v` 输出与 Type 0 的结果进行比较,尤其是输出张量布局和 PPU 元数据。

<!-- cell: benchmark-yolox-heading src: 2ae99d790d -->
### 5.5 使用 `dxrun` 比较 PPU 与非 PPU

对 YOLOX-S 重复同样的受控测试。非 PPU 参考是 5.1 节下载的已发布 Model Zoo DXNN;它不在本教程中编译。

```bash
dxrun -m models/yolox-s_640x640_non_ppu.dxnn --use-ort -t 5
dxrun -m outputs/yolox_s_type1_ppu/<compiled-model>.dxnn --use-ort -t 5
```

请结合 `dxparse` 显示的任务结构来解读这个合成基准测试。即使这个单一的 FPS 数字没有提高,PPU 模型也可以减少主机侧处理和输出传输量。

<!-- cell: topk-heading src: a8f8add50a -->
## 6. YOLO26 基于 TopK 的优化(n / s / m / l / x)

YOLO26 使用一对一的解耦检测头。`yolo26_postprocess` 变换把 TopK 选择移到开销较大的 CPU 侧解码工作之前。这与 PPU Type 0/1 不同:它重写 ONNX 图,减少 NPU 推理之后需要处理的候选框工作量。

```text
Baseline: 8,400 candidates → decode and score all candidates → TopK 300
Optimized: 8,400 candidates → TopK 300 → decode and score only 300 candidates
```

两个模型保持相同的检测结果契约 `[1, 300, 6]`,但 CPU 侧工作的顺序发生了变化。

本节适用于 Model Zoo 中的所有 YOLO26 尺寸。6.1 中的一个变量 `YOLO26_VARIANT` 用于选择模型;下面的每个文件名、检测头张量检查、编译和基准测试都随之变化。五个导出模型共享相同的检测头布局,因此 TopK 变换完全相同。先从 `n` 开始,然后修改该变量并重新运行本节以处理更大的模型。

| `YOLO26_VARIANT` | ONNX 下载量 | 每个模型的编译时间(x86_64 台式机;本节编译两个模型) |
|---|---:|---|
| `n`(默认) | 10 MB | 约 2 分钟(实测) |
| `s` | 37 MB | 约 2 分钟(实测) |
| `m` | 78 MB | 未实测;预计数分钟 |
| `l` | 95 MB | 未实测;预计 10 分钟或更长 |
| `x` | 213 MB | 未实测;预计 10 分钟或更长 |

每个变体在 `models/`、`configs/` 和 `outputs/` 下使用各自的文件,因此切换回已编译过的尺寸时会跳过编译。

<!-- cell: download-yolo26-heading src: 306e030b72 -->
### 6.1 下载并验证基线模型

设置 `YOLO26_VARIANT` 并运行单元格。基线 ONNX 及其配套的 Q-Lite JSON 来自 Model Zoo(`yolo26-<variant>_640x640.onnx` 和 `.json`)。安全下载器可以避免 shell 下载失败而下一个笔记本单元格仍继续执行时可能出现的空文件错误。

<!-- cell: verify-heads-heading src: adfa304cc8 -->
### 6.2 验证六个检测头张量

该变换需要三个检测尺度上各一个边界框张量和一个类别置信度张量。这些是**输出张量名**,而不是从其他模型复制来的显示标签。导出到 Model Zoo 的所有 YOLO26 尺寸共享相同的检测头布局(边界框为 `/model.23/one2one_cv2.<scale>/...`,类别得分为 `/model.23/one2one_cv3.<scale>/...`),因此下面的名称由一个模式生成,再对照实际下载的 ONNX 进行检查。如果检查失败,请在 Netron 中打开模型并更新该模式。

<!-- cell: apply-topk-heading src: 1ac20c8599 -->
### 6.3 应用 `yolo26_postprocess`

该变换使用 DX-COM Python 环境运行,因为 `dx_com` 安装在那里。它会创建一个单独的 ONNX 文件,因此基线保持不变。

脚本以基线 ONNX 和目标路径作为参数,因此同一个脚本适用于所有变体。接下来的两个单元格分别写入脚本并运行它。对于默认的 `n` 变体,运行它等价于:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
python optimize_yolo26.py models/yolo26-n_640x640.onnx models/yolo26-n_640x640_topk300.onnx
```

<!-- cell: adapt-yolo26-heading src: f1bf82c079 -->
### 6.4 调整 Model Zoo 配置

基线模型和优化模型必须使用相同的校准与预处理配置。只更改不可用的数据集路径。这样可以保证性能比较是受控的。

<!-- cell: compile-yolo26-heading src: e70f6d348a -->
### 6.5 编译并检查两个模型

将每个 ONNX 编译到单独的输出目录。两条命令使用相同的 JSON 和编译器选项;ONNX 图是唯一的实验变量。

<!-- cell: compile-yolo26-baseline-command src: fa81283b97 -->
#### 6.5.1 编译基线模型

对于默认的 `n` 变体,下一个代码单元格等价于在单独的终端中输入以下命令(其他变体相应地更改 `yolo26-n` 和 `yolo26n`):

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/yolo26-n_640x640.onnx \
      -c configs/yolo26-n_640x640.local.json \
      -o outputs/yolo26n_baseline \
      --gen_log \
      --export_html
```

如果输出目录中已经存在 DXNN 文件,代码单元格会报告该文件并跳过 `dxcom`。

<!-- cell: compile-yolo26-topk-command src: 1c1158411a -->
#### 6.5.2 编译 TopK 模型

对于默认的 `n` 变体,下一个代码单元格等价于在单独的终端中输入以下命令(其他变体相应地更改 `yolo26-n` 和 `yolo26n`):

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/yolo26-n_640x640_topk300.onnx \
      -c configs/yolo26-n_640x640.local.json \
      -o outputs/yolo26n_topk300 \
      --gen_log \
      --export_html
```

如果输出目录中已经存在 DXNN 文件,代码单元格会报告该文件并跳过 `dxcom`。

<!-- cell: benchmark-heading src: 917423f3f9 -->
### 6.6 使用 `dxrun` 比较基线与 TopK 性能

`dxrun --use-ort -t 5` 使用合成输入对每个 DXNN 进行五秒的基准测试。`--use-ort` 除 NPU 图外还会执行 CPU 子图,这是本次比较所必需的。

请在同一设备上、相近的温度和系统负载条件下运行两个模型。该结果测量的是运行时吞吐量,而不是检测精度或完整视频流水线的 FPS。做发布决策时请重复测量。

对于默认的 `n` 变体:

```bash
dxrun -m outputs/yolo26n_baseline/yolo26-n_640x640.dxnn --use-ort -t 5
dxrun -m outputs/yolo26n_topk300/yolo26-n_640x640_topk300.dxnn --use-ort -t 5
```

<!-- cell: 6be8d246-d16c-429d-a82f-e1c45594974c src: b7be7c85e3 -->
## 7. 故障排除

| 症状 | 看到的内容 | 可能原因 | 解决方法 |
|---|---|---|---|
| 找不到 PPU 节点名 | `ValueError: ... not found in the ONNX graph` | PPU 配置中的 `layer` 名称属于另一个导出模型 | 在 Netron 中检查 ONNX 并更新 JSON 中的节点名 |
| `dxrun` 找不到设备 | `No device found` 或 `Fail to initialize device` | 本主机上没有 DEEPX NPU | 在配有 NPU 的主机上运行比较;仅编译的步骤仍可正常工作 |
| 无法读取 FPS | `Could not read FPS from dxrun output` | `dxrun` 的输出格式已更改 | 在上方打印完整输出并手动读取 FPS 行 |
| 修改 JSON 没有效果 | `Skip compilation: found ...` | 输出目录中已经存在 DXNN | 删除 `workspace/outputs/<dir>` 并重新运行 |
| 下载不完整 | `workspace/models` 中出现极小或零字节的文件 | 下载被中断 | 删除该文件;下载辅助函数会在使用前验证大小 |

<!-- cell: summary src: a42f00d89b -->
## 8. 总结

### 8.1 中级优化路线图

```text
Representative calibration data
              │
              ▼
     Inspect the detection head
              │
      ┌───────┴────────┐
      │                │
Anchor-based      Anchor-free
  YOLOv7             YOLOX
      │                │
PPU Type 0        PPU Type 1
      │                │
      └───────┬────────┘
              │
              ▼
 Reduce host-side filtering
 and class-selection workload

YOLO26 decoupled head
          │
          ▼
 TopK-first ONNX rewrite
          │
          ▼
 Reduce candidates before
 CPU-side decoding
```

### 8.2 三条优化路径

| 主题 | 使用的模型 | 主要决策 | 优化位置 | 主机 CPU 的职责 |
|---|---|---|---|---|
| PPU Type 0 | YOLOv7 | 选择基于锚框的检测头并映射其 Conv 节点 | 硬件 PPU | NMS 和应用逻辑 |
| PPU Type 1 | YOLOX-S | 映射 bbox、objectness 和类别分支 | 硬件 PPU | NMS 和应用逻辑 |
| TopK 优先优化 | YOLO26(`YOLO26_VARIANT`,默认 n) | 把 TopK 移到开销较大的解码之前 | 重写后的 ONNX 图 | 解码更少的候选框 |

### 8.3 已完成的实验

| 实验 | 比较的模型 | 控制变量 | 验证方式 |
|---|---|---|---|
| Type 0 PPU | YOLOv7 非 PPU 与 PPU | 硬件后处理 | `dxparse`、`dxrun` |
| Type 1 PPU | YOLOX-S 非 PPU 与 PPU | 硬件后处理 | `dxparse`、`dxrun` |
| TopK 优化 | YOLO26 基线与 TopK 300(所选尺寸) | ONNX 图结构 | 输出契约、`dxparse`、`dxrun` |

### 8.4 完成清单

- [ ] 将校准数据和预处理视为模型定义的一部分
- [ ] 识别 NPU、PPU 与主机 CPU 处理之间的边界
- [ ] 验证 YOLOv7 Type 0 的 Conv 节点映射
- [ ] 编译并检查 Type 0 PPU 模型
- [ ] 验证 YOLOX-S Type 1 的 bbox、objectness 和类别映射
- [ ] 编译并检查 Type 1 PPU 模型
- [ ] 在相同的基准测试条件下比较 PPU 与非 PPU 模型
- [ ] 对 YOLO26 模型应用 TopK 优先优化(并尝试第二种尺寸)
- [ ] 验证基线模型与 TopK 模型保持相同的输出契约
- [ ] 比较基线 DXNN 与 TopK DXNN 的运行时性能
- [ ] 使用真实数据验证精度、主机 CPU 占用率和完整流水线延迟

### 8.5 如何选择

| 如果您的模型具有... | 从...开始 |
|---|---|
| 基于锚框的 YOLO 检测头 | PPU Type 0 |
| 独立的 bbox、objectness 和类别分支 | PPU Type 1 |
| 大量候选框后接 TopK | TopK 优先的图优化 |
| 未知或自定义的检测头 | 在选择优化方案前检查确切的 ONNX 图 |

> **请记住:** 更低的主机工作量与更高的合成输入 FPS 并不是同一回事。先用 `dxrun` 进行受控的运行时比较,再使用真实的应用工作负载测量任务精度、主机 CPU 占用率和端到端延迟。

### 8.6 下一步

继续学习**高级教程**,了解:

- Q-Lite、Q-PRO 和 Q-Master 的选择;
- 量化诊断;
- QXNN 恢复工作流程;
- QAT;
- Python API 编译;以及
- 高级编译器控制。
