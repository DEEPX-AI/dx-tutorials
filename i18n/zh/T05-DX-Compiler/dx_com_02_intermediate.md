<!-- i18n source: notebooks/T05-DX-Compiler/dx_com_02_intermediate.ipynb -->
<!-- i18n lang: zh -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: intro src: 680d42d3c9 -->
# DEEPX Tutorial 05-2 - DX-COM 中级

本笔记本从一次成功的编译出发,进一步介绍可控且可解释的编译工作流程。

您将改进校准决策,使用 PPU Type 0 和 Type 1 构建真实的 Model Zoo 模型,并对 YOLO26 检测、分割和姿态模型应用 TopK 优先优化(默认为 n;只需修改一个变量即可切换到 s、m、l 和 x)。

<!-- cell: 6dffc67d-0a31-4b77-b975-65a83973f314 src: caf8fba665 -->
> 本教程共 3 部分,这是第 2 部分。课程地图和完整的部分列表见入门笔记本。

<!-- cell: objectives src: ef4f57c018 -->
## 学习目标

完成本教程后,您将能够:

- 选择具有代表性的校准数据并复现模型预处理,
- 说明哪些检测操作在 NPU 的 PPU 上运行、哪些仍留在主机 CPU 上,
- 根据检测头架构选择 PPU Type 0 或 Type 1,
- 对照 ONNX 图验证 PPU 层映射,
- 使用硬件 PPU 编译 Model Zoo 的 YOLOv7 和 YOLOX-S 模型,
- 比较 PPU 与非 PPU 的 Model Zoo DXNN 运行时行为,
- 说明 YOLO26 DXNN 的哪一部分在主机 CPU 上运行,以及 TopK 优先优化会在那里改变什么,
- 自动找到 YOLO26 ONNX 模型的检测头张量,
- 对任意尺寸(n、s、m、l、x)的 YOLO26 检测、分割和姿态模型应用 TopK 优先优化,
- 检查优化后的模型给出相同的结果,并测量它节省的 CPU 时间和端到端吞吐量,以及
- 在不修改应用程序的情况下,在 DX-APP 中运行优化后的 DXNN。

本笔记本不会修改 SDK 源码树。所有生成的文件都存放在 `<dx-tutorials>/notebooks/T05-DX-Compiler/workspace` 下,该目录被 git 忽略。

<!-- cell: 64617a25-7ca0-407c-97d2-80bcd28d505c src: 38b24ba0d7 -->
## 前提条件

- 已完成教程 05-1:`venv-dx-compiler-local` 中的 `dxcom` 以及示例校准数据集。
- 已安装 DX-RT 并配有 DEEPX NPU,用于第 4 节和第 6 节的 `dxrun` 比较。
- 第 6.8 节需要:DX-RT Python 环境(`venv-dx-runtime`,教程 01)和 DX-APP(教程 02)。缺少它们时,只会跳过 DX-APP 相关的单元格。
- 下载量:来自 Model Zoo 的 YOLOv7 和 YOLOX 的 ONNX 与 DXNN 文件(约 290 MB),以及第 6 节使用的 YOLO26 检测、分割和姿态 ONNX 文件(默认 `n` 尺寸约 33 MB)。
- 耗时:YOLOv7 640x640 的编译是本系列中最长的;在笔记本电脑上预计需要数十分钟。第 6 节会编译六个 YOLO26 模型(检测、分割和姿态各一个基线模型和一个 TopK 模型);对于 `n`,在 x86_64 台式机上每个模型约需 2 分钟。

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

<!-- cell: ppu-types src: 69849e3a90 -->
### 3.2 根据模型架构选择模式

| 模式 | 架构 | 典型模型 | 层映射 | 执行位置 |
|---|---|---|---|---|
| PPU Type 0 | 基于锚框(anchor-based) | YOLOv3、YOLOv4、YOLOv5、YOLOv7 | 检测 Conv 节点 → 锚框数量 | PPU 硬件 |
| PPU Type 1 | 无锚框(anchor-free) | YOLOX、YOLOv8–YOLOv12 | `bbox`、`obj_conf`(如存在)和 `cls_conf` 节点 | PPU 硬件 |
| `pre_optimize()` | TopK 优先的 ONNX 重写 | YOLOv8、v9、v11、v12、v13 和 YOLO26(检测、分割、姿态);RTMDet | 每个尺度的检测头输出张量 | NPU + 主机 CPU 图 |

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

<!-- cell: topk-heading src: 7c5984ab3c -->
## 6. YOLO26 TopK 优先优化(n / s / m / l / x)

PPU Type 0 和 Type 1 把检测后处理移到 NPU 硬件中。YOLO26 走的是另一条路:它的后处理仍留在主机 CPU 上,而 DX-COM 的 `pre_optimize()` 会重写这一 CPU 部分,使其工作量大幅减少。本节说明这些工作在哪里运行、重写改变了什么、如何确认结果保持不变,以及能节省多少时间。随后将同样的步骤应用于分割和姿态模型。

### YOLO26 后处理在哪里运行

编译后的 YOLO26 DXNN 包含两个任务。您将在 6.6 中用 `dxparse` 看到它们。

```text
image ─► npu_0 [NPU] ──────────► head tensors ─► cpu_0 [CPU, ONNX Runtime on the host] ─► [1, 300, 6]
         backbone, neck, and     box and class    Sigmoid on the class logits, box decoding,
         the head Convs          logits, 3 scales  keep the best 300 candidates
```

NPU 任务在每个检测头分支的最后一个 Conv 处结束。之后的部分都是普通的 ONNX,由 DX-RT 在主机 CPU 上用 ONNX Runtime 运行;这就是 `dxrun` 需要 `--use-ort` 的原因。在 640x640 下,检测头会产生 8,400 个候选框,因此基线 CPU 任务在每一帧中都要先对 8,400 x 80 = 672,000 个类别 logit 执行 Sigmoid,然后才保留其中的 300 个。

### `pre_optimize()` 改变了什么

```text
Baseline : 8,400 candidates → Sigmoid on every class logit → decode → TopK 300
Optimized: 8,400 candidates → TopK 300 on the raw logits    → Sigmoid and decode for those 300 only
```

Sigmoid 不会改变得分的顺序,因此对原始 logit 执行 TopK 会选出相同的候选框。输出仍为 `[1, 300, 6]`(边界框 `x1 y1 x2 y2`、得分、类别 id),6.5 节会表明其数值完全相同。改变的只是主机 CPU 的工作量。

YOLO26 使用的 pass 是 `yolo_no_dfl_postprocess`(YOLO26 的边界框不使用 DFL)。DX-COM 还提供用于 YOLOv8 至 v13 的 `yolo_dfl_postprocess` 和用于 RTMDet 的 `rtmdet_postprocess`;DX-COM 用户手册的 *Pre-Optimize API* 一节介绍了这三个 pass。

### 何时值得使用

节省发生在主机 CPU 上。当 CPU 部分在每帧中占比较大时,这种节省才有意义:例如 Arm 开发板等较慢的主机 CPU,以及类别数较多的模型。在快速的 x86_64 台式机上,或者对于只有少量类别的模型,收益很小。6.7 节会分别测量 CPU 部分和端到端吞吐量,以便您评估自己的主机。

### 应用该 pass 的两种方式

| 方法 | 做法 | 适用场景 |
|---|---|---|
| Python API | `dx_com.pre_optimize(model, passes=...)` 写出一个新的 ONNX 文件,然后再编译该文件 | 希望在编译前查看并检查重写后的 ONNX(本节采用这种方式) |
| 编译配置 | 在编译 JSON 中加入 `pre_optimize` 块;`dxcom` 在编译时应用该 pass | pass 设置已经过检查,并且希望只用一个编译步骤(见 6.3 中的说明) |

### 本节导览

| 步骤 | 操作内容 | 结果 |
|---|---|---|
| 6.1 | 下载 Model Zoo 的 YOLO26 检测模型 | 基线 ONNX 和 JSON |
| 6.2 | 自动找到检测头张量 | pass 设置 |
| 6.3 | 应用该 pass | TopK ONNX |
| 6.4 | 比较优化前后的 CPU 部分 | 算子与形状表 |
| 6.5 | 检查结果是否完全相同 | 在真实图像上进行 ONNX Runtime 比较 |
| 6.6 | 编译两个模型 | 两个 DXNN 文件及其 NPU/CPU 划分 |
| 6.7 | 测量 | CPU 部分耗时和 `dxrun` FPS |
| 6.8 | 在 DX-APP 中运行 TopK DXNN | 结果图像 |
| 6.9 | 对分割和姿态模型重复 6.1 至 6.8 | |

6.1 中的 `YOLO26_VARIANT` 用于选择 Model Zoo 的模型尺寸,下面的每个文件名、编译和基准测试都随之变化。先从 `n` 开始,然后修改该变量并重新运行本节以处理更大的模型。每个尺寸在 `models/`、`configs/` 和 `outputs/` 下使用各自的文件,因此切换回已编译过的尺寸时会跳过编译。

| `YOLO26_VARIANT` | 检测 ONNX 下载量 | 每个模型的编译时间(x86_64 台式机) |
|---|---:|---|
| `n`(默认) | 10 MB | 约 2 分钟(实测) |
| `s` | 37 MB | 约 2 分钟(实测) |
| `m` | 78 MB | 未实测;预计数分钟 |
| `l` | 95 MB | 未实测;预计 10 分钟或更长 |
| `x` | 213 MB | 未实测;预计 10 分钟或更长 |

<!-- cell: download-yolo26-heading src: 306e030b72 -->
### 6.1 下载并验证基线模型

设置 `YOLO26_VARIANT` 并运行单元格。基线 ONNX 及其配套的 Q-Lite JSON 来自 Model Zoo(`yolo26-<variant>_640x640.onnx` 和 `.json`)。安全下载器可以避免 shell 下载失败而下一个笔记本单元格仍继续执行时可能出现的空文件错误。

<!-- cell: verify-heads-heading src: 86c8f319d0 -->
### 6.2 找到检测头张量

`pre_optimize()` 需要知道 NPU 部分在哪里结束:即每个检测尺度上,每个检测头分支中最后一个 Conv 的输出张量。这些是来自您所用的那个 ONNX 文件的张量名。它们取决于模型和导出器版本,因此切勿从其他模型复制。

检测头张量很容易识别:

1. 它是某个 `Conv` 的输出,形状为 `[1, C, H, W]`,其中 `H = 640 / stride`,stride 为 8、16 或 32(80x80、40x40、20x20)。
2. 它后面没有激活函数;只有 `Reshape` 或 `Concat` 使用它。
3. 它的通道数 `C` 表明它的角色:

| 角色 | 通道数 | task |
|---|---|---|
| `bbox` | 4(到边界框四条边的距离) | 全部 |
| `cls_conf` | 类别数(类别 logit) | 全部 |
| `mask_coeff` | 32 | `seg` |
| `kpt` | 关键点数 x 每个关键点的值数(17 x 3 = 51) | `pose` |

实际模型还有两个细节:

- **通道数相同。** 在 4 类模型中,`cls_conf` 与 `bbox` 一样有 4 个通道。只有类别得分分支会直接进入 `Sigmoid`,据此可以区分二者。
- **两个检测头。** YOLO26 训练时使用one-to-many 和 one-to-one 两个检测头。推理使用 one-to-one 检测头,即张量名中的 `one2one_*`。只包含 one-to-many 检测头的导出模型无法使用该 pass(参见 6.3 中关于 `nms=False` 的说明)。

下一个单元格会写入 `yolo26_heads.py`,这是一个应用上述规则的小模块。它从 ONNX 文件的 Ultralytics 元数据中读取 task、类别数和关键点布局,并在某条规则不成立时停止并给出说明。它只需要 `onnx` 包,因此笔记本和 DX-COM 环境都可以导入它。DX-COM 用户手册的 *Identifying Conv Head Tensor Names* 一节说明了同样的规则,以及如何在 Netron 中检查它们。

<!-- cell: apply-topk-heading src: 45a61021f4 -->
### 6.3 应用 `yolo_no_dfl_postprocess`

`optimize_yolo26.py` 在安装了 `dx_com` 的 DX-COM Python 环境中运行。它使用 `yolo26_heads.py` 找到检测头张量,打印 pass 设置,并把 TopK 模型写入一个单独的 ONNX 文件,因此基线保持不变。同一个脚本适用于所有尺寸和 task,之后也适用于您自己的模型。

pass 设置如下:

| 键 | 值 | 说明 |
|---|---|---|
| `task` | `base`、`seg` 或 `pose` | 必需;决定每个 layer 必须具有哪些检测头张量 |
| `layers` | 每个尺度一个条目,按 stride 8、16、32 排序 | 来自 6.2 |
| `num_classes` | 类别数 | 默认为 80 |
| `topk` | 解码前保留的候选框数量 | 默认为 300;Model Zoo 的pre-optimized 配置使用 512 |

> **DX-COM v2.5.0 中的变更。** 原名 `yolo26_postprocess` 的 pass 现在是 `yolo_no_dfl_postprocess`,`yolo_postprocess`(YOLOv8 到 v13)现在是 `yolo_dfl_postprocess`。使用旧名称会报错。每个 YOLO pass 都需要一个 `task` 键,并且不再接受 `input_height` / `input_width`(输入尺寸从模型中读取)。

> **您自己的 YOLO26 模型。** 如果您自己训练的模型是以 NMS-free 检测头导出的,该脚本同样适用:`yolo export model=best.pt format=onnx imgsz=640 batch=1 nms=False`。如果不加 `nms=False`,当前的 Ultralytics 版本(已在 8.4.208 上测试)会导出 one-to-many 检测头,其输出为 `[1, 4 + classes, 8400]`,此时 `yolo26_heads.py` 会停止并提示重新导出。请保持 `imgsz` 固定;不要使用 `dynamic=True`。`num_classes` 以及姿态模型的 `kpt_shape` 来自您所导出模型的元数据。

> **在编译 JSON 中应用该 pass。** 脚本会以 `{"yolo_no_dfl_postprocess": {...}}` 的形式打印设置。若要让 `dxcom` 在编译原始 ONNX 时应用该 pass,请将该字典放入编译 JSON 中 `pre_optimize` 下的列表:`"pre_optimize": [{"yolo_no_dfl_postprocess": {...}}]`。编译日志随后会显示 `[pre_optimize] applying pass 'yolo_no_dfl_postprocess' from config`,并且 DXNN 沿用该 ONNX 的文件名。不要把 `pre_optimize` 和 `ppu` 放在同一个 JSON 中:DX-COM 2.5.0 会使用 `ppu` 进行编译并丢弃 `pre_optimize`,只打印 `falling back to legacy: ValueError: config 'pre_optimize' and 'ppu' are mutually exclusive`。

接下来的两个单元格分别写入脚本并运行它。对于默认的 `n` 变体,这等价于:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
python optimize_yolo26.py models/yolo26-n_640x640.onnx models/yolo26-n_640x640_topk300.onnx
```

<!-- cell: cpu-part-heading src: 94cb953e24 -->
### 6.4 比较优化前后的 CPU 部分

编译后 DXNN 的 CPU 任务是从检测头张量到模型输出的 ONNX 子图。下一个单元格从两个模型中裁剪出该子图(`onnx.utils.extract_model`),并列出工作量随候选框数量增长的算子。

需要关注的内容:

- **基线:** `Sigmoid` 在 `[1, 80, 8400]` 上运行,即每个候选框的每个类别各一个值,而 TopK 在它之后执行。
- **TopK 模型:** `ReduceMax` 和 `TopK` 先从原始 logit 中选出 300 个候选框,然后 `Sigmoid` 在 `[1, 300, 80]` 上运行,值的数量只有原来的 1/28。

TopK 模型的节点更多,因为它增加了挑选这 300 个候选框的 gather 步骤。这些额外节点处理的是 300 行,而不是 8,400 行。裁剪出的 CPU 部分还会在 6.7 中用于测量 CPU 工作的耗时。

<!-- cell: verify-results-heading src: 368a5b670d -->
### 6.5 检查结果是否完全相同

修改后的图必须先给出相同的结果,才值得编译。接下来的单元格使用 ONNX Runtime 在同一张真实图像上运行两个 ONNX 模型,比较它们的输出,并绘制每个模型的检测结果。

图像的准备方式与 Model Zoo JSON 准备校准图像的方式相同:用值 114 填充为 640x640 的正方形,从 BGR 转换为 RGB,除以 255,并转换为 NCHW。测试图像来自 DX-APP 示例;如果未安装 DX-APP,则使用第一张校准图像。

比较范围涵盖得分高于 0.001(DX Model Zoo 进行 mAP 评估时使用的阈值)的每个候选框,以及其他所有输出(例如分割原型)。得分更低的候选框是有意排除的:在列表靠后的位置,许多候选框的得分完全相同,而 TopK 可能以任意顺序返回这些得分并列的候选框。重写后的模型对原始 logit 而不是 Sigmoid 输出进行排序,因此这类并列项的顺序可能不同。在本笔记本的测试图像上,第一个并列项出现在得分低于 0.0001 的位置。

该比较检查的是重写后的 ONNX 图。编译时会分别对每个模型进行量化,因此两个 DXNN 文件可能略有差异;6.8 会在 NPU 上运行这两个模型。

<!-- cell: verify-results-note src: 98e55a730d -->
如果最大差异为 0 且候选框数量相同,就说明对于这张图像,重写给出了相同的结果。对于您自己的模型,请用您数据中的多张图像运行同样的检查,包括包含大量目标的图像和不含任何目标的图像。

<!-- cell: adapt-yolo26-heading src: 1cce1fabf3 -->
### 6.6 编译两个模型

#### 6.6.1 调整 Model Zoo 配置

两个模型使用相同的校准与预处理配置,因此 ONNX 图是两次编译之间唯一的差异。只更改指向 Model Zoo 服务器的数据集路径。

<!-- cell: compile-yolo26-baseline-command src: 9ec494a9de -->
#### 6.6.2 编译基线模型

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

<!-- cell: compile-yolo26-topk-command src: 6679ececa5 -->
#### 6.6.3 编译 TopK 模型

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

<!-- cell: task-split-heading src: 5ea330f2c1 -->
#### 6.6.4 查看 NPU/CPU 划分

`dxparse -v` 会列出 DXNN 的各个任务以及在任务之间传递的张量。下一个单元格只打印这一部分。在基线模型中,`npu_0` 在 6.2 中找到的检测头张量处结束,`cpu_0` 将它们转换为 `output0`。在 TopK 模型中,NPU 还会为 CPU 部分改变检测头张量的布局(`preopt_*_tr_*`,channels last),而 `cpu_0` 生成 `preopt_output`。

<!-- cell: benchmark-heading src: 021a6c65d3 -->
### 6.7 测量效果

两项测量分别回答两个问题:

1. **该 pass 去掉了多少 CPU 工作?** 6.4 中的 CPU 部分使用 ONNX Runtime 在单线程上运行,输入为测试图像的真实检测头张量。这样可以单独衡量该 pass 所改变的工作。
2. **整个流水线是否变快?** `dxrun --use-ort -t 5` 使用合成输入将每个 DXNN 运行五秒,NPU 部分和 CPU 部分一起执行。

DX-RT 会让相邻帧的 NPU 任务和 CPU 任务重叠执行,因此端到端 FPS 由两者中较慢的一方决定。当 NPU 是较慢的阶段时(这在快速的 x86_64 主机上很常见),缩短 CPU 部分对 FPS 的影响很小。在较慢的主机 CPU 上,CPU 部分会成为瓶颈,测量 1 中节省的时间就会转化为 FPS 的提升。

对于默认的 `n` 变体,`dxrun` 命令如下:

```bash
dxrun -m outputs/yolo26n_baseline/yolo26-n_640x640.dxnn --use-ort -t 5
dxrun -m outputs/yolo26n_topk300/yolo26-n_640x640_topk300.dxnn --use-ort -t 5
```

请在同一设备上、相近的系统负载和温度条件下运行两个模型,并在得出结论前重复测量。这些数字既不衡量精度,也不代表完整视频应用的 FPS。

<!-- cell: benchmark-results src: 7013e3c5ea -->
如何解读结果:

- **CPU 部分。** 这是该 pass 所改变的部分。对于 Model Zoo 模型,它缩短约 35% 至 45%。绝对耗时取决于主机 CPU:同样的工作在 Arm 核心上所需的时间是在 x86_64 台式机核心上的数倍。
- **`dxrun` FPS。** 只有在 CPU 部分限制流水线的程度内,它才会提升。在用于测试本笔记本的配备 DX-M1 的 x86_64 台式机上,YOLO26-n 的 CPU 部分实测约为 0.74 ms → 0.41 ms,但 `dxrun` FPS 只提升约 1.06×,因为 NPU 是较慢的阶段。
- **在 Arm 主机上,CPU 部分是瓶颈。** 以下是本节中相同的 DXNN 文件在配备 DX-M1 的 Raspberry Pi 5 上的测量结果(`dxrun -m <model> --use-ort -t 10`,默认设置):

  | YOLO26-n | 基线 FPS | TopK 300 FPS | 加速比 |
  |---|---:|---:|---:|
  | det | 100.79 | 179.54 | 1.78× |
  | seg | 64.13 | 95.61 | 1.49× |
  | pose | 142.48 | 265.44 | 1.86× |

  对于三个基线模型,DX-RT 还打印了 `CPU TASK [cpu_0] Inference Worker - Average Input Queue Load : ...% (DXRT_DYNAMIC_CPU_THREAD: OFF) - To improve FPS, set: 'export DXRT_DYNAMIC_CPU_THREAD=ON'`。该消息表示有帧在等待 CPU 任务。TopK 模型没有出现该消息。如果在您的开发板上看到它,说明 CPU 部分是瓶颈,这项优化很可能有帮助。`DXRT_DYNAMIC_CPU_THREAD=ON` 是另一项独立的设置,它为 CPU 任务分配更多线程;上述数据未使用该设置。
- **DX-COM 手册中的参考结果**,测试环境为配备 ARM Cortex-A53 主机的 DX-M1(`DXRT_DYNAMIC_CPU_THREAD=ON dxrun -m <model> -l 1000 --use-ort`):det 184.89 → 315.73 FPS(1.7×),seg 130.82 → 195.05 FPS(1.5×),pose 242.77 → 304.38 FPS(1.3×)。

若要为您自己的产品做决策,请直接在目标开发板上运行本节。

<!-- cell: dx-app-heading src: a0178a7c3f -->
### 6.8 在 DX-APP 中运行 TopK DXNN

应用程序代码无需修改。DX-APP 的 YOLO26 后处理器按位置读取第一个输出,而 DX-STREAM 的 `yolo26_od`、`yolo26_seg` 和 `yolo26_pose` 后处理库按形状选择输出(`[1, 300, 6]`、至少 38 列、至少 57 列)。因此,输出改名(`output0` → `preopt_output`)无需任何修改。两者仍然需要 ONNX Runtime 来执行 CPU 任务。

下一个单元格针对该模型运行 DX-APP Python 示例,分别用两个 DXNN 处理同一张图像,并显示两个结果。它使用教程 01 中的 DX-RT Python 环境和教程 02 中的 DX-APP 源码;如果缺少它们,该单元格会跳过。对于默认的 `n` 变体,它等价于:

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_app
<DX_ALL_SUITE_DIR>/dx-runtime/venv-dx-runtime/bin/python \
  src/python_example/object_detection/yolo26/yolo26-n_640x640/yolo26-n_640x640_sync.py \
  --model <workspace>/outputs/yolo26n_topk300/yolo26-n_640x640_topk300.dxnn \
  --image sample/img/sample_people.jpg --no-display --save --save-dir <workspace>/outputs/dx_app_results
```

在本笔记本的测试机器上,两个 DXNN 文件检测到的目标相同;由于每个模型是分别量化的,边界框角点最多相差约 2 个像素,得分最多相差约 0.05。部署前请在验证集上检查精度,例如使用 DX Model Zoo 的评估工具。

<!-- cell: seg-pose-heading src: 61a29534c1 -->
### 6.9 分割与姿态:只改 `task` 的同一个 pass

DX-COM v2.5.0 将 `yolo_no_dfl_postprocess` 扩展到了 YOLO26 分割和姿态模型。pass 名称、`topk` 以及 `bbox` 和 `cls_conf` 张量都与 6.3 相同。变化只有两处:`task` 键,以及每个尺度上多出的一个检测头张量。

| `task` | Model Zoo 模型(`n`) | 各尺度的检测头张量 | `num_classes` | 该 pass 处理后的输出 |
|---|---|---|---:|---|
| `base` | `yolo26-n_640x640` | `bbox`(4 ch)、`cls_conf`(80 ch) | 80 | `[1, 300, 6]` |
| `seg` | `yolo26-n-seg_640x640` | + `mask_coeff`(32 ch),`one2one_cv4.<scale>` | 80 | `[1, 300, 38]` 以及原型 `[1, 32, 160, 160]` |
| `pose` | `yolo26-n-pose_640x640` | + `kpt`(51 ch),`one2one_cv4_kpts.<scale>`,`kpt_shape` 为 `[17, 3]` | 1 | `[1, 300, 57]` |

- **分割。** 保留下来的 300 个候选框各自带有 32 个掩码系数。掩码原型输出不会被修改,原样传出;将系数与原型组合成掩码的工作仍由主机完成。
- **姿态。** 关键点检测头只有一个 Conv,因此其张量名称(`/model.23/one2one_cv4_kpts.<scale>/Conv_output_0`)不符合其他分支的 `...<scale>.2/Conv_output_0` 模式。6.2 中的规则不依赖于名称,因此 `yolo26_heads.py` 仍然能找到它。`kpt_shape` 来自 ONNX 元数据,且 17 x 3 必须等于关键点 Conv 的 51 个通道。
- **校验。** `pre_optimize` 在修改图之前,会按照 `task` 检查检测头张量:`seg` 必须提供 `mask_coeff`,`pose` 必须提供 `kpt` 和 `kpt_shape`,且各自拒绝对方的张量。但它无法发现以 `task="base"` 运行分割模型的情况:这样可以编译成功,但掩码会丢失。因此 `yolo26_heads.py` 会把 `task` 与模型元数据进行比较。

接下来的单元格对这两个模型重复 6.1 至 6.8 的步骤,每个单元格对应一个步骤。由于 Model Zoo 为每种尺寸都发布了 `seg` 和 `pose` 模型,这些单元格同样遵循 `YOLO26_VARIANT`。

<!-- cell: seg-pose-prepare-heading src: fcd279e22c -->
#### 6.9.1 下载模型并找到检测头张量

<!-- cell: seg-pose-apply-heading src: 1c5e139d9d -->
#### 6.9.2 应用该 pass 并比较 CPU 部分

对于默认的 `n` 变体,优化步骤等价于:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
python optimize_yolo26.py models/yolo26-n-seg_640x640.onnx models/yolo26-n-seg_640x640_topk300.onnx --task seg
python optimize_yolo26.py models/yolo26-n-pose_640x640.onnx models/yolo26-n-pose_640x640_topk300.onnx --task pose
```

此处 `--task` 是可选的,因为脚本会从元数据中读取它。显式传入它可以表明您的预期,如果模型与之不符,脚本会停止。在姿态模型的表格中,还要关注关键点的 `Sigmoid`:它从 `[1, 17, 1, 8400]` 变为 `[1, 300, 17, 1]`。

<!-- cell: seg-pose-verify-heading src: 7657781fd1 -->
#### 6.9.3 检查结果是否完全相同

对于分割,比较涵盖掩码系数和原型;对于姿态,比较涵盖关键点。图像中绘制的掩码和关键点都来自各个模型自身的输出。

<!-- cell: seg-pose-compile-heading src: e98f1b66b9 -->
#### 6.9.4 编译基线模型和 TopK 模型

该单元格为每个模型写入本地 JSON(与 6.6.1 相同),并使用与 6.6 相同的选项编译四个模型。对于默认的 `n` 变体,它等价于以下命令(姿态模型请将 `seg` 替换为 `pose`):

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/yolo26-n-seg_640x640.onnx \
      -c configs/yolo26-n-seg_640x640.local.json \
      -o outputs/yolo26n_seg_baseline \
      --gen_log \
      --export_html
dxcom -m models/yolo26-n-seg_640x640_topk300.onnx \
      -c configs/yolo26-n-seg_640x640.local.json \
      -o outputs/yolo26n_seg_topk300 \
      --gen_log \
      --export_html
```

如果输出目录中已经有 DXNN 文件,该单元格会报告该文件并跳过对应模型的 `dxcom`。

<!-- cell: seg-pose-benchmark-heading src: 083b768e06 -->
#### 6.9.5 测量检测、分割和姿态

该单元格按照 6.7 的方式测量四个新 DXNN 文件的 CPU 部分和 `dxrun` FPS,并把 6.7 中的检测结果一并加入表格。

<!-- cell: seg-pose-results src: 967a128494 -->
三个 task 的比较:

- **检测和分割**在同一处节省 CPU 时间:对 80 个类别 x 8,400 个候选框执行的类别得分 `Sigmoid`。分割还保留了掩码原型输出(该 pass 不会修改它),而且主机仍需构建掩码,因此其相对收益较小。
- **姿态**只有一个类别,因此其类别得分的计算量很小。它的节省来自关键点可见性 `Sigmoid`(17 x 8,400 个值)和关键点解码,这两者现在只对 300 个候选框运行。其影响有多大取决于主机:在上面的 Raspberry Pi 5 上,姿态模型的收益最大(1.86×),而在手册的 Cortex-A53 结果中,它的收益最小(1.3×)。

<!-- cell: seg-pose-dx-app-heading src: 42492b118d -->
#### 6.9.6 在 DX-APP 中运行分割和姿态 DXNN 文件

<!-- cell: 6be8d246-d16c-429d-a82f-e1c45594974c src: 8bd09629d5 -->
## 7. 故障排除

| 症状 | 看到的内容 | 可能原因 | 解决方法 |
|---|---|---|---|
| 找不到 PPU 节点名 | `ValueError: ... not found in the ONNX graph` | PPU 配置中的 `layer` 名称属于另一个导出模型 | 在 Netron 中检查 ONNX 并更新 JSON 中的节点名 |
| `dxrun` 找不到设备 | `No device found` 或 `Fail to initialize device` | 本主机上没有 DEEPX NPU | 在配有 NPU 的主机上运行比较;仅编译的步骤仍可正常工作 |
| 无法读取 FPS | `Could not read FPS from dxrun output` | `dxrun` 的输出格式已更改 | 在上方打印完整输出并手动读取 FPS 行 |
| 修改 JSON 没有效果 | `Skip compilation: found ...` | 输出目录中已经存在 DXNN | 删除 `workspace/outputs/<dir>` 并重新运行 |
| `pre_optimize` 拒绝检测头张量 | `ValueError: [yolo_no_dfl_postprocess] task='seg' requires 'mask_coeff' on every layer` 或 `Cannot find shape for mask_coeff=...` | `task` 和检测头张量与模型不符 | 在这个确切的 ONNX 文件上用 `find_yolo26_heads()` 查找张量 |
| head finder在 task 检查处停止 | `The model metadata says task='seg', not 'base'` | task 参数与模型不符 | 传入消息中给出的 task;若使用 `base`,分割模型会在没有掩码的情况下完成编译 |
| head finder报告 one-to-many 检测头 | `This ONNX file has the one-to-many head (output [1, 4 + classes, 8400])` | 导出模型时没有使用 `nms=False` | 使用 `yolo export ... format=onnx nms=False` 重新导出 |
| head finder找不到 `cls_conf` Conv | `expected one cls_conf head Conv with N channels, found none` | 类别数错误,或元数据中缺少类别数 | 传入 `num_classes`(脚本使用 `--num-classes`) |
| head finder报告动态输入 | `The input height is dynamic` | 导出模型时使用了 `dynamic=True` | 使用固定的 `imgsz` 重新导出 |
| TopK DXNN 在 `dxrun` 中没有变快 | 相对吞吐量接近 1.00× | 限制流水线的是 NPU 而不是主机 CPU(在 x86_64 上很常见),或者模型的类别数很少 | 比较 6.7 中的 CPU 部分耗时,并在目标开发板上测量 |
| JSON 中的 `pre_optimize` 没有效果 | `falling back to legacy: ... 'pre_optimize' and 'ppu' are mutually exclusive` | JSON 中还有 `ppu` 块 | 从 JSON 中删除 `ppu` |
| DX-APP 对具有其他类别的模型显示 COCO 标签 | 您的目标上显示的是 COCO 类别名称 | 标签来自示例自身的 `config.json`;`--config` 不会改变它们 | 使用 `scripts/extract_model_package.sh --no-prune` 导出独立包(教程 02),并在其 `config.json` 中设置 `num_classes` 和 `visualizer.kwargs.custom_labels` |
| 下载不完整 | `workspace/models` 中出现极小或零字节的文件 | 下载被中断 | 删除该文件;下载辅助函数会在使用前验证大小 |

<!-- cell: summary src: a00cdd65b0 -->
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

YOLO26 det / seg / pose head
          │
          ▼
 Find head tensors → TopK-first ONNX rewrite
 (task = base / seg / pose)
          │
          ▼
 Same results, less host-CPU work
 per frame
```

### 8.2 三条优化路径

| 主题 | 使用的模型 | 主要决策 | 优化位置 | 主机 CPU 的职责 |
|---|---|---|---|---|
| PPU Type 0 | YOLOv7 | 选择基于锚框的检测头并映射其 Conv 节点 | 硬件 PPU | NMS 和应用逻辑 |
| PPU Type 1 | YOLOX-S | 映射 bbox、objectness 和类别分支 | 硬件 PPU | NMS 和应用逻辑 |
| TopK 优先优化 | YOLO26 检测、分割和姿态(`YOLO26_VARIANT`,默认 n) | 把 TopK 移到 Sigmoid 和解码之前;设置 `task` 和检测头张量 | 重写后的 ONNX 图(Python API 或编译 JSON) | 解码更少的候选框(及其掩码系数或关键点) |

### 8.3 已完成的实验

| 实验 | 比较的模型 | 控制变量 | 验证方式 |
|---|---|---|---|
| Type 0 PPU | YOLOv7 非 PPU 与 PPU | 硬件后处理 | `dxparse`、`dxrun` |
| Type 1 PPU | YOLOX-S 非 PPU 与 PPU | 硬件后处理 | `dxparse`、`dxrun` |
| TopK 优化 | YOLO26 检测、分割、姿态:基线与 TopK 300(所选尺寸) | ONNX 图结构 | ONNX Runtime 输出比较、CPU 部分耗时、`dxparse`、`dxrun`、DX-APP |

### 8.4 完成清单

- [ ] 将校准数据和预处理视为模型定义的一部分
- [ ] 识别 NPU、PPU 与主机 CPU 处理之间的边界
- [ ] 验证 YOLOv7 Type 0 的 Conv 节点映射
- [ ] 编译并检查 Type 0 PPU 模型
- [ ] 验证 YOLOX-S Type 1 的 bbox、objectness 和类别映射
- [ ] 编译并检查 Type 1 PPU 模型
- [ ] 在相同的基准测试条件下比较 PPU 与非 PPU 模型
- [ ] 说明 YOLO26 DXNN 的哪一部分在主机 CPU 上运行
- [ ] 使用 `find_yolo26_heads()` 查找 YOLO26 检测头张量,而不是复制名称
- [ ] 对 YOLO26 模型应用 TopK 优先优化(并尝试第二种尺寸)
- [ ] 使用 ONNX Runtime 确认基线模型与 TopK 模型给出完全相同的输出
- [ ] 测量 CPU 部分耗时和 `dxrun` FPS,并解释二者为何不同
- [ ] 在不修改应用程序的情况下,在 DX-APP 中运行 TopK DXNN
- [ ] 使用 `task` = `seg` 和 `pose` 将同一个 pass 应用于 YOLO26 分割和姿态模型
- [ ] 使用真实数据验证精度、主机 CPU 占用率和完整流水线延迟

### 8.5 如何选择

| 如果您的模型具有... | 从...开始 |
|---|---|
| 基于锚框的 YOLO 检测头 | PPU Type 0 |
| 独立的 bbox、objectness 和类别分支 | PPU Type 1 |
| 大量候选框后接 TopK,且所在主机上 CPU 部分限制了 FPS | TopK 优先的图优化 |
| 分割或姿态检测头(YOLO26 或 YOLOv8 至 v13) | 使用 `task` = `seg` 或 `pose` 的 TopK 优先图优化 |
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
