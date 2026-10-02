<!-- i18n source: notebooks/T06-DX-Runtime/dx_rt_01_beginner.ipynb -->
<!-- i18n lang: zh -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 56ed92cd src: dfd7b6b490 -->
# DEEPX Tutorial 06-1 - DX-RT 入门

本笔记本介绍在 DEEPX NPU 上执行编译后 DXNN 模型的运行时层。

## 学习目标

完成本教程后,您将能够:

- 说明 DX-RT 负责什么,以及哪些仍属于应用程序的职责,
- 验证已安装的运行时工具和 NPU 状态,
- 检查 DXNN 模型的输入、输出和任务图,
- 区分单次运行、最大吞吐量和目标 FPS 三种模式,
- 使用 `dxrun` 运行受控的虚拟输入基准测试,
- 对包含 CPU 任务的模型比较仅 NPU 与启用 ORT 两种执行方式,以及
- 为检查、推理和监控选择正确的 CLI 工具。

本教程不会修改 SDK 源码树。所有教程产物和日志都保存在 `<dx-tutorials>/notebooks/T06-DX-Runtime/workspace` 下。

<!-- cell: a6368870 src: a2bad27896 -->
## 课程地图

| 笔记本 | 主要内容 |
|---|---|
| 入门 | DX-RT 的角色、设备检查、DXNN 检查和 CLI 推理 |
| 中级 | Python 和 C++ API、同步/异步执行、批处理和缓冲区 |
| 高级 | 资源绑定、性能分析、监控、多输入/内存加载以及发布验证 |

除非您已经理解 DXNN 模型契约和 DX-RT CLI,否则请按顺序完成这些笔记本。

<!-- cell: ffd44045-7886-415c-b0c5-017237e17a82 src: 427df03b3b -->
## 前提条件

- 已安装 DX-RT(教程 01 第 3 节),且 DEEPX NPU 以 `/dev/dxrt0` 的形式可见。
- ResNet50 示例模型;如果缺失,设置单元格会用 `dx_app/setup.sh` 下载它。
- 第 6 节使用的 YOLO26-S 示例模型(21 MB);如果缺失,该节会以同样的方式下载。
- 不需要 `sudo`。预计耗时约 10 分钟。

下一个单元格会定位 SDK、检查这些要求并打印状态表。任何标记为 `MISSING` 的项目都会附上提供它的步骤。

<!-- cell: 51bdb967 src: a53c9cd719 -->
## 1. DX-RT 所处的位置

DX-COM 生成 `.dxnn` 模型。DX-RT 加载该模型、管理推理缓冲区和作业、与设备驱动通信,并返回输出张量。

<img src="assets/dx-rt-runtime-workflow.svg" style="max-width: 1100px; width: 100%;" alt="DX-RT 推理工作流">

| DX-RT 负责 | 您的应用程序负责 |
|---|---|
| DXNN 加载与验证 | 输入采集 |
| 设备选择与 NPU 调度 | 模型相关的预处理 |
| 运行时输入/输出缓冲区 | 模型相关的后处理 |
| 同步与异步作业 | 产品行为与可视化 |
| 运行时性能分析与设备查询 | 端到端精度与服务级指标 |

> 一次成功的运行时调用只能证明模型已执行。它不能证明预处理、解码或应用程序精度是正确的。

<!-- cell: e3eb37dc src: f7cdc491e8 -->
## 2. 初始化教程工作区

设置单元格从 `config.json` 读取共享的 SDK 位置。它只在本教程内部创建本地链接和目录:

```text
T06-DX-Runtime/
├── assets/
├── dx_rt_01_beginner.ipynb
├── dx_rt_02_intermediate.ipynb
├── dx_rt_03_advanced.ipynb
└── workspace/
    ├── models/       # links to SDK models
    ├── reports/      # dxparse and benchmark reports
    ├── profiler/     # profiler JSON and visualizations
    ├── cpp/          # generated C++ examples and build files
    ├── multi_input/  # link to the two-input model from Tutorial 05-3 (Advanced)
    └── .venv-dxrt/   # DX-RT Python binding environment (created in Intermediate section 2)
```

本教程使用由 DX-APP 资源设置下载的 `resnet50_224x224.dxnn`。如果它缺失,设置单元格会打印获取它所需的准确终端命令。

<!-- cell: 4ac70adf src: d712a2cdc0 -->
### 2.1 在笔记本之外使用的命令

上面打印的模型下载命令是一条普通的 shell 命令:

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_app
bash setup.sh --models resnet50 --no-force
```

`--no-force` 会保留已经下载的资源。随后教程会在自己的 `workspace/models` 目录下创建一个符号链接;它不会复制或修改 SDK 模型。

<!-- cell: 02e64674 src: 8ca79cf862 -->
## 3. 验证运行时安装

以下命令与您在终端中输入的命令完全相同。它们不会创建输出文件。

<!-- cell: e1dcae80 src: 07bed7df89 -->
### 3.1 检查设备

除非显式选择了某个设备,否则 `dxcli --status` 会查询所有可用的加速器。

```bash
dxcli --status
dxcli -s          # short form of the same command
```

正常的结果应至少识别出一个设备。如果没有出现任何设备,请在此停下,先检查驱动、固件、物理连接和安装状态,再测试模型。

<!-- cell: ca70321e src: 51e030d707 -->
## 4. 检查 DXNN 模型契约

运行时应用程序必须在以下所有项目上与编译后的模型保持一致:

| 契约项目 | 重要性 |
|---|---|
| 输入张量名称与数量 | 多输入模型需要正确的映射 |
| 形状 | 应用程序必须分配所需数量的元素 |
| 数据类型 | dtype 不匹配可能导致错误或无效数据 |
| 布局 | NHWC 和 NCHW 以不同顺序存储相同的值 |
| 预处理边界 | 部分预处理可能已经编译进 DXNN |
| 输出张量 | 后处理必须使用正确的名称、形状和顺序 |
| CPU 任务 | 它们需要启用 ORT 的运行时路径 |

`dxparse` 无需运行推理即可读取此契约。`-v` 会额外显示任务依赖关系和内存信息。标准的 shell 重定向会保存报告供下一个单元格使用。`dxparse` 即使写入文件时也会用 ANSI 转义码为输出着色,因此 `sed` 过滤器会去掉这些转义码,使保存的报告保持为纯文本。

等效的终端命令:

```bash
dxparse -m <T06-DX-Runtime>/workspace/models/resnet50_224x224.dxnn -v \
  | sed 's/\x1b\[[0-9;]*m//g' \
  > <T06-DX-Runtime>/workspace/reports/resnet50_dxparse.txt
```

<!-- cell: a6c68f9c src: 9918c774c5 -->
### 4.1 查看保存的契约

下一个单元格读取在教程工作区内创建的报告。请关注模型版本、编译器版本、输入/输出张量、任务类型和内存大小。

<!-- cell: a5f4ec9d src: c5ebfa0c53 -->
## 5. 理解 `dxrun` 的模式

| 模式 | 选项 | 主要行为 | 适用场景 |
|---|---|---|---|
| 单次 | `--single` | 在一个核心上顺序执行单输入推理 | 基本执行与延迟检查 |
| 基准测试 | `--benchmark` | 让可用的运行时流水线保持忙碌 | 最大吞吐量比较 |
| 目标 FPS | `--fps N` | 以请求的速率提交工作 | 在产品负载下进行容量和稳定性检查 |

`--time` 控制测试时长并覆盖 `--loops`。`--warmup-runs` 将初始预热运行排除在测量之外。比较结果时请使用相同的模型、选项、时长和系统状态。

<!-- cell: a11603a4 src: 88e192a8dc -->
### 5.1 运行一次请求

等效的终端命令:

```bash
cd <T06-DX-Runtime>/workspace
dxrun -m models/resnet50_224x224.dxnn --single --loops 1 --verbose
```

<!-- cell: a0ae40f8 src: 126816c0da -->
### 5.2 测量最大吞吐量

此基准测试使用虚拟输入。它适合测量运行时性能,但不适合测量分类精度。

等效的终端命令:

```bash
cd <T06-DX-Runtime>/workspace
dxrun -m models/resnet50_224x224.dxnn \
      --benchmark \
      --time 5 \
      --warmup-runs 5 \
      --buffer-count 6
```

<!-- cell: d6f353b5 src: 21dc67134a -->
### 5.3 测试目标工作负载

目标 FPS 运行回答的是系统能否维持请求的到达速率。它不同于最大吞吐量基准测试。

等效的终端命令:

```bash
cd <T06-DX-Runtime>/workspace
dxrun -m models/resnet50_224x224.dxnn --fps 30 --time 5 --warmup-runs 5
```

<!-- cell: c3631420 src: 393fccae2d -->
## 6. CPU 任务与 `--use-ort`

DXNN 图可以同时包含 NPU 任务和 CPU 任务。`--use-ort` 为不在 NPU 上执行的 CPU 侧子图启用 ONNX Runtime。

- 不要盲目添加 `--use-ort`。
- 先用 `dxparse -v` 检查任务图。
- 在同一组比较中,所有结果使用相同的 ORT 设置。
- 如果运行时在构建时未启用 ORT 支持,则无法执行 CPU 任务。

目前使用的 ResNet50 模型是仅 NPU 的图,因此该选项对它没有任何影响。Model Zoo 中的 YOLO26-S 则不同:它的检测头以一个 CPU 任务(`npu_0 -> cpu_0`)结尾,该任务将六个原始检测头张量解码为最终的 `[1, 300, 6]` 检测结果。接下来的三个单元格会在模型缺失时下载它、显示其任务图,并运行两次 `dxrun`:先仅 NPU,再加上 `--use-ort`。

等效的终端命令:

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_app
bash setup.sh --models yolo26s --no-force        # only when the model is missing

cd <T06-DX-Runtime>/workspace
dxparse -m models/yolo26-s_640x640.dxnn -v
dxrun -m models/yolo26-s_640x640.dxnn --benchmark --time 5 --warmup-runs 5
dxrun -m models/yolo26-s_640x640.dxnn --benchmark --time 5 --warmup-runs 5 --use-ort
```

<!-- cell: a8230861 src: 9c5259ac39 -->
**观察要点**

| | 仅 NPU | `--use-ort` |
|---|---|---|
| `dxrun` 列出的任务 | `Task[0] npu_0` | `Task[0] npu_0` 和 `Task[1] cpu_0` |
| 输出 | 六个原始检测头张量(`.../Conv_output_0`) | `output0 [1, 300, 6]`: 解码后的检测结果 |
| FPS | 纯 NPU 吞吐量 | 包含 CPU 解码的端到端吞吐量 |

在高速桌面 CPU 上,两个 FPS 值通常相差不过几个百分点,因为 DX-RT 以流水线方式运行 CPU 任务,同时 NPU 已在处理下一个输入。在性能较弱的主机 CPU 上,FPS 差距会变大,此时 CPU 任务成为瓶颈。始终不同的是输出:不使用 `--use-ort` 时,应用程序必须自行解码这六个原始张量。对于带有 CPU 任务的检测模型,应与应用程序实测吞吐量进行比较的是启用 ORT 的数值。

<!-- cell: 228a7f19 src: 10632970eb -->
## 7. CLI 名称与向后兼容

当前教程使用新的命令名称。旧脚本通过兼容别名继续工作。

| 当前命令 | 旧别名 | 用途 |
|---|---|---|
| `dxparse` | `parse_model` | 检查 DXNN 模型 |
| `dxrun` | `run_model` | 执行模型或进行基准测试 |
| `dxcli` | `dxrt-cli` | 查询运行时设备接口 |

在新代码中请使用当前名称,以保持日志和文档的一致性。

<!-- cell: 69b2a38e src: 351f9dcc85 -->
## 8. 实时监控

`dxtop` 是交互式的,会持续重绘终端。请在单独的 JupyterLab 终端中运行它,而不是在笔记本单元格中:

```bash
dxtop
```

使用 **File → New → Terminal**,然后运行该命令。按 `q` 退出。在另一个终端运行 `dxrun` 的同时,监控利用率、NPU 内存、温度、电压和时钟。

<!-- cell: 5e501953 src: bbfc3360e6 -->
## 9. 故障排除

| 症状 | 看到的内容 | 可能原因 | 解决方法 |
|---|---|---|---|
| 找不到命令 | `dxrun: command not found` | 未安装 DX-RT,或 `/usr/local/bin` 不在 `PATH` 中 | 完成教程 01 第 3 节并打开新终端 |
| 没有设备 | `dxcli --status` 中显示 `No device found` | 驱动、固件或物理连接 | 检查 `lsmod | grep dxrt`、重启或将主机断电再开机 |
| 模型被拒绝 | `dxrun` 报告版本或格式错误 | DXNN 由不兼容的 DX-COM 编译 | 比较 `dxparse` 打印的版本与已安装的运行时;必要时重新编译 |
| CPU 任务错误 | 带有 CPU 任务的模型出现 ORT 错误 | 运行时在构建时未包含 ONNX Runtime | 使用启用 ORT 的构建并传入 `--use-ort` |
| 性能异常 | 各次运行之间 FPS 波动 | 没有预热、热节流或存在其他负载 | 使用 `--warmup-runs`、固定时长、相同选项和空闲的主机 |

<!-- cell: 3578de13 src: 817b273196 -->
## 10. 总结

### 10.1 已完成的运行时工作流

**验证工具和设备**  
→ **检查 DXNN 契约**  
→ **运行一次推理**  
→ **测量最大吞吐量**  
→ **测试目标工作负载**  
→ **比较仅 NPU 与启用 ORT 的执行**

<img src="assets/dx-rt-runtime-workflow.svg" style="max-width: 1000px; width: 100%;" alt="DX-RT 推理工作流">

### 10.2 工具一览

| 问题 | 工具或选项 | 证据 |
|---|---|---|
| NPU 是否可见? | `dxcli --status` | 设备状态 |
| 模型期望什么输入? | `dxparse -v` | 张量与任务契约 |
| 能否运行一次请求? | `dxrun --single` | 基本执行与延迟 |
| 最大吞吐量是多少? | `dxrun --benchmark` | 受控虚拟输入 FPS |
| 能否维持 30 FPS? | `dxrun --fps 30` | 目标负载下的行为 |
| 模型是否需要 CPU? | `dxparse -v`、`dxrun --use-ort` | 任务图与解码后的输出 |
| 随时间变化的情况如何? | `dxtop` | 设备实时状态 |

### 10.3 完成清单

- [ ] 识别 DX-RT 与应用程序之间的边界
- [ ] 验证已安装的 CLI 工具
- [ ] 查询 NPU 状态
- [ ] 保存并查看详细的 DXNN 报告
- [ ] 运行单次、基准测试和目标 FPS 模式
- [ ] 对带有 CPU 任务的模型比较仅 NPU 与 `--use-ort` 运行
- [ ] 区分运行时性能与模型精度
- [ ] 使用真实应用数据验证预处理、后处理和精度

> **请记住:** 比较性能时,请保持模型、命令、运行时版本、设备状态和工作负载完全一致。

### 10.4 下一步

继续学习**中级教程**,通过 Python 和 C++ API 实现同样的运行时概念,然后比较同步、异步、批处理和缓冲区管理的不同选择。
