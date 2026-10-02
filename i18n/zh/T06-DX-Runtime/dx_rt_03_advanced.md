<!-- i18n source: notebooks/T06-DX-Runtime/dx_rt_03_advanced.ipynb -->
<!-- i18n lang: zh -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 46c94efe src: bf00e17c30 -->
# DEEPX Tutorial 06-3 - DX-RT 高级

本笔记本将一次可正常工作的推理调用变成一个可观测、可复现的运行时实验。

## 学习目标

完成本教程后,您将能够:

- 配置设备选择、NPU 核心绑定、ORT 使用方式以及缓冲区数量,
- 通过受控基准测试调优缓冲区数量,
- 生成并可视化 DX-RT 性能分析器跟踪,
- 读取每个任务的 H2D、NPU、D2H、格式处理器和 CPU 任务指标,
- 使用变异系数(Coefficient of Variation,CoV)检测不稳定的计时,
- 查询设备内存、利用率和热状态,
- 从内存缓冲区加载 DXNN 模型,
- 为多输入模型准备命名输入,
- 注册运行时事件处理器,
- 解释稳定的 C ABI、仅头文件的 C++ 封装以及运行时 IPC 边界,
- 解释、构建、启用并通过 C++、Python 和 `dxrun` 对 NFH 与 CPU 算子加速进行 A/B 测试,以及
- 创建一份精简的发布验证记录。

生成的所有跟踪、报告、源文件和结果都保存在 <code>&lt;dx-tutorials&gt;/notebooks/T06-DX-Runtime/workspace</code> 下。

<!-- cell: 7bedacd4 src: 215e801d3c -->
> 本教程共 3 部分,这是第 3 部分。课程地图和完整的部分列表见入门笔记本。

<!-- cell: ef4ee62f-6a13-4f7a-bc9d-be05af55b919 src: 5ddb64a396 -->
## 前提条件

- 已完成教程 06-2,特别是第 2 节(此处会复用 `workspace/.venv-dxrt` 环境)。
- 一块 DEEPX NPU;性能分析器和基准测试章节会运行真实推理。
- 可选: 教程 05-3 第 8 节生成的多输入模型。该模型不存在时,对应实验会被跳过。
- 阅读时间约 15 分钟。单元格本身在一到两分钟内即可运行完成。

下一个单元格会定位 SDK、检查上述要求并打印状态表。任何标记为 `MISSING` 的项目都会附带提供它的步骤。

<!-- cell: 4c12f0dc src: be70bbc8b4 -->
## 1. 初始化教程工作区

本笔记本要求中级教程创建的 T06 本地环境中已安装匹配的 <code>dx_engine</code> wheel。在可用时它还会链接第二个模型,以便 <code>dxbenchmark</code> 演示多模型报告。

<!-- cell: b8287091 src: 431918d739 -->
## 2. 运行时资源控制

<code>InferenceOption</code> 控制一个引擎实例在哪里运行以及如何运行。

| 选项 | 默认含义 | 何时修改 |
|---|---|---|
| <code>devices</code> | 空列表表示使用所有可用设备 | 隔离某个设备或在多个设备间分配引擎 |
| <code>bound_option</code> | <code>NPU_ALL</code> | 预留一个核心或一对核心 |
| <code>use_ort</code> | 取决于构建 | 模型包含 CPU 任务 |
| <code>buffer_count</code> | 运行时默认值,通常为 6 | 实测的排队或内存行为需要调优 |

设备选择的是一个物理加速器。绑定选项选择的是每个选定设备内部的核心。不要混淆这两个层级。

<!-- cell: 01c44f7b src: b4e173b1c2 -->
### 2.1 绑定安全

如果已有的引擎实例已经预留了单独的 NPU 核心,再以 <code>NPU_ALL</code> 创建另一个引擎可能会阻塞,直到所有所需核心都被释放。在多线程和多进程服务中,请保持资源归属明确。

| 目标 | 典型起点 |
|---|---|
| 单个引擎的最大吞吐量 | 一个或多个设备,<code>NPU_ALL</code> |
| 隔离两条独立的流水线 | 分配不同的设备或互不重叠的核心绑定 |
| 复现单核心延迟测试 | 一个设备和一个核心 |
| 跨进程共享硬件 | 在创建引擎之前定义归属 |

绑定是一个部署决策。本笔记本中实际执行的实验都在设备 0 上以 <code>NPU_ALL</code> 运行。

<!-- cell: 0391fabe src: 9280b3bc19 -->
## 3. 通过受控实验调优缓冲区数量

<img src="assets/dx-rt-observability.svg" style="max-width: 1100px; width: 100%;" alt="DX-RT 可观测性与调优流程">

下面四条命令仅在 <code>--buffer-count</code> 上有所不同。它们使用相同的模型、预热次数、持续时间和执行模式。请比较吞吐量和延迟;不要机械地选择最大的数值。

等效的终端命令模式:

~~~bash
cd <T06-DX-Runtime>/workspace
dxrun -m models/resnet50_224x224.dxnn \
      --benchmark --time 3 --warmup-runs 5 --buffer-count N
~~~

<!-- cell: 2177a506 src: 4687dde300 -->
选择在不违反内存或延迟限制的前提下,能达到所需稳定吞吐量的最小值。请在真实的应用负载下重复该实验,因为预处理、后处理和其他进程都会改变队列压力。

<!-- cell: 9c4ffa77 src: 63add5ff96 -->
## 4. 分析运行时时间线

性能分析器将时间划分为若干阶段,例如输入格式化、主机到设备传输、NPU 计算、设备到主机传输、输出格式化以及 CPU 任务。

以下命令在 T06 的 profiler 目录中生成 <code>profiler.json</code>:

~~~bash
cd <T06-DX-Runtime>/workspace/profiler
dxrun -m ../models/resnet50_224x224.dxnn \
      --benchmark --time 5 --warmup-runs 5 --profiler
~~~

<!-- cell: fb34227e src: 4811beb9fe -->
### 4.1 将跟踪转换为图像

SDK 的绘图命令直接显示在下方。<code>--auto-select</code> 聚焦于一段稳定的中心区域,输出仍保留在 T06 下。

等效的终端命令:

~~~bash
python <DX_RT_DIR>/tool/profiler/plot.py \
  --input <T06>/workspace/profiler/profiler.json \
  --output <T06>/workspace/profiler/profiler.png \
  --auto-select
~~~

<!-- cell: 0dff6899 src: 9e9a10036d -->
### 4.2 解读性能分析器事件

| 事件 | 测量内容 | 首先要问的问题 |
|---|---|---|
| Buffer Wait | 等待空闲的推理缓冲区 | 队列深度或 CPU 压力是否过高? |
| NPU Input Format Handler | 填充与布局转换 | 格式转换是否占了显著比例? |
| PCIe Write / H2D | 主机到设备传输 | 输入大小或传输是否是瓶颈? |
| NPU Core | NPU 计算 | 模型是否主导了总时间? |
| PCIe Read / D2H | 设备到主机传输 | 输出是否异常大? |
| NPU Output Format Handler | 输出切片与布局转换 | 转换是否是瓶颈? |
| CPU Task Queue Wait | 等待 CPU 执行 | CPU 流水线是否已饱和? |
| cpu_N | CPU 算子执行 | 计算密集型 CPU 算子是否占主导? |

NPU 任务总时间涵盖格式处理、传输、计算和输出处理。它比单纯的 NPU 核心计算范围更广。

<!-- cell: 100ed5cc src: 4831be0e8a -->
## 5. 通过 Python API 读取每个任务的指标

DX-RT v3.4 新增了 <code>get_job_metrics(job_id)</code>。请在 <code>wait(job_id)</code> 之后立即调用它。旧的性能数据方法已被弃用。

下面的代码启用性能分析,运行 20 个异步任务,并打印最后一个任务的有效指标。重复运行的任务也为 5.1 节的 CoV 表提供了足够的样本。

<!-- cell: 381e6586 src: 6eb64f8da6 -->
### 5.1 用变异系数衡量稳定性

变异系数(CoV)等于标准差除以均值:

~~~text
CoV (%) = standard deviation / mean × 100
~~~

它用于比较平均耗时不同的各阶段之间的相对抖动。数值越低越稳定,但不存在通用的发布阈值。请根据产品的延迟预算和运行条件自行定义。

| 结果 | 解读 |
|---|---|
| 均值低,CoV 高 | 通常很快,但偶尔不稳定 |
| 均值高,CoV 低 | 可预测地慢 |
| p99 高,均值中等 | 需要调查尾延迟 |
| NPU 稳定,主机时间不稳定 | 检查主机队列、CPU 负载和预处理 |

<!-- cell: 9994a301 src: 646fb2cbbd -->
## 6. 监控设备健康状态

监控服务大约每秒更新一次共享状态。更快地轮询通常只会得到同一个样本。

下一个单元格记录两次快照。请将 <code>is_valid() == False</code> 视为过期或不可用的监控数据。

<!-- cell: c0c43234 src: 68ad41aa24 -->
若需持续检查,请在应用运行期间在单独的终端中运行 <code>dxtop</code>。将温度、利用率、内存和降频证据与延迟、吞吐量一起收集;短时间的冷机基准测试可能掩盖生产环境中的热行为。

<!-- cell: 91656971 src: cf8de1ebce -->
## 7. 从内存加载模型

当模型来自加密存储、软件包、网络服务或其他托管数据源时,从内存加载非常有用。NumPy 数组必须是 C 连续的,并且在引擎的整个生命周期内必须保持有效。

<!-- cell: bf637c62 src: 8a0360883c -->
## 8. 多输入模型契约

对于多输入模型,最安全的接口是以精确张量名称为键的字典:

~~~python
inputs = {
    "left_image": left_tensor,
    "right_image": right_tensor,
}
outputs = engine.run_multi_input(inputs)
~~~

DX-RT 也支持有序列表和单个拼接缓冲区,但命名输入可以减少顺序错误。

本实验会复用 T05 高级教程第 8 节创建的双输入 DXNN(如果存在)。它不会编译模型,也不会在 T06 之外创建文件。链接放在 `workspace/multi_input/` 而不是 `workspace/models/` 下,这样第 11 节的目录基准测试仍然只比较单输入分类模型。

<!-- cell: d7f33ae1 src: c7292df319 -->
## 9. 运行时事件与服务集成

<code>RuntimeEventDispatcher</code> 集中处理设备警告、错误、恢复通知、内存事件和降频事件。产品级的处理器应当快速、线程安全,并将结构化事件转发到应用的日志或健康系统。

下一个单元格注册一个处理器并派发一个**合成的教程事件**以验证路由。它不会模拟真实的硬件故障。

<!-- cell: 068ed27c src: 82e871f553 -->
### 9.1 ABI 与 IPC 边界

DX-RT v3.4 将公开集成接口与内部实现分离开来:

| 层 | 用途 | 产品指引 |
|---|---|---|
| 稳定的 C ABI,<code>dxrt_c_api.h</code> | 来自 <code>libdxrt.so</code> 的带版本 C 符号 | 用于语言绑定和二进制分发 |
| 仅头文件的 C++14 封装,<code>dxrt_cxx_api.h</code> | 建立在 C ABI 之上的现代 C++ 接口 | 推荐用于新的 C++ 应用 |
| 旧版桥接头文件 | 现有 <code>dxrt_api.h</code> 的源码兼容性 | 保证旧产品可继续构建;有计划地迁移 |
| 共享内存 IPC 与运行时服务 | 进程到运行时的高效通信 | 视为内部传输机制,而非应用层张量 API |

不要依赖 <code>libdxrt.so</code> 中隐藏的 C++ 符号。公开的 C/C++ 头文件才是受支持的集成边界。

<!-- cell: 1dbfded8 src: 4febd54669 -->
## 10. 可选的 CPU 侧加速

这些功能加速的是主机侧的运行时工作。它们不会改变 DXNN 图,也不会让 NPU 计算层更快。只有在性能分析器确认了对应的瓶颈之后才应启用它们。

| 功能 | 加速的工作 | x86_64 实现 | aarch64 实现 | 适用场景 |
|---|---|---|---|---|
| `NPU_FORMAT_CONVERSION_ACCELERATION` | NPU Format Handler(NFH): NPU 任务前后的转置、填充、切片和设备布局转换 | Intel IPP | ARM NEON/ASIMD | 输入或输出格式处理器耗时显著 |
| `CPU_OP_ACCELERATION` | CPU 回退子图中的 ONNX Runtime CPU 算子 | OpenVINO Execution Provider | XNNPACK Execution Provider | ORT CPU 时间主要由 Conv 或 MatMul 等计算密集型算子占据 |

NFH 加速**不会**取代应用的预处理。缩放、颜色转换、归一化以及其他模型特定的工作仍然遵循模型契约。CPU 算子加速**不会**把 CPU 算子迁移到 NPU 上;它只是选择一个更优化的 CPU 执行提供程序。

### 10.1 两道门槛: 构建支持与运行时启用

两道门槛都必须打开:

| 门槛 | 用途 | 默认值 |
|---|---|---|
| 构建时 CMake 选项 | 将该功能及其平台库编译进 DX-RT | `OFF` |
| 运行时设置 | 为当前进程启用已编译的功能 | `OFF` |

如果缺少构建支持,那么 C++ 枚举、Python 枚举以及对应的 `dxrun` 选项都不会存在。仅靠运行时启用无法补上缺失的实现。

### 10.2 构建带加速支持的 DX-RT

在 DX-RT 源码树中,编辑 `<DX_RT_DIR>/cmake/dxrt.cfg.cmake`,只修改以下两个选项:

~~~cmake
option(USE_NPU_FORMAT_CONVERSION_ACCELERATION
       "Accelerate NPU data format conversion (transpose/padding)" ON)
option(USE_CPU_OP_ACCELERATION
       "Accelerate CPU-side ONNX operations" ON)
~~~

然后执行一次干净构建,让 CMake 重新检测所需的库:

~~~bash
cd <DX_RT_DIR>
./build.sh --clean
~~~

`CPU_OP_ACCELERATION` 还要求 DX-RT 构建启用了 ORT,因为它会选择一个优化的 ONNX Runtime 执行提供程序。干净构建可能会下载或安装平台依赖。当所需的库或平台要求不可用时,构建会禁用所请求的功能。请检查 CMake 输出,而不是假定 `ON` 已生效。安装新的运行时之后,请重新安装应用所使用的、与之匹配的 `dx_engine` wheel。

<!-- cell: ac100003 src: 7d2752ab1c -->
### 10.3 验证已安装的运行时是否提供这些功能

请分别检查 Python wheel 和系统 CLI。即使它们报告相同的版本,其构建能力也可能不同。下面的单元格是只读的。一致的部署应当在产品使用的每个接口中都提供所需的功能。

> **标准安装会打印的内容:** 打包的 `dx_engine` wheel 会将两个功能都报告为 `available`,而系统 `dxrun` 则打印 `No acceleration options`。这是预期行为,不是错误:这两个二进制文件使用了不同的 CMake 选项构建(10.1 节),而 Python 枚举成员只存在于启用了该功能构建的 wheel 中。这正是本单元格要揭示的不一致。

<!-- cell: ac100005 src: 324af8acdf -->
### 10.4 在应用中启用这些功能

请在构造第一个 `InferenceEngine` **之前**配置加速。当 DXNN 包含 CPU 任务时,请保持 ORT 启用。

**C++**

~~~cpp
#include <dxrt/dxrt_cxx_api.h>

int main()
{
    auto& config = dxrt::Configuration::GetInstance();

#ifdef DXRT_NFH_ACCELERATION_AVAILABLE
    config.SetEnable(
        dxrt::Configuration::ITEM::NFH_ACCELERATION, true);
#endif

#ifdef DXRT_CPU_OP_ACCELERATION_AVAILABLE
    config.SetEnable(
        dxrt::Configuration::ITEM::CPU_OP_ACCELERATION, true);
#endif

    dxrt::InferenceOption option;
    option.useORT = true;  // Required only when the graph has CPU tasks.
    dxrt::InferenceEngine engine("model.dxnn", &option);
    // Prepare input and run inference.
}
~~~

预处理器守卫可以在已安装的 DX-RT 头文件中不存在某项加速功能时,仍然保持源码可构建。产品也可以选择把缺失的功能视为配置错误。

**Python**

~~~python
from dx_engine import Configuration, InferenceEngine, InferenceOption

config = Configuration()
required_items = ("NFH_ACCELERATION", "CPU_OP_ACCELERATION")
missing = [name for name in required_items if not hasattr(Configuration.ITEM, name)]
if missing:
    raise RuntimeError(f"DX-RT was built without: {', '.join(missing)}")

config.set_enable(Configuration.ITEM.NFH_ACCELERATION, True)
config.set_enable(Configuration.ITEM.CPU_OP_ACCELERATION, True)

option = InferenceOption()
option.use_ort = True  # Required only when the graph has CPU tasks.
with InferenceEngine("model.dxnn", option) as engine:
    # Prepare input and run inference.
    pass
~~~

Python wheel 必须与新安装的运行时版本和 Python ABI 匹配。否则枚举或原生扩展可能与共享库不匹配。

<!-- cell: ac100006 src: 1c4141f606 -->
### 10.5 使用 `dxrun` 测试

首先确认 `dxrun --help` 列出了 `--accel-nfh` 和 `--accel-cpu`。然后使用一个包含 CPU 任务的 DXNN 模型进行 A/B 比较。保持模型、ORT 设置、持续时间、预热、缓冲区数量、设备绑定和系统负载完全相同。

~~~bash
MODEL=/path/to/model-with-cpu-tasks.dxnn

# 1. Baseline
dxrun -m "$MODEL" --use-ort --benchmark --time 10 --warmup-runs 10 --buffer-count 6

# 2. Accelerate only NPU format conversion
dxrun -m "$MODEL" --use-ort --benchmark --time 10 --warmup-runs 10 --buffer-count 6 --accel-nfh

# 3. Accelerate only ORT CPU operators
dxrun -m "$MODEL" --use-ort --benchmark --time 10 --warmup-runs 10 --buffer-count 6 --accel-cpu

# 4. Enable both features
dxrun -m "$MODEL" --use-ort --benchmark --time 10 --warmup-runs 10 --buffer-count 6 --accel-nfh --accel-cpu
~~~

当您需要阶段级的证据时,请在一次较短的诊断运行中加上 `--profiler`。仅凭吞吐量无法判断发生变化的是 NFH、CPU 算子、传输还是 NPU 计算。

| 结果 | 解读 |
|---|---|
| NFH 时间减少 | 格式转换加速正在生效 |
| CPU 任务时间减少 | 优化的 ORT 执行提供程序正在发挥作用 |
| FPS 不变 | 瓶颈在其他阶段,或被加速的工作量太小 |
| 延迟或 CPU 使用率变差 | 对此工作负载禁用该功能并保留基线 |

`CPU_OP_ACCELERATION` 主要对计算密集型 CPU 操作有用。Reshape、Transpose 和 Concat 通常受内存带宽限制,可能几乎没有改善。两项功能都不保证一定带来性能提升。

### 10.6 相关选项: 动态 CPU 线程

如果性能分析器显示的是 CPU 任务队列压力,而不是单个算子开销大,请单独测试动态 CPU 线程:

~~~bash
export DXRT_DYNAMIC_CPU_THREAD=ON
dxrun -m "$MODEL" --use-ort --benchmark --time 10 --warmup-runs 10
~~~

这是另一个独立的 A/B 实验。它不能替代 NFH 或 CPU 算子加速。

<!-- cell: f43487a0 src: a4a30ab9f0 -->
## 11. 对多个模型进行基准测试

<code>dxbenchmark</code> 会在目录中查找 DXNN 文件,并生成机器可读和可视化的报告。结果路径位于 T06 内。每次运行都会新增一组带时间戳的 <code>DXBENCHMARK_&lt;date&gt;.csv/.html/.json</code> 文件,并在结果目录中写入一个 <code>profiler.json</code>;若想进行干净的比较,请删除旧文件。该目录只包含第 1 节链接的单输入模型;第 8 节的双输入模型被特意放在 <code>workspace/multi_input/</code> 中。

等效的终端命令:

~~~bash
cd <T06>/workspace/reports/dxbenchmark
dxbenchmark --dir <T06>/workspace/models \
            --result-path <T06>/workspace/reports/dxbenchmark \
            --time 3 \
            --warmup 3 \
            --sort fps \
            --order desc
~~~

<!-- cell: 669618a2 src: 39da7d0246 -->
## 12. 创建发布验证记录

<img src="assets/dx-rt-release-loop.svg" style="max-width: 1100px; width: 100%;" alt="生产运行时验证循环">

运行时发布记录应当把确切的二进制文件和环境与测得的证据关联起来。下一个单元格会在 T06 内写入一份精简的 JSON 记录。在做出真正的发布决策之前,请补充产品精度、端到端延迟、主机 CPU、内存和热测试结果。

<!-- cell: b2f25d6a-8d4f-4316-9883-6a704948cc37 src: 921089640f -->
## 13. 故障排除

| 症状 | 看到的内容 | 可能原因 | 解决方法 |
|---|---|---|---|
| 缺少 DX-RT Python 环境 | `Complete Intermediate Section 2 first` | 未运行教程 06-2 的第 2 节 | 运行一次即可;本笔记本复用 `workspace/.venv-dxrt` |
| 缺少 `profiler.json` | 针对 `workspace/profiler/profiler.json` 的 `FileNotFoundError` | 未运行 `dxrun --profiler` | 先运行第 4 节 |
| 多输入实验被跳过 | `Multi-input execution lab skipped` | 缺少教程 05-3 第 8 节的模型 | 可选;运行该节或直接继续 |
| Python 与 `dxrun` 报告的加速支持不一致 | Python 中为 `available`,`dxrun` 中为 `No acceleration options` | 两个二进制文件使用了不同的选项构建 | 以您将用于部署的那个为准;按需使用所需选项重新构建 DX-RT(第 10 节) |
| `dxbenchmark --warmup` 似乎被忽略 | 预热运行次数不匹配 | `--warmup` 的单位是秒,而不是运行次数 | 使用 `--warmup <seconds>`;`--warmup-runs` 属于 `dxrun` |

<!-- cell: 703286a9 src: d75aea9e39 -->
## 14. 总结

### 14.1 已完成的生产工作流

**冻结契约和工作负载**  
→ **测量基线**  
→ **分析每个运行时阶段**  
→ **一次只调优一种资源**  
→ **监控稳定性和设备健康状态**  
→ **保存可复现的发布证据**

<img src="assets/dx-rt-release-loop.svg" style="max-width: 1000px; width: 100%;" alt="DX-RT 生产验证循环">

### 14.2 高级控制面板

| 决策 | 先看证据 | 控制手段 |
|---|---|---|
| 队列深度 | Buffer Wait、延迟、内存 | <code>buffer_count</code> |
| 资源放置 | 设备/核心利用率 | <code>devices</code>、<code>bound_option</code> |
| CPU 回退 | 任务图和 CPU 任务时间 | <code>use_ort</code> |
| 格式转换 | 输入/输出格式处理器时间 | NFH 加速(如已编译) |
| CPU 算子开销 | CPU 任务类型和耗时 | CPU 算子加速(如已编译) |
| CPU 队列压力 | 队列等待和主机 CPU | 动态 CPU 线程 |
| 服务健康状态 | 运行时事件和设备快照 | 事件处理器和监控策略 |

### 14.3 产出的证据

| 产物 | 位置 | 用途 |
|---|---|---|
| 性能分析器 JSON | <code>workspace/profiler/profiler.json</code> | 原始事件时间线 |
| 性能分析器图像 | <code>workspace/profiler/profiler*.png</code> | 可视化瓶颈检查 |
| 基准测试报告 | <code>workspace/reports/dxbenchmark/DXBENCHMARK_*.csv/.html/.json</code>(每次运行一组)以及 <code>profiler.json</code> | 多模型比较 |
| 发布记录 | <code>workspace/reports/release_validation.json</code> | 可追溯的发布清单 |

### 14.4 完成清单

- [ ] 配置设备、核心绑定、ORT 和缓冲区数量
- [ ] 在同一固定工作负载下比较多个缓冲区数量
- [ ] 生成并可视化性能分析器跟踪
- [ ] 读取每个任务的阶段指标
- [ ] 将 CoV 用作计时稳定性信号
- [ ] 查询内存、利用率、时钟和热状态
- [ ] 从内存加载 DXNN 模型
- [ ] 准备命名的多输入推理路径
- [ ] 用合成事件验证运行时事件处理器
- [ ] 区分加速功能的构建支持与运行时启用
- [ ] 将 NFH 和 CPU 算子加速对应到 C++、Python 和 `dxrun` 的控制方式
- [ ] 在 T06 下生成基准测试和发布记录产物
- [ ] 使用真实的预处理和后处理运行同样的测试
- [ ] 用有代表性的标注数据验证任务精度
- [ ] 在部署主机上运行热浸泡测试和尾延迟测试

> **请记住:** 优化性能分析器所指出的那个阶段。孤立的 NPU 结果再快,在精度、端到端延迟、CPU 负载、内存、稳定性和热行为被一并验证之前,都不算发布结果。

> **下一步:** 继续学习演示教程(教程 10 为 Python OCR 流水线,教程 20-24 为 C++ 应用),将运行时应用到完整的应用程序中。
