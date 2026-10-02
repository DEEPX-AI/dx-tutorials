<!-- i18n source: notebooks/T06-DX-Runtime/dx_rt_02_intermediate.ipynb -->
<!-- i18n lang: zh -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 20e090c0 src: bc1aa2e5f2 -->
# DEEPX Tutorial 06-2 - DX-RT 中级

本笔记本从 CLI 验证进入应用层面的推理,使用 DX-RT 的 Python 和 C++ API。

## 学习目标

完成本教程后,您将能够:

- 在 Jupyter 内核中安装匹配的预构建 <code>dx_engine</code> wheel,
- 读取张量元数据,并按所需的形状和 dtype 分配输入,
- 实现同步推理,
- 使用作业 ID 和 <code>wait()</code> 实现异步推理,
- 说明回调与缓冲区生命周期的规则,
- 使用批处理 API 将相互独立的样本分组,
- 有意识地使用 <code>InferenceOption.buffer_count</code>,
- 使用 <code>dxrt_cxx_api.h</code> 构建一个最小的 C++14 应用,以及
- 根据延迟、吞吐量和所有权要求选择执行模式。

所有生成的 Python、C++、构建和报告文件都保留在 <code>&lt;dx-tutorials&gt;/notebooks/T06-DX-Runtime/workspace</code> 下。

<!-- cell: 550595f9 src: caf8fba665 -->
> 本教程共 3 部分,这是第 2 部分。课程地图和完整的部分列表见入门笔记本。

<!-- cell: e054aa4b-cc8a-4145-a2f1-3087227ff739 src: d38e092018 -->
## 前提条件

- 已完成教程 06-1。
- `uv` 和 `cmake`(以及 `build-essential` 中的 `g++`),分别用于 Python 环境和 C++ 构建。
- 网络访问: 第 2 节会在 `workspace/.venv-dxrt` 下创建一个 81 MB 的 Python 环境。`dx_engine` wheel 是本地的,但 `uv` 会下载其 NumPy 依赖,因此离线主机需要本地软件包索引。
- 预计阅读约 10 分钟。下载缓存就绪后,单元格本身大约一分钟即可运行完毕。

下一个单元格会定位 SDK、检查这些要求并打印状态表。标记为 `MISSING` 的项目都会附带提供它的步骤。

<!-- cell: 1d7d338e src: 98620d01b8 -->
## 1. 初始化教程工作区

<!-- cell: 0478080a src: d6685c848d -->
## 2. 在 T06 工作区中安装 Python 绑定

DX-RT Debian 包在 <code>/usr/share/libdxrt-bin/python</code> 下提供了针对特定版本的 wheel。为 Python 3.12 构建的 wheel 无法被 Python 3.13 内核导入,因此本笔记本会选择 <code>cpXY</code> 标签与当前运行内核匹配的 wheel。

为了把所有生成的文件都保留在 T06 内,本教程会在 <code>workspace/.venv-dxrt</code> 下创建一个小型专用环境,并把它的 site-packages 目录追加到内核的模块搜索路径中。追加(而不是前置)很重要:该环境中还包含作为 wheel 依赖的 NumPy,而内核必须继续使用自己的 NumPy。只有内核中没有的 <code>dx_engine</code> 才会从本地环境解析。这些命令等价于:

~~~bash
uv venv --python <current-jupyter-python> <T06>/workspace/.venv-dxrt
uv pip install --python <T06>/workspace/.venv-dxrt/bin/python \
  /usr/share/libdxrt-bin/python/dx_engine-<version>-cp<XY>-*.whl
~~~

这不会修改共享的 Jupyter 环境,也不会重新构建 DX-RT。当本地环境中已经包含匹配的 wheel 时,重新运行这些单元格会跳过创建和安装。

<!-- cell: 3d715559 src: 463872b41b -->
## 3. 分配内存前先读取张量元数据

不要根据模型名称推断输入形状或 dtype。请向引擎查询其契约。DX-RT v3.4 会校验 NumPy dtype,以防止未定义行为。

下面的辅助函数使用 <code>np.empty(...)</code>,然后填充缓冲区。这会强制分配真正可写的页面,避免在 DMA 固定(pinning)期间出现写时复制零页问题。

<!-- cell: b3a1916a src: b039b75719 -->
### 3.1 张量所有权检查清单

| 检查项 | 安全做法 |
|---|---|
| 形状 | 根据 <code>get_input_tensors_info()</code> 分配 |
| Dtype | 严格使用返回的 NumPy dtype |
| 布局 | 与编译后模型的可见布局一致 |
| 连续性 | 传入 C 连续数组 |
| 生命周期 | 异步输入在完成之前保持存活 |
| 输出作用域 | 仅当回调输出必须在回调之外继续使用时才复制 |

<!-- cell: 62cafeb6 src: 658cf437ee -->
## 4. 同步推理

<code>run()</code> 会阻塞调用线程,直到结果就绪。对于顺序执行或面向延迟的应用,它是最清晰的起点。

<img src="assets/dx-rt-execution-modes.svg" style="max-width: 1100px; width: 100%;" alt="同步、异步和批处理执行模式">

<!-- cell: 3e183092 src: de13644a11 -->
这三个计时回答的是不同的问题:

- **主机观测时间** 包括 Python 调用及其周围的主机端开销。
- **DX-RT 延迟** 是运行时最近一次请求的延迟。
- **NPU 推理时间** 只涵盖 NPU 执行,仅是端到端延迟的一部分。

不要把 NPU 时间标注为从摄像头到显示的延迟。

<!-- cell: 54f7a81f src: bb3c7d9b98 -->
## 5. 使用作业 ID 的异步推理

<code>run_async()</code> 返回一个作业 ID。之后 <code>wait(job_id)</code> 返回对应的输出。这使应用可以让提交、NPU 执行和其他工作相互重叠。

<img src="assets/dx-rt-buffer-lifecycle.svg" style="max-width: 1100px; width: 100%;" alt="异步缓冲区所有权生命周期">

<!-- cell: d0e724c4 src: cbe255be5a -->
### 5.1 等待与回调的对比

| 完成方式 | 优势 | 主要职责 |
|---|---|---|
| <code>run_async()</code> + <code>wait(job_id)</code> | 请求与结果的显式匹配 | 保存作业 ID,并在合适的线程中等待 |
| 注册的回调 | 低延迟的完成处理 | 保持回调快速且线程安全 |
| <code>run()</code> | 最简单的控制流 | 接受调用线程被阻塞 |

回调输出仅在回调作用域内有效。如果下游工作必须保留它们,请在回调内复制所需数据,并把繁重的工作转移到另一个队列。

<!-- cell: af9ec43d src: 35292b89d9 -->
## 6. 批处理 API

DX-RT 批处理执行将多个相互独立的样本分组,并在运行时内部异步调度它们。它**不会**改变编译后模型的批处理维度;DXNN 模型通常仍使用批处理大小 1。当前 API 使用带批处理格式输入和显式输出缓冲区的 `run()`;`run_batch()` 仅作为已弃用的兼容包装保留。

嵌套的 Python 形式为:

~~~python
[
    [sample_0_input_0],
    [sample_1_input_0],
    [sample_2_input_0],
]
~~~

<!-- cell: 4b77e8c1 src: 3d13107715 -->
## 7. 缓冲区数量即流水线容量

<code>InferenceOption.buffer_count</code> 控制内部推理缓冲区的数量。更大的值可以允许更多进行中的工作,但也会消耗更多内存,并且在流水线填满之后可能不再带来收益。

| 过小 | 均衡 | 过大 |
|---|---|---|
| 提交方需要等待缓冲区 | 有足够的工作让 NPU 保持忙碌 | 额外内存却收益甚微 |
| 吞吐量可能降低 | 稳定的吞吐量和有界的内存 | 更长的队列可能增加延迟 |

请测量有代表性的取值,而不是假设最大值就是最佳值。高级教程会进行一次受控的缓冲区数量实验。

<!-- cell: 485e925a src: f6386e2aa5 -->
## 8. 构建一个最小的 C++14 应用

DX-RT v3.4 提供稳定的 C ABI 和一个仅头文件的 C++14 包装器。新的 C++ 代码只应包含:

~~~cpp
#include <dxrt/dxrt_cxx_api.h>
~~~

不要在同一个翻译单元中同时包含 <code>dxrt_api.h</code> 和 <code>dxrt_cxx_api.h</code>。

DX-RT 安装了 CMake 包配置(<code>/usr/local/lib/cmake/dxrt/dxrtConfig.cmake</code>),因此项目只需要 <code>find_package(dxrt REQUIRED)</code> 和导入目标 <code>dxrt::dxrt</code>。该目标自带包含目录和 <code>pthread</code> 依赖。演示教程(20-24)把同样的查找封装在一个小型 CMake 辅助脚本中;下面的几行是其原始形式。

接下来的两个单元格只会在 <code>T06-DX-Runtime/workspace/cpp</code> 下创建源文件。

<!-- cell: 78076ec1 src: 8d9e43b968 -->
### 8.1 配置与构建

这些笔记本单元格运行的命令与您在终端中输入的完全相同:

~~~bash
cmake -S <T06>/workspace/cpp -B <T06>/workspace/cpp/build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build <T06>/workspace/cpp/build --parallel
~~~

构建目录保留在 T06 内。

<!-- cell: 1d3ac8cb src: d3fa81e622 -->
### 8.2 运行 C++ 应用

等价的终端命令:

~~~bash
<T06>/workspace/cpp/build/dxrt_sync \
  <T06>/workspace/models/resnet50_224x224.dxnn
~~~

<!-- cell: 68c92258 src: 34a1319a55 -->
## 9. 选择执行方式

| 需求 | 推荐的起点 | 原因 |
|---|---|---|
| 最简单的顺序流程 | 同步 <code>run()</code> | 清晰的所有权和错误处理 |
| 最低单请求延迟研究 | 同步 <code>run()</code> | 没有刻意的队列深度 |
| 更高的流式吞吐量 | 异步 + <code>wait()</code> 或回调 | 让相互独立的流水线工作重叠 |
| 同时处理多个独立样本 | 批处理 API | 将提交和完成分组 |
| 紧密集成和低 Python 开销 | C++ API | 直接的 C++14 应用控制 |
| 现有的 Python 流水线 | Python API | 与 NumPy 数据快速集成 |

正确的选择取决于完整的应用。如果队列过深,更高的推理 FPS 仍可能导致用户可见的延迟变差。

<!-- cell: 3fd7908b-0ab7-49ad-a0a4-2513fcb1771d src: 04785bad99 -->
## 10. 故障排除

| 症状 | 看到的内容 | 可能原因 | 解决方法 |
|---|---|---|---|
| 缺少 `uv` 或 `cmake` | `[MISSING] uv` 或 `[MISSING] cmake` | 未安装 | `curl -LsSf https://astral.sh/uv/install.sh | sh`;`sudo apt install cmake build-essential` |
| 没有适用于此 Python 的 wheel | `No dx_engine wheel for cp3XY` | `/usr/share/libdxrt-bin/python` 下没有与内核 Python 版本匹配的 wheel | 使用受支持的 Python 版本创建教程的 `.venv` |
| `import dx_engine` 失败 | `ImportError` 或版本不匹配的消息 | `workspace/.venv-dxrt` 中的 wheel 比 `libdxrt.so` 旧 | 重新运行第 2 节以重新安装 wheel |
| CMake 找不到 DX-RT | `Could not find a package configuration file provided by "dxrt"` | DX-RT 安装在自定义前缀下,因此 `dxrtConfig.cmake` 不在默认搜索路径中 | 在 `cmake -S ... -B ...` 行中添加 `-DCMAKE_PREFIX_PATH=<prefix>` |
| 异步输出看起来不正确 | `wait()` 之后出现垃圾值 | 输入缓冲区在 `wait()` 返回之前被释放或复用 | 在 `wait()` 返回之前保持输入数组存活 |

<!-- cell: b69fd536 src: 8d0b95c249 -->
## 11. 总结

### 11.1 已完成的 API 工作流

**读取张量元数据**  
→ **按精确的 dtype 和形状分配**  
→ **同步运行**  
→ **异步提交并等待**  
→ **将相互独立的样本分组**  
→ **用 C++ 构建相同的流程**

<img src="assets/dx-rt-execution-modes.svg" style="max-width: 1000px; width: 100%;" alt="DX-RT 执行模式">

### 11.2 执行概览

| 模式 | 提交 | 完成 | 首选指标 |
|---|---|---|---|
| 同步 | 单个请求 | 从 <code>run()</code> 返回 | 请求延迟 |
| 异步 + 等待 | 多个作业 ID | <code>wait(job_id)</code> | 吞吐量和队列延迟 |
| 回调 | 多个请求 | 运行时回调 | 回调开销和吞吐量 |
| 批处理 | 一组样本 | 一组输出 | 有效样本数/秒 |
| C++ | 相同的运行时概念 | C++ 张量 | 端到端应用开销 |

### 11.3 完成清单

- [ ] 安装与 Jupyter Python ABI 匹配的 wheel
- [ ] 根据运行时元数据分配连续张量
- [ ] 测量同步模式下的主机、运行时和 NPU 计时
- [ ] 在 <code>wait()</code> 返回之前保持异步输入的生命周期
- [ ] 在不改变模型批处理大小的情况下使用批处理 API
- [ ] 说明 <code>buffer_count</code> 的内存/吞吐量权衡
- [ ] 构建并运行一个 C++14 DX-RT 应用
- [ ] 用产品的预处理替换虚拟输入并验证输出

> **请记住:** 缓冲区所有权是正确性的一部分。性能调优应在验证形状、dtype、布局、生命周期和输出映射之后进行。

### 11.4 下一步

继续学习**高级教程**,绑定设备和核心、分析各个阶段的性能、监控设备健康状态、测试内存加载和多输入模型,并构建可复现的发布证据。
