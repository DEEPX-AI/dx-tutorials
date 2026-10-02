<!-- i18n source: notebooks/T04-DX-STREAM/dx_stream.ipynb -->
<!-- i18n lang: zh -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 1726e9e4-5821-4f34-add4-37f1fb44ee77 src: c7877fefea -->
# DEEPX Tutorial 04 - DX-STREAM 工作流

本教程介绍 DX-STREAM v3.1.x,并展示如何在 DEEPX NPU 上构建端到端的视觉 AI 流水线。

## 学习目标

您将:

- 了解 DX-STREAM 在 DEEPX SDK 中所处的位置
- 使用 `DxPreprocess → DxInfer → DxPostprocess → DxTracker → DxOsd` 构建流水线
- 学习 DX 多流域(multi-stream domain)及选择器边界
- 使用推理后端、缩放、转换和消息代理元素
- 为自定义 AI 模型适配自定义后处理库
- 添加由元数据驱动的人数统计叠加层
- 检查元数据并诊断 GStreamer 流水线

<!-- cell: 32d7336d-3f4d-4c4a-aa70-2662f4558edc src: 6645080682 -->
## 1. DX-STREAM 概述

DX-STREAM 是一组用于在 DEEPX NPU 上构建视觉 AI 流水线的 GStreamer 元素。

```text
Source → Decode → DxPreprocess → DxInfer → DxPostprocess → DxTracker → DxOsd → Sink
```

![DX-STREAM 流水线](assets/dx-stream-pipeline.png)

| 元素 | 用途 |
|---|---|
| `dxpreprocess` | 对帧进行缩放、裁剪,并转换为模型输入张量 |
| `dxinfer` | 运行编译后的 `.dxnn` 模型 |
| `dxpostprocess` | 解码输出张量并写入推理元数据 |
| `dxtracker` | 为检测到的目标分配持久 ID |
| `dxosd` | 绘制边界框、标签、姿态和分割结果 |
| `dxgather` | 合并由同一源创建的分支 |
| `dxinputselector` | 将输入流合并进 DX 多流域 |
| `dxoutputselector` | 按流 ID 拆分 DX 多流域 |
| `dxrate` | 控制输出帧率 |
| `dxmsgconv` | 将推理元数据转换为结构化消息 |
| `dxmsgbroker` | 将消息发布到 MQTT 或 Kafka |
| `dxscale` | 更改帧分辨率 |
| `dxconvert` | 更改帧颜色格式 |

完整的元素与属性参考,请参阅 [DEEPX Developer Portal](https://developer.deepx.ai/download/?id=583) 提供的 DX-STREAM 用户手册。需要登录。

<!-- cell: 2d53a070-40a1-4658-950d-62c7c796af9b src: fe4feb49aa -->
## 2. 前提条件

- 已按教程 01 使用 `./dx-runtime/install.sh --all` 完成安装:DX-RT、DX-APP 和 DX-STREAM 已构建,NPU 可见。推荐先完成教程 03;第 5 节在其叉车与工人(Forklift and Worker)模型存在时会复用该模型。
- 用于显示窗口的图形桌面会话。设置单元格会检测显示器:有显示器时,流水线单元格会打开 DX-STREAM 窗口;没有显示器时(例如通过 SSH),它们会以**无头(headless)**方式使用 `fakesink` 运行,以便笔记本仍能完整执行。`HEADLESS` 可以强制设为任意一种方式。第 4 节的 `run_demo.sh` 演示和摄像头单元格需要手动启用(`RUN_DEMOS`、`RUN_CAMERA`),因为每个都需要数分钟。
- 可选硬件:用于 4.3 的 V4L2 摄像头(`RUN_CAMERA`)、用于 4.9 的可访问 RTSP 流、用于 4.11 的 MQTT 或 Kafka 代理。
- 工具:`gstreamer1.0-tools`、`v4l-utils`、`jq`(`setup.sh` 使用)、用于 7.3 的 `graphviz`。5.7 节使用 `meson` 重新构建 DX-STREAM,需要 `sudo`。
- DX-STREAM GStreamer 插件必须对 `gst-inspect-1.0` 可见。`build.sh` 会打印 `export GST_PLUGIN_PATH=...` 这一行;设置单元格会检查这一点,并在插件位于默认位置时为内核设置该变量。
- 下载量:`setup.sh` 获取 17 个示例模型(约 300 MB)和共享示例视频归档(约 1.1 GB,存放在 `<dx-all-suite>/workspace/res/videos` 下,与 DX-APP 共享)。第 5 节复用教程 03 的模型和视频,否则下载 90 MB。
- 本教程创建的文件存放在 `notebooks/T04-DX-STREAM/workspace/`,git 会忽略该目录。SDK 树中唯一的改动是 5.6 中的后处理补丁,这正是该节的目的。

下一个单元格会定位 SDK、检查这些要求并打印状态表。

<!-- cell: 53136a15-c25c-4168-aa38-a07629a40996 src: 1605721eba -->
### 2.1 加载 SDK 路径

下面的单元格根据 `DX_ALL_SUITE_DIR` 推导出 `DX_STREAM_DIR` 和其他 SDK 路径,然后把笔记本的工作目录切换到 DX-STREAM 仓库根目录,这是 `run_demo.sh` 和示例路径所要求的。它还定义了下文使用的显示标志和工作区路径。

<!-- cell: 8d9cf255-57c7-4b99-be3b-741331f6e532 src: 75741dab3d -->
### 2.2 (可选)Intel GPU 前提条件

仅针对 Intel x86_64 iGPU/dGPU 系统,请按照 [`docs/intel_gpu.md`](../../docs/intel_gpu.md) 操作。

<!-- cell: e7c0a679-122a-4e4e-bae9-1c034689e281 src: 5ae0009348 -->
### 2.3 下载示例资源

`setup.sh` 会把捆绑流水线所需的模型和视频下载到 `dx_stream/samples/`。当前列表包含 17 个模型(约 300 MB)和共享示例视频归档(约 1.1 GB;`setup.sh` 将其解压到 `<dx-all-suite>/workspace/res/videos` 下,并链接为 `dx_stream/samples/videos`)。首次运行预计需要几分钟。当示例已经存在时,单元格会跳过该脚本。

<!-- cell: fe12c34c-a475-437f-bba9-d6ed0668f6b0 src: de693a0b81 -->
### 2.4 验证已安装的 DX-STREAM 元素

该插件应提供以下 13 个元素:

`dxconvert`、`dxgather`、`dxinfer`、`dxinputselector`、`dxmsgbroker`、`dxmsgconv`、`dxosd`、`dxoutputselector`、`dxpostprocess`、`dxpreprocess`、`dxrate`、`dxscale` 和 `dxtracker`。

<!-- cell: 6a2cff9b-d3eb-48ad-be1e-c18fb204726d src: 3e2aa3dfa3 -->
## 3. 快速入门

### 3.1 运行 YOLO26n 目标检测流水线

此流水线读取一段视频,对每一帧进行预处理,运行 YOLO26n,解码结果并渲染带标注的输出。在桌面会话中(`HEADLESS = False`,检测到显示器时的默认值),结果会在窗口中打开,单元格会一直处于运行状态,直到视频结束或您按下停止按钮(■)。没有显示器时(`HEADLESS = True`),帧会发送到 `fakesink`,单元格只打印流水线状态消息;此时的重点在于每个元素都能链接、模型能加载、推理能运行。单元格会打印它使用的 sink。

若要改为在终端中运行(File > New > Terminal),请 `cd` 到设置单元格打印的 DX-STREAM 目录,粘贴下面的 `gst-launch-1.0` 命令,并以 `videoconvert ! fpsdisplaysink sync=false` 作为最后一个元素。按 `Ctrl+C` 停止。

<!-- cell: 31792737-11bd-409e-975b-6d068cf213a8 src: b9702f410a -->
### 3.2 检查核心元素

`gst-inspect-1.0` 会显示已安装版本的 pad 模板、属性、默认值和支持的枚举值。

<!-- cell: ce012ae3-7b15-4cde-89fa-9e7a998f985f src: e7ac256517 -->
#### 3.2.1 DxPreprocess

```bash
dxpreprocess preprocess-id=1 resize-width=640 resize-height=640
```

`preprocess-id` 标识生成的输入张量。下游的 `dxinfer` 必须引用相同的 ID。

<!-- cell: 286b0cb5-2662-454e-a1a3-ae94aff066ce src: 5260c0d019 -->
#### 3.2.2 DxInfer 与推理后端

```bash
dxinfer preprocess-id=1 inference-id=1 model-path=/path/to/model.dxnn backend=auto
```

- `auto`: 选择一个可用的已编译后端
- `dxrt`: 使用 DEEPX Runtime 后端
- `dxvnpu`: 当 DX-STREAM 以 VNPU 支持构建时,使用 VNPU 后端

`inference-id` 标识供 `dxpostprocess` 消费的输出张量。DX-STREAM v3.1.x 内部使用异步 Put/Get 后端接口;流水线语法保持不变。

<!-- cell: eba1a455-60a5-4898-99b4-ab0a40a00f21 src: b9001c0bcb -->
#### 3.2.3 DxPostprocess

```bash
dxpostprocess inference-id=1 \
  library-file-path=/usr/local/share/gstdxstream/lib/libpostprocess_yolo26od.so \
  function-name=PostProcess
```

`inference-id` 必须与上游的推理输出匹配。共享库和函数必须与模型架构匹配。

<!-- cell: 98989ca3-7203-4c23-9311-c49d59adf254 src: 42550f73c9 -->
#### 3.2.4 DxOsd

DxOsd 读取推理元数据,并把可视化结果叠加到视频帧上。

<!-- cell: 98f4122d-835f-46a0-b359-58d35c5ef222 src: 86672b6017 -->
### 3.3 缩放与转换视频帧

`dxscale` 更改分辨率,`dxconvert` 更改颜色格式。在受支持的平台上会自动选择加速内核;不受支持的组合会回退到软件实现。

```bash
... ! dxscale width=640 height=480 ! \
      dxconvert ! video/x-raw,format=RGB ! ...
```

`dxconvert` 不会调整帧大小,`dxscale` 也不会选择最终的颜色格式。

本笔记本中的流水线在显示 sink 之前以软件 `videoconvert` 结尾。硬件 `dxconvert` 速度更快,但在本 SDK 版本上,它对某些片段(例如第 5 节的叉车与工人视频)会间歇性地产生全绿的帧。捆绑脚本通过其 `VIDEOCONVERT_PIPELINE` 变量按平台选择转换器;如果您把 `SINK` 切换为 `dxconvert` 后看到绿帧,请切换回来。

<!-- cell: a2bacdbc-f65c-4c63-804d-88241bb9f07a src: 17eb05a204 -->
## 4. 运行捆绑的演示流水线

`run_demo.sh` 提供 12 个选项:`0` 到 `9`、代表 Secondary Mode 的 `-`,以及代表深度估计的 `=`。如果十秒内未收到选择,它会运行选项 `0`。下一个单元格打印已安装的菜单。

这些演示会打开原生窗口,并在流水线退出前一直占用单元格,因此下面的单元格仅在桌面会话中且 `RUN_DEMOS = True`(设置单元格)时才会运行。推荐的方式是使用终端(File > New > Terminal):

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_stream     # the setup cell printed your actual path
./run_demo.sh
```

请及时输入菜单项(提示会在十秒后超时),按 `Ctrl+C` 停止流水线。每个演示之后都附有它运行的脚本,以及关于其中关注要点的说明。

<!-- cell: cdce0b1a-4fae-4368-9720-5bfb211b0776 src: 63e050e95e -->
使用笔记本的停止按钮(`■`)停止正在运行的流水线。

### 4.1 目标检测 - YOLO26n

![单路目标检测流水线](assets/pipline-single-detection.png)

关注要点: 与快速入门相同的五个 DX 元素。`model-path` 和 `library-file-path=.../libpostprocess_yolo26od.so` 直接设置在 `dxinfer` 和 `dxpostprocess` 上。

<!-- cell: f9415924-9b2c-4691-8811-46da31c7b397 src: c79045ee08 -->
### 4.2 使用 PPU 的目标检测 - YOLOv5s

关注要点: 没有 `library-file-path`。每个 DX 元素都读取 `dx_stream/configs/YoloV5S_PPU/` 下的 JSON `config-file-path`。PPU 意味着 NPU 已经解码了边界框,因此后处理配置只需映射 PPU 输出。

<!-- cell: 0598f06b-a1de-40bd-8339-d6c93eff57a5 src: ef74dca18a -->
### 4.3 人脸检测

关注要点: 4.1 的结构,配合人脸模型和 `libpostprocess_yolov5s_face.so`;`dxosd` 绘制人脸框和关键点。

<!-- cell: 95a894ed-f224-4e72-b6f3-b2fc702b4f9e src: 51efb7d6f3 -->
下一个示例使用带 PPU 输出的 SCRFD500M 模型。

关注要点: 通用的 `libpostprocess_ppu.so`。对于 PPU 模型,一个共享库可以服务多种架构。

<!-- cell: bed6a8d7-6e12-444d-85c4-72c6e7dab466 src: f29ec878ae -->
#### 使用摄像头源

第一个示例请求原始视频。第二个示例请求 MJPEG,并非所有 USB 摄像头都支持。请根据 `v4l2-ctl --list-formats-ext --device=/dev/video0` 的输出调整设备和 caps。

<!-- cell: db029aad-2a05-4904-946c-d7b67e8e27cc src: 9f586331fc -->
### 4.4 姿态估计

关注要点: `libpostprocess_yolo26pose.so` 把关键点写入 `DXObjectMeta`,`dxosd` 据此绘制骨架。

<!-- cell: 70f47140-89cd-477b-abfe-e5c5ed869042 src: b99cd2cfe3 -->
下一个示例使用带 PPU 输出的 YOLOv5 Pose。

关注要点: 再次使用 `libpostprocess_ppu.so`。与 4.4 对比,只有模型和库发生了变化。

<!-- cell: fc15ee5e-4fc9-4595-a4e2-3da2f54f5420 src: eece8565d4 -->
### 4.5 实例分割 - YOLO26n-Seg

选项 `6` 运行分割演示。在当前 SDK 布局中,该脚本位于 `instance_segmentation` 下。

关注要点: 位于 `dxpostprocess` 和 `dxosd` 之间的 `dxtracker` 元素;分割演示还会为掩码分配跟踪 ID。

<!-- cell: ba5d8d52-5fe5-4cd2-93b8-b2db52570a1a src: 2a8e38c8f9 -->
### 4.6 多目标跟踪

选项 `7` 先运行目标检测,再进行跟踪。

![单路跟踪流水线](assets/pipline-single-tracking.png)

关注要点: 4.2 的检测链加上 `dxtracker config-file-path=.../tracker_config.json`。跟踪器配置选择算法及其阈值。

<!-- cell: 154b4e5c-5a1f-4036-9be6-fd8e8e1a39ab src: ea853b3cad -->
### 4.7 多通道目标检测

选项 `8` 运行四个独立的推理分支,并将它们的输出合成在一起。

![多通道流水线](assets/pipeline-multi-stream.png)

关注要点: 四个 `filesrc ... ! dxscale` 分支,各自拥有自己的预处理、推理和后处理,由 `compositor` 合并。模型在每个分支中各加载一次。

<!-- cell: 30471517-0937-4f1a-bd77-f41ac4749987 src: 3b21ecdbaf -->
### 4.8 DX 多流域

DX-STREAM v3.1.0 引入了 `application/x-dxvideoraw` caps 域。它允许多个流共享一条处理链,同时保留每个流的身份、尺寸、格式、元数据和时间线。

```text
N × video/x-raw
       ↓
dxinputselector                 domain entry
       ↓ application/x-dxvideoraw
dxpreprocess → dxinfer → dxpostprocess → dxosd
       ↓ application/x-dxvideoraw
dxoutputselector                domain exit
       ↓
N × video/x-raw
```

`dxinputselector` 分配流 ID,并在需要时创建 `DXFrameMeta`。`dxoutputselector` 路由每个缓冲区,并恢复对应的每流事件。

该域保留每个流的 `STREAM_START`、`CAPS`、`SEGMENT`、`TAG`、`EOS` 和 `GAP` 事件。因此,慢速流可以自行处理 QoS,而不会限制所有输入。

<!-- cell: 0ecc9dfd-6d20-4385-b4c3-f5bcc9b6bf70 src: aa2cd3fde1 -->
#### 元素放置规则

| 元素 | 位于 `application/x-dxvideoraw` 内部 | 放置说明 |
|---|---:|---|
| `dxpreprocess`、`dxinfer`、`dxpostprocess`、`dxtracker`、`dxosd`、`dxrate` | 是 | 这些元素可在单流模式或域模式下工作 |
| `dxscale`、`dxconvert` | 否 | 请将它们放在 `dxinputselector` 之前或 `dxoutputselector` 之后 |
| `dxgather` | 否 | 它合并来自同一源的分支,而不是不同的流 ID |
| `videoconvert`、`videoscale`、`compositor` 等标准元素 | 否 | 它们接受 `video/x-raw`,而非 `application/x-dxvideoraw` |
| `dxinputselector`、`dxoutputselector` | 仅限边界 | 它们负责进入和退出该域 |

放置错误会在 caps 协商阶段失败,而不会产生含糊不清的运行时结果。

<!-- cell: e89082d6-2e11-4491-acc9-b64f356b528a src: 76d7544217 -->
#### 共享一条推理链

当所有通道使用同一个模型时,单条共享推理链可以避免为每个通道各加载一次模型,并降低 NPU 内存占用。

<img src="assets/pipeline-multi-stream-single-infer.png" style="max-width: 1400px;">

当前示例使用 `run_multi_stream_selector.sh`。

关注要点: 环绕一条共享链的 `dxinputselector name=in` 和 `dxoutputselector name=out`。数一数 DX 元素的数量,并与 4.7 对比。

<!-- cell: e8f05808-0f68-475f-b0b0-d9ee7ba85240 src: c7e15732ef -->
### 4.9 多通道 RTSP

选项 `9` 演示多个 RTSP 源。实时流水线受益于 v3.1.x 中修正的延迟和 QoS 报告。

关注要点: 4.8 的选择器布局,输入换成 `rtspsrc`。流 URL 是脚本顶部的变量;请根据您的摄像头进行编辑。

<!-- cell: 317ba344-632d-4528-964e-4411d31d7b56 src: 98c893db6e -->
### 4.10 Secondary Mode

![Secondary Mode 流水线](assets/pipeline-secondary.png)

- **主模式(Primary Mode)** 对整帧进行预处理和推理。后处理通常会创建新的 `DXObjectMeta` 对象。
- **次级模式(Secondary Mode)** 对检测到的目标区域进行预处理。后处理会更新或丰富已有的目标元数据。

关注要点: 第二组预处理、推理和后处理三元组上的 `secondary-mode=true`。SCRFD 找到人脸,然后 EfficientNet 对每个人脸区域而非整帧进行分类。

<!-- cell: c73d9a02-064d-4b78-a062-5598aa8546cc src: 6c9c479a23 -->
### 4.11 将推理结果发布到 MQTT 或 Kafka

`dxmsgconv` 使用自定义消息转换库转换推理元数据。`dxmsgbroker` 把载荷发布到 MQTT 或 Kafka 代理。

```text
... → dxpostprocess → dxmsgconv → dxmsgbroker
```

在 v3.1.x 中,`include-frame=true` 会把当前帧作为 Base64 编码的 JPEG 添加到消息中。这会增加 JPEG 编码工作量、消息大小和网络流量,因此仅在消费者需要图像时才启用。

```bash
dxmsgconv library-file-path=/path/to/libmessage_convert.so include-frame=true ! \
dxmsgbroker broker-name=mqtt conn-info=localhost:1883 topic=test
```

代理必须在流水线启动前处于运行状态。捆绑脚本提供了完整的 MQTT 和 Kafka 示例,但不属于交互式 `run_demo.sh` 菜单。

<!-- cell: 4f9da17d-7791-42e5-9930-16a25ace619a src: a8c8d5a58b -->
## 5. 编写您自己的应用

本节集成教程 03 中创建的叉车与工人检测器。该模型使用 YOLOv7 输出格式,但只有两个类别而不是 80 个 COCO 类别,因此必须适配其后处理配置。

![自定义流水线](assets/custom-pipeline.png)

<img src="assets/detection-goal.jpg" style="max-width: 1400px;">

<!-- cell: 66060d59-25bd-463a-8c02-e67713decc74 src: 423cab8027 -->
### 5.1 DX-STREAM v3.1.x 自定义库 API

为旧版本编写的自定义预处理和后处理库必须按如下方式更新:

- 已移除 `DXFrameMeta::_buf`,以避免循环缓冲区引用。
- 自定义函数的第一个参数现在是 `GstBuffer *buf`。
- 使用 `dx_acquire_obj_meta_from_pool()` 创建目标元数据。
- 使用 `dx_add_obj_meta_to_frame()` 附加新目标。
- 在 Primary Mode 下,后处理创建结果对象。
- 在 Secondary Mode 下,后处理更新通过 `object_meta` 传入的对象。

当前的 YoloV7 库已经使用 v3.1.x 的函数形式:

```cpp
extern "C" void PostProcess(
    GstBuffer* buf,
    std::vector<dxs::DXTensor> network_output,
    DXFrameMeta* frame_meta,
    DXObjectMeta* object_meta)
{
    DXObjectMeta* result = dx_acquire_obj_meta_from_pool();

    // Decode tensors and populate result.

    dx_add_obj_meta_to_frame(frame_meta, result);
}
```

<!-- cell: 48523baa-d664-4f24-836e-f876a6c6959e src: e73e58aa61 -->
### 5.2 元数据层次结构

推理结果随 `GstBuffer` 一起传递,而不是通过单独的旁路通道。

```text
GstBuffer
└── DXFrameMeta
    ├── stream ID, width, height, format, ROI
    ├── input tensors  (preprocess ID → tensors)
    ├── output tensors (inference ID → tensors)
    ├── frame-level classification or segmentation
    ├── DXObjectMeta[]
    │   ├── label, confidence, box, tracking ID
    │   ├── keypoints, features, OBB, face, segmentation
    │   └── DXUserMeta[]
    └── DXUserMeta[]
```

`DXUserMeta` 可以把应用特定的数据附加到帧或目标上。用户元数据的实现必须同时提供复制和释放函数,以便在 GStreamer 复制或释放缓冲区时元数据保持有效。

<!-- cell: 712b8026-aa30-4662-a7ec-5f7a2e79e179 src: 77b28e4653 -->
### 5.3 获取自定义 DXNN 模型

教程 03 把 `yolov7-forklift-person.dxnn` 编译(或下载)到了它的工作区。该单元格在文件存在时复制它,否则下载同一模型(71 MB)。

```bash
wget -nc https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/yolov7-forklift-person.dxnn
```

<!-- cell: 8e29f8a5-4356-4e58-93a9-9c247bfa7c77 src: fa60065b2f -->
### 5.4 检查编译后的模型

在修改后处理器之前,先检查模型的输出。`dxparse -v` 会打印输入和输出张量。自定义模型保留了 YOLOv7 的头部布局:原始检测头有 `3 × (5 + 2) = 21` 个通道(三个锚框、四个框值、目标置信度、两个类别分数),而不是 COCO 模型的 `3 × (5 + 80) = 255`。`libpostprocess_yolov7.so` 中的解码器已经能处理这种布局;只有类别数量和类别名称不同,而这正是 5.5 中的补丁所修改的内容。

<!-- cell: 6972d420-7312-4a41-bd63-1a01205af19a src: 0127f93eee -->
### 5.5 适配 YoloV7 后处理配置

正如 5.4 所示,输出解码器已经与模型架构匹配。只需更改类别的数量和名称。补丁会写入工作区,并在 5.6 中应用到 SDK 源代码。

<!-- cell: b16382e7-7b3a-45e5-88bd-9bb80cc61480 src: 1e14d10b82 -->
### 5.6 应用补丁

下面的单元格可以安全地多次运行。它在需要时应用补丁,在相同补丁已存在时报告成功,在源代码存在冲突修改时停止且不更改文件。它不会重置或丢弃已有的工作。

<!-- cell: 4f49345d-1354-4454-9bfe-bf1de2b099b7 src: b1a18419d0 -->
### 5.7 重新构建 DX-STREAM

`build.sh` 使用 `meson` 重新构建插件和自定义库,把它们安装到配置的前缀下,并为第 6 节创建带有 `pydxs` 的 `venv-dx_stream`。安装步骤和清理 root 所有的构建目录都需要 `sudo`,因此与教程 01 一样,单元格仅在 `sudo` 可以免密运行时才在此处执行构建,否则打印供终端使用的命令。完整重新构建需要几分钟。如果因为已有的 `builddir` 是由其他 meson 版本生成而导致增量构建失败,单元格会回退到 `./build.sh --clean`,从头重新构建。

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_stream
./build.sh
```

<!-- cell: 5d801612-1e0a-44fa-8345-74636b91a421 src: 9808c3e4d1 -->
### 5.8 创建推理配置

`dxinfer` 的属性可以直接在流水线中提供,也可以通过 JSON 文件提供。

| JSON 字段 | 含义 |
|---|---|
| `preprocess_id` | 选择由具有相同 ID 的 `dxpreprocess` 元素创建的输入张量 |
| `inference_id` | 标记模型输出张量;`dxpostprocess` 必须使用相同的 ID |
| `model_path` | 编译后的 `.dxnn` 模型路径;相对路径相对于进程工作目录解析,因此单元格写入的是绝对路径 |
| `backend` | 选择 `auto`、`dxrt` 或已编译的可选后端 |

在包含多个预处理、推理或后处理元素的流水线中,使用显式 ID 非常重要。

<!-- cell: 1341a38d-276d-4ac6-85e3-c454925cb1a9 src: ff1c547c4c -->
### 5.9 获取测试视频

与教程 03 使用的同一段片段;存在时从该工作区复制,否则下载(19 MB)。

<!-- cell: ccf95200-0466-4b6c-91fb-5dcecd0bc6cf src: f4aec25b4e -->
### 5.10 运行自定义流水线

各 ID 构成如下关系:

```text
dxpreprocess(preprocess-id=1)
        ↓
dxinfer(preprocess_id=1, inference_id=1)
        ↓
dxpostprocess(inference-id=1)
```

`library-file-path` 指向 5.7 安装的库。下面的路径是默认安装前缀;如果 DX-STREAM 是用 `./build.sh --prefix=<dir>` 构建的,请使用 `<dir>/share/gstdxstream/lib/libpostprocess_yolov7.so`。输出发送到 `SINK`:在桌面会话中为窗口,否则为 `fakesink`。

<!-- cell: 83d74722 src: 77c4166080 -->
## 6. 构建 YOLO26s 人数统计应用

本练习首先用 Python 实现人数统计,以便每个集成点都易于检查。它运行 YOLO26s 目标检测,读取附加在每个视频帧上的检测元数据,统计 COCO 类别 `0`(`person`)的数量,并更新文本叠加层。

完成本节后,您将理解:

- `DXObjectMeta` 在何处定义和创建,
- 检测元数据如何随 `GstBuffer` 一起传递,
- `pydxs` 如何把 C++ 元数据暴露给 Python,
- 一个 Python 字符串如何变成正在运行的 GStreamer 流水线,以及
- 为什么 pad 探针回调和显示更新使用不同的执行上下文。

<!-- cell: 08a854bd src: 2272569272 -->
### 6.1 准备 YOLO26s 模型和视频

`yolo26-s_640x640.dxnn` 是教程 01 第 3.3 节下载到共享工作区的模型之一,因此这里直接复用;只有在缺失时,单元格才会用 DX-APP 的设置脚本下载它。选择 DX-STREAM 示例中的滑雪板视频,是因为它包含多个人物。

<!-- cell: 254abcd0 src: 269af0cb87 -->
### 6.2 理解检测元数据

DX-STREAM 把像素和推理结果放在一起,而不把检测数据写入图像本身。一个解码后的视频帧由一个 `GstBuffer` 承载;DX-STREAM 为该缓冲区附加一个 `DXFrameMeta`,帧元数据拥有一个已检测目标的列表。

```text
GstBuffer — one video frame moving through the pipeline
│
├── Video memory / pixels
│
└── DXFrameMeta — frame-level GstMeta
    ├── stream_id, width, height, format, frame_rate, ROI, ...
    │
    └── object_meta_list
        ├── DXObjectMeta #0
        │   ├── label          = 0
        │   ├── label_name     = "person"
        │   ├── confidence     = 0.92
        │   └── box            = [x1, y1, x2, y2]
        ├── DXObjectMeta #1
        └── ...
```

这一关系很重要:Python 回调不会再次运行推理,也不会解析图像。它只读取 `dxpostprocess` 已经生成的目标元数据。

<!-- cell: d59665ef src: cd784d857f -->
#### 6.2.1 `DXObjectMeta` 在何处定义和创建

SDK 在以下相对于 `DX_STREAM_DIR` 的头文件中定义这些结构:

| 用途 | SDK 源文件 |
|---|---|
| 帧元数据及其目标列表 | `gst-dxstream-plugin/metadata/gst-dxframemeta.hpp` |
| 每个目标的检测元数据 | `gst-dxstream-plugin/metadata/gst-dxobjectmeta.hpp` |
| 两种结构的 Python 绑定 | `bindings/python/pydxs/src/metadata_binding.cpp` |

对于主目标检测,`dxpostprocess` 调用配置的 C++ `PostProcess` 函数。该函数按以下生命周期把 YOLO 张量输出转换为目标:

```cpp
DXObjectMeta* object = dx_acquire_obj_meta_from_pool();
object->_label       = class_id;
object->_label_name  = class_name;
object->_confidence  = score;
object->_box         = {x1, y1, x2, y2};
dx_add_obj_meta_to_frame(frame_meta, object);
```

结果被附加到当前帧,因此下游元素看到的是同一组检测结果:

```text
dxinfer                 dxpostprocess                         downstream
tensor output ────────▶ create DXObjectMeta ────────┬──────▶ dxosd draws boxes
                                                    └──────▶ Python probe counts people
```

DX-STREAM 随缓冲区一起管理元数据的生命周期。如果某个元素创建新缓冲区并复制 `DXFrameMeta`,其目标元数据也会一并复制。尽管如此,应用代码仍应只在处理当前缓冲区回调期间把 Python 元数据引用视为有效;不要保存它以供日后使用。

<!-- cell: b9bcdd40 src: d486fcc937 -->
#### 6.2.2 C++ 元数据如何呈现在 Python 中

`pydxs` 把 C++ 成员映射为对 Python 友好的属性:

| C++ 字段 | Python 属性 | 在本练习中的含义 |
|---|---|---|
| `DXFrameMeta::_object_meta_list` | `frame_meta.object_meta_list` 或对 `frame_meta` 进行迭代 | 当前帧中的所有检测结果 |
| `DXObjectMeta::_label` | `obj_meta.label` | COCO 类别 ID;`0` 表示 `person` |
| `DXObjectMeta::_label_name` | `obj_meta.label_name` | 人类可读的类别名称 |
| `DXObjectMeta::_confidence` | `obj_meta.confidence` | 检测置信度 |
| `DXObjectMeta::_box` | `obj_meta.box` | 帧坐标系下的 `[left, top, right, bottom]` |
| `DXObjectMeta::_track_id` | `obj_meta.track_id` | 跟踪器 ID;本流水线中未分配 |

回调接收的是即将离开 `dxpostprocess` 的同一个 `Gst.Buffer`:

```python
buffer = info.get_buffer()
frame_meta = pydxs.dx_get_frame_meta(hash(buffer))

people_count = sum(
    1 for obj_meta in frame_meta
    if obj_meta.label == PERSON_CLASS_ID
)
```

`hash(buffer)` 提供绑定所需的原生 `GstBuffer` 地址。`frame_meta` 可迭代,是因为 `pydxs` 通过 Python 的迭代器协议暴露了 `_object_meta_list`。

因此,计数就是**当前帧**中被接受的 `person` 检测框的数量。Python 代码不会再应用另一个置信度阈值。如果需要置信度过滤和重复抑制,必须在后处理器中、在目标被附加之前完成。这不是独立访客计数;要随时间维持身份,需要使用跟踪。

<!-- cell: b962bea0 src: 2f6d8fbc24 -->
### 6.3 理解 Python 如何嵌入 GStreamer 流水线

该应用并不以子进程方式调用 `gst-launch-1.0`。相反,`pipeline_description` 以 Python 格式化字符串的形式包含同样的 GStreamer launch 语法,`Gst.parse_launch()` 把这段文本转换为通过 pad 连接的真实 GStreamer 元素。

```text
Python application
│
├── pipeline_description = f''' ... '''
│       │
│       └── Gst.parse_launch(description)
│                    │
│                    ▼
│    ┌────────┐  ┌──────────┐  ┌───────┐  ┌─────────────┐  ┌───────┐
└───▶│ source │─▶│preprocess│─▶│ infer │─▶│ postprocess │─▶│  osd  │─▶ display
     └────────┘  └──────────┘  └───────┘  └──────┬──────┘  └───────┘
                                                  │
                                    source-pad BUFFER probe
                                                  │
                                                  ▼
                                    read metadata → count people
                                                  │
                                                  ▼
                                      update `textoverlay` text
```

探针只是观察缓冲区;它不是另一条流水线分支,也不会复制帧。

| 流水线组成部分 | 职责 |
|---|---|
| `urisourcebin ! decodebin` | 读取视频并解码压缩帧 |
| `dxpreprocess` | 调整帧大小并为模型准备输入 |
| `dxinfer` | 运行 YOLO26s 并附加其输出张量 |
| `dxpostprocess name=detector_postprocess` | 解码张量并附加 `DXObjectMeta` 条目 |
| `dxosd` | 读取目标元数据并绘制边界框和标签 |
| `videoconvert` | 为标准的叠加/显示路径准备视频格式 |
| `textoverlay name=people_overlay` | 显示由 Python 控制的计数 |
| `fpsdisplaysink` | 呈现帧并测量显示 FPS |

<!-- cell: 98bc52b7 src: aaf166a292 -->
#### 6.3.1 从流水线字符串到具名的 Python 对象

三行代码把流水线文本与应用逻辑连接起来:

```python
self.pipeline = Gst.parse_launch(pipeline_description)
self.people_overlay = self.pipeline.get_by_name("people_overlay")
postprocess = self.pipeline.get_by_name("detector_postprocess")
```

这些名称来自 launch 字符串中的属性:

```text
dxpostprocess name=detector_postprocess ...
textoverlay   name=people_overlay ...
```

随后,应用把一个回调附加到后处理器的 source pad 上:

```python
src_pad = postprocess.get_static_pad("src")
src_pad.add_probe(Gst.PadProbeType.BUFFER, self._postprocess_probe)
```

`dxpostprocess` 的每个输出缓冲区在继续流向 `dxosd` 之前都会调用 `_postprocess_probe`。返回 `Gst.PadProbeReturn.OK` 允许同一缓冲区继续向下游传递。

<!-- cell: af6d6eca src: 427e4a7436 -->
#### 6.3.2 流式回调与 GLib 主循环

GStreamer 从流式线程调用 pad 探针。缓慢的回调会延迟后续的每一帧,因此探针只获取元数据、统计目标数量并调度显示更新。

```text
GStreamer streaming thread                    GLib main-loop thread
──────────────────────────                    ─────────────────────
buffer reaches postprocess.src
          │
          ▼
_postprocess_probe()
  ├── get GstBuffer
  ├── get DXFrameMeta
  ├── count label == 0
  └── GLib.idle_add(_update_overlays, count) ───────────────┐
          │                                                 ▼
          └── return OK                           _update_overlays()
                    │                               └── set text property
                    ▼
             buffer continues                     UI remains responsive
```

总线也会把流水线范围的事件报告给主循环:

```text
GStreamer bus ── ERROR ──▶ print details and stop
              └─ EOS   ──▶ stop the main loop
```

`pipeline.set_state(Gst.State.PLAYING)` 启动数据流,而 `GLib.MainLoop().run()` 让 Python 应用保持运行,以便它能处理空闲回调和总线消息。

<!-- cell: 6ab93ab6 src: bebe7b0bf7 -->
### 6.4 创建 Python 应用

下面的完整应用把上文描述的流水线、元数据探针、叠加层更新、总线处理和命令行参数组合在一起。

<!-- cell: 13351f19 src: 0781fa93bf -->
### 6.5 验证 DX-STREAM Python 环境

该应用必须使用 DX-STREAM 虚拟环境,因为它提供了 `pydxs` 模块和 GStreamer 的 Python 绑定。

<!-- cell: e6d0e88f src: 89e064ba44 -->
### 6.6 运行应用

在桌面会话中(`HEADLESS = False`),应用会打开一个窗口,显示 YOLO 边界框和 `People: N` 叠加层,并一直运行到视频结束或按下停止按钮。没有显示器时(`HEADLESS = True`),它以 `--no-display` 运行:流水线、元数据探针和计数仍然运行,帧发送到 `fakesink`,单元格在 EOS 时以一行摘要结束。

该应用必须使用 `venv-dx_stream/bin/python`,而不是 Jupyter 内核,因为 `pydxs` 和 GStreamer Python 绑定位于 DX-STREAM 环境中。在终端中:

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_stream
./venv-dx_stream/bin/python <workspace>/yolo26_people_counter.py \
    --model <DX_ALL_SUITE_DIR>/workspace/res/models/yolo26-s_640x640.dxnn \
    --video dx_stream/samples/videos/snowboard.mp4
```

<!-- cell: 5c93c78a src: 946e670e83 -->
### 6.7 预期结果与扩展点

常规的 YOLO 边界框和标签仍通过 `dxosd` 可见。一个标准的 GStreamer 文本叠加层添加了应用状态:

- 右上角:`People: N`,每当计数变化时更新。

显示的数字应与该帧中被接受的 `person` 框数量一致。如果一个人产生了重叠的边界框,请更改后处理器的置信度/NMS 策略,而不是在 Python 叠加层中修正数字。

对于生产环境的占用率系统,请添加 `dxtracker`、定义进入/离开区域,并统计稳定的跟踪 ID,而不是原始的逐帧检测结果。

<!-- cell: e899a740-4b78-4712-8073-a1904d5b76b2 src: 0a5e5189de -->
## 7. 调试

### 7.1 使用有针对性的 GStreamer 日志

先用级别 3 查看初始化和状态变化,然后仅对正在排查的元素启用级别 4 或 5。

```bash
# All DX-STREAM elements at INFO level
GST_DEBUG=dx*:3 ./run_demo.sh

# Inference and tracker details
GST_DEBUG=dxinfer:4,dxtracker:4 ./run_demo.sh

# Metadata lifecycle
GST_DEBUG=dxmeta:5 ./run_demo.sh

# Save logs to a file
GST_DEBUG=2,dxinfer:4 GST_DEBUG_FILE=/tmp/dxstream.log ./run_demo.sh
```

在进行性能测量时请关闭详细日志,因为日志输出会影响时序。

<!-- cell: 13445965-9c30-429f-80ff-3acf97fd6cee src: 771d6ca90f -->
### 7.2 理解 v3.1.x 的失败行为

- 不兼容的链接会在 caps 协商阶段失败。
- 模型文件缺失、NPU 初始化失败和自定义库错误会作为 GStreamer 元素错误报告,而不是以 `abort()` 终止。
- 处理元素会把它们测得的时间贡献给 LATENCY 查询。
- 内部状态在 FLUSH 时重置,改善了定位和重放行为。
- EOS、FLUSH 和状态转换能更可靠地清理工作线程,包括单帧输入的情况。

当多流流水线无法链接时,请先检查是否把标准的 `video/x-raw` 元素放到了 `application/x-dxvideoraw` 域内部。

<!-- cell: a340afc0-dfc5-471d-98aa-6c7e8e2e0f53 src: af81fae94c -->
### 7.3 导出流水线图

设置了 `GST_DEBUG_DUMP_DOT_DIR` 后,GStreamer 会在每次流水线状态变化时写入一个 DOT 文件。Graphviz 可以把 `PAUSED_PLAYING` 文件转换为真实元素图的图片。如有需要,安装一次即可:

```bash
sudo apt install graphviz
```

下一个单元格为内核设置该变量,并再次运行 3.1 的快速入门流水线(按配置以无头方式或带窗口运行),这样 DOT 文件就会落在教程工作区中。等效的终端命令:

```bash
GST_DEBUG_DUMP_DOT_DIR=<workspace>/dot gst-launch-1.0 filesrc location=... ! ... ! fakesink
```

<!-- cell: af463f4c-5ea2-4477-b67c-fcc4cad871b7 src: f4e26083b6 -->
### 7.4 DX-STREAM v3.1.x 中的变化

本教程使用 DX-STREAM v3.1.2 进行了验证。v3.1.x 系列引入了本教程所依赖的运行时行为:

- 基于 `application/x-dxvideoraw` 的统一多流域
- 在共享处理链中保留每个流的生命周期和时间线
- `dxinfer` 后端抽象和异步 Put/Get 处理
- 用于 Base64 编码 JPEG 帧的 `dxmsgconv include-frame`
- 更新后的自定义库和元数据 API(5.1 节)
- 改进的延迟报告、FLUSH 恢复、caps 协商和错误报告

可用的后端和元素属性取决于 DX-STREAM 的构建方式。请使用 `gst-inspect-1.0` 检查已安装的构建。

<!-- cell: 325d5d11-7b12-4ddb-83e2-f1cb7ce17a91 src: 9ef3c41d4e -->
## 8. 故障排除

| 症状 | 看到的内容 | 可能原因 | 解决方法 |
|---|---|---|---|
| 找不到 DX-STREAM 插件 | `DX-STREAM plugin not found` 或 `no element "dxinfer"` | 启动 JupyterLab 的环境中未设置 `GST_PLUGIN_PATH` | 在 `./run-jupyter-lab.sh` 之前导出 `GST_PLUGIN_PATH=/usr/local/lib/x86_64-linux-gnu/gstreamer-1.0`(`build.sh` 打印的路径),或用 `./build.sh` 重新构建 |
| 没有窗口出现 | `Could not open display` 或流水线保持沉默 | 没有图形会话 | 在桌面会话中运行;若只是快速检查,可把 `fpsdisplaysink` 替换为 `fakesink` |
| 摄像头流水线失败 | `v4l2src: Cannot open device` 或 caps 协商错误 | 设备路径错误或分辨率不受支持 | 运行 `v4l2-ctl --list-formats-ext`,并在单元格中编辑设备和 caps |
| RTSP 演示无法连接 | `Could not connect to server` | `run_RTSP.sh` 中的示例 RTSP URL 在您的网络中不可访问 | 编辑 `run_RTSP.sh`,填入可访问的流 |
| `build.sh` 停止 | 单元格内出现 `sudo` 密码提示或 `meson` 错误 | 构建会进行系统级安装,需要 `sudo` | 在终端(File > New > Terminal)中运行 `./build.sh` |
| 某些帧完全是绿色 | 窗口或保存的帧间歇性地闪现绿色 | 显示 sink 之前的 `dxconvert` 错误处理了某些解码后的缓冲区 | 使用 `videoconvert ! fpsdisplaysink`(笔记本默认值);仅在 3.3 的 `dxscale`/`dxconvert` 实验中保留 `dxconvert` |
| 未生成 DOT 图 | `No DOT file was generated` | 演示在流水线进入 PLAYING 之前就被停止了 | 让视频开始播放,然后再停止单元格 |
| 示例模型在加载时被拒绝 | 来自 `dxinfer` 的 DX-RT 版本或格式错误 | 示例模型是为 Model Zoo 发布版 `2_4_0` 编译的;较旧的运行时无法加载它们 | 使用本教程验证所用的 DX-RT 版本(3.4.2),或用您的 DX-COM 重新编译模型 |
| `setup.sh` 要求输入密码 | 安装 `jq` 时出现 `[sudo] password for ...` | 缺少 `jq`,脚本用 `apt` 安装它 | 在终端中运行一次 `sudo apt install jq`,然后重新运行单元格 |
| `import pydxs` 失败 | `ModuleNotFoundError: No module named 'pydxs'` | DX-STREAM 尚未构建,或使用了 Jupyter 的 Python 而不是 `venv-dx_stream` | 运行 5.7(构建),然后像单元格那样使用 `venv-dx_stream/bin/python` |
| `build.sh` 立即失败 | `Build data file ... was generated with an old version of meson` | 上一个 DX-STREAM 版本遗留的 `builddir` | 运行 `./build.sh --clean`(5.7 中的单元格在首次尝试失败时会自动执行此操作) |

<!-- cell: 51fafd7a-3597-4c54-a355-0cad94216211 src: de9bc8a30a -->
## 9. 总结

您现在已经:

- 构建并检查了一条端到端的 DX-STREAM 推理流水线
- 运行了捆绑的目标检测、PPU、人脸检测、姿态、分割、跟踪、多通道、RTSP 和 Secondary Mode 演示,并阅读了它们背后的脚本
- 检查了推理后端、`dxscale`、`dxconvert` 以及 MQTT/Kafka 消息代理脚本(代理演示需要运行中的代理,且不属于菜单)
- 学习了 `application/x-dxvideoraw` 多流域及其元素放置规则
- 为双类别的叉车与工人模型适配了 YOLOv7 后处理器和推理配置
- 跟踪了检测结果从 `GstBuffer` 经 `DXFrameMeta` 和 `DXObjectMeta`、再通过 `pydxs` 进入 Python 的全过程
- 在 Python 中嵌入了 GStreamer 流水线,并结合 GLib 主循环使用 pad 探针显示逐帧人数
- 使用有针对性的 GStreamer 日志和 DOT 图诊断了流水线行为

### 9.1 完成清单

- [ ] 用 `gst-inspect-1.0` 确认 13 个 DX-STREAM 元素
- [ ] 运行快速入门的 YOLO26n 流水线以及至少三个捆绑演示
- [ ] 解释多流域的放置规则
- [ ] 打补丁、重新构建并运行双类别的叉车与工人后处理器
- [ ] 运行 Python 人数统计应用,并通过 `pydxs` 读取 `DXObjectMeta`
- [ ] 导出并打开流水线 DOT 图

> **下一步:** 继续学习教程 05,深入了解 DX-Compiler 工作流:ONNX 验证、校准、PPU、图优化和量化策略。
