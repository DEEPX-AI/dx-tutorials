<!-- i18n source: notebooks/T01-Getting-Started/getting_started.ipynb -->
<!-- i18n lang: zh -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 7561e296-41b4-44c3-ab4f-4c8582f0593f src: 14f392f9fc -->
# DEEPX Tutorial 01 - 如何安装 DEEPX SDK
第一篇教程介绍如何安装 DEEPX SDK 并验证安装是否成功。您将学习如何准备环境、安装 SDK,并确认系统已正确识别 DEEPX NPU 设备(DX-M1、DX-M1M 和 DX-H1 Quattro)。

## 学习目标

完成本教程后,您将能够成功安装 DX-All Suite,并在 DEEPX NPU 设备上运行一个基本流程。

<!-- cell: 389418ec-fa11-4a60-9822-183b9d441952 src: 32aa4d182a -->
## 前提条件

<!-- cell: 9c427531-ff8d-4e74-93db-8f41d1237db6 src: 2c603254b1 -->
**注意:** 以下要求仅针对本教程,**并非使用 DEEPX 产品的必要条件。**
- 操作系统: Linux (Ubuntu 20.04/22.04/24.04/26.04, Debian 12/13)
- 内存: 8G(使用 DX-Compiler 时需 16G)
- 存储: 至少 40 GB
- DEEPX NPU: DX-M1、DX-M1M 和 DX-H1 Quattro
- CPU: x86_64 上支持 DX-Compiler + DX-Runtime,aarch64 上仅支持 DX-Runtime

<!-- cell: 2dcfff26-61ce-46f1-b08f-5669e3ead9a8 src: d3372f2340 -->
## DXNN® - DEEPX NPU SDK 简介 (DX-AS: DX-All Suite)

<!-- cell: 52b3f559-cb24-4e41-8286-6802b4d96c4d src: 99488f0f51 -->
DX-AS(DX-All Suite)是一个集成了框架与工具的环境,用于在 DEEPX 设备上编译和推理 AI 模型。用户也可以逐个安装工具来搭建环境,但 DX-AS 通过统一各工具的版本来保持最佳兼容性。

![](https://github.com/DEEPX-AI/dx-all-suite/raw/main/docs/source/resources/dxnn_sdk_illustration.png)

DEEPX SDK 主要分为两个关键部分。

第一部分是 AI 模型编译环境,它将您的 AI 模型转换为可在 DEEPX NPU 上高效执行的优化格式。

第二部分是 AI 模型运行时环境,它在实际的 DEEPX NPU 硬件上执行编译后的 AI 模型并生成结果。

借助 DX-All Suite,您无需分别管理这两个组件即可一次性完成配置。

![](https://github.com/DEEPX-AI/dx-all-suite/raw/main/docs/source/img/dx-as.png)


![](https://github.com/DEEPX-AI/dx-all-suite/raw/main/docs/source/img/DXNN-SDK-Simple-Architecture.png)



为便于理解,这里有两段关于 DX-SDK 的 YouTube 视频:
- [Youtube - DEEPX SDK Introduction](https://www.youtube.com/watch?v=Js6Soex0WI4) | 下载: [EN](https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/1.DXNN_Introduce_ENG.mp4) · [中文](https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/1.DXNN_Introduce_CH.mp4) · [한국어](https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/1.DXNN_Introduce_KO.mp4)

<img src="assets/youtube-dx-sdk.png" style="max-width: 1000px;">

<!-- cell: 2af4c675-fced-4ac2-ab69-05a2a8abbeb9 src: accfc71fa0 -->
## 1. 下载 DX-All Suite

<!-- cell: 3b0789e3-bf0c-4304-8545-bed41a13b3be src: c90ee8eafd -->
### 1.1 配置安装

教程按以下顺序查找 DX-All Suite:

1. 环境变量 `DX_ALL_SUITE_DIR`,
2. `dx-tutorials/config.json`(由本节创建,不被 git 跟踪),
3. 自动检测常见位置,例如 `~/dx-all-suite` 以及与 `dx-tutorials` 同级的 `dx-all-suite` 文件夹,
4. 默认位置 `~/dx-all-suite`。

下一个单元格会打印找到的位置以及该值的来源。如果 SDK 尚未安装,默认位置会显示为 `MISSING`;在克隆步骤之前这是正常的。

<!-- cell: dx-all-suite-directory-heading src: b988a5d1c2 -->
#### 1.1.1 安装目录

若要使用上面打印的位置,请保持下一个单元格不变。只有当您想把 DX-All Suite 安装到其他位置(或已经安装在其他位置)时,才需要修改 `DX_ALL_SUITE_DIR`。运行该单元格会把选择保存到 `dx-tutorials/config.json`,这样其他所有教程都会使用同一位置。

您也可以随时在终端中设置(在 `dx-tutorials` 目录下运行):

```bash
python tutorial_paths.py --set ~/my/dx-all-suite
```

<!-- cell: dx-all-suite-branch-heading src: 635edf5993 -->
#### 1.1.2 Git 分支

只有在需要其他 SDK 版本时才修改下面的分支名。
当前分支名为 `main`。

<!-- cell: 8d2496bb-a3a2-460c-a891-1979b3cfab1d src: 187f9b06f0 -->
### 1.2 克隆 DX-All Suite

下一个单元格**仅当所配置的位置不存在 DX-All Suite 时**才会运行以下命令。目录和分支来自您上面保存的设置,在单元格中显示为 `{DX_ALL_SUITE_DIR}` 和 `{DX_ALL_SUITE_BRANCH}`。

```bash
git clone --depth 1 --shallow-submodules --recurse-submodules --progress \
    --branch main https://github.com/DEEPX-AI/dx-all-suite.git ~/dx-all-suite
```

- `--depth 1 --shallow-submodules` 只获取每个仓库的最新提交:下载约 600 MB、占用磁盘 1.6 GB,而不是数 GB。
- `--recurse-submodules` 同时检出子模块(`dx-runtime`、`dx-compiler`、`dx-modelzoo` 及其内部的仓库)。
- `--progress` 即使输出不是终端也会打印进度,因此您可以在单元格中查看进度。

根据网络情况预计需要 1 到 10 分钟。您可以用 ■ 按钮停止单元格,稍后再运行。完成时最后几行会显示 `Submodule path ... checked out`。

如果上一次克隆在顶层检出之后被中断,目录中已有 `.git`,但部分子模块为空。此时单元格不会重新克隆,而是运行以下命令:

```bash
git -C ~/dx-all-suite submodule update --init --recursive --depth 1 --progress
```

您也可以不使用单元格,而是在终端(**File > New > Terminal**)中自行运行上述任一命令。下面的状态单元格在两种情况下都同样有效。

<!-- cell: b6993bc2-9b2c-48ec-995b-4f1777a8844c src: 034c0df327 -->
### 1.3 当前状态与后续步骤

下一个单元格显示目前已安装的内容,并针对每个标记为 `MISSING` 的项目指出提供它的具体步骤。每次回到本笔记本时都可以再次运行它;重复运行是安全的。

<!-- cell: ee6ce747-f96c-417f-917f-9cccd54be3ec src: 64b774a178 -->
### 1.4 DX-All Suite 包含的内容

克隆下来的仓库包含三个 Git 子模块:
 - `dx-compiler`(**DX-Compiler**): 将 ONNX 模型转换为 DXNN。在第 2 节安装。
 - `dx-runtime`(**DX-Runtime**): 在 DEEPX NPU 上运行编译后的模型。在第 3 节安装。
 - `dx-modelzoo`: 后续教程使用的预编译模型和编译器配置。

<!-- cell: 2ab0cf89-7974-40e6-8117-18165d1ef10f src: 3d336a7ce6 -->
DX-Runtime 提供:
 - `DX-APP`: 从用户应用的角度展示如何使用 DX NPU 的 DEEPX 应用模板
 - `DX-FW`: NPU 固件二进制文件
 - `DX-RT`: 为使用 DX NPU 优化执行推理任务而设计的框架
 - `DX-NPU Driver`: DX NPU 的 Linux 内核驱动
 - `DX-STREAM`: 面向 DX NPU 的基于 GStreamer 的视觉 AI 应用开发工具

<!-- cell: 912db406-4064-4ea1-b075-c455abeab01f src: b20e1b138c -->
## 2. 安装 DX-Compiler

<!-- cell: c342f780-67f2-4914-a52d-321b5fe55b72 src: 7b700f5a1e -->
更多细节请参阅 [DX-All Suite 安装指南](https://github.com/DEEPX-AI/dx-all-suite/blob/main/docs/source/02_Setting_Up_Environment.md)。

DX-Compiler 环境提供预构建的二进制文件,不包含源代码。安装程序会从远程服务器下载各个模块:

| 命令 | 安装内容 | `sudo` |
|---|---|---|
| `./dx-compiler/install.sh --target=dx_com` | 安装在独立 Python 环境中的 DX-COM 编译器(`dxcom`) | 即使已存在也会运行 `apt-get install python3-dev python3-venv` |
| `./dx-compiler/install.sh --target=dx_tron` | DX-TRON 模型查看器(`.deb` 包) | 使用 `apt-get` 安装该包 |
| `./dx-compiler/install.sh` | 两者 | 以上两项 |

本教程在 2.1 安装 DX-COM。DX-TRON 为可选项,在 2.3 介绍。

<!-- cell: 3bdb9b57-de73-4186-bac9-ac1effa245da src: 814d4a60eb -->
### 2.1 安装 DX-COM

要运行的命令是:

```bash
cd ~/dx-all-suite
./dx-compiler/install.sh --target=dx_com
```

由于 `run-jupyter-lab.sh` 启动 JupyterLab 时不会激活教程的 Python 环境,所以无论在笔记本单元格还是终端中运行,此命令看到的都是系统 `python3`。安装程序会在 `dx-compiler/venv-dx-compiler-local` 创建自己的环境,不会改动 Jupyter 的环境。

唯一决定*在哪里*运行的因素是 `sudo`:安装程序总是会为 `python3-dev` 和 `python3-venv` 调用 `sudo apt-get`,而笔记本单元格无法响应密码提示。因此下面的三个单元格分别:

1. **检查** `dxcom` 是否已安装,以及此处的 `sudo` 是否可以免密运行,
2. **安装**:运行上述命令(仅当检查单元格提示可以运行时才运行此单元格),
3. **验证** `dxcom` 是否存在。

下载量为几百 MB,预计需要 2 到 10 分钟。输出末尾会出现绿色的 `[HINT] dx_com installation completed!` 块,下文附有参考。

<!-- cell: 891804c8-7521-4f34-a453-f8261151d1f4 src: ae07f58239 -->
安装成功完成后,安装程序会打印以下提示块。它创建的虚拟环境 `dx-compiler/venv-dx-compiler-local` 与 Jupyter 环境保持独立。2.2 中的单元格在每个 `!` 行的 shell 内用 `source venv-dx-compiler-local/bin/activate` 激活它,这不会改变 Jupyter 环境。

<pre style="color: green">
[HINT] ==================================================================== 
[HINT]   dx_com installation completed! 
[HINT]  
[HINT]   To use dx_com, activate the virtual environment first: 
[HINT]     $ source <path_to_dx_all_suite>/dx-compiler/venv-dx-compiler-local/bin/activate 
[HINT]  
[HINT]   Then you can run dxcom: 
[HINT]     $ dxcom -h 
[HINT]  
[HINT] ==================================================================== 
</pre>

接下来,验证 DX-Compiler 安装的组件:

<!-- cell: 63c15643-8cfc-4a6f-a6ee-67451ef0b5fb src: c2d63e41ac -->
现在,让我们看看 dx_com 下的文件和文件夹:

<!-- cell: cffd0d41-fa76-48c4-af02-6a07dcf47057 src: d96c946dd1 -->
### 2.2 验证 DX-Compiler

2.1 中的验证单元格已经确认 `dxcom` 存在。本节实际运行它:先输出帮助信息,然后真正编译一个示例模型。

<!-- cell: 1803112e-39d6-4066-8abc-c1c2d61fdfbc src: dbf5bd0d1f -->
#### 2.2.1 查看 `dxcom` 帮助

`dxcom` 位于安装程序创建的虚拟环境中,因此在终端里调用前必须先激活该环境。下一个单元格精确地运行以下命令:

```bash
cd ~/dx-all-suite/dx-compiler
source venv-dx-compiler-local/bin/activate
dxcom -h
```

笔记本中的每个 `!` 行都会启动一个新的 shell,因此这三条命令用 `&&` 连接在一行中。激活仅对该 shell 有效,Jupyter 环境不会改变。

<!-- cell: 9cf00564-56e7-42cf-bbf6-873292fee8da src: bf66ec5280 -->
#### 2.2.2 编译 `MobileNetV2-1.onnx` 生成 `MobileNetV2-1.dxnn`

下一个单元格运行以下命令。编译器会打印每个阶段,`--gen_log` 还会把它们写入 `compiler.log`。耗时从几秒到大约一分钟。

```bash
cd ~/dx-all-suite/dx-compiler
source venv-dx-compiler-local/bin/activate
cd dx_com
dxcom -m sample_models/onnx/MobileNetV2-1.onnx \
      -c sample_models/json/MobileNetV2-1.json \
      -o output/MobileNetV2-1 \
      --gen_log
```

<!-- cell: 8233469e-d738-4c1b-84ae-a451a733a87a src: e5dd23e1a4 -->
检查是否已生成 `MobileNetV2-1.dxnn` 文件:

<!-- cell: 3b83c7c6-a23c-4fd3-ba83-890dfa2b3584 src: adf789f4ad -->
`--gen_log` 选项会把编译日志输出到名为 `compiler.log` 的文件中。

让我们看看保存的日志信息:

<!-- cell: d93f93a6-c5f7-476e-9632-695968e6cc3c src: 2e60c3c391 -->
### 2.3 DX-Tron

**DX-TRON** 是一个图形化可视化工具,用于查看由 DEEPX 工具链编译的 `.dxnn` 模型文件。 

它允许用户加载并检查模型结构,通过彩色编码的图查看 NPU 与 CPU 之间的工作负载分布。 

借助 DX-TRON,用户可以更好地理解模型执行流程并提升整体性能。

> **注意:** DX-TRON 仍随 SDK 一起发布,但已不再积极维护。对于编译后的模型,推荐使用 `dxcom --export_html` 生成的 HTML 摘要(教程 05)作为报告;DX-TRON 仍适合快速直观地查看 `.dxnn` 文件。

<img src="assets/sc-dxtron.png" style="max-width: 600px;">

**主要功能:**
- **支持 .dxnn 文件**: 加载并可视化由 DEEPX 工具链编译的模型文件。
- **工作负载可视化**: 以颜色区分显示工作负载的执行情况:
- 红色: 在 NPU 上执行的算子
- 蓝色: 在 CPU 或主机上执行的算子
- **模型导航控件**: 随时使用左下角的后退箭头返回模型概览界面。
- **交互式节点检查**: 双击图中的任意节点可查看相关算子的详细信息。
- 说明: DX-TRON 基于 [netron](https://netron.app/) 开发,以支持 DXNN。

<!-- cell: cdc69460-e92c-460b-aaa5-dc0651eba2dc src: 9f6e1afb38 -->
#### 2.3.1 安装 DX-TRON

DX-TRON 以 Debian 包的形式发布,因此安装程序使用 `sudo apt-get`:

```bash
cd ~/dx-all-suite
./dx-compiler/install.sh --target=dx_tron
```

与 2.1 一样,后面有三个单元格:**检查**(`dxtron` 是否已安装、`sudo` 能否免密运行)、**安装**(上述命令;仅当检查单元格提示可以运行时才运行,或者在终端中运行)、**验证**。

<!-- cell: 061c3fe9-86fa-4dc3-aa5a-bf48c2365459 src: eb9122c32f -->
#### 2.3.2 运行 DX-TRON

下一个单元格在 DX-TRON 中打开 2.2.2 编译的 MobileNetV2 模型。窗口打开期间,单元格保持运行状态。
> **注意:** 点击上方的停止按钮('■')即可停止 `dxtron`!

<!-- cell: e8158e1f-0476-46b1-b72a-94006d838653 src: bcd38bb9d2 -->
## 3. 安装 DX-Runtime
更多细节请参阅 [DX-All Suite 安装指南](https://github.com/DEEPX-AI/dx-all-suite/blob/main/docs/source/02_Setting_Up_Environment.md)。

<!-- cell: 1dee6063-b928-4c16-8100-bbde4cc779db src: a530d80148 -->
### 3.1 (可选)安装前的前提条件(仅限 `Orangepi-5 plus`)
 - 如果使用 Orange Pi 官方镜像,可能未安装内核头文件。安装 NPU 驱动需要内核头文件。
 - 请参阅[此处](../../docs/orangepi5p.md)链接的文档。

### 3.2 (可选)安装前的前提条件(仅限 `Raspberrypi-5`)
 - PCIe 默认配置为 Gen2。可以设置为 Gen3 以提高带宽。 
 - 请参阅[此处](../../docs/raspberrypi5.md)链接的文档。 

<!-- cell: 5a923f2b-8ed3-454b-b171-6162b87349ed src: d59a17ec7e -->
DX-Runtime 环境包含各模块的源代码。这些仓库作为 Git 子模块(`dx_rt_npu_linux_driver`、`dx_fw`、`dx_rt`、`dx_app` 和 `dx_stream`)管理在 `./dx-runtime` 下。

让我们看看 DX-Runtime 安装脚本的所有选项:

<!-- cell: f46c6f59-c240-40ef-beef-c391b6b02ad7 src: bdef7c465b -->
### 3.1 安装 DX-Runtime

后续教程会使用 DX-APP 和 DX-STREAM,因此使用 `--all` 安装全部内容:

```bash
cd ~/dx-all-suite
./dx-runtime/install.sh --all
```

如果只需要核心运行时(NPU 驱动、固件和 DX-RT),请用 `--runtime-only` 代替 `--all`。

安装程序全程使用 `sudo`:它会构建并加载 NPU 内核驱动、安装 Debian 包,并注册 `dxrt.service` 守护进程。它不会提问。与 2.1 一样,后面有三个单元格:**检查**(是否已安装、`sudo` 能否免密运行)、**安装**(上述命令;仅当检查单元格提示可以运行时才运行,或者在终端中运行)、**验证**。预计需要 10 到 30 分钟,主要用于构建 DX-RT、DX-APP 和 DX-STREAM。

安装 NPU 驱动后需要**重启**,固件更新后安装程序建议完全断电。重启后,再次启动 `./run-jupyter-lab.sh`,打开本笔记本,运行第一个代码单元格,然后从 3.2 继续。

<!-- cell: 379be8a6-b287-462b-a2d7-3444b12b2b06 src: e04e936876 -->
### 3.2 验证安装

配置单元格已经把安装目录和 Git 分支保存到了 `dx-tutorials/config.json`。本验证检查仓库、分支、子模块、DX-Compiler 环境、DX-Runtime CLI 以及 NPU 设备节点。之后的单元格会更详细地检查 PCIe 链路、内核驱动和服务。

<!-- cell: 8019bed5-064a-4d2f-b1c6-28e4608c977e src: b7b11a0e49 -->
### 3.3 推荐:仅下载选定的模型

下一个代码单元格使用 `SELECTED_MODELS` 作为允许列表,只下载这些模型。这是本教程推荐的默认方式。如果需要其他模型集合,请在运行单元格前编辑该列表。

> **警告 — 除非确有必要,否则不要使用全量下载命令。**  
> 运行 `!cd $DX_ALL_SUITE_DIR/dx-runtime/dx_app && bash setup.sh <<< ""` 会向交互式提示提供空答案。空答案会选中所有类别和所有模型,因此会下载全部 349 个模型。完整集合需要大量网络流量和约 29 GB 的存储空间。

推荐的单元格把 `SELECTED_MODELS` 传给 `setup.sh --models`,并使用 `--no-force`,这样已有的模型文件不会被重复下载。列出的 24 个模型是教程 02 到 24 使用的模型;根据网络情况,下载需要几分钟。

<!-- cell: 6549dfd3-3e52-45ca-b3a7-29a0c3a23134 src: de74293afb -->
### 3.4 DX-RT 提供的实用工具

<!-- cell: f2fe6a42-d9fc-426f-9ec3-06b64fd9180b src: a05bf69c17 -->
让我们看看 DX-RT 支持的 CLI 工具。

<!-- cell: dxrt-cli-compatibility-heading src: eda0521192 -->
**当前 CLI 名称与向后兼容名称**

DX-RT 把当前的 CLI 二进制文件安装在 `/usr/local/bin` 下。旧的命令名作为指向当前二进制文件的符号链接仍然可用:

| 当前名称 | 旧的兼容名称 | 实现 |
|---|---|---|
| `dxcli` | `dxrt-cli` | `dxrt-cli -> dxcli` |
| `dxrun` | `run_model` | `run_model -> dxrun` |
| `dxparse` | `parse_model` | `parse_model -> dxparse` |

因此两个名称执行的是同一个二进制文件,接受相同的选项。帮助文本可能显示调用时使用的名称。`dxbenchmark`、`dxtop` 和 `dxrtd` 保留现有名称,作为独立的二进制文件安装。

<!-- cell: 5ead4347-03ba-4d2a-8779-4333e6ddbe5d src: 688be0a7e3 -->
#### 3.4.1 dxbenchmark

`dxbenchmark` 是一个 CLI 工具,用于运行编译后的 .dxnn 模型以测试功能并测量性能。

<!-- cell: 34c9aeab-e537-4c77-a570-1848c07b9e03 src: 6a7b03d552 -->
打开网页浏览器,以 HTML 格式查看 `dxbenchmark` 的结果。

<!-- cell: ae8b23c7-05b3-432a-a55b-aa7fd57fcd05 src: 39068fd181 -->
#### 3.4.2 dxtop

`dxtop` 是一个类似 htop 的工具,用于实时监控 DEEPX NPU 指标,例如利用率、温度和内存。

> dxtop 的输出格式在 Jupyter Notebook 代码单元格中无法正确显示。请改为**打开单独的终端**运行 dxtop 命令。 </br>
> 如何打开单独的终端? `File > New > Terminal`
> 
> ![](assets/open-terminal.png)

<!-- cell: 8cacae4f-7187-4e97-9c82-bbd2d4605826 src: e3b55470f3 -->
> <img src="assets/sc-dxtop.png" style="max-width: 600px;">

<!-- cell: b8469434-2b9d-466a-a5d1-2ba415a678e2 src: 277f9a2a24 -->
#### 3.4.3 dxcli

`dxcli`(为兼容也可用 `dxrt-cli`)用于查询和监控 DEEPX DX-RT 设备,并管理 NPU 固件。

<!-- cell: fff91477-4f16-4cd2-b0bd-7962bacdfa4d src: c1380e457f -->
#### 可选:刷写 NPU 固件

DX-Runtime 安装程序已经写入了与此 SDK 版本匹配的固件,因此通常**不需要**此步骤。仅当 DEEPX 支持团队要求重新刷写,或切换到其他 SDK 分支之后才使用。在下一个单元格中设置 `FLASH_FIRMWARE = True` 以启用;为 `False` 时单元格只打印将要执行的操作。

命令如下:

```bash
sudo systemctl stop dxrt.service
dxcli -u ~/dx-all-suite/dx-runtime/dx_fw/m1/latest/mdot2/fw.bin   # DX-H1 使用 h1/fw.bin
sleep 5
sudo systemctl start dxrt.service
```

<!-- cell: b5e02530-6e11-43f9-a4c6-12587efc5739 src: 3d4e32fb2f -->
#### 3.4.4 dxrun

`dxrun`(为兼容也可用 `run_model`)是一个 CLI 工具,用于运行编译后的 .dxnn 模型以测试功能并测量性能。

<!-- cell: d0b1d604-a0cc-4c22-bd42-c0ec1c0ff8a6 src: 3548e92d89 -->
`yolo26-s_640x640.dxnn` 由两部分组成(NPU 和 CPU)。

 ![](assets/sc-yolo26s.png)

如果某个算子不受 NPU 支持,它会通过 CPU 卸载,使用 ONNX Runtime 在 CPU 上处理。 

使用 `dxrun` 时加上 `--use-ort` 选项会同时执行 NPU 和 CPU 部分,从而测量整个 .dxnn AI 流水线的性能。

如果省略 `--use-ort` 选项,性能测量仅限于 NPU 部分。

<!-- cell: 96b76758-a19a-41ba-81f8-3b82b74d88dc src: 30f541bc2b -->
#### 3.4.5 dxparse

`dxparse`(为兼容也可用 `parse_model`)是一个命令行工具,用于读取编译后的 .dxnn 模型并显示其结构、输入/输出和元数据。

<!-- cell: dae58bf1-ed01-4469-8cab-5f1ebc529039 src: dab48f2d09 -->
## 4. 故障排除

| 症状 | 看到的内容 | 可能原因 | 解决方法 |
|---|---|---|---|
| 安装单元格停在 `sudo` | `sudo: a terminal is required to read the password` | 未配置免密 sudo,安装程序无法在单元格内请求密码 | 在终端(File > New > Terminal)中运行打印出的命令,然后运行验证单元格 |
| 克隆单元格失败 | `fatal: destination path ... already exists and is not an empty directory` | 目标位置存在不属于 git 检出的文件 | 移走该目录或在 1.1 中选择其他位置,然后重新运行单元格 |
| `dxcom` 一直是 MISSING | 状态表中显示 `[MISSING] dx_com` | DX-COM 安装程序未完成 | 向上滚动查看安装程序输出;必须出现 `[HINT]` 块。重新运行 2.1 |
| 没有 NPU 设备 | `[MISSING] npu` 或 `ls: cannot access '/dev/dxrt*'` | 内核驱动未加载,通常是因为 3.1 之后没有重启主机 | 重启。然后检查 `lsmod \| grep dxrt`;如果设备仍然不存在,请将主机完全断电后再开机 |
| `dxrt.service` 未运行 | `systemctl status dxrt.service` 显示 `Active: failed` | 驱动或固件不匹配 | `sudo systemctl restart dxrt.service`;如果再次失败,在终端中运行 `./dx-runtime/install.sh --runtime-only` |
| 模型未下载 | `setup.sh` 报告未知的模型名 | 名称与 Model Zoo 清单不一致(区分大小写) | 在 `dx_app` 中用 `grep name scripts/modelzoo_manifest.json` 查看准确名称 |

<!-- cell: 2e5f3528-1de5-45e1-87db-c8018aa6c1d6 src: da34d3824a -->
## 5. 总结

<!-- cell: 7acbc00f-59ec-4386-b048-d447c0507bff src: 077c9eb1db -->
| 项目 | DX-Compiler | DX-Runtime |
|---|---|---|
| 角色 | 模型编译 | 模型推理执行 |
| 输入 | ONNX | DXNN |
| 输出 | `.dxnn` | 推理结果 |
| 系统 | 仅 x86_64 | x86_64、aarch64 |
| 安装方式 | `./dx-compiler/install.sh --target=dx_com` | `./dx-runtime/install.sh --all` |
| 主要 CLI | `dxcom`(位于 `venv-dx-compiler-local` 中) | `dxcli`、`dxrun`、`dxparse`、`dxbenchmark`、`dxtop` |

### 5.1 完成清单

- [ ] `config.json` 指向 DX-All Suite 目录,且 1.3 的状态单元格中所有项目都显示为 `OK`
- [ ] `dxcom -h` 可以运行,且已编译出 `MobileNetV2-1.dxnn`(2.2)
- [ ] `dxtron` 可以打开 `.dxnn` 文件(2.3,可选)
- [ ] `/dev/dxrt0` 存在、`dxrt.service` 处于活动状态,且 `dxcli -s` 列出了 NPU(3.2)
- [ ] 选定的模型已下载到共享工作区(3.3)
- [ ] 已用 `dxrun` 和 `dxbenchmark` 在 NPU 上运行模型(3.4)

> **下一步:** 继续学习教程 02,使用这里下载的模型运行 DX-APP 示例(分类、检测、姿态、分割)。
