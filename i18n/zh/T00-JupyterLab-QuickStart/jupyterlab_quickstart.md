<!-- i18n source: notebooks/T00-JupyterLab-QuickStart/jupyterlab_quickstart.ipynb -->
<!-- i18n lang: zh -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 9f7ec700 src: f635a0ef60 -->
# DEEPX Tutorial 00 - JupyterLab 快速入门

本笔记本清晰、准确地介绍如何使用 Jupyter Notebook 和 JupyterLab。

<!-- cell: 803b9ba4-59c8-4cc5-beb7-0681a9e58862 src: 5b33f5bea8 -->
## 学习目标

完成本教程后,您将能够:

- 区分 Markdown 单元格和代码单元格,并在两者之间转换;
- 按顺序运行单元格,并解释执行顺序为何重要;
- 在单元格中用 `!` 运行 shell 命令;
- 中断正在运行的单元格并重启内核;
- 使用 DEEPX 教程所依赖的键盘快捷键。

<!-- cell: 22dda5c1 src: 95b6d9b019 -->
## 1. 什么是 Jupyter Notebook 和 JupyterLab?

### 1.1 Jupyter Notebook
Jupyter Notebook 是一种文件(`.ipynb`),它将以下内容组合在一起:
- Python 代码
- 文本文档
- 输出结果
- 图表和日志

它被广泛用于:
- 数据科学
- AI 训练
- 科研
- 教育

### 1.2 JupyterLab
JupyterLab 是运行笔记本的基于 Web 的环境。

它允许您:
- 同时打开多个笔记本
- 管理文件
- 打开终端
- 交互式运行代码

简单理解:
- **Notebook = 文档**
- **JupyterLab = 工作区**

<!-- cell: f40193e8 src: 3a643b0e06 -->
## 2. JupyterLab 界面导览

### 2.1 左侧边栏
- 文件浏览器
- 运行中的会话
- 目录

### 2.2 主区域
- 已打开的笔记本
- 分屏视图
- 多个标签页

### 2.3 顶部菜单
- File
- Edit
- View
- Run
- Kernel
- Settings
- Help

<!-- cell: ded6ef5a src: a32d2b19c2 -->
## 3. 关于单元格(Markdown 单元格与代码单元格)

笔记本由单元格组成。

### 3.1 Markdown 单元格
用于:
- 标题
- 说明
- 列表
- 文档

### 3.2 代码单元格
用于:
- Python 代码
- 命令

### 3.3 创建单元格并从 `Code cell` 转换为 `Markdown cell`
**要在 Jupyter 中创建新的 Markdown 单元格**,请点击工具栏中的 `+` 按钮。默认情况下,Jupyter 中所有新建的单元格都是代码单元格,因此需要更改单元格格式,才能被识别并渲染为 Markdown 单元格。为此,先用光标点击该单元格以确保它处于激活状态。然后点击工具栏上显示“Code”的下拉框(位于 ⏭ 按钮旁边),将其从“Code”改为“Markdown”:
![](https://datasciencebook.ca/_main_files/figure-html/convert-to-markdown-cell-1.png)

<!-- cell: 6d235ea4 src: 9c1ab7b997 -->
## 4. 单元格执行与规则

代码在一个称为 **Kernel(内核)** 的进程中运行。

重要规则:

1. 所有单元格共享同一块内存。
2. 执行顺序很重要。
3. 如果结果异常,请重启内核。
4. 从上到下依次运行以获得干净的结果。


**要单独运行某个代码单元格**,需要先激活该单元格。用光标点击它即可。Jupyter 会在单元格左侧显示蓝色矩形高亮,表示该单元格已激活。激活后,可以点击工具栏中的 Run(▶)按钮,或使用键盘快捷键 `Shift + Enter` 来运行该单元格。

![](https://datasciencebook.ca/_main_files/figure-html/activate-and-run-button-1.png)

> **下一个单元格会出现预期中的错误:** `a` 尚未定义,因此先运行 `print(a)` 会引发 `NameError`。这是为了演示执行顺序而有意设计的。观察到错误后,运行后面定义 `a` 的单元格,然后再运行第二个 `print(a)` 单元格。

<!-- cell: a66c6696 src: 3997b8ee35 -->
## 5. 代码单元格的用法(Python、Shell 命令)

### 5.1 Python 示例

<!-- cell: 2a2a34cb src: 3bca83ca12 -->
### 5.2 Shell 命令示例

使用 `!` 运行系统命令。

示例:
- `!ls`
- `!pwd`
- 先 `import sys`,然后 `!uv pip list --python "{sys.executable}"`

<!-- cell: c556e004-b18c-4922-b33e-46e1a48933c9 src: 9f7641a317 -->
## 6. 中断正在运行的单元格

代码单元格执行时,单元格左侧会显示 `*`。这表示该单元格正在运行。
> [*]

单元格运行结束后,`*` 会变为一个数字。该数字表示执行顺序。

> [1]</br>
> [2]</br>
> [3]</br>

如果某个单元格运行时间过长(例如无限循环),可以点击工具栏中的停止按钮(■):

![](https://media.geeksforgeeks.org/wp-content/uploads/20231010191734/Screenshot-from-2023-10-10-18-55-07.png)

<!-- cell: 9127087a src: d06b76240e -->
## 7. 打开终端

要打开终端:

`File → New → Terminal`

您可以:
- 运行 Linux 命令
- 使用 Git
- 安装软件包

<!-- cell: c7054257 src: c8e4c98164 -->
## 8. 常用键盘快捷键

运行:
- `Shift + Enter` → 运行单元格并跳到下一个单元格(最常用)
- `Ctrl + Enter` → 运行单元格并停留在当前位置

单元格控制:
- `A` → 在上方插入
- `B` → 在下方插入
- `D, D` → 删除
- `M` → Markdown
- `Y` → Code

模式:
- `Esc` → 命令模式
- `Enter` 或 `Mouse L btn double-click` → 编辑模式

<!-- cell: 4539e1ac src: c1dfdcb589 -->
## 9. 实用技巧

✔ 经常保存(`Ctrl + S`)  
✔ 必要时重启内核  
✔ 避免随意运行单元格  
✔ 笔记本变慢时清除输出  
✔ 使用清晰的变量名
