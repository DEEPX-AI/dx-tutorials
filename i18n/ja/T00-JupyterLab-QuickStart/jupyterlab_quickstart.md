<!-- i18n source: notebooks/T00-JupyterLab-QuickStart/jupyterlab_quickstart.ipynb -->
<!-- i18n lang: ja -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 9f7ec700 src: f635a0ef60 -->
# DEEPX Tutorial 00 - JupyterLab クイックスタート

このノートブックでは、Jupyter Notebook と JupyterLab を分かりやすく正しく使う方法を説明します。

<!-- cell: 803b9ba4-59c8-4cc5-beb7-0681a9e58862 src: 5b33f5bea8 -->
## 学習目標

このチュートリアルを終えると、次のことができるようになります。

- Markdown セルとコードセルを見分け、相互に変換する
- セルを順番に実行し、実行順序が重要な理由を説明する
- `!` を使ってセルからシェルコマンドを実行する
- 実行中のセルを中断し、カーネルを再起動する
- DEEPX チュートリアルで使用するキーボードショートカットを使う

<!-- cell: 22dda5c1 src: 95b6d9b019 -->
## 1. Jupyter Notebook と JupyterLab とは?

### 1.1 Jupyter Notebook
Jupyter Notebook は、次の要素を 1 つにまとめたファイル(`.ipynb`)です。
- Python コード
- テキストによるドキュメント
- 出力結果
- プロットとログ

次のような分野で広く使われています。
- データサイエンス
- AI の学習
- 研究
- 教育

### 1.2 JupyterLab
JupyterLab は、ノートブックを実行する Web ベースの環境です。

次のことができます。
- 複数のノートブックを同時に開く
- ファイルを管理する
- ターミナルを開く
- コードを対話的に実行する

簡単に言えば:
- **Notebook = ドキュメント**
- **JupyterLab = ワークスペース**

<!-- cell: f40193e8 src: 3a643b0e06 -->
## 2. JupyterLab インターフェースの紹介

### 2.1 左サイドバー
- ファイルブラウザー
- 実行中のセッション
- 目次

### 2.2 メインエリア
- 開いているノートブック
- 分割表示
- 複数のタブ

### 2.3 トップメニュー
- File
- Edit
- View
- Run
- Kernel
- Settings
- Help

<!-- cell: ded6ef5a src: a32d2b19c2 -->
## 3. セルについて (Markdown セルとコードセル)

ノートブックはセルで構成されています。

### 3.1 Markdown セル
用途:
- タイトル
- 説明
- リスト
- ドキュメント

### 3.2 コードセル
用途:
- Python コード
- コマンド

### 3.3 セルを作成して `Code cell` から `Markdown cell` に変換する
Jupyter で**新しい Markdown セルを作成するには**、ツールバーの `+` ボタンをクリックします。Jupyter では新しいセルはすべて既定でコードセルとして作成されるため、Markdown セルとして認識・表示されるようにセルの形式を変更する必要があります。そのためには、まずセルをカーソルでクリックしてアクティブにします。次に、ツールバーにある「Code」と表示されたドロップダウンボックス(⏭ ボタンの隣にあります)をクリックし、「Code」から「Markdown」に変更します。
![](https://datasciencebook.ca/_main_files/figure-html/convert-to-markdown-cell-1.png)

<!-- cell: 6d235ea4 src: 9c1ab7b997 -->
## 4. セルの実行とルール

コードは **Kernel**(カーネル)と呼ばれるプロセスの中で実行されます。

重要なルール:

1. すべてのセルは同じメモリを共有します。
2. 実行順序が重要です。
3. 結果がおかしい場合はカーネルを再起動してください。
4. 正しい結果を得るには、上から下へ順番に実行してください。


**コードセルを単独で実行するには**、まずセルをアクティブにする必要があります。これはカーソルでセルをクリックすることで行います。セルがアクティブになると、Jupyter はセルの左側を青い枠で強調表示します。セルがアクティブになったら、ツールバーの Run(▶)ボタンを押すか、キーボードショートカット `Shift + Enter` でセルを実行できます。

![](https://datasciencebook.ca/_main_files/figure-html/activate-and-run-button-1.png)

> **次のセルで想定されるエラー:** `a` はまだ定義されていないため、先に `print(a)` を実行すると `NameError` が発生します。これは実行順序を示すための意図的なデモです。エラーを確認したら、続く `a` を定義するセルを実行し、その後 2 つ目の `print(a)` セルを実行してください。

<!-- cell: a66c6696 src: 3997b8ee35 -->
## 5. コードセルの使い方 (Python、シェルコマンド)

### 5.1 Python の例

<!-- cell: 2a2a34cb src: 3bca83ca12 -->
### 5.2 シェルコマンドの例

システムコマンドを実行するには `!` を使います。

例:
- `!ls`
- `!pwd`
- `import sys` の後に `!uv pip list --python "{sys.executable}"`

<!-- cell: c556e004-b18c-4922-b33e-46e1a48933c9 src: 9f7641a317 -->
## 6. 実行中のセルを中断する

コードセルの実行中は、セルの左側に `*` が表示されます。これはセルが現在実行中であることを意味します。
> [*]

セルの実行が終わると、`*` は数字に変わります。この数字は実行順序を示します。

> [1]</br>
> [2]</br>
> [3]</br>

セルの実行が長すぎる場合(たとえば無限ループ)は、ツールバーの停止ボタン(■)をクリックできます。

![](https://media.geeksforgeeks.org/wp-content/uploads/20231010191734/Screenshot-from-2023-10-10-18-55-07.png)

<!-- cell: 9127087a src: d06b76240e -->
## 7. ターミナルを開く

ターミナルを開くには:

`File → New → Terminal`

次のことができます。
- Linux コマンドを実行する
- Git を使う
- パッケージをインストールする

<!-- cell: c7054257 src: c8e4c98164 -->
## 8. 便利なキーボードショートカット

実行:
- `Shift + Enter` → セルを実行して次のセルへ移動(最もよく使う)
- `Ctrl + Enter` → セルを実行してそのまま留まる

セル操作:
- `A` → 上に挿入
- `B` → 下に挿入
- `D, D` → 削除
- `M` → Markdown に変更
- `Y` → Code に変更

モード:
- `Esc` → コマンドモード
- `Enter` または `Mouse L btn double-click` → 編集モード

<!-- cell: 4539e1ac src: c1dfdcb589 -->
## 9. 便利なヒント

✔ こまめに保存する(`Ctrl + S`)  
✔ 必要に応じてカーネルを再起動する  
✔ セルを無秩序に実行しない  
✔ ノートブックが遅い場合は出力をクリアする  
✔ 分かりやすい変数名を使う
