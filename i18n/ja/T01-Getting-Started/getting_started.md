<!-- i18n source: notebooks/T01-Getting-Started/getting_started.ipynb -->
<!-- i18n lang: ja -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 7561e296-41b4-44c3-ab4f-4c8582f0593f src: 14f392f9fc -->
# DEEPX Tutorial 01 - DEEPX SDK のインストール方法
この最初のチュートリアルでは、DEEPX SDK をインストールし、セットアップが正常に完了したことを確認する方法を説明します。環境の準備、SDK のインストール、そして DEEPX NPU デバイス(DX-M1、DX-M1M、DX-H1 Quattro)がシステムで正しく認識されていることの確認までを学びます。

## 学習目標

このチュートリアルを終えると、DX-All Suite を正しくインストールし、DEEPX NPU デバイス上で基本的なフローを実行できるようになります。

<!-- cell: 389418ec-fa11-4a60-9822-183b9d441952 src: 32aa4d182a -->
## 前提条件

<!-- cell: 9c427531-ff8d-4e74-93db-8f41d1237db6 src: 2c603254b1 -->
**注:** 以下の要件はこのチュートリアルのためのものであり、**DEEPX 製品を使用するための必須条件ではありません。**
- OS: Linux (Ubuntu 20.04/22.04/24.04/26.04, Debian 12/13)
- RAM: 8G (DX-Compiler を使う場合は 16G)
- ストレージ: 40 GB 以上
- DEEPX NPU: DX-M1、DX-M1M、DX-H1 Quattro
- CPU: x86_64 では DX-Compiler + DX-Runtime、aarch64 では DX-Runtime のみ

<!-- cell: 2dcfff26-61ce-46f1-b08f-5669e3ead9a8 src: d3372f2340 -->
## DXNN® - DEEPX NPU SDK の紹介 (DX-AS: DX-All Suite)

<!-- cell: 52b3f559-cb24-4e41-8286-6802b4d96c4d src: 99488f0f51 -->
DX-AS(DX-All Suite)は、DEEPX デバイスを使って AI モデルのコンパイルと推論を行うためのフレームワークとツールの統合環境です。個別のツールをインストールして統合環境を構築することもできますが、DX-AS は各ツールのバージョンを揃えることで最適な互換性を維持します。

![](https://github.com/DEEPX-AI/dx-all-suite/raw/main/docs/source/resources/dxnn_sdk_illustration.png)

DEEPX SDK は大きく 2 つの部分に分かれています。

1 つ目は AI モデルコンパイル環境で、AI モデルを DEEPX NPU 上で効率的に実行できる最適化された形式に変換します。

2 つ目は AI モデルランタイム環境で、コンパイル済みの AI モデルを実際の DEEPX NPU ハードウェア上で実行して結果を生成します。

DX-All Suite を使えば、両方のコンポーネントを個別に管理することなく、まとめてセットアップできます。

![](https://github.com/DEEPX-AI/dx-all-suite/raw/main/docs/source/img/dx-as.png)


![](https://github.com/DEEPX-AI/dx-all-suite/raw/main/docs/source/img/DXNN-SDK-Simple-Architecture.png)



理解を助けるために、DX-SDK に関する YouTube 動画が 2 本あります。
- [Youtube - DEEPX SDK Introduction](https://www.youtube.com/watch?v=Js6Soex0WI4) | ダウンロード: [EN](https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/1.DXNN_Introduce_ENG.mp4) · [中文](https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/1.DXNN_Introduce_CH.mp4) · [한국어](https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/1.DXNN_Introduce_KO.mp4)

<img src="assets/youtube-dx-sdk.png" style="max-width: 1000px;">

<!-- cell: 2af4c675-fced-4ac2-ab69-05a2a8abbeb9 src: accfc71fa0 -->
## 1. DX-All Suite のダウンロード

<!-- cell: 3b0789e3-bf0c-4304-8545-bed41a13b3be src: c90ee8eafd -->
### 1.1 インストールの設定

チュートリアルは次の順序で DX-All Suite を探します。

1. 環境変数 `DX_ALL_SUITE_DIR`、
2. `dx-tutorials/config.json`(この節で作成され、git では追跡されません)、
3. `~/dx-all-suite` や `dx-tutorials` の隣にある `dx-all-suite` フォルダーなど、一般的な場所の自動検出、
4. 既定の場所 `~/dx-all-suite`。

次のセルは、見つかった場所とその値の出所を表示します。SDK がまだインストールされていない場合、既定の場所は `MISSING` と表示されます。クローン手順の前では正常な状態です。

<!-- cell: dx-all-suite-directory-heading src: b988a5d1c2 -->
#### 1.1.1 インストールディレクトリ

上に表示された場所をそのまま使う場合は、次のセルを変更しないでください。DX-All Suite を別の場所にインストールしたい場合(またはすでにインストール済みの場合)にのみ `DX_ALL_SUITE_DIR` を編集します。セルを実行すると選択内容が `dx-tutorials/config.json` に保存され、他のすべてのチュートリアルが同じ場所を使います。

ターミナルからいつでも設定することもできます(`dx-tutorials` ディレクトリで実行)。

```bash
python tutorial_paths.py --set ~/my/dx-all-suite
```

<!-- cell: dx-all-suite-branch-heading src: 635edf5993 -->
#### 1.1.2 Git ブランチ

別の SDK バージョンが必要な場合にのみ、下のブランチ名を変更してください。
現在のブランチ名は `main` です。

<!-- cell: 8d2496bb-a3a2-460c-a891-1979b3cfab1d src: 187f9b06f0 -->
### 1.2 DX-All Suite のクローン

次のセルは、設定した場所に **DX-All Suite が存在しない場合にのみ**、以下のコマンドを実行します。ディレクトリとブランチは上で保存した設定から取得され、セル内では `{DX_ALL_SUITE_DIR}` と `{DX_ALL_SUITE_BRANCH}` として表示されます。

```bash
git clone --depth 1 --shallow-submodules --recurse-submodules --progress \
    --branch main https://github.com/DEEPX-AI/dx-all-suite.git ~/dx-all-suite
```

- `--depth 1 --shallow-submodules` はすべてのリポジトリの最新コミットだけを取得します。数 GB ではなく約 600 MB のダウンロードとディスク上 1.6 GB で済みます。
- `--recurse-submodules` はサブモジュール(`dx-runtime`、`dx-compiler`、`dx-modelzoo` とその内部のリポジトリ)もチェックアウトします。
- `--progress` は出力がターミナルでなくても進捗を表示するので、セル内で進み具合を確認できます。

ネットワークによって 1〜10 分かかります。■ ボタンでセルを停止し、後で再実行できます。完了すると最後の行に `Submodule path ... checked out` と表示されます。

前回のクローンがトップレベルのチェックアウト後に中断された場合、ディレクトリには `.git` がありますが一部のサブモジュールが空です。その場合、セルは再クローンの代わりに次のコマンドを実行します。

```bash
git -C ~/dx-all-suite submodule update --init --recursive --depth 1 --progress
```

セルを使わずに、ターミナル(**File > New > Terminal**)でどちらかのコマンドを自分で実行しても構いません。下のステータスセルはどちらの場合も同じように動作します。

<!-- cell: b6993bc2-9b2c-48ec-995b-4f1777a8844c src: 034c0df327 -->
### 1.3 現在の状態と次のステップ

次のセルは、これまでにインストールされたものと、`MISSING` と表示された項目についてはそれを提供する正確な手順を表示します。このノートブックに戻ってくるたびに再実行してください。繰り返し実行しても安全です。

<!-- cell: ee6ce747-f96c-417f-917f-9cccd54be3ec src: 64b774a178 -->
### 1.4 DX-All Suite の中身

クローンには 3 つの Git サブモジュールが含まれています。
 - `dx-compiler` (**DX-Compiler**): ONNX モデルを DXNN に変換します。第 2 節でインストールします。
 - `dx-runtime` (**DX-Runtime**): コンパイル済みモデルを DEEPX NPU 上で実行します。第 3 節でインストールします。
 - `dx-modelzoo`: 後のチュートリアルで使用するコンパイル済みモデルとコンパイラ設定です。

<!-- cell: 2ab0cf89-7974-40e6-8117-18165d1ef10f src: 3d336a7ce6 -->
DX-Runtime は次を提供します。
 - `DX-APP`: ユーザーアプリケーションの観点から DX NPU の使い方を示す DEEPX アプリケーションテンプレート
 - `DX-FW`: NPU ファームウェアバイナリ
 - `DX-RT`: DX NPU を使った推論タスクを最適化して実行するために設計されたフレームワーク
 - `DX-NPU Driver`: DX NPU 用の Linux カーネルドライバー
 - `DX-STREAM`: DX NPU 向けの GStreamer ベースのビジョン AI アプリケーション開発ツール

<!-- cell: 912db406-4064-4ea1-b075-c455abeab01f src: b20e1b138c -->
## 2. DX-Compiler のインストール

<!-- cell: c342f780-67f2-4914-a52d-321b5fe55b72 src: 7b700f5a1e -->
詳細は [DX-All Suite インストールガイド](https://github.com/DEEPX-AI/dx-all-suite/blob/main/docs/source/02_Setting_Up_Environment.md)を参照してください。

DX-Compiler 環境はビルド済みのバイナリを提供し、ソースコードは含みません。インストーラーは各モジュールをリモートサーバーからダウンロードします。

| コマンド | インストール対象 | `sudo` |
|---|---|---|
| `./dx-compiler/install.sh --target=dx_com` | 専用の Python 環境に入る DX-COM コンパイラ(`dxcom`) | すでに存在していても `apt-get install python3-dev python3-venv` を実行 |
| `./dx-compiler/install.sh --target=dx_tron` | DX-TRON モデルビューアー(`.deb` パッケージ) | `apt-get` でパッケージをインストール |
| `./dx-compiler/install.sh` | 両方 | 上記の両方 |

このチュートリアルでは 2.1 で DX-COM をインストールします。DX-TRON は任意で、2.3 で扱います。

<!-- cell: 3bdb9b57-de73-4186-bac9-ac1effa245da src: 814d4a60eb -->
### 2.1 DX-COM のインストール

実行するコマンドは次のとおりです。

```bash
cd ~/dx-all-suite
./dx-compiler/install.sh --target=dx_com
```

`run-jupyter-lab.sh` はチュートリアルの Python 環境を有効化せずに JupyterLab を起動するため、このコマンドはノートブックのセルで実行してもターミナルで実行してもシステムの `python3` を使います。インストーラーは `dx-compiler/venv-dx-compiler-local` に独自の環境を作成し、Jupyter の環境には手を加えません。

*どこで*実行するかを決める唯一の要素は `sudo` です。インストーラーは `python3-dev` と `python3-venv` のために常に `sudo apt-get` を呼び出しますが、ノートブックのセルはパスワード入力に応答できません。そのため、以下の 3 つのセルは次のことを行います。

1. **確認**: `dxcom` がすでにインストールされているか、この環境で `sudo` がパスワードなしで動作するかを確認します。
2. **インストール**: 上記のコマンドを実行します(確認セルが実行してよいと示した場合のみ実行してください)。
3. **検証**: `dxcom` が存在することを確認します。

ダウンロードは数百 MB で、2〜10 分かかります。出力の最後に緑色の `[HINT] dx_com installation completed!` ブロックが表示されます。参考のために下に掲載しています。

<!-- cell: 891804c8-7521-4f34-a453-f8261151d1f4 src: ae07f58239 -->
インストールが正常に完了すると、インストーラーは次のヒントブロックを表示します。作成された仮想環境 `dx-compiler/venv-dx-compiler-local` は Jupyter の環境とは分離されています。2.2 のセルは各 `!` 行のシェル内で `source venv-dx-compiler-local/bin/activate` を実行して有効化するため、Jupyter の環境は変更されません。

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

次に、DX-Compiler によってインストールされたコンポーネントを確認します。

<!-- cell: 63c15643-8cfc-4a6f-a6ee-67451ef0b5fb src: c2d63e41ac -->
では、dx_com 配下のファイルとフォルダーを見てみましょう。

<!-- cell: cffd0d41-fa76-48c4-af02-6a07dcf47057 src: d96c946dd1 -->
### 2.2 DX-Compiler の検証

2.1 の検証セルで `dxcom` が存在することはすでに確認しました。この節では実際に実行します。まずヘルプを表示し、次にサンプルモデルを実際にコンパイルします。

<!-- cell: 1803112e-39d6-4066-8abc-c1c2d61fdfbc src: dbf5bd0d1f -->
#### 2.2.1 `dxcom` のヘルプを確認する

`dxcom` はインストーラーが作成した仮想環境の中にあるため、ターミナルでは呼び出す前にその環境を有効化する必要があります。次のセルは正確に以下のコマンドを実行します。

```bash
cd ~/dx-all-suite/dx-compiler
source venv-dx-compiler-local/bin/activate
dxcom -h
```

ノートブックの `!` 行は毎回新しいシェルを起動するため、3 つのコマンドを `&&` で 1 行につないでいます。有効化はそのシェルにのみ適用され、Jupyter の環境は変更されません。

<!-- cell: 9cf00564-56e7-42cf-bbf6-873292fee8da src: bf66ec5280 -->
#### 2.2.2 `MobileNetV2-1.onnx` をコンパイルして `MobileNetV2-1.dxnn` を生成する

次のセルは以下のコマンドを実行します。コンパイラは各ステージを出力し、`--gen_log` によって同じ内容が `compiler.log` にも書き込まれます。数秒から 1 分程度かかります。

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
`MobileNetV2-1.dxnn` ファイルが生成されたか確認します。

<!-- cell: 3b83c7c6-a23c-4fd3-ba83-890dfa2b3584 src: adf789f4ad -->
`--gen_log` オプションは、コンパイルログを `compiler.log` という名前のファイルに出力します。

保存されたログメッセージを見てみましょう。

<!-- cell: d93f93a6-c5f7-476e-9632-695968e6cc3c src: 2e60c3c391 -->
### 2.3 DX-Tron

**DX-TRON** は、DEEPX ツールチェーンでコンパイルした `.dxnn` モデルファイルを探索するためのグラフィカルな可視化ツールです。 

モデル構造を読み込んで確認し、色分けされたグラフで NPU と CPU の間のワークロード分配を確認できます。 

DX-TRON を使うと、モデルの実行フローをより深く理解し、全体の性能を改善できます。

> **注:** DX-TRON は引き続き SDK と共に配布されていますが、積極的なメンテナンスは行われていません。コンパイル済みモデルのレポートには、`dxcom --export_html` が生成する HTML サマリー(チュートリアル 05)を推奨します。DX-TRON は `.dxnn` ファイルを手早く視覚的に確認する用途では今も便利です。

<img src="assets/sc-dxtron.png" style="max-width: 600px;">

**主な機能:**
- **.dxnn ファイル対応**: DEEPX ツールチェーンでコンパイルしたモデルファイルを読み込んで可視化します。
- **ワークロードの視覚的表現**: ワークロードの実行を色分けして表示します。
- 赤: NPU で実行される演算
- 青: CPU またはホストで実行される演算
- **モデルナビゲーション**: 左下の戻る矢印で、いつでもモデル概要画面に戻れます。
- **対話的なノード検査**: グラフ内のノードをダブルクリックすると、関連する演算の詳細情報を表示できます。
- 参考: DX-TRON は DXNN をサポートするために [netron](https://netron.app/) をベースに開発されました。

<!-- cell: cdc69460-e92c-460b-aaa5-dc0651eba2dc src: 9f6e1afb38 -->
#### 2.3.1 DX-TRON のインストール

DX-TRON は Debian パッケージとして配布されるため、インストーラーは `sudo apt-get` を使用します。

```bash
cd ~/dx-all-suite
./dx-compiler/install.sh --target=dx_tron
```

2.1 と同様に 3 つのセルが続きます。**確認**(`dxtron` がインストールされているか、`sudo` がパスワードなしで動作するか)、**インストール**(上記のコマンド。確認セルが実行してよいと示した場合のみ実行するか、ターミナルで実行してください)、**検証**です。

<!-- cell: 061c3fe9-86fa-4dc3-aa5a-bf48c2365459 src: eb9122c32f -->
#### 2.3.2 DX-TRON の実行

次のセルは、2.2.2 でコンパイルした MobileNetV2 モデルを DX-TRON で開きます。ウィンドウが開いている間、セルは実行中のままになります。
> **注:** 上の停止ボタン('■')をクリックすると `dxtron` を終了できます!

<!-- cell: e8158e1f-0476-46b1-b72a-94006d838653 src: bcd38bb9d2 -->
## 3. DX-Runtime のインストール
詳細は [DX-All Suite インストールガイド](https://github.com/DEEPX-AI/dx-all-suite/blob/main/docs/source/02_Setting_Up_Environment.md)を参照してください。

<!-- cell: 1dee6063-b928-4c16-8100-bbde4cc779db src: a530d80148 -->
### 3.1 (任意) インストール前の前提条件 (`Orangepi-5 plus` の場合のみ)
 - Orange Pi の公式イメージを使用している場合、カーネルヘッダーがインストールされていないことがあります。NPU ドライバーのインストールにはカーネルヘッダーが必要です。
 - [こちら](../../docs/orangepi5p.md)にリンクされたドキュメントを参照してください。

### 3.2 (任意) インストール前の前提条件 (`Raspberrypi-5` の場合のみ)
 - PCIe は既定で Gen2 に設定されています。帯域幅を増やすために Gen3 に設定できます。 
 - [こちら](../../docs/raspberrypi5.md)にリンクされたドキュメントを参照してください。 

<!-- cell: 5a923f2b-8ed3-454b-b171-6162b87349ed src: d59a17ec7e -->
DX-Runtime 環境には各モジュールのソースコードが含まれています。リポジトリは `./dx-runtime` 配下の Git サブモジュール(`dx_rt_npu_linux_driver`、`dx_fw`、`dx_rt`、`dx_app`、`dx_stream`)として管理されています。

DX-Runtime インストールスクリプトのすべてのオプションを見てみましょう。

<!-- cell: f46c6f59-c240-40ef-beef-c391b6b02ad7 src: bdef7c465b -->
### 3.1 DX-Runtime のインストール

後のチュートリアルで DX-APP と DX-STREAM を使用するため、`--all` ですべてをインストールします。

```bash
cd ~/dx-all-suite
./dx-runtime/install.sh --all
```

コアランタイム(NPU ドライバー、ファームウェア、DX-RT)だけが必要な場合は、`--all` の代わりに `--runtime-only` を使用してください。

インストーラーは全体を通して `sudo` を使用します。NPU カーネルドライバーのビルドとロード、Debian パッケージのインストール、`dxrt.service` デーモンの登録を行います。質問はしません。2.1 と同様に 3 つのセルが続きます。**確認**(インストール済みか、`sudo` がパスワードなしで動作するか)、**インストール**(上記のコマンド。確認セルが実行してよいと示した場合のみ実行するか、ターミナルで実行してください)、**検証**です。10〜30 分かかり、大部分は DX-RT、DX-APP、DX-STREAM のビルド時間です。

NPU ドライバーのインストール後は**再起動**が必要で、ファームウェア更新後はインストーラーが完全な電源オフを推奨します。再起動後、`./run-jupyter-lab.sh` を再度起動し、このノートブックを開いて最初のコードセルを実行してから 3.2 に進んでください。

<!-- cell: 379be8a6-b287-462b-a2d7-3444b12b2b06 src: e04e936876 -->
### 3.2 インストールの検証

インストールディレクトリと Git ブランチは、設定セルによってすでに `dx-tutorials/config.json` に保存されています。この検証では、リポジトリ、ブランチ、サブモジュール、DX-Compiler 環境、DX-Runtime CLI、NPU デバイスノードを確認します。その後のセルでは、PCIe リンク、カーネルドライバー、サービスをより詳しく調べます。

<!-- cell: 8019bed5-064a-4d2f-b1c6-28e4608c977e src: b7b11a0e49 -->
### 3.3 推奨: 選択したモデルのみダウンロードする

次のコードセルは `SELECTED_MODELS` を許可リストとして使い、それらのモデルだけをダウンロードします。これがこのチュートリアルで推奨する既定の方法です。別のモデルセットが必要な場合は、セルを実行する前にリストを編集してください。

> **警告 — 必要な場合を除き、全件ダウンロードのコマンドは使用しないでください。**  
> `!cd $DX_ALL_SUITE_DIR/dx-runtime/dx_app && bash setup.sh <<< ""` を実行すると、対話型プロンプトに空の回答が与えられます。空の回答はすべてのカテゴリとすべてのモデルを選択するため、349 個のモデルがすべてダウンロードされます。全セットには大量のネットワークトラフィックと約 29 GB のストレージが必要です。

推奨セルは `SELECTED_MODELS` を `setup.sh --models` に渡し、`--no-force` を使って既存のモデルファイルを再ダウンロードしないようにします。リストされている 24 個のモデルはチュートリアル 02〜24 で使用するもので、ダウンロードにはネットワークによって数分かかります。

<!-- cell: 6549dfd3-3e52-45ca-b3a7-29a0c3a23134 src: de74293afb -->
### 3.4 DX-RT が提供する便利なツール

<!-- cell: f2fe6a42-d9fc-426f-9ec3-06b64fd9180b src: a05bf69c17 -->
DX-RT がサポートする CLI ツールを見てみましょう。

<!-- cell: dxrt-cli-compatibility-heading src: eda0521192 -->
**現在の CLI 名と後方互換の名前**

DX-RT は現在の CLI バイナリを `/usr/local/bin` にインストールします。以前のコマンド名は、現在のバイナリへのシンボリックリンクとして引き続き利用できます。

| 現在の名前 | 以前の互換名 | 実装 |
|---|---|---|
| `dxcli` | `dxrt-cli` | `dxrt-cli -> dxcli` |
| `dxrun` | `run_model` | `run_model -> dxrun` |
| `dxparse` | `parse_model` | `parse_model -> dxparse` |

したがって、どちらの名前でも同じバイナリが実行され、同じオプションを受け付けます。ヘルプには呼び出しに使った名前が表示されることがあります。`dxbenchmark`、`dxtop`、`dxrtd` は既存の名前のままで、独立したバイナリとしてインストールされます。

<!-- cell: 5ead4347-03ba-4d2a-8779-4333e6ddbe5d src: 688be0a7e3 -->
#### 3.4.1 dxbenchmark

`dxbenchmark` は、コンパイル済みの .dxnn モデルを実行して機能をテストし、性能を測定する CLI ツールです。

<!-- cell: 34c9aeab-e537-4c77-a570-1848c07b9e03 src: 6a7b03d552 -->
Web ブラウザーを開いて、`dxbenchmark` の結果を HTML 形式で確認します。

<!-- cell: ae8b23c7-05b3-432a-a55b-aa7fd57fcd05 src: 39068fd181 -->
#### 3.4.2 dxtop

`dxtop` は、使用率、温度、メモリなどの DEEPX NPU メトリクスをリアルタイムで監視する htop 風のツールです。

> dxtop の出力形式は Jupyter Notebook のコードセルでは正しく表示されません。代わりに**別のターミナルを開いて** dxtop コマンドを実行してください。 </br>
> 別のターミナルを開くには: `File > New > Terminal`
> 
> ![](assets/open-terminal.png)

<!-- cell: 8cacae4f-7187-4e97-9c82-bbd2d4605826 src: e3b55470f3 -->
> <img src="assets/sc-dxtop.png" style="max-width: 600px;">

<!-- cell: b8469434-2b9d-466a-a5d1-2ba415a678e2 src: 277f9a2a24 -->
#### 3.4.3 dxcli

`dxcli`(互換性のため `dxrt-cli` としても利用可能)は、DEEPX DX-RT デバイスの照会と監視、および NPU ファームウェアの管理を行います。

<!-- cell: fff91477-4f16-4cd2-b0bd-7962bacdfa4d src: c1380e457f -->
#### 任意: NPU ファームウェアの書き込み

DX-Runtime インストーラーはこの SDK バージョンに対応するファームウェアをすでに書き込んでいるため、この手順は通常**不要**です。DEEPX サポートから再書き込みを求められた場合や、別の SDK ブランチに切り替えた後にのみ使用してください。次のセルで `FLASH_FIRMWARE = True` に設定すると有効になります。`False` の場合、セルは実行予定の内容を表示するだけです。

コマンドは次のとおりです。

```bash
sudo systemctl stop dxrt.service
dxcli -u ~/dx-all-suite/dx-runtime/dx_fw/m1/latest/mdot2/fw.bin   # DX-H1 は h1/fw.bin
sleep 5
sudo systemctl start dxrt.service
```

<!-- cell: b5e02530-6e11-43f9-a4c6-12587efc5739 src: 3d4e32fb2f -->
#### 3.4.4 dxrun

`dxrun`(互換性のため `run_model` としても利用可能)は、コンパイル済みの .dxnn モデルを実行して機能をテストし、性能を測定する CLI ツールです。

<!-- cell: d0b1d604-a0cc-4c22-bd42-c0ec1c0ff8a6 src: 3548e92d89 -->
`yolo26-s_640x640.dxnn` は 2 つの部分(NPU と CPU)で構成されています。

 ![](assets/sc-yolo26s.png)

NPU がサポートしない演算子は、CPU オフロードによって ONNX Runtime を使い CPU で処理されます。 

`dxrun` に `--use-ort` オプションを追加すると NPU と CPU の両方の部分が実行され、.dxnn AI パイプライン全体の性能を測定できます。

`--use-ort` オプションを省略すると、性能測定は NPU 部分のみに限定されます。

<!-- cell: 96b76758-a19a-41ba-81f8-3b82b74d88dc src: 30f541bc2b -->
#### 3.4.5 dxparse

`dxparse`(互換性のため `parse_model` としても利用可能)は、コンパイル済みの .dxnn モデルを読み込み、その構造、入出力、メタデータを表示するコマンドラインツールです。

<!-- cell: dae58bf1-ed01-4469-8cab-5f1ebc529039 src: dab48f2d09 -->
## 4. トラブルシューティング

| 症状 | 表示される内容 | 考えられる原因 | 対処 |
|---|---|---|---|
| インストールセルが `sudo` で止まる | `sudo: a terminal is required to read the password` | パスワードなしの sudo が設定されておらず、インストーラーがセル内でパスワードを要求できない | 表示されたコマンドをターミナル(File > New > Terminal)で実行し、その後検証セルを実行する |
| クローンセルが失敗する | `fatal: destination path ... already exists and is not an empty directory` | 対象の場所に git チェックアウトではないファイルが存在する | ディレクトリを移動するか 1.1 で別の場所を選び、セルを再実行する |
| `dxcom` が MISSING のまま | ステータス表に `[MISSING] dx_com` | DX-COM インストーラーが最後まで完了していない | 上にスクロールしてインストーラーの出力を確認する。`[HINT]` ブロックが必要。2.1 を再実行する |
| NPU デバイスがない | `[MISSING] npu` または `ls: cannot access '/dev/dxrt*'` | カーネルドライバーがロードされていない。多くは 3.1 の後にホストを再起動していない | 再起動する。次に `lsmod \| grep dxrt` を確認し、それでもデバイスがなければホストの電源を完全に切ってから入れ直す |
| `dxrt.service` が動作していない | `systemctl status dxrt.service` に `Active: failed` | ドライバーまたはファームウェアの不一致 | `sudo systemctl restart dxrt.service`。再び失敗する場合はターミナルで `./dx-runtime/install.sh --runtime-only` を実行する |
| モデルがダウンロードされない | `setup.sh` が不明なモデル名を報告する | 名前が Model Zoo のマニフェストと異なる(大文字小文字を区別) | `dx_app` 内で `grep name scripts/modelzoo_manifest.json` を実行して正確な名前を確認する |

<!-- cell: 2e5f3528-1de5-45e1-87db-c8018aa6c1d6 src: da34d3824a -->
## 5. まとめ

<!-- cell: 7acbc00f-59ec-4386-b048-d447c0507bff src: 077c9eb1db -->
| 項目 | DX-Compiler | DX-Runtime |
|---|---|---|
| 役割 | モデルのコンパイル | モデル推論の実行 |
| 入力 | ONNX | DXNN |
| 出力 | `.dxnn` | 推論結果 |
| システム | x86_64 のみ | x86_64、aarch64 |
| インストール方法 | `./dx-compiler/install.sh --target=dx_com` | `./dx-runtime/install.sh --all` |
| 主な CLI | `dxcom`(`venv-dx-compiler-local` 内) | `dxcli`、`dxrun`、`dxparse`、`dxbenchmark`、`dxtop` |

### 5.1 完了チェックリスト

- [ ] `config.json` が DX-All Suite ディレクトリを指し、1.3 のステータスセルですべての項目が `OK` と表示される
- [ ] `dxcom -h` が実行でき、`MobileNetV2-1.dxnn` がコンパイルされた (2.2)
- [ ] `dxtron` で `.dxnn` ファイルを開ける (2.3、任意)
- [ ] `/dev/dxrt0` が存在し、`dxrt.service` がアクティブで、`dxcli -s` に NPU が表示される (3.2)
- [ ] 選択したモデルが共有ワークスペースにダウンロードされた (3.3)
- [ ] `dxrun` と `dxbenchmark` で NPU 上でモデルを実行した (3.4)

> **次へ:** チュートリアル 02 に進み、ここでダウンロードしたモデルで DX-APP のサンプル(分類、検出、姿勢推定、セグメンテーション)を実行してみましょう。
