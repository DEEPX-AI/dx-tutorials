<!-- i18n source: notebooks/T05-DX-Compiler/dx_com_01_beginner.ipynb -->
<!-- i18n lang: ja -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 263bea17 src: 05ca91a778 -->
# DEEPX Tutorial 05-1 - DX-COM 入門

このノートブックでは、小さく再現可能な例を使って、ONNX から DXNN までの完全なワークフローを紹介します。

## 学習目標

このチュートリアルを終えると、次のことができるようになります。

1. DX-COM の全体的なプロセスを理解する
2. DX-COM のインストールを検証する
3. MobileNetV2 の ONNX モデルをエクスポートして検証する
4. キャリブレーション設定を作成する
5. DXNN モデルをコンパイルして検査する
6. DEEPX Model Zoo から ONNX モデルとその JSON 設定をダウンロードしてコンパイルする

このノートブックは SDK のソースツリーを変更しません。生成されるファイルはすべて `<dx-tutorials>/notebooks/T05-DX-Compiler/workspace` 配下に保存され、git では無視されます。

<!-- cell: 09404f54 src: 15d2514cfa -->
## コースマップ

| ノートブック | 主な内容 |
|---|---|
| 入門 | インストール、ONNX の検証、JSON の基礎、初めてのコンパイル、Model Zoo |
| 中級 | キャリブレーション品質、ハードウェア PPU、YOLO26 TopK 最適化 |
| 上級 | Q-PRO、診断、QXNN レジューム、QAT、Python API、高度なコンパイラ制御 |

DX-COM の設定とキャリブレーションをすでに理解している場合を除き、ノートブックは順番に進めてください。

<!-- cell: dx-com-01-prerequisites src: 70ef62da25 -->
## 前提条件

- チュートリアル 01 が完了していること: DX-COM が `dx-compiler/venv-dx-compiler-local` にサンプルキャリブレーションデータセットと共にインストール済みであること。第 5 節と第 6 節の `dxparse` と `dxrun` のセルには DX-RT と DEEPX NPU が必要です。
- ダウンロード: 第 3 節のエクスポートで使う CPU 専用の PyTorch ホイールと ONNX ツール(初回インストール時に数百 MB)、および Model Zoo の ResNet50 ONNX(97 MB)
- `sudo` は不要です。所要時間は約 20 分で、2 回のコンパイルはそれぞれ 1 分未満です
- このチュートリアルが作成するものはすべて `notebooks/T05-DX-Compiler/workspace/` に置かれ、git では無視されます

第 2 節のセットアップセルが SDK を探し、これらの要件を確認してステータス表を表示します。`MISSING` と表示された項目には、それを提供する手順が併記されます。

<!-- cell: dxcom-workflow-overview src: 44853a9f8f -->
## 1. コンパイルワークフロー

全体のコンパイルプロセスは次の 3 つのステップで構成されます。

1. **学習済みモデルの取得** — モデルを学習するか、PyTorch などのフレームワークから互換性のある学習済みモデルを入手します。
2. **モデルを ONNX に変換** — フレームワークのモデルを ONNX にエクスポートし、入力名、入力形状、演算子、出力を確認します。
3. **ONNX を DXNN にコンパイル** — JSON 設定と代表的なキャリブレーションデータを使って DX-COM を実行し、DEEPX NPU 向けのモデルを生成します。

<img src="assets/dx-com-workflow.jpg" style="max-width: 800px; width: 100%;" alt="学習済みモデルから ONNX、DXNN に至る DX-COM コンパイルワークフロー">

この入門チュートリアルでは、MobileNetV2 を使って同じワークフロー(PyTorch → ONNX → DXNN)をたどります。

詳細は DX-Compiler ユーザーガイドを参照してください [ダウンロード](https://developer.deepx.ai/download/?id=581)

> **注:** ユーザーガイドをダウンロードするには、まず https://developer.deepx.ai/ にログインする必要があります。

<!-- cell: dxcom-artifact-glossary src: 6e807062e7 -->
### 1.1 コンパイル時に使用・生成されるファイル

これらの成果物は区別して考えてください。それぞれワークフローの中で異なる役割を持ちます。

| 成果物 | 役割 | 使用・生成するもの |
|---|---|---|
| `.pt` などのフレームワークモデル | 元のフレームワークにおける学習済みの重みとネットワーク定義 | ONNX エクスポート時に使用 |
| `.onnx` | DX-COM が読み込む、フレームワークに依存しないモデルグラフ | DX-COM への入力 |
| コンパイラ `.json` | 入力形状、キャリブレーション、前処理、量子化、および任意のコンパイラ設定 | DX-COM への入力 |
| キャリブレーションデータセット | 量子化範囲の推定に使う代表的なサンプル | キャリブレーション中に DX-COM が読み込む |
| `.dxnn` | DEEPX ランタイムが実行するコンパイル済みモデル | DX-COM の主な出力 |
| `compiler.log` | 詳細なコンパイルメッセージ、警告、エラー | `--gen_log` で生成 |
| `*_summary.html` | グラフとコンパイラ情報を含む視覚的なコンパイルレポート | `--export_html` で生成 |

`.onnx`、コンパイラ `.json`、キャリブレーションデータセットはコンパイルの入力です。`.dxnn`、ログ、HTML レポートは出力です。

<img src="assets/dx-compile-progress.png" style="max-width: 1000px; width: 100%;" alt="必要なファイルを伴う DX-COM のコンパイル">

<!-- cell: f668eb12 src: 50dcc961e3 -->
## 2. 要件とワークスペース

DX-COM は **x86-64 Linux** をサポートします。**バッチサイズ 1** の静的な **ONNX 入力**形状を使用してください。実用的なホストには少なくとも **16 GB の RAM** と 8 GB の空きストレージが必要です。

このチュートリアルには、SDK のインストールから次の 2 つが必要です。

| 要件 | 提供元 | `MISSING` と報告された場合 |
|---|---|---|
| `dx_com`(`venv-dx-compiler-local` 内の `dxcom` コンパイラ) | `./dx-compiler/install.sh` | チュートリアル 01 の第 2 節、または下記の 2.1 に従う |
| `calibration_dataset`(`dx_com/` 配下のサンプル画像) | `dx-compiler/example/2-download_sample_calibration_dataset.sh` | そのスクリプトをターミナルで一度実行する |

次のセルは `tutorial_paths.py` を通じて(環境変数、`config.json`、自動検出の順で)DX-All Suite を探し、見つかった内容を表示して、このチュートリアルのワークスペースを準備します。要件が不足していても停止しません。停止するチェックは、インストールオプションの説明の後、2.2 節にあります。

<!-- cell: 16e77545 src: 141db488f9 -->
### 2.1 DX-COM のインストールオプション

DX-COM は Python パッケージであり、**専用の Python 仮想環境へのインストールを強く推奨します**。仮想環境は DX-COM とその依存関係をシステムの Python および Jupyter カーネルから分離するため、バージョンの競合を減らし、コンパイラ環境の再現や削除を容易にします。

最も簡単なスタンドアロンインストールは、[PyPI](https://pypi.org/project/dx-com/) で公開されているパッケージを使う方法です。別のターミナルで次のコマンドを実行してください。

```bash
# 1. Create a dedicated environment.
python3 -m venv ~/venv-dx-com

# 2. Activate it in the current terminal.
source ~/venv-dx-com/bin/activate

# 3. Install DX-COM from PyPI.
python -m pip install --upgrade pip
pip install dx-com

# 4. Confirm that the virtual environment owns the commands.
which python
which dxcom
dxcom --version
```

2 つの `which` コマンドは `~/venv-dx-com/` 配下のパスを表示するはずです。新しいターミナルを開くたびに `source ~/venv-dx-com/bin/activate` を再度実行し、終了するときは `deactivate` を実行してください。

このノートブックは `dx-all-suite/dx-compiler/install.sh` が `venv-dx-compiler-local` に作成した専用環境を使用するため、そのコンパイラは Jupyter カーネルの Python 環境に依存しません。上記の PyPI の手順は、DX-COM を個別にインストールする場合に推奨される代替手段です。

> `dx-com` をインストールするとコンパイラパッケージと `dxcom` コマンドが提供されます。完全なコンパイルとデプロイのワークフローには、引き続き DEEPX SDK の残りの部分とサポート対象の Linux x86-64 ホストが必要です。

<!-- cell: c7a528a2 src: dd9c03046a -->
### 2.2 DX-COM インストールの検証

次のセルは、まず第 2 節の 2 つの要件が揃っていることを確認します。いずれかが不足している場合は、実行すべき正確な手順を表示して停止します。その手順を完了してから、セルを再実行してください。

セルの残りの部分は、次のターミナルコマンドと同等です。

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
dxcom --version
dxcom -h
```

`<DX_ALL_SUITE_DIR>` は、上のセットアップセルが表示した場所です。

<!-- cell: f8556d44 src: 6e8fd67cf4 -->
## 3. 小さな ONNX モデルのエクスポート

このプロジェクトは `uv` で管理された Jupyter 環境を使用しており、`pip` モジュールが含まれていない場合があります。

`uv pip install --python "{sys.executable}"` は、現在の Jupyter カーネルに明示的にパッケージをインストールします。 

これらのパッケージは ONNX のエクスポートと検査に使用します。DX-COM 自体は引き続き `DX_COMPILER_VENV` から実行されます。

`onnxscript` は、最近の torch バージョン(2.9 以降)の ONNX エクスポーターに必要です。`--extra-index-url` は `uv` に CPU 専用の PyTorch ホイールを指定します。エクスポートに GPU は不要で、CPU ホイールは数 GB ではなく数百 MB で済みます。すでにインストールされている torch はそのまま残されます。

<!-- cell: 206f9840 src: a88480414f -->
### 3.1 ONNX コントラクトの検証

JSON ファイルの入力名は、ONNX グラフの入力名と完全に一致している必要があります。形状推論と `onnx.checker` により、コンパイル前に多くのエクスポートエラーを検出できます。

<!-- cell: 96ab8536 src: 372ae5a1e1 -->
ONNX ファイルを [Netron](https://netron.app/) で開いて、テンソル名、形状、演算子を確認することもできます。コンパイル済みの `.dxnn` については、`--export_html` で生成される DX-COM の HTML サマリー、または DX-TRON(チュートリアル 01、2.3 節)を使用してください。DX-TRON は引き続き SDK と共に配布されていますが、積極的なメンテナンスは行われていないため、HTML サマリーが推奨されるレポートです。

<!-- cell: 0898f061 src: 1303aaeedb -->
## 4. キャリブレーション設定の作成

キャリブレーション画像は、量子化プロセス中に高い精度を維持するために非常に重要です。

<img src="assets/calibration.jpeg" style="max-width: 1000px;" alt="量子化中に使用される代表的なキャリブレーション画像">

設定は、モデルの入力コントラクトと、元画像を入力テンソルに変換する方法の両方を記述します。

| フィールド | 目的 |
|---|---|
| `inputs` | 正確な ONNX 入力名と静的な形状 |
| `calibration_method` | 量子化範囲の推定に使うオブザーバー |
| `calibration_num` | 使用する代表サンプルの数 |
| `default_loader.dataset_path` | キャリブレーション入力を含むディレクトリ |
| `preprocessings` | 画像からテンソルへの順序付きの変換 |

前処理の順序は重要です。値は、元のモデルの学習時および評価時に使われた前処理と一致している必要があります。

<!-- cell: mobilenet-preflight-checklist src: 1110d1b61b -->
### 4.1 コンパイル前のプリフライトチェックリスト

時間のかかる可能性のあるコンパイルを始める前に、次の項目を確認してください。

- ONNX ファイルが `onnx.checker` を通過する
- すべての入力次元が静的である
- ONNX の入力名と形状がコンパイラ JSON と完全に一致する
- キャリブレーションディレクトリが存在し、サポートされるファイルを含んでいる
- キャリブレーションの前処理が、モデルの学習時および評価時の前処理と一致する
- 選択した DX-COM 実行ファイルが存在する

> **重要な要件 — バッチサイズは `1` でなければなりません。** ONNX の入力と JSON の `inputs` エントリの両方が `1` で始まる必要があります(例: `[1, 3, 224, 224]`)。
>
> DXNN モデルは DEEPX NPU の単一サンプル実行コントラクトを対象としているため、異なるバッチ次元や動的なバッチ次元はコンパイルおよびランタイムのコントラクトと一致しません。スループットを向上させるには、複数の推論リクエストを送信するか非同期パイプラインを使用してください。モデルのバッチ次元を増やしてはいけません。

次のセルはこのチェックリストを実行可能なチェックに変換し、要件が満たされていない場合は DX-COM の前で停止します。

<!-- cell: be4871a6 src: 4880623ba4 -->
## 5. DXNN へのコンパイル

### 5.1 コンパイル

`--gen_log` はコンパイラのログを保存し、`--export_html` はモデルサマリーを生成します。このコマンドはエラーを見えるようにしたまま、専用の出力ディレクトリを使用します。出力ディレクトリにすでに DXNN ファイルがある場合、セルは `Skip compilation: found ...` と表示して `dxcom` を再実行しません。強制的に再コンパイルするには、そのディレクトリを削除してください。

次のセルは、次のターミナルコマンドと同等です。

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
dxcom -m models/mobilenet_v2.onnx \
      -c configs/mobilenet_v2.json \
      -o outputs/mobilenet_v2_q_lite \
      --gen_log \
      --export_html
```

> **コードセル末尾の `2>&1 | sed ... | grep ...` について:** `dxcom` は Jupyter が描画できないカーソル移動のエスケープコードでプログレスバーを描くため、そのままでは数百行の空行として表示されます。このフィルターはプログレスバーの行だけを取り除き、`[INFO]`、`[WARNING]`、`[ERROR]` のメッセージはすべて表示されたままになります。ターミナルではフィルターを省略できます。`--gen_log` により完全な出力が `compiler.log` に保存されます。このノートブックのすべてのコンパイルセルで同じフィルターを使用しています。

<!-- cell: c56da884 src: 824e864a9a -->
### 5.2 検査とベンチマーク

`dxparse -v` はコンパイル済みモデルの構造とテンソルのメタデータを表示します。続いて `dxrun --use-ort -t 5` が、合成入力による 5 秒間のベンチマークを実行します。

<!-- cell: 85f6c539 src: 70726cba9e -->
コンパイラが `[INFO] Added nodes` と報告した場合は、どの前処理演算がグラフに挿入されたかを確認してください。ランタイムアプリケーションで同じ正規化、色変換、転置を再度適用してはいけません。

<!-- cell: 8bc0a25d-367f-411d-96b4-5a0ca6d02ad4 src: cfccad8843 -->
### 5.3 性能ベンチマークと精度評価は別物です

| | `dxrun` 合成ベンチマーク | データセットによる精度評価 |
|---|---|---|
| 入力 | 生成されたダミー入力 | 実際のラベル付き検証データセット |
| 主な結果 | ランタイムのスループットとレイテンシ | Top-1、Top-5、mAP、mIoU などのタスク指標 |
| 確認できること | コンパイル済みモデルが実行可能であることと、そのランタイム性能 | 前処理、推論、後処理、予測品質 |
| 証明**できない**こと | モデルの精度 | 最終的なアプリケーション負荷におけるデプロイ時のレイテンシ |

> **コンパイルの成功と高速な `dxrun` の結果は、モデルの精度を証明しません。**

精度の測定については、[DEEPX-AI/dx-modelzoo](https://github.com/DEEPX-AI/dx-modelzoo) とその[モデル評価ガイド](https://github.com/DEEPX-AI/dx-modelzoo/blob/main/docs/source/guides/evaluation.md)を参照してください。DX-ModelZoo は、データセット、前処理、ランタイムプロファイル、後処理、評価器をモデル設定を通じて接続し、タスク固有の指標を報告します。ONNX と DXNN の結果を比較する際は、同じラベル付き検証セットを使用してください。

<!-- cell: 74b9a787 src: d8d71d6109 -->
## 6. DEEPX Model Zoo のモデルをコンパイルする

[DEEPX Model Zoo](https://developer.deepx.ai/modelzoo/) は、検索可能なモデルのメタデータと、ONNX モデル、DXNN モデル、サポートされる量子化バリアント用のコンパイラ JSON ファイルなどのダウンロード可能な成果物を提供します。

この演習では、ONNX ファイルが小さい **Resnet50** を使用します。より大きなモデルでもワークフローは同じです。

1. モデルと量子化バリアントを選択する
2. その ONNX と対応する JSON をダウンロードする
3. ONNX の入力コントラクトを確認する
4. 環境固有の JSON の値を調整する
5. 新しい出力ディレクトリにコンパイルする

<!-- cell: 0d7e5ebb-7b55-4331-ad08-ef736c05424c src: 683283911f -->
### 6.1 Resnet50 の ONNX ファイルと dxcom 設定ファイル(json)のダウンロード

<!-- cell: 250b29cb src: a1a1b7cc23 -->
### 6.2 dxcom 設定ファイル(json)のキャリブレーションパスの更新

Model Zoo の JSON ファイルには、公開されたモデルを生成する際に使用されたキャリブレーション設定が含まれています。そのデータセットパスはビルド環境のものであり、通常はお使いのコンピューターには存在しません。モデル固有の前処理はそのまま残し、データセットパスをローカルの代表的なデータセットに置き換えてください。

ここで SDK のサンプル画像を使うのは、コンパイラのワークフローを再現可能にするためだけです。精度に関する判断には、実際のデプロイ領域のサンプルを使用してください。

ダウンロードしたファイルは `configs/resnet50_224x224.modelzoo.json` として変更されずに残り、調整したコピーは `configs/resnet50_224x224.local.json` に書き込まれます。2 つを分けておくことで、ダウンロードセルを再実行しても編集内容が上書きされることはなく、`wget --continue` が編集済みファイルに追記することもありません。

<!-- cell: b45307f5-9e35-4bcb-9e33-7814434037be src: 3464f02b41 -->
### 6.3 コンパイル

次のセルは、次のターミナルコマンドと同等です。

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
dxcom -m models/resnet50_224x224.onnx \
      -c configs/resnet50_224x224.local.json \
      -o outputs/resnet50_224x224_q_lite \
      --gen_log \
      --export_html
```

5.1 と同様に、出力ディレクトリにすでに DXNN ファイルがある場合、セルは `dxcom` をスキップし、プログレスバーはフィルターで除去されます。

<!-- cell: dbd331d3-99c3-4615-acb9-4ea826e758fd src: 29085b4480 -->
### 6.4 検査とベンチマーク

<!-- cell: 9236bfd1 src: 5dfd2e5d63 -->
### 6.5 最新の HTML コンパイルレポートを開く

DX-COM は `--export_html` が有効な場合に HTML モデルサマリーを生成します。次のセルは、このチュートリアルの出力ディレクトリ配下で最も新しく生成されたレポートを見つけ、新しいブラウザータブでそれを開くボタンを表示します。

<!-- cell: eb56f5d4 src: 9857c6dd8b -->
## 7. トラブルシューティング

| 症状 | 表示される内容 | 考えられる原因 | 対処 |
|---|---|---|---|
| `dxcom: command not found` | ターミナルで | DX-COM 環境が有効化されていない | まず `source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate` を実行する。セルは各 `!` 行でこれを行っている |
| 入力名のエラー | `KeyError` または `input ... not found in model` | JSON の `inputs` が `model.graph.input` と一致していない | 第 4 節の PASS チェックと名前を比較し、JSON を修正する |
| 動的形状またはバッチのエラー | `Dynamic shape is not supported` またはバッチサイズのエラー | ONNX エクスポートで動的な軸、または 1 より大きいバッチサイズが使われた | バッチサイズ 1 の静的形状でエクスポートする |
| キャリブレーションファイルがない | `dataset_path` 配下で `No images found` | パスまたはファイル拡張子が誤っている | JSON の `dataset_path` と `file_extensions` を確認する |
| JSON の変更が反映されない | `Skip compilation: found ...` | 出力ディレクトリにすでに DXNN がある | `workspace/outputs/<dir>` を削除してコンパイルセルを再実行する |
| ONNX エクスポートが失敗する | `ModuleNotFoundError: No module named 'onnxscript'` | torch 2.9 以降の ONNX エクスポーターには `onnxscript` が必要 | 第 3 節のインストールセルを再実行する(`onnxscript` がインストールされる)。あるいは `torch.onnx.export` に `dynamo=False` を渡す |
| コンパイルセルが venv を見つけられない | `{DX_COMPILER_VENV}/bin/activate: No such file or directory` | 前のセルが失敗したためコマンド内の変数が未定義になり、IPython がコマンド全体を展開せずに残している | 上にスクロールして最初に失敗したセルを修正して再実行し、その後コンパイルセルを再実行する |
| `wget` が `416 Requested Range Not Satisfiable` を報告する | ダウンロードセルの再実行時 | `--continue` がすでに完全なファイルを見つけた | 対処不要。ファイルは完全である |
| ランタイムで予測が誤っている | `dxcom` は成功したのに精度が低い | アプリケーションの前処理が JSON と異なる | チャンネル順、スケーリング、正規化、リサイズポリシーを確認し、ラベル付きデータで精度を測定する |

各実験はそれぞれ専用の出力ディレクトリに保存してください。これによりコンパイラレポートとバイナリの追跡が容易になります。

<!-- cell: de985ffc-1b44-4182-94cd-8a16c7dfb545 src: 335b61b8ab -->
## 8. まとめ

### 8.1 完了したワークフロー

**PyTorch モデル**  
→ **ONNX のエクスポートと検証**  
→ **JSON 設定とキャリブレーションデータ**  
→ **DX-COM によるコンパイル**  
→ **DXNN の検査とベンチマーク**

<img src="assets/dx-compile-progress.png"
     style="max-width: 1000px; width: 100%;"
     alt="DX-COM コンパイルワークフロー">

### 8.2 入力と出力

| ステージ | 主な成果物 | 検証方法 |
|---|---|---|
| モデルの準備 | `mobilenet_v2.onnx` | `onnx.checker` |
| コンパイラ設定 | `mobilenet_v2.json` | プリフライトチェックリスト |
| コンパイル | `mobilenet_v2.dxnn` | 出力ファイルの確認 |
| 構造の検査 | DXNN メタデータ | `dxparse -v` |
| 性能の確認 | ダミー入力によるベンチマーク | `dxrun` |
| 精度評価 | ラベル付き検証データセット | DX-ModelZoo |

### 8.3 完了チェックリスト

- [ ] PyTorch モデルを ONNX にエクスポートする
- [ ] ONNX の入力名と静的形状を確認する
- [ ] バッチサイズが `1` であることを確認する
- [ ] キャリブレーション設定を作成する
- [ ] ONNX を DXNN にコンパイルする
- [ ] `dxparse` でコンパイル済みモデルを検査する
- [ ] `dxrun` でランタイム性能を測定する
- [ ] ラベル付きデータセットでモデルの精度を評価する

> **覚えておいてください:** コンパイルの成功と高速な `dxrun` の結果は、
> モデルの精度を証明しません。精度は代表的なラベル付きデータで
> 別途測定する必要があります。

### 8.4 次へ

**中級チュートリアル**に進んで、次の内容を学びましょう。

- キャリブレーションデータセットの設計
- Type 0 と Type 1 のハードウェア PPU
- PPU ありと PPU なしの性能比較
- YOLO26 TopK ベースの最適化
