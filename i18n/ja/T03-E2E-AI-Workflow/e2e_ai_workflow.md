<!-- i18n source: notebooks/T03-E2E-AI-Workflow/e2e_ai_workflow.ipynb -->
<!-- i18n lang: ja -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 2dbf2d92-815e-4a2e-b468-26f71cd8dbe3 src: e823f81eaf -->
# DEEPX Tutorial 03 - DEEPX NPU を使った AI プロジェクトワークフロー

<!-- cell: 74c3b426-7d6e-4844-bf75-737a94d1055f src: 7da7d09c6b -->
この 3 つ目のチュートリアルでは、DEEPX ハードウェア上に AI モデルをデプロイするためのエンドツーエンドのワークフロー全体を説明します。

フォークリフトと作業者を検出するモデルを学習し、DX-Compiler ツールを使って DXNN 形式に変換し、最終的な AI アプリケーションを DEEPX NPU 上で実行します。この一連の流れを通じて、DEEPX NPU 開発パイプラインの全体像を把握できます。 

<!-- cell: 38bde5ed-95d6-4750-a6fb-174273a70167 src: 7bfac3c725 -->
## 学習目標

このチュートリアルを終えると、次のことができるようになります。

- モデルの選定から NPU へのデプロイまで、AI プロジェクト全体の流れをたどる
- 独自に学習した ONNX モデルを、`dxcom` と Model Zoo の JSON 設定を使ってコンパイルする
- 関係する 2 つの JSON ファイル(コンパイラ設定と DX-APP ランタイム設定)を説明する
- 標準の DX-APP 実行ファイルを使って、独自の 2 クラス検出器を画像と動画に対して実行する

<!-- cell: 445ad29e-2955-4f2e-beac-2e2a34bbe81f src: 6275be0099 -->
## ハンズオンプロジェクトの概要

<!-- cell: e37c1d96-a612-4246-af9c-ebf798355fba src: 08e68a9f0e -->
- **検出クラス**: Forklift、Worker
- **ベース AI モデル**: YOLOv7
- **データセット**: [Kaggle](https://www.kaggle.com/datasets/hakantaskiner/personforklift-dataset/data) の Forklift & Worker 画像 1448 枚
- **学習**: GPU メモリ 24 GB 以上の NVIDIA GPU(学習には高性能な GPU が必要ですが、デプロイには DEEPX NPU だけで十分です)
- **推論 NPU**: `DX-M1`
- **AI アプリケーション**: DX-APP の yolo デモを修正して再利用
- **期待される出力**:
<img src="assets/detection-goal.jpg" style="max-width: 1200px;">

<!-- cell: e8217ad7-fc19-421b-967d-98ff2dfbc762 src: 73e0f10840 -->
## AI ワークフローの概要

<!-- cell: 955d81f9-0699-45e6-ba37-e27a481ad370 src: 2d9b410433 -->
この図は、AI プロジェクトの一般的なワークフローを示しています。

目標を定義し、データを収集してラベル付けし、モデルを学習します。
DX-Compiler は、DX NPU 向けにモデルをより高速かつ軽量(INT8)にするのを助けます。
最後のステップは、DX-APP または DX-STREAM を使ってモデルを DEEPX NPU にデプロイすることです。

各ステップは、作業者とフォークリフトの検出のような実世界の AI ソリューションへとつながっていきます。

  <img src="assets/workflow2.jpg" style="max-width: 1200px;">

<!-- cell: 1b3d8491-f89b-4fbf-b601-2cfc061e96a5 src: 22621a4e9d -->
## 前提条件

- チュートリアル 01 が完了していること: DX-COM、DX-RT、DX-APP がインストールされ、NPU が認識されている。
- 第 4 節(コンパイル)には DX-COM とキャリブレーションデータセットが必要です。または、オプション B でコンパイル済みモデルをダウンロードします。
- 第 5 節には DEEPX NPU と、動画結果を表示するためのディスプレイが必要です。
- 学習(第 3 節)は任意で、24 GB メモリの NVIDIA GPU が必要です。Colab ノートブックへのリンクはその節にあります。
- ダウンロード: `cs.deepx.ai` から ONNX モデル(139 MB)、コンパイル済み DXNN(71 MB)、テスト動画(19 MB)。任意の YOLOv7 コンパイルには 10 分以上かかります。
- このチュートリアルがダウンロードまたは生成するものはすべて `notebooks/T03-E2E-AI-Workflow/workspace/` に置かれ、git では無視されます。SDK のチェックアウトはクリーンなままです。

次のセルは SDK の場所を特定し、これらの要件を確認してステータス表を表示します。`MISSING` と表示された項目には、それを提供する手順が併記されます。

<!-- cell: 691e8f9d-0115-48d7-a02a-1b64c84c3b72 src: 92bb42d767 -->
## 1. AI ワークフロー - ユースケースに基づくモデル選定

<!-- cell: afa66140-dbee-4bf2-a9ce-82f4316a8512 src: 9e254de77f -->
AI プロジェクトを始めるには、ユースケースに合った AI モデルを選ぶ必要があります。

このチュートリアルの目標は、フォークリフトと作業者を検出することです。
物体検出でよく知られたモデルである YOLOv7 を使用します。

- Forklift & Worker を検出するために YOLOv7 を選択
- YOLOv7 の詳細: [リンク](https://docs.ultralytics.com/models/yolov7/)
- YOLOv7 の使い方: [リンク](https://github.com/WongKinYiu/yolov7)

<!-- cell: 0a81b7c5-ec05-412b-bc79-338f77d02b18 src: dbe5a538f7 -->
## 2. AI ワークフロー - データ準備とアノテーション

<!-- cell: ac53d3cc-8142-4d59-ac44-fb7f2f836e74 src: 0be5dc9c10 -->
Kaggle からフォークリフトと人物のラベル付きデータセットをダウンロードします。
 - 参考: [Kaggle リンク](https://www.kaggle.com/datasets/hakantaskiner/personforklift-dataset)

<!-- cell: e071e952-82bb-4cf9-8c88-6a007b150d3b src: 0aadb5e072 -->
## 3. AI ワークフロー - 学習

<!-- cell: cbd31d3c-e849-49d3-b7f4-10efefca0861 src: 88b6d45076 -->
モデルを効率よく学習するには、24GB 以上のグラフィックメモリを搭載した GPU を使用してください。

 - YOLOv7 の学習方法: [リンク](https://colab.research.google.com/drive/1dAdjJuhXqFM_Qcd0QqAn7_AGx7abA5aX?usp=sharing)

学習はここではなく、その Colab ノートブックで行います。その成果物であるエクスポート済み ONNX ファイル `yolov7-forklift-person.onnx` は第 4 節でダウンロードできるため、GPU がなくても先に進めます。学習から引き継がれる点が 1 つあります。データセットのラベルにおけるクラス順序は `0 = Forklift`、`1 = Worker` であり、5.1 節の `class_names` は正確にこの順序で記載する必要があります。

<!-- cell: f7dd943a-796f-4aec-8341-a524c71c4e2f src: 9802354bde -->
## 4. AI ワークフロー - DX-Compiler による最適化

<!-- cell: 6347d2b0-3fd8-4a96-bc0e-84c5885586eb src: 70970cc292 -->
学習済みの AI モデルを DXNN 形式にコンパイルしましょう。

全体の流れは次のとおりです。
1. pytorch フレームワークで学習済みモデルを用意する
2. ONNX 形式に変換する
3. ONNX を DXNN にコンパイルする(DX-Compiler の詳細はユーザーガイド [こちら](https://developer.deepx.ai/download/?id=581) を参照してください)
                                                                                             
> **注:** ユーザーガイドをダウンロードするには、まず https://developer.deepx.ai/ にログインする必要があります。

<img src="assets/dx-com-workflow.jpg" style="max-width: 1200px;">

<!-- cell: bdca7205-5cef-453c-863a-794925e61820 src: 0df77eb438 -->
DX-Compiler のソース構成は次のとおりです。
```bash
dx_com
 ├── calibration_dataset   # Dataset used to optimize model accuracy
 └── sample_models         # Sample configuration file and ONNX files 
```

<!-- cell: 6655c5ab-7718-4a75-a305-54516a723d1d src: eb8bb286ef -->
### 4.1 エクスポート済み ONNX モデルと YOLOv7 用 DX-Compiler 設定の準備

<!-- cell: a2886542-8a78-4e17-801b-121deb3b41e0 src: 9e9aa01488 -->
独自の YOLOv7 ONNX モデルは直接ダウンロードされます。このチュートリアルには YOLOv7 の Q-Lite JSON 設定のコピーも含まれているため、ブラウザーのダウンロード先を前提とせずにノートブックを先頭から最後まで実行できます。

元の設定は [DEEPX Model Zoo](https://developer.deepx.ai/modelzoo/) から取得したものです: Object Detection、YOLOv7、Q-Lite JSON ダウンロード。

<img src="assets/sc-modelzoo-yolov7.png" style="max-width: 1000px;">

  
  **注:** DEEPX Model Zoo は、DEEPX が検証した AI モデルとコンパイラ設定を提供します。

<!-- cell: acb10634-544d-4ace-ba8a-0fd870fbf87e src: e1d203991b -->
### 4.2 YOLOv7 の json ファイルを独自環境向けに修正する

<!-- cell: 97b44e9e-20ad-4a1a-be68-321b3e981ec2 src: 618757da53 -->
### 4.3 json 設定を参照して ONNX を DXNN にコンパイルする

<!-- cell: t03-dxcom-environment-guide src: 8e71d364a7 -->
DX-Compiler は、SDK のインストール時に作成された専用の Python 仮想環境を使用します。JupyterLab は別の `dx-tutorials` 環境で動作しているため、通常はノートブックの `PATH` から `dxcom` を利用できません。

次のコードセルは `config.json` から DX-Compiler のパスを導出し、DX-Compiler 環境を有効化し、`dx_com` ワークスペースに移動して、同じシェル内で `dxcom` を実行します。各 `!` コードセルは新しいシェルを起動するため、有効化はセルをまたいで持続しません。このため、`dxcom` を実行するすべてのノートブックセルで有効化コマンドを繰り返しています。

このチュートリアル以外では、ターミナルを開いて環境を一度だけ有効化してください。

```bash
cd <DX_ALL_SUITE_DIR>/dx-compiler
source venv-dx-compiler-local/bin/activate
cd dx_com
dxcom -h
```

有効化した環境は、`deactivate` を実行するかターミナルを閉じるまで、そのターミナル内で有効なままです。

<!-- cell: 87678442-bbd9-47c0-a4cd-631987888e24 src: aed9a3d45a -->
DXNN を取得する 2 つの方法のいずれかを選びます。次のセルで `COMPILE_OPTION` を設定してください。`"A"` は ONNX を自分でコンパイルし(10 分以上)、`"B"` は DEEPX が同じコマンドでコンパイルした DXNN をダウンロードします(71 MB)。どちらのセルも、ワークスペースに DXNN がすでに存在する場合は自動的にスキップされます。

<!-- cell: 39c4982e-a87b-48c0-8a3f-1ed63295c4cd src: cfeb99e9ac -->
#### 4.3.1 オプション A: `dxcom` で ONNX をコンパイルする

これはチュートリアル 01 の 2.2.2 節と同じコマンドで、独自の ONNX と編集済みの JSON を使います。`-o` はワークスペースを指しているため、SDK ツリーには何も書き込まれません。

```bash
cd <DX_ALL_SUITE_DIR>/dx-compiler
source venv-dx-compiler-local/bin/activate
cd dx_com
dxcom -m <workspace>/yolov7-forklift-person.onnx -c <workspace>/yolov7-forklift-person.json -o <workspace>/output --gen_log
```

> **注:** コンパイルにはホストによって **10 分以上**かかります。このセルは `COMPILE_OPTION` が `"A"` の場合にのみ実行されます。

<!-- cell: ddb0f169-582f-4a96-9c49-d8325df7e616 src: 6d1d5efeaf -->
#### 4.3.2 オプション B: コンパイル済み DXNN をダウンロードする

DEEPX は同じ ONNX を同じ JSON でコンパイルし、その結果を公開しています。ダウンロードすれば、数秒で同一のファイルが得られます。

```bash
cd <workspace>/output
wget -nc https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/yolov7-forklift-person.dxnn
```

<!-- cell: 6daf1b20-9182-4a77-82d0-8188d9c6e257 src: c0768f5b3c -->
## 5. AI ワークフロー - DEEPX NPU へのデプロイ

<!-- cell: 4d54f93a-31ff-4f9c-8fd5-1de4776cc622 src: f2166618d1 -->
### 5.1 独自の YOLOv7 モデル向けに DX-APP を設定する

現在の DX-APP では、独自のクラス数に対応するために C++ ソースを修正する必要はありません。YOLOv7 ファクトリはランタイム設定ファイルから `num_classes` と `class_names` を読み込み、後処理器が検出結果をデコードする際にそれらの値を使用します。

> **参考:** この節は [DX-APP YOLO Customizing Guide](https://github.com/DEEPX-AI/dx_app/blob/main/docs/source/docs/12_DX-APP_YOLO_Customizing_Guide.md#step-3-tune-configjson)(チェックアウト内の `dx_app/docs/source/docs/12_DX-APP_YOLO_Customizing_Guide.md`)の **Step 3「Tune `config.json`」**に沿っています。このガイドには YOLO 後処理器が読み込むすべてのキー(`obj_threshold`、`score_threshold`、`nms_threshold`、`num_classes`、`class_names`、`anchors`、`strides`)が列挙され、3 クラスモデル向けの同じファイルの例が示されています。同梱の YOLOv7 サンプルは既定値を `src/cpp_example/object_detection/yolov7/config.json` に持っており、以下で書き出すファイルは 2 クラスの Forklift と Worker モデル向けにそれらを上書きします。

このチュートリアルで使う 2 つの JSON 設定ファイルを混同しないでください。

| 設定 | 使用する側 | 目的 |
|---|---|---|
| `yolov7-forklift-person.json` | `dxcom -c` | モデルのコンパイル、入力形状、前処理、キャリブレーションデータを記述します。 |
| `yolov7-forklift-runtime.json` | DX-APP `--config` | ランタイムの後処理しきい値、クラス数、表示ラベルを記述します。 |

コンパイル済みモデルのデコード後の出力形状は `[1, 25200, 7]` です。各行には 4 つのボックス値、1 つの objectness 値、2 つのクラススコアが含まれます: `4 + 1 + 2 = 7`。生の検出ヘッドは 21 チャネルを持ちますが、これは各ヘッドが 3 つのアンカーを使うためです: `3 × (5 + 2) = 21`。

<!-- cell: c8598424-1ee1-4413-9ecd-ab7f6fd6e2d6 src: 37fd0c267c -->
#### ランタイム設定フィールド

| フィールド | 説明 |
|---|---|
| `obj_threshold` | objectness スコアがこの値未満の候補を除外します。 |
| `score_threshold` | 最終的なクラス信頼度がこの値未満の検出を除外します。YOLOv7 では、最終信頼度は objectness とクラススコアに基づきます。 |
| `nms_threshold` | 重複する検出を取り除くために Non-Maximum Suppression が使用する IoU しきい値です。値が小さいほど重複ボックスをより積極的に除去します。 |
| `num_classes` | モデルが出力するクラス数です。このモデルは 2 クラスを出力するため、値は `2` である必要があります。 |
| `class_names` | 各クラス ID に対して表示されるラベルです。配列の順序は学習時に使用したクラス順序と正確に一致している必要があります。 |

このモデルでは、クラス ID `0` が `Forklift`、クラス ID `1` が `Worker` です。ラベルの文字列だけを変更してもモデルの動作は変わらず、検出されたクラス ID の表示方法が変わるだけです。`num_classes`、`class_names`、モデル出力の間に不一致があると、誤ったデコードや誤ったラベル表示が発生することがあります。

> **注:** ガイドが警告しているとおり、`num_classes` はコンパイル済みモデルの出力と一致している必要があります。クラス数はコンパイル時に固定されるため、ランタイム JSON だけを変更してもテンソル形状の不一致は解消できません。

<!-- cell: 12ba0bdd-1df3-47d2-a454-a1bc39c8768c src: d725d9b9f9 -->
### 5.2 YOLOv7 実行ファイルの確認

ランタイム設定は、DX-APP が YOLOv7 後処理器を生成する前に `--config` で読み込まれます。C++ ソースは変更しないため、`bin/yolov7_async` がすでに存在していれば DX-APP の再ビルドは不要です。

モデルの実行時、DX-APP は `Config loaded: yolov7-forklift-runtime.json (4 keys)` と記録します。この数はスカラー値のみを数えたもので、`class_names` はリストとして別に保存されるため、5 つの設定すべてが有効になっています。

実行ファイルがない場合は、ターミナル(File > New > Terminal)でこのターゲットだけをビルドしてください。

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_app
./build.sh --target yolov7_async
```

<!-- cell: aef5fbb5-57c5-425b-bb18-be6c09f84008 src: 548c63ca47 -->
### 5.3 画像で独自モデルを実行する

独自の DXNN とそのランタイム設定を標準の YOLOv7 実行ファイルに渡します。共有の DX-APP ソースは変更しません。セルはヘッドレス(`--no-display --save`)で動作するためディスプレイがなくても実行でき、保存された結果を下に表示します。代わりに DX-APP のウィンドウを開くには `--no-display` を外してください。その場合、ウィンドウを閉じるまでセルは実行中のままになります。

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_app
./bin/yolov7_async -m <workspace>/output/yolov7-forklift-person.dxnn \
    --config <workspace>/yolov7-forklift-runtime.json \
    -i <tutorial>/assets/forklift-worker.png --no-display --save --save-dir <workspace>/outputs
```

<!-- cell: c6d86bb5-c77a-4144-8072-4c30ac0e0386 src: 2dbebc5c03 -->
### 5.4 動画で独自モデルを実行する

動画での実行は、同じコマンドに `-v` を付けたものです。クリップの長さは 80 秒(2001 フレーム)です。`SHOW_WINDOW = False`(既定)では、アプリケーションはヘッドレスで動作し、レンダリング済みの動画をワークスペース配下に書き出し、**完了するまで何も出力しません**。DX-M1 では約 20 秒かかるため、セルが固まっているわけではありません。その後、保存された動画がノートブックに表示されます。代わりに DX-APP のウィンドウでライブ表示したい場合は `SHOW_WINDOW = True` に設定してください(ウィンドウを閉じるまでセルは実行中のままになります)。

```bash
./bin/yolov7_async -m <workspace>/output/yolov7-forklift-person.dxnn \
    --config <workspace>/yolov7-forklift-runtime.json -v <workspace>/forklift-worker.mp4 \
    --no-display --save --save-dir <workspace>/outputs
```

<!-- cell: 0fefce5e-afd0-44ab-a03f-ed3a18e92272 src: 015011ee11 -->
## 6. トラブルシューティング

| 症状 | 表示される内容 | 考えられる原因 | 対処 |
|---|---|---|---|
| ダウンロードが失敗する | `wget: unable to resolve host` または `404` | インターネットに接続できない、またはアーカイブが移動した | 接続を確認して再試行する。モデルについては 4.3.2 のオプション B を使う |
| `dxcom` がキャリブレーション画像を見つけられない | `No images found` またはキャリブレーションエラー | JSON の `dataset_path` が誤っている | 4.2 のセルを再実行する。`dx_com/calibration_dataset` の絶対パスが書き込まれる |
| コンパイルに非常に時間がかかる | エラーなしで 10 分以上 | 640x640 の YOLOv7 は CPU 負荷が高い | 待つか、オプション B(コンパイル済み DXNN)を使う |
| `No DXNN at .../workspace/output/...` | 4.3.2 の後に `FileNotFoundError` | どちらのオプションでもファイルが生成されていない | `COMPILE_OPTION` を `"A"` または `"B"` に設定し、対応するセルを再実行する |
| 5.3 または 5.4 でウィンドウが表示されない | セルが性能サマリーだけで終了する | セルは既定でヘッドレスで動作する | `workspace/outputs/` 配下の保存結果を開くか、デスクトップセッションで `--no-display` を外す / `SHOW_WINDOW = True` に設定する |
| ボックスのラベルが誤っている | フォークリフトが Worker とラベル付けされる、またはその逆 | `class_names` の順序が学習時の順序と異なる | ランタイム JSON の順序を修正して再実行する。`num_classes` は 2 のままにする |

<!-- cell: 2df122e0-2ea3-4c2e-b294-6fee2d7c5b24 src: 4848d5487c -->
## 7. まとめ

### 7.1 完了したワークフロー

```text
Choose a model (YOLOv7) and a labeled dataset
       │
       ▼
Train on a GPU (Colab notebook)
       │
       ▼
Export ONNX and compile to DXNN with dxcom
       │
       ▼
Describe classes and thresholds in a DX-APP runtime config
       │
       ▼
Run the custom model on the NPU with yolov7_async
```

### 7.2 完了チェックリスト

- [ ] コンパイラ用に ONNX モデルと Model Zoo の JSON 設定を準備した
- [ ] `dataset_path` をローカルのキャリブレーションデータセットに向けた
- [ ] `dxcom` でモデルをコンパイルした、またはコンパイル済み DXNN をダウンロードした
- [ ] `num_classes` と `class_names` を含む DX-APP ランタイム設定を作成した
- [ ] 独自モデルを画像と動画に対して実行した

> **次へ:** チュートリアル 04 に進み、同じ Forklift と Worker の検出器を DX-STREAM で GStreamer パイプラインに組み込んでみましょう。
