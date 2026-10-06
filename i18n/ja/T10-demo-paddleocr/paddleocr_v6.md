<!-- i18n source: notebooks/T10-demo-paddleocr/paddleocr_v6.ipynb -->
<!-- i18n lang: ja -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: t10-v6-001 src: b22c49c292 -->
# DEEPX Tutorial 10 - DEEPX NPU 上の PP-OCRv6

このチュートリアルでは PP-OCRv6 を紹介し、DEEPX NPU 上での ONNX から DXNN までの完全な OCR ワークフローを実演します。コンパイルと授業での作業時間を短く保てるため、対象ティアとして PP-OCRv6_tiny を選択しています。

ワークフローは、個別に確認できる次のステージに分かれています。

1. 元の ONNX モデルをダウンロードする
2. 動的な入力次元を固定形状に置き換える
3. 固定形状のモデルが ONNX の結果を維持していることを検証する
4. モデルを DXNN にコンパイルする
5. 検出とテキスト認識を 1 つのアプリケーションとして実行する

<!-- cell: t10-v6-002 src: f4377ce97d -->
## 学習目標

このチュートリアルを終えると、次のことができるようになります。

- PP-OCRv6 のアーキテクチャを説明し、tiny、small、medium のティアから選択する
- 検出 → 認識の OCR パイプラインを説明する
- 1 つの動的な認識モデルが 6 つの固定形状 DXNN モデルになる理由を説明する
- 元の ONNX ファイルを安全にダウンロードして検証する
- 前処理がアプリケーションと一致するキャリブレーション設定を作成する
- 失敗を隠さずに <code>dxcom</code> でモデルをコンパイルする
- 生成された DXNN 成果物を検査する
- レビュー済みのカメラアプリケーションを実行し、実用上の OCR 精度の限界を把握する

<!-- cell: t10-npu-pattern src: deed40455a -->
## このアプリケーションの NPU の使い方

| 項目 | このチュートリアルでは | 学んだ場所 |
|---|---|---|
| エンジン | 6 つの `InferenceEngine` オブジェクトが常駐します。検出器 1 つと認識器 5 つで、テキストの切り出しごとにアスペクト比で選択されます | T06-2 §3 |
| 実行 | 検出は同期的に実行されます(`run()`)。認識器はフレーム内の各切り出しを `run_async()` で実行し、`wait()` で結果を収集します | T06-2 §4, §5 |
| タスクグラフ | コンパイル済みの各モデルには NPU タスクとそれに続く `cpu_0` タスクがあります(DX-COM が CPU に残した演算子を ONNX Runtime が実行します) | T06-1 §6 |
| リクエストあたりの入力 | DET `[1, 640, 640, 3]` UINT8 = 1.2 MB。REC の切り出しは高さ 48 ピクセル、幅 120〜1200 | T06-3 §4 |
| 測定するもの | 検出器と認識器 1 つの `dxrun` スループット(6.3 節)をアプリケーションのフレームあたり時間と比較します。単語の多いフレームは REC リクエストも多くなるため、レイテンシはテキスト数に対して報告します | T06-1 §5 |

<!-- cell: 659aa80e-30b9-4cf8-ae33-1ed4bf0cbd53 src: cf3775d539 -->
## 前提条件

- チュートリアル 01 が完了していること: DX-COM、DEEPX NPU を備えた DX-RT、および `uv`。
- ダウンロード: Hugging Face からの ONNX モデル 2 つ(1.8 MB と 4.5 MB)と、`cs.deepx.ai` からのキャリブレーションアーカイブ(110 MB)。
- 所要時間: 約 15 分。第 6 節の 6 回の `dxcom` コンパイルはそれぞれ数秒(検出器は約 10 秒)です。7.1 節では `workspace/.venv-ocr` 配下に 90 MB の Python 環境を作成します。
- DX-RT 3.4.2 と DX-COM 2.4.1 で検証済みです。モデルはここでコンパイルするため、常にお使いのコンパイラと一致します。
- 第 8 節のカメラアプリケーションには USB カメラとディスプレイが必要です。`--check` スモークテストはどちらがなくても実行できます。

次のセルは SDK を探し、これらの要件を確認してステータス表を表示します。`MISSING` と表示された項目には、それを提供する手順が併記されます。

<!-- cell: t10-v6-overview src: fd7df8aac3 -->
## 1. PP-OCRv6 の概要

### 1.1 OCR とは?

**光学文字認識**(OCR)は、さまざまな種類の文書(スキャンした紙の文書、PDF ファイル、デジタルカメラで撮影した画像)を、編集や検索が可能なデータに変換する技術です。

AI に「目」を与えるものと考えてください。一般的に、次の 2 段階のパイプラインで動作します。
1. テキスト検出: 画像内のどこにテキストがあるかを特定します(その周りにボックスを描きます)。
2. テキスト認識: そのボックス内の文字が何であるかを読み取ります。

<!-- cell: 5b35baad-56a0-4995-aaa9-9b39b7510f74 src: e4261a9a39 -->
### 1.2 PP-OCRv6 とは?

PaddleOCR は、Baidu が PaddlePaddle フレームワークをベースに開発した超軽量のオープンソース OCR システムです。

[PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR) は、PaddleOCR の汎用 OCR モデルファミリーの最新世代です。テキスト検出とテキスト認識の両方に新しい **PPLCNetV4** バックボーンを使用し、エッジデバイスからサーバーまでをカバーする 3 つのデプロイティアを提供します。

[公式の PP-OCRv6 技術ドキュメント](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)に記載されている主な改善点は次のとおりです。

- **スケーラブルな単一モデルファミリー:** tiny、small、medium のティアは 1.5M から 34.5M パラメータまでをカバーします
- **統一された多言語認識:** small と medium は 1 つのモデルで 50 言語をサポートし、tiny は 49 言語をサポートします(日本語は含まれません)
- **テキスト検出の改善:** RepLKFPN がネックのパラメータを削減しつつ受容野を拡大します
- **テキスト認識の改善:** EncoderWithLightSVTR が局所的なコンテキストと大域的なアテンションを組み合わせ、CTC が効率的な並列デコードを提供します
- **特殊シーンのカバー:** 公式の評価には、手書き文字、回転した文字やアート文字、デジタル表示、ドットマトリクス文字、タイヤの刻印、その他の産業用テキストが含まれます

<img src="assets/ppocrv6-backbone.jpg" style="max-width: 1000px; width: 100%;" alt="PP-OCRv6 の認識と検出のための PPLCNetV4 バックボーン設計">

*PPLCNetV4 はタスクに適応したダウンサンプリングを使用します。認識では水平方向のシーケンス情報を保持し、検出ではマルチスケールの特徴マップを生成します。出典: [公式 PaddleOCR PP-OCRv6 ドキュメント](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.md)。*

<!-- cell: t10-v6-tier-comparison src: 998aa1c53c -->
### 1.3 tiny、small、medium

各ティアは同じ PP-OCRv6 設計を異なるスケールで使用します。一般に、大きなティアほど難しいケースでの精度は向上しますが、モデルサイズと計算コストが増加します。

| ティア | パラメータ数 | 想定ターゲット | 検出 Hmean (%) | 認識精度 (%) | Intel Xeon OpenVINO (秒/画像) | NVIDIA A100 PaddlePaddle (秒/画像) |
|---|---:|---|---:|---:|---:|---:|
| **tiny** | 1.5M | エッジ / IoT | 80.6 | 73.5 | **0.20** | **0.13** |
| **small** | 7.7M | モバイル / デスクトップ | 84.1 | 81.3 | 0.59 | 0.25 |
| **medium** | 34.5M | サーバー / 最高精度 | **86.2** | **83.2** | 1.40 | 0.29 |

> **表の読み方:** 精度の値は PaddleOCR 内部のマルチシナリオベンチマークによるものです。速度は、一般画像と文書画像 200 枚に対する画像 1 枚あたりのエンドツーエンドの秒数で、画像 I/O、前処理、後処理、推論を含みます。これらの CPU/GPU の数値はティア間の相対的なトレードオフを説明するものであり、**DEEPX NPU の測定値ではなく**、DX-COM のコンパイル時間を予測するものでもありません。

<img src="assets/ppocrv6-performance.png" style="max-width: 1100px; width: 100%;" alt="公式の PP-OCRv6 検出・認識精度の比較">

*左: テキスト検出の平均 Hmean。右: テキスト認識の加重平均精度。出典: [公式 PaddleOCR PP-OCRv6 ドキュメント](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.md)。*

<!-- cell: t10-v6-tiny-selection src: c7b6b6fe09 -->
### 1.4 このチュートリアルが PP-OCRv6_tiny を選ぶ理由

このチュートリアルでは、授業中のモデル準備とコンパイル時間を最小限に抑えるために **tiny** ティアを選択しています。1.5M パラメータという小ささは、エッジや IoT へのデプロイの自然な出発点にもなります。これはチュートリアルの時間を考慮した判断であり、tiny が最良の本番モデルであると主張するものではありません。測定された精度の向上が追加リソースに見合う場合は、small または medium を選択してください。

tiny には 2 つの重要なトレードオフがあります。

1. 公式の検出精度と認識精度が small および medium より低い
2. 49 言語をサポートし日本語は含まれないのに対し、small と medium は 50 言語をサポートする

> 第 4 節で公式の PP-OCRv6_tiny 検出器と認識器の ONNX モデルをダウンロードします。続く 7.3 節では、ライブデモの前に、アプリケーションの認識辞書、前処理、テンソルコントラクトがこれらのモデルと一致していることを確認します。

<img src="assets/ppocrv6-detection-comparison.jpg" style="max-width: 1000px; width: 100%;" alt="産業用および難しいテキストに対する公式の PP-OCRv6 medium テキスト検出比較">

*この公式の定性的な図は PP-OCRv6_medium を使用しており、難しい検出シナリオを示しています。tiny ティアの精度結果ではありません。出典: [公式 PaddleOCR PP-OCRv6 ドキュメント](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.md)。*

<!-- cell: t10-v6-006 src: 243e5834d1 -->
## 2. チュートリアル環境の準備

このノートブックは、共有のチュートリアルパスヘルパーを通じて <code>config.json</code> を読み込みます。SDK がチュートリアルリポジトリの中にインストールされていることは前提としません。

<!-- cell: t10-v6-008 src: a5b14575cb -->
### 2.1 現在の uv 環境への Python パッケージのインストール

Jupyter 環境は uv で作成されており、<code>pip</code> モジュールが含まれていない場合があります。そのため次のセルでは、<code>%pip</code> の代わりに <code>uv pip install --python ...</code> を使用します。

セルの再実行は安全です。uv は要件をすでに満たしているパッケージを再利用します。

<!-- cell: t10-v6-004 src: 33b7724658 -->
## 3. OCR パイプラインを理解する

PP-OCRv6 は 2 つのモデルステージで動作します。検出してから認識するという同じフローが、すべてのティア(tiny、small、medium)に適用されます。

| ステージ | 入力 | 出力 | 目的 |
|---|---|---|---|
| テキスト検出 | 画像全体、640 × 640 | テキストのポリゴン | テキスト領域を見つける |
| テキスト認識 | テキストの切り出し 1 つ、高さ 48 | 文字列 | ピクセルをテキストに変換する |

<img src="assets/ocr-workflow.jpg" style="max-width: 980px;" alt="PP-OCR ワークフロー">

PaddleOCR を DX NPU に適用するには、次の 4 つのステップが必要です。

1. PaddleOCR の ONNX モデルをダウンロードする

2. 動的な入力形状を固定する

3. DX NPU 向けに ONNX を *.dxnn にコンパイルする

4. DEEPX-SDK で OCR アプリケーションを実装する

<!-- cell: t10-v6-010 src: 1523e85531 -->
## 4. リソースのダウンロードと検査

### 4.1 ONNX モデルのダウンロード

このチュートリアルでは、Hugging Face 上の公式 PaddlePaddle ONNX リポジトリから **tiny** の検出器と認識器を直接ダウンロードします。

- [PP-OCRv6_tiny_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx)
- [PP-OCRv6_tiny_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx)

ダウンロードするのは **DET** と **REC** の ONNX モデルのみです。任意のテキスト行方向分類器は、このチュートリアルでは使用しません。

チュートリアルの結果を再現可能にするため、各 URL は公式リポジトリの特定のリビジョンに固定されています。ダウンロードは HTTPS 証明書を検証し、一時的な <code>.part</code> ファイルに書き込み、SHA-256 チェックサムを検証してから保存先を置き換えます。一致する既存ファイルは再利用され、古いファイルや異なるファイルは再度ダウンロードされます。

<!-- cell: t10-v6-calibration-download src: fcfdb536d1 -->
### 4.2 キャリブレーションデータセットのダウンロード

OCR のキャリブレーション画像をダウンロードして、このチュートリアルディレクトリに展開します。アーカイブは次のディレクトリを作成します。

```text
models/
├── det_dataset/
└── rec_dataset/
```

アーカイブファイルが完了マーカーの役割を果たします。<code>ocr-dataset.tar.gz</code> がすでに存在する場合、ダウンロードと展開の両方がスキップされます。

<!-- cell: t10-v6-013 src: d18bf1f9fc -->
元のモデルでは、バッチサイズと画像の寸法が動的になっています。

動的形状は汎用の ONNX Runtime アプリケーションには便利ですが、DEEPX のコンパイルでは毎回、具体的な入力形状が必要です。

<!-- cell: t10-v6-014 src: f5160e4612 -->
## 5. 動的な入力形状の固定

次の表が、固定形状モデルに関する唯一の信頼できる情報源です。形状は NCHW 順(バッチ、チャンネル、高さ、幅)です。

<!-- cell: b7060f4d-1c5c-4ac6-b461-2eb3a5598a95 src: e31dd6c875 -->
### 5.1 テキスト認識モデルの入力形状を固定する

**なぜ 5 つの異なるモデルが必要なのか?**

NPU は固定の入力形状を必要とするため、短い単語('Hi' など)と長い文を同じ幅 1200 のボックスで扱うと、過剰なパディングが発生し、細部が失われてしまいます。

異なるアスペクト比の「バケット」を作ることで、テキストをより効率的かつ正確に処理できます。そのため、このチュートリアルではアスペクト比の異なる 5 つの独立した認識モデルを使用します(検出モデル 1 つを加えて、合計 6 つの DXNN ファイル)。

各ケースで、検出されたテキストの比率に最も合うモデルを選択して適用します。

<img src="assets/ocr-ratio.png" style="max-width: 800px;">

次の GIF アニメーションは、実際に検出されたテキストの比率に基づいてどのテキスト認識モデルが選ばれるかを示しています。

<img src="assets/ocr-ratio.gif" style="max-width: 800px;">

<!-- cell: t10-v6-016 src: 231600d020 -->
### 5.2 `onnxsim` を使って動的な入力形状を固定する

ノートブックは、現在のカーネルの Python インタープリターを通じて ONNX Simplifier を呼び出します。たとえば、最初の変換は次と同等です。

~~~bash
python -m onnxsim models/det.onnx models/det_fixed.onnx           --overwrite-input-shape x:1,3,640,640
python -m onnxsim models/rec.onnx models/rec_fixed_ratio_2_5.onnx --overwrite-input-shape x:1,3,48,120
python -m onnxsim models/rec.onnx models/rec_fixed_ratio_5.onnx   --overwrite-input-shape x:1,3,48,240
...
~~~

空でない既存の固定形状モデルは確認のうえ再利用されます。

<!-- cell: t10-v6-018 src: 5fa1f3bd19 -->
### 5.3 形状と数値的等価性の検証

構造的な検証だけでは十分ではありません。次のセルでは以下を行います。

1. 固定形状のすべての ONNX モデルを確認する
2. 正確な入力形状を確認する
3. 検出と代表的な 2 つの認識バケットについて ONNX Runtime の出力を比較する

すべての認識バケットは同じ元のグラフから生成されています。チュートリアルの検証を手早く済ませるため、数値的な認識チェックには比率 2.5 と比率 5 のモデルを使用します。

<!-- cell: t10-v6-021 src: 803f189d08 -->
## 6. ONNX を DX NPU 向けの *.dxnn にコンパイルする

### 6.1 DX-COM のキャリブレーション設定を作成する

キャリブレーション時の前処理はアプリケーションの前処理と一致していなければなりません。色順、スケーリング、正規化、レイアウトのいずれかが食い違うと、コンパイルが成功しても精度が低下することがあります。

| モデル | キャリブレーション画像 | リサイズ | 平均 / 標準偏差 |
|---|---|---:|---|
| 検出 | <code>det_dataset</code> | 640 × 640 | ImageNet の値 |
| 認識 | 対応するアスペクト比のバケット | 固定幅 × 48 | [0.5, 0.5, 0.5] |

<!-- cell: t10-v6-024 src: b5cb9cbaae -->
コンパイルの前に、少なくとも 1 つの設定を確認してください。特に NCHW の入力形状、キャリブレーションデータセット、リサイズサイズ、チャネル順、正規化、transpose の順序を確認します。

<!-- cell: t10-v6-026 src: 7df5a7156c -->
### 6.2 ONNX モデルを DXNN にコンパイルする

各モデルはそれぞれ専用のディレクトリに書き出されます。有効な DXNN ファイルがすでに存在する場合はスキップされるため、授業を繰り返しても再コンパイルに時間を取られません。

セルはモデル名を表示したうえで、各モデルに対して正確に次のコマンドを実行します。

~~~bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
dxcom -m <fixed-model.onnx> \
      -c <calibration-config.json> \
      -o <output-directory> \
      --gen_log \
      --export_html
~~~

コンパイラの出力はそのまま表示され、コンパイルに失敗した場合は黙って続行せずにエラーでセルが停止します。

<!-- cell: t10-v6-029 src: 9a976ceadb -->
### 6.3 コンパイル成果物を検証する

完全な OCR アプリケーションには、検出 1 個と認識バケット 5 個、合計ちょうど 6 個の DXNN ファイルが必要です。必要なファイルが 1 つでも欠けていると、このセルは早い段階で失敗します。

<!-- cell: t10-v6-031 src: f143ae02c1 -->
<code>dxparse</code> で 2 つのモデルの役割を確認します。認識モデルはアスペクト比 2.5 のものが代表例で、他の認識ファイルとの違いは主に固定入力幅だけです。

<!-- cell: t10-dxrun-baseline-md src: 97352c10c3 -->
同じ 2 つのモデルについて CLI のベースラインを測定します。どちらのグラフも CPU タスクで終わるため `--use-ort` が必須で、`-v` を付けるとレイテンシの内訳が表示されます。これらの数値は後でアプリケーションと比較します。カメラループは検出した単語ごとに DET 1 回と REC 1 回のリクエストを実行するため、フレーム時間はテキスト量に応じて増えます。

```bash
dxrun -m <workspace>/outputs/paddleocr_v6/det_fixed/det_fixed.dxnn --use-ort -t 3 -v
dxrun -m <workspace>/outputs/paddleocr_v6/rec_fixed_ratio_2_5/rec_fixed_ratio_2_5.dxnn --use-ort -t 3 -v
```

<!-- cell: t10-v6-033 src: 877fbbedf2 -->
## 7. DEEPX-SDK で OCR アプリケーションをテストする

再利用可能な Python アプリケーションは <code>app/</code> にあります。<code>outputs/paddleocr_v6/</code> にコンパイル済みの PP-OCRv6 tiny モデルを使い、各フレームを次のように処理します。

<img src="assets/ppocrv6-camera-pipeline.png" alt="PP-OCRv6 カメラ推論パイプライン" style="max-width: 1000px; width: 100%; height: auto;">

検出器と 5 つの認識エンジンは一度だけロードされ、常駐したままになります。

<!-- cell: t10-v6-034 src: d8ac433600 -->
### 7.1 アプリケーションの依存関係をインストールする

アプリケーションには <code>dx_engine</code>、OpenCV、NumPy、Pillow が必要です。チュートリアル 06-2 と同様に、これらは共有の Jupyter 環境ではなく <code>workspace/.venv-ocr</code> 配下の小さな環境にインストールします。DX-RT パッケージは <code>/usr/share/libdxrt-bin/python</code> にビルド済みの <code>dx_engine</code> wheel を同梱しており、セルはカーネルの Python に合う <code>cpXY</code> タグの wheel を選びます。<code>app/run_camera.sh</code> も同じ環境を使います。セルの再実行は安全で、すでに存在するものはスキップされます。

```bash
uv venv --python <jupyter-python> <T10>/workspace/.venv-ocr
uv pip install --python <T10>/workspace/.venv-ocr/bin/python -r app/requirements.txt \
    /usr/share/libdxrt-bin/python/dx_engine-<version>-cp<XY>-*.whl
```

<!-- cell: t10-v6-036 src: 507594f64b -->
### 7.2 アプリケーションの構成と安全策

この Notebook は、別のランナーを生成する代わりに <code>app/</code> 配下で保守されているファイルを使います。

| ファイル | 役割 |
|---|---|
| <code>app/ocr_engine.py</code> | DXNN のロード、DET の後処理、切り出し、5 バケットの REC ルーティング、CTC デコード |
| <code>app/camera_app.py</code> | CLI の検証、640×480 のカメラキャプチャ、OpenCV プレビュー、BBOX と多言語テキストのオーバーレイ |
| <code>app/run_camera.sh</code> | <code>workspace/.venv-ocr</code> でアプリケーションを起動(<code>PYTHON_BIN</code> で上書き可能) |
| <code>app/assets/ppocrv6_tiny_dict.txt</code> | 6906 クラスの REC 出力に対応する公式 tiny 辞書 |

モデルの欠落、無効なテンソル形状、辞書の不一致、カメラのオープン/読み取り失敗に対して、アプリケーションは明確に失敗します。既存の DXNN ファイルを変更することはありません。

<!-- cell: t10-v6-037 src: 90057954f1 -->
### 7.3 カメラなしでアプリケーションのスモークテストを実行する

<code>--check</code> モードはカメラを開かず、GUI ウィンドウも作成しません。パッケージ化されたアプリケーションをロードし、検出器と 5 つの認識器すべてに合成入力の推論を 1 回通します。これにより、ライブデモの前に NPU ランタイム、入出力テンソルのコントラクト、tiny 辞書のクラス数を検証できます。

下で実行するコマンドは、モデルと辞書のパスを明示して <code>workspace/.venv-ocr/bin/python app/camera_app.py --check</code> を実行するのと同等です。

<!-- cell: t10-v6-039 src: f2bee9052c -->
### 7.4 640×480 のカメラアプリケーションをテストする

ライブアプリケーションは対話型の OpenCV ウィンドウを開きます。Notebook のカーネルをブロックせず、カメラを正しく解放できるように、**JupyterLab のターミナル**で実行してください。`run_camera.sh` は 7.1 で作成した `workspace/.venv-ocr` 環境で起動します。

既定の入力は <code>/dev/video0</code> の 640×480、15 FPS です。プレビューウィンドウで **q** または **Esc** を押すと終了します。別のデバイスを選ぶには <code>--camera /dev/videoN</code> を使います。

<!-- cell: t10-v6-041 src: 4c25749e1a -->
<img src="assets/sc-ocr-app.png" style="max-width: 800px;" alt="期待される PP-OCR の結果">

<!-- cell: t10-v6-042 src: 4c5e64d2e3 -->
## 8. 精度と性能のチェックリスト

コンパイルが成功しても、OCR の精度が十分であるとは限りません。代表的な画像でパイプライン全体を検証してください。

| 確認項目 | 重要な理由 | 実際の対応 |
|---|---|---|
| 検出解像度 | 小さな文字は 640 × 640 で消えることがある | 想定するカメラ距離で再現率を比較する |
| キャリブレーションのカバー範囲 | 量子化はキャリブレーション統計に従う | 実際の照明、フォント、ぼけ、背景を含める |
| 前処理の一致 | 色や正規化の不一致はモデル入力をずらす | コンパイル時と実行時の前処理を同一に保つ |
| 認識バケット | 過度なリサイズやパディングは文字を損なう | テキストのアスペクト比の分布を測定する |
| 認識の信頼度 | しきい値が低いと誤ったテキストが表示され、高いとテキストが落ちる | ラベル付き検証セットで調整する |
| 遠近と向き | カメラで撮った文書は常に平らで正立とは限らない | 必要に応じて文書の向き補正と歪み補正を追加する |
| エンドツーエンドのレイテンシ | 1 フレームに多数の認識用切り出しが含まれることがある | FPS だけでなくテキスト数に対するレイテンシを報告する |

見栄えのよいカメラサンプル 1 枚に合わせて調整しないでください。固定の検証セットを維持し、設定を変更するたびに検出の再現率、認識精度、エンドツーエンドのレイテンシを記録します。

<!-- cell: t10-v6-043 src: bf21a3d708 -->
## 9. トラブルシューティング

| 症状 | 表示される内容 | 考えられる原因 | 対処 |
|---|---|---|---|
| `dxcom` がない | ステータス表に `[MISSING] dx_com` | DX-COM がインストールされていないか、`config.json` が別の場所を指している | チュートリアル 01 の第 2 節を完了し、`python tutorial_paths.py --show` を確認する |
| ダウンロードが拒否される | `Checksum mismatch` または `Incomplete download` | 転送が中断されたか、上流のファイルが変更された | `workspace/models` 配下の該当モデルだけを削除して 4.1 節を再実行する |
| 固定モデルの形状が違う | `AssertionError: ... expected [1, 3, 48, 240]` | 古い固定 ONNX ファイルが残っている | その固定 ONNX ファイルを削除して第 5 節を再実行する |
| コンパイルが失敗する | `Error Type: DataNotFoundError` などの `dxcom` の `[ERROR]` 行 | `dataset_path` が間違っているか、指定した拡張子の画像がない | `configs_v6/<model>.json` を 6.1 で表示されたディレクトリと比較する。完全なログは `outputs/paddleocr_v6/<model>/compiler.log` にある |
| 7.1 で `dx_engine` のインポートに失敗する | `ModuleNotFoundError: No module named 'dx_engine'` または wheel/ABI の不一致 | 別の Python で環境が作られたか、DX-RT がアップグレードされた | `workspace/.venv-ocr` を削除して 7.1 を再実行する。wheel はインストール済みの `libdxrt-bin` と一致している必要がある |
| `--check` がモデルの欠落を報告する | `Application test files are missing` | 6 個の DXNN ファイルがすべてコンパイルされていない | `SELECTED_COMPILES` にすべてのモデルを含めて 6.2 を再実行する |
| カメラを開けない | `Could not open camera /dev/video0` | カメラがない、デバイスパスが違う、またはユーザーが `video` グループに入っていない | `v4l2-ctl --list-devices` を実行し、`--camera /dev/videoN` を渡す。`sudo usermod -aG video $USER` の後に再ログインする |
| OpenCV の表示エラー | `cannot open display` | グラフィカルセッションがない | デスクトップセッション(有効な `DISPLAY` を持つ JupyterLab ターミナル)から実行する |
| テキストが表示されない | テキストのないボックス、またはボックス自体がない | 照明、フォーカス、しきい値 | 代表的な画像で `--det-threshold`、`--box-threshold`、`--rec-threshold` を調整する |

<!-- cell: t10-v6-044 src: 7776b639fb -->
## 10. まとめ

### 10.1 完了したワークフロー

```text
Official PP-OCRv6 tiny DET + REC ONNX
                    │
                    ▼
       Fix dynamic input dimensions
          ┌─────────┴─────────┐
          │                   │
   DET 640×640       REC width buckets ×5
          └─────────┬─────────┘
                    ▼
     Calibration configuration + DX-COM
                    │
                    ▼
           Six verified DXNN models
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 camera_app.py --check    640×480 camera
                              │
                              ▼
                   BBOX + recognized text
```

### 10.2 成果物と検証のダッシュボード

| 段階 | 成果物またはコマンド | 検証内容 |
|---|---|---|
| 元のモデル | <code>det.onnx</code>、<code>rec.onnx</code> | 公式リビジョンと SHA-256 チェックサム |
| 固定 ONNX | DET 1 個と REC 5 個のモデル | 静的な入力形状と ONNX の有効性 |
| 数値チェック | ONNX Runtime による比較 | 固定形状への変換が代表的な出力を保つこと |
| キャリブレーション | <code>configs_v6/*.json</code> | データセット、リサイズ、色順、正規化、レイアウト |
| コンパイル | <code>dxcom</code> | ログとレポートが見える状態での ONNX から DXNN への変換 |
| ランタイムコントラクト | <code>camera_app.py --check</code> | DET と 5 つの REC モデルが期待どおりのテンソルコントラクトで実行されること |
| ライブアプリケーション | <code>app/run_camera.sh</code> | 640×480 のキャプチャ、アスペクト比によるルーティング、CTC デコード、BBOX、テキストオーバーレイ |

### 10.3 ランタイムのモデルマップ

以下の形状は DX-RT が実行時に報告するもの(`dxparse`、`get_input_tensors_info()`)で、バッチ、高さ、幅、チャネルの順(NHWC、UINT8)です。第 5 節では ONNX の入力を NCHW の float 順で固定しました。DX-COM がレイアウト変換と正規化をコンパイル済みモデルに畳み込むため、アプリケーションは生の HWC ピクセルをそのまま入力できます。

| 役割 | 固定入力 | 選択ルール |
|---|---:|---|
| DET | `[1, 640, 640, 3]` | カメラフレームごとに 1 回 |
| REC 2.5 | `[1, 48, 120, 3]` | 切り出しのアスペクト比 ≤ 2.5 |
| REC 5 | `[1, 48, 240, 3]` | 切り出しのアスペクト比 ≤ 5 |
| REC 10 | `[1, 48, 480, 3]` | 切り出しのアスペクト比 ≤ 10 |
| REC 15 | `[1, 48, 720, 3]` | 切り出しのアスペクト比 ≤ 15 |
| REC 25 | `[1, 48, 1200, 3]` | 切り出しのアスペクト比 ≤ 25、またはそれより横長の切り出しのフォールバック |

### 10.4 完了チェックリスト

- [ ] 公式の PP-OCRv6 tiny DET と REC の ONNX モデルをダウンロードする
- [ ] 元のモデルのチェックサムと ONNX 構造を検証する
- [ ] 固定 DET モデル 1 個と固定 REC モデル 5 個を作成する
- [ ] 代表的な元モデルと固定モデルの出力を比較する
- [ ] キャリブレーション設定を作成し、DXNN モデルをコンパイルする
- [ ] コンパイル成果物を確認し、6 個のモデルすべてを NPU でテストする
- [ ] DET から REC への OpenCV カメラアプリケーションを準備する
- [ ] ラベル付き検証セットで検出精度と認識精度を測定する
- [ ] 代表的なカメラ画像でしきい値を調整する
- [ ] エンドツーエンドのレイテンシ、スループット、ホスト CPU 使用率を記録する

### 10.5 small または medium ティアを試す

より大きな PP-OCRv6 ティアで同じワークフローを繰り返すには、対応する公式 PaddlePaddle ONNX リポジトリから検出器と認識器の両方をダウンロードします。

| ティア | 検出 ONNX リポジトリ | 認識 ONNX リポジトリ |
|---|---|---|
| **small** | [PP-OCRv6_small_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx) | [PP-OCRv6_small_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx) |
| **medium** | [PP-OCRv6_medium_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx) | [PP-OCRv6_medium_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx) |

<!-- cell: 87889eeb-8768-43a7-97e9-5c0805d6cd64 src: 78b7602a47 -->
### 10.6 OCR の性能を向上させるには

このチュートリアルでは、`Document Image Orientation Classification`、`Text Image Unwarping`、`Text Line Orientation Classification` といった前処理ブロックは実装していません。

より高い OCR 性能が必要な場合は、不足している AI モデルを実装して OCR の AI パイプラインに組み込んでください。

<img src="assets/ppocrv6-full-pipeline.png" style="max-width: 1200px;">

精度を高めるには:

- **実際の運用環境**のテキスト画像でキャリブレーションデータセットを構築する。
- 実際のカメラ、距離、文字サイズ、アスペクト比、フォント、言語、照明、ぼけ、遠近をカバーする。
- 現在の**キャリブレーションデータセット**には、アスペクト比 5、15、25 の 3 つの大まかな REC バケットがある。運用時のテキスト形状のばらつきが大きい場合は、これらのデータセットのバケットをより細かく分割する。
- これらのキャリブレーション用バケットは、実行時の 5 つの REC モデルルート(アスペクト比 2.5、5、10、15、25)とは別のものである。
- 動的な DET と REC の入力を、運用データの測定結果に基づいて選んだ固定形状に変換する。
- キャリブレーションと実行時の前処理を同一に保つ。

| 運用データ | 調整 |
|---|---|
| 現在のバケットの中間にあるアスペクト比 | 中間の REC バケットを追加する |
| 縦長、細い、または長いテキスト | 対応する固定 REC 形状を使う |
| 非常に小さい文字 | より大きな DET 入力を評価する |
| 特殊なカメラのアスペクト比 | DET の入力形状を合わせる |
| リサイズ後の歪み | 形状と前処理を調整する |

リリース前に:

- 同じラベル付きデータセットで ONNX と DXNN の精度を比較する。
- 検出の再現率、認識精度、レイテンシ、メモリ、CPU 負荷を測定する。
- テンソルコントラクト、前処理、キャリブレーション、認識辞書を再確認する。
- 測定可能な価値がある場合にだけ形状を追加する。形状が増えるとビルド時間、ストレージ、ルーティングの複雑さが増える。

> **次へ:** チュートリアル 20 に進み、同じランタイム上でマルチチャネルの YOLO アプリケーションを C++ でビルドして実行しましょう。
