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

