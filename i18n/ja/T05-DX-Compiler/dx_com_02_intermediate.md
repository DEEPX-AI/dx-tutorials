<!-- i18n source: notebooks/T05-DX-Compiler/dx_com_02_intermediate.ipynb -->
<!-- i18n lang: ja -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: intro src: 680d42d3c9 -->
# DEEPX Tutorial 05-2 - DX-COM 中級編

このノートブックでは、コンパイルが成功した段階から一歩進んで、制御可能で説明可能なコンパイルワークフローを構築します。

キャリブレーションの判断を改善し、PPU Type 0 と Type 1 を使って実際の Model Zoo モデルをビルドし、YOLO26 の検出、セグメンテーション、ポーズモデルに TopK 先行最適化を適用します(既定では n。s、m、l、x は変数を 1 つ変えるだけで切り替えられます)。

<!-- cell: 6dffc67d-0a31-4b77-b975-65a83973f314 src: caf8fba665 -->
> このチュートリアルの全 3 部のうちの第 2 部です。コースマップと各部の一覧は入門ノートブックにあります。

<!-- cell: objectives src: ef4f57c018 -->
## 学習目標

このチュートリアルを終えると、次のことができるようになります。

- 代表的なキャリブレーションデータを選択し、モデルの前処理を再現する
- 検出処理のうち、どの演算が NPU の PPU で実行され、どの演算がホスト CPU に残るかを説明する
- 検出ヘッドのアーキテクチャから PPU Type 0 または Type 1 を選択する
- PPU のレイヤーマッピングを ONNX グラフと照らし合わせて検証する
- Model Zoo の YOLOv7 と YOLOX-S モデルをハードウェア PPU 付きでコンパイルする
- PPU 版と非 PPU 版の Model Zoo DXNN のランタイム挙動を比較する
- YOLO26 DXNN のどの部分がホスト CPU で実行され、TopK 先行最適化がそこで何を変えるのかを説明する
- YOLO26 ONNX モデルのヘッドテンソルを自動的に見つける
- 任意のサイズ(n、s、m、l、x)の YOLO26 検出、セグメンテーション、ポーズモデルに TopK 先行最適化を適用する
- 最適化後のモデルが同じ結果を返すことを確認し、削減される CPU 時間とエンドツーエンドのスループットを測定する
- アプリケーションを変更せずに、最適化後の DXNN を DX-APP で実行する

このノートブックは SDK のソースツリーを変更しません。生成されるファイルはすべて `<dx-tutorials>/notebooks/T05-DX-Compiler/workspace` 配下に保存され、git では無視されます。

<!-- cell: 64617a25-7ca0-407c-97d2-80bcd28d505c src: 38b24ba0d7 -->
## 前提条件

- チュートリアル 05-1 の完了: `venv-dx-compiler-local` 内の `dxcom` と、サンプルのキャリブレーションデータセット。
- 第 4 節と第 6 節の `dxrun` 比較のために、DEEPX NPU を搭載したマシンに DX-RT がインストールされていること。
- 6.8 節のために: DX-RT の Python 環境(`venv-dx-runtime`、チュートリアル 01)と DX-APP(チュートリアル 02)。これらがない場合でも、スキップされるのは DX-APP のセルだけです。
- ダウンロード: Model Zoo の YOLOv7 と YOLOX の ONNX および DXNN ファイル(約 290 MB)と、第 6 節で使う YOLO26 の検出、セグメンテーション、ポーズの ONNX ファイル(既定の `n` サイズで約 33 MB)。
- 所要時間: YOLOv7 640x640 のコンパイルはこのシリーズで最も時間がかかります。ラップトップでは数十分を見込んでください。第 6 節では YOLO26 モデルを 6 つ(検出、セグメンテーション、ポーズそれぞれのベースラインと TopK)コンパイルします。`n` の場合、x86_64 デスクトップで 1 モデルあたり約 2 分です。

次のセルは SDK の場所を特定し、これらの要件を確認してステータス表を表示します。`MISSING` と表示された項目には、それを提供する手順が示されます。

<!-- cell: workspace-heading src: 98620d01b8 -->
## 1. チュートリアルワークスペースの初期化

<!-- cell: calibration src: d60c6fc7ed -->
## 2. キャリブレーションはモデル定義の一部

量子化は、キャリブレーション入力によって生成される浮動小数点テンソルを観測します。手近にあるだけで無関係な画像セットを使っても DXNN ファイルは生成できますが、精度は低下します。

| 判断項目 | 推奨される実践 |
|---|---|
| サンプル | 実際の照明条件、スケール、背景、クラス分布を網羅する |
| 前処理 | 学習時および評価時と完全に一致させる |
| 枚数 | モデルのレシピから始め、安定性を測定してからのみ増やす |
| 手法 | 公開されている設定から始め、メトリクスを使って手法を比較する |
| 検証 | ホールドアウトデータセットで ONNX と DXNN の出力を比較する |

SDK の `calibration_dataset` リンクはこのチュートリアルを実行可能に保つためのものであり、タスク固有のデータセットの代わりにはなりません。

<!-- cell: preprocessing-order src: 1a5af2c2fa -->
### 2.1 前処理の順序

典型的な検出パイプラインは次のとおりです。

```text
image → letterbox/pad → BGR-to-RGB → divide by 255 → HWC-to-CHW → add batch axis
```

この順序をそのまま鵜呑みにしないでください。モデルのエクスポーターと学習コードを確認してください。パディングの位置、パディング値、色の順序、正規化が異なると、キャリブレーションテンソルの分布が変わります。

<!-- cell: ppu-overview src: 906986dee8 -->
## 3. PPU の概要

**後処理ユニット(Post-Processing Unit、PPU)** は DEEPX NPU 内部のハードウェアです。サポートされている YOLO 検出ヘッドに対して、ホスト CPU が処理しなければならない生の候補の数を削減します。

PPU は選択された次の 2 つの演算を実行します。

1. **信頼度フィルタリング**: コンパイル時に指定したしきい値を下回る候補を除去します。
2. **クラス予測**: 残った各候補について最も強いクラスを選択します。

PPU は Non-Maximum Suppression(NMS)を**実行しません**。NMS とアプリケーションレベルの解釈は引き続きホスト CPU で実行されます。また、PPU は画像の前処理や NPU 推論を置き換えるものでもありません。

<!-- cell: ppu-flow src: 2b907dff84 -->
### 3.1 検出パイプラインにおける PPU の位置づけ

```text
Without hardware PPU
┌──────────┐   ┌───────────────┐   ┌──────────────────────────────┐   ┌─────────┐
│ Input    │ → │ NPU inference │ → │ CPU filtering/class selection│ → │ CPU NMS │
└──────────┘   └───────────────┘   └──────────────────────────────┘   └─────────┘

With PPU Type 0 or Type 1
┌──────────┐   ┌───────────────┐   ┌────────────────────────────┐   ┌─────────┐
│ Input    │ → │ NPU inference │ → │ PPU filter/class prediction│ → │ CPU NMS │
└──────────┘   └───────────────┘   └────────────────────────────┘   └─────────┘
                                           hardware                    host
```

| ステージ | PPU なし | PPU Type 0/1 あり |
|---|---|---|
| ニューラルネットワーク推論 | NPU | NPU |
| 信頼度フィルタリング | ホスト CPU | PPU ハードウェア |
| 最良クラスの選択 | ホスト CPU | PPU ハードウェア |
| バウンディングボックスのデコード | モデル/タイプに依存 | モデル/タイプに依存 |
| NMS | ホスト CPU | ホスト CPU |

主な利点は、ホスト CPU の処理量が減り、中間の検出データが少なくなることです。エンドツーエンドでの正確な効果は、モデル、しきい値、シーン、ホスト CPU、アプリケーションの後処理によって異なります。

<!-- cell: ppu-types src: 69849e3a90 -->
### 3.2 モデルアーキテクチャからモードを選択する

| モード | アーキテクチャ | 代表的なモデル | レイヤーマッピング | 実行場所 |
|---|---|---|---|---|
| PPU Type 0 | アンカーベース(anchor-based) | YOLOv3, YOLOv4, YOLOv5, YOLOv7 | 検出 Conv ノード → アンカー数 | PPU ハードウェア |
| PPU Type 1 | アンカーフリー(anchor-free) | YOLOX, YOLOv8–YOLOv12 | `bbox`、`obj_conf`(存在する場合)、`cls_conf` ノード | PPU ハードウェア |
| `pre_optimize()` | TopK 先行の ONNX 書き換え | YOLOv8, v9, v11, v12, v13 および YOLO26(検出、セグメンテーション、ポーズ)、RTMDet | スケールごとのヘッド出力テンソル | NPU + ホスト CPU グラフ |

モデル名だけでタイプを選択しないでください。エクスポートされた ONNX のヘッド構造を確認し、まさにそのファイルに含まれるノード名を使用してください。

<!-- cell: ppu-config-anatomy src: 7ae6e1120b -->
### 3.3 PPU 設定を読み解く

```text
PPU configuration
├── type           selects the supported head architecture
├── conf_thres     fixed confidence threshold compiled into the DXNN
├── num_classes    class count of this exported model
├── activation     Type 0 activation, usually Sigmoid
└── layer          exact ONNX head-node mapping
```

| フィールド | Type 0 | Type 1 | 重要な理由 |
|---|:---:|:---:|---|
| `type` | 必須 | 必須 | ハードウェアのデータパスを選択する |
| `conf_thres` | 必須 | 必須 | PPU から出力される候補の数を制御する |
| `num_classes` | 必須 | 必須 | ヘッドのチャネルレイアウトと一致している必要がある |
| `activation` | 必須 | 未使用 | アンカーベースのスコア活性化を適用する |
| `layer` | 辞書 | リスト | PPU の入力を正確な ONNX ノードに接続する |
| `num_anchors` | レイヤーごと | 未使用 | 各スケールのアンカー数と一致している必要がある |
| `obj_conf` | 未使用 | モデルに依存 | YOLOX には独立した objectness ブランチがある |

**重要:** `conf_thres` はコンパイル時に固定されます。後から変更するには新しい DXNN が必要です。値を高くするとホストの処理量は減りますが、有効な検出まで除去してしまう可能性があります。デプロイ前に精度を測定してください。

<!-- cell: type0-heading src: 269378c531 -->
## 4. PPU Type 0 ラボ: Model Zoo の YOLOv7

YOLOv7 はアンカーベースの検出ヘッドを使用するため、このラボでは PPU Type 0 を使用します。実際の Model Zoo の ONNX と対応する PPU JSON をダウンロードし、マッピングされた Conv ノードを検証し、データセットのパスを調整して DXNN をコンパイルします。

<!-- cell: download-helper-heading src: 110755aed3 -->
### 4.1 Model Zoo ファイルを安全にダウンロードする

次のセルは `tutorial_paths.py` から `download_file` をインポートします(チュートリアル 05-2 と 05-3 で共有されるヘルパーです。内容はファイルを開いて確認してください)。空でない宛先ファイルがすでに存在する場合、ヘルパーはネットワークリクエストを行わずにダウンロードをスキップします。新しいコピーをダウンロードしたい場合は、まずローカルのファイルを削除してください。新規ダウンロードの場合、ヘルパーはリモートファイルのサイズを確認し、一時的な `.part` ファイルに書き込み、転送が完了した後にのみ宛先ファイルを置き換えます。DNS やネットワークの障害は例外を発生させ、空の最終ファイルを残しません。

このラボでは、コンパイル用に ONNX と PPU JSON をダウンロードします。また、ベンチマークの参照用に公開されている非 PPU 版 DXNN もダウンロードします。

<!-- cell: inspect-type0-heading src: cb55365789 -->
### 4.2 ONNX と Type 0 マッピングを確認する

Type 0 では `layer` は辞書です。各キーは検出ヘッドの Conv ノード名でなければならず、`num_anchors` はそのスケールと一致している必要があります。出力チャネル数は次の式に従います。

```text
channels = num_anchors × (5 + num_classes)
         = 3 × (5 + 80)
         = 255
```

<!-- cell: yolov7-diagram src: d3c1a04b7a -->
ハイライトされた Conv ノードは、Model Zoo の PPU 設定で使用される 3 つのスケール固有の検出ヘッドです。

![YOLOv7 Type 0 PPU ヘッドマッピング](assets/yolov7-class-n80-ppu.png)

<!-- cell: adapt-type0-heading src: e20da5e57c -->
### 4.3 環境固有のパスだけを調整する

Model Zoo の JSON には、このモデルで使用された前処理レシピが含まれています。ラボではそのまま変更せず、利用できないキャリブレーションデータセットのパスだけを置き換えます。製品モデルの場合は、デプロイ対象ドメインの代表的なデータセットを使用してください。

<!-- cell: compile-helper-heading src: 320ebf992e -->
### 4.4 Type 0 DXNN をコンパイルして確認する

次のコードセルは DX-COM 環境を有効化し、`dxcom` を直接実行します。これは別のターミナルで次のコマンドを入力するのと同等です。

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/yolov7_640x640.onnx \
      -c configs/yolov7_640x640_ppu.local.json \
      -o outputs/yolov7_type0_ppu \
      --gen_log \
      --export_html
```

実際の SDK とチュートリアルのパスは異なる場合があります。コードセルは `config.json` から読み込んだパスを使用します。出力ディレクトリにすでに DXNN ファイルが存在する場合は、スキップメッセージを表示して `dxcom` を再実行しません。

> **コードセル末尾の `2>&1 | sed ... | grep ...` について:** `dxcom` はカーソル移動のエスケープコードを使ってプログレスバーを描画しますが、Jupyter はこれを表示できず、何百行もの空行として表示されてしまいます。このフィルターはそのプログレスバーの行だけを取り除き、`[INFO]`、`[WARNING]`、`[ERROR]` のメッセージはすべてそのまま表示されます。ターミナルではフィルターを省略しても構いません。`--gen_log` によって完全な出力が `compiler.log` に保存されます。このノートブックのすべてのコンパイルセルで同じフィルターを使用しています。

<!-- cell: benchmark-yolov7-heading src: 50a17e9138 -->
### 4.5 `dxrun` で PPU 版と非 PPU 版を比較する

非 PPU 版 DXNN は Model Zoo から直接ダウンロードしたもので、PPU 版 DXNN は 4.4 節でコンパイルした結果です。次のセルは、合成入力を使って両方のモデルを 5 秒間実行します。

```bash
dxrun -m models/yolov7_640x640_non_ppu.dxnn --use-ort -t 5
dxrun -m outputs/yolov7_type0_ppu/<compiled-model>.dxnn --use-ort -t 5
```

非 PPU 版モデルは NPU タスクと CPU タスクを含み、生の検出値を返します。PPU 版モデルはハードウェアが生成した `BBOX` データを返し、ホストが処理する出力を削減します。PPU は合成入力での FPS 向上を**保証するものではありません**。モデルのスケジューリング、出力転送、デバイスの状態によっては、この単独テストでは非 PPU 版の方が速くなることもあります。製品としての判断には、実際の入力を使ってフルパイプラインのレイテンシとホスト CPU 使用率も測定してください。

<!-- cell: type1-heading src: 103ddfd632 -->
## 5. PPU Type 1 ラボ: Model Zoo の YOLOX-S

YOLOX は分離型(decoupled)のアンカーフリー検出ヘッドを使用します。各スケールにバウンディングボックス、objectness、クラス信頼度の独立したブランチがあるため、このラボでは PPU Type 1 を使用します。

<!-- cell: download-type1-heading src: dcfccc5610 -->
### 5.1 ONNX と Type 1 JSON をダウンロードする

同じ安全なダウンローダーを再利用します。ONNX と PPU JSON はコンパイルに使用し、公開されている非 PPU 版 DXNN はベンチマークの参照として使用します。

<!-- cell: inspect-type1-heading src: 80d640adfd -->
### 5.2 ONNX と Type 1 マッピングを確認する

YOLOX では、各スケールが 3 つの名前付きノードをマッピングします。

```text
feature map ─┬─ bbox branch ─────→ bbox
             ├─ object branch ───→ obj_conf
             └─ class branch ────→ cls_conf
```

3 つのエントリは、80×80、40×40、20×20 の検出スケールに対応します。

<!-- cell: yolox-diagram src: 8a4b542192 -->
色は、各スケールで対にする必要がある `bbox`、`obj_conf`、`cls_conf` のブランチを示しています。

![YOLOX Type 1 PPU ヘッドマッピング](assets/yolox-class-n80-ppu.png)

<!-- cell: adapt-type1-heading src: 7559b7cea7 -->
### 5.3 キャリブレーションデータセットのパスを調整する

ダウンロードした Type 1 マッピングと YOLOX の前処理はそのまま維持します。この演習では、Model Zoo のビルドマシン上のデータセットパスだけを置き換えます。

<!-- cell: compile-type1-heading src: 818827cba8 -->
### 5.4 Type 1 DXNN をコンパイルして確認する

次のコードセルは、別のターミナルで次のコマンドを入力するのと同等です。

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/yolox-s_640x640.onnx \
      -c configs/yolox-s_640x640_ppu.local.json \
      -o outputs/yolox_s_type1_ppu \
      --gen_log \
      --export_html
```

コードセルは実際のパスを自動的に解決します。出力ディレクトリにすでに DXNN ファイルが存在する場合は `dxcom` をスキップします。その `dxparse -v` の出力を Type 0 の結果と比較してください。特に出力テンソルのレイアウトと PPU メタデータに注目してください。

<!-- cell: benchmark-yolox-heading src: 2ae99d790d -->
### 5.5 `dxrun` で PPU 版と非 PPU 版を比較する

YOLOX-S でも同じ条件を揃えたテストを繰り返します。非 PPU 版の参照モデルは 5.1 節でダウンロードした公開 Model Zoo DXNN であり、このチュートリアルではコンパイルしません。

```bash
dxrun -m models/yolox-s_640x640_non_ppu.dxnn --use-ort -t 5
dxrun -m outputs/yolox_s_type1_ppu/<compiled-model>.dxnn --use-ort -t 5
```

この合成ベンチマークは、`dxparse` が示すタスク構造と合わせて解釈してください。PPU 版モデルは、この単一の FPS 値が向上しない場合でも、ホスト側の処理と出力トラフィックを削減できます。

<!-- cell: topk-heading src: 7c5984ab3c -->
## 6. YOLO26 の TopK 先行最適化 (n / s / m / l / x)

PPU Type 0 と Type 1 は、検出の後処理を NPU ハードウェアに移します。YOLO26 は別の方法をとります。後処理はホスト CPU に残したまま、DX-COM の `pre_optimize()` がその CPU 部分を書き換えて、処理量を大幅に減らします。この節では、処理がどこで実行されるか、書き換えによって何が変わるか、結果が変わらないことをどう確認するか、どれだけ時間を削減できるかを示します。その後、同じ手順をセグメンテーションモデルとポーズモデルに適用します。

### YOLO26 の後処理が実行される場所

コンパイルされた YOLO26 DXNN には 2 つのタスクが含まれています。これらは 6.6 で `dxparse` を使って確認します。

```text
image ─► npu_0 [NPU] ──────────► head tensors ─► cpu_0 [CPU, ONNX Runtime on the host] ─► [1, 300, 6]
         backbone, neck, and     box and class    Sigmoid on the class logits, box decoding,
         the head Convs          logits, 3 scales  keep the best 300 candidates
```

NPU タスクは、各ヘッドブランチの最後の Conv で終わります。それ以降はすべて通常の ONNX であり、DX-RT がホスト CPU 上で ONNX Runtime を使って実行します。`dxrun` に `--use-ort` が必要なのはこのためです。640x640 では、ヘッドは 8,400 個の候補を出力します。そのためベースラインの CPU タスクは、毎フレーム 8,400 x 80 = 672,000 個のクラスロジットに Sigmoid を適用してから、そのうち 300 個を残します。

### `pre_optimize()` が変えるもの

```text
Baseline : 8,400 candidates → Sigmoid on every class logit → decode → TopK 300
Optimized: 8,400 candidates → TopK 300 on the raw logits    → Sigmoid and decode for those 300 only
```

Sigmoid はスコアの順序を保つため、生のロジットに対して TopK を行っても同じ候補が選ばれます。出力は `[1, 300, 6]`(ボックス `x1 y1 x2 y2`、スコア、クラス ID)のままであり、6.5 節で値が同一であることを示します。変わるのはホスト CPU の処理量だけです。

YOLO26 用のパスは `yolo_no_dfl_postprocess` です(YOLO26 のボックスは DFL を使用しません)。DX-COM には、YOLOv8〜v13 用の `yolo_dfl_postprocess` と RTMDet 用の `rtmdet_postprocess` もあります。3 つすべてについて、DX-COM ユーザーマニュアルの *Pre-Optimize API* で説明されています。

### 効果が得られる場合

削減されるのはホスト CPU の処理です。そのため、CPU 部分が各フレームの大きな割合を占める場合、つまり Arm ボードのような低速なホスト CPU や、クラス数の多いモデルで効果を発揮します。高速な x86_64 デスクトップや、クラス数が少ないモデルでは、向上はわずかです。6.7 節では CPU 部分とエンドツーエンドのスループットを分けて測定するので、お使いのホストでの効果を判断できます。

### パスを適用する 2 つの方法

| 方法 | 手順 | 使用する場面 |
|---|---|---|
| Python API | `dx_com.pre_optimize(model, passes=...)` が新しい ONNX ファイルを書き出し、それをコンパイルする | コンパイルの前に、書き換えた ONNX を確認・検証したい場合(この節) |
| コンパイル設定 | コンパイル JSON 内の `pre_optimize` ブロック。`dxcom` がコンパイル中にパスを適用する | パスの設定を検証済みで、コンパイルを 1 ステップで済ませたい場合(6.3 の注記を参照) |

### この節の構成

| ステップ | 実施内容 | 結果 |
|---|---|---|
| 6.1 | Model Zoo の YOLO26 検出モデルをダウンロードする | ベースラインの ONNX と JSON |
| 6.2 | ヘッドテンソルを自動的に見つける | パスの設定 |
| 6.3 | パスを適用する | TopK ONNX |
| 6.4 | CPU 部分を前後で比較する | 演算と形状の表 |
| 6.5 | 結果が同一であることを確認する | 実画像を使った ONNX Runtime での比較 |
| 6.6 | 両方のモデルをコンパイルする | 2 つの DXNN ファイルと、その NPU/CPU の分割 |
| 6.7 | 測定する | CPU 部分の時間と `dxrun` の FPS |
| 6.8 | TopK DXNN を DX-APP で実行する | 結果画像 |
| 6.9 | セグメンテーションとポーズについて 6.1〜6.8 を繰り返す | |

6.1 の `YOLO26_VARIANT` で Model Zoo のサイズを選択し、以下のすべてのファイル名、コンパイル、ベンチマークがそれに従います。まず `n` から始め、その後に変数を変更してより大きなモデルでこの節を再実行してください。各サイズは `models/`、`configs/`、`outputs/` 配下に独自のファイルを使用するため、すでにコンパイル済みのサイズに戻した場合はコンパイルがスキップされます。

| `YOLO26_VARIANT` | 検出 ONNX ダウンロード | モデル 1 つあたりのコンパイル時間(x86_64 デスクトップ) |
|---|---:|---|
| `n`(既定) | 10 MB | 約 2 分(実測) |
| `s` | 37 MB | 約 2 分(実測) |
| `m` | 78 MB | 未測定。数分を見込む |
| `l` | 95 MB | 未測定。10 分以上を見込む |
| `x` | 213 MB | 未測定。10 分以上を見込む |

<!-- cell: download-yolo26-heading src: 306e030b72 -->
### 6.1 ベースラインモデルをダウンロードして検証する

`YOLO26_VARIANT` を設定してセルを実行します。ベースラインの ONNX と対応する Q-Lite JSON は Model Zoo から取得します(`yolo26-<variant>_640x640.onnx` と `.json`)。安全なダウンローダーにより、シェルのダウンロードが失敗しても次のノートブックセルが続行してしまうことで発生する空ファイルのエラーを防ぎます。

<!-- cell: verify-heads-heading src: 86c8f319d0 -->
### 6.2 ヘッドテンソルを見つける

`pre_optimize()` は、NPU 部分がどこで終わるかを知る必要があります。つまり、各検出スケールについて、各ヘッドブランチの最後の Conv の出力テンソルです。これらは、お使いの ONNX ファイルそのものに含まれるテンソル名です。モデルやエクスポーターのバージョンによって異なるため、決して他のモデルからコピーしないでください。

ヘッドテンソルは簡単に見分けられます。

1. 形状が `[1, C, H, W]` の `Conv` の出力であり、`H = 640 / stride`、stride は 8、16、32 のいずれかです(80x80、40x40、20x20)。
2. 後に活性化関数が続きません。これを使用するのは `Reshape` または `Concat` だけです。
3. チャネル数 `C` によって役割がわかります。

| 役割 | チャネル数 | タスク |
|---|---|---|
| `bbox` | 4(ボックスの 4 辺までの距離) | すべて |
| `cls_conf` | クラス数(クラスロジット) | すべて |
| `mask_coeff` | 32 | `seg` |
| `kpt` | キーポイント数 x キーポイントあたりの値の数(17 x 3 = 51) | `pose` |

実際のモデルでは、さらに次の 2 点に注意が必要です。

- **チャネル数が等しい場合。** 4 クラスのモデルでは、`cls_conf` は `bbox` と同じく 4 チャネルになります。直接 `Sigmoid` に入るのはクラススコアのブランチだけなので、それによって 2 つを区別できます。
- **2 つのヘッド。** YOLO26 は one-to-many ヘッドと one-to-one ヘッドを使って学習されます。推論では one-to-one ヘッドを使用し、テンソル名には `one2one_*` が含まれます。one-to-many ヘッドだけを含むエクスポートは、このパスでは動作しません(6.3 の `nms=False` に関する注記を参照)。

次のセルは、これらのルールを適用する小さなモジュール `yolo26_heads.py` を書き出します。このモジュールは ONNX ファイル内の Ultralytics メタデータから task、クラス数、キーポイントのレイアウトを読み取り、ルールが成り立たない場合は説明を表示して停止します。必要なのは `onnx` パッケージだけなので、ノートブックと DX-COM 環境のどちらからでもインポートできます。DX-COM ユーザーマニュアルの *Identifying Conv Head Tensor Names* でも、同じルールと Netron での確認方法が説明されています。

<!-- cell: apply-topk-heading src: 45a61021f4 -->
### 6.3 `yolo_no_dfl_postprocess` を適用する

`optimize_yolo26.py` は、`dx_com` がインストールされている DX-COM の Python 環境で実行します。このスクリプトは `yolo26_heads.py` でヘッドテンソルを見つけ、パスの設定を表示し、TopK モデルを別の ONNX ファイルに書き出すため、ベースラインは変更されません。同じスクリプトがすべてのサイズと task に使え、後にはお使いの独自モデルにも使えます。

パスの設定は次のとおりです。

| キー | 値 | 備考 |
|---|---|---|
| `task` | `base`、`seg`、`pose` のいずれか | 必須。各レイヤーが持つべきヘッドテンソルを選択する |
| `layers` | スケールごとに 1 エントリ、stride 8、16、32 の順 | 6.2 で取得 |
| `num_classes` | クラス数 | 既定は 80 |
| `topk` | デコード前に残す候補の数 | 既定は 300。Model Zoo のpre-optimized プロファイルでは 512 を使用 |

> **DX-COM v2.5.0 での変更点。** 以前 `yolo26_postprocess` と呼ばれていたパスは `yolo_no_dfl_postprocess` に、`yolo_postprocess`(YOLOv8〜v13)は `yolo_dfl_postprocess` に名前が変わりました。旧名を使うとエラーになります。すべての YOLO パスに `task` キーが必要で、`input_height` / `input_width` は受け付けられなくなりました(入力サイズはモデルから取得されます)。

> **独自の YOLO26 モデル。** NMS-free ヘッドでエクスポートしたモデルであれば、独自に学習したモデルにもこのスクリプトを使えます: `yolo export model=best.pt format=onnx imgsz=640 batch=1 nms=False`。`nms=False` を指定しない場合、現在の Ultralytics リリース(8.4.208 で検証)は出力が `[1, 4 + classes, 8400]` の one-to-many ヘッドをエクスポートし、`yolo26_heads.py` は再エクスポートを促すメッセージを表示して停止します。`imgsz` は固定し、`dynamic=True` は使用しないでください。`num_classes` と、ポーズの場合の `kpt_shape` は、エクスポートしたモデルのメタデータから取得されます。

> **コンパイル JSON からパスを適用する。** スクリプトは設定を `{"yolo_no_dfl_postprocess": {...}}` の形式で表示します。元の ONNX をコンパイルする際に `dxcom` にパスを適用させるには、この辞書をリストに入れて、コンパイル JSON の `pre_optimize` の下に置きます: `"pre_optimize": [{"yolo_no_dfl_postprocess": {...}}]`。するとコンパイルログに `[pre_optimize] applying pass 'yolo_no_dfl_postprocess' from config` と表示され、DXNN のファイル名は ONNX のファイル名と同じになります。`pre_optimize` と `ppu` を同じ JSON に入れないでください。DX-COM 2.5.0 は `ppu` を使ってコンパイルして `pre_optimize` を破棄し、`falling back to legacy: ValueError: config 'pre_optimize' and 'ppu' are mutually exclusive` と表示するだけです。

次の 2 つのセルでスクリプトを書き出して実行します。既定の `n` バリアントの場合、実行内容は次と同等です。

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
python optimize_yolo26.py models/yolo26-n_640x640.onnx models/yolo26-n_640x640_topk300.onnx
```

<!-- cell: cpu-part-heading src: 94cb953e24 -->
### 6.4 CPU 部分を前後で比較する

コンパイルされた DXNN の CPU タスクは、ヘッドテンソルからモデル出力までの ONNX サブグラフです。次のセルは、両方のモデルからそのサブグラフを切り出し(`onnx.utils.extract_model`)、候補の数に応じて処理量が増える演算を一覧表示します。

確認するポイント:

- **ベースライン:** `Sigmoid` が `[1, 80, 8400]`、つまりすべての候補についてクラスごとに 1 つずつの値に対して実行され、TopK はその後に来ます。
- **TopK モデル:** まず `ReduceMax` と `TopK` が生のロジットから 300 個の候補を選択し、`Sigmoid` は `[1, 300, 80]` に対して実行されます。値の数は 28 分の 1 です。

TopK モデルのほうがノード数は多くなります。300 個の候補を取り出す gather の処理が追加されるためです。ただし、これらの追加ノードが扱うのは 8,400 行ではなく 300 行です。切り出した CPU 部分は、6.7 で CPU 処理の時間を計測する際にも使用します。

<!-- cell: verify-results-heading src: 368a5b670d -->
### 6.5 結果が同一であることを確認する

グラフを変更した場合は、コンパイルする前に、同じ結果が得られることを確認する必要があります。次のセルでは、同じ実画像に対して両方の ONNX モデルを ONNX Runtime で実行し、出力を比較して、各モデルの検出結果を描画します。

画像は、Model Zoo の JSON がキャリブレーション画像を準備するのと同じ方法で準備します。値 114 で 640x640 の正方形にパディングし、BGR から RGB に変換し、255 で割り、NCHW に変換します。テスト画像は DX-APP のサンプルから取得します。DX-APP がインストールされていない場合は、最初のキャリブレーション画像を使用します。

比較の対象は、スコアが 0.001(DX Model Zoo が mAP 評価に使用するしきい値)を超えるすべての候補と、その他すべての出力(たとえばセグメンテーションのプロトタイプ)です。スコアがそれより低い候補は意図的に除外しています。リストの下位では多くの候補がまったく同じスコアを持ち、TopK は同点の候補をどちらの順序で返すこともあるためです。書き換えたモデルは Sigmoid の出力ではなく生のロジットをソートするため、そのような同点候補の順序が異なる場合があります。このノートブックのテスト画像では、最初の同点はスコア 0.0001 未満で現れました。

この比較で検証するのは、書き換えた ONNX グラフです。コンパイルでは各モデルが個別に量子化されるため、2 つの DXNN ファイルはわずかに異なる場合があります。6.8 では両方を NPU で実行します。

<!-- cell: verify-results-note src: 98e55a730d -->
候補の数が同じで最大差が 0 であれば、この画像に対して書き換えが同じ結果を返すことを意味します。独自のモデルの場合は、物体が多数写っている画像や 1 つも写っていない画像を含め、お使いのデータの複数の画像で同じ確認を行ってください。

<!-- cell: adapt-yolo26-heading src: 1cce1fabf3 -->
### 6.6 両方のモデルをコンパイルする

#### 6.6.1 Model Zoo の設定を調整する

両方のモデルは同じキャリブレーションと前処理の設定を使用するため、2 つのコンパイルの違いは ONNX グラフだけになります。変更するのは、Model Zoo のサーバーを指しているデータセットパスだけです。

<!-- cell: compile-yolo26-baseline-command src: 9ec494a9de -->
#### 6.6.2 ベースラインをコンパイルする

既定の `n` バリアントの場合、次のコードセルは別のターミナルで次のコマンドを入力するのと同等です(他のバリアントでは `yolo26-n` と `yolo26n` がそれに応じて変わります)。

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/yolo26-n_640x640.onnx \
      -c configs/yolo26-n_640x640.local.json \
      -o outputs/yolo26n_baseline \
      --gen_log \
      --export_html
```

出力ディレクトリにすでに DXNN ファイルが存在する場合、コードセルはそのファイルを報告して `dxcom` をスキップします。

<!-- cell: compile-yolo26-topk-command src: 6679ececa5 -->
#### 6.6.3 TopK モデルをコンパイルする

既定の `n` バリアントの場合、次のコードセルは別のターミナルで次のコマンドを入力するのと同等です(他のバリアントでは `yolo26-n` と `yolo26n` がそれに応じて変わります)。

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/yolo26-n_640x640_topk300.onnx \
      -c configs/yolo26-n_640x640.local.json \
      -o outputs/yolo26n_topk300 \
      --gen_log \
      --export_html
```

出力ディレクトリにすでに DXNN ファイルが存在する場合、コードセルはそのファイルを報告して `dxcom` をスキップします。

<!-- cell: task-split-heading src: 5ea330f2c1 -->
#### 6.6.4 NPU/CPU の分割を確認する

`dxparse -v` は、DXNN のタスクと、タスク間で受け渡されるテンソルを一覧表示します。次のセルはその部分だけを表示します。ベースラインでは、`npu_0` は 6.2 で見つけたヘッドテンソルで終わり、`cpu_0` がそれらを `output0` に変換します。TopK モデルでは、NPU が CPU 部分のためにヘッドテンソルのレイアウトも変更し(`preopt_*_tr_*`、channels last)、`cpu_0` が `preopt_output` を出力します。

<!-- cell: benchmark-heading src: 021a6c65d3 -->
### 6.7 効果を測定する

2 つの測定で、2 つの問いに答えます。

1. **パスによってどれだけの CPU 処理が削減されるか。** 6.4 の CPU 部分を ONNX Runtime で 1 スレッドで実行し、テスト画像の実際のヘッドテンソルを入力します。これにより、パスが変更する処理だけを切り分けて測定できます。
2. **パイプライン全体が速くなるか。** `dxrun --use-ort -t 5` は、合成入力を使って各 DXNN を NPU 部分と CPU 部分を合わせて 5 秒間実行します。

DX-RT は連続するフレームの NPU タスクと CPU タスクを重ねて実行するため、エンドツーエンドの FPS は 2 つのうち遅いほうで決まります。高速な x86_64 ホストでよくあるように NPU のほうが遅いステージである場合、CPU 部分が短くなっても FPS はほとんど変わりません。ホスト CPU が低速な場合は CPU 部分が制約となり、測定 1 で削減された時間が FPS の向上につながります。

既定の `n` バリアントの場合、`dxrun` コマンドは次のとおりです。

```bash
dxrun -m outputs/yolo26n_baseline/yolo26-n_640x640.dxnn --use-ort -t 5
dxrun -m outputs/yolo26n_topk300/yolo26-n_640x640_topk300.dxnn --use-ort -t 5
```

両方のモデルを同じデバイス上で、同程度の負荷と温度条件のもとで実行し、結論を出す前に測定を繰り返してください。これらの数値は、精度やフルビデオアプリケーションの FPS を測定するものではありません。

<!-- cell: benchmark-results src: 7013e3c5ea -->
結果の読み方:

- **CPU 部分。** パスが変更するのはこの部分です。Model Zoo のモデルでは約 35〜45% 短縮されます。絶対的な時間はホスト CPU によって異なり、同じ処理でも Arm コアでは x86_64 デスクトップのコアより数倍長くかかります。
- **`dxrun` の FPS。** CPU 部分がパイプラインを制約している範囲でしか向上しません。このノートブックの検証に使用した DX-M1 搭載の x86_64 デスクトップでは、YOLO26-n の CPU 部分は約 0.74 ms → 0.41 ms になりましたが、NPU のほうが遅いステージだったため、`dxrun` の FPS の向上は約 1.06× にとどまりました。
- **Arm ホストでは CPU 部分が制約になります。** この節で作成したものと同じ DXNN ファイルを、DX-M1 を搭載した Raspberry Pi 5 で測定した結果は次のとおりです(`dxrun -m <model> --use-ort -t 10`、既定の設定)。

  | YOLO26-n | ベースライン FPS | TopK 300 FPS | 向上 |
  |---|---:|---:|---:|
  | det | 100.79 | 179.54 | 1.78× |
  | seg | 64.13 | 95.61 | 1.49× |
  | pose | 142.48 | 265.44 | 1.86× |

  3 つのベースラインモデルでは、DX-RT が `CPU TASK [cpu_0] Inference Worker - Average Input Queue Load : ...% (DXRT_DYNAMIC_CPU_THREAD: OFF) - To improve FPS, set: 'export DXRT_DYNAMIC_CPU_THREAD=ON'` というメッセージも表示しました。このメッセージは、フレームが CPU タスクを待っていたことを意味します。TopK モデルでは表示されませんでした。お使いのボードでこのメッセージが表示される場合は CPU 部分がボトルネックであり、この最適化が効果を発揮する可能性が高いと言えます。`DXRT_DYNAMIC_CPU_THREAD=ON` は CPU タスクにより多くのスレッドを割り当てる別の設定であり、これらの数値の測定には使用していません。
- **DX-COM マニュアルの参考値**(ARM Cortex-A53 ホストの DX-M1、`DXRT_DYNAMIC_CPU_THREAD=ON dxrun -m <model> -l 1000 --use-ort`): det 184.89 → 315.73 FPS(1.7×)、seg 130.82 → 195.05 FPS(1.5×)、pose 242.77 → 304.38 FPS(1.3×)。

ご自身の製品について判断する際は、ターゲットボード上でこの節を実行してください。

<!-- cell: dx-app-heading src: a0178a7c3f -->
### 6.8 TopK DXNN を DX-APP で実行する

アプリケーションコードは変更しません。DX-APP の YOLO26 用後処理は最初の出力を位置で読み取り、DX-STREAM の `yolo26_od`、`yolo26_seg`、`yolo26_pose` 後処理ライブラリは出力を形状(`[1, 300, 6]`、38 列以上、57 列以上)で選択します。そのため、名前が変わった出力(`output0` → `preopt_output`)に合わせた変更は必要ありません。どちらも、CPU タスクには引き続き ONNX Runtime が必要です。

次のセルは、そのモデル用の DX-APP Python サンプルを各 DXNN で同じ画像に対して実行し、両方の結果を表示します。チュートリアル 01 の DX-RT Python 環境とチュートリアル 02 の DX-APP ソースを使用します。これらがない場合、セルはスキップされます。既定の `n` バリアントの場合、次と同等です。

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_app
<DX_ALL_SUITE_DIR>/dx-runtime/venv-dx-runtime/bin/python \
  src/python_example/object_detection/yolo26/yolo26-n_640x640/yolo26-n_640x640_sync.py \
  --model <workspace>/outputs/yolo26n_topk300/yolo26-n_640x640_topk300.dxnn \
  --image sample/img/sample_people.jpg --no-display --save --save-dir <workspace>/outputs/dx_app_results
```

このノートブックのテストマシンでは、両方の DXNN ファイルが同じ物体を検出しました。各モデルが個別に量子化されるため、ボックスの角の座標は最大で約 2 ピクセル、スコアは最大で約 0.05 異なりました。デプロイ前に、たとえば DX Model Zoo の評価ツールを使って、検証セットで精度を確認してください。

<!-- cell: seg-pose-heading src: 61a29534c1 -->
### 6.9 セグメンテーションとポーズ: `task` を変えるだけの同じパス

DX-COM v2.5.0 では、`yolo_no_dfl_postprocess` が YOLO26 のセグメンテーションモデルとポーズモデルにも対応しました。パス名、`topk`、`bbox` と `cls_conf` のテンソルは 6.3 と同じです。変わるのは 2 点で、`task` キーと、スケールごとに 1 つ加わるヘッドテンソルです。

| `task` | Model Zoo モデル(`n`) | スケールごとのヘッドテンソル | `num_classes` | パス適用後の出力 |
|---|---|---|---:|---|
| `base` | `yolo26-n_640x640` | `bbox`(4 ch)、`cls_conf`(80 ch) | 80 | `[1, 300, 6]` |
| `seg` | `yolo26-n-seg_640x640` | + `mask_coeff`(32 ch)、`one2one_cv4.<scale>` | 80 | `[1, 300, 38]` とプロトタイプ `[1, 32, 160, 160]` |
| `pose` | `yolo26-n-pose_640x640` | + `kpt`(51 ch)、`one2one_cv4_kpts.<scale>`、`kpt_shape` `[17, 3]` と併用 | 1 | `[1, 300, 57]` |

- **セグメンテーション。** 残った 300 個の候補それぞれが、32 個のマスク係数を保持します。マスクのプロトタイプ出力は変更されずにそのまま渡され、係数とプロトタイプを組み合わせてマスクを作る処理は引き続きホストが行います。
- **ポーズ。** キーポイントヘッドは単一の Conv なので、テンソル名(`/model.23/one2one_cv4_kpts.<scale>/Conv_output_0`)は他のブランチの `...<scale>.2/Conv_output_0` というパターンに従いません。6.2 のルールは名前に依存しないため、`yolo26_heads.py` はそれでもこのテンソルを見つけられます。`kpt_shape` は ONNX メタデータから取得し、17 x 3 がキーポイント Conv のチャネル数 51 と一致する必要があります。
- **検証。** `pre_optimize` はグラフを編集する前に、ヘッドテンソルを `task` と照合します。`seg` には `mask_coeff`、`pose` には `kpt` と `kpt_shape` が必須で、それぞれ相手側のテンソルは拒否されます。ただし、セグメンテーションモデルを `task="base"` で実行した場合は検出できません。コンパイルは通りますが、マスクが失われます。そのため `yolo26_heads.py` は `task` をモデルのメタデータと比較します。

次のセルでは、2 つのモデルについて 6.1〜6.8 を、1 セルにつき 1 ステップずつ繰り返します。Model Zoo はすべてのサイズで `seg` と `pose` のモデルを公開しているため、これらのセルも `YOLO26_VARIANT` に従います。

<!-- cell: seg-pose-prepare-heading src: fcd279e22c -->
#### 6.9.1 モデルをダウンロードしてヘッドテンソルを見つける

<!-- cell: seg-pose-apply-heading src: 1c5e139d9d -->
#### 6.9.2 パスを適用して CPU 部分を比較する

既定の `n` バリアントの場合、最適化の手順は次と同等です。

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
python optimize_yolo26.py models/yolo26-n-seg_640x640.onnx models/yolo26-n-seg_640x640_topk300.onnx --task seg
python optimize_yolo26.py models/yolo26-n-pose_640x640.onnx models/yolo26-n-pose_640x640_topk300.onnx --task pose
```

スクリプトがメタデータから task を読み取るため、ここでの `--task` は省略可能です。指定すると期待する task を明示することになり、モデルと一致しない場合はスクリプトが停止します。ポーズの表では、キーポイントの `Sigmoid` にも注目してください。`[1, 17, 1, 8400]` から `[1, 300, 17, 1]` に変わります。

<!-- cell: seg-pose-verify-heading src: 7657781fd1 -->
#### 6.9.3 結果が同一であることを確認する

比較の対象は、セグメンテーションではマスク係数とプロトタイプ、ポーズではキーポイントです。画像には、各モデル自身の出力からマスクとキーポイントを描画します。

<!-- cell: seg-pose-compile-heading src: e98f1b66b9 -->
#### 6.9.4 ベースラインと TopK モデルをコンパイルする

このセルは各モデルのローカル JSON を書き出し(6.6.1 と同様)、6.6 と同じオプションで 4 つのモデルをコンパイルします。既定の `n` バリアントの場合、次のコマンドと同等です(ポーズモデルでは `seg` を `pose` に置き換えてください)。

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/yolo26-n-seg_640x640.onnx \
      -c configs/yolo26-n-seg_640x640.local.json \
      -o outputs/yolo26n_seg_baseline \
      --gen_log \
      --export_html
dxcom -m models/yolo26-n-seg_640x640_topk300.onnx \
      -c configs/yolo26-n-seg_640x640.local.json \
      -o outputs/yolo26n_seg_topk300 \
      --gen_log \
      --export_html
```

出力ディレクトリにすでに DXNN ファイルがある場合、セルはそのファイルを報告し、そのモデルの `dxcom` をスキップします。

<!-- cell: seg-pose-benchmark-heading src: 083b768e06 -->
#### 6.9.5 検出、セグメンテーション、ポーズを測定する

このセルは 6.7 と同様に新しい 4 つの DXNN ファイルの CPU 部分と `dxrun` の FPS を測定し、6.7 の検出結果を表に加えます。

<!-- cell: seg-pose-results src: 967a128494 -->
3 つのタスクの比較:

- **検出とセグメンテーション**は、同じ箇所で CPU 時間を削減します。80 クラス x 8,400 候補に対するクラススコアの `Sigmoid` です。セグメンテーションはさらに、パスが変更しないマスクのプロトタイプ出力を保持し、マスクの構築も引き続きホストが行うため、相対的な向上は小さくなります。
- **ポーズ**はクラスが 1 つしかないため、クラススコアの処理はわずかです。ポーズでの削減は、キーポイントの可視性の `Sigmoid`(17 x 8,400 個の値)とキーポイントのデコードが、300 個の候補に対してのみ実行されるようになることで得られます。その効果の大きさはホストによって異なります。上記の Raspberry Pi 5 ではポーズの向上が最も大きく(1.86×)、マニュアルの Cortex-A53 の結果では最も小さくなっています(1.3×)。

<!-- cell: seg-pose-dx-app-heading src: 42492b118d -->
#### 6.9.6 セグメンテーションとポーズの DXNN ファイルを DX-APP で実行する

<!-- cell: 6be8d246-d16c-429d-a82f-e1c45594974c src: 8bd09629d5 -->
## 7. トラブルシューティング

| 症状 | 表示される内容 | 考えられる原因 | 対処 |
|---|---|---|---|
| PPU のノード名が見つからない | `ValueError: ... not found in the ONNX graph` | PPU 設定の `layer` の名前が別のエクスポートのものである | Netron で ONNX を確認し、JSON のノード名を更新する |
| `dxrun` がデバイスを見つけられない | `No device found` または `Fail to initialize device` | このホストに DEEPX NPU がない | NPU を搭載したホストで比較を実行する。コンパイルのみの手順はそのまま動作する |
| FPS を読み取れない | `Could not read FPS from dxrun output` | `dxrun` の出力形式が変わった | 上に完全な出力を表示し、FPS の行を手動で読み取る |
| JSON の変更が反映されない | `Skip compilation: found ...` | 出力ディレクトリにすでに DXNN が存在する | `workspace/outputs/<dir>` を削除して再実行する |
| `pre_optimize` がヘッドテンソルを拒否する | `ValueError: [yolo_no_dfl_postprocess] task='seg' requires 'mask_coeff' on every layer` または `Cannot find shape for mask_coeff=...` | `task` とヘッドテンソルがモデルと一致していない | まさにこの ONNX ファイルに対して `find_yolo26_heads()` でテンソルを見つける |
| head finderが task の確認で停止する | `The model metadata says task='seg', not 'base'` | task 引数がモデルと一致していない | メッセージに表示された task を渡す。`base` のままだと、セグメンテーションモデルがマスクなしでコンパイルされる |
| head finderが one-to-many ヘッドを報告する | `This ONNX file has the one-to-many head (output [1, 4 + classes, 8400])` | モデルが `nms=False` なしでエクスポートされた | `yolo export ... format=onnx nms=False` で再エクスポートする |
| head finderが `cls_conf` の Conv を見つけられない | `expected one cls_conf head Conv with N channels, found none` | クラス数が誤っているか、メタデータにない | `num_classes` を渡す(スクリプトでは `--num-classes`) |
| head finderが動的入力を報告する | `The input height is dynamic` | モデルが `dynamic=True` でエクスポートされた | 固定の `imgsz` で再エクスポートする |
| TopK DXNN が `dxrun` で速くならない | 相対スループットが 1.00× に近い | パイプラインを制約しているのがホスト CPU ではなく NPU である(x86_64 でよくある)、またはモデルのクラス数が少ない | 6.7 の CPU 部分の時間を比較し、ターゲットボードで測定する |
| JSON の `pre_optimize` が反映されない | `falling back to legacy: ... 'pre_optimize' and 'ppu' are mutually exclusive` | JSON に `ppu` ブロックも含まれている | JSON から `ppu` を削除する |
| 別のクラスを持つモデルで DX-APP に COCO のラベルが表示される | 物体に COCO のクラス名が表示される | ラベルはサンプル自身の `config.json` から取得され、`--config` では変更されない | `scripts/extract_model_package.sh --no-prune` でスタンドアロンパッケージをエクスポートし(チュートリアル 02)、その `config.json` で `num_classes` と `visualizer.kwargs.custom_labels` を設定する |
| ダウンロードが不完全 | `workspace/models` に極小または 0 バイトのファイルがある | ダウンロードが中断された | ファイルを削除する。ダウンロードヘルパーは使用前にサイズを検証する |

<!-- cell: summary src: a00cdd65b0 -->
## 8. まとめ

### 8.1 中級編の最適化マップ

```text
Representative calibration data
              │
              ▼
     Inspect the detection head
              │
      ┌───────┴────────┐
      │                │
Anchor-based      Anchor-free
  YOLOv7             YOLOX
      │                │
PPU Type 0        PPU Type 1
      │                │
      └───────┬────────┘
              │
              ▼
 Reduce host-side filtering
 and class-selection workload

YOLO26 det / seg / pose head
          │
          ▼
 Find head tensors → TopK-first ONNX rewrite
 (task = base / seg / pose)
          │
          ▼
 Same results, less host-CPU work
 per frame
```

### 8.2 3 つの最適化パス

| トピック | 使用モデル | 主な判断 | 最適化の場所 | ホスト CPU の役割 |
|---|---|---|---|---|
| PPU Type 0 | YOLOv7 | アンカーベースのヘッドを選択し、その Conv ノードをマッピングする | ハードウェア PPU | NMS とアプリケーションロジック |
| PPU Type 1 | YOLOX-S | bbox、objectness、クラスの各ブランチをマッピングする | ハードウェア PPU | NMS とアプリケーションロジック |
| TopK 先行最適化 | YOLO26 の検出、セグメンテーション、ポーズ(`YOLO26_VARIANT`、既定は n) | TopK を Sigmoid とデコードの前に移動し、`task` とヘッドテンソルを指定する | 書き換えた ONNX グラフ(Python API またはコンパイル JSON) | より少ない候補(とそのマスク係数またはキーポイント)をデコードする |

### 8.3 完了した実験

| 実験 | 比較したモデル | 統制した変数 | 検証方法 |
|---|---|---|---|
| Type 0 PPU | YOLOv7 非 PPU 版と PPU 版 | ハードウェア後処理 | `dxparse`、`dxrun` |
| Type 1 PPU | YOLOX-S 非 PPU 版と PPU 版 | ハードウェア後処理 | `dxparse`、`dxrun` |
| TopK 最適化 | YOLO26 の検出、セグメンテーション、ポーズ: ベースラインと TopK 300(選択したサイズ) | ONNX グラフ構造 | ONNX Runtime による出力比較、CPU 部分の時間、`dxparse`、`dxrun`、DX-APP |

### 8.4 完了チェックリスト

- [ ] キャリブレーションデータと前処理をモデル定義の一部として扱う
- [ ] NPU、PPU、ホスト CPU の処理の境界を特定する
- [ ] YOLOv7 Type 0 の Conv ノードマッピングを検証する
- [ ] Type 0 PPU モデルをコンパイルして確認する
- [ ] YOLOX-S Type 1 の bbox、objectness、クラスのマッピングを検証する
- [ ] Type 1 PPU モデルをコンパイルして確認する
- [ ] 同じベンチマーク条件で PPU 版と非 PPU 版のモデルを比較する
- [ ] YOLO26 DXNN のどの部分がホスト CPU で実行されるかを説明する
- [ ] 名前をコピーするのではなく、`find_yolo26_heads()` で YOLO26 のヘッドテンソルを見つける
- [ ] YOLO26 モデルに TopK 先行最適化を適用する(そして 2 つ目のサイズも試す)
- [ ] ベースラインと TopK モデルが同一の出力を返すことを ONNX Runtime で確認する
- [ ] CPU 部分の時間と `dxrun` の FPS を測定し、両者が異なる理由を説明する
- [ ] アプリケーションを変更せずに TopK DXNN を DX-APP で実行する
- [ ] `task` = `seg`、`pose` で YOLO26 のセグメンテーションモデルとポーズモデルに同じパスを適用する
- [ ] 実データで精度、ホスト CPU 使用率、フルパイプラインのレイテンシを検証する

### 8.5 選び方

| モデルが持つもの | まず試すもの |
|---|---|
| アンカーベースの YOLO ヘッド | PPU Type 0 |
| 独立した bbox、objectness、クラスのブランチ | PPU Type 1 |
| 大量の候補の後に続く TopK(CPU 部分が FPS を制約しているホストの場合) | TopK 先行のグラフ最適化 |
| セグメンテーションまたはポーズのヘッド(YOLO26 または YOLOv8〜v13) | `task` = `seg` または `pose` での TopK 先行のグラフ最適化 |
| 不明またはカスタマイズされたヘッド | 最適化を選択する前に、実際の ONNX グラフを確認する |

> **覚えておくこと:** ホストのワークロード低減と合成入力での FPS 向上は同じ結果ではありません。`dxrun` で条件を揃えたランタイム比較を行い、その後に実際のアプリケーションワークロードでタスク精度、ホスト CPU 使用率、エンドツーエンドのレイテンシを測定してください。

### 8.6 次のステップ

**上級編チュートリアル**に進み、次の内容を学びましょう。

- Q-Lite、Q-PRO、Q-Master の選択
- 量子化の診断
- QXNN の再開ワークフロー
- QAT
- Python API によるコンパイル
- 高度なコンパイラ制御
