<!-- i18n source: notebooks/T05-DX-Compiler/dx_com_02_intermediate.ipynb -->
<!-- i18n lang: ja -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: intro src: bb61e22e24 -->
# DEEPX Tutorial 05-2 - DX-COM 中級編

このノートブックでは、コンパイルが成功した段階から一歩進んで、制御可能で説明可能なコンパイルワークフローを構築します。

キャリブレーションの判断を改善し、PPU Type 0 と Type 1 を使って実際の Model Zoo モデルをビルドし、YOLO26(既定では n。s、m、l、x は変数を 1 つ変えるだけで切り替え可能)に対する TopK 先行最適化の効果を測定します。

<!-- cell: 6dffc67d-0a31-4b77-b975-65a83973f314 src: caf8fba665 -->
> このチュートリアルの全 3 部のうちの第 2 部です。コースマップと各部の一覧は入門ノートブックにあります。

<!-- cell: objectives src: 46a3025a59 -->
## 学習目標

このチュートリアルを終えると、次のことができるようになります。

- 代表的なキャリブレーションデータを選択し、モデルの前処理を再現する
- 検出処理のうち、どの演算が NPU の PPU で実行され、どの演算がホスト CPU に残るかを説明する
- 検出ヘッドのアーキテクチャから PPU Type 0 または Type 1 を選択する
- PPU のレイヤーマッピングを ONNX グラフと照らし合わせて検証する
- Model Zoo の YOLOv7 と YOLOX-S モデルをハードウェア PPU 付きでコンパイルする
- PPU 版と非 PPU 版の Model Zoo DXNN のランタイム挙動を比較する
- 任意のサイズ(n、s、m、l、x)の YOLO26 モデルに TopK 先行最適化を適用する
- `dxrun` でベースラインと最適化後の DXNN の性能を比較する

このノートブックは SDK のソースツリーを変更しません。生成されるファイルはすべて `<dx-tutorials>/notebooks/T05-DX-Compiler/workspace` 配下に保存され、git では無視されます。

<!-- cell: 64617a25-7ca0-407c-97d2-80bcd28d505c src: f0911d99ba -->
## 前提条件

- チュートリアル 05-1 の完了: `venv-dx-compiler-local` 内の `dxcom` と、サンプルのキャリブレーションデータセット。
- 第 4 節と第 6 節の `dxrun` 比較のために、DEEPX NPU を搭載したマシンに DX-RT がインストールされていること。
- ダウンロード: Model Zoo の YOLOv7 と YOLOX の ONNX および DXNN ファイル(約 290 MB)。
- 所要時間: YOLOv7 640x640 のコンパイルはこのシリーズで最も時間がかかります。ラップトップでは数十分を見込んでください。

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

<!-- cell: ppu-types src: 7508872dcf -->
### 3.2 モデルアーキテクチャからモードを選択する

| モード | アーキテクチャ | 代表的なモデル | レイヤーマッピング | 実行場所 |
|---|---|---|---|---|
| PPU Type 0 | アンカーベース(anchor-based) | YOLOv3, YOLOv4, YOLOv5, YOLOv7 | 検出 Conv ノード → アンカー数 | PPU ハードウェア |
| PPU Type 1 | アンカーフリー(anchor-free) | YOLOX, YOLOv8–YOLOv12 | `bbox`、`obj_conf`(存在する場合)、`cls_conf` ノード | PPU ハードウェア |
| `pre_optimize()` | TopK 先行の ONNX 書き換え | YOLOv8 ファミリー, YOLOv10, YOLO26 | スケールごとの bbox/class 出力テンソル | NPU + ホスト CPU グラフ |

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

<!-- cell: topk-heading src: a8f8add50a -->
## 6. YOLO26 の TopK ベース最適化 (n / s / m / l / x)

YOLO26 は one-to-one の分離型検出ヘッドを使用します。`yolo26_postprocess` 変換は、TopK 選択を CPU 側の高コストなデコード処理の前に移動します。これは PPU Type 0/1 とは異なり、ONNX グラフを書き換えて、NPU 推論後に処理される候補のワークロードを削減します。

```text
Baseline: 8,400 candidates → decode and score all candidates → TopK 300
Optimized: 8,400 candidates → TopK 300 → decode and score only 300 candidates
```

両方のモデルは同じ検出結果のコントラクト `[1, 300, 6]` を維持しますが、CPU 側の処理の順序が変わります。

この節は Model Zoo のすべての YOLO26 サイズで動作します。6.1 の変数 `YOLO26_VARIANT` 1 つでモデルを選択し、以下のすべてのファイル名、ヘッドテンソルの確認、コンパイル、ベンチマークがそれに従います。5 つのエクスポートはすべて同じ検出ヘッドレイアウトを共有しているため、TopK 変換は同一です。まず `n` から始め、その後に変数を変更してより大きなモデルでこの節を再実行してください。

| `YOLO26_VARIANT` | ONNX ダウンロード | モデル 1 つあたりのコンパイル時間(x86_64 デスクトップ。この節では 2 つのモデルをコンパイルします) |
|---|---:|---|
| `n`(既定) | 10 MB | 約 2 分(実測) |
| `s` | 37 MB | 約 2 分(実測) |
| `m` | 78 MB | 未測定。数分を見込む |
| `l` | 95 MB | 未測定。10 分以上を見込む |
| `x` | 213 MB | 未測定。10 分以上を見込む |

各バリアントは `models/`、`configs/`、`outputs/` 配下に独自のファイルを使用するため、すでにコンパイル済みのサイズに戻した場合はコンパイルがスキップされます。

<!-- cell: download-yolo26-heading src: 306e030b72 -->
### 6.1 ベースラインモデルをダウンロードして検証する

`YOLO26_VARIANT` を設定してセルを実行します。ベースラインの ONNX と対応する Q-Lite JSON は Model Zoo から取得します(`yolo26-<variant>_640x640.onnx` と `.json`)。安全なダウンローダーにより、シェルのダウンロードが失敗しても次のノートブックセルが続行してしまうことで発生する空ファイルのエラーを防ぎます。

<!-- cell: verify-heads-heading src: adfa304cc8 -->
### 6.2 6 つのヘッドテンソルを検証する

この変換には、3 つの検出スケールそれぞれにバウンディングボックステンソル 1 つとクラス信頼度テンソル 1 つが必要です。これらは**出力テンソル名**であり、他のモデルからコピーした表示ラベルではありません。Model Zoo にエクスポートされたすべての YOLO26 サイズは同じヘッドレイアウト(ボックスは `/model.23/one2one_cv2.<scale>/...`、クラススコアは `/model.23/one2one_cv3.<scale>/...`)を共有しているため、以下の名前は 1 つのパターンから生成され、実際にダウンロードした ONNX と照合されます。確認に失敗した場合は、Netron でモデルを開いてパターンを更新してください。

<!-- cell: apply-topk-heading src: 1ac20c8599 -->
### 6.3 `yolo26_postprocess` を適用する

`dx_com` がインストールされているのは DX-COM の Python 環境なので、変換はその環境で実行します。別の ONNX ファイルを作成するため、ベースラインは変更されません。

スクリプトはベースラインの ONNX と出力先パスを引数として受け取るため、同じスクリプトがすべてのバリアントに使えます。次の 2 つのセルでスクリプトを書き出して実行します。既定の `n` バリアントの場合、実行内容は次と同等です。

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
python optimize_yolo26.py models/yolo26-n_640x640.onnx models/yolo26-n_640x640_topk300.onnx
```

<!-- cell: adapt-yolo26-heading src: f1bf82c079 -->
### 6.4 Model Zoo の設定を調整する

ベースラインモデルと最適化モデルには、同じキャリブレーションと前処理の設定を使用する必要があります。変更するのは利用できないデータセットパスだけです。これにより性能比較の条件が揃います。

<!-- cell: compile-yolo26-heading src: e70f6d348a -->
### 6.5 両方のモデルをコンパイルして確認する

各 ONNX を別々の出力ディレクトリにコンパイルします。両方のコマンドは同じ JSON とコンパイラオプションを使用し、ONNX グラフだけが唯一の実験変数です。

<!-- cell: compile-yolo26-baseline-command src: fa81283b97 -->
#### 6.5.1 ベースラインをコンパイルする

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

<!-- cell: compile-yolo26-topk-command src: 1c1158411a -->
#### 6.5.2 TopK モデルをコンパイルする

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

<!-- cell: benchmark-heading src: 917423f3f9 -->
### 6.6 `dxrun` でベースラインと TopK の性能を比較する

`dxrun --use-ort -t 5` は、合成入力を使って各 DXNN を 5 秒間ベンチマークします。`--use-ort` は NPU グラフに加えて CPU サブグラフも実行するもので、この比較には必須です。

両方のモデルを同じデバイス上で、同程度の温度条件とシステム負荷のもとで実行してください。この結果が測定するのはランタイムのスループットであり、検出精度やフルビデオパイプラインの FPS ではありません。リリース判断の際は測定を繰り返してください。

既定の `n` バリアントの場合:

```bash
dxrun -m outputs/yolo26n_baseline/yolo26-n_640x640.dxnn --use-ort -t 5
dxrun -m outputs/yolo26n_topk300/yolo26-n_640x640_topk300.dxnn --use-ort -t 5
```

<!-- cell: 6be8d246-d16c-429d-a82f-e1c45594974c src: b7be7c85e3 -->
## 7. トラブルシューティング

| 症状 | 表示される内容 | 考えられる原因 | 対処 |
|---|---|---|---|
| PPU のノード名が見つからない | `ValueError: ... not found in the ONNX graph` | PPU 設定の `layer` の名前が別のエクスポートのものである | Netron で ONNX を確認し、JSON のノード名を更新する |
| `dxrun` がデバイスを見つけられない | `No device found` または `Fail to initialize device` | このホストに DEEPX NPU がない | NPU を搭載したホストで比較を実行する。コンパイルのみの手順はそのまま動作する |
| FPS を読み取れない | `Could not read FPS from dxrun output` | `dxrun` の出力形式が変わった | 上に完全な出力を表示し、FPS の行を手動で読み取る |
| JSON の変更が反映されない | `Skip compilation: found ...` | 出力ディレクトリにすでに DXNN が存在する | `workspace/outputs/<dir>` を削除して再実行する |
| ダウンロードが不完全 | `workspace/models` に極小または 0 バイトのファイルがある | ダウンロードが中断された | ファイルを削除する。ダウンロードヘルパーは使用前にサイズを検証する |

<!-- cell: summary src: a42f00d89b -->
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

YOLO26 decoupled head
          │
          ▼
 TopK-first ONNX rewrite
          │
          ▼
 Reduce candidates before
 CPU-side decoding
```

### 8.2 3 つの最適化パス

| トピック | 使用モデル | 主な判断 | 最適化の場所 | ホスト CPU の役割 |
|---|---|---|---|---|
| PPU Type 0 | YOLOv7 | アンカーベースのヘッドを選択し、その Conv ノードをマッピングする | ハードウェア PPU | NMS とアプリケーションロジック |
| PPU Type 1 | YOLOX-S | bbox、objectness、クラスの各ブランチをマッピングする | ハードウェア PPU | NMS とアプリケーションロジック |
| TopK 先行最適化 | YOLO26(`YOLO26_VARIANT`、既定は n) | TopK を高コストなデコードの前に移動する | 書き換えた ONNX グラフ | より少ない候補をデコードする |

### 8.3 完了した実験

| 実験 | 比較したモデル | 統制した変数 | 検証方法 |
|---|---|---|---|
| Type 0 PPU | YOLOv7 非 PPU 版と PPU 版 | ハードウェア後処理 | `dxparse`、`dxrun` |
| Type 1 PPU | YOLOX-S 非 PPU 版と PPU 版 | ハードウェア後処理 | `dxparse`、`dxrun` |
| TopK 最適化 | YOLO26 ベースラインと TopK 300(選択したサイズ) | ONNX グラフ構造 | 出力コントラクト、`dxparse`、`dxrun` |

### 8.4 完了チェックリスト

- [ ] キャリブレーションデータと前処理をモデル定義の一部として扱う
- [ ] NPU、PPU、ホスト CPU の処理の境界を特定する
- [ ] YOLOv7 Type 0 の Conv ノードマッピングを検証する
- [ ] Type 0 PPU モデルをコンパイルして確認する
- [ ] YOLOX-S Type 1 の bbox、objectness、クラスのマッピングを検証する
- [ ] Type 1 PPU モデルをコンパイルして確認する
- [ ] 同じベンチマーク条件で PPU 版と非 PPU 版のモデルを比較する
- [ ] YOLO26 モデルに TopK 先行最適化を適用する(そして 2 つ目のサイズも試す)
- [ ] ベースラインと TopK モデルが同じ出力コントラクトを維持していることを検証する
- [ ] ベースラインと TopK の DXNN ランタイム性能を比較する
- [ ] 実データで精度、ホスト CPU 使用率、フルパイプラインのレイテンシを検証する

### 8.5 選び方

| モデルが持つもの | まず試すもの |
|---|---|
| アンカーベースの YOLO ヘッド | PPU Type 0 |
| 独立した bbox、objectness、クラスのブランチ | PPU Type 1 |
| 大量の候補の後に続く TopK | TopK 先行のグラフ最適化 |
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
