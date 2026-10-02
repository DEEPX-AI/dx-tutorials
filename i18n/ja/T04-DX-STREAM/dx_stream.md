<!-- i18n source: notebooks/T04-DX-STREAM/dx_stream.ipynb -->
<!-- i18n lang: ja -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 1726e9e4-5821-4f34-add4-37f1fb44ee77 src: c7877fefea -->
# DEEPX Tutorial 04 - DX-STREAM ワークフロー

このチュートリアルでは DX-STREAM v3.1.x を紹介し、DEEPX NPU 上でエンドツーエンドの Vision AI パイプラインを構築する方法を説明します。

## 学習目標

このチュートリアルでは次のことを学びます。

- DEEPX SDK における DX-STREAM の位置づけを理解する
- `DxPreprocess → DxInfer → DxPostprocess → DxTracker → DxOsd` でパイプラインを構築する
- DX マルチストリームドメインとセレクターの境界を学ぶ
- 推論バックエンド、スケーリング、変換、メッセージブローカーの各エレメントを使う
- カスタム AI モデル向けにカスタム後処理ライブラリを適応させる
- メタデータ駆動の人数カウントオーバーレイを追加する
- メタデータを調べ、GStreamer パイプラインを診断する

<!-- cell: 32d7336d-3f4d-4c4a-aa70-2662f4558edc src: 6645080682 -->
## 1. DX-STREAM の概要

DX-STREAM は、DEEPX NPU 上で Vision AI パイプラインを構築するための GStreamer エレメント群です。

```text
Source → Decode → DxPreprocess → DxInfer → DxPostprocess → DxTracker → DxOsd → Sink
```

![DX-STREAM パイプライン](assets/dx-stream-pipeline.png)

| エレメント | 用途 |
|---|---|
| `dxpreprocess` | フレームをリサイズ、クロップし、モデル入力テンソルに変換します |
| `dxinfer` | コンパイル済みの `.dxnn` モデルを実行します |
| `dxpostprocess` | 出力テンソルをデコードし、推論メタデータを書き込みます |
| `dxtracker` | 検出されたオブジェクトに永続的な ID を割り当てます |
| `dxosd` | ボックス、ラベル、姿勢、セグメンテーション結果を描画します |
| `dxgather` | 同じソースから作成されたブランチを統合します |
| `dxinputselector` | 入力ストリームを DX マルチストリームドメインに統合します |
| `dxoutputselector` | DX マルチストリームドメインをストリーム ID ごとに分割します |
| `dxrate` | 出力フレームレートを制御します |
| `dxmsgconv` | 推論メタデータを構造化メッセージに変換します |
| `dxmsgbroker` | メッセージを MQTT または Kafka に配信します |
| `dxscale` | フレームの解像度を変更します |
| `dxconvert` | フレームのカラーフォーマットを変更します |

エレメントとプロパティの完全なリファレンスは、[DEEPX Developer Portal](https://developer.deepx.ai/download/?id=583) から入手できる DX-STREAM User Manual を参照してください。ログインが必要です。

<!-- cell: 2d53a070-40a1-4658-950d-62c7c796af9b src: fe4feb49aa -->
## 2. 前提条件

- チュートリアル 01 を `./dx-runtime/install.sh --all` で完了していること: DX-RT、DX-APP、DX-STREAM がビルド済みで、NPU が認識されていること。チュートリアル 03 の完了を推奨します。第 5 節では、そこで作成した Forklift and Worker モデルが存在すれば再利用します。
- ウィンドウ表示のためのグラフィカルデスクトップセッション。セットアップセルはディスプレイを検出します。ディスプレイがあればパイプラインのセルは DX-STREAM ウィンドウを開き、ない場合(たとえば SSH 経由)は `fakesink` を使って**ヘッドレス**で実行し、ノートブックが最後まで完了するようにします。`HEADLESS` はどちらの値にも強制設定できます。第 4 節の `run_demo.sh` デモとカメラのセルはそれぞれ数分かかるため、オプトイン(`RUN_DEMOS`、`RUN_CAMERA`)になっています。
- 任意のハードウェア: 4.3 用の V4L2 カメラ(`RUN_CAMERA`)、4.9 用の到達可能な RTSP ストリーム、4.11 用の MQTT または Kafka ブローカー。
- ツール: `gstreamer1.0-tools`、`v4l-utils`、`jq`(`setup.sh` が使用)、7.3 用の `graphviz`。5.7 節では `meson` で DX-STREAM を再ビルドするため、`sudo` が必要です。
- DX-STREAM GStreamer プラグインが `gst-inspect-1.0` から見えていること。`build.sh` は `export GST_PLUGIN_PATH=...` の行を表示します。セットアップセルはこれを確認し、プラグインが既定の場所にある場合はカーネル用にこの変数を設定します。
- ダウンロード: `setup.sh` はサンプルモデル 17 個(約 300 MB)と共有サンプル動画アーカイブ(約 1.1 GB。`<dx-all-suite>/workspace/res/videos` に保存され、DX-APP と共有されます)を取得します。第 5 節はチュートリアル 03 のモデルと動画を再利用するか、90 MB をダウンロードします。
- このチュートリアルで作成されるファイルは `notebooks/T04-DX-STREAM/workspace/` に置かれ、git では無視されます。SDK ツリー内で変更されるのは 5.6 の後処理パッチだけで、それがその節の目的です。

次のセルは SDK の場所を特定し、これらの要件を確認してステータス表を表示します。

<!-- cell: 53136a15-c25c-4168-aa38-a07629a40996 src: 1605721eba -->
### 2.1 SDK パスの読み込み

次のセルは `DX_ALL_SUITE_DIR` から `DX_STREAM_DIR` とその他の SDK パスを導出し、ノートブックの作業ディレクトリを、`run_demo.sh` とサンプルのパスが前提とする DX-STREAM リポジトリのルートに変更します。また、以降で使用する表示フラグとワークスペースのパスも定義します。

<!-- cell: 8d9cf255-57c7-4b99-be3b-741331f6e532 src: 75741dab3d -->
### 2.2 (任意) Intel GPU の前提条件

Intel x86_64 iGPU/dGPU システムの場合のみ、[`docs/intel_gpu.md`](../../docs/intel_gpu.md) に従ってください。

<!-- cell: e7c0a679-122a-4e4e-bae9-1c034689e281 src: 5ae0009348 -->
### 2.3 サンプルリソースのダウンロード

`setup.sh` は、同梱パイプラインに必要なモデルと動画を `dx_stream/samples/` にダウンロードします。現在のリストにはモデル 17 個(約 300 MB)と共有サンプル動画アーカイブ(約 1.1 GB。`setup.sh` が `<dx-all-suite>/workspace/res/videos` に展開し、`dx_stream/samples/videos` としてリンクします)が含まれます。初回実行では数分かかります。サンプルがすでに存在する場合、セルはスクリプトをスキップします。

<!-- cell: fe12c34c-a475-437f-bba9-d6ed0668f6b0 src: de693a0b81 -->
### 2.4 インストール済み DX-STREAM エレメントの確認

プラグインは次の 13 個のエレメントを公開しているはずです。

`dxconvert`、`dxgather`、`dxinfer`、`dxinputselector`、`dxmsgbroker`、`dxmsgconv`、`dxosd`、`dxoutputselector`、`dxpostprocess`、`dxpreprocess`、`dxrate`、`dxscale`、`dxtracker`。

<!-- cell: 6a2cff9b-d3eb-48ad-be1e-c18fb204726d src: 3e2aa3dfa3 -->
## 3. クイックスタート

### 3.1 YOLO26n 物体検出パイプラインの実行

このパイプラインは動画を読み込み、各フレームを前処理して YOLO26n を実行し、結果をデコードしてアノテーション付きの出力をレンダリングします。デスクトップセッション(`HEADLESS = False`、ディスプレイが検出された場合の既定値)では結果がウィンドウに表示され、動画が終了するか停止ボタン(■)を押すまでセルは実行中のままになります。ディスプレイがない場合(`HEADLESS = True`)、フレームは `fakesink` に送られ、セルはパイプラインの状態メッセージだけを表示します。この場合の目的は、すべてのエレメントがリンクされ、モデルが読み込まれ、推論が実行されることを確認することです。セルは使用するシンクを表示します。

代わりにターミナル(File > New > Terminal)で実行するには、セットアップセルが表示した DX-STREAM ディレクトリに `cd` し、下の `gst-launch-1.0` コマンドの最後のエレメントを `videoconvert ! fpsdisplaysink sync=false` にして貼り付けます。`Ctrl+C` で停止します。

<!-- cell: 31792737-11bd-409e-975b-6d068cf213a8 src: b9702f410a -->
### 3.2 コアエレメントの確認

`gst-inspect-1.0` は、インストールされているバージョンのパッドテンプレート、プロパティ、既定値、サポートされる列挙値を表示します。

<!-- cell: ce012ae3-7b15-4cde-89fa-9e7a998f985f src: e7ac256517 -->
#### 3.2.1 DxPreprocess

```bash
dxpreprocess preprocess-id=1 resize-width=640 resize-height=640
```

`preprocess-id` は生成される入力テンソルを識別します。下流の `dxinfer` は同じ ID を参照する必要があります。

<!-- cell: 286b0cb5-2662-454e-a1a3-ae94aff066ce src: 5260c0d019 -->
#### 3.2.2 DxInfer と推論バックエンド

```bash
dxinfer preprocess-id=1 inference-id=1 model-path=/path/to/model.dxnn backend=auto
```

- `auto`: 利用可能なコンパイル済みバックエンドを選択します
- `dxrt`: DEEPX Runtime バックエンドを使用します
- `dxvnpu`: DX-STREAM が VNPU サポート付きでビルドされている場合に VNPU バックエンドを使用します

`inference-id` は `dxpostprocess` が消費する出力テンソルを識別します。DX-STREAM v3.1.x は内部で非同期の Put/Get バックエンドインターフェースを使用しますが、パイプラインの構文は変わりません。

<!-- cell: eba1a455-60a5-4898-99b4-ab0a40a00f21 src: b9001c0bcb -->
#### 3.2.3 DxPostprocess

```bash
dxpostprocess inference-id=1 \
  library-file-path=/usr/local/share/gstdxstream/lib/libpostprocess_yolo26od.so \
  function-name=PostProcess
```

`inference-id` は上流の推論出力と一致している必要があります。共有ライブラリと関数はモデルアーキテクチャに対応したものでなければなりません。

<!-- cell: 98989ca3-7203-4c23-9311-c49d59adf254 src: 42550f73c9 -->
#### 3.2.4 DxOsd

DxOsd は推論メタデータを読み取り、視覚的な結果を映像フレームに重ねて描画します。

<!-- cell: 98f4122d-835f-46a0-b359-58d35c5ef222 src: 86672b6017 -->
### 3.3 映像フレームのスケーリングと変換

`dxscale` は解像度を変更し、`dxconvert` はカラーフォーマットを変更します。対応プラットフォームでは高速化されたカーネルが自動的に選択され、サポートされない組み合わせはソフトウェア実装にフォールバックします。

```bash
... ! dxscale width=640 height=480 ! \
      dxconvert ! video/x-raw,format=RGB ! ...
```

`dxconvert` はフレームをリサイズせず、`dxscale` は最終的なカラーフォーマットを選択しません。

このノートブックのパイプラインは、表示シンクの前にソフトウェアの `videoconvert` を置いて終わります。ハードウェアの `dxconvert` の方が高速ですが、この SDK リリースでは一部のクリップ(たとえば第 5 節の Forklift and Worker 動画)で断続的に全面緑色のフレームが生じました。同梱スクリプトは `VIDEOCONVERT_PIPELINE` 変数でプラットフォームごとにコンバーターを選択します。`SINK` を `dxconvert` に切り替えて緑色のフレームが表示される場合は、元に戻してください。

<!-- cell: a2bacdbc-f65c-4c63-804d-88241bb9f07a src: 17eb05a204 -->
## 4. 同梱デモパイプラインの実行

`run_demo.sh` には 12 の選択肢があります。`0` から `9`、Secondary Mode の `-`、Depth Estimation の `=` です。10 秒以内に選択がなければオプション `0` を実行します。次のセルはインストールされているメニューを表示します。

これらのデモはネイティブウィンドウを開き、パイプラインが終了するまでセルを占有するため、以下のセルはデスクトップセッションで `RUN_DEMOS = True`(セットアップセル)のときだけ実行されます。推奨される方法はターミナル(File > New > Terminal)です。

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_stream     # the setup cell printed your actual path
./run_demo.sh
```

メニュー項目はすぐに入力し(プロンプトは 10 秒でタイムアウトします)、`Ctrl+C` でパイプラインを停止します。各デモの後には、実行されるスクリプトと、その中で注目すべき点についてのメモが続きます。

<!-- cell: cdce0b1a-4fae-4368-9720-5bfb211b0776 src: 63e050e95e -->
実行中のパイプラインはノートブックの停止ボタン(`■`)で停止します。

### 4.1 物体検出 - YOLO26n

![単一の物体検出パイプライン](assets/pipline-single-detection.png)

注目ポイント: クイックスタートと同じ 5 つの DX エレメント。`model-path` と `library-file-path=.../libpostprocess_yolo26od.so` が `dxinfer` と `dxpostprocess` に直接設定されています。

<!-- cell: f9415924-9b2c-4691-8811-46da31c7b397 src: c79045ee08 -->
### 4.2 PPU を使った物体検出 - YOLOv5s

注目ポイント: `library-file-path` がありません。各 DX エレメントは `dx_stream/configs/YoloV5S_PPU/` 配下の JSON `config-file-path` を読み込みます。PPU とは NPU がすでにボックスをデコード済みであることを意味し、後処理の設定は PPU 出力のマッピングだけを行います。

<!-- cell: 0598f06b-a1de-40bd-8339-d6c93eff57a5 src: ef74dca18a -->
### 4.3 顔検出

注目ポイント: 4.1 の構造に顔モデルと `libpostprocess_yolov5s_face.so` を組み合わせたもの。`dxosd` が顔のボックスとランドマークを描画します。

<!-- cell: 95a894ed-f224-4e72-b6f3-b2fc702b4f9e src: 51efb7d6f3 -->
次の例は PPU 出力付きの SCRFD500M モデルを使用します。

注目ポイント: 汎用の `libpostprocess_ppu.so`。PPU モデルでは、1 つの共有ライブラリが複数のアーキテクチャに対応します。

<!-- cell: bed6a8d7-6e12-444d-85c4-72c6e7dab466 src: f29ec878ae -->
#### カメラソースの使用

最初の例は raw 映像を要求します。2 つ目は MJPEG を要求しますが、すべての USB カメラがサポートしているわけではありません。`v4l2-ctl --list-formats-ext --device=/dev/video0` の結果に合わせてデバイスと caps を調整してください。

<!-- cell: db029aad-2a05-4904-946c-d7b67e8e27cc src: 9f586331fc -->
### 4.4 姿勢推定

注目ポイント: `libpostprocess_yolo26pose.so` がキーポイントを `DXObjectMeta` に書き込み、`dxosd` がそれをもとにスケルトンを描画します。

<!-- cell: 70f47140-89cd-477b-abfe-e5c5ed869042 src: b99cd2cfe3 -->
次の例は PPU 出力付きの YOLOv5 Pose を使用します。

注目ポイント: ここでも `libpostprocess_ppu.so` です。4.4 と比較してください。変わるのはモデルとライブラリだけです。

<!-- cell: fc15ee5e-4fc9-4595-a4e2-3da2f54f5420 src: eece8565d4 -->
### 4.5 インスタンスセグメンテーション - YOLO26n-Seg

オプション `6` はセグメンテーションデモを実行します。現在の SDK レイアウトでは、スクリプトは `instance_segmentation` 配下にあります。

注目ポイント: `dxpostprocess` と `dxosd` の間にある `dxtracker` エレメント。セグメンテーションデモはマスクにもトラック ID を割り当てます。

<!-- cell: ba5d8d52-5fe5-4cd2-93b8-b2db52570a1a src: 2a8e38c8f9 -->
### 4.6 複数物体追跡

オプション `7` は物体検出に続けて追跡を実行します。

![単一の追跡パイプライン](assets/pipline-single-tracking.png)

注目ポイント: 4.2 の検出チェーンに `dxtracker config-file-path=.../tracker_config.json` を加えたもの。トラッカーの設定でアルゴリズムとそのしきい値を選択します。

<!-- cell: 154b4e5c-5a1f-4036-9be6-fd8e8e1a39ab src: ea853b3cad -->
### 4.7 マルチチャネル物体検出

オプション `8` は 4 つの独立した推論ブランチを実行し、その出力を合成します。

![マルチチャネルパイプライン](assets/pipeline-multi-stream.png)

注目ポイント: 4 つの `filesrc ... ! dxscale` ブランチがそれぞれ独自の前処理、推論、後処理を持ち、`compositor` で統合されます。モデルはブランチごとに 1 回ずつ読み込まれます。

<!-- cell: 30471517-0937-4f1a-bd77-f41ac4749987 src: 3b21ecdbaf -->
### 4.8 DX マルチストリームドメイン

DX-STREAM v3.1.0 では `application/x-dxvideoraw` caps ドメインが導入されました。これにより、各ストリームの識別情報、寸法、フォーマット、メタデータ、タイムラインを保ったまま、複数のストリームが 1 つの処理チェーンを共有できます。

```text
N × video/x-raw
       ↓
dxinputselector                 domain entry
       ↓ application/x-dxvideoraw
dxpreprocess → dxinfer → dxpostprocess → dxosd
       ↓ application/x-dxvideoraw
dxoutputselector                domain exit
       ↓
N × video/x-raw
```

`dxinputselector` はストリーム ID を割り当て、必要に応じて `DXFrameMeta` を作成します。`dxoutputselector` は各バッファをルーティングし、対応するストリームごとのイベントを復元します。

このドメインはストリームごとの `STREAM_START`、`CAPS`、`SEGMENT`、`TAG`、`EOS`、`GAP` イベントを保持します。そのため、遅いストリームはすべての入力を抑制することなく、自身の QoS を独自に処理できます。

<!-- cell: 0ecc9dfd-6d20-4385-b4c3-f5bcc9b6bf70 src: aa2cd3fde1 -->
#### エレメント配置のルール

| エレメント | `application/x-dxvideoraw` 内 | 配置に関する注意 |
|---|---:|---|
| `dxpreprocess`、`dxinfer`、`dxpostprocess`、`dxtracker`、`dxosd`、`dxrate` | 可 | これらのエレメントは単一ストリームモードでもドメインモードでも動作します |
| `dxscale`、`dxconvert` | 不可 | `dxinputselector` の前か `dxoutputselector` の後に配置します |
| `dxgather` | 不可 | 1 つのソースから分かれたブランチを統合するもので、異なるストリーム ID を統合するものではありません |
| `videoconvert`、`videoscale`、`compositor` などの標準エレメント | 不可 | `application/x-dxvideoraw` ではなく `video/x-raw` を受け付けます |
| `dxinputselector`、`dxoutputselector` | 境界のみ | ドメインへの入口と出口になります |

誤った配置は、曖昧な実行時の結果を生むのではなく、caps ネゴシエーションの段階で失敗します。

<!-- cell: e89082d6-2e11-4491-acc9-b64f356b528a src: 76d7544217 -->
#### 1 つの推論チェーンを共有する

すべてのチャネルが同じモデルを使う場合、単一の共有推論チェーンにすることで、チャネルごとにモデルを読み込むことを避け、NPU のメモリ使用量を削減できます。

<img src="assets/pipeline-multi-stream-single-infer.png" style="max-width: 1400px;">

現在の例は `run_multi_stream_selector.sh` を使用します。

注目ポイント: 1 つの共有チェーンを `dxinputselector name=in` と `dxoutputselector name=out` が囲んでいます。DX エレメントの数を数えて 4.7 と比較してください。

<!-- cell: e8f05808-0f68-475f-b0b0-d9ee7ba85240 src: c7e15732ef -->
### 4.9 マルチチャネル RTSP

オプション `9` は複数の RTSP ソースを実演します。ライブパイプラインは、v3.1.x で修正されたレイテンシと QoS のレポートの恩恵を受けます。

注目ポイント: 4.8 のセレクター構成に `rtspsrc` 入力を組み合わせたもの。ストリーム URL はスクリプト先頭の変数です。お使いのカメラに合わせて編集してください。

<!-- cell: 317ba344-632d-4528-964e-4411d31d7b56 src: 98c893db6e -->
### 4.10 Secondary Mode

![Secondary Mode パイプライン](assets/pipeline-secondary.png)

- **Primary Mode(プライマリモード)** はフレーム全体を前処理して推論します。後処理は通常、新しい `DXObjectMeta` オブジェクトを作成します。
- **Secondary Mode(セカンダリモード)** は検出されたオブジェクト領域を前処理します。後処理は既存のオブジェクトメタデータを更新または拡充します。

注目ポイント: 2 組目の前処理・推論・後処理の 3 つ組に設定された `secondary-mode=true`。SCRFD が顔を検出し、その後 EfficientNet がフレーム全体ではなく各顔領域を分類します。

<!-- cell: c73d9a02-064d-4b78-a062-5598aa8546cc src: 6c9c479a23 -->
### 4.11 推論結果を MQTT または Kafka に配信する

`dxmsgconv` はカスタムのメッセージ変換ライブラリを使って推論メタデータを変換します。`dxmsgbroker` はそのペイロードを MQTT または Kafka ブローカーに配信します。

```text
... → dxpostprocess → dxmsgconv → dxmsgbroker
```

v3.1.x では、`include-frame=true` を指定すると現在のフレームが Base64 エンコードされた JPEG としてメッセージに追加されます。これにより JPEG エンコードの処理、メッセージサイズ、ネットワークトラフィックが増えるため、コンシューマーが画像を必要とする場合にのみ有効にしてください。

```bash
dxmsgconv library-file-path=/path/to/libmessage_convert.so include-frame=true ! \
dxmsgbroker broker-name=mqtt conn-info=localhost:1883 topic=test
```

パイプラインを開始する前にブローカーが起動している必要があります。同梱スクリプトには完全な MQTT と Kafka の例が含まれていますが、対話型の `run_demo.sh` メニューには含まれていません。

<!-- cell: 4f9da17d-7791-42e5-9930-16a25ace619a src: a8c8d5a58b -->
## 5. 独自アプリケーションの作成

この節では、チュートリアル 03 で作成した Forklift and Worker 検出器を統合します。このモデルは YOLOv7 の出力形式を使いますが、COCO の 80 クラスではなく 2 クラスしか持たないため、後処理の設定を適応させる必要があります。

![カスタムパイプライン](assets/custom-pipeline.png)

<img src="assets/detection-goal.jpg" style="max-width: 1400px;">

<!-- cell: 66060d59-25bd-463a-8c02-e67713decc74 src: 423cab8027 -->
### 5.1 DX-STREAM v3.1.x カスタムライブラリ API

旧バージョン向けに書かれたカスタムの前処理・後処理ライブラリは、次のように更新する必要があります。

- 循環的なバッファ参照を避けるため、`DXFrameMeta::_buf` は削除されました。
- カスタム関数の最初の引数は `GstBuffer *buf` になりました。
- オブジェクトメタデータは `dx_acquire_obj_meta_from_pool()` で作成します。
- 新しいオブジェクトは `dx_add_obj_meta_to_frame()` でアタッチします。
- Primary Mode では、後処理が結果オブジェクトを作成します。
- Secondary Mode では、後処理が `object_meta` で渡されたオブジェクトを更新します。

現在の YoloV7 ライブラリは、すでに v3.1.x の関数形式を使用しています。

```cpp
extern "C" void PostProcess(
    GstBuffer* buf,
    std::vector<dxs::DXTensor> network_output,
    DXFrameMeta* frame_meta,
    DXObjectMeta* object_meta)
{
    DXObjectMeta* result = dx_acquire_obj_meta_from_pool();

    // Decode tensors and populate result.

    dx_add_obj_meta_to_frame(frame_meta, result);
}
```

<!-- cell: 48523baa-d664-4f24-836e-f876a6c6959e src: e73e58aa61 -->
### 5.2 メタデータの階層

推論結果は、別のサイドチャネルではなく `GstBuffer` と共に流れます。

```text
GstBuffer
└── DXFrameMeta
    ├── stream ID, width, height, format, ROI
    ├── input tensors  (preprocess ID → tensors)
    ├── output tensors (inference ID → tensors)
    ├── frame-level classification or segmentation
    ├── DXObjectMeta[]
    │   ├── label, confidence, box, tracking ID
    │   ├── keypoints, features, OBB, face, segmentation
    │   └── DXUserMeta[]
    └── DXUserMeta[]
```

`DXUserMeta` を使うと、アプリケーション固有のデータをフレームやオブジェクトにアタッチできます。GStreamer がバッファをコピーまたは解放したときにメタデータが有効であり続けるよう、ユーザーメタの実装はコピー関数と解放関数の両方を提供する必要があります。

<!-- cell: 712b8026-aa30-4662-a7ec-5f7a2e79e179 src: 77b28e4653 -->
### 5.3 カスタム DXNN モデルの取得

チュートリアル 03 は `yolov7-forklift-person.dxnn` をそのワークスペースにコンパイル(またはダウンロード)しました。セルはそのファイルが存在すればコピーし、存在しなければ同じモデル(71 MB)をダウンロードします。

```bash
wget -nc https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/yolov7-forklift-person.dxnn
```

<!-- cell: 8e29f8a5-4356-4e58-93a9-9c247bfa7c77 src: fa60065b2f -->
### 5.4 コンパイル済みモデルの確認

後処理に手を入れる前に、モデルが何を出力するかを確認します。`dxparse -v` は入力テンソルと出力テンソルを表示します。カスタムモデルは YOLOv7 のヘッドレイアウトを維持しており、raw 検出ヘッドは COCO モデルの `3 × (5 + 80) = 255` ではなく `3 × (5 + 2) = 21` チャネル(3 つのアンカー、4 つのボックス値、objectness、2 つのクラススコア)を持ちます。`libpostprocess_yolov7.so` のデコーダーはすでにこのレイアウトを処理できます。異なるのはクラス数とクラス名だけであり、5.5 のパッチが変更するのはまさにその部分です。

<!-- cell: 6972d420-7312-4a41-bd63-1a01205af19a src: 0127f93eee -->
### 5.5 YoloV7 後処理設定の適応

5.4 で示したとおり、出力デコーダーはすでにモデルアーキテクチャに一致しています。変更が必要なのはクラスの数と名前だけです。パッチはワークスペースに書き込まれ、5.6 で SDK ソースに適用されます。

<!-- cell: b16382e7-7b3a-45e5-88bd-9bb80cc61480 src: 1e14d10b82 -->
### 5.6 パッチの適用

以下のセルは複数回実行しても安全です。必要なときにパッチを適用し、同じパッチがすでに存在する場合は成功と報告し、ソースに競合する変更がある場合はファイルを変更せずに停止します。既存の作業をリセットしたり破棄したりすることはありません。

<!-- cell: 4f49345d-1354-4454-9bfe-bf1de2b099b7 src: b1a18419d0 -->
### 5.7 DX-STREAM の再ビルド

`build.sh` は `meson` でプラグインとカスタムライブラリを再ビルドし、設定されたプレフィックスにインストールして、第 6 節用に `pydxs` を含む `venv-dx_stream` を作成します。インストール手順と root 所有のビルドディレクトリのクリーンアップには `sudo` を使うため、チュートリアル 01 と同様に、セルは `sudo` がパスワードなしで動作する場合にのみここでビルドを実行し、そうでない場合はターミナル用のコマンドを表示します。フルリビルドには数分かかります。既存の `builddir` が別の meson バージョンで生成されていたためにインクリメンタルビルドが失敗した場合、セルは `./build.sh --clean` にフォールバックし、最初からビルドし直します。

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_stream
./build.sh
```

<!-- cell: 5d801612-1e0a-44fa-8345-74636b91a421 src: 9808c3e4d1 -->
### 5.8 推論設定の作成

`dxinfer` のプロパティは、パイプライン内で直接指定することも、JSON ファイルで指定することもできます。

| JSON フィールド | 意味 |
|---|---|
| `preprocess_id` | 同じ ID を持つ `dxpreprocess` エレメントが作成した入力テンソルを選択します |
| `inference_id` | モデル出力テンソルにラベルを付けます。`dxpostprocess` は同じ ID を使う必要があります |
| `model_path` | コンパイル済み `.dxnn` モデルへのパス。相対パスはプロセスの作業ディレクトリから解決されるため、セルは絶対パスを書き込みます |
| `backend` | `auto`、`dxrt`、またはコンパイル済みの任意バックエンドを選択します |

複数の前処理、推論、後処理エレメントを含むパイプラインでは、明示的な ID を使うことが重要です。

<!-- cell: 1341a38d-276d-4ac6-85e3-c454925cb1a9 src: ff1c547c4c -->
### 5.9 テスト動画の取得

チュートリアル 03 で使用したのと同じクリップです。そのワークスペースに存在すればコピーし、なければダウンロード(19 MB)します。

<!-- cell: ccf95200-0466-4b6c-91fb-5dcecd0bc6cf src: f4aec25b4e -->
### 5.10 カスタムパイプラインの実行

ID は次の関係を形成します。

```text
dxpreprocess(preprocess-id=1)
        ↓
dxinfer(preprocess_id=1, inference_id=1)
        ↓
dxpostprocess(inference-id=1)
```

`library-file-path` は 5.7 でインストールしたライブラリを指します。以下のパスは既定のインストールプレフィックスです。DX-STREAM を `./build.sh --prefix=<dir>` でビルドした場合は `<dir>/share/gstdxstream/lib/libpostprocess_yolov7.so` を使用してください。出力は `SINK` に送られます。デスクトップセッションではウィンドウ、それ以外では `fakesink` です。

<!-- cell: 83d74722 src: 77c4166080 -->
## 6. YOLO26s 人数カウントアプリケーションの構築

この演習では、各統合ポイントを確認しやすくするために、まず Python で人数カウントを実装します。YOLO26s の物体検出を実行し、各映像フレームにアタッチされた検出メタデータを読み取り、COCO クラス `0`(`person`)を数えてテキストオーバーレイを更新します。

この節を終えると、次のことが理解できます。

- `DXObjectMeta` がどこで定義され、作成されるか
- 検出メタデータがどのように `GstBuffer` と共に流れるか
- `pydxs` がどのように C++ のメタデータを Python に公開するか
- Python の文字列がどのように実行中の GStreamer パイプラインになるか
- パッドプローブのコールバックと表示の更新が、なぜ異なる実行コンテキストを使うのか

<!-- cell: 08a854bd src: 2272569272 -->
### 6.1 YOLO26s モデルと動画の準備

`yolo26-s_640x640.dxnn` はチュートリアル 01 の 3.3 節で共有ワークスペースにダウンロードしたモデルの 1 つなので、ここではそれを再利用します。セルは、モデルが見つからない場合にのみ DX-APP のセットアップスクリプトでダウンロードします。DX-STREAM サンプルのスノーボード動画は、複数の人物が映っているため選ばれています。

<!-- cell: 254abcd0 src: 269af0cb87 -->
### 6.2 検出メタデータを理解する

DX-STREAM は、検出データを画像そのものに書き込むことなく、ピクセルと推論結果を一緒に保持します。デコードされた 1 つの映像フレームは 1 つの `GstBuffer` で運ばれ、DX-STREAM はそのバッファに 1 つの `DXFrameMeta` をアタッチし、フレームメタデータが検出オブジェクトのリストを所有します。

```text
GstBuffer — one video frame moving through the pipeline
│
├── Video memory / pixels
│
└── DXFrameMeta — frame-level GstMeta
    ├── stream_id, width, height, format, frame_rate, ROI, ...
    │
    └── object_meta_list
        ├── DXObjectMeta #0
        │   ├── label          = 0
        │   ├── label_name     = "person"
        │   ├── confidence     = 0.92
        │   └── box            = [x1, y1, x2, y2]
        ├── DXObjectMeta #1
        └── ...
```

この関係は重要です。Python のコールバックは推論を再実行せず、画像を解析もしません。`dxpostprocess` がすでに生成したオブジェクトメタデータを読み取るだけです。

<!-- cell: d59665ef src: cd784d857f -->
#### 6.2.1 `DXObjectMeta` が定義され、作成される場所

SDK は、`DX_STREAM_DIR` からの相対パスで次のヘッダーにこれらの構造体を定義しています。

| 用途 | SDK ソース |
|---|---|
| フレームメタデータとそのオブジェクトリスト | `gst-dxstream-plugin/metadata/gst-dxframemeta.hpp` |
| オブジェクトごとの検出メタデータ | `gst-dxstream-plugin/metadata/gst-dxobjectmeta.hpp` |
| 両構造体の Python バインディング | `bindings/python/pydxs/src/metadata_binding.cpp` |

プライマリの物体検出では、`dxpostprocess` が設定された C++ の `PostProcess` 関数を呼び出します。この関数は次のライフサイクルで YOLO のテンソル出力をオブジェクトに変換します。

```cpp
DXObjectMeta* object = dx_acquire_obj_meta_from_pool();
object->_label       = class_id;
object->_label_name  = class_name;
object->_confidence  = score;
object->_box         = {x1, y1, x2, y2};
dx_add_obj_meta_to_frame(frame_meta, object);
```

結果は現在のフレームにアタッチされるため、下流のエレメントは同じ検出結果を参照します。

```text
dxinfer                 dxpostprocess                         downstream
tensor output ────────▶ create DXObjectMeta ────────┬──────▶ dxosd draws boxes
                                                    └──────▶ Python probe counts people
```

DX-STREAM はメタデータの寿命をバッファと共に管理します。あるエレメントが新しいバッファを作成して `DXFrameMeta` をコピーすると、そのオブジェクトメタデータもコピーされます。それでもアプリケーションコードは、Python のメタデータ参照を現在のバッファコールバックの処理中にのみ有効なものとして扱い、後で使うために保存しないでください。

<!-- cell: b9bcdd40 src: d486fcc937 -->
#### 6.2.2 C++ のメタデータが Python でどう見えるか

`pydxs` は C++ のメンバーを Python で扱いやすいプロパティにマッピングします。

| C++ フィールド | Python プロパティ | この演習での意味 |
|---|---|---|
| `DXFrameMeta::_object_meta_list` | `frame_meta.object_meta_list` または `frame_meta` の反復処理 | 現在のフレーム内のすべての検出 |
| `DXObjectMeta::_label` | `obj_meta.label` | COCO クラス ID。`0` は `person` を意味します |
| `DXObjectMeta::_label_name` | `obj_meta.label_name` | 人間が読めるクラス名 |
| `DXObjectMeta::_confidence` | `obj_meta.confidence` | 検出の信頼度 |
| `DXObjectMeta::_box` | `obj_meta.box` | フレーム座標での `[left, top, right, bottom]` |
| `DXObjectMeta::_track_id` | `obj_meta.track_id` | トラッカー ID。このパイプラインでは割り当てられません |

コールバックは、`dxpostprocess` から出ようとしているのと同じ `Gst.Buffer` を受け取ります。

```python
buffer = info.get_buffer()
frame_meta = pydxs.dx_get_frame_meta(hash(buffer))

people_count = sum(
    1 for obj_meta in frame_meta
    if obj_meta.label == PERSON_CLASS_ID
)
```

`hash(buffer)` は、バインディングが期待するネイティブの `GstBuffer` アドレスを渡します。`pydxs` が `_object_meta_list` を Python のイテレータープロトコルで公開しているため、`frame_meta` は反復処理できます。

したがってこのカウントは、**現在のフレーム**で受理された `person` 検出ボックスの数です。Python コードは追加の信頼度しきい値を適用しません。信頼度によるフィルタリングや重複の抑制を行う場合は、オブジェクトがアタッチされる前に後処理側で行う必要があります。これはユニークな来訪者数ではありません。時間をまたいで同一性を維持するには追跡が必要です。

<!-- cell: b962bea0 src: 2f6d8fbc24 -->
### 6.3 Python が GStreamer パイプラインを埋め込む仕組みを理解する

このアプリケーションは `gst-launch-1.0` をサブプロセスとして起動しません。代わりに、`pipeline_description` に GStreamer の launch 構文と同じ内容を Python のフォーマット済み文字列として保持し、`Gst.parse_launch()` がそのテキストを、パッドで接続された実際の GStreamer エレメントに変換します。

```text
Python application
│
├── pipeline_description = f''' ... '''
│       │
│       └── Gst.parse_launch(description)
│                    │
│                    ▼
│    ┌────────┐  ┌──────────┐  ┌───────┐  ┌─────────────┐  ┌───────┐
└───▶│ source │─▶│preprocess│─▶│ infer │─▶│ postprocess │─▶│  osd  │─▶ display
     └────────┘  └──────────┘  └───────┘  └──────┬──────┘  └───────┘
                                                  │
                                    source-pad BUFFER probe
                                                  │
                                                  ▼
                                    read metadata → count people
                                                  │
                                                  ▼
                                      update `textoverlay` text
```

プローブはバッファを観察するだけです。別のパイプラインブランチではなく、フレームをコピーすることもありません。

| パイプラインの構成要素 | 役割 |
|---|---|
| `urisourcebin ! decodebin` | 動画を読み込み、圧縮フレームをデコードします |
| `dxpreprocess` | フレームをリサイズしてモデル用に準備します |
| `dxinfer` | YOLO26s を実行し、その出力テンソルをアタッチします |
| `dxpostprocess name=detector_postprocess` | テンソルをデコードし、`DXObjectMeta` エントリをアタッチします |
| `dxosd` | オブジェクトメタデータを読み取り、ボックスとラベルを描画します |
| `videoconvert` | 標準のオーバーレイ/表示パス向けに映像フォーマットを準備します |
| `textoverlay name=people_overlay` | Python が制御するカウントを表示します |
| `fpsdisplaysink` | フレームを表示し、表示 FPS を測定します |

<!-- cell: 98bc52b7 src: aaf166a292 -->
#### 6.3.1 パイプライン文字列から名前付き Python オブジェクトへ

次の 3 行がパイプラインのテキストとアプリケーションロジックを結び付けます。

```python
self.pipeline = Gst.parse_launch(pipeline_description)
self.people_overlay = self.pipeline.get_by_name("people_overlay")
postprocess = self.pipeline.get_by_name("detector_postprocess")
```

名前は launch 文字列内のプロパティに由来します。

```text
dxpostprocess name=detector_postprocess ...
textoverlay   name=people_overlay ...
```

その後、アプリケーションは後処理エレメントのソースパッドにコールバックをアタッチします。

```python
src_pad = postprocess.get_static_pad("src")
src_pad.add_probe(Gst.PadProbeType.BUFFER, self._postprocess_probe)
```

`dxpostprocess` から出力されるすべてのバッファは、`dxosd` に進む前に `_postprocess_probe` を呼び出します。`Gst.PadProbeReturn.OK` を返すと、同じバッファがそのまま下流に進みます。

<!-- cell: af6d6eca src: 427e4a7436 -->
#### 6.3.2 ストリーミングコールバックと GLib メインループ

GStreamer はストリーミングスレッドからパッドプローブを呼び出します。コールバックが遅いと後続のすべてのフレームが遅延するため、プローブはメタデータの取得、オブジェクトのカウント、表示更新のスケジューリングだけを行います。

```text
GStreamer streaming thread                    GLib main-loop thread
──────────────────────────                    ─────────────────────
buffer reaches postprocess.src
          │
          ▼
_postprocess_probe()
  ├── get GstBuffer
  ├── get DXFrameMeta
  ├── count label == 0
  └── GLib.idle_add(_update_overlays, count) ───────────────┐
          │                                                 ▼
          └── return OK                           _update_overlays()
                    │                               └── set text property
                    ▼
             buffer continues                     UI remains responsive
```

バスはパイプライン全体のイベントもメインループに報告します。

```text
GStreamer bus ── ERROR ──▶ print details and stop
              └─ EOS   ──▶ stop the main loop
```

`pipeline.set_state(Gst.State.PLAYING)` がデータフローを開始し、`GLib.MainLoop().run()` が Python アプリケーションを存続させて、アイドルコールバックとバスメッセージを処理できるようにします。

<!-- cell: 6ab93ab6 src: bebe7b0bf7 -->
### 6.4 Python アプリケーションの作成

以下の完全なアプリケーションは、上で説明したパイプライン、メタデータプローブ、オーバーレイ更新、バス処理、コマンドライン引数を組み合わせたものです。

<!-- cell: 13351f19 src: 0781fa93bf -->
### 6.5 DX-STREAM Python 環境の確認

アプリケーションは DX-STREAM の仮想環境を使う必要があります。`pydxs` モジュールと GStreamer の Python バインディングを提供しているのがこの環境だからです。

<!-- cell: e6d0e88f src: 89e064ba44 -->
### 6.6 アプリケーションの実行

デスクトップセッション(`HEADLESS = False`)では、アプリケーションは YOLO のボックスと `People: N` オーバーレイを表示するウィンドウを開き、動画が終了するか停止ボタンが押されるまで実行中のままになります。ディスプレイがない場合(`HEADLESS = True`)は `--no-display` で実行されます。パイプライン、メタデータプローブ、カウントはそのまま動作し、フレームは `fakesink` に送られ、セルは EOS で要約行を表示して終了します。

`pydxs` と GStreamer の Python バインディングは DX-STREAM 環境にあるため、アプリケーションは Jupyter カーネルではなく `venv-dx_stream/bin/python` を使う必要があります。ターミナルでは次のように実行します。

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_stream
./venv-dx_stream/bin/python <workspace>/yolo26_people_counter.py \
    --model <DX_ALL_SUITE_DIR>/workspace/res/models/yolo26-s_640x640.dxnn \
    --video dx_stream/samples/videos/snowboard.mp4
```

<!-- cell: 5c93c78a src: 946e670e83 -->
### 6.7 期待される結果と拡張ポイント

通常の YOLO のボックスとラベルは `dxosd` によって引き続き表示されます。標準の GStreamer テキストオーバーレイがアプリケーションの状態を追加します。

- 右上: `People: N`。カウントが変わるたびに更新されます。

表示される数は、そのフレームで受理された `person` ボックスの数と一致するはずです。1 人の人物が重なり合うボックスを生成する場合は、Python のオーバーレイで数を補正するのではなく、後処理側の信頼度/NMS ポリシーを変更してください。

本番の在室者数システムでは、`dxtracker` を追加し、入退場領域を定義して、フレームごとの raw 検出ではなく安定したトラック ID を数えてください。

<!-- cell: e899a740-4b78-4712-8073-a1904d5b76b2 src: 0a5e5189de -->
## 7. デバッグ

### 7.1 対象を絞った GStreamer ログを使う

初期化と状態遷移の確認にはレベル 3 から始め、調査対象のエレメントに対してのみレベル 4 または 5 を有効にします。

```bash
# All DX-STREAM elements at INFO level
GST_DEBUG=dx*:3 ./run_demo.sh

# Inference and tracker details
GST_DEBUG=dxinfer:4,dxtracker:4 ./run_demo.sh

# Metadata lifecycle
GST_DEBUG=dxmeta:5 ./run_demo.sh

# Save logs to a file
GST_DEBUG=2,dxinfer:4 GST_DEBUG_FILE=/tmp/dxstream.log ./run_demo.sh
```

ログ出力はタイミングに影響するため、性能測定中は詳細ログを無効にしてください。

<!-- cell: 13445965-9c30-429f-80ff-3acf97fd6cee src: 771d6ca90f -->
### 7.2 v3.1.x の失敗時の挙動を理解する

- 互換性のないリンクは caps ネゴシエーションの段階で失敗します。
- モデルファイルの欠落、NPU 初期化の失敗、カスタムライブラリのエラーは、`abort()` で終了する代わりに GStreamer エレメントのエラーとして報告されます。
- 処理エレメントは、測定した処理時間を LATENCY クエリに反映します。
- FLUSH 時に内部状態がリセットされ、シークと再生のやり直しの挙動が改善されています。
- EOS、FLUSH、状態遷移の際に、単一フレーム入力を含めてワーカースレッドがより確実にクリーンアップされます。

マルチストリームパイプラインがリンクできない場合は、まず標準の `video/x-raw` エレメントが `application/x-dxvideoraw` ドメインの内側に置かれていないかを確認してください。

<!-- cell: a340afc0-dfc5-471d-98aa-6c7e8e2e0f53 src: af81fae94c -->
### 7.3 パイプライングラフのエクスポート

`GST_DEBUG_DUMP_DOT_DIR` が設定されていると、GStreamer はパイプラインの状態遷移のたびに DOT ファイルを書き出します。Graphviz は `PAUSED_PLAYING` ファイルを実際のエレメントグラフの画像に変換します。必要であれば一度だけインストールしてください。

```bash
sudo apt install graphviz
```

次のセルはカーネル用にこの変数を設定し、3.1 のクイックスタートパイプラインを(設定に応じてヘッドレスまたはウィンドウ付きで)再度実行するため、DOT ファイルはチュートリアルのワークスペースに出力されます。同等のターミナルコマンドは次のとおりです。

```bash
GST_DEBUG_DUMP_DOT_DIR=<workspace>/dot gst-launch-1.0 filesrc location=... ! ... ! fakesink
```

<!-- cell: af463f4c-5ea2-4477-b67c-fcc4cad871b7 src: f4e26083b6 -->
### 7.4 DX-STREAM v3.1.x での変更点

このチュートリアルは DX-STREAM v3.1.2 で検証されました。v3.1.x 系列では、このチュートリアルが前提とする次のランタイム動作が導入されました。

- `application/x-dxvideoraw` に基づく統一されたマルチストリームドメイン
- 共有処理チェーンにおけるストリームごとのライフサイクルとタイムラインの保持
- `dxinfer` バックエンドの抽象化と非同期 Put/Get 処理
- Base64 エンコードされた JPEG フレーム用の `dxmsgconv include-frame`
- カスタムライブラリとメタデータ API の更新(5.1 節)
- レイテンシレポート、FLUSH からの復帰、caps ネゴシエーション、エラー報告の改善

利用可能なバックエンドとエレメントのプロパティは、DX-STREAM のビルド方法によって異なります。インストールされているビルドは `gst-inspect-1.0` で確認してください。

<!-- cell: 325d5d11-7b12-4ddb-83e2-f1cb7ce17a91 src: 9ef3c41d4e -->
## 8. トラブルシューティング

| 症状 | 表示される内容 | 考えられる原因 | 対処 |
|---|---|---|---|
| DX-STREAM プラグインが見つからない | `DX-STREAM plugin not found` または `no element "dxinfer"` | JupyterLab を起動した環境で `GST_PLUGIN_PATH` が設定されていない | `./run-jupyter-lab.sh` の前に `GST_PLUGIN_PATH=/usr/local/lib/x86_64-linux-gnu/gstreamer-1.0`(`build.sh` が表示したパス)をエクスポートするか、`./build.sh` で再ビルドする |
| ウィンドウが表示されない | `Could not open display`、またはパイプラインが何も出力しない | グラフィカルセッションがない | デスクトップセッションで実行する。簡易確認には `fpsdisplaysink` を `fakesink` に置き換える |
| カメラパイプラインが失敗する | `v4l2src: Cannot open device` または caps ネゴシエーションエラー | デバイスパスが誤っているか、解像度がサポートされていない | `v4l2-ctl --list-formats-ext` を実行し、セル内のデバイスと caps を編集する |
| RTSP デモが接続できない | `Could not connect to server` | `run_RTSP.sh` のサンプル RTSP URL にお使いのネットワークから到達できない | `run_RTSP.sh` を到達可能なストリームに編集する |
| `build.sh` が止まる | セル内で `sudo` のパスワードプロンプトまたは `meson` エラー | ビルドはシステム全体にインストールするため `sudo` が必要 | ターミナル(File > New > Terminal)で `./build.sh` を実行する |
| 一部のフレームが全面緑色になる | ウィンドウまたは保存されたフレームが断続的に緑色に点滅する | 表示シンクの前の `dxconvert` が一部のデコード済みバッファを正しく処理できない | `videoconvert ! fpsdisplaysink`(ノートブックの既定)を使う。`dxconvert` は 3.3 の `dxscale`/`dxconvert` の実験にのみ使う |
| DOT グラフが生成されない | `No DOT file was generated` | パイプラインが PLAYING に達する前にデモが停止された | 動画の再生が始まってからセルを停止する |
| サンプルモデルが読み込み時に拒否される | `dxinfer` からの DX-RT バージョンまたはフォーマットのエラー | サンプルモデルは Model Zoo リリース `2_4_0` 向けにコンパイルされており、古いランタイムでは読み込めない | このチュートリアルの検証に使用した DX-RT バージョン(3.4.2)を使うか、お使いの DX-COM でモデルを再コンパイルする |
| `setup.sh` がパスワードを要求する | `jq` のインストール中に `[sudo] password for ...` | `jq` がなく、スクリプトが `apt` でインストールしようとしている | ターミナルで一度 `sudo apt install jq` を実行してからセルを再実行する |
| `import pydxs` が失敗する | `ModuleNotFoundError: No module named 'pydxs'` | DX-STREAM がまだビルドされていないか、`venv-dx_stream` ではなく Jupyter の Python が使われた | 5.7(ビルド)を実行し、セルと同様に `venv-dx_stream/bin/python` を使う |
| `build.sh` がすぐに失敗する | `Build data file ... was generated with an old version of meson` | 以前の DX-STREAM バージョンが残した `builddir` | `./build.sh --clean` を実行する(5.7 のセルは最初の試行が失敗すると自動的にこれを行う) |

<!-- cell: 51fafd7a-3597-4c54-a355-0cad94216211 src: de9bc8a30a -->
## 9. まとめ

ここまでで次のことを行いました。

- エンドツーエンドの DX-STREAM 推論パイプラインを構築し、確認した
- 同梱の物体検出、PPU、顔検出、姿勢推定、セグメンテーション、追跡、マルチチャネル、RTSP、Secondary Mode の各デモを実行し、その背後にあるスクリプトを読んだ
- 推論バックエンド、`dxscale`、`dxconvert`、MQTT/Kafka メッセージブローカーのスクリプトを確認した(ブローカーのデモは稼働中のブローカーが必要で、メニューには含まれない)
- `application/x-dxvideoraw` マルチストリームドメインとそのエレメント配置ルールを学んだ
- 2 クラスの Forklift and Worker モデル向けに YOLOv7 の後処理と推論設定を適応させた
- 検出結果を `GstBuffer` から `DXFrameMeta`、`DXObjectMeta` を経て `pydxs` 経由で Python まで追跡した
- GStreamer パイプラインを Python に埋め込み、パッドプローブと GLib メインループを使ってフレームごとの人数カウントを表示した
- 対象を絞った GStreamer ログと DOT グラフを使ってパイプラインの挙動を診断した

### 9.1 完了チェックリスト

- [ ] `gst-inspect-1.0` で 13 個の DX-STREAM エレメントを確認した
- [ ] クイックスタートの YOLO26n パイプラインと、同梱デモを 3 つ以上実行した
- [ ] マルチストリームドメインの配置ルールを説明できる
- [ ] 2 クラスの Forklift and Worker 後処理にパッチを当て、再ビルドして実行した
- [ ] Python の人数カウントアプリケーションを実行し、`pydxs` 経由で `DXObjectMeta` を読み取った
- [ ] パイプラインの DOT グラフをエクスポートして開いた

> **次へ:** チュートリアル 05 に進み、DX-Compiler のワークフロー(ONNX の検証、キャリブレーション、PPU、グラフ最適化、量子化戦略)を詳しく学びましょう。
