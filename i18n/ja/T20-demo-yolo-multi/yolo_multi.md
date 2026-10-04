<!-- i18n source: notebooks/T20-demo-yolo-multi/yolo_multi.ipynb -->
<!-- i18n lang: ja -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 7bc75515 src: 7038880fdd -->
# DEEPX Tutorial 20 - YOLO マルチチャネル C++ デモ

このチュートリアルでは、C++ アプリケーションが DEEPX NPU を使って複数の動画チャネルで YOLO 物体検出を実行する仕組みを説明します。また、リソースのダウンロード、アプリケーションのビルド、動画デモとカメラデモの実行方法も示します。

![YOLO マルチチャネルデモ](assets/yolo-multi-sc.png)

<!-- cell: 8f86f5a0 src: 6807896955 -->
## 学習目標

このチュートリアルを終えると、次のことができるようになります。

- C++ プロジェクトの構造を理解する
- JSON ファイルのモデル、入力、表示の設定を読み取る
- 非同期マルチチャネル推論のフローを追う
- Release モードでアプリケーションをビルドする
- 動画とカメラのサンプルを実行する

<!-- cell: t20-npu-pattern src: 80e34194de -->
## このアプリケーションの NPU の使い方

| 項目 | このチュートリアルでは | 学んだ場所 |
|---|---|---|
| エンジン | 36 個のチャネルスレッドすべてが共有する **1 つ**の `InferenceEngine`。DX-RT がそれらのリクエストをキューに入れ、NPU 上でスケジューリングします | T06-3 §2 |
| 実行 | 各チャネルは `RunAsync()` を呼び出し、すでに次のフレームを準備している間にコールバックでボックスを受け取ります | T06-2 §5 |
| タスクグラフ | `YOLOV5S_PPU` は単一の NPU タスクです。PPU がデコード済みのボックスを返すため、ホストは NMS だけを実行します。36 チャネルでは、ここで PPU の効果が発揮されます(T05-2 §4 と比較) | T05-2 §3 |
| リクエストあたりの入力 | `[1, 512, 512, 3]` UINT8 = 786 KB。30 FPS の 36 チャネルでは H2D トラフィックが 850 MB/s になるため、表示タイルは小さくなっています | T06-3 §4 |
| 測定対象 | 3.2 節の `dxrun` ベースライン(単一ストリーム、デコードや表示なし)と、ウィンドウヘッダーに表示される FPS の比較 | T06-1 §5 |

<!-- cell: t20-prerequisites src: c5253820d0 -->
## 前提条件

- チュートリアル 01 が完了していること: DX-RT がインストール済みで、DEEPX NPU が `/dev/dxrt0` として認識されていること。`cmake` と `g++`(`build-essential`)。
- OpenCV ウィンドウ用のグラフィカルなデスクトップセッション。カメラデモ用の `/dev/video0` にある V4L2 カメラ(任意)。
- ダウンロード: 約 134 MB のアーカイブ 1 つ(YOLOv5s PPU モデルと 45 本のサンプル動画)を `assets/` に展開します。このディレクトリは git で無視されます。
- 所要時間: 約 20 分。ビルドには 1〜2 分かかります。パッケージが不足している場合の第 2 節の `apt-get` 行を除き、`sudo` は不要です。
- DX-RT 3.4.2 で検証済みです。モデルは DX-COM 2.2.0 でコンパイルされており、このランタイムで読み込めます。

次のセルは SDK の場所を特定し、これらの要件を確認してステータス表を表示します。`MISSING` と表示された項目には、それを提供する手順が添えられます。

<!-- cell: 8ea9a7fc src: a4b88669a3 -->
## 1. チュートリアルファイルの場所を特定する

次のセルは、JupyterLab がリポジトリのルートから起動された場合でも、このノートブックのディレクトリから起動された場合でも、チュートリアルディレクトリを見つけます。

<!-- cell: 77ab9dbf src: 9a378a9fc7 -->
### 1.1 プロジェクト構成

```text
T20-demo-yolo-multi/
├── get_resources.sh
├── assets/
│   ├── models/
│   └── videos/
├── app/
│   ├── build.sh
│   ├── run_camera.sh
│   ├── run_video.sh
│   ├── config/
│   ├── include/
│   ├── src/
│   ├── lib/
│   ├── extern/
│   └── sample/
└── yolo_multi.ipynb
```

`extern/` にはヘッダーオンリーの cxxopts と RapidJSON の依存関係が含まれています。`sample/` には表示 UI で使用するフォントと画像が含まれています。

<!-- cell: 4173a889 src: 76d6941463 -->
## 2. 環境の確認

Debian または Ubuntu のパッケージをインストールします。

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config ca-certificates curl tar \
    libopencv-dev libopencv-contrib-dev libfreetype-dev ffmpeg v4l-utils \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
```

リソースアーカイブは約 140 MB です(PPU モデル 1 つと 45 本のサンプル動画)。

<!-- cell: e29c2c86 src: 4446f1b209 -->
## 3. リソースのダウンロードと確認

どちらのデモも `assets/models/YOLOV5S_PPU.dxnn` と `assets/videos/` 配下のサンプル MP4 ファイルを使用します。`get_resources.sh` はアーカイブを 1 つダウンロードして `assets/` に展開し、展開が成功した後にアーカイブを削除します。

<!-- cell: d87f310e src: 670c65bcaf -->
次のセルは、リソースが不足している場合にのみ実行してください。同じ名前の既存ファイルは展開時に置き換えられることがあります。

<!-- cell: 49a25e54 src: 2ba54cf8ba -->
### 3.1 `dxparse` でモデルを調べる

`dxparse` を使って、モデルの構造、タスクグラフ、テンソル、メモリ使用量、依存関係を調べます。

```bash
dxparse -m assets/models/YOLOV5S_PPU.dxnn -v
```

詳細出力ではモデルの入力が次のように表示されます。

```text
  Inputs
     -  images, UINT8, [1, 512, 512, 3 ]
```

次元はバッチ、高さ、幅、チャネルです。したがって、このモデルは 640 x 640 ではなく 512 x 512 の 3 チャネル入力を期待しています。

<!-- cell: 16d5dc87 src: 339ac4ce39 -->
### 3.2 `dxrun` でモデルをベンチマークする

自動生成されたダミー入力を使って 5 秒間の CLI ベンチマークを実行します。

```bash
dxrun -m assets/models/YOLOV5S_PPU.dxnn --use-ort -t 5
```

`dxrun` は `--single` も `--fps` も指定されていない場合、既定でベンチマークモードを使用します。`-t 5` は測定時間を 5 秒に設定し、`--use-ort` はモデルグラフ内の CPU タスクに対して ONNX Runtime を有効にします。報告される結果は、マルチチャネルの動画デコードや表示のオーバーヘッドを含まない、コマンドラインでの性能ベースラインとなります。

<!-- cell: edb6cae2 src: 47f55de289 -->
## 4. デモ設定を理解する

アプリケーションはすべてのランタイム設定を JSON ファイルから読み取ります。重要なフィールドは次のとおりです。

- `model_path`: `app/` からの相対パスで指定する DXNN モデルのパス
- `model_name`: 対応する YOLO 後処理パラメーターを選択する
- `video_sources`: 入力パス、入力タイプ、および任意の保存フレーム数
- `display_config`: 出力サイズ、グリッド、FPS、レイアウトの設定
- `num_devices`: ヘッダーに表示する NPU デバイスの数

<!-- cell: 3313e802 src: 0facc52619 -->
カメラ設定には `/dev/video0` の入力 1 つと動画入力 32 個が含まれています。カメラの強調表示は `expand_mode` とは独立しているため、カメラは拡大された中央領域に黄色の枠付きで配置されます。

動画設定には 6 x 6 グリッドの動画入力 36 個が含まれ、`expand_mode` は `false` に設定されています。アプリケーションの拡張レイアウト(中央のタイル 1 つを拡大)はソース数が 33、41、61、73 の場合にのみ存在します。そのため、33 ソースのカメラ設定ではこれが使われ、36 ソースの動画設定では使われません。

<!-- cell: 44889672 src: 4575c7a014 -->
## 5. C++ コードを読む

このノートブックは、現在のソースファイルから選択した部分を直接表示します。これにより、ノートブック内に C++ コードの 2 つ目のコピーを保持する必要がなくなります。

<!-- cell: 940a9b9a src: c69dca4723 -->
### 5.1 ビルド設定

CMake は C++14 の実行ファイルを 1 つビルドし、DXRT、OpenCV、pthread、OpenMP、および C++14 で必要なファイルシステムライブラリをリンクします。OpenCV の FreeType サポートは利用可能な場合に使用されます。`cmake/dxdemo.function.cmake` の `add_dxrt_lib()` ヘルパーは、チュートリアル 06-2 や他のデモが使うのと同じ `find_package(dxrt)` の検索をラップし、インポートされたターゲット `dxrt::dxrt` をリンクします。その追加の分岐はクロスコンパイルと Windows ビルドのためだけのものです。

<!-- cell: cfc22a04 src: f378c3defe -->
### 5.2 設定の解析

`ApplicationJsonParser()` は JSON フィールドを検証し、`AppConfig` を埋めます。パーサーは任意の表示設定に対する既定値も提供します。

<!-- cell: b299367c src: 4f1996b60f -->
### 5.3 マルチチャネル推論パイプライン

```text
JSON configuration
        |
        v
DXRT InferenceEngine (shared)
        |
        +-- ObjectDetection: channel 1 --+
        +-- ObjectDetection: channel 2 --+--> output grid --> OpenCV window
        +-- ObjectDetection: channel N --+
```

`main()` は DXRT の `InferenceEngine` を 1 つと、入力ソースごとに `ObjectDetection` オブジェクトを 1 つ作成します。各チャネルは独自のワーカースレッドで動作します。

<!-- cell: b25e522d src: 0cf1ca9c29 -->
### 5.4 チャネルごとの非同期推論

`ObjectDetection::threadFunc()` は前処理済みの入力フレームを取得して `RunAsync()` を呼び出します。コールバックは最新のバウンディングボックスを更新し、その間にチャネルスレッドは表示フレームを準備します。ワーカーとコールバックの間で共有されるデータはミューテックスで保護されます。

<!-- cell: 966230b7 src: 614c770d7d -->
### 5.5 YOLO 後処理

`Yolo::PostProc()` はモデルの出力タイプに応じたデコーダーを選択します。これらのデモは PPU 出力パスを使用します。デコードされた候補はクラスごとにソートされ、重なり合うボックスを除去するために非最大抑制(NMS)に渡されます。

<!-- cell: bf10922b src: 576016cf2b -->
## 6. アプリケーションのビルド

`build.sh` は `app/build/` を作成し、Release モードで CMake を構成して、`nproc` が報告するすべての CPU コアで `make` を実行します。完全な再ビルドが必要な場合は `./build.sh --clean` を使用し、DX-RT がカスタムプレフィックスにインストールされている場合は、たとえば `./build.sh -DCMAKE_PREFIX_PATH=<prefix>` のように CMake オプションをそのまま渡します。

<!-- cell: a3cc6532 src: 1521315f86 -->
## 7. 36 チャネル動画デモの実行

次のセルは OpenCV ウィンドウを開き、デモが終了するまで待機します。`Esc` または `q` を押すか、`EXIT` ボタンをクリックしてください。

<!-- cell: 34f09e43 src: 1e56cf0884 -->
## 8. 33 チャネルカメラデモの実行

既定の設定では `/dev/video0` に V4L2 カメラがあることを想定しています。アプリケーションはサポートされているカメラモードを自動的に選択し、カメラのレートを最大 30 FPS に制限します。

<!-- cell: cec0fe9e src: 0cee838170 -->
## 9. 操作方法

- `Esc` または `q`: 終了
- `t`: 検出ボックスの表示/非表示
- `EXIT` ボタン: マウスで終了

<!-- cell: 0c8262f3-3b63-458d-a93a-f9bb5f21dc27 src: c73952716d -->
## 10. トラブルシューティング

| 症状 | 表示される内容 | 考えられる原因 | 対処 |
|---|---|---|---|
| CMake が DXRT を見つけられない | `Could not find a package configuration file provided by "dxrt"` | DX-RT がインストールされていないか、カスタムプレフィックスにある | DX-Runtime をインストールする(チュートリアル 01 第 3 節)か、`./build.sh -DCMAKE_PREFIX_PATH=<prefix>` を実行する |
| モデルまたは動画を開けない | アプリケーションログに `[ER] ... cannot open` | リソースが不足している | 第 3 節のリソースセルを実行する |
| モデルとランタイムのバージョンが一致しない | `The version of the compiled model is not compatible with the version of the runtime` | DXNN が別の SDK リリース向けにコンパイルされている(このモデル: DX-COM 2.2.0) | このチュートリアルで検証した DX-RT バージョン(3.4.2)を使用するか、お使いの DX-COM でモデルを再コンパイルする |
| カメラを開けない | `cannot open /dev/video0` | カメラがないか、ユーザーが `video` グループに属していない | `v4l2-ctl --list-devices` を実行し、`sudo usermod -aG video $USER` の後に再ログインする。別のデバイスを使う場合はカメラ用 JSON を編集する |
| ウィンドウが表示されない | `cannot open display` | グラフィカルセッションがない | デスクトップセッションで実行する |

<!-- cell: 1da13663 src: 0ee7a9d3d1 -->
## 11. まとめ

アプリケーションは 1 つの DXRT 推論エンジンを複数のチャネルワーカーで共有します。各ワーカーは非同期推論、YOLO PPU 後処理、フレーム描画を実行します。メインループはチャネルのフレームを設定可能な 1 つのグリッドにまとめ、ランタイム情報を OpenCV ウィンドウに表示します。

### 11.1 完了チェックリスト

- [ ] `get_resources.sh` でモデルと動画をダウンロードする
- [ ] `dxparse` で PPU モデルを調べ、`dxrun` でベンチマークする
- [ ] チャネルワーカーが 1 つの `InferenceEngine` を共有する仕組みを読む
- [ ] アプリケーションをビルドして動画デモを実行する
- [ ] JSON 設定でグリッドまたは入力リストを変更して再実行する

> **次へ:** チュートリアル 21 に進み、1 つの Qt アプリケーションで物体検出、姿勢推定、セグメンテーション、深度推定をまとめて実行してみましょう。
