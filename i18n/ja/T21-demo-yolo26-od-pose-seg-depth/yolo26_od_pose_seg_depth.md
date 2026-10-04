<!-- i18n source: notebooks/T21-demo-yolo26-od-pose-seg-depth/yolo26_od_pose_seg_depth.ipynb -->
<!-- i18n lang: ja -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: fa62a5df src: a9203579d1 -->
# DEEPX Tutorial 21 - YOLO26 物体検出・姿勢推定・セグメンテーション・深度推定デモ

このチュートリアルでは、同じカメラまたは動画ストリームに対して、以下の 4 つの YOLO26s モデルを DEEPX NPU で実行する Qt ベースの C++ アプリケーションを解説します。
- YOLO26s 物体検出
- YOLO26s 姿勢推定
- YOLO26s インスタンスセグメンテーション
- YOLO26s 単眼深度推定

![YOLO マルチチャネルデモ](assets/yolo26-od-pos-seg-depth.png)

<!-- cell: 7b3ff195 src: d25198ce78 -->
## 学習目標

このチュートリアルを終えると、次のことができるようになります。

- 自己完結型の C++ プロジェクト構造を理解する
- 検出、姿勢推定、セグメンテーション、深度推定の各パイプラインを識別する
- 1 つの入力フレームが 4 つの非同期ワーカーにどのように配布されるかを理解する
- Qt アプリケーションを Release モードでビルドする
- カメラまたは動画ファイルでデモを実行する

<!-- cell: t21-npu-pattern src: 0c9b4db3ff -->
## このアプリケーションの NPU の使い方

| 項目 | このチュートリアルでは | 学んだ場所 |
|---|---|---|
| エンジン | タスクごとに 1 つ、合計 **4 つ**の `InferenceEngine` オブジェクトをすべて同じデバイス上で使用し、DX-RT がそれらのリクエストを NPU コア上でインターリーブする | T06-3 §2 |
| 実行 | 各ワーカーは `bufferCount` を自身のインフライト上限に設定して `RunAsync()` を呼び出し、完了コールバック内で描画する。最新フレームキューはレイテンシを積み上げる代わりに古いフレームを破棄する | T06-2 §5, §7 |
| タスクグラフ | 4 つのモデルはすべて `cpu_0` タスク(ONNX Runtime による CPU 上でのヘッドデコード)で終わるため、モデルを追加するたびにホスト CPU の負荷が増える | T06-1 §6 |
| リクエストあたりの入力 | 3 つのモデルは `[1, 640, 640, 3]` UINT8(1.2 MB)、深度モデルは `[1, 768, 768, 3]`(1.8 MB)を受け取る。1 枚のカメラフレームが 4 つのレターボックス済み入力になる | T06-3 §4 |
| 測定するもの | 3.1 節のモデルごとの `dxrun` ベースラインと、ウィンドウ内の 4 つの FPS ラベルを比較する。4 つのパネルのレートの合計が、NPU がどのように共有されているかを示す | T06-1 §5 |

<!-- cell: t21-prerequisites src: 2ab76b757a -->
## 前提条件

- チュートリアル 01 が完了していること: DX-RT がインストール済みで、DEEPX NPU が `/dev/dxrt0` として認識されていること。Qt ウィンドウ用に `cmake`、`g++`、`qtbase5-dev` が必要です。
- グラフィカルなデスクトップセッション。カメラデモ用の V4L2 カメラ(任意)。
- ダウンロード: 約 80 MB のアーカイブ 1 つ(深度モデルとサンプル動画、およびチュートリアル 01 がすでに共有ワークスペースに配置した 3 つの YOLO26s モデルのコピー)。
- 所要時間: 約 20 分。ビルドには 1〜2 分かかります。パッケージが不足している場合の第 2 節の `apt-get` 行を除き、`sudo` は不要です。
- DX-RT 3.4.2 で検証済みです。モデルは DX-COM 2.4.0 でコンパイルされています。

次のセルは SDK の場所を特定し、これらの要件を確認してステータス表を表示します。`MISSING` と表示された項目には、それを提供する手順が添えられます。

<!-- cell: f8c4f71c src: bafe0141d6 -->
## 1. チュートリアルファイルの場所を確認する

<!-- cell: 9c6f9a02 src: fa676f3bae -->
### 1.1 プロジェクト構成

```text
T21-demo-yolo26-od-pose-seg-depth/
├── get_resources.sh
├── assets/
│   ├── models/
│   └── videos/
├── app/
│   ├── build.sh
│   ├── run_camera.sh
│   ├── run_video.sh
│   ├── CMakeLists.txt
│   ├── yolo26s_4.cpp
│   ├── common/
│   │   ├── base/
│   │   ├── processors/
│   │   └── utility/
│   ├── factory/
│   └── extern/
└── yolo26_od_pose_seg_depth.ipynb
```

<!-- cell: afe9307e src: b0dd491e5c -->
## 2. 環境を確認する

Debian または Ubuntu のパッケージをインストールします。

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config qtbase5-dev libopencv-dev ffmpeg v4l-utils
```

リソースアーカイブは約 84 MB です(モデル 4 つとサンプル動画 1 つ)。

<!-- cell: c457d2df src: 9ff0a2e16c -->
## 3. リソースをダウンロードして確認する

アプリケーションは `assets/` 配下に 4 つのモデルと 1 つのサンプル動画があることを前提としています。

```text
assets/
├── models/
│   ├── yolo26-s_640x640.dxnn
│   ├── yolo26-s-pose_640x640.dxnn
│   ├── yolo26-s-seg_640x640.dxnn
│   └── yolo26-depth-s_768x768_q-lite.dxnn
└── videos/
    └── dance-960-540.mp4
```

検出、姿勢推定、セグメンテーションの 3 つのモデルは、チュートリアル 01 の 3.3 節で共有ワークスペース(`<DX_ALL_SUITE_DIR>/workspace/res/models`)にダウンロードしたものと同じファイルなので、次のセルはそれらが存在する場合にそこからリンクします。深度モデルと動画はこのチュートリアルのアーカイブ(約 80 MB)にのみ含まれています。`get_resources.sh` がそれをダウンロードして `assets/` に展開し、展開が成功した後にアーカイブを削除します。

<!-- cell: bbbdc0e0 src: 670c65bcaf -->
次のセルはリソースが不足している場合にのみ実行してください。同じ名前の既存ファイルは展開時に置き換えられることがあります。

<!-- cell: t21-baseline-md src: e5aeb7a350 -->
### 3.1 4 つのモデルを検査してベンチマークする

アプリケーションを読む前に、4 つのエンジンが実行するものを確認しましょう。`dxparse -v` を見ると、すべてのモデルが NPU タスクの後に `cpu_0` タスクを持つことがわかります。検出ヘッド(および深度モデルの出力ステージ)は DX-RT 内部の ONNX Runtime でデコードされるため、意味のあるベンチマークには `--use-ort` が必要です。`dxrun` の数値は単一モデルのベースラインです。アプリケーションは同じ NPU 上で 4 つすべてを同時に実行するため、各パネルの FPS はそのベースラインより低くなります。

```bash
cd <T21>/assets/models
dxparse -m yolo26-s_640x640.dxnn -v
dxrun -m yolo26-s_640x640.dxnn --use-ort -t 3 -v
```

<!-- cell: 2deb9884 src: 1d367207ac -->
## 4. アプリケーションアーキテクチャ

```text
Camera or video
       |
       v
CaptureThread
       |
       +--> Detection worker    --> Object Detection panel
       +--> Pose worker         --> Pose Estimation panel
       +--> Segmentation worker --> Instance Segmentation panel
       +--> Depth worker        --> Depth Estimation panel
```

キャプチャスレッドは各 BGR フレームを 4 つの最新フレームキューに公開します。検出、姿勢推定、セグメンテーションはタスク固有のファクトリを使用します。`DepthWorker` は 768 x 768 のレターボックス前処理、非同期 DXRT 推論、レターボックスの除去、ソースフレームへのリサイズ、Turbo カラーマッピングを行います。Qt は 4 つのライブ結果をすべて全画面の 2 x 2 グリッドに表示します。

<!-- cell: 51d08357 src: 44c389b817 -->
## 5. C++ コードを読む

次のヘルパーは、現在のソースファイルから選択したセクションを表示します。

<!-- cell: 5e81e61e src: aa11e7c86a -->
### 5.1 ビルド構成

CMake は 1 つの C++17 実行ファイルをビルドし、Qt5 Widgets、OpenCV、DXRT をリンクします。`PROJECT_ROOT_DIR` はチュートリアルディレクトリを指し、既定のモデルパスが `assets/models/` 配下に解決されるようにします。

<!-- cell: 02a0873a src: 4e66e3d6da -->
### 5.2 コマンドラインオプションと既定のリソース

`AppArgs` は 4 つのモデルパス、カメラ設定、任意の動画パス、デバッグオプションを定義します。`--model-depth` は既定の深度モデルを上書きします。`--video` を省略するとカメラモードになります。V4L2 デバイスを選ぶには `-c` または `--camera` を、キャプチャ設定を要求するには `--width`、`--height`、`--fps` を使用します。これらのオプションを省略した場合の既定値は `/dev/video0`、1280 x 720、30 FPS です。

<!-- cell: 3a6d2550 src: 5104ab68bd -->
### 5.3 タスクファクトリ

各ファクトリは、1 つのタスクに対して適切な前処理、後処理、可視化コンポーネントを生成します。

<!-- cell: 70995906 src: e8239c4efd -->
### 5.4 非同期結果ワーカー

各ワーカーは独自の `InferenceEngine` を生成し、非同期コールバックを登録します。検出、姿勢推定、セグメンテーションはタスク固有のファクトリを使用します。専用の深度ワーカーは `[1, 768, 768, 3]` UINT8 入力と `[1, 1, 768, 768]` FLOAT 出力のコントラクトを検証し、コールバックが完了するまで入力バッファを保持し、レターボックスのパディングを除去し、ソースフレームのジオメトリを復元して、OpenCV の Turbo カラーマップを適用します。各最新フレームキューは、モデルが入力ストリームより遅い場合にレイテンシを積み上げる代わりに古いフレームを置き換えます。

<!-- cell: 01af8912 src: 6bab0dfd1b -->
### 5.5 カメラと動画のキャプチャ

`CaptureThread` は OpenCV の `VideoCapture` を使用します。動画入力は `--no-loop-video` が設定されていない限りループ再生されます。カメラ入力は V4L2 バックエンドを使用し、MJPG 形式を要求します。

<!-- cell: 6ec88213 src: f00de1deb0 -->
### 5.6 Qt 2 x 2 ウィンドウ

`QuadWindow` は 4 つのライブパネルを生成し、4 つの推論ワーカーとキャプチャスレッドを起動し、FPS ラベルを更新し、整然とシャットダウンを行います。

<!-- cell: fef60649 src: 2836bd7037 -->
## 6. アプリケーションをビルドする

`build.sh` は `app/build/` を作成し、CMake を Release モードで構成し、`nproc` が報告するすべての CPU コアを使って `make` を実行します。完全な再ビルドには `./build.sh --clean` を、DX-RT がカスタムプレフィックス配下にインストールされている場合は `./build.sh -DCMAKE_PREFIX_PATH=<prefix>` を使用してください。

<!-- cell: 6e6f1fe2 src: b5bfeb4275 -->
## 7. カメラデモを実行する

既定のデバイスは `/dev/video0` で、1280 x 720 と 30 FPS を要求します。別のカメラやキャプチャ設定を選ぶには、下の変数を変更してください。実行セルは全画面の Qt ウィンドウを開き、アプリケーションが終了するまでブロックします。

<!-- cell: c267e0de src: db9fcae673 -->
## 8. 動画デモを実行する

`VIDEO_PATH` に `assets/videos/` 配下のファイルを設定します。`run_video.sh` は最初の引数として動画パスを受け取ります。

<!-- cell: 9dee3c0f src: ca736df157 -->
## 9. 操作方法

- `Esc` または `q`: 終了
- `EXIT` ボタン: マウスで終了

<!-- cell: 05fc0113-d263-4d67-a4af-92c174b45b94 src: 6093c888ab -->
## 10. トラブルシューティング

| 症状 | 表示される内容 | 考えられる原因 | 対処 |
|---|---|---|---|
| CMake が Qt5 を見つけられない | `Could not find a package configuration file provided by "Qt5"` | `qtbase5-dev` がインストールされていない | `sudo apt install qtbase5-dev` |
| CMake が DXRT を見つけられない | `Could not find a package configuration file provided by "dxrt"` | DX-RT がインストールされていないか、カスタムプレフィックス配下にある | DX-Runtime をインストールする(チュートリアル 01 第 3 節)か、`./build.sh -DCMAKE_PREFIX_PATH=<prefix>` を実行する |
| モデルがロード時に拒否される | DX-RT のバージョンまたは形式のエラー | DXNN が別の SDK リリース向けにコンパイルされた(これらのモデルは DX-COM 2.4.0) | このチュートリアルが検証された DX-RT バージョン(3.4.2)を使用するか、お使いの DX-COM で再コンパイルする |
| モデルを開けない | `assets/models` 配下のファイル名を示すロードエラー | リソースが不足している | 第 3 節のリソースセルを実行する |
| カメラを開けない | `cannot open /dev/video0` | カメラがないか、権限がない | `v4l2-ctl --list-devices` を実行し、ユーザーを `video` グループに追加し、`--camera` で別のデバイスを指定する |
| Qt ウィンドウが表示されない | `could not connect to display` | グラフィカルセッションがない | デスクトップセッションで実行する |

<!-- cell: 83f9ac2c src: 18f95b8d4f -->
## 11. まとめ

1 つのキャプチャスレッドが各フレームを 4 つの独立した非同期 DXRT パイプラインに送ります。タスク固有のファクトリが物体検出、姿勢推定、インスタンスセグメンテーションを処理します。専用の深度ワーカーはフレームをレターボックス処理し、768 x 768 の深度モデルを実行し、深度マップをソースのジオメトリに復元して、Turbo カラーマップを適用します。Qt は 4 つのライブ結果をすべて全画面の 2 x 2 レイアウトに表示します。

### 11.1 完了チェックリスト

- [ ] 4 つのモデルとサンプル動画をダウンロードする
- [ ] 1 つのキャプチャスレッドが 4 つの非同期パイプラインに入力を供給する仕組みを読む
- [ ] Qt アプリケーションをビルドする
- [ ] 動画デモとカメラデモを実行する

> **次へ:** チュートリアル 22 に進み、CLIP でカメラフレームをテキストクエリと照合してみましょう。
