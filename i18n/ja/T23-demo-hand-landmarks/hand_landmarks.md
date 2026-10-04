<!-- i18n source: notebooks/T23-demo-hand-landmarks/hand_landmarks.ipynb -->
<!-- i18n lang: ja -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: title src: 08f7a63301 -->
# DEEPX Tutorial 23 - 手のランドマーク C++ デモ

このノートブックでは、検出した各手の 21 個のランドマークを追跡する C++ アプリケーションを解説し、ビルドして実行します。カメラまたは動画を入力として受け付け、2 つの推論ステージの両方に DEEPX NPU を使用します。

<!-- cell: 169e124a src: c366966d31 -->
![手のランドマークデモ](assets/hand-landmarks-sc.png)

<!-- cell: 99f74090-1591-4f3c-83d3-c7911115ef61 src: e791a70079 -->
## 学習目標

このチュートリアルを終えると、次のことができるようになります。

- 手のひら検出と手のランドマークの 2 ステージパイプラインを説明する
- デモ用のモデルとサンプル動画をダウンロードする
- `dxparse` で両方の DXNN モデルを検査する
- Qt アプリケーションをビルドし、カメラまたは動画で実行する
- コマンドラインから検出しきい値と表示オプションを調整する

<!-- cell: t23-npu-pattern src: 1878462df3 -->
## このアプリケーションの NPU の使い方

| 項目 | このチュートリアルでは | 学んだ場所 |
|---|---|---|
| エンジン | フレームごとに 2 つの `InferenceEngine` オブジェクトが順番に実行されます。手のひら検出器を 1 回、次に検出した手ごとにランドマークモデルを 1 回実行します | T06-2 §4 |
| 実行 | 手のひら検出は同期的に実行されます。各手のランドマークリクエストは `RunAsync()` で投入され、コールバック内でフレーム座標に戻されるため、複数の手の処理が NPU 上で重なります | T06-2 §5 |
| タスクグラフ | 手のひら検出器は `cpu_0` タスク(ONNX Runtime によるアンカーのデコード)で終わります。ランドマークモデルは NPU のみです | T06-1 §6 |
| リクエストあたりの入力 | `[1, 192, 192, 3]` UINT8(110 KB)と `[1, 224, 224, 3]` UINT8(150 KB)。入力が小さいため、ホスト側のクロップと回転が支配的になります | T06-3 §4 |
| 測定するもの | 4.1 節の 2 つの `dxrun` ベースラインと、ウィンドウに表示される FPS の比較。`--max-hands` を使うと、フレームあたりのランドマークリクエスト数が制限されます | T06-1 §5 |

<!-- cell: pipeline src: 148af37d15 -->
## 1. 処理パイプライン

このアプリケーションには手のパイプラインだけが含まれています。

```text
Camera or video frame
        |
        v
Palm detector, 192 x 192
        |
        v
Palm box and rotated hand region
        |
        v
Hand crop, 224 x 224 -> landmark model
        |
        v
21 landmarks and handedness -> Qt GUI
```

まず手のひら検出が手の領域を見つけます。各領域は回転・クロップされ、ランドマークモデルに渡されます。結果には、画像空間の 21 個の点、ワールドランドマーク、手の存在信頼度、左右の判定が含まれます。

<!-- cell: prerequisites src: 21ce181bb7 -->
## 2. 前提条件

- チュートリアル 01 が完了していること: DX-RT がインストール済みで、DEEPX NPU が `/dev/dxrt0` として認識されている(`dxcli -s` で確認)。Qt ウィンドウ用に `cmake`、`g++`、`qtbase5-dev` が必要です。
- グラフィカルなデスクトップセッション。カメラデモ用の V4L2 カメラ(任意)。
- ダウンロード: 約 10 MB のアーカイブ 1 個(モデル 2 個とサンプル動画)を `assets/` に展開します。このディレクトリは git で無視されます。
- 所要時間: 約 15 分。ビルドには約 1 分かかります。パッケージが不足している場合の下記 `apt` 行を除き、`sudo` は不要です。
- DX-RT 3.4.2 で検証済みです。モデルは DX-COM 2.3.0 でコンパイルされています。

C++ ビルドと GUI の依存関係は次のコマンドでインストールします。

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev qtbase5-dev ffmpeg v4l-utils
```

次のセルは SDK の場所を特定し、これらの要件を確認してステータス表を表示します。`MISSING` と表示された項目には、それを提供する手順が添えられます。

<!-- cell: layout-heading src: e62a57c223 -->
## 3. プロジェクト構成

アプリケーションのコードとスクリプトは `app/` 配下にあります。モデルと動画はビルドツリーの外の `assets/` 配下に置かれます。

<!-- cell: resources src: 46ee82775e -->
## 4. リソースのダウンロードと確認

2 つの推論モデルは、Google AI Edge が提供する [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) モデルバンドルに由来します。その手のひら検出モデルと手のランドマークモデルを、DEEPX NPU 推論用に DXNN 形式へ変換しています。

`get_resources.sh` はリソースアーカイブをダウンロードし、モデルとサンプル動画を `assets/` に展開して、展開が成功した後にダウンロードしたアーカイブを削除します。

```text
assets/
├── models/
│   ├── hand-detector_192x192.dxnn
│   └── HandLandmarkLite.dxnn
└── videos/
    └── hands.mp4
```

手のひらモデルの入力は UINT8 `[1, 192, 192, 3]` でなければなりません。ランドマークモデルの入力は UINT8 `[1, 224, 224, 3]` でなければなりません。

<!-- cell: resource-download-note src: 670c65bcaf -->
次のセルはリソースが不足している場合にのみ実行してください。同じ名前の既存ファイルは展開時に置き換えられることがあります。

<!-- cell: inspect-heading src: c632a33a79 -->
### 4.1 任意: モデルの検査

`dxparse` とモデルが利用可能な場合、次のセルはそれぞれのテンソル情報と `dxrun` のベースラインを表示します。手のひら検出器は `cpu_0` タスク(アンカーは DX-RT 内の ONNX Runtime でデコードされます)で終わるため、`--use-ort` が必要です。ランドマークモデルは NPU のみです。利用できない場合、セルは失敗せずに確認をスキップします。

<!-- cell: code-heading src: 14c08de877 -->
## 5. C++ コードガイド

実装は `app/hand_landmarks.cpp` にあります。次のセルは、チュートリアル 20〜22 と同様にソースファイルから主要な処理セクションを直接表示するため、ノートブックがコードの複製を持つことはありません。

<!-- cell: code-explanation src: 1a18d06026 -->
### 5.1 実装の主なステージ

1. `Options` と `parse_args` が入力、モデル、しきい値、表示モード、カメラ設定を選択します。
2. `preprocess_palm_frame` が BGR を RGB に変換し、192 x 192 の手のひら検出器入力を作成します。既定ではレターボックス処理を使用します。
3. `decode_palm_detections` が生のテンソルを手のひらボックスと 7 個のキーポイントに変換し、重み付きの非最大抑制(NMS)を適用します。
4. 手首と中指のキーポイントから回転した手の領域を定義します。`make_landmark_input` がそれを 224 x 224 にワープします。
5. `run_landmark_async` が検出した手のひらごとに 1 回のランドマーク推論を投入します。そのコールバックが結果をフレーム座標に戻します。
6. `draw_hand_landmarks` が 21 個の点を接続して描画します。`FrameView` がフレームと性能メトリクスを表示します。
7. `run_detection_loop` がキャプチャ、2 つの推論ステージ、描画、任意の動画保存、再生ペース調整を結合します。

<!-- cell: build-heading src: b00b1b09c7 -->
## 6. ビルド

`build.sh` は CMake を Release モードで設定し、利用可能なすべての CPU コアで `make` を実行します。先に以前のビルドディレクトリを削除するには `--clean` を使用してください。

<!-- cell: camera-heading src: 176345be53 -->
## 7. カメラで実行する

スクリプトはカメラインデックス 0 を使用し、1280 x 720、30 FPS を要求します。`RUN_CAMERA` を `True` に設定するのは、NPU、カメラ、モデルの準備が整ったグラフィカルセッションでのみにしてください。`Esc` または `Q` を押すとアプリケーションが終了します。

<!-- cell: video-heading src: 51c4ab12d3 -->
## 8. 動画で実行する

`run_video.sh` は `assets/videos/hands.mp4` を読み込んでループ再生します。モデルと動画を所定の場所に配置してから `RUN_VIDEO` を `True` に設定してください。

<!-- cell: custom-run src: 1a1d864d58 -->
## 9. カスタムコマンド

別のカメラとキャプチャモードを選択する:

```bash
cd app
./run_camera.sh --camera 2 --width 1920 --height 1080 --fps 30
```

カスタム動画を使用し、デバッグ用に手のひら領域を表示する:

```bash
./build/hand_landmarks --video /path/to/input.mp4 --loop --show-palm
```

描画結果を保存する。ファイル `output-XX.mp4` はカレントディレクトリに書き込まれるため、`app/` をきれいに保つにはチュートリアルの `workspace/` から実行してください。

```bash
mkdir -p ../workspace && cd ../workspace
../app/build/hand_landmarks --video /path/to/input.mp4 --save --landmark-only
```

`Esc` または `Q` で終了し、`F` でフルスクリーンモードを切り替えます。

<!-- cell: c1eb325f-0e0f-441f-979a-f3b1a4643a01 src: ecf000b5c7 -->
## 10. モデルの出所

このデモで使用する手のひら検出モデルと手のランドマークモデルは、Google AI Edge が提供する [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) モデルバンドルに由来します。モデルは DEEPX NPU 推論用に DXNN 形式へ変換され、`hand-detector_192x192.dxnn` と `HandLandmarkLite.dxnn` として使用されます。

<!-- cell: 86932e85-d1bd-4636-8069-5dd13cbb56ac src: 59326af863 -->
## 11. 便利なオプション

- `--max-hands N`: フレームあたりの手の最大数を設定します。既定値は 4 です。
- `--palm-conf VALUE`: 手のひらの信頼度しきい値を設定します。既定値は 0.2 です。
- `--landmark-conf VALUE`: ランドマークの存在しきい値を設定します。既定値は 0.5 です。
- `--show-palm`: 手のひらボックスと回転領域も描画します。
- `--landmark-only`: 手のランドマークのみを描画します。
- `--windowed`: フルスクリーンの代わりに 1280 x 720 のウィンドウを使用します。
- `--save`: 描画結果をカレントディレクトリに次に利用可能な `output-XX.mp4` ファイルとして保存します(`workspace/` から実行してください。第 9 節を参照)。

完全なオプション一覧は `./build/hand_landmarks --help` を実行して確認してください。`Esc` または `Q` で終了し、`F` でフルスクリーンモードを切り替えます。

<!-- cell: troubleshooting src: 22102be3a3 -->
## 12. トラブルシューティング

| 症状 | 表示される内容 | 考えられる原因 | 対処 |
|---|---|---|---|
| モデルがない | リソース確認でファイルが `missing` と表示される | `get_resources.sh` が実行されていないか、アーカイブが変更された | リソースセルを実行し、`assets/models/` 配下の両方のファイル名を確認する |
| モデルが拒否される | ロード時に形状または dtype のエラー | モデルファイルが誤っている | `dxparse` で入力形状と UINT8 dtype を確認する |
| カメラを開けない | `cannot open camera 0` | カメラがないか、権限がない | `v4l2-ctl --list-devices` を実行し、`--camera` で別のインデックスを試す |
| ウィンドウが表示されない | `could not connect to display` | グラフィカルセッションがない | デスクトップセッションで実行する |
| CMake が DXRT を見つけられない | `Could not find ... dxrt` | DX-RT がインストールされていないか、カスタムのプレフィックス配下にある | DX-Runtime をインストールする(チュートリアル 01 第 3 節)か、`./build.sh -DCMAKE_PREFIX_PATH=<prefix>` を実行する |

<!-- cell: 24061ba3-6ced-4492-93d3-3508af97fdf0 src: 99d0cc5118 -->
## 13. まとめ

2 つの DXRT モデルが順番に実行されます。手のひら検出器が手を見つけ、ランドマークモデルが手ごとに 21 個のキーポイントを予測します。Qt アプリケーションはフレームをキャプチャし、両方のモデルに非同期で入力を与え、ライブ表示にランドマークを描画します。

### 13.1 完了チェックリスト

- [ ] `get_resources.sh` でモデルとサンプル動画をダウンロードする
- [ ] `dxparse` で両方のモデルを検査する
- [ ] `build.sh` でアプリケーションをビルドする
- [ ] 動画またはカメラでアプリケーションを実行する

> **次へ:** チュートリアル 24 に進み、ライブの後処理コントロール付きで PIDNet セマンティックセグメンテーションを実行しましょう。
