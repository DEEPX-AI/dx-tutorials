<!-- i18n source: notebooks/T24-demo-pidnet-cityscapes/pidnet_cityscapes.ipynb -->
<!-- i18n lang: ja -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: title src: a79a64189a -->
# DEEPX Tutorial 24 - PIDNet Cityscapes C++ デモ

このノートブックでは、Qt5 C++ のセマンティックセグメンテーションアプリケーションを解説し、ビルドして実行します。カメラまたは動画のフレームを DEEPX NPU 上の PIDNet で処理し、下部のコントロールバーに大きな入力テンソルを強調表示し、PIDNet の argmax スケールを調整するコンパクトなライブスライダーを提供します。

<!-- cell: 062ad116-becd-4b26-a4b3-475220ad077f src: cb4f28770c -->
## 学習目標

このチュートリアルを終えると、次のことができるようになります。

- PIDNet のロジットが色付きの Cityscapes オーバーレイになる仕組みを説明する
- デモ用のモデルとサンプル動画をダウンロードする
- Qt アプリケーションをビルドし、カメラまたは動画で実行する
- Argmax スケールスライダーを使って、後処理の解像度と CPU 時間をトレードオフする
- 使用するホストに合った処理中(in-flight)リクエスト数を選ぶ

<!-- cell: t24-npu-pattern src: 0b3416de9c -->
## このアプリケーションの NPU の使い方

| 項目 | このチュートリアルでの内容 | 学んだ場所 |
|---|---|---|
| エンジン | `InferenceEngine` 1 つ。モデルは NPU のみ(CPU タスクなし)なので、DX-RT は生のクラスロジットを返し、後処理全体はアプリケーションが担当する | T06-1 §6 |
| 実行 | キャプチャスレッドは最大 `--inflight`(既定 4)個の `RunAsync()` リクエストをキューに保持する。NPU が次のフレームを処理している間に、完了コールバック内で argmax とレンダリングが実行される | T06-2 §5, T06-3 §3 |
| リクエストあたりの入力 | `[1, 1024, 2048, 3]` UINT8 = 6.3 MB で、これらのチュートリアルの中で最大の入力。出力は `[1, 19, 128, 256]` FLOAT = 2.5 MB。ホストからデバイスへ、およびデバイスからホストへの転送が各リクエストの中で無視できない割合を占める | T06-3 §4 |
| CPU 側 | 19 クラスに対するピクセルごとの argmax は CPU の作業。Argmax スケールスライダーは、NPU のレートを維持したまま、その解像度と CPU 時間をトレードオフする | T06-3 §10 |
| 測定するもの | 4.1 節の `dxrun -v`(NPU 処理時間とレイテンシの比較)と、異なるスライダー位置および `--inflight` 値での GUI FPS | T06-1 §5 |

<!-- cell: pipeline src: 2dfd7495e4 -->
## 1. 処理パイプライン

```text
Camera or video frame
        |
        v
Resize to model input and convert BGR to RGB
        |
        v
Asynchronous PIDNet inference -> class logits
        |
        v
Bilinear interpolation at the selected argmax scale
        |
        v
Per-pixel argmax -> Cityscapes color mask -> Qt5 GUI
```

既定のモデルは UINT8 `[1, 1024, 2048, 3]` の入力を受け取り、FLOAT32 `[1, 19, 128, 256]` のロジットを返します。

<!-- cell: prerequisites src: 9341de1df6 -->
## 2. 前提条件

- チュートリアル 01 が完了していること: DX-RT がインストール済みで、DEEPX NPU が `/dev/dxrt0` として認識されている(`dxcli -s` で確認)。Qt ウィンドウのために `cmake`、`g++`、`qtbase5-dev` が必要です。
- グラフィカルなデスクトップセッション。カメラデモには V4L2 カメラ(任意)。
- ダウンロード: 約 20 MB のアーカイブ 1 つ(モデルとサンプル動画)を `assets/` に配置します。このディレクトリは git で無視されます。
- 所要時間: 約 15 分。ビルドには約 1 分かかります。パッケージが不足している場合の以下の `apt` 行を除き、`sudo` は不要です。
- DX-RT 3.4.2 で検証済み。モデルは DX-COM 2.3.0 でコンパイルされています。

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev qtbase5-dev ffmpeg v4l-utils
```

次のセルは SDK の場所を特定し、これらの要件を確認してステータス表を表示します。`MISSING` と表示された項目には、それを提供する手順が添えられます。

<!-- cell: layout-heading src: 572e07788b -->
## 3. プロジェクト構成

C++ コードと起動スクリプトは `app/` 配下にあります。ダウンロードしたモデルと動画は `assets/` 配下に配置します。

<!-- cell: resources src: 163f686043 -->
## 4. リソースのダウンロードと確認

このデモは公式の [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet) プロジェクトを参照しています。その PyTorch チェックポイント `PIDNet_S_Cityscapes_val.pt` を DEEPX NPU 推論用に DXNN 形式へ変換しました。アプリケーションは変換後の `pidnet_s_cityscapes_val_fixed.dxnn` モデルを使用します。

`get_resources.sh` はリソースアーカイブをダウンロードし、モデルとサンプル動画を `assets/` に展開し、展開が成功したらダウンロードしたアーカイブを削除します。

```text
assets/
├── models/
│   └── pidnet_s_cityscapes_val_fixed.dxnn
└── videos/
    └── pidnet.mp4
```

<!-- cell: resource-download-note src: 670c65bcaf -->
次のセルはリソースが不足している場合にのみ実行してください。同じ名前の既存ファイルは展開時に置き換えられることがあります。

<!-- cell: inspect-heading src: bf1c859947 -->
### 4.1 任意: モデルの検査

`dxparse` とモデルが利用可能な場合、次のセルは入力テンソルと出力テンソル、および `dxrun -v` のベースラインを表示します。NPU 処理時間とエンドツーエンドのレイテンシを比較してください。6.3 MB の入力によって、ホストからデバイスへの転送がすべてのリクエストの中で目に見える部分を占めます。非同期の処理中(in-flight)キューが隠すのはまさにこの部分です。利用できない場合、セルは失敗せずに確認をスキップします。

<!-- cell: scale-heading src: affd9c896d -->
## 5. `pidnet_argmax_scale` を理解する

PIDNet は低解像度のクラスロジットを生成します。各ピクセルで最も可能性の高いクラスを選択する前に、アプリケーションはそのロジットをバイリニア補間します。スケールは中間の補間サイズを制御します。

- `0.1`: CPU コストが最も低く、境界は粗くなります。
- `0.4`: 速度と精細さのバランスを取った既定値です。
- `1.0`: フルフレームの argmax 解像度で、CPU コストが最も高くなります。

GUI のスライダーは 0.10 から 1.00 までの値を 0.05 刻みで使用します。スライダーはアトミックな値に書き込み、各非同期完了コールバックはフレームを処理する前にその値を 1 回読み取ります。これにより、すべてのフレームが一貫した 1 つのスケールを持ちます。

<!-- cell: code-heading src: ed5809056c -->
## 6. C++ コードガイド

実装は `app/pidnet_cityscapes.cpp` にあります。次のセルは、チュートリアル 20〜23 と同様に、主要なセクションをソースファイルから直接表示します。

<!-- cell: code-explanation src: f2781bc990 -->
### 6.1 実装の主な段階

1. `Options` と `parse_args` は、カメラまたは動画、モデル、カメラ設定、初期スケール、オーバーレイの不透明度を選択します。
2. `preprocess_frame` は BGR を RGB に変換し、フレームをモデルの入力テンソルに直接リサイズします。
3. `compute_argmax_mask` は現在のスケールですべてのクラスチャンネルを補間し、最もスコアの高いクラスを選択します。
4. `render_segmentation` はクラス ID を Cityscapes の色に対応付け、マスクを入力フレームとブレンドします。
5. `MainWindow` は、ランタイムの入力形状、コンパクトなスケールスライダー、Exit ボタンを含む下部のコントロールバーの上に動画を配置します。
6. キャプチャスレッドは `RunAsync` で最大 4 つのリクエストを投入し、NPU 推論と CPU 後処理を重ね合わせられるようにします。
7. 完了コールバックは、最新のレンダリング結果だけを 1 フレーム分のメールボックスに公開します。Qt タイマーは古い GUI イベントを蓄積せずにそれを消費します。
8. OpenCV の操作は内部スレッドを 1 つだけ使用します。フレーム単位のコールバックの並列処理が、利用可能な CPU コアをすでに使い切っているためです。

<!-- cell: build-heading src: 6910ddc7ad -->
## 7. ビルド

`build.sh` は Release ビルドを構成し、利用可能なすべての CPU コアで `make` を実行します。先に古いビルドディレクトリを削除するには `--clean` を使用してください。

<!-- cell: camera-heading src: 6ce2ff2946 -->
## 8. カメラで実行する

スクリプトはカメラインデックス 0 を 1280 x 720、30 FPS で要求します。NPU、モデル、カメラ、グラフィカルセッションの準備が整っている場合にのみ `RUN_CAMERA` を `True` に設定してください。

<!-- cell: video-heading src: 85d23010af -->
## 9. 動画で実行する

`run_video.sh` は `assets/videos/pidnet.mp4` を読み込み、ループ再生します。モデルと動画を所定の場所に配置した後、`RUN_VIDEO` を `True` に設定してください。

<!-- cell: custom-run src: 3eda8227c7 -->
## 10. カスタムコマンド

別のスライダー値で開始する:

```bash
cd app
./run_video.sh --pidnet-argmax-scale 0.7
```

ウィンドウモードで別の動画を使用する:

```bash
./build/pidnet_cityscapes --video /path/to/input.mp4 --loop --windowed
```

別のカメラを選択し、マスクの不透明度を下げる:

```bash
./run_camera.sh --camera 2 --width 1920 --height 1080 --fps 30 --alpha 0.45
```

終了するには `Esc`、`Q`、または Exit ボタンを使用します。`F` を押すとフルスクリーンモードを切り替えます。既定の `--inflight 4` は 4 コアの Raspberry Pi 5 を想定しています。さらにスループットが必要な場合は `--inflight 6` を試してください。

<!-- cell: d4dfe950-47a4-4a4e-adb8-c9defa2b6703 src: 5c302aa913 -->
## 11. モデルの出所

このデモは公式の [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet) プロジェクトを参照しています。PyTorch チェックポイント `PIDNet_S_Cityscapes_val.pt` はそのプロジェクトから取得し、DEEPX NPU 推論用に DXNN 形式へ変換しました。変換後の `pidnet_s_cityscapes_val_fixed.dxnn` モデルをこのアプリケーションで使用しています。

<!-- cell: 508b6045-6a18-4e39-8844-ad638d890d04 src: 08a0e7b298 -->
## 12. その他のオプションと操作

- `--alpha VALUE`: セグメンテーションオーバーレイの不透明度を 0.0〜1.0 で設定します。既定値は 0.6 です。
- `--inflight N`: 非同期リクエストの最大数を 1〜6 で設定します。既定値は 4 コアの Raspberry Pi 5 向けの 4 です。
- `--no-pace`: 元の FPS に合わせるのではなく、動画を可能な限り高速に処理します。
- `--windowed`: 1280 x 800 のウィンドウで起動します。
- `--full-screen`: フルスクリーンモードで起動します。スクリプトの既定値です。
- `Esc`、`Q`、または **Exit** ボタン: アプリケーションを終了します。
- `F`: フルスクリーンモードを切り替えます。

オプションの一覧は `./build/pidnet_cityscapes --help` を実行して確認してください。

<!-- cell: 81d36d4a-2c61-4c7b-aa91-004b609ab880 src: 12bee04ca1 -->
## 13. 性能設計

キャプチャスレッドは `RunAsync` でフレームを投入します。NPU が他のフレームの処理を開始している間に、完了コールバックが argmax とマスクのレンダリングを実行します。上限付きの処理中(in-flight)リミッターが、メモリの無制限な増加を防ぎます。OpenCV の内部ワーカープールは 1 スレッドに制限されています。並列処理はすでにコールバック間で行われているためで、これにより 4 コアシステムでの入れ子になったオーバーサブスクリプションを回避します。

完了したフレームは 1 フレーム分のメールボックスに書き込まれます。Qt スレッドは常に最新の結果を取り出すため、遅延したフレームが GUI キューに溜まることはありません。これにより、CPU の後処理が入力レートに追いつけない場合でも操作の応答性が保たれます。

既定の `--inflight 4` は Raspberry Pi 5 の CPU 数に合わせたものです。CPU 使用率と温度が許容範囲内であれば、`--inflight 6` を試して DXRT のバッファプール全体を使用してください。

```bash
./run_video.sh --inflight 6
```

<!-- cell: troubleshooting src: 2b346be6d4 -->
## 14. トラブルシューティング

| 症状 | 表示される内容 | 考えられる原因 | 対処 |
|---|---|---|---|
| モデルがない | リソース確認で `missing` と表示される | `get_resources.sh` が実行されていない | リソースセルを実行し、`assets/models/` 配下のファイル名を確認する |
| 動画がない | `pidnet.mp4` が見つからない | リソースが不完全 | `pidnet.mp4` を `assets/videos/` 配下に配置するか、`run_video.sh` に別のパスを渡す |
| カメラを開けない | `cannot open camera 0` | カメラがないか、権限がない | `v4l2-ctl --list-devices` を実行し、別のインデックスを試す |
| ウィンドウが表示されない | `could not connect to display` | グラフィカルセッションがない | デスクトップセッションで実行する |
| CMake が DXRT を見つけられない | `Could not find a package configuration file provided by "dxrt"` | DX-RT がインストールされていないか、カスタムプレフィックス配下にある | DX-Runtime をインストールする(チュートリアル 01 第 3 節)か、`./build.sh -DCMAKE_PREFIX_PATH=<prefix>` を実行する |
| ロード時にモデルが拒否される | DX-RT のバージョンまたは形式のエラー | DXNN が古いランタイム向けに DX-COM 2.3.0 でコンパイルされている | このチュートリアルで検証済みの DX-RT バージョン(3.4.2)を使用する |
| 後処理が遅い | NPU がアイドル状態なのに FPS が低い | フル解像度での CPU argmax | Argmax スケールスライダーを 0.1 の方向へ動かすか、`--inflight` を下げる |

<!-- cell: 06f03620-4bef-40dd-84a5-9fcfb0ea5fc8 src: eae16a4f91 -->
## 15. まとめ

PIDNet は NPU 上で動作し、アプリケーションはそのクラスロジットを CPU 上で色付きの Cityscapes セグメンテーションオーバーレイに変換します。Argmax スケールスライダーは、モデルをフルスピードで動作させたまま、後処理の解像度と CPU 時間をトレードオフします。

### 15.1 完了チェックリスト

- [ ] `get_resources.sh` でモデルとサンプル動画をダウンロードする
- [ ] `dxparse` でモデルを検査する
- [ ] `build.sh` でアプリケーションをビルドする
- [ ] アプリケーションを実行し、Argmax スケールを調整する

> **次へ:** デモシリーズは以上で完了です。チュートリアルの全一覧はリポジトリの README を参照してください。
