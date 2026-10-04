<!-- i18n source: notebooks/T22-demo-clip-single/clip_single.ipynb -->
<!-- i18n lang: ja -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: t22-01 src: e41c8db501 -->
# DEEPX Tutorial 22 - CLIP シングルストリーム C++ デモ

このチュートリアルでは、カメラまたは動画入力を使う C++ CLIP アプリケーションのビルド方法と実行方法を説明します。テキストクエリは ONNX Runtime でエンコードし、画像は DEEPX NPU で非同期にエンコードし、Qt GUI が類似度スコアを表示します。

<!-- cell: t22-02 src: 4b7588c5da -->
![CLIP シングルストリームデモ](assets/clip-single-sc.png)

<!-- cell: t22-03 src: 0d359c5f65 -->
## 学習目標

このチュートリアルを終えると、次のことを理解できるようになります。

- CLIP が画像の埋め込みとテキストの埋め込みをどのように比較するか
- C++ アプリケーションが ONNX Runtime と DXRT をどのように組み合わせているか
- カメラと動画のフレームが画像エンコーダー向けにどのように準備されるか
- 非同期 NPU 推論とフレームスキップがどのように動作するか
- アプリケーションのビルド方法と実行方法

<!-- cell: t22-npu-pattern src: f43836dc91 -->
## このアプリケーションの NPU の使い方

| 項目 | このチュートリアルでは | 学んだ場所 |
|---|---|---|
| エンジン | ViT-L/14 画像エンコーダー用に `InferenceEngine` を 1 つ。テキストエンコーダーは DX-RT の外で ONNX Runtime を使い CPU 上でクエリごとに 1 回実行され、その埋め込みはキャッシュされる | T06-2 §4 |
| 実行 | `bufferCount` を処理中(in flight)のフレーム数と等しくした `RunAsync()`。投入された各フレームは、コールバックが 768 値の埋め込みをコピーするまで入力バッファを保持する | T06-2 §5, §7 |
| タスクグラフ | この DXNN は `cpu_0` タスクで始まり NPU で終わるため、CPU 部分がクリティカルパス上にあり、`--use-ort` が必須 | T06-1 §6 |
| リクエストごとの入力 | ホスト側でリサイズ、センタークロップ、正規化を行った後の `[1, 3, 224, 224]` FLOAT = 602 KB | T06-3 §4 |
| 測定するもの | 第 4 節の `dxrun` ベースラインと、`--skip-frames 0` での GUI のレートとの比較。`--skip-frames` はランキングの鮮度と NPU 負荷をトレードオフするつまみ | T06-1 §5 |

<!-- cell: 307b3492-06ab-4f38-8103-9d48f85e1fd8 src: 5091fc3c20 -->
## 前提条件

- DX-RT がインストール済みの DEEPX NPU(チュートリアル 01 第 3 節)と `cmake`。
- テキストエンコーダー用の ONNX Runtime C++ ヘッダーと共有ライブラリ。DX-Runtime はこれらを `/usr/local` 配下にインストールします。別のプレフィックスの場合は `./build.sh -DONNXRUNTIME_ROOT=<prefix>` を実行してください。
- Qt ウィンドウ用のディスプレイと、カメラデモ用の V4L2 カメラ。
- ダウンロード: リソースアーカイブは約 1.4 GB(画像エンコーダー DXNN、テキストエンコーダー ONNX、語彙、サンプル動画)で、git が無視する `assets/` に展開されます。
- 所要時間: ダウンロードを除いて約 20 分。ビルドには約 1 分かかります。
- DX-RT 3.4.2 で検証済み。画像エンコーダーは DX-COM 2.2.1 でコンパイルされています。

Debian または Ubuntu のパッケージをインストールします。

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev qtbase5-dev zlib1g-dev
```

次のセルは SDK の場所を特定し、これらの要件を確認してステータス表を表示します。`MISSING` と表示された項目には、それを提供する手順が添えられます。

<!-- cell: t22-05 src: 22cfa55977 -->
## 1. プロジェクト構成

```text
T22-demo-clip-single/
├── README.md
├── clip_single.ipynb
├── get_resources.sh
├── assets/
│   ├── models/
│   ├── videos/
│   └── images/
└── app/
    ├── CMakeLists.txt
    ├── build.sh
    ├── run_camera.sh
    ├── run_video.sh
    ├── main.cpp
    ├── clip_tokenizer.cpp
    └── clip_tokenizer.hpp
```

<!-- cell: t22-07 src: 25298446ba -->
## 2. リソースのダウンロードと確認

`get_resources.sh` はリソースアーカイブをダウンロードし、画像エンコーダー、テキストエンコーダー、BPE 語彙、サンプル動画を `assets/` に展開し、展開が成功した後にダウンロードしたアーカイブを削除します。テキストエンコーダーは重みを外部データとして保存しているため、ONNX の `.data` ファイルが必要です。

<!-- cell: t22-08-download-note src: 670c65bcaf -->
次のセルは、リソースが不足している場合にのみ実行してください。同じ名前の既存ファイルは展開時に置き換えられることがあります。

<!-- cell: t22-09 src: 49c8ad89f7 -->
## 3. 推論パイプライン

```text
Text queries -> BPE tokens -> ONNX Runtime text encoder -> text embeddings
                                                              |
Camera/video -> resize and center crop -> DXRT image encoder -> image embedding
                                                              |
                                                              v
                                      L2 normalization -> dot products -> ranked GUI scores
```

テキストの埋め込みは起動時に計算されてキャッシュされます。選択された各動画フレームは 224 x 224 の CHW float テンソルに変換されます。DXRT は画像推論を非同期に投入するため、キャプチャと GUI の更新は各 NPU リクエストの完了を待ちません。

<!-- cell: t22-11 src: 2cc1224d10 -->
### 3.1 コマンドラインオプション

`AppOptions` は、モデルのパス、入力ソース、カメラ設定、フレームスキップの間隔、GUI フラグを定義します。`--input` を省略すると、アプリケーションは `--camera` で選択されたカメラを使用します。

<!-- cell: t22-13 src: d79b113959 -->
### 3.2 画像の前処理

画像はアスペクト比を保ったままリサイズされ、224 x 224 にセンタークロップされ、BGR から RGB の順序に変換され、正規化されて CHW レイアウトで格納されます。

<!-- cell: t22-15 src: 160edac714 -->
### 3.3 テキストエンコードとキャッシュ

`TextEncoder` は各クエリをトークン化し、ONNX テキストエンコーダーを実行します。`TextFeatureStore` は正規化された埋め込みを、モデルとテキストリストをキーとして `.cache/text_features_cpp/` 配下に保存します。

<!-- cell: t22-17 src: dd4b9fd45c -->
### 3.4 非同期の画像エンコード

`ImageEncoderAsync` は複数のバッファを持つ DXRT 推論エンジンを作成します。投入された各フレームは、完了コールバックが 768 値の画像埋め込みをコピーするまで入力メモリを所有します。

<!-- cell: t22-19 src: f266352d14 -->
### 3.5 類似度によるランキング

任意の L2 正規化の後、アプリケーションはテキストクエリごとに 1 つの内積を計算します。正規化された埋め込みでは、これはコサイン類似度になります。GUI は、設定されたスコアしきい値を超える最も強い一致を強調表示します。

<!-- cell: t22-21 src: 39b6816859 -->
## 4. DXNN 画像エンコーダーを調べる

モデルが利用可能な場合、`dxparse` はモデルの入力テンソルと出力テンソルを表示します。アプリケーションは `[1, 3, 224, 224]` 形状の float 入力と 768 値の画像埋め込みを期待します。タスクの順序に注意してください。このモデルは `cpu_0` タスク(パッチ埋め込みが CPU で実行されます)で始まり NPU で終わるため、`--use-ort` が必須であり、CPU 部分がすべてのリクエストのクリティカルパス上にあります。その後 `dxrun` が、GUI と比較するための単一モデルのベースラインを示します。

<!-- cell: t22-23 src: b823314d41 -->
## 5. アプリケーションのビルド

`build.sh` は CMake を Release モードで構成し、`nproc` が報告するすべての CPU コアを使って `make` を実行します。既存のビルドディレクトリを先に削除するには `./build.sh --clean` を、カスタムプレフィックス配下のライブラリには `./build.sh -DONNXRUNTIME_ROOT=<prefix>` または `-DCMAKE_PREFIX_PATH=<prefix>` を使用してください。ビルドにはモデルファイルや動画ファイルは不要です。

<!-- cell: t22-26 src: 54770cb65b -->
## 6. カメラデモの実行

既定のソースは `/dev/video0` で、1280 x 720、30 FPS を要求します。リソースとカメラの準備ができたら `RUN_CAMERA = True` に設定してください。GUI が閉じられるまでセルはブロックされます。

<!-- cell: t22-28 src: 2048774b7b -->
## 7. 動画デモの実行

`run_video.sh` は `assets/videos/CLIP-demo.mp4` を使用し、ファイルの末尾でループします。リソースの準備ができたら `RUN_VIDEO = True` に設定してください。

<!-- cell: t22-30 src: 021a013221 -->
## 8. 便利なオプション

- `--skip-frames N` は `N + 1` フレームごとに 1 回画像推論を実行します。`0` はすべてのフレームを、`2`(既定値)は 3 フレームごとにエンコードします。値を大きくすると NPU と CPU の負荷が下がりますが、ランキングの更新頻度は低くなります。
- `--no-normalize` は類似度計算前の L2 正規化を無効にします。
- `--camera SOURCE` は `--input` を省略したときに使うカメラを選択します。
- `--input SOURCE` はカメラインデックス、カメラデバイス、または動画パスを受け付けます。
- `--full-screen` は Qt ウィンドウをフルスクリーンモードで開きます。
- `--exit-btn` は Exit ボタンを追加します。

GUI を閉じるには `Esc` または `Q` を押します。実行スクリプトには既定のテキストクエリが含まれています。完全に独自のクエリリストを使いたい場合は、`build/clip_single` を直接実行してください。

<!-- cell: 8d0c598f-eb24-4c6c-8238-f1ea3b1878b0 src: 73c3cfa45c -->
## 9. カスタムテキストクエリを使う

実行スクリプトはタスクに適した既定のテキストクエリを提供します。自分のクエリだけを使うには、実行ファイルを直接実行します。

```bash
cd notebooks/T22-demo-clip-single/app
./build/clip_single \
    --texts "A person" "A bicycle" "A cup" \
    --camera /dev/video0 \
    --skip-frames 2 \
    --full-screen \
    --exit-btn
```

カスタム動画の場合:

```bash
./build/clip_single \
    --texts "Cars are driving" "An empty road" \
    --input /path/to/video.mp4 \
    --exit-btn
```

すべてのオプションを確認するには `./build/clip_single --help` を実行してください。アプリケーションを閉じるには、`Esc` または `Q` を押すか、Exit ボタンをクリックします。

<!-- cell: 666c080f-6927-4134-9841-3a3cb4999793 src: 3fcbd3e27f -->
## 10. トラブルシューティング

| 症状 | 表示される内容 | 考えられる原因 | 対処 |
|---|---|---|---|
| CMake が ONNX Runtime を見つけられない | `Could not find ONNXRUNTIME` | ONNX Runtime C++ パッケージがインストールされていないか、既定のプレフィックスにない | インストールする(DX-Runtime が提供)か、`./build.sh -DONNXRUNTIME_ROOT=<prefix>` を実行する |
| CMake が DXRT を見つけられない | `Could not find a package configuration file provided by "dxrt"` | DX-RT がインストールされていないか、カスタムプレフィックス配下にある | DX-Runtime をインストールする(チュートリアル 01 第 3 節)か、`./build.sh -DCMAKE_PREFIX_PATH=<prefix>` を実行する |
| 画像エンコーダーがロード時に拒否される | DX-RT のバージョンまたは形式のエラー | DXNN が古いランタイム向けに DX-COM 2.2.1 でコンパイルされている | このチュートリアルの検証に使用した DX-RT バージョン(3.4.2)を使用する |
| ダウンロードが遅い、または止まる | `get_resources.sh` が長時間実行される | アーカイブが約 1.4 GB ある | 完了するまで待つ。セルを再実行すると `.part` ファイルから再開される |
| カメラを開けない | `cannot open /dev/video0` | カメラがないか、権限がない | `v4l2-ctl --list-devices` を実行し、`--camera` で別のデバイスを指定する |
| Qt ウィンドウが表示されない | `could not connect to display` | グラフィカルセッションがない | デスクトップセッションで実行する |

<!-- cell: t22-31 src: e079ee85d4 -->
## 11. まとめ

このアプリケーションは、CPU のテキストエンコーダーと非同期の NPU 画像エンコーダーを 1 つの C++ GUI に組み合わせています。テキスト特徴は再利用でき、画像特徴は継続的に生成され、正規化された内積によってタスク固有の分類器なしでゼロショットのテキストと画像のマッチングが実現されます。

### 11.1 完了チェックリスト

- [ ] 画像エンコーダー、テキストエンコーダー、語彙をダウンロードする
- [ ] `dxparse` で画像エンコーダーを調べる
- [ ] アプリケーションをビルドして `--help` を実行する
- [ ] 動画またはカメラでデモを実行する
- [ ] 自分のテキストクエリで `clip_single` を実行する

> **次へ:** チュートリアル 23 に進み、C++ アプリケーションでカメラからハンドランドマークを追跡してみましょう。
