<!-- i18n source: notebooks/T06-DX-Runtime/dx_rt_02_intermediate.ipynb -->
<!-- i18n lang: ja -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 20e090c0 src: d28eecdb2f -->
# DEEPX Tutorial 06-2 - DX-RT 中級編

このノートブックでは、CLI による検証から一歩進み、DX-RT の Python API と C++ API を使ったアプリケーションレベルの推論を扱います。

## 学習目標

このチュートリアルを終えると、次のことができるようになります。

- 対応するビルド済みの <code>dx_engine</code> wheel を Jupyter カーネルにインストールする
- テンソルのメタデータを読み取り、要求される shape と dtype で入力を確保する
- 同期推論を実装し、前処理から top-5 ラベルまで実際の画像を分類する
- ジョブ ID と <code>wait()</code>、および登録したコールバックを使った非同期推論を実装する
- バッファのライフタイムに関するルールを説明する
- バッチ API で独立したサンプルをまとめる
- <code>InferenceOption.buffer_count</code> を意図を持って使う
- <code>dxrt_cxx_api.h</code> を使って最小限の C++14 アプリケーションをビルドする
- レイテンシ、スループット、所有権の要件から実行モードを選択する

生成される Python、C++、ビルド、レポートのファイルはすべて <code>&lt;dx-tutorials&gt;/notebooks/T06-DX-Runtime/workspace</code> 配下に置かれます。

<!-- cell: 550595f9 src: caf8fba665 -->
> このチュートリアルは全 3 部のうちの第 2 部です。コースマップとすべてのパートの一覧は入門ノートブックにあります。

<!-- cell: e054aa4b-cc8a-4145-a2f1-3087227ff739 src: d38e092018 -->
## 前提条件

- チュートリアル 06-1 を完了していること。
- Python 環境と C++ ビルドのための `uv` と `cmake`(および `build-essential` の `g++`)。
- ネットワークアクセス: 第 2 節で `workspace/.venv-dxrt` 配下に 81 MB の Python 環境を作成します。`dx_engine` wheel はローカルにありますが、`uv` が依存関係の NumPy をダウンロードするため、オフラインのホストではローカルのパッケージインデックスが必要です。
- 読む時間は約 10 分を見込んでください。ダウンロードがキャッシュされていれば、セル自体は 1 分程度で実行できます。

次のセルは SDK の場所を特定し、これらの要件を確認してステータス表を表示します。`MISSING` と表示された項目には、それを提供する手順が併記されます。

<!-- cell: 1d7d338e src: 98620d01b8 -->
## 1. チュートリアルワークスペースの初期化

<!-- cell: 0478080a src: d6685c848d -->
## 2. T06 ワークスペースへの Python バインディングのインストール

DX-RT の Debian パッケージは、<code>/usr/share/libdxrt-bin/python</code> 配下にバージョン固有の wheel を提供します。Python 3.12 向けにビルドされた wheel は Python 3.13 のカーネルからはインポートできないため、このノートブックは <code>cpXY</code> タグが実行中のカーネルと一致する wheel を選択します。

生成されるファイルをすべて T06 内に収めるため、このチュートリアルは <code>workspace/.venv-dxrt</code> 配下に小さな専用環境を作成し、その site-packages ディレクトリをカーネルのモジュール検索パスの末尾に追加します。先頭ではなく末尾に追加することが重要です。この環境には wheel の依存関係として NumPy も含まれており、カーネルは引き続き自身の NumPy を使い続けなければなりません。カーネルが持っていない <code>dx_engine</code> だけがローカル環境から解決されます。コマンドは次と同等です。

~~~bash
uv venv --python <current-jupyter-python> <T06>/workspace/.venv-dxrt
uv pip install --python <T06>/workspace/.venv-dxrt/bin/python \
  /usr/share/libdxrt-bin/python/dx_engine-<version>-cp<XY>-*.whl
~~~

これは共有の Jupyter 環境を変更せず、DX-RT の再ビルドも行いません。ローカル環境にすでに一致する wheel が含まれている場合、セルを再実行しても作成とインストールはスキップされます。

<!-- cell: 3d715559 src: d07404bdbe -->
## 3. メモリを確保する前にテンソルのメタデータを読み取る

入力の shape や dtype をモデル名から推測しないでください。エンジンにその契約を問い合わせます。v3.4 以降、DX-RT は未定義動作を防ぐために NumPy の dtype を検証します。

下のヘルパーは <code>np.empty(...)</code> を使ってからバッファを埋めます。これにより実際に書き込み可能なページが確保され、DMA のピン留め時にコピーオンライトのゼロページに起因する問題を回避できます。`<DX_ALL_SUITE_DIR>/dx-runtime/dx_rt/examples/python` にある SDK の Python サンプルも同じパターンを使用しています。

<!-- cell: b3a1916a src: b039b75719 -->
### 3.1 テンソル所有権のチェックリスト

| 確認項目 | 安全な方法 |
|---|---|
| Shape | <code>get_input_tensors_info()</code> から確保する |
| Dtype | 返された NumPy の dtype をそのまま使う |
| レイアウト | コンパイル済みモデルの可視レイアウトに合わせる |
| 連続性 | C 連続の配列を渡す |
| ライフタイム | 非同期の入力は完了まで生存させておく |
| 出力のスコープ | コールバックの出力は、コールバックより長く必要な場合にのみコピーする |

<!-- cell: 62cafeb6 src: 658cf437ee -->
## 4. 同期推論

<code>run()</code> は結果が準備できるまで呼び出し元スレッドをブロックします。逐次処理やレイテンシ重視のアプリケーションにとって、最も分かりやすい出発点です。

<img src="assets/dx-rt-execution-modes.svg" style="max-width: 1100px; width: 100%;" alt="同期、非同期、バッチの実行モード">

<!-- cell: 3e183092 src: 726f414385 -->
4 つの計測値はそれぞれ異なる問いに答えます。

- **ホスト側で観測した時間**には、Python 呼び出しとその周辺のホスト側オーバーヘッドが含まれます。
- **E2E 応答時間**は、DX-RT が記録した直近のリクエストの、投入から結果までの時間で、NPU バッファの待ち時間とキューの待ち時間を含みます。DX-RT v3.5.0 以降、`get_latency()` はこれと同じ値を返します。
- **コアレイテンシ**は、E2E 応答時間からバッファプールとキューの待ち時間を除いたものです(`get_core_latency()`、v3.5.0 で新規追加)。両者の差が大きい場合、リクエストは NPU 上で遅く実行されているのではなく、ランタイム内で待たされていることを意味します。
- **NPU 推論時間**は NPU の実行をカバーし、エンドツーエンドのレイテンシの一部にすぎません。

NPU 時間をカメラから表示までのレイテンシとして扱わないでください。

<!-- cell: real-image-heading src: 404e895e5a -->
### 4.1 実際の画像を分類する

上のダミー入力は、呼び出しが動作することを証明するだけで、結果が正しいことは証明しません。次のセルでは、同じ `run()` を実際の写真に対して実行し、出力を ImageNet のラベルに変換します。これは DX-RT を中心とした最小限のアプリケーション、つまり前処理、推論、後処理です。

前処理は第 3 節のモデルの契約によって決まります。

| 契約項目 | `resnet50_224x224.dxnn` での値 | アプリケーションが行うこと |
|---|---|---|
| 入力 | `[1, 224, 224, 3]`、`uint8` | 画像を 224x224 にリサイズし、8 ビットの値と HWC の順序を保ったまま、バッチ軸を追加する |
| 色の順序 | RGB | OpenCV は BGR で読み込むため、BGR を RGB に変換する |
| 正規化 | DXNN にコンパイル済み | 何もしない: `uint8` 入力であることから、モデルが内部で正規化していることが分かる |
| 出力 | `[1, 1000]`、`float32` のクラススコア | softmax を適用し、上位 5 クラスを取り出す |

これらは DX-APP の ResNet50 サンプル(`SimpleResizePreprocessor` と `ClassificationPostprocessor`)と同じ手順です。画像はこのサンプルの DX-APP 既定のサンプル画像で、ImageNet のラベル一覧も DX-APP(`src/python_example/common/utility/labels.py`)から取得しています。

<!-- cell: real-image-notes src: e33455d041 -->
結果の読み方:

- 写真には柴犬が写っています。ImageNet-1k には柴犬のクラスがないため、確率は最も近いスピッツ系の犬種(ハスキー、ノルウェジアン・エルクハウンド、シベリアン・ハスキーなど)に分散します。このような分散はラベルセットの性質によるもので、ランタイムのエラーではありません。
- DX-APP の `resnet50_224x224_sync.py` は、この画像に対して同じ top-5 を出力します。自分のパイプラインと参照アプリケーションの結果が一致すれば、前処理と出力のマッピングは正しいと言えます。
- 色の順序が誤っていたりリサイズが抜けていたりしても、エラーなく実行され、異なるラベルが出るだけです。パイプラインを信頼する前に、ラベル付きデータで精度を確認してください。

<!-- cell: 54f7a81f src: bb3c7d9b98 -->
## 5. ジョブ ID を使った非同期推論

<code>run_async()</code> はジョブ ID を返します。その後 <code>wait(job_id)</code> が対応する出力を返します。これにより、アプリケーションは送信、NPU の実行、その他の処理を重ね合わせることができます。

<img src="assets/dx-rt-buffer-lifecycle.svg" style="max-width: 1100px; width: 100%;" alt="非同期バッファの所有権のライフサイクル">

<!-- cell: d0e724c4 src: cbe255be5a -->
### 5.1 wait とコールバックの比較

| 完了の受け取り方 | 強み | 主な責務 |
|---|---|---|
| <code>run_async()</code> + <code>wait(job_id)</code> | リクエストと結果の明示的な対応付け | ジョブ ID を保持し、適切なスレッドから wait する |
| 登録したコールバック | 低レイテンシの完了処理 | コールバックを高速かつスレッドセーフに保つ |
| <code>run()</code> | 最も単純な制御フロー | 呼び出し元スレッドがブロックされることを受け入れる |

コールバックの出力はコールバックのスコープ内でのみ有効です。後続処理でそれらを保持する必要がある場合は、必要なデータをコールバック内でコピーし、重い処理は別のキューに移してください。

<!-- cell: callback-intro src: eb74f2ee1e -->
次のセルではコールバック方式を使います。`register_callback()` は、各 `run_async()` ジョブが完了したときに DX-RT が自身の worker thread 上で呼び出す関数を設定します。`user_arg` はジョブとともに渡されるため、コールバックはどのリクエスト(たとえばフレーム番号)が完了したかを知ることができます。このセルは次のことを行います。

1. コールバックを登録する
2. 4.1 節の実際の画像を使って 8 個のジョブを投入する。各ジョブの `user_arg` にはそのインデックスを渡す
3. 最後のジョブの後にコールバックがセットするイベントを待つ
4. `register_callback(None)` でコールバックを解除する

コールバックは共有データを変更する前にロックを取得し、必要な小さな結果(top-1 ラベル)だけを保持して、`0` を返します。重い後処理は別のスレッドやキューで行い、ランタイムの worker thread をブロックしないようにします。これは DX-APP の `AsyncRunner` が基盤としているパターンです。

<!-- cell: af9ec43d src: 35292b89d9 -->
## 6. バッチ API

DX-RT のバッチ実行は、複数の独立したサンプルをまとめてランタイム内部で非同期にスケジュールします。コンパイル済みモデルのバッチ次元は変更**しません**。DXNN モデルは通常どおりバッチサイズ 1 のままです。現在の API はバッチ形式の入力と明示的な出力バッファを伴う `run()` を使用し、`run_batch()` は非推奨の互換ラッパーとしてのみ残されています。

ネストした Python の形式は次のとおりです。

~~~python
[
    [sample_0_input_0],
    [sample_1_input_0],
    [sample_2_input_0],
]
~~~

<!-- cell: 4b77e8c1 src: 3d13107715 -->
## 7. バッファ数はパイプラインの容量

<code>InferenceOption.buffer_count</code> は内部の推論バッファの数を制御します。値を大きくすると同時に処理中の作業を増やせますが、その分メモリを消費し、パイプラインが満杯になった後は効果がなくなることがあります。

| 小さすぎる | バランスが取れている | 大きすぎる |
|---|---|---|
| 送信側がバッファ待ちになる | NPU を稼働させ続けるのに十分な作業量 | メモリが増えるだけで効果は小さい |
| スループットが低下する可能性がある | 安定したスループットと上限のあるメモリ | キューが長くなりレイテンシが増える可能性がある |

最大値が最良だと決めつけず、代表的な値を測定してください。上級編チュートリアルでは、バッファ数を制御した実験を行います。

<!-- cell: 485e925a src: fcff455a41 -->
## 8. 最小限の C++14 アプリケーションのビルド

v3.4 以降、DX-RT は安定した C ABI とヘッダーオンリーの C++14 ラッパーを提供します。新しい C++ コードでインクルードするのは次のものだけにしてください。

~~~cpp
#include <dxrt/dxrt_cxx_api.h>
~~~

<code>dxrt_api.h</code> と <code>dxrt_cxx_api.h</code> を同じ翻訳単位でインクルードしないでください。

DX-RT は CMake のパッケージ構成(<code>/usr/local/lib/cmake/dxrt/dxrtConfig.cmake</code>)をインストールするため、プロジェクトに必要なのは <code>find_package(dxrt REQUIRED)</code> とインポートされたターゲット <code>dxrt::dxrt</code> だけです。このターゲットはインクルードディレクトリと <code>pthread</code> への依存関係を持っています。デモチュートリアル(20〜24)では同じ検索処理を小さな CMake ヘルパーでラップしていますが、以下の行はその素の形です。

次の 2 つのセルは、<code>T06-DX-Runtime/workspace/cpp</code> 配下にのみソースファイルを作成します。

<!-- cell: 78076ec1 src: 8d9e43b968 -->
### 8.1 構成とビルド

これらのノートブックセルは、ターミナルで入力するのと同じコマンドを実行します。

~~~bash
cmake -S <T06>/workspace/cpp -B <T06>/workspace/cpp/build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build <T06>/workspace/cpp/build --parallel
~~~

ビルドディレクトリは T06 の中に収まります。

<!-- cell: 1d3ac8cb src: 98e8643d77 -->
### 8.2 C++ アプリケーションの実行

同等のターミナルコマンドは次のとおりです。

~~~bash
<T06>/workspace/cpp/build/dxrt_sync \
  <T06>/workspace/models/resnet50_224x224.dxnn
~~~

C++ での非同期、コールバック、マルチスレッド、複数入力、プロファイラーのプログラムについては、`<DX_ALL_SUITE_DIR>/dx-runtime/dx_rt/examples/cpp` にある SDK サンプル(たとえば `run_async_model`、`run_async_model_thread`、`run_batch_model`、`multi_input_model_inference`)から始めてください。

<!-- cell: 68c92258 src: 34a1319a55 -->
## 9. 実行スタイルの選択

| 要件 | 推奨される出発点 | 理由 |
|---|---|---|
| 最も単純な逐次フロー | 同期 <code>run()</code> | 所有権とエラー処理が明確 |
| 単一リクエストの最小レイテンシの調査 | 同期 <code>run()</code> | 意図的なキュー深さがない |
| より高いストリーミングスループット | 非同期 + <code>wait()</code> またはコールバック | 独立したパイプライン処理を重ね合わせる |
| 複数の独立したサンプルを一度に処理 | バッチ API | 送信と完了をまとめる |
| 密な統合と低い Python オーバーヘッド | C++ API | C++14 アプリケーションを直接制御 |
| 既存の Python パイプライン | Python API | NumPy データとの迅速な統合 |

正しい選択はアプリケーション全体によって決まります。キューが深くなりすぎると、推論 FPS が高くてもユーザーから見たレイテンシはかえって悪化することがあります。

<!-- cell: 3fd7908b-0ab7-49ad-a0a4-2513fcb1771d src: 9921b5ad10 -->
## 10. トラブルシューティング

| 症状 | 表示される内容 | 考えられる原因 | 対処 |
|---|---|---|---|
| `uv` または `cmake` がない | `[MISSING] uv` または `[MISSING] cmake` | インストールされていない | `curl -LsSf https://astral.sh/uv/install.sh | sh`、`sudo apt install cmake build-essential` |
| この Python 向けの wheel がない | `No dx_engine wheel for cp3XY` | カーネルの Python バージョンに一致する wheel が `/usr/share/libdxrt-bin/python` 配下にない | サポートされている Python バージョンでチュートリアルの `.venv` を作成する |
| `import dx_engine` が失敗する | インポート時のバージョン不一致に関する `RuntimeError`、または `ImportError` | DX-RT v3.5.0 以降、wheel の major.minor バージョンがインストール済みの `libdxrt.so` と異なると `import dx_engine` は停止する。`workspace/.venv-dxrt` 内の wheel は古いランタイムのものだった | 第 2 節を再実行して一致する wheel をインストールする。`DX_ENGINE_SKIP_VERSION_CHECK=1` でチェックをスキップできるが、診断目的でのみ使用する |
| コールバックが届かない | `TimeoutError: Only N of 8 callbacks arrived` | コールバック内で例外が発生した、またはジョブの完了前にコールバックが解除された | コールバックは短く、例外を出さないようにする。`register_callback(None)` での解除は最後のジョブの後にのみ行う |
| CMake が DX-RT を見つけられない | `Could not find a package configuration file provided by "dxrt"` | DX-RT がカスタムプレフィックスにインストールされており、`dxrtConfig.cmake` が既定の検索パスにない | `cmake -S ... -B ...` の行に `-DCMAKE_PREFIX_PATH=<prefix>` を追加する |
| 非同期の出力がおかしい | `wait()` の後にゴミの値が入っている | `wait()` が返る前に入力バッファが解放または再利用された | `wait()` が返るまで入力配列を生存させておく |

<!-- cell: b69fd536 src: 5b51c96421 -->
## 11. まとめ

### 11.1 完了した API ワークフロー

**テンソルのメタデータを読み取る**  
→ **正確な dtype と shape で確保する**  
→ **同期で実行する**  
→ **実際の画像を分類する**  
→ **非同期で送信して wait する、またはコールバックで完了する**  
→ **独立したサンプルをまとめる**  
→ **同じフローを C++ で構築する**

<img src="assets/dx-rt-execution-modes.svg" style="max-width: 1000px; width: 100%;" alt="DX-RT の実行モード">

### 11.2 実行ダッシュボード

| モード | 送信 | 完了 | 最初に見るべき指標 |
|---|---|---|---|
| 同期 | 1 リクエスト | <code>run()</code> からの復帰 | リクエストのレイテンシ |
| 非同期 + wait | 複数のジョブ ID | <code>wait(job_id)</code> | スループットとキュー遅延 |
| コールバック | 複数のリクエスト | ランタイムのコールバック | コールバックのコストとスループット |
| バッチ | サンプルのグループ | 出力のグループ | 実効サンプル数/秒 |
| C++ | 同じランタイムの概念 | C++ のテンソル | エンドツーエンドのアプリケーションコスト |

### 11.3 完了チェックリスト

- [ ] Jupyter の Python ABI に一致する wheel をインストールした
- [ ] ランタイムのメタデータから連続したテンソルを確保した
- [ ] 同期実行のホスト、ランタイム、NPU の時間を測定した
- [ ] 実際の画像を分類し、DX-APP サンプルの結果と一致させた
- [ ] `user_arg` を使ったスレッドセーフなコールバックで非同期ジョブを完了した
- [ ] <code>wait()</code> まで非同期入力のライフタイムを維持した
- [ ] モデルのバッチサイズを変えずにバッチ API を使用した
- [ ] <code>buffer_count</code> のメモリとスループットのトレードオフを説明できる
- [ ] C++14 の DX-RT アプリケーションをビルドして実行した
- [ ] ダミー入力を製品の前処理に置き換え、出力を検証した

> **覚えておくこと:** バッファの所有権は正しさの一部です。性能チューニングは、shape、dtype、レイアウト、ライフタイム、出力のマッピングを検証した後に行います。

### 11.4 SDK サンプル

SDK には、このノートブックの各 API について実行可能なサンプルが付属しており、独自のコードを書く際の出発点として適しています。

| 言語 | ディレクトリ | サンプル |
|---|---|---|
| Python | `<DX_ALL_SUITE_DIR>/dx-runtime/dx_rt/examples/python` | `run_sync_model.py`, `run_async_model_wait.py`, `run_async_model_callback.py`, `run_async_model_thread.py`, `run_batch_model.py`, `run_sync_model_bufcount.py`, `multi_input_model_inference.py` |
| C++ | `<DX_ALL_SUITE_DIR>/dx-runtime/dx_rt/examples/cpp` | `run_sync_model`, `run_async_model_wait`, `run_async_model_thread`, `run_batch_model`, `run_async_model_bufcount`, `multi_input_model_inference` |

### 11.5 次のステップ

**上級編チュートリアル**に進み、デバイスとコアのバインド、個々のステージのプロファイリング、デバイスの健全性の監視、メモリロード型モデルと複数入力モデルのテスト、再現可能なリリース証跡の作成を学びましょう。
