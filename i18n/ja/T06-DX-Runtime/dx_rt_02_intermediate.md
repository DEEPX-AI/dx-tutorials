<!-- i18n source: notebooks/T06-DX-Runtime/dx_rt_02_intermediate.ipynb -->
<!-- i18n lang: ja -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 20e090c0 src: bc1aa2e5f2 -->
# DEEPX Tutorial 06-2 - DX-RT 中級編

このノートブックでは、CLI による検証から一歩進み、DX-RT の Python API と C++ API を使ったアプリケーションレベルの推論を扱います。

## 学習目標

このチュートリアルを終えると、次のことができるようになります。

- 対応するビルド済みの <code>dx_engine</code> wheel を Jupyter カーネルにインストールする
- テンソルのメタデータを読み取り、要求される shape と dtype で入力を確保する
- 同期推論を実装する
- ジョブ ID と <code>wait()</code> を使った非同期推論を実装する
- コールバックとバッファのライフタイムに関するルールを説明する
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

<!-- cell: 3d715559 src: 463872b41b -->
## 3. メモリを確保する前にテンソルのメタデータを読み取る

入力の shape や dtype をモデル名から推測しないでください。エンジンにその契約を問い合わせます。DX-RT v3.4 は未定義動作を防ぐために NumPy の dtype を検証します。

下のヘルパーは <code>np.empty(...)</code> を使ってからバッファを埋めます。これにより実際に書き込み可能なページが確保され、DMA のピン留め時にコピーオンライトのゼロページに起因する問題を回避できます。

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

<!-- cell: 3e183092 src: de13644a11 -->
3 つの計測値はそれぞれ異なる問いに答えます。

- **ホスト側で観測した時間**には、Python 呼び出しとその周辺のホスト側オーバーヘッドが含まれます。
- **DX-RT レイテンシ**は、ランタイムが記録した直近のリクエストのレイテンシです。
- **NPU 推論時間**は NPU の実行をカバーし、エンドツーエンドのレイテンシの一部にすぎません。

NPU 時間をカメラから表示までのレイテンシとして扱わないでください。

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

<!-- cell: 485e925a src: f6386e2aa5 -->
## 8. 最小限の C++14 アプリケーションのビルド

DX-RT v3.4 は安定した C ABI とヘッダーオンリーの C++14 ラッパーを提供します。新しい C++ コードでインクルードするのは次のものだけにしてください。

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

<!-- cell: 1d3ac8cb src: d3fa81e622 -->
### 8.2 C++ アプリケーションの実行

同等のターミナルコマンドは次のとおりです。

~~~bash
<T06>/workspace/cpp/build/dxrt_sync \
  <T06>/workspace/models/resnet50_224x224.dxnn
~~~

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

<!-- cell: 3fd7908b-0ab7-49ad-a0a4-2513fcb1771d src: 04785bad99 -->
## 10. トラブルシューティング

| 症状 | 表示される内容 | 考えられる原因 | 対処 |
|---|---|---|---|
| `uv` または `cmake` がない | `[MISSING] uv` または `[MISSING] cmake` | インストールされていない | `curl -LsSf https://astral.sh/uv/install.sh | sh`、`sudo apt install cmake build-essential` |
| この Python 向けの wheel がない | `No dx_engine wheel for cp3XY` | カーネルの Python バージョンに一致する wheel が `/usr/share/libdxrt-bin/python` 配下にない | サポートされている Python バージョンでチュートリアルの `.venv` を作成する |
| `import dx_engine` が失敗する | `ImportError` またはバージョン不一致のメッセージ | `workspace/.venv-dxrt` 内の wheel が `libdxrt.so` より古い | 第 2 節を再実行して wheel を再インストールする |
| CMake が DX-RT を見つけられない | `Could not find a package configuration file provided by "dxrt"` | DX-RT がカスタムプレフィックスにインストールされており、`dxrtConfig.cmake` が既定の検索パスにない | `cmake -S ... -B ...` の行に `-DCMAKE_PREFIX_PATH=<prefix>` を追加する |
| 非同期の出力がおかしい | `wait()` の後にゴミの値が入っている | `wait()` が返る前に入力バッファが解放または再利用された | `wait()` が返るまで入力配列を生存させておく |

<!-- cell: b69fd536 src: 8d0b95c249 -->
## 11. まとめ

### 11.1 完了した API ワークフロー

**テンソルのメタデータを読み取る**  
→ **正確な dtype と shape で確保する**  
→ **同期で実行する**  
→ **非同期で送信して wait する**  
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
- [ ] <code>wait()</code> まで非同期入力のライフタイムを維持した
- [ ] モデルのバッチサイズを変えずにバッチ API を使用した
- [ ] <code>buffer_count</code> のメモリとスループットのトレードオフを説明できる
- [ ] C++14 の DX-RT アプリケーションをビルドして実行した
- [ ] ダミー入力を製品の前処理に置き換え、出力を検証した

> **覚えておくこと:** バッファの所有権は正しさの一部です。性能チューニングは、shape、dtype、レイアウト、ライフタイム、出力のマッピングを検証した後に行います。

### 11.4 次のステップ

**上級編チュートリアル**に進み、デバイスとコアのバインド、個々のステージのプロファイリング、デバイスの健全性の監視、メモリロード型モデルと複数入力モデルのテスト、再現可能なリリース証跡の作成を学びましょう。
