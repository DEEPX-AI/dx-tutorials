<!-- i18n source: notebooks/T06-DX-Runtime/dx_rt_03_advanced.ipynb -->
<!-- i18n lang: ja -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 46c94efe src: bf00e17c30 -->
# DEEPX Tutorial 06-3 - DX-RT 上級編

このノートブックでは、動作している推論呼び出しを、観測可能で再現可能なランタイム実験へと発展させます。

## 学習目標

このチュートリアルを終えると、次のことができるようになります。

- デバイス選択、NPU コアバインディング、ORT の使用、バッファ数を設定する
- 制御された条件のベンチマークでバッファ数をチューニングする
- DX-RT プロファイラーのトレースを生成して可視化する
- ジョブごとの H2D、NPU、D2H、フォーマットハンドラー、CPU タスクのメトリクスを読み取る
- 変動係数(Coefficient of Variation、CoV)を使って不安定なタイミングを検出する
- デバイスのメモリ、使用率、温度状態を照会する
- メモリ上のバッファから DXNN モデルを読み込む
- 複数入力モデル向けに名前付き入力を準備する
- ランタイムイベントハンドラーを登録する
- 安定した C ABI、ヘッダーオンリーの C++ ラッパー、ランタイム IPC の境界を説明する
- NFH および CPU 演算アクセラレーションを C++、Python、`dxrun` から説明・ビルド・有効化・A/B テストする
- コンパクトなリリース検証レコードを作成する

生成されるトレース、レポート、ソースファイル、結果はすべて <code>&lt;dx-tutorials&gt;/notebooks/T06-DX-Runtime/workspace</code> 配下に保存されます。

<!-- cell: 7bedacd4 src: 215e801d3c -->
> このチュートリアルの第 3 部(全 3 部)です。コースマップと全パートの一覧は入門ノートブックにあります。

<!-- cell: ef4ee62f-6a13-4f7a-bc9d-be05af55b919 src: 5ddb64a396 -->
## 前提条件

- チュートリアル 06-2 を完了していること。特に第 2 節(`workspace/.venv-dxrt` 環境をここで再利用します)。
- DEEPX NPU。プロファイラーとベンチマークの節では実際に推論を実行します。
- 任意: チュートリアル 05-3 第 8 節の複数入力モデル。存在しない場合、そのラボはスキップされます。
- 所要時間は読む時間で約 15 分です。セル自体は 1〜2 分で実行が完了します。

次のセルは SDK を探し、これらの要件を確認してステータス表を表示します。`MISSING` と表示された項目には、それを提供する手順が示されます。

<!-- cell: 4c12f0dc src: be70bbc8b4 -->
## 1. チュートリアルワークスペースの初期化

このノートブックは、中級編チュートリアルで作成した T06 ローカル環境に、対応する <code>dx_engine</code> wheel が入っていることを前提とします。また、利用可能な場合は 2 つ目のモデルをリンクし、<code>dxbenchmark</code> で複数モデルのレポートを実演できるようにします。

<!-- cell: b8287091 src: 431918d739 -->
## 2. ランタイムリソースの制御

<code>InferenceOption</code> は、1 つのエンジンインスタンスをどこでどのように実行するかを制御します。

| オプション | 既定の意味 | 変更する場面 |
|---|---|---|
| <code>devices</code> | 空のリストは利用可能なデバイスを使用 | デバイスを分離する、またはエンジンを分散させるとき |
| <code>bound_option</code> | <code>NPU_ALL</code> | 1 コアまたはコアペアを予約するとき |
| <code>use_ort</code> | ビルドに依存 | モデルに CPU タスクが含まれるとき |
| <code>buffer_count</code> | ランタイムの既定値、通常は 6 | 測定されたキューイングやメモリの挙動からチューニングが必要なとき |

デバイスは物理的なアクセラレーターを選択します。バインドオプションは、選択した各デバイス内のコアを選択します。この 2 つのレベルを混同しないでください。

<!-- cell: 01c44f7b src: b4e173b1c2 -->
### 2.1 バインディングの安全性

既存のエンジンインスタンスがすでに個別の NPU コアを予約している場合、<code>NPU_ALL</code> で別のエンジンを作成すると、必要なコアがすべて解放されるまでブロックされることがあります。マルチスレッドおよびマルチプロセスのサービスでは、リソースの所有権を明示的に管理してください。

| 目的 | 一般的な出発点 |
|---|---|
| 1 つのエンジンで最大スループット | 1 つ以上のデバイス、<code>NPU_ALL</code> |
| 独立した 2 つのパイプラインを分離 | 異なるデバイス、または重複しないコアバインディングを割り当てる |
| シングルコアのレイテンシテストを再現 | 1 つのデバイスと 1 つのコア |
| プロセス間でハードウェアを共有 | エンジン作成前に所有権を定義する |

バインディングはデプロイ時の判断事項です。このノートブックで実行するラボは、デバイス 0 と <code>NPU_ALL</code> を使います。

<!-- cell: 0391fabe src: 9280b3bc19 -->
## 3. 制御された実験によるバッファ数のチューニング

<img src="assets/dx-rt-observability.svg" style="max-width: 1100px; width: 100%;" alt="DX-RT の観測性とチューニングのフロー">

以下の 4 つのコマンドは <code>--buffer-count</code> だけが異なります。モデル、ウォームアップ回数、実行時間、実行モードは同じです。スループットとレイテンシを比較してください。最大の値を自動的に選ばないでください。

同等のターミナルでのパターン:

~~~bash
cd <T06-DX-Runtime>/workspace
dxrun -m models/resnet50_224x224.dxnn \
      --benchmark --time 3 --warmup-runs 5 --buffer-count N
~~~

<!-- cell: 2177a506 src: 4687dde300 -->
メモリやレイテンシの制限を超えずに、必要な安定したスループットに到達する最小の値を選んでください。前処理、後処理、他のプロセスがキューへの負荷を変化させるため、実際のアプリケーション負荷の下で実験を繰り返してください。

<!-- cell: 9c4ffa77 src: 63add5ff96 -->
## 4. ランタイムタイムラインのプロファイリング

プロファイラーは、入力フォーマット変換、ホストからデバイスへの転送、NPU 計算、デバイスからホストへの転送、出力フォーマット変換、CPU タスクといったステージごとに時間を分離します。

次のコマンドは T06 のプロファイラーディレクトリ内に <code>profiler.json</code> を作成します。

~~~bash
cd <T06-DX-Runtime>/workspace/profiler
dxrun -m ../models/resnet50_224x224.dxnn \
      --benchmark --time 5 --warmup-runs 5 --profiler
~~~

<!-- cell: fb34227e src: 4811beb9fe -->
### 4.1 トレースを画像に変換する

SDK のプロット用コマンドをすぐ下に示します。<code>--auto-select</code> は安定した中央領域に注目し、出力は T06 配下に保存されます。

同等のターミナルコマンド:

~~~bash
python <DX_RT_DIR>/tool/profiler/plot.py \
  --input <T06>/workspace/profiler/profiler.json \
  --output <T06>/workspace/profiler/profiler.png \
  --auto-select
~~~

<!-- cell: 0dff6899 src: 9e9a10036d -->
### 4.2 プロファイラーイベントの解釈

| イベント | 測定対象 | 最初に確認する問い |
|---|---|---|
| Buffer Wait | 空き推論バッファの待ち時間 | キューの深さや CPU 負荷が高すぎないか? |
| NPU Input Format Handler | パディングとレイアウト変換 | フォーマット変換の割合が大きくないか? |
| PCIe Write / H2D | ホストからデバイスへの転送 | 入力サイズや転送が制限になっていないか? |
| NPU Core | NPU の計算 | モデルが全体時間の大部分を占めているか? |
| PCIe Read / D2H | デバイスからホストへの転送 | 出力が異常に大きくないか? |
| NPU Output Format Handler | 出力のスライスとレイアウト変換 | 変換がボトルネックになっていないか? |
| CPU Task Queue Wait | CPU 実行の待ち時間 | CPU パイプラインが飽和していないか? |
| cpu_N | CPU 演算子の実行 | 計算負荷の高い CPU 演算子が支配的でないか? |

NPU タスク全体には、フォーマット処理、転送、計算、出力処理が含まれます。NPU コアの計算のみよりも広い範囲を指します。

<!-- cell: 100ed5cc src: 4831be0e8a -->
## 5. Python API でジョブごとのメトリクスを読み取る

DX-RT v3.4 では <code>get_job_metrics(job_id)</code> が追加されました。<code>wait(job_id)</code> の直後に呼び出してください。従来のパフォーマンスデータ取得メソッドは非推奨です。

以下のコードはプロファイリングを有効にし、20 個の非同期ジョブを実行して、最後のジョブの有効なメトリクスを表示します。繰り返し実行したジョブは、5.1 節の CoV 表に必要なサンプル数も提供します。

<!-- cell: 381e6586 src: 6eb64f8da6 -->
### 5.1 変動係数(CoV)による安定性の評価

変動係数(Coefficient of Variation、CoV)は、標準偏差を平均で割った値です。

~~~text
CoV (%) = standard deviation / mean × 100
~~~

平均時間の異なるステージ間で、相対的なジッターを比較できます。低いほど安定していますが、普遍的なリリース基準値はありません。製品のレイテンシ予算と動作条件から基準値を定義してください。

| 結果 | 解釈 |
|---|---|
| 平均が低く、CoV が高い | 通常は高速だが、ときどき不安定 |
| 平均が高く、CoV が低い | 予測どおりに遅い |
| p99 が高く、平均は中程度 | テールレイテンシの調査が必要 |
| NPU は安定、ホスト時間は不安定 | ホストのキュー、CPU 負荷、前処理を調べる |

<!-- cell: 9994a301 src: 646fb2cbbd -->
## 6. デバイスの健全性の監視

監視サービスは共有ステータスをおよそ 1 秒に 1 回更新します。それより速くポーリングしても、通常は同じサンプルが返ります。

次のセルは 2 つのスナップショットを記録します。<code>is_valid() == False</code> は、監視データが古いか利用できないことを意味すると扱ってください。

<!-- cell: c0c43234 src: 68ad41aa24 -->
継続的に確認するには、アプリケーションの実行中に別のターミナルで <code>dxtop</code> を実行してください。温度、使用率、メモリ、スロットリングの証拠を、レイテンシおよびスループットと一緒に収集してください。冷えたシステムでの短時間のベンチマークでは、本番環境での熱的な挙動が見えないことがあります。

<!-- cell: 91656971 src: cf8de1ebce -->
## 7. メモリからモデルを読み込む

メモリからの読み込みは、モデルが暗号化ストレージ、パッケージ、ネットワークサービス、その他の管理されたデータソースから提供される場合に便利です。NumPy 配列は C 連続(C-contiguous)であり、エンジンの生存期間中は有効なまま保持されている必要があります。

<!-- cell: bf637c62 src: 8a0360883c -->
## 8. 複数入力モデルのコントラクト

複数入力モデルでは、正確なテンソル名をキーとする辞書が最も安全なインターフェースです。

~~~python
inputs = {
    "left_image": left_tensor,
    "right_image": right_tensor,
}
outputs = engine.run_multi_input(inputs)
~~~

DX-RT は順序付きリストや 1 つに連結したバッファもサポートしますが、名前付き入力を使うと順序の間違いを減らせます。

このラボは、T05 上級編の第 8 節で作成した 2 入力の DXNN が存在する場合にそれを再利用します。モデルのコンパイルや、T06 の外へのファイル作成は行いません。リンクは `workspace/models/` ではなく `workspace/multi_input/` 配下に置かれるため、第 11 節のディレクトリベンチマークは単一入力の分類モデルだけを比較し続けます。

<!-- cell: d7f33ae1 src: c7292df319 -->
## 9. ランタイムイベントとサービス統合

<code>RuntimeEventDispatcher</code> は、デバイスの警告、エラー、復旧通知、メモリイベント、スロットリングイベントを一元管理します。製品のハンドラーは高速かつスレッドセーフで、構造化されたイベントをアプリケーションのロギングやヘルスシステムに転送するべきです。

次のセルはハンドラーを登録し、経路を検証するために **チュートリアル用の合成イベント** を 1 つディスパッチします。実際のハードウェア障害をシミュレートするものではありません。

<!-- cell: 068ed27c src: 82e871f553 -->
### 9.1 ABI と IPC の境界

DX-RT v3.4 では、公開された統合インターフェースと内部実装が分離されています。

| レイヤー | 目的 | 製品向けの指針 |
|---|---|---|
| 安定した C ABI、<code>dxrt_c_api.h</code> | <code>libdxrt.so</code> のバージョン管理された C シンボル | 言語バインディングとバイナリ配布に使用する |
| ヘッダーオンリーの C++14 ラッパー、<code>dxrt_cxx_api.h</code> | C ABI 上のモダンな C++ インターフェース | 新規の C++ アプリケーションに推奨 |
| レガシーブリッジヘッダー | 既存の <code>dxrt_api.h</code> とのソース互換性 | 既存製品のビルドを維持し、計画的に移行する |
| 共有メモリ IPC とランタイムサービス | プロセスとランタイム間の効率的な通信 | 内部トランスポートとして扱い、アプリケーションのテンソル API としては扱わない |

<code>libdxrt.so</code> の非公開 C++ シンボルに依存しないでください。公開された C/C++ ヘッダーがサポート対象の統合境界です。

<!-- cell: 1dbfded8 src: 4febd54669 -->
## 10. 任意: CPU 側のアクセラレーション

これらの機能はホスト側のランタイム処理を高速化します。DXNN グラフを変更したり、NPU の計算レイヤーを高速化したりはしません。プロファイラーが対応するボトルネックを特定した後にのみ有効にしてください。

| 機能 | 高速化される処理 | x86_64 の実装 | aarch64 の実装 | 有効な場面 |
|---|---|---|---|---|
| `NPU_FORMAT_CONVERSION_ACCELERATION` | NPU Format Handler(NFH): NPU タスク前後の転置、パディング、スライス、デバイスレイアウト変換 | Intel IPP | ARM NEON/ASIMD | 入力または出力のフォーマットハンドラー時間が大きいとき |
| `CPU_OP_ACCELERATION` | CPU フォールバックサブグラフ内の ONNX Runtime CPU 演算子 | OpenVINO Execution Provider | XNNPACK Execution Provider | ORT の CPU 時間が Conv や MatMul などの計算負荷の高い演算子に支配されているとき |

NFH アクセラレーションはアプリケーションの前処理を置き換える**ものではありません**。リサイズ、色変換、正規化、その他モデル固有の処理は引き続きモデルのコントラクトに従います。CPU 演算アクセラレーションは CPU 演算子を NPU に移す**ものではなく**、より最適化された CPU 実行プロバイダーを選択するものです。

### 10.1 2 つのゲート: ビルドサポートとランタイムでのオプトイン

両方のゲートが開いている必要があります。

| ゲート | 目的 | 既定値 |
|---|---|---|
| ビルド時の CMake オプション | 機能とそのプラットフォームライブラリを DX-RT にコンパイルして組み込む | `OFF` |
| ランタイム設定 | コンパイル済みの機能をプロセスに対して有効化する | `OFF` |

ビルドサポートがない場合、C++ の enum、Python の enum、対応する `dxrun` オプションは存在しません。ランタイムでのオプトインだけでは、欠けている実装を追加することはできません。

### 10.2 アクセラレーションサポート付きで DX-RT をビルドする

DX-RT のソースツリーで `<DX_RT_DIR>/cmake/dxrt.cfg.cmake` を編集し、次の 2 つのオプションだけを変更します。

~~~cmake
option(USE_NPU_FORMAT_CONVERSION_ACCELERATION
       "Accelerate NPU data format conversion (transpose/padding)" ON)
option(USE_CPU_OP_ACCELERATION
       "Accelerate CPU-side ONNX operations" ON)
~~~

その後、CMake が必要なライブラリを再検出できるようにクリーンビルドを実行します。

~~~bash
cd <DX_RT_DIR>
./build.sh --clean
~~~

`CPU_OP_ACCELERATION` は最適化された ONNX Runtime 実行プロバイダーを選択するため、ORT が有効な DX-RT ビルドも必要です。クリーンビルドではプラットフォームの依存関係がダウンロードまたはインストールされることがあります。必要なライブラリやプラットフォーム要件が利用できない場合、ビルドは要求された機能を無効化します。`ON` が成功したと決めつけず、CMake の出力を確認してください。新しいランタイムをインストールした後は、アプリケーションが使用する対応バージョンの `dx_engine` wheel を再インストールしてください。

<!-- cell: ac100003 src: 7d2752ab1c -->
### 10.3 インストール済みランタイムが機能を公開していることを確認する

Python wheel とシステムの CLI を別々に確認してください。同じバージョンを報告していても、ビルド時の機能が異なる場合があります。次のセルは読み取り専用です。一貫したデプロイでは、製品が使用するすべてのインターフェースで必要な機能が公開されているべきです。

> **標準的なインストールで表示される内容:** パッケージ化された `dx_engine` wheel は両方の機能を `available` と報告する一方で、システムの `dxrun` は `No acceleration options` と表示します。これは想定どおりの動作であり、エラーではありません。2 つのバイナリは異なる CMake オプションでビルドされており(10.1 節)、Python の enum メンバーはその機能付きでビルドされた wheel にのみ存在します。このセルはまさにこの不一致を明らかにするためのものです。

<!-- cell: ac100005 src: 324af8acdf -->
### 10.4 アプリケーションで機能を有効にする

最初の `InferenceEngine` を構築する**前に**アクセラレーションを設定してください。DXNN に CPU タスクが含まれる場合は ORT を有効のままにしてください。

**C++**

~~~cpp
#include <dxrt/dxrt_cxx_api.h>

int main()
{
    auto& config = dxrt::Configuration::GetInstance();

#ifdef DXRT_NFH_ACCELERATION_AVAILABLE
    config.SetEnable(
        dxrt::Configuration::ITEM::NFH_ACCELERATION, true);
#endif

#ifdef DXRT_CPU_OP_ACCELERATION_AVAILABLE
    config.SetEnable(
        dxrt::Configuration::ITEM::CPU_OP_ACCELERATION, true);
#endif

    dxrt::InferenceOption option;
    option.useORT = true;  // Required only when the graph has CPU tasks.
    dxrt::InferenceEngine engine("model.dxnn", &option);
    // Prepare input and run inference.
}
~~~

プリプロセッサーのガードにより、インストール済みの DX-RT ヘッダーにアクセラレーション機能が存在しない場合でもソースをビルドできます。製品によっては、機能の欠如を設定エラーとして扱う方針も取れます。

**Python**

~~~python
from dx_engine import Configuration, InferenceEngine, InferenceOption

config = Configuration()
required_items = ("NFH_ACCELERATION", "CPU_OP_ACCELERATION")
missing = [name for name in required_items if not hasattr(Configuration.ITEM, name)]
if missing:
    raise RuntimeError(f"DX-RT was built without: {', '.join(missing)}")

config.set_enable(Configuration.ITEM.NFH_ACCELERATION, True)
config.set_enable(Configuration.ITEM.CPU_OP_ACCELERATION, True)

option = InferenceOption()
option.use_ort = True  # Required only when the graph has CPU tasks.
with InferenceEngine("model.dxnn", option) as engine:
    # Prepare input and run inference.
    pass
~~~

Python wheel は、新しくインストールしたランタイムのバージョンと Python ABI に一致している必要があります。一致しない場合、enum やネイティブ拡張が共有ライブラリと整合しないことがあります。

<!-- cell: ac100006 src: 1c4141f606 -->
### 10.5 `dxrun` でテストする

まず `dxrun --help` に `--accel-nfh` と `--accel-cpu` が表示されることを確認します。次に、CPU タスクを含む DXNN モデルで A/B 比較を実行します。モデル、ORT 設定、実行時間、ウォームアップ、バッファ数、デバイスバインディング、システム負荷は同一に保ってください。

~~~bash
MODEL=/path/to/model-with-cpu-tasks.dxnn

# 1. Baseline
dxrun -m "$MODEL" --use-ort --benchmark --time 10 --warmup-runs 10 --buffer-count 6

# 2. Accelerate only NPU format conversion
dxrun -m "$MODEL" --use-ort --benchmark --time 10 --warmup-runs 10 --buffer-count 6 --accel-nfh

# 3. Accelerate only ORT CPU operators
dxrun -m "$MODEL" --use-ort --benchmark --time 10 --warmup-runs 10 --buffer-count 6 --accel-cpu

# 4. Enable both features
dxrun -m "$MODEL" --use-ort --benchmark --time 10 --warmup-runs 10 --buffer-count 6 --accel-nfh --accel-cpu
~~~

ステージレベルの証拠が必要な場合は、短めの診断用の実行に `--profiler` を追加してください。スループットだけでは、NFH、CPU 演算子、転送、NPU 計算のどれが変化したかは分かりません。

| 結果 | 解釈 |
|---|---|
| NFH 時間が減少 | フォーマット変換アクセラレーションが機能している |
| CPU タスク時間が減少 | 最適化された ORT 実行プロバイダーが効果を出している |
| FPS が変わらない | 別のステージがボトルネックか、高速化された処理が小さすぎる |
| レイテンシや CPU 使用率が悪化 | このワークロードでは機能を無効にし、ベースラインを維持する |

`CPU_OP_ACCELERATION` は主に計算負荷の高い CPU 処理に有効です。Reshape、Transpose、Concat はメモリ帯域に律速されることが多く、改善がほとんど見られない場合があります。どちらの機能も性能向上を保証するものではありません。

### 10.6 関連オプション: 動的 CPU スレッディング

プロファイラーが、個々の高コストな演算子ではなく CPU タスクキューの負荷を示している場合は、動的 CPU スレッディングを別途テストしてください。

~~~bash
export DXRT_DYNAMIC_CPU_THREAD=ON
dxrun -m "$MODEL" --use-ort --benchmark --time 10 --warmup-runs 10
~~~

これはもう 1 つの独立した A/B 実験です。NFH や CPU 演算アクセラレーションの代わりになるものではありません。

<!-- cell: f43487a0 src: a4a30ab9f0 -->
## 11. 複数モデルのベンチマーク

<code>dxbenchmark</code> はディレクトリ内の DXNN ファイルを見つけ、機械可読なレポートと視覚的なレポートを作成します。結果のパスは T06 内です。実行のたびにタイムスタンプ付きの <code>DXBENCHMARK_&lt;date&gt;.csv/.html/.json</code> セットが追加され、結果ディレクトリに <code>profiler.json</code> も書き込まれます。きれいな状態で比較したい場合は古いファイルを削除してください。このディレクトリには第 1 節でリンクした単一入力モデルだけが含まれ、第 8 節の 2 入力モデルは意図的に <code>workspace/multi_input/</code> に置かれています。

同等のターミナルコマンド:

~~~bash
cd <T06>/workspace/reports/dxbenchmark
dxbenchmark --dir <T06>/workspace/models \
            --result-path <T06>/workspace/reports/dxbenchmark \
            --time 3 \
            --warmup 3 \
            --sort fps \
            --order desc
~~~

<!-- cell: 669618a2 src: 39da7d0246 -->
## 12. リリース検証レコードの作成

<img src="assets/dx-rt-release-loop.svg" style="max-width: 1100px; width: 100%;" alt="本番ランタイム検証ループ">

ランタイムのリリースレコードは、正確なバイナリと環境を、測定された証拠と結び付けるべきです。次のセルは T06 内にコンパクトな JSON レコードを書き込みます。実際のリリース判断の前に、製品の精度、エンドツーエンドのレイテンシ、ホスト CPU、メモリ、熱的な結果を追加してください。

<!-- cell: b2f25d6a-8d4f-4316-9883-6a704948cc37 src: 921089640f -->
## 13. トラブルシューティング

| 症状 | 表示される内容 | 考えられる原因 | 対処 |
|---|---|---|---|
| DX-RT の Python 環境がない | `Complete Intermediate Section 2 first` | チュートリアル 06-2 の第 2 節が実行されていない | 一度実行する。このノートブックは `workspace/.venv-dxrt` を再利用する |
| `profiler.json` がない | `workspace/profiler/profiler.json` に対する `FileNotFoundError` | `dxrun --profiler` が実行されていない | まず第 4 節を実行する |
| 複数入力ラボがスキップされる | `Multi-input execution lab skipped` | チュートリアル 05-3 第 8 節のモデルが存在しない | 任意。その節を実行するか、そのまま進む |
| Python と `dxrun` が異なるアクセラレーションサポートを報告する | Python では `available`、`dxrun` では `No acceleration options` | 2 つのバイナリが異なるオプションでビルドされている | デプロイに使う方を信頼する。必要なオプションで DX-RT を再ビルドする(第 10 節) |
| `dxbenchmark --warmup` が無視されているように見える | ウォームアップ回数が一致しない | `--warmup` は回数ではなく秒数 | `--warmup <seconds>` を使う。`--warmup-runs` は `dxrun` のオプション |

<!-- cell: 703286a9 src: d75aea9e39 -->
## 14. まとめ

### 14.1 完了した本番ワークフロー

**コントラクトとワークロードを固定する**  
→ **ベースラインを測定する**  
→ **各ランタイムステージをプロファイリングする**  
→ **リソースを 1 つずつチューニングする**  
→ **安定性とデバイスの健全性を監視する**  
→ **再現可能なリリース証拠を保存する**

<img src="assets/dx-rt-release-loop.svg" style="max-width: 1000px; width: 100%;" alt="DX-RT 本番検証ループ">

### 14.2 上級制御ダッシュボード

| 判断事項 | まず確認する証拠 | 制御手段 |
|---|---|---|
| キューの深さ | Buffer Wait、レイテンシ、メモリ | <code>buffer_count</code> |
| リソース配置 | デバイス/コアの使用率 | <code>devices</code>、<code>bound_option</code> |
| CPU フォールバック | タスクグラフと CPU タスク時間 | <code>use_ort</code> |
| フォーマット変換 | 入力/出力のフォーマットハンドラー時間 | NFH アクセラレーション(コンパイル済みの場合) |
| CPU 演算子のコスト | CPU タスクの種類と所要時間 | CPU 演算アクセラレーション(コンパイル済みの場合) |
| CPU キューの負荷 | キュー待ち時間とホスト CPU | 動的 CPU スレッディング |
| サービスの健全性 | ランタイムイベントとデバイススナップショット | イベントハンドラーと監視ポリシー |

### 14.3 生成された証拠

| 成果物 | 場所 | 目的 |
|---|---|---|
| プロファイラー JSON | <code>workspace/profiler/profiler.json</code> | 生のイベントタイムライン |
| プロファイラー画像 | <code>workspace/profiler/profiler*.png</code> | 視覚的なボトルネックの確認 |
| ベンチマークレポート | <code>workspace/reports/dxbenchmark/DXBENCHMARK_*.csv/.html/.json</code>(実行ごとに 1 セット)と <code>profiler.json</code> | 複数モデルの比較 |
| リリースレコード | <code>workspace/reports/release_validation.json</code> | 追跡可能なリリースチェックリスト |

### 14.4 完了チェックリスト

- [ ] デバイス、コアバインディング、ORT、バッファ数を設定した
- [ ] 1 つの固定ワークロードで複数のバッファ数を比較した
- [ ] プロファイラーのトレースを生成して可視化した
- [ ] ジョブごとのステージメトリクスを読み取った
- [ ] CoV をタイミング安定性の指標として使用した
- [ ] メモリ、使用率、クロック、温度状態を照会した
- [ ] メモリから DXNN モデルを読み込んだ
- [ ] 名前付きの複数入力推論パスを準備した
- [ ] 合成イベントでランタイムイベントハンドラーを検証した
- [ ] アクセラレーションのビルドサポートとランタイムでのオプトインを区別した
- [ ] NFH および CPU 演算アクセラレーションを C++、Python、`dxrun` の制御手段に対応付けた
- [ ] T06 配下にベンチマークとリリースレコードの成果物を作成した
- [ ] 実際の前処理と後処理を含めて同じテストを実行した
- [ ] 代表的なラベル付きデータでタスク精度を検証した
- [ ] デプロイ先ホストで熱的ソークテストとテールレイテンシテストを実行した

> **覚えておくこと:** プロファイラーが特定したステージを最適化してください。単独の NPU 結果が速くなっても、精度、エンドツーエンドのレイテンシ、CPU 負荷、メモリ、安定性、熱的な挙動を一緒に検証するまではリリース結果とは言えません。

> **次へ:** デモチュートリアル(Python の OCR パイプラインはチュートリアル 10、C++ アプリケーションはチュートリアル 20〜24)に進み、ランタイムを完全なアプリケーションに適用してみましょう。
