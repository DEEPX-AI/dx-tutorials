<!-- i18n source: notebooks/T05-DX-Compiler/dx_com_03_advanced.ipynb -->
<!-- i18n lang: ja -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: advanced-title src: ca77fa067f -->
# DEEPX Tutorial 05-3 - DX-COM 上級編

このチュートリアルでは、精度回復、量子化診断、再利用可能な QXNN チェックポイント、Quantization-Aware Training(QAT)、そして `dx_com` Python API によるプログラムからのコンパイルを扱います。

最初のラボではワークフローに集中できるように SqueezeNet を使用します。最後のラボでは、Python API が必要になるケースを示すために、コンパクトな 2 入力ステレオ融合モデルを使用します。

> **始める前に:** まず入門と中級のチュートリアルを完了してください。コンパイルには数分かかることがあります。各コンパイルセルは既存の DXNN を確認し、重複した作業をスキップします。

<!-- cell: ac85dc95-b527-4b0a-8cad-e82a0d512e16 src: 215e801d3c -->
> このチュートリアルの第 3 部(全 3 部)です。コースマップと全パートの一覧は入門ノートブックにあります。

<!-- cell: learning-objectives src: c53dcc566f -->
## 学習目標

このチュートリアルを終えると、次のことができるようになります。

- Q-Lite PTQ、Q-PRO 拡張 PTQ、Q-Master QAT を比較し、適切なワークフローを選択する
- 同じモデル、データ、前処理から、統制された Q-Lite 実験と自動 Q-PRO 実験を構築する
- 量子化診断レポートを使って、一度に 1 変数ずつ変える精度回復実験を特定する
- 完全な ONNX コンパイルを繰り返さずに、QXNN チェックポイントを IQR 再キャリブレーションと自動 Q-PRO に再利用する
- Q-Master/QAT 設定を準備し、パイプラインのスモークテストと意味のある精度トレーニングを区別する
- モデルとワークフローの要件に基づいて `dxcom` CLI と `dx_com.compile()` Python API を使い分ける
- 名前をキーとする PyTorch `DataLoader` を実装し、Python API で 2 入力モデルをコンパイルする
- 特定のグラフや性能に関する問いに答える場合にのみ、高度なコンパイラ制御を適用する
- リリースレビューのために、モデル、キャリブレーション、コンパイラ、精度、性能、再現性のエビデンスを収集する

このノートブックは SDK のソースツリーを変更しません。生成されるファイルはすべて `<dx-tutorials>/notebooks/T05-DX-Compiler/workspace` 配下に保存され、git では無視されます。

<!-- cell: 2120bfff-8ac4-408b-ac67-6dfb87c7f8ab src: 709b2f76c9 -->
## 前提条件

- チュートリアル 05-1 と 05-2 が完了していること。
- DX-COM、キャリブレーションデータセット、DX-RT の `dxparse`。
- ダウンロード: Model Zoo の SqueezeNet ONNX と JSON(約 5 MB)。第 8 節の 2 入力モデルはローカルで生成します。
- 所要時間: このノートブックは異なる量子化設定で SqueezeNet を 5 回コンパイルし、さらに小さな Python API コンパイルを 1 回行います。Q-Lite ベースライン、診断、Python API のコンパイルは数秒で終わります。2 回の自動 Q-PRO コンパイル(第 4 節と 6.2 節)はそれぞれ約 10 分以上かかります。

次のセルは SDK を見つけ、これらの要件を確認し、ステータス表を表示します。`MISSING` と表示された項目には、それを提供する手順が併記されます。

<!-- cell: workspace-heading src: be980c7ccd -->
## 1. チュートリアルワークスペースの初期化

セットアップセルは `config.json` からすべての SDK パスを解決し、分離されたワークスペースを作成し、シンボリックリンクを通じて SDK のキャリブレーション画像を再利用します。

生成されるチュートリアルファイルは次の場所に保存されます。

```text
<dx-tutorials>/notebooks/T05-DX-Compiler/workspace/      (ignored by git)
├── models/                 downloaded and generated ONNX files
├── configs/                compiler JSON files
├── outputs/                one directory per compile
└── calibration_dataset -> <DX_COM_DIR>/calibration_dataset
```

<!-- cell: notebook-dependencies-heading src: 8bf0df0e3a -->
Jupyter カーネルと DX-COM コンパイラは別々の Python 環境を使用します。これは意図的な構成です。

```text
Jupyter code       -> <dx-tutorials>/.venv/bin/python
DX-COM Python API  -> <DX_COMPILER_DIR>/venv-dx-compiler-local/bin/python
```

ノートブック専用の検査用パッケージは `uv` で Jupyter 環境にインストールします。この uv 管理の環境では `%pip` を使用しないでください。

<!-- cell: prepare-model-heading src: 2002bc78a7 -->
## 2. コンパクトな参照モデルの準備

Model Zoo の SqueezeNet ONNX と、それに対応する Q-Lite JSON を使用します。ダウンロードした前処理はそのまま変更せず、ローカルのキャリブレーションデータセットのパスだけを置き換えます。

次のセルは `tutorial_paths.py` から `download_file` をインポートします(チュートリアル 05-2 でも使用している共有ヘルパーです)。このヘルパーは 2 つのルールに従います。

1. 空でない宛先ファイルが存在する場合、ネットワーク要求を行わずにダウンロードをスキップします。
2. 新しいファイルの場合、`.part` にダウンロードし、サイズチェックに合格した後にのみ宛先を置き換えます。

意図的に新しいコピーが必要な場合は、まずローカルファイルを削除してください。

<!-- cell: quantization-strategy src: 9e98660596 -->
## 3. 量子化戦略の選択

3 つの方式はいずれも INT8 の DXNN を生成しますが、使用する手法が異なり、必要なデータ量と計算量も異なります。

<img src="assets/q-lite-q-pro-q-master.png" alt="Q-Lite、Q-PRO、Q-Master の比較" style="max-width: 600px;">

| 項目 | Q-Lite | Q-PRO | Q-Master |
|---|---|---|---|
| 量子化手法 | 標準 PTQ | 拡張 PTQ | QAT(Quantization-Aware Training) |
| 主な仕組み | 浮動小数点の範囲をキャリブレーション | 自動または手動の DXQ 拡張ステージを適用 | 量子化された学生モデルをファインチューニング |
| 必要なデータ | 代表的なキャリブレーションサンプル | 同じ代表的なキャリブレーションサンプル | トレーニングと検証用サンプル。タスク損失を使う場合はラベルが必要 |
| トレーニング | なし | なし | あり |
| 一般的な計算コスト | 最も低い | Q-Lite より高い | 最も高い。通常は CUDA GPU を使用 |
| DX-COM のエントリポイント | 通常のコンパイル | `--use_q_pro` または手動の DXQ スキーム | 設定に `qmaster` ブロックを追加 |
| 推奨される役割 | 統制されたベースライン | 最初の PTQ 精度回復実験 | PTQ でも精度目標に届かない場合に使用 |

このチュートリアルで **Q-Master** とは、JSON の `qmaster` ブロックで有効化される DX-COM の QAT ワークフローを指します。名前だけで手法を選ばないでください。

<img src="assets/quantization-strategy-decision-flow.png" alt="Q-Lite から Q-PRO、Q-Master への判断フロー" style="max-width: 800px; width: 100%;">

<!-- cell: baseline-heading src: 0c8f1bdab9 -->
### 3.1 Q-Lite ベースラインの確立

公平なベースラインは、以降のすべての実験と同じ ONNX、キャリブレーションデータ、前処理、コンパイラバージョン、検証プロトコルを使用します。

次のセルは、別のターミナルで次のコマンドを入力するのと同等です。

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/squeezenet-1.0_224x224.onnx \
      -c configs/squeezenet.local.json \
      -o outputs/squeezenet_q_lite_baseline \
      --gen_log \
      --export_html
```

ノートブックは実際のパスを自動的に解決します。出力ディレクトリにすでに DXNN が含まれている場合、コンパイルはスキップされます。

> **コードセル末尾の `2>&1 | sed ... | grep ...` について:** `dxcom` はカーソル移動のエスケープコードで進捗バーを描画しますが、Jupyter はこれを描画できず、そのままでは何百行もの空行として表示されます。このフィルターは進捗バーの行だけを取り除き、`[INFO]`、`[WARNING]`、`[ERROR]` の各メッセージはすべて表示されたままになります。ターミナルではフィルターを省略して構いません。`--gen_log` は完全な出力を `compiler.log` に保存します。このノートブックのすべてのコンパイルセルで同じフィルターを使用しています。

<!-- cell: baseline-artifacts-heading src: 2bc6375342 -->
HTML サマリーと `compiler.log` は実験エビデンスの一部です。これらはタスク精度の検証を置き換えるものではありませんが、ビルドをレビュー可能かつ再現可能にします。

<!-- cell: qpro-heading src: c75215ec57 -->
## 4. 自動 Q-PRO の実行

Q-PRO は DXQ ファミリーから選択された量子化拡張ステージを試行します。`--use_q_pro` は DX-COM が組み合わせを自動的に選択するため、最も簡単な出発点です。

重要なルール:

- Q-Lite と Q-PRO の入力は同一に保ちます。
- `--use_q_pro` と手動で選択した `enhanced_scheme` は同時に指定できません。
- コンパイルコストの増加が正当化されるのは、測定したタスク精度が向上した場合だけです。

同等のターミナルコマンド:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/squeezenet-1.0_224x224.onnx \
      -c configs/squeezenet.local.json \
      -o outputs/squeezenet_q_pro \
      --use_q_pro \
      --opt_level 1 \
      --gen_log \
      --export_html
```

<!-- cell: compare-qpro src: a0cd8c050e -->
### 4.1 実験を正しく比較する

両方のモデルに同じホールドアウト検証データセットを使用します。1 つの数値だけでなく、複数の値を記録してください。

| エビデンス | 答えられる問い |
|---|---|
| タスク指標 | Q-PRO は有用な精度を回復したか? |
| コンパイル時間 | どれだけの開発コストが追加されたか? |
| DXNN サイズ | デプロイ成果物は大きく変わったか? |
| `dxrun` スループット | 単体のランタイムスループットは変わったか? |
| エンドツーエンドのレイテンシと CPU | アプリケーション全体は改善したか? |

Q-Lite がすでに製品目標を満たしている場合、Q-PRO が自動的に優れているわけではありません。

<!-- cell: diagnosis-heading src: 7923517f6c -->
## 5. 量子化損失の診断

診断は、ONNX と DXNN の精度差を再現できるようになってから使用します。`--quant_diagnosis` は次を生成します。

- `quant_diagnosis/diagnosis_report.html`: 領域ごとの品質エビデンスと再試行のガイダンス
- `quant_diagnosis/<model>.qxnn`: 再量子化のための再利用可能なチェックポイント

同等のターミナルコマンド:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/squeezenet-1.0_224x224.onnx \
      -c configs/squeezenet.local.json \
      -o outputs/squeezenet_diagnosis \
      --quant_diagnosis \
      --gen_log \
      --export_html
```

コンパイルがスキップされるのは、DXNN、診断 HTML、QXNN チェックポイントのすべてが存在する場合だけです。

<!-- cell: read-diagnosis src: bcd73e08c0 -->
### 5.1 レポートを実験計画として読む

1. **Warning** または **Critical** と表示された領域を見つけます。
2. 浮動小数点と量子化後の挙動が最初に乖離する場所を確認します。
3. エビデンスと推奨される再コンパイルの意図を読みます。
4. キャリブレーション手法や Q-PRO など、変更を 1 つ選びます。
5. QXNN チェックポイントから再開します。
6. 同じホールドアウトのタスク指標を再度測定します。

一度に変更する変数は 1 つにしてください。データセット、前処理、オブザーバー、コンパイラオプションを同時に変更すると、どの変更が効果をもたらしたのか特定できません。

<!-- cell: resume-heading src: c604a33ce2 -->
## 6. QXNN 再開による再量子化

QXNN 再開は、以前の ONNX コンパイル作業をスキップし、量子化に依存するステージだけを再実行します。

```text
Normal compile: ONNX -> optimize/partition -> quantize -> codegen -> DXNN
                                |
                                +-> diagnosis QXNN checkpoint
                                             |
Resume:                             new quantization setting -> DXNN
```

ルール:

- `--checkpoint` と `--model_path` は同時に指定できません。
- 元の設定はチェックポイントに埋め込まれているため、`-c` は不要です。
- `--dataset_path` は、埋め込まれた場所を意図的に上書きする場合にのみ使用します。

<!-- cell: resume-iqr-heading src: 8455073c65 -->
### 6.1 IQR 再キャリブレーションで再開する

同等のターミナルコマンド:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom --checkpoint outputs/squeezenet_diagnosis/quant_diagnosis/<model>.qxnn \
      -o outputs/squeezenet_resume_iqr \
      --recalibration_method iqr \
      --dataset_path calibration_dataset
```

<!-- cell: resume-qpro-heading src: 3df09ba500 -->
### 6.2 自動 Q-PRO で再開する

この実験は同じチェックポイントを再利用しますが、IQR の代わりに自動 Q-PRO を適用します。

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom --checkpoint outputs/squeezenet_diagnosis/quant_diagnosis/<model>.qxnn \
      -o outputs/squeezenet_resume_q_pro \
      --use_q_pro \
      --dataset_path calibration_dataset
```

<!-- cell: qat-heading src: d9acd3286b -->
## 7. Q-Master(QAT)実験の準備

Q-Master は、PTQ 手法が精度目標を満たさない場合に Quantization-Aware Training を使用します。通常の JSON 設定に `qmaster` ブロックを追加するだけで、別個の CLI フラグは不要です。

```text
Calibration -> QAT training -> best QXNN checkpoint -> final DXNN
```

意味のある QAT 実行には、代表的なトレーニングデータと検証データ、そして通常は CUDA GPU が必要です。チュートリアルのキャリブレーション画像はトレーニングデータセットではないため、この節では高コストで誤解を招くトレーニングを開始する代わりに、レビュー済みのテンプレートを作成します。

重要な考え方:

- `default_loader` は、キャリブレーションとトレーニングの両方における前処理の供給源です。
- `qmaster` にはトレーニングのハイパーパラメータが含まれます。
- `fast_run: true` は 1 エポック・1 バッチでパイプラインを検証します。精度は検証**しません**。
- データセットのリビジョン、設定、トレーニングログ、最良のチェックポイント、コンパイラバージョン、検証結果を保存してください。

<!-- cell: run-qat-guidance src: 6a412f8ffc -->
データセットのパスを置き換え、ハイパーパラメータを確認したら、別のターミナルで QAT を実行します。

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/squeezenet-1.0_224x224.onnx \
      -c configs/squeezenet.qat.template.json \
      -o outputs/squeezenet_qat \
      --gen_log \
      --export_html
```

重要な出力:

| 成果物 | 用途 |
|---|---|
| `qat_checkpoint/qat_checkpoint.qxnn` | 最良のトレーニングチェックポイント。コンパイルのみの再開に再利用可能 |
| `*.dxnn` | 最終的なデプロイ成果物 |
| `compiler.log` とトレーニングログ | デバッグと再現性のためのエビデンス |

パイプラインのスモークテストだけを行う場合は、`qmaster` の中に `"fast_run": true` を追加します。`fast_run` は JSON キーであり、`dxcom` のオプションではありません。

<!-- cell: python-api-overview src: 666d46f4f0 -->
## 8. Python API によるプログラムからのコンパイル

CLI と Python API は同じ DX-COM コンパイラエンジンを使用します。Python API を使っても標準的なモデルが本質的に速くなったり精度が上がったりするわけではありません。変わるのは、アプリケーションがモデル、キャリブレーションデータ、コンパイラオプションを渡す方法です。

### 8.1 どちらのインターフェースをいつ使うべきか

| 要件 | `dxcom` CLI | `dx_com.compile()` Python API |
|---|---:|---:|
| JSON 前処理による標準的な単一画像入力 | 推奨 | 対応 |
| 再現可能なシェルコマンドや CI ステップ | 推奨 | 対応 |
| メモリ上の `onnx.ModelProto` | 不可 | 可 |
| カスタムの PyTorch 前処理/データパイプライン | 不可 | 可 |
| 画像以外のキャリブレーションデータ | 不可 | 可 |
| 複数入力モデル | 不可 | **必須** |
| プログラムによる実験ループと例外処理 | 限定的 | 推奨 |
| アプリケーションからの直接的な QAT/チェックポイント制御 | 限定的 | 推奨 |

Python API の利点:

- ONNX パスまたはメモリ上の `onnx.ModelProto` を渡せる
- PyTorch `DataLoader` で正確なキャリブレーションテンソルを提供できる
- 複数入力モデルや画像以外のモデルに対応できる
- シェル出力の解析なしで繰り返し可能な実験ループを構築できる
- Python 例外を処理し、構造化されたメタデータを記録できる
- モデルのエクスポート、検証、成果物管理のコードにコンパイルを統合できる

よくある間違いを防ぐための 2 つの制約があります。

1. `config=` と `dataloader=` はどちらか一方だけを指定します。両者は同時に指定できません。
2. `dataloader=` を使う場合、前処理は `Dataset.__getitem__()` の中で行い、デプロイ時の前処理と一致させる必要があります。

<!-- cell: inspect-python-api-heading src: c31ad53df9 -->
### 8.2 インストール済みの API を確認する

DX-COM は専用のコンパイラ環境にインストールされているため、ノートブックのカーネルは `dx_com` を直接インポートしません。次のセルは、正確に以下のコマンドを 1 行の `!` につないで実行します。

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
python -c 'import inspect, dx_com; print(inspect.signature(dx_com.compile))'
```

<!-- cell: python-api-parameters src: ab00c54c22 -->
最も重要なパラメータは次のとおりです。

| パラメータ | 意味 |
|---|---|
| `model` | ONNX ファイルパスまたは `onnx.ModelProto` |
| `output_dir` | DXNN とレポートを受け取るディレクトリ |
| `config` | 標準的な単一入力画像モデル向けの JSON 駆動のキャリブレーション |
| `dataloader` | プログラムによるキャリブレーションテンソル。この複数入力ラボでは必須 |
| `calibration_method` | 通常のコンパイルでは `ema` または `minmax` |
| `calibration_num` | DataLoader から消費するサンプル数 |
| `opt_level` | 高速な実験には `0`、完全な最適化には `1` |
| `use_q_pro` | 自動 Q-PRO を有効化。手動の `enhanced_scheme` とは併用不可 |
| `quant_diagnosis` | 診断 HTML と再開可能な QXNN を生成 |
| `gen_log`、`export_html` | レビューとデバッグのエビデンスを保存 |

標準的な単一入力の呼び出しは次のようになります。

```python
import dx_com

dx_com.compile(
    model="models/model.onnx",
    config="configs/model.json",
    output_dir="outputs/model",
    gen_log=True,
    export_html=True,
)
```

カスタム DataLoader を使う場合は `config=` を `dataloader=` に置き換えます。両方を渡してはいけません。

<!-- cell: multi-input-model-heading src: c96a18bc8d -->
### 8.3 Python API を必要とするモデル

ステレオ深度、オプティカルフロー、特徴融合のネットワークは、同期した 2 つのテンソルを入力とすることが一般的です。このラボでは、長時間のモデルダウンロードやコンパイルなしに複数入力の要件を示せるため、コンパクトなステレオ特徴融合モデルを選びました。`dxcom` CLI の JSON ローダーはモデル入力を 1 つしかサポートしないため、両方のテンソルを供給できません。Python API なら、ONNX の正確な入力名をキーとする辞書を返すことができます。

```text
left image  -- preprocessing -- tensor "left_image"  --+
                                                        +--> stereo-fusion ONNX --> DXNN
right image -- preprocessing -- tensor "right_image" --+

Dataset.__getitem__()
    -> {"left_image": Tensor[C,H,W], "right_image": Tensor[C,H,W]}
DataLoader(batch_size=1)
    -> {"left_image": Tensor[1,C,H,W], "right_image": Tensor[1,C,H,W]}
```

代表的なグラフは、特徴融合の前に両方の入力をエンコードします。これにより 2 枚の画像を 1 つの入力として扱うことを避け、本物の 2 入力 DXNN コントラクトを維持します。

```text
left_image  -> Conv -> left_features  --+
                                          Add -> ReLU -> fused_features
right_image -> Conv -> right_features --+
```

このラボではコンパクトで代表的なステレオ融合グラフを構築します。演習が API コントラクトに集中できるよう、意図的に小さくしています。合成されたキャリブレーションデータはコンパイルの仕組みを検証するだけです。本番のステレオモデルには、同期した代表的な左右サンプルが必要です。

<!-- cell: multi-input-dataloader-heading src: 80154869ac -->
### 8.4 名前付きキャリブレーション DataLoader の実装

各データセット項目にはバッチ次元を含めません。`DataLoader(batch_size=1)` が自動的に追加します。

辞書のキーはタプルよりも安全です。

- 辞書: 入力は ONNX 名で対応付けられる
- タプル/リスト: 入力はモデル内部の入力順序で対応付けられる

スクリプトはコンパイラを呼び出す前に最初のバッチを検証し、DXNN がすでに存在する場合はコンパイルをスキップします。

<!-- cell: run-multi-input-heading src: d764daa244 -->
### 8.5 DX-COM Python 環境でコンパイルする

次のノートブックセルは、コンパイラ環境の Python インタープリターでスクリプトを実行します。これは次と同等です。

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
python compile_stereo_fusion.py
```

このモデルには意図的に `dxcom` CLI での同等コマンドがありません。コンパイラはカスタム DataLoader が供給する 2 つのテンソルを必要とするためです。

<!-- cell: inspect-multi-input-heading src: f6368021a6 -->
### 8.6 結果の確認

`dxparse -v` には `left_image` と `right_image` の 2 つのモデル入力が表示されるはずです。これにより、コンパイルされた成果物が複数入力のインターフェースを維持していることが確認できます。

キャリブレーション DataLoader は ONNX コントラクト(`float32`、NCHW)に一致しています。DX-COM はサポートされる入力変換を NPU パスに移動するため、コンパイル後の NPU インターフェースは `uint8`、NHWC として報告されることがあります。デプロイでは、ONNX のレイアウトが変わっていないと仮定するのではなく、生成された DXNN について報告される入力形状、dtype、前処理のガイダンスに従ってください。

<!-- cell: advanced-controls-heading src: 7bf0d30ccb -->
## 9. 高度なコンパイラ制御を意図的に使う

これらのオプションは、グラフ、性能、再現性に関する特定の問いに答えるために使うべきものです。「さらに最適化する」ための汎用的なセットとして有効化しないでください。

| CLI オプション | Python API パラメータ | 目的 | 主な注意点 |
|---|---|---|---|
| `--opt_level {0,1}` | `opt_level` | コンパイル時間と完全な最適化のトレードオフ | 統制されたベースラインと比較する |
| `--aggressive_partitioning` | `aggressive_partitioning` | より多くの NPU パーティショニングを探索 | 実験的。出力とエンドツーエンドのレイテンシを検証する |
| `--compile_input_nodes` | `input_nodes` | 選択した ONNX 演算子ノードから開始 | コンパイル後のインターフェースが変わる |
| `--compile_output_nodes` | `output_nodes` | 選択した ONNX 演算子ノードで終了 | 残りの処理はホスト側に移る |
| `--float64_calibration` | `float64_calibration` | CPU 間でのキャリブレーションの決定性を向上 | 計算とメモリのコストが高い |
| `--gen_log` | `gen_log` | 詳細なコンパイラログを保存 | リリース成果物と一緒に保管する |
| `--export_html` | `export_html` | 自己完結型のサマリーを生成 | タスク検証の代わりにはならない |

サブグラフのコンパイルには、テンソル名ではなく **ONNX 演算子ノード名**を使用します。結果として得られる入出力コントラクトと、ホスト側に残る演算を文書化してください。

<!-- cell: release-validation-heading src: 82c9e17d3c -->
## 10. リリース検証記録の作成

コンパイラが正常終了することは必要条件ですが、リリースの十分なエビデンスではありません。

| 領域 | 最低限のエビデンス |
|---|---|
| ソースモデル | エクスポートコマンド、フレームワーク/バージョン、ONNX チェッカーの結果 |
| キャリブレーション | データセットのリビジョン、サンプル数、前処理、手法 |
| コンパイラ | DX-COM バージョン、完全なコマンド/API 引数、ログ、HTML レポート |
| 精度 | ONNX とすべての DXNN 候補に対する同一のタスク指標 |
| 性能 | ウォームアップ方針、実行時間、デバイス、スループット、レイテンシ、CPU 負荷 |
| 統合 | 入出力形状、前処理の責任範囲、デコーダーのコントラクト |
| 再現性 | 成果物のハッシュと分離された出力ディレクトリ |

ダミー入力を使った `dxrun` はスループットの確認には有用ですが、タスク精度やアプリケーション全体のレイテンシ測定の代わりにはなりません。

<!-- cell: 20d9ebdb-92ef-4b3e-9c73-c4bf02326bd8 src: 175675be1b -->
## 11. トラブルシューティング

| 症状 | 表示される内容 | 考えられる原因 | 対処 |
|---|---|---|---|
| `--use_q_pro` と `--enhanced_scheme` が競合する | オプションの競合に関するコンパイラエラー | 自動と手動の Q-PRO 選択が同時に要求された | 1 回のコンパイルではどちらか一方だけを使う |
| `--checkpoint` ファイルが見つからない | `.qxnn` ファイルに対する `FileNotFoundError` | `quant_diagnosis/*.qxnn` を書き出す第 5 節の診断コンパイルがスキップされたか失敗した | 先に第 5 節を実行し、チェックポイントのパスが表示されることを確認する |
| 第 7 節で QAT が実行されない | `configs/squeezenet.qat.template.json` が書き出されるだけ | 意図的な動作: サンプル画像では有用なモデルをトレーニングできない | データセットのパスを代表的なトレーニングデータに置き換え、第 7 節のコマンドをターミナルで実行する |
| JSON の変更が反映されない | `Skip compilation: found ...` | 出力ディレクトリにすでに DXNN が含まれている | `workspace/outputs/<dir>` を削除して再実行する |
| `dxparse` がない | `[MISSING] dx_rt` | DX-RT がインストールされていない | DX-Runtime をインストールする(チュートリアル 01 第 3 節) |

<!-- cell: summary src: 0316911a19 -->
## 12. まとめ

### 12.1 上級の精度回復マップ

```text
Build a controlled Q-Lite baseline
                │
                ▼
       Accuracy target met?
          ┌─────┴─────┐
         Yes          No
          │            │
          │            ▼
          │      Try automatic Q-PRO
          │            │
          │            ▼
          │   Accuracy target met?
          │      ┌─────┴─────┐
          │     Yes          No
          │      │            │
          │      │            ▼
          │      │    Diagnose quantization loss
          │      │            │
          │      │            ▼
          │      │    Resume from QXNN and test
          │      │    one change at a time
          │      │            │
          │      │            ▼
          │      │    Prepare Q-Master QAT
          └──────┴────────────┘
                │
                ▼
 Validate accuracy, performance,
 integration, and reproducibility
```

### 12.2 上級ワークフローのダッシュボード

| ワークフロー | 使う場面 | 主な入力 | 主な結果 |
|---|---|---|---|
| Q-Lite ベースライン | 最初の統制された PTQ 実験 | ONNX、JSON、代表的なキャリブレーションデータ | ベースラインの DXNN とエビデンス |
| 自動 Q-PRO | Q-Lite が精度目標に届かない | 同じモデル、データ、前処理、検証プロトコル | 拡張 PTQ の候補 |
| 診断と QXNN 再開 | ONNX から DXNN への精度差が再現可能 | 診断レポートと QXNN チェックポイント | 一度に 1 変数ずつの高速な試行 |
| Q-Master QAT | PTQ 手法でも目標に届かない | 代表的なトレーニングデータと検証データ | トレーニング済みの量子化候補 |
| Python API | キャリブレーションにカスタムテンソル、画像以外のデータ、複数入力が必要 | ONNX と名前をキーとする PyTorch DataLoader | プログラムからコンパイルされた DXNN |

### 12.3 完了した実験

| 実験 | 統制された比較またはコントラクト | 生成されたエビデンス |
|---|---|---|
| SqueezeNet の Q-Lite と Q-PRO の比較 | 同じ ONNX、キャリブレーション、前処理、コンパイラ設定 | 個別の DXNN ファイル、ログ、HTML レポート |
| 量子化診断 | 設定を変更する前に精度損失の候補を再現 | 診断 HTML と再利用可能な QXNN チェックポイント |
| QXNN 再開 | 同じチェックポイントからの IQR 再キャリブレーションと自動 Q-PRO の比較 | より高速な量子化実験 |
| Q-Master の準備 | スモークテスト設定と意味のある QAT トレーニングの区別 | レビュー済みの QAT テンプレートと検証要件 |
| ステレオ融合の Python API | 1 つの DataLoader が供給する 2 つの名前付き入力 | `dxparse` で検証した 2 入力 DXNN |

### 12.4 完了チェックリスト

- [ ] 再利用可能な Q-Lite ベースラインを構築した
- [ ] 統制された PTQ 実験として自動 Q-PRO を実行した
- [ ] 量子化診断レポートを生成して解釈した
- [ ] QXNN チェックポイントを IQR と Q-PRO の試行に再利用した
- [ ] Q-Master のスモークテストと意味のある QAT を区別した
- [ ] ワークフローの要件から CLI または Python API を選択した
- [ ] 名前をキーとする複数入力のキャリブレーション DataLoader を実装した
- [ ] 2 入力 DXNN をコンパイルして検査した
- [ ] リリースレビューに必要なエビデンスを特定した
- [ ] チュートリアルのサンプルを代表的な製品データに置き換えた
- [ ] ONNX とすべての DXNN 候補を同じホールドアウトのタスク指標で比較した
- [ ] アプリケーション全体のレイテンシ、スループット、ホスト CPU 使用率を測定した

### 12.5 選び方

| 必要なこと | 始めるべきもの |
|---|---|
| 再現可能な INT8 リファレンスを確立する | Q-Lite |
| トレーニングなしで PTQ の精度を回復する | 自動 Q-PRO |
| 量子化損失がどこから始まるかを見つける | 量子化診断 |
| 完全な ONNX コンパイルを繰り返さずに別の量子化設定を試す | QXNN 再開 |
| PTQ のオプションを使い切った後に精度を回復する | Q-Master QAT |
| カスタム、画像以外、メモリ上、または複数の名前付き入力を供給する | `dx_com.compile()` Python API |
| 標準的なモデル向けに再現可能なシェルや CI のコンパイルステップを構築する | `dxcom` CLI |

> **覚えておいてください:** コンパイラの成功メッセージ、生成された DXNN、ダミー入力での高速なベンチマークは部分的なエビデンスにすぎません。リリースの判断には、代表的なデータ、ONNX と DXNN に対する同じホールドアウトのタスク指標、パイプライン全体の性能測定、そして再現可能な成果物が必要です。

### 12.6 このワークフローを自分のモデルに適用する

1. ソース ONNX、前処理、データセットのリビジョン、検証指標を固定します。
2. 量子化設定を変更する前に Q-Lite ベースラインを確立します。
3. 一度に 1 変数ずつ変更し、すべての結果を分離された出力ディレクトリに保存します。
4. Q-Lite から Q-PRO、診断/QXNN 再開、そして最後に Q-Master へと段階を上げるのは、測定した精度がそれを必要とする場合だけにします。
5. コマンドまたは API 引数、コンパイラログ、レポート、ハッシュ、精度、性能の結果をリリース候補と一緒に保存します。

> **次へ:** チュートリアル 06 に進み、DX-Runtime でコンパイル済みモデルを実行しましょう。CLI ツール、Python と C++ の API、プロファイリング、リリース検証を扱います。
