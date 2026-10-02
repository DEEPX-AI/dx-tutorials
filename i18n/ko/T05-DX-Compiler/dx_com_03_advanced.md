<!-- i18n source: notebooks/T05-DX-Compiler/dx_com_03_advanced.ipynb -->
<!-- i18n lang: ko -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: advanced-title src: ca77fa067f -->
# DEEPX Tutorial 05-3 - DX-COM 고급

이 튜토리얼은 정확도 복구, 양자화 진단, 재사용 가능한 QXNN 체크포인트, 양자화 인식 학습(QAT), 그리고 `dx_com` Python API를 사용한 프로그래밍 방식의 컴파일을 다룹니다.

앞부분의 실습은 워크플로에 집중할 수 있도록 SqueezeNet을 사용합니다. 마지막 실습은 Python API가 필요한 사례를 보여 주기 위해 입력이 두 개인 소형 스테레오 퓨전 모델을 사용합니다.

> **시작하기 전에:** 먼저 초급과 중급 튜토리얼을 완료하세요. 컴파일에는 몇 분이 걸릴 수 있습니다. 모든 컴파일 셀은 기존 DXNN이 있는지 확인하고 반복 작업을 건너뜁니다.

<!-- cell: ac85dc95-b527-4b0a-8cad-e82a0d512e16 src: 215e801d3c -->
> 이 튜토리얼의 3부 중 3부입니다. 코스 맵과 전체 파트 목록은 초급 노트북에 있습니다.

<!-- cell: learning-objectives src: c53dcc566f -->
## 학습 목표

이 튜토리얼을 마치면 다음을 할 수 있습니다.

- Q-Lite PTQ, Q-PRO 향상 PTQ, Q-Master QAT를 비교하고 적절한 워크플로를 선택합니다.
- 같은 모델, 데이터, 전처리로 통제된 Q-Lite 실험과 자동 Q-PRO 실험을 구성합니다.
- 양자화 진단 리포트를 활용해 한 번에 하나의 변수만 바꾸는 정확도 복구 실험을 식별합니다.
- 전체 ONNX 컴파일을 반복하지 않고 QXNN 체크포인트를 재사용해 IQR 재캘리브레이션과 자동 Q-PRO를 수행합니다.
- Q-Master/QAT 설정을 준비하고 파이프라인 스모크 테스트와 의미 있는 정확도 학습을 구분합니다.
- 모델과 워크플로 요구 사항에 따라 `dxcom` CLI와 `dx_com.compile()` Python API 중 하나를 선택합니다.
- 이름을 키로 사용하는 PyTorch `DataLoader`를 구현하고 Python API로 입력이 두 개인 모델을 컴파일합니다.
- 특정 그래프 또는 성능 질문에 답할 때만 고급 컴파일러 제어 옵션을 적용합니다.
- 릴리스 검토를 위한 모델, 캘리브레이션, 컴파일러, 정확도, 성능, 재현성 증거를 수집합니다.

이 노트북은 SDK 소스 트리를 수정하지 않습니다. 생성되는 모든 파일은 git이 무시하는 `<dx-tutorials>/notebooks/T05-DX-Compiler/workspace` 아래에 저장됩니다.

<!-- cell: 2120bfff-8ac4-408b-ac67-6dfb87c7f8ab src: 709b2f76c9 -->
## 사전 요구 사항

- 튜토리얼 05-1과 05-2 완료.
- DX-COM, 캘리브레이션 데이터셋, DX-RT의 `dxparse`.
- 다운로드: Model Zoo의 SqueezeNet ONNX와 JSON(약 5 MB). 8절의 입력 두 개 모델은 로컬에서 생성됩니다.
- 소요 시간: 이 노트북은 서로 다른 양자화 설정으로 SqueezeNet을 다섯 번 컴파일하고, 작은 Python API 컴파일을 한 번 수행합니다. Q-Lite 기준선, 진단, Python API 컴파일은 몇 초면 끝나며, 두 번의 자동 Q-PRO 컴파일(4절과 6.2절)은 각각 10분 이상 걸립니다.

다음 셀은 SDK를 찾고, 이 요구 사항을 확인한 뒤 상태 표를 출력합니다. `MISSING`으로 표시된 항목에는 그것을 제공하는 단계가 함께 표시됩니다.

<!-- cell: workspace-heading src: be980c7ccd -->
## 1. 튜토리얼 워크스페이스 초기화

설정 셀은 `config.json`에서 모든 SDK 경로를 확인하고, 격리된 워크스페이스를 만들고, 심볼릭 링크를 통해 SDK 캘리브레이션 이미지를 재사용합니다.

생성되는 튜토리얼 파일은 다음 위치에 저장됩니다.

```text
<dx-tutorials>/notebooks/T05-DX-Compiler/workspace/      (ignored by git)
├── models/                 downloaded and generated ONNX files
├── configs/                compiler JSON files
├── outputs/                one directory per compile
└── calibration_dataset -> <DX_COM_DIR>/calibration_dataset
```

<!-- cell: notebook-dependencies-heading src: 8bf0df0e3a -->
Jupyter 커널과 DX-COM 컴파일러는 서로 다른 Python 환경을 사용합니다. 이는 의도된 구성입니다.

```text
Jupyter code       -> <dx-tutorials>/.venv/bin/python
DX-COM Python API  -> <DX_COMPILER_DIR>/venv-dx-compiler-local/bin/python
```

노트북 전용 검사 패키지는 `uv`로 Jupyter 환경에 설치합니다. uv로 관리되는 이 환경에서는 `%pip`을 사용하지 마세요.

<!-- cell: prepare-model-heading src: 2002bc78a7 -->
## 2. 소형 참조 모델 준비

Model Zoo의 SqueezeNet ONNX와 이에 맞는 Q-Lite JSON을 사용합니다. 다운로드한 전처리는 그대로 유지하고, 로컬 캘리브레이션 데이터셋 경로만 교체합니다.

다음 셀은 `tutorial_paths.py`에서 `download_file`을 가져옵니다(튜토리얼 05-2에서도 사용한 공용 헬퍼). 이 함수는 두 가지 규칙을 따릅니다.

1. 비어 있지 않은 대상 파일이 있으면 네트워크 요청 없이 다운로드를 건너뜁니다.
2. 새 파일은 `.part`로 다운로드하고 크기 검사를 통과한 뒤에만 대상 파일을 교체합니다.

의도적으로 새 사본이 필요하면 먼저 로컬 파일을 삭제하세요.

<!-- cell: quantization-strategy src: 9e98660596 -->
## 3. 양자화 전략 선택

세 경로 모두 INT8 DXNN을 생성하지만, 사용하는 방법이 다르고 필요한 데이터와 연산량도 다릅니다.

<img src="assets/q-lite-q-pro-q-master.png" alt="Q-Lite, Q-PRO, Q-Master 비교" style="max-width: 600px;">

| 항목 | Q-Lite | Q-PRO | Q-Master |
|---|---|---|---|
| 양자화 방법 | 표준 PTQ | 향상 PTQ | QAT(양자화 인식 학습) |
| 주요 메커니즘 | 부동소수점 범위 캘리브레이션 | 자동 또는 수동 DXQ 향상 단계 적용 | 양자화된 학생 모델 파인튜닝 |
| 필요한 데이터 | 대표성 있는 캘리브레이션 샘플 | 동일한 대표성 있는 캘리브레이션 샘플 | 학습 및 검증 샘플. 태스크 손실을 사용할 때는 레이블 필요 |
| 학습 | 아니요 | 아니요 | 예 |
| 일반적인 연산 비용 | 가장 낮음 | Q-Lite보다 높음 | 가장 높음. 보통 CUDA GPU 사용 |
| DX-COM 진입점 | 일반 컴파일 | `--use_q_pro` 또는 수동 DXQ 스킴 | 설정에 `qmaster` 블록 추가 |
| 권장 역할 | 통제된 기준선 | 첫 번째 PTQ 정확도 복구 실험 | PTQ로도 정확도 목표에 미달할 때 사용 |

이 튜토리얼에서 **Q-Master**는 JSON `qmaster` 블록으로 활성화되는 DX-COM QAT 워크플로를 뜻합니다. 이름만 보고 방법을 선택하지 마세요.

<img src="assets/quantization-strategy-decision-flow.png" alt="Q-Lite에서 Q-PRO, Q-Master로 이어지는 결정 흐름" style="max-width: 800px; width: 100%;">

<!-- cell: baseline-heading src: 0c8f1bdab9 -->
### 3.1 Q-Lite 기준선 수립

공정한 기준선은 이후의 모든 실험과 동일한 ONNX, 캘리브레이션 데이터, 전처리, 컴파일러 버전, 검증 프로토콜을 사용합니다.

다음 셀은 별도의 터미널에서 다음 명령을 입력하는 것과 동일합니다.

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/squeezenet-1.0_224x224.onnx \
      -c configs/squeezenet.local.json \
      -o outputs/squeezenet_q_lite_baseline \
      --gen_log \
      --export_html
```

노트북은 실제 경로를 자동으로 확인합니다. 출력 디렉터리에 이미 DXNN이 있으면 컴파일을 건너뜁니다.

> **코드 셀 끝의 `2>&1 | sed ... | grep ...`에 관하여:** `dxcom`은 Jupyter가 렌더링할 수 없는 커서 이동 이스케이프 코드로 진행 표시줄을 그리는데, 이것이 그대로 출력되면 수백 줄의 빈 줄로 나타납니다. 이 필터는 진행 표시줄 줄만 제거하며, 모든 `[INFO]`, `[WARNING]`, `[ERROR]` 메시지는 그대로 표시됩니다. 터미널에서는 필터를 생략해도 됩니다. `--gen_log`는 전체 출력을 `compiler.log`에 보존합니다. 이 노트북의 모든 컴파일 셀에서 같은 필터를 사용합니다.

<!-- cell: baseline-artifacts-heading src: 2bc6375342 -->
HTML 요약과 `compiler.log`는 실험 증거의 일부입니다. 태스크 정확도 검증을 대신하지는 않지만, 빌드를 검토하고 재현할 수 있게 해 줍니다.

<!-- cell: qpro-heading src: c75215ec57 -->
## 4. 자동 Q-PRO 실행

Q-PRO는 DXQ 계열에서 선택한 양자화 향상 단계를 시도합니다. `--use_q_pro`는 DX-COM이 조합을 자동으로 선택하므로 가장 쉬운 출발점입니다.

중요한 규칙:

- Q-Lite와 Q-PRO의 입력을 동일하게 유지합니다.
- `--use_q_pro`와 수동으로 선택한 `enhanced_scheme`은 함께 사용할 수 없습니다.
- 더 높은 컴파일 비용은 측정된 태스크 정확도가 개선될 때만 정당화됩니다.

동일한 터미널 명령:

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
### 4.1 실험을 올바르게 비교하기

두 모델 모두 동일한 홀드아웃 검증 데이터셋을 사용합니다. 숫자 하나가 아니라 여러 항목을 기록하세요.

| 증거 | 답할 수 있는 질문 |
|---|---|
| 태스크 지표 | Q-PRO가 유용한 정확도를 복구했는가? |
| 컴파일 시간 | 개발 비용이 얼마나 추가되었는가? |
| DXNN 크기 | 배포 산출물이 실질적으로 달라졌는가? |
| `dxrun` 처리량 | 단독 런타임 처리량이 달라졌는가? |
| 엔드투엔드 지연 시간과 CPU | 전체 애플리케이션이 개선되었는가? |

Q-Lite가 이미 제품 목표를 충족한다면 Q-PRO가 자동으로 더 나은 것은 아닙니다.

<!-- cell: diagnosis-heading src: 7923517f6c -->
## 5. 양자화 손실 진단

ONNX와 DXNN 사이의 정확도 차이를 재현할 수 있게 된 뒤에 진단을 사용하세요. `--quant_diagnosis`는 다음을 생성합니다.

- `quant_diagnosis/diagnosis_report.html`: 영역별 품질 증거와 재시도 안내,
- `quant_diagnosis/<model>.qxnn`: 재양자화에 재사용할 수 있는 체크포인트.

동일한 터미널 명령:

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

DXNN, 진단 HTML, QXNN 체크포인트가 모두 존재할 때만 컴파일을 건너뜁니다.

<!-- cell: read-diagnosis src: bcd73e08c0 -->
### 5.1 리포트를 실험 계획으로 읽기

1. **Warning** 또는 **Critical**로 표시된 영역을 찾습니다.
2. 부동소수점 동작과 양자화 동작이 처음 갈라지는 지점을 확인합니다.
3. 증거와 권장 재컴파일 의도를 읽습니다.
4. 캘리브레이션 방법이나 Q-PRO 같은 변경 사항을 하나 선택합니다.
5. QXNN 체크포인트에서 재개합니다.
6. 동일한 홀드아웃 태스크 지표를 다시 측정합니다.

한 번에 하나의 변수만 바꾸세요. 데이터셋, 전처리, 옵저버, 컴파일러 옵션이 모두 함께 바뀌면 어떤 변경이 도움이 되었는지 알 수 없습니다.

<!-- cell: resume-heading src: c604a33ce2 -->
## 6. QXNN 재개로 재양자화

QXNN 재개는 앞단의 ONNX 컴파일 작업을 건너뛰고 양자화에 의존하는 단계만 다시 실행합니다.

```text
Normal compile: ONNX -> optimize/partition -> quantize -> codegen -> DXNN
                                |
                                +-> diagnosis QXNN checkpoint
                                             |
Resume:                             new quantization setting -> DXNN
```

규칙:

- `--checkpoint`와 `--model_path`는 함께 사용할 수 없습니다.
- 원본 설정이 체크포인트에 포함되어 있으므로 `-c`는 필요하지 않습니다.
- `--dataset_path`는 포함된 위치를 의도적으로 덮어쓸 때만 사용합니다.

<!-- cell: resume-iqr-heading src: 8455073c65 -->
### 6.1 IQR 재캘리브레이션으로 재개

동일한 터미널 명령:

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom --checkpoint outputs/squeezenet_diagnosis/quant_diagnosis/<model>.qxnn \
      -o outputs/squeezenet_resume_iqr \
      --recalibration_method iqr \
      --dataset_path calibration_dataset
```

<!-- cell: resume-qpro-heading src: 3df09ba500 -->
### 6.2 자동 Q-PRO로 재개

이 실험은 같은 체크포인트를 재사용하되 IQR 대신 자동 Q-PRO를 적용합니다.

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom --checkpoint outputs/squeezenet_diagnosis/quant_diagnosis/<model>.qxnn \
      -o outputs/squeezenet_resume_q_pro \
      --use_q_pro \
      --dataset_path calibration_dataset
```

<!-- cell: qat-heading src: d9acd3286b -->
## 7. Q-Master(QAT) 실험 준비

Q-Master는 PTQ 방법으로 정확도 목표를 달성하지 못할 때 양자화 인식 학습을 사용합니다. 일반 JSON 설정에 `qmaster` 블록을 추가하면 되며, 별도의 CLI 플래그는 필요하지 않습니다.

```text
Calibration -> QAT training -> best QXNN checkpoint -> final DXNN
```

의미 있는 QAT 실행에는 대표성 있는 학습 및 검증 데이터와 보통 CUDA GPU가 필요합니다. 튜토리얼의 캘리브레이션 이미지는 학습 데이터셋이 아니므로, 이 절에서는 비용이 많이 들고 오해를 부를 수 있는 학습을 시작하는 대신 검토된 템플릿을 생성합니다.

핵심 개념:

- `default_loader`는 캘리브레이션과 학습 모두의 전처리 소스입니다.
- `qmaster`에는 학습 하이퍼파라미터가 들어 있습니다.
- `fast_run: true`는 에폭 하나와 배치 하나로 파이프라인을 검증합니다. 정확도를 검증하는 것은 **아닙니다**.
- 데이터셋 리비전, 설정, 학습 로그, 최적 체크포인트, 컴파일러 버전, 검증 결과를 보존하세요.

<!-- cell: run-qat-guidance src: 6a412f8ffc -->
데이터셋 경로를 교체하고 하이퍼파라미터를 검토한 뒤, 별도의 터미널에서 QAT를 실행합니다.

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/squeezenet-1.0_224x224.onnx \
      -c configs/squeezenet.qat.template.json \
      -o outputs/squeezenet_qat \
      --gen_log \
      --export_html
```

중요한 출력:

| 산출물 | 용도 |
|---|---|
| `qat_checkpoint/qat_checkpoint.qxnn` | 최적 학습 체크포인트. 컴파일 전용 재개에 재사용 가능 |
| `*.dxnn` | 최종 배포 산출물 |
| `compiler.log`와 학습 로그 | 디버깅과 재현성을 위한 증거 |

파이프라인 스모크 테스트만 하려면 `qmaster` 안에 `"fast_run": true`를 추가하세요. `fast_run`은 JSON 키이지 `dxcom` 옵션이 아닙니다.

<!-- cell: python-api-overview src: 666d46f4f0 -->
## 8. Python API로 프로그래밍 방식 컴파일

CLI와 Python API는 같은 DX-COM 컴파일러 엔진을 사용합니다. Python API가 표준 모델을 본질적으로 더 빠르거나 더 정확하게 만들지는 않습니다. 달라지는 것은 애플리케이션이 모델, 캘리브레이션 데이터, 컴파일러 옵션을 제공하는 방식입니다.

### 8.1 어떤 인터페이스를 언제 사용해야 할까요?

| 요구 사항 | `dxcom` CLI | `dx_com.compile()` Python API |
|---|---:|---:|
| JSON 전처리를 사용하는 표준 단일 이미지 입력 | 권장 | 지원 |
| 재현 가능한 셸 명령 또는 CI 단계 | 권장 | 지원 |
| 메모리 내 `onnx.ModelProto` | 불가 | 가능 |
| 사용자 정의 PyTorch 전처리/데이터 파이프라인 | 불가 | 가능 |
| 이미지가 아닌 캘리브레이션 데이터 | 불가 | 가능 |
| 다중 입력 모델 | 불가 | **필수** |
| 프로그래밍 방식의 실험 루프와 예외 처리 | 제한적 | 권장 |
| 애플리케이션에서 직접 QAT/체크포인트 제어 | 제한적 | 권장 |

Python API의 장점:

- ONNX 경로 또는 메모리 내 `onnx.ModelProto`를 전달할 수 있습니다.
- PyTorch `DataLoader`로 정확한 캘리브레이션 텐서를 제공할 수 있습니다.
- 다중 입력 모델과 이미지가 아닌 모델을 지원합니다.
- 셸 출력 파싱 없이 반복 가능한 실험 루프를 구성할 수 있습니다.
- Python 예외를 처리하고 구조화된 메타데이터를 기록할 수 있습니다.
- 컴파일을 모델 내보내기, 검증, 산출물 관리 코드에 통합할 수 있습니다.

흔한 실수를 막아 주는 두 가지 제약이 있습니다.

1. `config=`와 `dataloader=` 중 정확히 하나만 지정합니다. 둘은 함께 사용할 수 없습니다.
2. `dataloader=`를 사용하면 전처리는 `Dataset.__getitem__()` 안에서 수행되어야 하며 배포 시 전처리와 일치해야 합니다.

<!-- cell: inspect-python-api-heading src: c31ad53df9 -->
### 8.2 설치된 API 확인

DX-COM은 자체 컴파일러 환경에 설치되어 있으므로 노트북 커널은 `dx_com`을 직접 가져오지 않습니다. 다음 셀은 정확히 아래 명령을 하나의 `!` 줄로 이어 실행합니다.

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
python -c 'import inspect, dx_com; print(inspect.signature(dx_com.compile))'
```

<!-- cell: python-api-parameters src: ab00c54c22 -->
가장 중요한 매개변수는 다음과 같습니다.

| 매개변수 | 의미 |
|---|---|
| `model` | ONNX 파일 경로 또는 `onnx.ModelProto` |
| `output_dir` | DXNN과 리포트가 저장되는 디렉터리 |
| `config` | 표준 단일 입력 이미지 모델을 위한 JSON 기반 캘리브레이션 |
| `dataloader` | 프로그래밍 방식의 캘리브레이션 텐서. 이 다중 입력 실습에 필수 |
| `calibration_method` | 일반 컴파일에서는 `ema` 또는 `minmax` |
| `calibration_num` | DataLoader에서 소비할 샘플 수 |
| `opt_level` | 빠른 실험은 `0`, 전체 최적화는 `1` |
| `use_q_pro` | 자동 Q-PRO 활성화. 수동 `enhanced_scheme`과 함께 사용할 수 없음 |
| `quant_diagnosis` | 진단 HTML과 재개 가능한 QXNN 생성 |
| `gen_log`, `export_html` | 검토 및 디버깅 증거 보존 |

표준 단일 입력 호출은 다음과 같습니다.

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

사용자 정의 DataLoader를 사용할 때는 `config=`를 `dataloader=`로 바꾸세요. 둘을 동시에 전달하면 안 됩니다.

<!-- cell: multi-input-model-heading src: c96a18bc8d -->
### 8.3 Python API가 필요한 모델

스테레오 깊이, 옵티컬 플로, 피처 퓨전 네트워크는 보통 동기화된 두 개의 텐서를 입력으로 받습니다. 이 실습에서는 긴 모델 다운로드나 컴파일 없이 다중 입력 요구 사항을 보여 줄 수 있는 소형 스테레오 피처 퓨전 모델을 선택합니다. `dxcom` CLI의 JSON 로더는 모델 입력 하나만 지원하므로 두 텐서를 모두 제공할 수 없습니다. Python API는 정확한 ONNX 입력 이름을 키로 하는 딕셔너리를 반환할 수 있습니다.

```text
left image  -- preprocessing -- tensor "left_image"  --+
                                                        +--> stereo-fusion ONNX --> DXNN
right image -- preprocessing -- tensor "right_image" --+

Dataset.__getitem__()
    -> {"left_image": Tensor[C,H,W], "right_image": Tensor[C,H,W]}
DataLoader(batch_size=1)
    -> {"left_image": Tensor[1,C,H,W], "right_image": Tensor[1,C,H,W]}
```

대표 그래프는 피처 퓨전 전에 두 입력을 각각 인코딩합니다. 이렇게 하면 두 이미지를 하나의 입력으로 취급하는 것을 피하고 진정한 두 입력 DXNN 계약을 유지할 수 있습니다.

```text
left_image  -> Conv -> left_features  --+
                                          Add -> ReLU -> fused_features
right_image -> Conv -> right_features --+
```

이 실습은 소형 대표 스테레오 퓨전 그래프를 만듭니다. 실습이 API 계약에 집중할 수 있도록 의도적으로 작게 만들었습니다. 여기서 사용하는 합성 캘리브레이션 데이터는 컴파일 메커니즘만 검증합니다. 실제 제품용 스테레오 모델에는 동기화된 대표성 있는 좌/우 샘플이 필요합니다.

<!-- cell: multi-input-dataloader-heading src: 80154869ac -->
### 8.4 이름 기반 캘리브레이션 DataLoader 구현

각 데이터셋 항목에는 배치 차원이 없습니다. `DataLoader(batch_size=1)`이 배치 차원을 자동으로 추가합니다.

딕셔너리 키가 튜플보다 안전합니다.

- 딕셔너리: 입력이 ONNX 이름으로 매칭됩니다.
- 튜플/리스트: 입력이 모델 내부의 입력 순서로 매칭됩니다.

스크립트는 컴파일러를 호출하기 전에 첫 번째 배치를 검증하고, DXNN이 이미 있으면 컴파일을 건너뜁니다.

<!-- cell: run-multi-input-heading src: d764daa244 -->
### 8.5 DX-COM Python 환경으로 컴파일

다음 노트북 셀은 컴파일러 환경의 Python 인터프리터로 스크립트를 실행합니다. 다음과 동일합니다.

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
python compile_stereo_fusion.py
```

이 모델에는 의도적으로 `dxcom` CLI에 해당하는 명령이 없습니다. 컴파일러에 사용자 정의 DataLoader가 제공하는 두 텐서가 필요하기 때문입니다.

<!-- cell: inspect-multi-input-heading src: f6368021a6 -->
### 8.6 결과 확인

`dxparse -v`에 `left_image`와 `right_image` 두 개의 모델 입력이 표시되어야 합니다. 이는 컴파일된 산출물이 다중 입력 인터페이스를 유지하고 있음을 확인해 줍니다.

캘리브레이션 DataLoader는 ONNX 계약(`float32`, NCHW)과 일치합니다. DX-COM이 지원되는 입력 변환을 NPU 경로로 옮기므로 컴파일된 NPU 인터페이스는 `uint8`, NHWC로 보고될 수 있습니다. 배포할 때는 ONNX 레이아웃이 그대로 유지된다고 가정하지 말고, 생성된 DXNN에 대해 보고된 입력 형상, dtype, 전처리 안내를 따르세요.

<!-- cell: advanced-controls-heading src: 7bf0d30ccb -->
## 9. 고급 컴파일러 제어 옵션의 신중한 사용

이 옵션들은 특정한 그래프, 성능, 재현성 질문에 답하기 위한 것입니다. 범용적인 "최적화 추가" 묶음으로 활성화하지 마세요.

| CLI 옵션 | Python API 매개변수 | 용도 | 주요 주의 사항 |
|---|---|---|---|
| `--opt_level {0,1}` | `opt_level` | 컴파일 시간과 전체 최적화 사이의 절충 | 통제된 기준선과 비교 |
| `--aggressive_partitioning` | `aggressive_partitioning` | 더 많은 NPU 파티셔닝 탐색 | 실험적 기능. 출력과 엔드투엔드 지연 시간 검증 필요 |
| `--compile_input_nodes` | `input_nodes` | 선택한 ONNX 연산자 노드에서 시작 | 컴파일된 인터페이스가 바뀜 |
| `--compile_output_nodes` | `output_nodes` | 선택한 ONNX 연산자 노드에서 종료 | 남은 작업이 호스트로 이동 |
| `--float64_calibration` | `float64_calibration` | CPU 간 캘리브레이션 결정성 향상 | 연산 및 메모리 비용 증가 |
| `--gen_log` | `gen_log` | 상세한 컴파일러 로그 보존 | 릴리스 산출물과 함께 보관 |
| `--export_html` | `export_html` | 독립형 요약 생성 | 태스크 검증을 대신하지 않음 |

서브그래프 컴파일에는 텐서 이름이 아니라 **ONNX 연산자 노드 이름**을 사용하세요. 결과로 생기는 입출력 계약과 호스트 측에 남는 연산을 문서화하세요.

<!-- cell: release-validation-heading src: 82c9e17d3c -->
## 10. 릴리스 검증 기록 작성

컴파일러가 성공적으로 종료되는 것은 필요조건이지만, 릴리스를 위한 충분한 증거는 아닙니다.

| 영역 | 최소 증거 |
|---|---|
| 소스 모델 | 내보내기 명령, 프레임워크/버전, ONNX 검사기 결과 |
| 캘리브레이션 | 데이터셋 리비전, 샘플 수, 전처리, 방법 |
| 컴파일러 | DX-COM 버전, 전체 명령/API 인수, 로그, HTML 리포트 |
| 정확도 | ONNX와 모든 DXNN 후보에 대한 동일한 태스크 지표 |
| 성능 | 워밍업 정책, 측정 시간, 장치, 처리량, 지연 시간, CPU 부하 |
| 통합 | 입출력 형상, 전처리 담당 주체, 디코더 계약 |
| 재현성 | 산출물 해시와 격리된 출력 디렉터리 |

더미 입력을 사용하는 `dxrun`은 처리량 확인에 유용하지만, 태스크 정확도나 전체 애플리케이션 지연 시간 측정을 대신할 수는 없습니다.

<!-- cell: 20d9ebdb-92ef-4b3e-9c73-c4bf02326bd8 src: 175675be1b -->
## 11. 문제 해결

| 증상 | 표시되는 내용 | 원인 | 해결 |
|---|---|---|---|
| `--use_q_pro`와 `--enhanced_scheme` 충돌 | 옵션 충돌에 관한 컴파일러 오류 | 자동 Q-PRO 선택과 수동 Q-PRO 선택을 동시에 요청함 | 컴파일마다 둘 중 하나만 사용 |
| `--checkpoint` 파일을 찾을 수 없음 | `.qxnn` 파일에 대한 `FileNotFoundError` | `quant_diagnosis/*.qxnn`을 생성하는 5절의 진단 컴파일을 건너뛰었거나 실패함 | 먼저 5절을 실행하고 체크포인트 경로가 출력되는지 확인 |
| 7절에서 QAT가 실행되지 않음 | `configs/squeezenet.qat.template.json`만 생성됨 | 의도된 동작: 샘플 이미지로는 유용한 모델을 학습할 수 없음 | 데이터셋 경로를 대표성 있는 학습 데이터로 교체하고 7절의 명령을 터미널에서 실행 |
| JSON 변경이 반영되지 않음 | `Skip compilation: found ...` | 출력 디렉터리에 이미 DXNN이 있음 | `workspace/outputs/<dir>`을 삭제하고 다시 실행 |
| `dxparse`가 없음 | `[MISSING] dx_rt` | DX-RT가 설치되지 않음 | DX-Runtime 설치(튜토리얼 01의 3절) |

<!-- cell: summary src: 0316911a19 -->
## 12. 요약

### 12.1 고급 정확도 복구 지도

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

### 12.2 고급 워크플로 대시보드

| 워크플로 | 사용 시점 | 주요 입력 | 주요 결과 |
|---|---|---|---|
| Q-Lite 기준선 | 첫 번째 통제된 PTQ 실험 | ONNX, JSON, 대표성 있는 캘리브레이션 데이터 | 기준선 DXNN과 증거 |
| 자동 Q-PRO | Q-Lite가 정확도 목표에 미달 | 동일한 모델, 데이터, 전처리, 검증 프로토콜 | 향상 PTQ 후보 |
| 진단과 QXNN 재개 | ONNX-DXNN 정확도 차이가 재현 가능 | 진단 리포트와 QXNN 체크포인트 | 더 빠른 단일 변수 실험 |
| Q-Master QAT | PTQ 방법으로도 목표에 미달 | 대표성 있는 학습 및 검증 데이터 | 학습된 양자화 후보 |
| Python API | 캘리브레이션에 사용자 정의 텐서, 이미지가 아닌 데이터, 다중 입력이 필요 | ONNX와 이름을 키로 사용하는 PyTorch DataLoader | 프로그래밍 방식으로 컴파일된 DXNN |

### 12.3 완료한 실험

| 실험 | 통제된 비교 또는 계약 | 생성된 증거 |
|---|---|---|
| SqueezeNet Q-Lite 대 Q-PRO | 동일한 ONNX, 캘리브레이션, 전처리, 컴파일러 설정 | 별도의 DXNN 파일, 로그, HTML 리포트 |
| 양자화 진단 | 설정을 바꾸기 전에 정확도 손실 후보를 재현 | 진단 HTML과 재사용 가능한 QXNN 체크포인트 |
| QXNN 재개 | 같은 체크포인트에서 IQR 재캘리브레이션 대 자동 Q-PRO | 더 빠른 양자화 실험 |
| Q-Master 준비 | 스모크 테스트 설정 대 의미 있는 QAT 학습 | 검토된 QAT 템플릿과 검증 요구 사항 |
| 스테레오 퓨전 Python API | 하나의 DataLoader가 제공하는 두 개의 이름 있는 입력 | `dxparse`로 확인한 두 입력 DXNN |

### 12.4 완료 체크리스트

- [ ] 재사용 가능한 Q-Lite 기준선 구축
- [ ] 통제된 PTQ 실험으로 자동 Q-PRO 실행
- [ ] 양자화 진단 리포트 생성 및 해석
- [ ] IQR과 Q-PRO 실험에 QXNN 체크포인트 재사용
- [ ] Q-Master 스모크 테스트와 의미 있는 QAT 구분
- [ ] 워크플로 요구 사항에 따라 CLI 또는 Python API 선택
- [ ] 이름을 키로 사용하는 다중 입력 캘리브레이션 DataLoader 구현
- [ ] 두 입력 DXNN 컴파일 및 검사
- [ ] 릴리스 검토에 필요한 증거 식별
- [ ] 튜토리얼 샘플을 대표성 있는 제품 데이터로 교체
- [ ] 동일한 홀드아웃 태스크 지표로 ONNX와 모든 DXNN 후보 비교
- [ ] 전체 애플리케이션의 지연 시간, 처리량, 호스트 CPU 사용량 측정

### 12.5 선택 방법

| 필요한 작업 | 시작점 |
|---|---|
| 재현 가능한 INT8 참조 수립 | Q-Lite |
| 학습 없이 PTQ 정확도 복구 | 자동 Q-PRO |
| 양자화 손실이 시작되는 지점 찾기 | 양자화 진단 |
| 전체 ONNX 컴파일을 반복하지 않고 다른 양자화 설정 테스트 | QXNN 재개 |
| PTQ 옵션을 모두 소진한 뒤 정확도 복구 | Q-Master QAT |
| 사용자 정의, 이미지가 아닌, 메모리 내, 또는 이름 있는 다중 입력 제공 | `dx_com.compile()` Python API |
| 표준 모델을 위한 재현 가능한 셸 또는 CI 컴파일 단계 구축 | `dxcom` CLI |

> **기억하세요:** 컴파일러 성공 메시지, 생성된 DXNN, 빠른 더미 입력 벤치마크는 부분적인 증거일 뿐입니다. 릴리스 결정에는 대표성 있는 데이터, ONNX와 DXNN에 대한 동일한 홀드아웃 태스크 지표, 전체 파이프라인 성능 측정, 재현 가능한 산출물이 필요합니다.

### 12.6 이 워크플로를 내 모델에 적용하기

1. 소스 ONNX, 전처리, 데이터셋 리비전, 검증 지표를 고정합니다.
2. 양자화 설정을 바꾸기 전에 Q-Lite 기준선을 수립합니다.
3. 한 번에 하나의 변수만 바꾸고 모든 결과를 격리된 출력 디렉터리에 보관합니다.
4. 측정된 정확도가 요구할 때만 Q-Lite에서 Q-PRO, 진단/QXNN 재개, 마지막으로 Q-Master 순으로 단계를 올립니다.
5. 명령 또는 API 인수, 컴파일러 로그, 리포트, 해시, 정확도, 성능 결과를 릴리스 후보와 함께 보관합니다.

> **다음:** 튜토리얼 06으로 이동해 DX-Runtime으로 컴파일된 모델을 실행해 보세요. CLI 도구, Python 및 C++ API, 프로파일링, 릴리스 검증을 다룹니다.
