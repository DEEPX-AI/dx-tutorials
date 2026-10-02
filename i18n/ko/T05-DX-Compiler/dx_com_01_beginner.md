<!-- i18n source: notebooks/T05-DX-Compiler/dx_com_01_beginner.ipynb -->
<!-- i18n lang: ko -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 263bea17 src: 05ca91a778 -->
# DEEPX Tutorial 05-1 - DX-COM 초급

이 노트북은 작고 재현 가능한 예제로 ONNX에서 DXNN까지의 전체 워크플로를 소개합니다.

## 학습 목표

이 튜토리얼을 마치면 다음을 할 수 있습니다.

1. DX-COM의 전체 프로세스를 이해하고,
2. DX-COM 설치를 검증하고,
3. MobileNetV2 ONNX 모델을 내보내고 검증하고,
4. 캘리브레이션 설정을 만들고,
5. DXNN 모델을 컴파일하고 검사하고,
6. DEEPX Model Zoo에서 ONNX 모델과 JSON 설정을 다운로드해 컴파일할 수 있습니다.

이 노트북은 SDK 소스 트리를 수정하지 않습니다. 생성되는 모든 파일은 git이 무시하는 `<dx-tutorials>/notebooks/T05-DX-Compiler/workspace` 아래에 저장됩니다.

<!-- cell: 09404f54 src: 15d2514cfa -->
## 코스 맵

| 노트북 | 주요 내용 |
|---|---|
| 초급 | 설치, ONNX 검증, JSON 기초, 첫 컴파일, Model Zoo |
| 중급 | 캘리브레이션 품질, 하드웨어 PPU, YOLO26 TopK 최적화 |
| 고급 | Q-PRO, 진단, QXNN 재개, QAT, Python API, 고급 컴파일러 제어 |

DX-COM 설정과 캘리브레이션을 이미 이해하고 있지 않다면 노트북을 순서대로 진행하세요.

<!-- cell: dx-com-01-prerequisites src: 70ef62da25 -->
## 사전 요구 사항

- 튜토리얼 01 완료: `dx-compiler/venv-dx-compiler-local`에 DX-COM이 샘플 캘리브레이션 데이터셋과 함께 설치되어 있어야 하며, 5절과 6절의 `dxparse` 및 `dxrun` 셀을 위해 DX-RT와 DEEPX NPU가 필요합니다.
- 다운로드: 3절의 내보내기에 사용하는 CPU 전용 PyTorch 휠과 ONNX 도구(처음 설치 시 수백 MB), Model Zoo의 ResNet50 ONNX(97 MB)
- `sudo` 불필요. 소요 시간은 약 20분이며, 두 번의 컴파일은 각각 1분 이내에 끝납니다.
- 이 튜토리얼이 생성하는 모든 파일은 git이 무시하는 `notebooks/T05-DX-Compiler/workspace/`에 저장됩니다.

2절의 설정 셀은 SDK 위치를 찾고, 이 요구 사항을 확인한 뒤 상태 표를 출력합니다. `MISSING`으로 표시된 항목에는 그것을 제공하는 단계가 함께 안내됩니다.

<!-- cell: dxcom-workflow-overview src: 44853a9f8f -->
## 1. 컴파일 워크플로

전체 컴파일 프로세스는 다음 세 단계로 구성됩니다.

1. **사전 학습된 모델 확보** — 모델을 직접 학습하거나 PyTorch 같은 프레임워크에서 호환되는 사전 학습 모델을 가져옵니다.
2. **모델을 ONNX로 변환** — 프레임워크 모델을 ONNX로 내보내고 입력 이름, 입력 형상, 연산자, 출력을 검증합니다.
3. **ONNX를 DXNN으로 컴파일** — JSON 설정과 대표 캘리브레이션 데이터를 사용해 DX-COM으로 DEEPX NPU용 모델을 생성합니다.

<img src="assets/dx-com-workflow.jpg" style="max-width: 800px; width: 100%;" alt="사전 학습 모델에서 ONNX, DXNN으로 이어지는 DX-COM 컴파일 워크플로">

이 초급 튜토리얼은 MobileNetV2로 같은 워크플로를 따라갑니다: PyTorch → ONNX → DXNN.

자세한 내용은 DX-Compiler 사용자 가이드를 참고하세요. [다운로드](https://developer.deepx.ai/download/?id=581)

> **참고:** 사용자 가이드를 다운로드하려면 먼저 https://developer.deepx.ai/ 에 로그인해야 합니다.

<!-- cell: dxcom-artifact-glossary src: 6e807062e7 -->
### 1.1 컴파일 과정에서 사용하고 생성하는 파일

다음 산출물을 구분해서 기억해 두세요. 각각 워크플로에서 다른 역할을 합니다.

| 산출물 | 역할 | 사용 또는 생성 주체 |
|---|---|---|
| `.pt` 같은 프레임워크 모델 | 원본 프레임워크의 학습된 가중치와 네트워크 정의 | ONNX 내보내기 시 사용 |
| `.onnx` | DX-COM이 읽는 프레임워크 독립적 모델 그래프 | DX-COM 입력 |
| 컴파일러 `.json` | 입력 형상, 캘리브레이션, 전처리, 양자화 및 선택적 컴파일러 설정 | DX-COM 입력 |
| 캘리브레이션 데이터셋 | 양자화 범위를 추정하는 데 사용하는 대표 샘플 | 캘리브레이션 중 DX-COM이 읽음 |
| `.dxnn` | DEEPX 런타임이 실행하는 컴파일된 모델 | DX-COM 주요 출력 |
| `compiler.log` | 상세 컴파일 메시지, 경고, 오류 | `--gen_log`로 생성 |
| `*_summary.html` | 그래프와 컴파일러 정보를 담은 시각적 컴파일 리포트 | `--export_html`로 생성 |

`.onnx`, 컴파일러 `.json`, 캘리브레이션 데이터셋은 컴파일 입력입니다. `.dxnn`, 로그, HTML 리포트는 출력입니다.

<img src="assets/dx-compile-progress.png" style="max-width: 1000px; width: 100%;" alt="필요한 파일과 함께 보는 DX-COM 컴파일">

<!-- cell: f668eb12 src: 50dcc961e3 -->
## 2. 요구 사항과 워크스페이스

DX-COM은 **x86-64 Linux**를 지원합니다. **배치 크기 1**의 정적 **ONNX 입력** 형상을 사용하세요. 실용적인 호스트 사양은 최소 **16 GB RAM**과 8 GB의 여유 저장 공간입니다.

이 튜토리얼은 SDK 설치에서 두 가지를 필요로 합니다.

| 요구 사항 | 제공 주체 | `MISSING`으로 보고되는 경우 |
|---|---|---|
| `dx_com` (`venv-dx-compiler-local` 안의 `dxcom` 컴파일러) | `./dx-compiler/install.sh` | 튜토리얼 01의 2절 또는 아래 2.1절을 따르세요 |
| `calibration_dataset` (`dx_com/` 아래의 샘플 이미지) | `dx-compiler/example/2-download_sample_calibration_dataset.sh` | 터미널에서 해당 스크립트를 한 번 실행하세요 |

다음 셀은 `tutorial_paths.py`를 통해 DX-All Suite를 찾고(환경 변수, `config.json`, 자동 탐지 순), 찾은 내용을 출력하고, 이 튜토리얼의 워크스페이스를 준비합니다. 요구 사항이 누락되어도 멈추지 않으며, 멈추는 확인은 설치 옵션을 설명한 뒤 2.2절에 있습니다.

<!-- cell: 16e77545 src: 141db488f9 -->
### 2.1 DX-COM 설치 옵션

DX-COM은 Python 패키지이며, **전용 Python 가상 환경에 설치하는 것을 강력히 권장합니다**. 가상 환경은 DX-COM과 그 의존성을 시스템 Python 및 Jupyter 커널과 분리하므로 버전 충돌이 줄어들고 컴파일러 환경을 재현하거나 제거하기 쉬워집니다.

가장 간단한 독립 설치 방법은 [PyPI](https://pypi.org/project/dx-com/)에 게시된 패키지를 사용하는 것입니다. 별도의 터미널에서 다음 명령을 실행하세요.

```bash
# 1. Create a dedicated environment.
python3 -m venv ~/venv-dx-com

# 2. Activate it in the current terminal.
source ~/venv-dx-com/bin/activate

# 3. Install DX-COM from PyPI.
python -m pip install --upgrade pip
pip install dx-com

# 4. Confirm that the virtual environment owns the commands.
which python
which dxcom
dxcom --version
```

두 `which` 명령은 `~/venv-dx-com/` 아래의 경로를 출력해야 합니다. 새 터미널을 열 때마다 `source ~/venv-dx-com/bin/activate`를 다시 실행하고, 작업을 마치면 `deactivate`를 실행하세요.

이 노트북은 `dx-all-suite/dx-compiler/install.sh`가 `venv-dx-compiler-local`에 만든 전용 환경을 사용하므로, 컴파일러가 Jupyter 커널의 Python 환경에 의존하지 않습니다. 위의 PyPI 절차는 DX-COM을 별도로 설치할 때 권장하는 대안입니다.

> `dx-com`을 설치하면 컴파일러 패키지와 `dxcom` 명령이 제공됩니다. 완전한 컴파일-배포 워크플로를 위해서는 나머지 DEEPX SDK와 지원되는 Linux x86-64 호스트가 여전히 필요합니다.

<!-- cell: c7a528a2 src: dd9c03046a -->
### 2.2 DX-COM 설치 검증

다음 셀은 먼저 2절의 두 요구 사항이 모두 갖춰져 있는지 확인합니다. 하나라도 없으면 실행해야 할 정확한 단계를 안내하며 멈추므로, 해당 단계를 완료한 뒤 셀을 다시 실행하세요.

셀의 나머지 부분은 다음 터미널 명령과 동일합니다.

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
dxcom --version
dxcom -h
```

`<DX_ALL_SUITE_DIR>`은 위 설정 셀이 출력한 위치입니다.

<!-- cell: f8556d44 src: 6e8fd67cf4 -->
## 3. 작은 ONNX 모델 내보내기

이 프로젝트는 `uv`로 관리되는 Jupyter 환경을 사용하며, 이 환경에는 `pip` 모듈이 없을 수 있습니다.

`uv pip install --python "{sys.executable}"`은 현재 Jupyter 커널에 패키지를 명시적으로 설치합니다. 

이 패키지들은 ONNX를 내보내고 검사하는 데 사용합니다. DX-COM 자체는 계속 `DX_COMPILER_VENV`에서 실행됩니다.

`onnxscript`는 최신 torch 버전(2.9 이상)의 ONNX 내보내기 도구에 필요합니다. `--extra-index-url`은 `uv`가 CPU 전용 PyTorch 휠을 사용하도록 지정합니다. 내보내기에는 GPU가 필요 없고, CPU 휠은 수 GB 대신 수백 MB입니다. 이미 설치된 torch는 그대로 둡니다.

<!-- cell: 206f9840 src: a88480414f -->
### 3.1 ONNX 계약 검증

JSON 파일의 입력 이름은 ONNX 그래프 입력 이름과 정확히 일치해야 합니다. 형상 추론과 `onnx.checker`는 컴파일 전에 많은 내보내기 오류를 잡아냅니다.

<!-- cell: 96ab8536 src: 372ae5a1e1 -->
ONNX 파일을 [Netron](https://netron.app/)에서 열어 텐서 이름, 형상, 연산자를 확인할 수도 있습니다. 컴파일된 `.dxnn`은 `--export_html`로 생성한 DX-COM HTML 요약이나 DX-TRON(튜토리얼 01, 2.3절)을 사용하세요. DX-TRON은 SDK와 함께 계속 배포되지만 더 이상 적극적으로 유지보수되지 않으므로 HTML 요약을 권장합니다.

<!-- cell: 0898f061 src: 1303aaeedb -->
## 4. 캘리브레이션 설정 만들기

캘리브레이션 이미지는 양자화 과정에서 높은 정확도를 유지하는 데 매우 중요합니다.

<img src="assets/calibration.jpeg" style="max-width: 1000px;" alt="양자화에 사용되는 대표 캘리브레이션 이미지">

설정은 모델 입력 계약과 원본 이미지가 입력 텐서로 변환되는 방식을 모두 기술합니다.

| 필드 | 용도 |
|---|---|
| `inputs` | 정확한 ONNX 입력 이름과 정적 형상 |
| `calibration_method` | 양자화 범위를 추정하는 데 사용하는 옵저버 |
| `calibration_num` | 사용할 대표 샘플 수 |
| `default_loader.dataset_path` | 캘리브레이션 입력이 들어 있는 디렉터리 |
| `preprocessings` | 순서가 있는 이미지-텐서 변환 |

전처리 순서가 중요합니다. 값은 원본 모델을 학습하고 평가할 때 사용한 전처리와 일치해야 합니다.

<!-- cell: mobilenet-preflight-checklist src: 1110d1b61b -->
### 4.1 컴파일 전 사전 점검 체크리스트

오래 걸릴 수 있는 컴파일을 시작하기 전에 다음을 확인하세요.

- ONNX 파일이 `onnx.checker`를 통과하는지,
- 모든 입력 차원이 정적인지,
- ONNX 입력 이름과 형상이 컴파일러 JSON과 정확히 일치하는지,
- 캘리브레이션 디렉터리가 존재하고 지원되는 파일이 들어 있는지,
- 캘리브레이션 전처리가 모델의 학습 및 평가 전처리와 일치하는지,
- 선택한 DX-COM 실행 파일이 존재하는지 확인합니다.

> **핵심 요구 사항 — 배치 크기는 `1`이어야 합니다.** ONNX 입력과 JSON의 `inputs` 항목 모두 `[1, 3, 224, 224]`처럼 `1`로 시작해야 합니다.
>
> DXNN 모델은 DEEPX NPU의 단일 샘플 실행 계약을 대상으로 하므로, 다른 값이거나 동적인 배치 차원은 컴파일 및 런타임 계약과 맞지 않습니다. 처리량을 높이려면 여러 추론 요청을 제출하거나 비동기 파이프라인을 사용하세요. 모델의 배치 차원을 늘리지 마세요.

다음 셀은 이 체크리스트를 실행 가능한 검사로 바꾸고, 요구 사항이 충족되지 않으면 DX-COM 실행 전에 멈춥니다.

<!-- cell: be4871a6 src: 4880623ba4 -->
## 5. DXNN으로 컴파일

### 5.1 컴파일

`--gen_log`는 컴파일러 로그를 보존하고 `--export_html`은 모델 요약을 생성합니다. 이 명령은 오류를 계속 볼 수 있게 하고 전용 출력 디렉터리를 사용합니다. 출력 디렉터리에 이미 DXNN 파일이 있으면 셀은 `Skip compilation: found ...`를 출력하고 `dxcom`을 다시 실행하지 않습니다. 강제로 다시 컴파일하려면 디렉터리를 삭제하세요.

다음 셀은 다음 터미널 명령과 동일합니다.

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
dxcom -m models/mobilenet_v2.onnx \
      -c configs/mobilenet_v2.json \
      -o outputs/mobilenet_v2_q_lite \
      --gen_log \
      --export_html
```

> **코드 셀 끝의 `2>&1 | sed ... | grep ...`에 대해:** `dxcom`은 Jupyter가 렌더링할 수 없는 커서 이동 이스케이프 코드로 진행 막대를 그리는데, 필터가 없으면 수백 줄의 빈 줄로 표시됩니다. 이 필터는 진행 막대 줄만 제거하며, 모든 `[INFO]`, `[WARNING]`, `[ERROR]` 메시지는 그대로 표시됩니다. 터미널에서는 필터를 생략해도 됩니다. `--gen_log`는 전체 출력을 `compiler.log`에 보관합니다. 이 노트북의 모든 컴파일 셀에서 같은 필터를 사용합니다.

<!-- cell: c56da884 src: 824e864a9a -->
### 5.2 검사와 벤치마크

`dxparse -v`는 컴파일된 모델 구조와 텐서 메타데이터를 보고합니다. 이어서 `dxrun --use-ort -t 5`가 합성 입력으로 5초 동안 벤치마크를 수행합니다.

<!-- cell: 85f6c539 src: 70726cba9e -->
컴파일러가 `[INFO] Added nodes`를 보고하면 어떤 전처리 연산이 그래프에 삽입되었는지 확인하세요. 런타임 애플리케이션에서 같은 정규화, 색상 변환, 전치를 다시 적용하지 마세요.

<!-- cell: 8bc0a25d-367f-411d-96b4-5a0ca6d02ad4 src: cfccad8843 -->
### 5.3 성능 벤치마크와 정확도 평가는 다릅니다

| | `dxrun` 합성 벤치마크 | 데이터셋 정확도 평가 |
|---|---|---|
| 입력 | 생성된 더미 입력 | 실제 레이블이 있는 검증 데이터셋 |
| 주요 결과 | 런타임 처리량과 지연 시간 | Top-1, Top-5, mAP, mIoU 같은 태스크 지표 |
| 확인하는 것 | 컴파일된 모델이 실행 가능한지와 런타임 성능 | 전처리, 추론, 후처리, 예측 품질 |
| 증명하지 **않는** 것 | 모델 정확도 | 최종 애플리케이션 워크로드에서의 배포 지연 시간 |

> **컴파일 성공과 빠른 `dxrun` 결과는 모델 정확도를 증명하지 않습니다.**

정확도 측정은 [DEEPX-AI/dx-modelzoo](https://github.com/DEEPX-AI/dx-modelzoo)와 [모델 평가 가이드](https://github.com/DEEPX-AI/dx-modelzoo/blob/main/docs/source/guides/evaluation.md)를 참고하세요. DX-ModelZoo는 모델 설정을 통해 데이터셋, 전처리, 런타임 프로필, 후처리, 평가기를 연결하고 태스크별 지표를 보고합니다. ONNX와 DXNN 결과를 비교할 때는 같은 레이블이 있는 검증 세트를 사용하세요.

<!-- cell: 74b9a787 src: d8d71d6109 -->
## 6. DEEPX Model Zoo의 모델 컴파일

[DEEPX Model Zoo](https://developer.deepx.ai/modelzoo/)는 검색 가능한 모델 메타데이터와 다운로드 가능한 산출물을 제공하며, 여기에는 ONNX 모델, DXNN 모델, 지원되는 양자화 변형별 컴파일러 JSON 파일이 포함됩니다.

이 실습에서는 ONNX 파일이 작은 **Resnet50**을 사용합니다. 더 큰 모델도 워크플로는 동일합니다.

1. 모델과 양자화 변형을 선택하고,
2. 해당 ONNX와 짝이 되는 JSON을 다운로드하고,
3. ONNX 입력 계약을 확인하고,
4. 환경에 따라 달라지는 JSON 값을 조정하고,
5. 새 출력 디렉터리에 컴파일합니다.

<!-- cell: 0d7e5ebb-7b55-4331-ad08-ef736c05424c src: 683283911f -->
### 6.1 Resnet50 ONNX 파일과 dxcom 설정 파일(json) 다운로드

<!-- cell: 250b29cb src: a1a1b7cc23 -->
### 6.2 dxcom 설정 파일(json)의 캘리브레이션 경로 수정

Model Zoo JSON 파일에는 게시된 모델을 만들 때 사용한 캘리브레이션 설정이 들어 있습니다. 그 데이터셋 경로는 빌드 환경의 것이므로 보통 사용자의 컴퓨터에는 존재하지 않습니다. 모델별 전처리는 유지하되, 데이터셋 경로는 로컬의 대표 데이터셋으로 바꾸세요.

여기서는 컴파일러 워크플로를 재현 가능하게 만들기 위해서만 SDK 샘플 이미지를 사용합니다. 정확도 판단을 위해서는 실제 배포 도메인의 샘플을 사용하세요.

다운로드한 파일은 `configs/resnet50_224x224.modelzoo.json`으로 변경 없이 유지되고, 수정된 사본은 `configs/resnet50_224x224.local.json`에 기록됩니다. 두 파일을 분리해 두면 다운로드 셀을 다시 실행해도 편집 내용을 덮어쓰지 않고, `wget --continue`가 편집된 파일에 이어 붙이는 일도 없습니다.

<!-- cell: b45307f5-9e35-4bcb-9e33-7814434037be src: 3464f02b41 -->
### 6.3 컴파일

다음 셀은 다음 터미널 명령과 동일합니다.

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
dxcom -m models/resnet50_224x224.onnx \
      -c configs/resnet50_224x224.local.json \
      -o outputs/resnet50_224x224_q_lite \
      --gen_log \
      --export_html
```

5.1과 마찬가지로, 출력 디렉터리에 이미 DXNN 파일이 있으면 셀은 `dxcom`을 건너뛰고 진행 막대는 필터링됩니다.

<!-- cell: dbd331d3-99c3-4615-acb9-4ea826e758fd src: 29085b4480 -->
### 6.4 검사와 벤치마크

<!-- cell: 9236bfd1 src: 5dfd2e5d63 -->
### 6.5 최신 HTML 컴파일 리포트 열기

DX-COM은 `--export_html`이 활성화되면 HTML 모델 요약을 생성합니다. 다음 셀은 이 튜토리얼의 출력 디렉터리에서 가장 최근에 생성된 리포트를 찾아 새 브라우저 탭에서 여는 버튼을 표시합니다.

<!-- cell: eb56f5d4 src: 9857c6dd8b -->
## 7. 문제 해결

| 증상 | 표시되는 내용 | 원인 | 해결 |
|---|---|---|---|
| `dxcom: command not found` | 터미널에서 | DX-COM 환경이 활성화되지 않음 | 먼저 `source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate`를 실행. 셀은 각 `!` 줄에서 이를 수행함 |
| 입력 이름 오류 | `KeyError` 또는 `input ... not found in model` | JSON의 `inputs`가 `model.graph.input`과 일치하지 않음 | 4절의 PASS 검사와 이름을 비교하고 JSON을 수정 |
| 동적 형상 또는 배치 오류 | `Dynamic shape is not supported` 또는 배치 크기 오류 | ONNX 내보내기에 동적 축을 사용했거나 배치 크기 > 1 | 배치 크기 1의 정적 형상으로 내보내기 |
| 캘리브레이션 파일 없음 | `dataset_path` 아래에 `No images found` | 잘못된 경로 또는 파일 확장자 | JSON의 `dataset_path`와 `file_extensions` 확인 |
| JSON 변경이 반영되지 않음 | `Skip compilation: found ...` | 출력 디렉터리에 이미 DXNN이 있음 | `workspace/outputs/<dir>`을 삭제하고 컴파일 셀을 다시 실행 |
| ONNX 내보내기 실패 | `ModuleNotFoundError: No module named 'onnxscript'` | torch 2.9+의 ONNX 내보내기 도구에 `onnxscript`가 필요함 | 3절의 설치 셀을 다시 실행하면 `onnxscript`가 설치됨. 또는 `torch.onnx.export`에 `dynamo=False`를 전달 |
| 컴파일 셀이 venv를 찾지 못함 | `{DX_COMPILER_VENV}/bin/activate: No such file or directory` | 앞선 셀이 실패해 명령의 변수가 정의되지 않았고, IPython이 명령 전체를 확장하지 않은 채로 둠 | 위로 스크롤해 처음 실패한 셀을 찾아 수정하고 다시 실행한 뒤 컴파일 셀을 다시 실행 |
| `wget`이 `416 Requested Range Not Satisfiable`를 보고 | 다운로드 셀을 다시 실행할 때 | `--continue`가 이미 완전한 파일을 발견함 | 할 일 없음. 파일이 완전함 |
| 런타임에서 잘못된 예측 | `dxcom`은 성공했지만 정확도가 낮음 | 애플리케이션의 전처리가 JSON과 다름 | 채널 순서, 스케일링, 정규화, 리사이즈 정책을 확인하고 레이블이 있는 데이터로 정확도를 측정 |

각 실험은 별도의 출력 디렉터리에 보관하세요. 그러면 컴파일러 리포트와 바이너리를 추적할 수 있습니다.

<!-- cell: de985ffc-1b44-4182-94cd-8a16c7dfb545 src: 335b61b8ab -->
## 8. 요약

### 8.1 완료한 워크플로

**PyTorch 모델**  
→ **ONNX 내보내기와 검증**  
→ **JSON 설정과 캘리브레이션 데이터**  
→ **DX-COM 컴파일**  
→ **DXNN 검사와 벤치마크**

<img src="assets/dx-compile-progress.png"
     style="max-width: 1000px; width: 100%;"
     alt="DX-COM 컴파일 워크플로">

### 8.2 입력과 출력

| 단계 | 주요 산출물 | 검증 |
|---|---|---|
| 모델 준비 | `mobilenet_v2.onnx` | `onnx.checker` |
| 컴파일러 설정 | `mobilenet_v2.json` | 사전 점검 체크리스트 |
| 컴파일 | `mobilenet_v2.dxnn` | 출력 파일 확인 |
| 구조 검사 | DXNN 메타데이터 | `dxparse -v` |
| 성능 확인 | 더미 입력 벤치마크 | `dxrun` |
| 정확도 평가 | 레이블이 있는 검증 데이터셋 | DX-ModelZoo |

### 8.3 완료 체크리스트

- [ ] PyTorch 모델을 ONNX로 내보내기
- [ ] ONNX 입력 이름과 정적 형상 확인
- [ ] 배치 크기가 `1`인지 확인
- [ ] 캘리브레이션 설정 만들기
- [ ] ONNX를 DXNN으로 컴파일
- [ ] `dxparse`로 컴파일된 모델 검사
- [ ] `dxrun`으로 런타임 성능 측정
- [ ] 레이블이 있는 데이터셋으로 모델 정확도 평가

> **기억하세요:** 컴파일 성공과 빠른 `dxrun` 결과는 모델 정확도를
> 증명하지 않습니다. 정확도는 대표성 있는 레이블 데이터로
> 별도로 측정해야 합니다.

### 8.4 다음 단계

**중급 튜토리얼**로 이어서 다음을 학습하세요.

- 캘리브레이션 데이터셋 설계
- Type 0과 Type 1 하드웨어 PPU
- PPU와 비PPU 성능 비교
- YOLO26 TopK 기반 최적화
