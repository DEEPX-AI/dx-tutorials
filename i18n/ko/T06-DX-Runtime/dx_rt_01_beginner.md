<!-- i18n source: notebooks/T06-DX-Runtime/dx_rt_01_beginner.ipynb -->
<!-- i18n lang: ko -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 56ed92cd src: dfd7b6b490 -->
# DEEPX Tutorial 06-1 - DX-RT 초급

이 노트북은 컴파일된 DXNN 모델을 DEEPX NPU에서 실행하는 런타임 계층을 소개합니다.

## 학습 목표

이 튜토리얼을 마치면 다음을 할 수 있습니다.

- DX-RT가 하는 일과 애플리케이션이 책임져야 하는 일을 설명한다.
- 설치된 런타임 도구와 NPU 상태를 확인한다.
- DXNN 모델의 입력, 출력, 태스크 그래프를 살펴본다.
- 단일 실행, 최대 처리량, 목표 FPS 모드를 구분한다.
- `dxrun`으로 통제된 더미 입력 벤치마크를 실행한다.
- CPU 태스크가 포함된 모델을 NPU 전용과 ORT 활성화 방식으로 실행해 비교한다.
- 검사, 추론, 모니터링에 알맞은 CLI 도구를 선택한다.

이 튜토리얼은 SDK 소스 트리를 수정하지 않습니다. 튜토리얼이 만드는 모든 산출물과 로그는 `<dx-tutorials>/notebooks/T06-DX-Runtime/workspace` 아래에 저장됩니다.

<!-- cell: a6368870 src: a2bad27896 -->
## 코스 맵

| 노트북 | 주요 내용 |
|---|---|
| 초급 | DX-RT의 역할, 장치 확인, DXNN 검사, CLI 추론 |
| 중급 | Python 및 C++ API, 동기/비동기 실행, 배치, 버퍼 |
| 고급 | 리소스 바인딩, 프로파일링, 모니터링, 다중 입력/메모리 로딩, 릴리스 검증 |

DXNN 모델 계약과 DX-RT CLI를 이미 이해하고 있지 않다면 노트북을 순서대로 진행하세요.

<!-- cell: ffd44045-7886-415c-b0c5-017237e17a82 src: 427df03b3b -->
## 사전 요구 사항

- DX-RT가 설치되어 있고(튜토리얼 01 3절) DEEPX NPU가 `/dev/dxrt0`으로 보여야 합니다.
- ResNet50 샘플 모델. 없으면 설정 셀이 `dx_app/setup.sh`로 다운로드합니다.
- 6절에서 사용하는 YOLO26-S 샘플 모델(21 MB). 없으면 해당 절에서 같은 방식으로 다운로드합니다.
- `sudo`는 필요하지 않습니다. 약 10분이 걸립니다.

다음 셀은 SDK를 찾고, 위 요구 사항을 확인한 뒤 상태 표를 출력합니다. `MISSING`으로 표시된 항목에는 그것을 제공하는 단계가 함께 표시됩니다.

<!-- cell: 51bdb967 src: a53c9cd719 -->
## 1. DX-RT의 위치

DX-COM은 `.dxnn` 모델을 만듭니다. DX-RT는 그 모델을 로드하고, 추론 버퍼와 작업을 관리하고, 장치 드라이버와 통신하고, 출력 텐서를 반환합니다.

<img src="assets/dx-rt-runtime-workflow.svg" style="max-width: 1100px; width: 100%;" alt="DX-RT 추론 워크플로">

| DX-RT가 담당 | 애플리케이션이 담당 |
|---|---|
| DXNN 로딩과 검증 | 입력 수집 |
| 장치 선택과 NPU 스케줄링 | 모델별 전처리 |
| 런타임 입출력 버퍼 | 모델별 후처리 |
| 동기 및 비동기 작업 | 제품 동작과 시각화 |
| 런타임 프로파일링과 장치 조회 | 엔드투엔드 정확도와 서비스 수준 지표 |

> 런타임 호출이 성공했다는 것은 모델이 실행되었다는 뜻입니다. 전처리, 디코딩, 애플리케이션 정확도가 올바르다는 뜻은 아닙니다.

<!-- cell: e3eb37dc src: f7cdc491e8 -->
## 2. 튜토리얼 워크스페이스 초기화

설정 셀은 `config.json`에서 공유 SDK 위치를 읽습니다. 이 튜토리얼 안에 로컬 링크와 디렉터리만 생성합니다.

```text
T06-DX-Runtime/
├── assets/
├── dx_rt_01_beginner.ipynb
├── dx_rt_02_intermediate.ipynb
├── dx_rt_03_advanced.ipynb
└── workspace/
    ├── models/       # links to SDK models
    ├── reports/      # dxparse and benchmark reports
    ├── profiler/     # profiler JSON and visualizations
    ├── cpp/          # generated C++ examples and build files
    ├── multi_input/  # link to the two-input model from Tutorial 05-3 (Advanced)
    └── .venv-dxrt/   # DX-RT Python binding environment (created in Intermediate section 2)
```

이 튜토리얼은 DX-APP 리소스 설정으로 다운로드되는 `resnet50_224x224.dxnn`을 사용합니다. 모델이 없으면 설정 셀이 모델을 받기 위한 정확한 터미널 명령을 출력합니다.

<!-- cell: 4ac70adf src: d712a2cdc0 -->
### 2.1 노트북 밖에서 사용하는 명령

위에 출력된 모델 다운로드 명령은 일반적인 셸 명령입니다.

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_app
bash setup.sh --models resnet50 --no-force
```

`--no-force`는 이미 다운로드된 리소스를 유지합니다. 그다음 튜토리얼은 자체 `workspace/models` 디렉터리 아래에 심볼릭 링크를 만들며, SDK 모델을 복사하거나 수정하지 않습니다.

<!-- cell: 02e64674 src: 8ca79cf862 -->
## 3. 런타임 설치 확인

다음 명령은 터미널에 입력하는 명령과 정확히 같습니다. 출력 파일은 생성하지 않습니다.

<!-- cell: e1dcae80 src: 07bed7df89 -->
### 3.1 장치 확인

`dxcli --status`는 장치를 명시적으로 선택하지 않으면 사용 가능한 모든 가속기를 조회합니다.

```bash
dxcli --status
dxcli -s          # short form of the same command
```

정상이라면 장치가 하나 이상 식별되어야 합니다. 장치가 보이지 않으면 여기서 멈추고, 모델을 테스트하기 전에 드라이버, 펌웨어, 물리적 연결, 설치 상태를 확인하세요.

<!-- cell: ca70321e src: 51e030d707 -->
## 4. DXNN 모델 계약 검사

런타임 애플리케이션은 컴파일된 모델과 다음 항목 모두에서 일치해야 합니다.

| 계약 항목 | 중요한 이유 |
|---|---|
| 입력 텐서 이름과 개수 | 다중 입력 모델은 올바른 매핑이 필요함 |
| 형상(shape) | 애플리케이션이 필요한 개수만큼 요소를 할당해야 함 |
| 데이터 타입 | dtype이 맞지 않으면 오류나 잘못된 데이터가 생길 수 있음 |
| 레이아웃 | NHWC와 NCHW는 같은 값을 다른 순서로 저장함 |
| 전처리 경계 | 일부 전처리는 이미 DXNN에 컴파일되어 있을 수 있음 |
| 출력 텐서 | 후처리는 올바른 이름, 형상, 순서를 사용해야 함 |
| CPU 태스크 | ORT가 활성화된 런타임 경로가 필요함 |

`dxparse`는 추론을 실행하지 않고 이 계약을 읽습니다. `-v`는 태스크 의존성과 메모리 정보를 추가합니다. 표준 셸 리디렉션으로 리포트를 저장해 다음 셀에서 사용합니다. `dxparse`는 파일에 쓸 때도 ANSI 이스케이프 코드로 출력에 색을 입히므로, `sed` 필터로 이를 제거해 저장된 리포트를 일반 텍스트로 유지합니다.

동일한 터미널 명령:

```bash
dxparse -m <T06-DX-Runtime>/workspace/models/resnet50_224x224.dxnn -v \
  | sed 's/\x1b\[[0-9;]*m//g' \
  > <T06-DX-Runtime>/workspace/reports/resnet50_dxparse.txt
```

<!-- cell: a6c68f9c src: 9918c774c5 -->
### 4.1 저장된 계약 검토

다음 셀은 튜토리얼 워크스페이스 안에 생성된 리포트를 읽습니다. 모델 버전, 컴파일러 버전, 입출력 텐서, 태스크 유형, 메모리 크기를 확인하세요.

<!-- cell: a5f4ec9d src: c5ebfa0c53 -->
## 5. `dxrun` 모드 이해

| 모드 | 옵션 | 주요 동작 | 용도 |
|---|---|---|---|
| 단일 | `--single` | 한 코어에서 단일 입력을 순차 추론 | 기본 실행과 지연 시간 확인 |
| 벤치마크 | `--benchmark` | 사용 가능한 런타임 파이프라인을 계속 바쁘게 유지 | 최대 처리량 비교 |
| 목표 FPS | `--fps N` | 요청한 속도로 작업을 제출 | 제품 부하에서의 용량과 안정성 확인 |

`--time`은 테스트 시간을 제어하며 `--loops`보다 우선합니다. `--warmup-runs`는 초기 워밍업 실행을 측정에서 제외합니다. 결과를 비교할 때는 같은 모델, 옵션, 시간, 시스템 상태를 사용하세요.

<!-- cell: a11603a4 src: 88e192a8dc -->
### 5.1 요청 하나 실행

동일한 터미널 명령:

```bash
cd <T06-DX-Runtime>/workspace
dxrun -m models/resnet50_224x224.dxnn --single --loops 1 --verbose
```

<!-- cell: a0ae40f8 src: 126816c0da -->
### 5.2 최대 처리량 측정

이 벤치마크는 더미 입력을 사용합니다. 런타임 성능을 측정하기에는 적합하지만 분류 정확도를 측정하기에는 적합하지 않습니다.

동일한 터미널 명령:

```bash
cd <T06-DX-Runtime>/workspace
dxrun -m models/resnet50_224x224.dxnn \
      --benchmark \
      --time 5 \
      --warmup-runs 5 \
      --buffer-count 6
```

<!-- cell: d6f353b5 src: 21dc67134a -->
### 5.3 목표 워크로드 테스트

목표 FPS 실행은 시스템이 요청한 도착 속도를 유지할 수 있는지를 묻습니다. 최대 처리량 벤치마크와는 다릅니다.

동일한 터미널 명령:

```bash
cd <T06-DX-Runtime>/workspace
dxrun -m models/resnet50_224x224.dxnn --fps 30 --time 5 --warmup-runs 5
```

<!-- cell: c3631420 src: 393fccae2d -->
## 6. CPU 태스크와 `--use-ort`

DXNN 그래프에는 NPU 태스크와 CPU 태스크가 포함될 수 있습니다. `--use-ort`는 NPU에서 실행되지 않는 CPU 측 서브그래프를 위해 ONNX Runtime을 활성화합니다.

- `--use-ort`를 무턱대고 추가하지 마세요.
- 먼저 `dxparse -v`로 태스크 그래프를 확인하세요.
- 비교하는 모든 결과에 같은 ORT 설정을 사용하세요.
- 런타임이 ORT 지원 없이 빌드되었다면 CPU 태스크 실행은 불가능합니다.

지금까지 사용한 ResNet50 모델은 NPU 전용 그래프이므로 이 옵션을 켜도 아무것도 바뀌지 않습니다. Model Zoo의 YOLO26-S는 다릅니다. 검출 헤드가 CPU 태스크(`npu_0 -> cpu_0`)로 끝나며, 이 태스크가 원시 헤드 텐서 6개를 최종 `[1, 300, 6]` 검출 결과로 디코딩합니다. 다음 세 셀은 모델이 없으면 다운로드하고, 태스크 그래프를 보여 준 뒤, `dxrun`을 NPU 전용과 `--use-ort` 두 가지로 실행합니다.

동일한 터미널 명령:

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_app
bash setup.sh --models yolo26s --no-force        # only when the model is missing

cd <T06-DX-Runtime>/workspace
dxparse -m models/yolo26-s_640x640.dxnn -v
dxrun -m models/yolo26-s_640x640.dxnn --benchmark --time 5 --warmup-runs 5
dxrun -m models/yolo26-s_640x640.dxnn --benchmark --time 5 --warmup-runs 5 --use-ort
```

<!-- cell: a8230861 src: 9c5259ac39 -->
**확인할 내용**

| | NPU 전용 | `--use-ort` |
|---|---|---|
| `dxrun`이 나열하는 태스크 | `Task[0] npu_0` | `Task[0] npu_0`과 `Task[1] cpu_0` |
| 출력 | 원시 헤드 텐서 6개(`.../Conv_output_0`) | `output0 [1, 300, 6]`: 디코딩된 검출 결과 |
| FPS | 순수 NPU 처리량 | CPU 디코딩을 포함한 엔드투엔드 처리량 |

빠른 데스크톱 CPU에서는 두 FPS 값이 몇 퍼센트 이내로 비슷한 경우가 많습니다. DX-RT가 NPU에서 다음 입력을 처리하는 동안 CPU 태스크를 파이프라인으로 실행하기 때문입니다. 호스트 CPU가 약하면 CPU 태스크가 병목이 되어 FPS 차이가 커집니다. 항상 달라지는 것은 출력입니다. `--use-ort`가 없으면 애플리케이션이 원시 텐서 6개를 직접 디코딩해야 합니다. CPU 태스크가 있는 검출 모델이라면, 애플리케이션에서 측정한 처리량과 비교해야 할 값은 ORT를 활성화한 수치입니다.

<!-- cell: 228a7f19 src: 10632970eb -->
## 7. CLI 이름과 하위 호환성

현재 튜토리얼은 새 명령 이름을 사용합니다. 이전 스크립트는 호환 별칭을 통해 계속 동작합니다.

| 현재 명령 | 이전 별칭 | 용도 |
|---|---|---|
| `dxparse` | `parse_model` | DXNN 모델 검사 |
| `dxrun` | `run_model` | 모델 실행 또는 벤치마크 |
| `dxcli` | `dxrt-cli` | 런타임 장치 인터페이스 조회 |

로그와 문서가 일관되도록 새 코드에서는 현재 이름을 사용하세요.

<!-- cell: 69b2a38e src: 351f9dcc85 -->
## 8. 실시간 모니터링

`dxtop`은 대화형이며 터미널을 계속 다시 그립니다. 노트북 셀이 아니라 별도의 JupyterLab 터미널에서 실행하세요.

```bash
dxtop
```

**File → New → Terminal**을 선택한 뒤 명령을 실행합니다. 종료하려면 `q`를 누릅니다. 다른 터미널에서 `dxrun`을 실행하는 동안 사용률, NPU 메모리, 온도, 전압, 클록을 모니터링하세요.

<!-- cell: 5e501953 src: bbfc3360e6 -->
## 9. 문제 해결

| 증상 | 표시되는 내용 | 원인 | 해결 |
|---|---|---|---|
| 명령을 찾을 수 없음 | `dxrun: command not found` | DX-RT가 설치되지 않았거나 `/usr/local/bin`이 `PATH`에 없음 | 튜토리얼 01 3절을 완료하고 새 터미널을 시작 |
| 장치 없음 | `dxcli --status`에 `No device found` | 드라이버, 펌웨어 또는 물리적 연결 | `lsmod | grep dxrt`를 확인하고, 재부팅하거나 호스트 전원을 완전히 껐다가 다시 켬 |
| 모델 거부 | `dxrun`의 버전 또는 형식 오류 | 호환되지 않는 DX-COM으로 DXNN을 컴파일함 | `dxparse`가 출력하는 버전을 설치된 런타임과 비교하고 필요하면 다시 컴파일 |
| CPU 태스크 오류 | CPU 태스크가 있는 모델에서 ORT 오류 | 런타임이 ONNX Runtime 없이 빌드됨 | ORT가 활성화된 빌드를 사용하고 `--use-ort`를 전달 |
| 예상치 못한 성능 | 실행마다 FPS가 달라짐 | 워밍업 없음, 열 스로틀링 또는 다른 부하 | `--warmup-runs`, 고정된 시간, 동일한 옵션, 유휴 상태의 호스트를 사용 |

<!-- cell: 3578de13 src: 817b273196 -->
## 10. 요약

### 10.1 완료한 런타임 워크플로

**도구와 장치 확인**  
→ **DXNN 계약 검사**  
→ **추론 1회 실행**  
→ **최대 처리량 측정**  
→ **목표 워크로드 테스트**  
→ **NPU 전용 실행과 ORT 활성화 실행 비교**

<img src="assets/dx-rt-runtime-workflow.svg" style="max-width: 1000px; width: 100%;" alt="DX-RT 추론 워크플로">

### 10.2 도구 대시보드

| 질문 | 도구 또는 옵션 | 근거 |
|---|---|---|
| NPU가 보이는가? | `dxcli --status` | 장치 상태 |
| 모델은 무엇을 기대하는가? | `dxparse -v` | 텐서와 태스크 계약 |
| 요청 하나를 실행할 수 있는가? | `dxrun --single` | 기본 실행과 지연 시간 |
| 최대 처리량은 얼마인가? | `dxrun --benchmark` | 통제된 더미 입력 FPS |
| 30 FPS를 유지할 수 있는가? | `dxrun --fps 30` | 목표 부하에서의 동작 |
| 모델에 CPU가 필요한가? | `dxparse -v`, `dxrun --use-ort` | 태스크 그래프와 디코딩된 출력 |
| 시간이 지나면 어떻게 되는가? | `dxtop` | 실시간 장치 상태 |

### 10.3 완료 체크리스트

- [ ] DX-RT와 애플리케이션 사이의 경계를 파악함
- [ ] 설치된 CLI 도구를 확인함
- [ ] NPU 상태를 조회함
- [ ] 상세 DXNN 리포트를 저장하고 검토함
- [ ] 단일, 벤치마크, 목표 FPS 모드를 실행함
- [ ] CPU 태스크가 있는 모델을 NPU 전용과 `--use-ort`로 실행해 비교함
- [ ] 런타임 성능과 모델 정확도를 구분함
- [ ] 실제 애플리케이션 데이터로 전처리, 후처리, 정확도를 검증함

> **기억하세요:** 성능을 비교할 때는 모델, 명령, 런타임 버전, 장치 상태, 워크로드를 정확히 동일하게 유지하세요.

### 10.4 다음 단계

**중급 튜토리얼**로 이어서 같은 런타임 개념을 Python 및 C++ API로 구현하고, 동기, 비동기, 배치, 버퍼 관리 방식을 비교해 보세요.
