<!-- i18n source: notebooks/T05-DX-Compiler/dx_com_02_intermediate.ipynb -->
<!-- i18n lang: ko -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: intro src: bb61e22e24 -->
# DEEPX Tutorial 05-2 - DX-COM 중급

이 노트북은 컴파일을 한 번 성공시키는 단계에서 벗어나, 제어 가능하고 설명 가능한 컴파일 워크플로로 나아갑니다.

캘리브레이션 결정을 개선하고, PPU Type 0과 Type 1로 실제 Model Zoo 모델을 빌드하며, YOLO26(기본값은 n이며 s, m, l, x는 변수 하나만 바꾸면 됩니다)에 TopK 우선 최적화를 적용했을 때의 효과를 측정합니다.

<!-- cell: 6dffc67d-0a31-4b77-b975-65a83973f314 src: caf8fba665 -->
> 이 튜토리얼의 3부 중 2부입니다. 코스 맵과 전체 파트 목록은 초급 노트북에 있습니다.

<!-- cell: objectives src: 46a3025a59 -->
## 학습 목표

이 튜토리얼을 마치면 다음을 할 수 있습니다.

- 대표성 있는 캘리브레이션 데이터를 선택하고 모델 전처리를 재현한다,
- 검출 연산 중 어떤 것이 NPU의 PPU에서 실행되고 어떤 것이 호스트 CPU에 남는지 설명한다,
- 검출 헤드 아키텍처를 보고 PPU Type 0 또는 Type 1을 선택한다,
- PPU 레이어 매핑을 ONNX 그래프와 대조해 검증한다,
- Model Zoo의 YOLOv7과 YOLOX-S 모델을 하드웨어 PPU로 컴파일한다,
- PPU 적용 모델과 비적용 모델의 Model Zoo DXNN 런타임 동작을 비교한다,
- 모든 크기(n, s, m, l, x)의 YOLO26 모델에 TopK 우선 최적화를 적용한다,
- `dxrun`으로 기준 모델과 최적화 모델의 DXNN 성능을 비교한다.

이 노트북은 SDK 소스 트리를 수정하지 않습니다. 생성되는 모든 파일은 git이 무시하는 `<dx-tutorials>/notebooks/T05-DX-Compiler/workspace` 아래에 저장됩니다.

<!-- cell: 64617a25-7ca0-407c-97d2-80bcd28d505c src: f0911d99ba -->
## 사전 요구 사항

- 튜토리얼 05-1 완료: `venv-dx-compiler-local`의 `dxcom`과 샘플 캘리브레이션 데이터셋.
- 4절과 6절의 `dxrun` 비교를 위해 DEEPX NPU가 있는 환경에 DX-RT 설치.
- 다운로드: Model Zoo의 YOLOv7 및 YOLOX ONNX, DXNN 파일(약 290 MB).
- 소요 시간: YOLOv7 640x640 컴파일은 이 시리즈에서 가장 오래 걸립니다. 노트북 PC에서는 수십 분을 예상하세요.

다음 셀은 SDK 위치를 찾고, 이 요구 사항을 확인하고, 상태 표를 출력합니다. `MISSING`으로 표시된 항목에는 그것을 제공하는 단계가 함께 표시됩니다.

<!-- cell: workspace-heading src: 98620d01b8 -->
## 1. 튜토리얼 워크스페이스 초기화

<!-- cell: calibration src: d60c6fc7ed -->
## 2. 캘리브레이션은 모델 정의의 일부입니다

양자화는 캘리브레이션 입력이 만들어 내는 부동소수점 텐서를 관찰합니다. 구하기 쉽지만 과제와 무관한 이미지 집합으로도 DXNN 파일은 생성되지만 정확도는 떨어질 수 있습니다.

| 결정 사항 | 권장 방식 |
|---|---|
| 샘플 | 실제 조명, 크기, 배경, 클래스 분포를 포괄 |
| 전처리 | 학습 및 평가와 정확히 일치 |
| 개수 | 모델 레시피 값으로 시작하고, 안정성을 측정한 뒤에만 늘림 |
| 방법 | 공개된 설정으로 시작한 뒤 지표를 기준으로 방법을 비교 |
| 검증 | 별도 보관(held-out) 데이터셋에서 ONNX와 DXNN 출력을 비교 |

SDK의 `calibration_dataset` 링크는 이 튜토리얼을 실행 가능하게 해 주지만, 과제별 데이터셋을 대신할 수는 없습니다.

<!-- cell: preprocessing-order src: 1a5af2c2fa -->
### 2.1 전처리 순서

일반적인 검출 파이프라인은 다음과 같습니다.

```text
image → letterbox/pad → BGR-to-RGB → divide by 255 → HWC-to-CHW → add batch axis
```

이 순서를 그대로 복사하지 마세요. 모델 내보내기(exporter) 코드와 학습 코드를 확인하세요. 패딩 위치, 패딩 값, 색상 순서, 정규화가 다르면 캘리브레이션 텐서의 분포가 달라집니다.

<!-- cell: ppu-overview src: 906986dee8 -->
## 3. PPU 개요

**후처리 유닛(Post-Processing Unit, PPU)**은 DEEPX NPU 내부의 하드웨어입니다. 지원되는 YOLO 검출 헤드에 대해 호스트 CPU가 처리해야 하는 원시 후보의 수를 줄여 줍니다.

PPU는 선택된 두 가지 연산을 수행합니다.

1. **신뢰도 필터링**: 컴파일 시점에 정한 임계값보다 낮은 후보를 제거합니다.
2. **클래스 예측**: 남은 각 후보에 대해 가장 강한 클래스를 선택합니다.

PPU는 비최대 억제(NMS)를 수행하지 **않습니다**. NMS와 애플리케이션 수준의 해석은 여전히 호스트 CPU에서 실행됩니다. PPU는 이미지 전처리나 NPU 추론을 대체하지도 않습니다.

<!-- cell: ppu-flow src: 2b907dff84 -->
### 3.1 검출 파이프라인에서 PPU의 위치

```text
Without hardware PPU
┌──────────┐   ┌───────────────┐   ┌──────────────────────────────┐   ┌─────────┐
│ Input    │ → │ NPU inference │ → │ CPU filtering/class selection│ → │ CPU NMS │
└──────────┘   └───────────────┘   └──────────────────────────────┘   └─────────┘

With PPU Type 0 or Type 1
┌──────────┐   ┌───────────────┐   ┌────────────────────────────┐   ┌─────────┐
│ Input    │ → │ NPU inference │ → │ PPU filter/class prediction│ → │ CPU NMS │
└──────────┘   └───────────────┘   └────────────────────────────┘   └─────────┘
                                           hardware                    host
```

| 단계 | PPU 미사용 | PPU Type 0/1 사용 |
|---|---|---|
| 신경망 추론 | NPU | NPU |
| 신뢰도 필터링 | 호스트 CPU | PPU 하드웨어 |
| 최적 클래스 선택 | 호스트 CPU | PPU 하드웨어 |
| 바운딩 박스 디코딩 | 모델/타입에 따라 다름 | 모델/타입에 따라 다름 |
| NMS | 호스트 CPU | 호스트 CPU |

주된 이점은 호스트 CPU 작업량 감소와 중간 검출 데이터 감소입니다. 정확한 종단 간(end-to-end) 이득은 모델, 임계값, 장면, 호스트 CPU, 애플리케이션 후처리에 따라 달라집니다.

<!-- cell: ppu-types src: 7508872dcf -->
### 3.2 모델 아키텍처에 따라 모드 선택

| 모드 | 아키텍처 | 대표 모델 | 레이어 매핑 | 실행 위치 |
|---|---|---|---|---|
| PPU Type 0 | 앵커 기반(anchor-based) | YOLOv3, YOLOv4, YOLOv5, YOLOv7 | 검출 Conv 노드 → 앵커 수 | PPU 하드웨어 |
| PPU Type 1 | 앵커 프리(anchor-free) | YOLOX, YOLOv8–YOLOv12 | `bbox`, (있는 경우) `obj_conf`, `cls_conf` 노드 | PPU 하드웨어 |
| `pre_optimize()` | TopK 우선 ONNX 재작성 | YOLOv8 계열, YOLOv10, YOLO26 | 스케일별 bbox/class 출력 텐서 | NPU + 호스트 CPU 그래프 |

모델 이름만으로 타입을 선택하지 마세요. 내보낸 ONNX의 헤드 구조를 확인하고, 바로 그 파일의 노드 이름을 사용하세요.

<!-- cell: ppu-config-anatomy src: 7ae6e1120b -->
### 3.3 PPU 설정 읽기

```text
PPU configuration
├── type           selects the supported head architecture
├── conf_thres     fixed confidence threshold compiled into the DXNN
├── num_classes    class count of this exported model
├── activation     Type 0 activation, usually Sigmoid
└── layer          exact ONNX head-node mapping
```

| 필드 | Type 0 | Type 1 | 중요한 이유 |
|---|:---:|:---:|---|
| `type` | 필수 | 필수 | 하드웨어 데이터 경로를 선택 |
| `conf_thres` | 필수 | 필수 | PPU를 통과하는 후보 수를 제어 |
| `num_classes` | 필수 | 필수 | 헤드 채널 레이아웃과 일치해야 함 |
| `activation` | 필수 | 사용 안 함 | 앵커 기반 점수 활성화 함수를 적용 |
| `layer` | 딕셔너리 | 리스트 | PPU 입력을 정확한 ONNX 노드에 연결 |
| `num_anchors` | 레이어별 | 사용 안 함 | 각 스케일의 앵커 수와 일치해야 함 |
| `obj_conf` | 사용 안 함 | 모델에 따라 다름 | YOLOX는 별도의 객체성(objectness) 브랜치를 가짐 |

**중요:** `conf_thres`는 컴파일 시 고정됩니다. 나중에 바꾸려면 DXNN을 새로 만들어야 합니다. 값이 높을수록 호스트 작업은 줄지만 유효한 검출까지 제거될 수 있습니다. 배포 전에 정확도를 측정하세요.

<!-- cell: type0-heading src: 269378c531 -->
## 4. PPU Type 0 실습: Model Zoo의 YOLOv7

YOLOv7은 앵커 기반 검출 헤드를 사용하므로 이 실습에서는 PPU Type 0을 사용합니다. 실제 Model Zoo ONNX와 그에 맞는 PPU JSON을 다운로드하고, 매핑된 Conv 노드를 확인하고, 데이터셋 경로를 조정한 뒤 DXNN을 컴파일합니다.

<!-- cell: download-helper-heading src: 110755aed3 -->
### 4.1 Model Zoo 파일을 안전하게 다운로드

다음 셀은 `tutorial_paths.py`에서 `download_file`을 가져옵니다(튜토리얼 05-2와 05-3이 공유하는 헬퍼이며, 파일을 열어 내용을 읽어 보세요). 비어 있지 않은 대상 파일이 이미 있으면 헬퍼는 네트워크 요청 없이 다운로드를 건너뜁니다. 새로 다운로드해야 할 때는 먼저 해당 로컬 파일을 삭제하세요. 새 다운로드의 경우 헬퍼는 원격 파일 크기를 확인하고, 임시 `.part` 파일에 기록한 뒤, 전송이 완전히 끝난 후에만 대상 파일을 교체합니다. DNS 또는 네트워크 오류가 발생하면 예외를 발생시키며 빈 최종 파일을 남기지 않습니다.

이 실습은 컴파일용 ONNX와 PPU JSON을 다운로드합니다. 또한 벤치마크 기준으로 사용할 공개 non-PPU DXNN도 다운로드합니다.

<!-- cell: inspect-type0-heading src: cb55365789 -->
### 4.2 ONNX와 Type 0 매핑 확인

Type 0에서 `layer`는 딕셔너리입니다. 각 키는 검출 헤드 Conv 노드의 이름이어야 하고, `num_anchors`는 해당 스케일과 일치해야 합니다. 출력 채널 수는 다음과 같이 계산됩니다.

```text
channels = num_anchors × (5 + num_classes)
         = 3 × (5 + 80)
         = 255
```

<!-- cell: yolov7-diagram src: d3c1a04b7a -->
강조 표시된 Conv 노드는 Model Zoo PPU 설정에서 사용하는 세 개의 스케일별 검출 헤드입니다.

![YOLOv7 Type 0 PPU 헤드 매핑](assets/yolov7-class-n80-ppu.png)

<!-- cell: adapt-type0-heading src: e20da5e57c -->
### 4.3 환경에 따라 달라지는 경로만 조정

Model Zoo JSON에는 이 모델에 사용된 전처리 레시피가 들어 있습니다. 실습에서는 이를 그대로 두고, 사용할 수 없는 캘리브레이션 데이터셋 경로만 교체하세요. 제품용 모델이라면 배포 도메인을 대표하는 데이터셋을 사용하세요.

<!-- cell: compile-helper-heading src: 320ebf992e -->
### 4.4 Type 0 DXNN 컴파일 및 확인

다음 코드 셀은 DX-COM 환경을 활성화하고 `dxcom`을 직접 실행합니다. 별도의 터미널에서 다음 명령을 입력하는 것과 동일합니다.

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/yolov7_640x640.onnx \
      -c configs/yolov7_640x640_ppu.local.json \
      -o outputs/yolov7_type0_ppu \
      --gen_log \
      --export_html
```

실제 SDK와 튜토리얼 경로는 다를 수 있습니다. 코드 셀은 `config.json`에서 불러온 경로를 사용합니다. 출력 디렉터리에 이미 DXNN 파일이 있으면 건너뛴다는 메시지를 출력하고 `dxcom`을 다시 실행하지 않습니다.

> **코드 셀 끝의 `2>&1 | sed ... | grep ...`에 대하여:** `dxcom`은 Jupyter가 렌더링할 수 없는 커서 이동 이스케이프 코드로 진행률 표시줄을 그리기 때문에, 필터가 없으면 수백 줄의 빈 줄로 표시됩니다. 이 필터는 진행률 표시줄 줄만 제거하며, 모든 `[INFO]`, `[WARNING]`, `[ERROR]` 메시지는 그대로 보입니다. 터미널에서는 필터를 생략해도 됩니다. `--gen_log`가 전체 출력을 `compiler.log`에 보관합니다. 이 노트북의 모든 컴파일 셀에서 같은 필터를 사용합니다.

<!-- cell: benchmark-yolov7-heading src: 50a17e9138 -->
### 4.5 `dxrun`으로 PPU와 non-PPU 비교

non-PPU DXNN은 Model Zoo에서 직접 다운로드한 것이고, PPU DXNN은 4.4절에서 컴파일한 결과물입니다. 다음 셀은 합성 입력으로 두 모델을 각각 5초 동안 실행합니다.

```bash
dxrun -m models/yolov7_640x640_non_ppu.dxnn --use-ort -t 5
dxrun -m outputs/yolov7_type0_ppu/<compiled-model>.dxnn --use-ort -t 5
```

non-PPU 모델은 NPU 태스크와 CPU 태스크를 포함하며 원시 검출 값을 반환합니다. PPU 모델은 하드웨어가 생성한 `BBOX` 데이터를 반환하고 호스트가 처리하는 출력을 줄입니다. PPU가 합성 입력 FPS의 향상을 보장하지는 **않습니다**. 모델 스케줄링, 출력 전송, 장치 상태에 따라 이 단독 테스트에서는 non-PPU 결과가 더 빠를 수도 있습니다. 제품 결정을 위해서는 실제 입력으로 전체 파이프라인 지연 시간과 호스트 CPU 사용량도 함께 측정하세요.

<!-- cell: type1-heading src: 103ddfd632 -->
## 5. PPU Type 1 실습: Model Zoo의 YOLOX-S

YOLOX는 분리형(decoupled) 앵커 프리 검출 헤드를 사용합니다. 각 스케일에 바운딩 박스, 객체성, 클래스 신뢰도 브랜치가 따로 있으므로 이 실습에서는 PPU Type 1을 사용합니다.

<!-- cell: download-type1-heading src: dcfccc5610 -->
### 5.1 ONNX와 Type 1 JSON 다운로드

같은 안전 다운로더를 재사용합니다. ONNX와 PPU JSON은 컴파일에 사용하고, 공개 non-PPU DXNN은 벤치마크 기준으로 사용합니다.

<!-- cell: inspect-type1-heading src: 80d640adfd -->
### 5.2 ONNX와 Type 1 매핑 확인

YOLOX에서는 각 스케일마다 이름이 지정된 세 개의 노드를 매핑합니다.

```text
feature map ─┬─ bbox branch ─────→ bbox
             ├─ object branch ───→ obj_conf
             └─ class branch ────→ cls_conf
```

세 항목은 각각 80×80, 40×40, 20×20 검출 스케일에 대응합니다.

<!-- cell: yolox-diagram src: 8a4b542192 -->
색상은 각 스케일에서 짝을 이뤄야 하는 `bbox`, `obj_conf`, `cls_conf` 브랜치를 나타냅니다.

![YOLOX Type 1 PPU 헤드 매핑](assets/yolox-class-n80-ppu.png)

<!-- cell: adapt-type1-heading src: 7559b7cea7 -->
### 5.3 캘리브레이션 데이터셋 경로 조정

다운로드한 Type 1 매핑과 YOLOX 전처리는 그대로 유지합니다. 이 실습에서는 Model Zoo 빌드 머신의 데이터셋 경로만 교체합니다.

<!-- cell: compile-type1-heading src: 818827cba8 -->
### 5.4 Type 1 DXNN 컴파일 및 확인

다음 코드 셀은 별도의 터미널에서 다음 명령을 입력하는 것과 동일합니다.

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/yolox-s_640x640.onnx \
      -c configs/yolox-s_640x640_ppu.local.json \
      -o outputs/yolox_s_type1_ppu \
      --gen_log \
      --export_html
```

코드 셀은 실제 경로를 자동으로 결정합니다. 출력 디렉터리에 이미 DXNN 파일이 있으면 `dxcom`을 건너뜁니다. `dxparse -v` 출력을 Type 0 결과와 비교하되, 특히 출력 텐서 레이아웃과 PPU 메타데이터를 살펴보세요.

<!-- cell: benchmark-yolox-heading src: 2ae99d790d -->
### 5.5 `dxrun`으로 PPU와 non-PPU 비교

YOLOX-S에 대해 같은 통제된 테스트를 반복합니다. non-PPU 기준 모델은 5.1절에서 다운로드한 공개 Model Zoo DXNN이며, 이 튜토리얼에서 컴파일한 것이 아닙니다.

```bash
dxrun -m models/yolox-s_640x640_non_ppu.dxnn --use-ort -t 5
dxrun -m outputs/yolox_s_type1_ppu/<compiled-model>.dxnn --use-ort -t 5
```

이 합성 벤치마크는 `dxparse`가 보여 주는 태스크 구조와 함께 해석하세요. PPU 모델은 이 단일 FPS 수치가 올라가지 않더라도 호스트 측 처리와 출력 트래픽을 줄일 수 있습니다.

<!-- cell: topk-heading src: a8f8add50a -->
## 6. YOLO26 TopK 기반 최적화 (n / s / m / l / x)

YOLO26은 일대일(one-to-one) 분리형 검출 헤드를 사용합니다. `yolo26_postprocess` 변환은 TopK 선택을 비용이 큰 CPU 측 디코딩 작업보다 앞으로 옮깁니다. 이는 PPU Type 0/1과는 다릅니다. ONNX 그래프를 재작성하여 NPU 추론 이후에 처리되는 후보 작업량을 줄입니다.

```text
Baseline: 8,400 candidates → decode and score all candidates → TopK 300
Optimized: 8,400 candidates → TopK 300 → decode and score only 300 candidates
```

두 모델 모두 같은 검출 결과 계약 `[1, 300, 6]`을 유지하지만, CPU 측 작업의 순서가 바뀝니다.

이 절은 Model Zoo의 모든 YOLO26 크기에서 동작합니다. 6.1의 변수 하나, `YOLO26_VARIANT`가 모델을 선택하며, 아래의 모든 파일 이름, 헤드 텐서 검사, 컴파일, 벤치마크가 이 값을 따릅니다. 다섯 가지 내보내기 모델은 모두 같은 검출 헤드 레이아웃을 공유하므로 TopK 변환도 동일합니다. 먼저 `n`으로 시작한 다음 변수를 바꿔 더 큰 모델로 이 절을 다시 실행해 보세요.

| `YOLO26_VARIANT` | ONNX 다운로드 | 모델당 컴파일 시간 (x86_64 데스크톱, 이 절은 모델 두 개를 컴파일) |
|---|---:|---|
| `n` (기본값) | 10 MB | 약 2분 (측정값) |
| `s` | 37 MB | 약 2분 (측정값) |
| `m` | 78 MB | 측정하지 않음, 수 분 예상 |
| `l` | 95 MB | 측정하지 않음, 10분 이상 예상 |
| `x` | 213 MB | 측정하지 않음, 10분 이상 예상 |

각 변형은 `models/`, `configs/`, `outputs/` 아래에 자체 파일을 사용하므로, 이미 컴파일한 크기로 다시 전환하면 컴파일을 건너뜁니다.

<!-- cell: download-yolo26-heading src: 306e030b72 -->
### 6.1 기준 모델 다운로드 및 검증

`YOLO26_VARIANT`를 설정하고 셀을 실행하세요. 기준 ONNX와 그에 맞는 Q-Lite JSON은 Model Zoo에서 가져옵니다(`yolo26-<variant>_640x640.onnx`와 `.json`). 안전 다운로더는 셸 다운로드가 실패했는데도 다음 노트북 셀이 계속 진행될 때 생길 수 있는 빈 파일 오류를 방지합니다.

<!-- cell: verify-heads-heading src: adfa304cc8 -->
### 6.2 여섯 개의 헤드 텐서 확인

이 변환에는 세 검출 스케일마다 바운딩 박스 텐서 하나와 클래스 신뢰도 텐서 하나가 필요합니다. 이것들은 다른 모델에서 복사한 표시용 레이블이 아니라 **출력 텐서 이름**입니다. Model Zoo에 내보낸 모든 YOLO26 크기는 같은 헤드 레이아웃(박스는 `/model.23/one2one_cv2.<scale>/...`, 클래스 점수는 `/model.23/one2one_cv3.<scale>/...`)을 공유하므로, 아래 이름은 하나의 패턴으로 만든 뒤 실제로 다운로드한 ONNX와 대조해 확인합니다. 확인에 실패하면 Netron에서 모델을 열어 패턴을 수정하세요.

<!-- cell: apply-topk-heading src: 1ac20c8599 -->
### 6.3 `yolo26_postprocess` 적용

`dx_com`이 설치된 곳이 DX-COM Python 환경이므로 변환은 이 환경에서 실행합니다. 별도의 ONNX 파일을 생성하므로 기준 모델은 그대로 유지됩니다.

스크립트는 기준 ONNX와 대상 경로를 인수로 받으므로 같은 스크립트를 모든 변형에 사용할 수 있습니다. 다음 두 셀은 스크립트를 작성하고 실행합니다. 기본 `n` 변형의 경우 실행은 다음과 동일합니다.

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
python optimize_yolo26.py models/yolo26-n_640x640.onnx models/yolo26-n_640x640_topk300.onnx
```

<!-- cell: adapt-yolo26-heading src: f1bf82c079 -->
### 6.4 Model Zoo 설정 조정

기준 모델과 최적화 모델에는 같은 캘리브레이션 및 전처리 설정을 사용해야 합니다. 사용할 수 없는 데이터셋 경로만 변경합니다. 이렇게 하면 성능 비교가 통제된 상태로 유지됩니다.

<!-- cell: compile-yolo26-heading src: e70f6d348a -->
### 6.5 두 모델 컴파일 및 확인

각 ONNX를 별도의 출력 디렉터리로 컴파일합니다. 두 명령은 같은 JSON과 컴파일러 옵션을 사용하며, ONNX 그래프만이 유일한 실험 변수입니다.

<!-- cell: compile-yolo26-baseline-command src: fa81283b97 -->
#### 6.5.1 기준 모델 컴파일

기본 `n` 변형의 경우 다음 코드 셀은 별도의 터미널에서 다음 명령을 입력하는 것과 동일합니다(다른 변형은 `yolo26-n`과 `yolo26n`을 그에 맞게 바꿉니다).

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/yolo26-n_640x640.onnx \
      -c configs/yolo26-n_640x640.local.json \
      -o outputs/yolo26n_baseline \
      --gen_log \
      --export_html
```

출력 디렉터리에 이미 DXNN 파일이 있으면 코드 셀은 해당 파일을 알려 주고 `dxcom`을 건너뜁니다.

<!-- cell: compile-yolo26-topk-command src: 1c1158411a -->
#### 6.5.2 TopK 모델 컴파일

기본 `n` 변형의 경우 다음 코드 셀은 별도의 터미널에서 다음 명령을 입력하는 것과 동일합니다(다른 변형은 `yolo26-n`과 `yolo26n`을 그에 맞게 바꿉니다).

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/yolo26-n_640x640_topk300.onnx \
      -c configs/yolo26-n_640x640.local.json \
      -o outputs/yolo26n_topk300 \
      --gen_log \
      --export_html
```

출력 디렉터리에 이미 DXNN 파일이 있으면 코드 셀은 해당 파일을 알려 주고 `dxcom`을 건너뜁니다.

<!-- cell: benchmark-heading src: 917423f3f9 -->
### 6.6 `dxrun`으로 기준 모델과 TopK 모델 성능 비교

`dxrun --use-ort -t 5`는 합성 입력으로 각 DXNN을 5초 동안 벤치마크합니다. `--use-ort`는 NPU 그래프뿐 아니라 CPU 서브그래프도 실행하며, 이 비교에는 필수입니다.

두 모델을 같은 장치에서 비슷한 발열 및 시스템 부하 조건으로 실행하세요. 이 결과는 런타임 처리량을 측정하는 것이지 검출 정확도나 전체 비디오 파이프라인 FPS를 측정하는 것이 아닙니다. 릴리스 결정을 위해서는 측정을 반복하세요.

기본 `n` 변형의 경우:

```bash
dxrun -m outputs/yolo26n_baseline/yolo26-n_640x640.dxnn --use-ort -t 5
dxrun -m outputs/yolo26n_topk300/yolo26-n_640x640_topk300.dxnn --use-ort -t 5
```

<!-- cell: 6be8d246-d16c-429d-a82f-e1c45594974c src: b7be7c85e3 -->
## 7. 문제 해결

| 증상 | 표시되는 내용 | 원인 | 해결 |
|---|---|---|---|
| PPU 노드 이름을 찾을 수 없음 | `ValueError: ... not found in the ONNX graph` | PPU 설정의 `layer` 이름이 다른 내보내기 모델의 것임 | Netron에서 ONNX를 확인하고 JSON의 노드 이름을 수정 |
| `dxrun`이 장치를 찾지 못함 | `No device found` 또는 `Fail to initialize device` | 이 호스트에 DEEPX NPU가 없음 | NPU가 있는 호스트에서 비교를 실행. 컴파일 전용 단계는 그대로 동작함 |
| FPS를 읽을 수 없음 | `Could not read FPS from dxrun output` | `dxrun` 출력 형식이 변경됨 | 위에서 전체 출력을 출력하고 FPS 줄을 직접 확인 |
| JSON 변경이 반영되지 않음 | `Skip compilation: found ...` | 출력 디렉터리에 이미 DXNN이 있음 | `workspace/outputs/<dir>`를 삭제하고 다시 실행 |
| 다운로드가 불완전함 | `workspace/models`에 아주 작거나 0바이트인 파일 | 다운로드가 중단됨 | 파일을 삭제. 다운로드 헬퍼가 사용 전에 크기를 검증함 |

<!-- cell: summary src: a42f00d89b -->
## 8. 요약

### 8.1 중급 최적화 지도

```text
Representative calibration data
              │
              ▼
     Inspect the detection head
              │
      ┌───────┴────────┐
      │                │
Anchor-based      Anchor-free
  YOLOv7             YOLOX
      │                │
PPU Type 0        PPU Type 1
      │                │
      └───────┬────────┘
              │
              ▼
 Reduce host-side filtering
 and class-selection workload

YOLO26 decoupled head
          │
          ▼
 TopK-first ONNX rewrite
          │
          ▼
 Reduce candidates before
 CPU-side decoding
```

### 8.2 세 가지 최적화 경로

| 주제 | 사용 모델 | 핵심 결정 | 최적화 위치 | 호스트 CPU 담당 |
|---|---|---|---|---|
| PPU Type 0 | YOLOv7 | 앵커 기반 헤드를 선택하고 Conv 노드를 매핑 | 하드웨어 PPU | NMS와 애플리케이션 로직 |
| PPU Type 1 | YOLOX-S | bbox, 객체성, 클래스 브랜치를 매핑 | 하드웨어 PPU | NMS와 애플리케이션 로직 |
| TopK 우선 최적화 | YOLO26 (`YOLO26_VARIANT`, 기본값 n) | 비용이 큰 디코딩 앞으로 TopK를 이동 | 재작성된 ONNX 그래프 | 더 적은 후보를 디코딩 |

### 8.3 완료한 실험

| 실험 | 비교 모델 | 통제 변수 | 검증 |
|---|---|---|---|
| Type 0 PPU | YOLOv7 non-PPU vs PPU | 하드웨어 후처리 | `dxparse`, `dxrun` |
| Type 1 PPU | YOLOX-S non-PPU vs PPU | 하드웨어 후처리 | `dxparse`, `dxrun` |
| TopK 최적화 | YOLO26 기준 모델 vs TopK 300 (선택한 크기) | ONNX 그래프 구조 | 출력 계약, `dxparse`, `dxrun` |

### 8.4 완료 체크리스트

- [ ] 캘리브레이션 데이터와 전처리를 모델 정의의 일부로 다룸
- [ ] NPU, PPU, 호스트 CPU 처리 사이의 경계를 식별함
- [ ] YOLOv7 Type 0 Conv 노드 매핑을 확인함
- [ ] Type 0 PPU 모델을 컴파일하고 확인함
- [ ] YOLOX-S Type 1의 bbox, 객체성, 클래스 매핑을 확인함
- [ ] Type 1 PPU 모델을 컴파일하고 확인함
- [ ] 같은 벤치마크 조건에서 PPU 모델과 non-PPU 모델을 비교함
- [ ] YOLO26 모델에 TopK 우선 최적화를 적용함 (두 번째 크기도 시도)
- [ ] 기준 모델과 TopK 모델이 같은 출력 계약을 유지하는지 확인함
- [ ] 기준 모델과 TopK 모델의 DXNN 런타임 성능을 비교함
- [ ] 실제 데이터로 정확도, 호스트 CPU 사용량, 전체 파이프라인 지연 시간을 검증함

### 8.5 선택 방법

| 모델이 다음에 해당하면... | 다음으로 시작... |
|---|---|
| 앵커 기반 YOLO 헤드 | PPU Type 0 |
| 분리된 bbox, 객체성, 클래스 브랜치 | PPU Type 1 |
| 많은 후보 집합 뒤에 TopK가 있음 | TopK 우선 그래프 최적화 |
| 알 수 없거나 커스터마이즈된 헤드 | 최적화를 선택하기 전에 실제 ONNX 그래프를 확인 |

> **기억하세요:** 호스트 작업량 감소와 합성 입력 FPS 향상은 같은 결과가 아닙니다. `dxrun`으로 통제된 런타임 비교를 수행한 다음, 실제 애플리케이션 워크로드로 과제 정확도, 호스트 CPU 사용량, 종단 간 지연 시간을 측정하세요.

### 8.6 다음 단계

**고급 튜토리얼**로 이어서 다음을 배웁니다.

- Q-Lite, Q-PRO, Q-Master 선택,
- 양자화 진단,
- QXNN 재개 워크플로,
- QAT,
- Python API 컴파일,
- 고급 컴파일러 제어.
