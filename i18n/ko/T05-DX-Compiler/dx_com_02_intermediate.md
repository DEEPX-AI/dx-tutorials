<!-- i18n source: notebooks/T05-DX-Compiler/dx_com_02_intermediate.ipynb -->
<!-- i18n lang: ko -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: intro src: 680d42d3c9 -->
# DEEPX Tutorial 05-2 - DX-COM 중급

이 노트북은 컴파일을 한 번 성공시키는 단계에서 벗어나, 제어 가능하고 설명 가능한 컴파일 워크플로로 나아갑니다.

캘리브레이션 결정을 개선하고, PPU Type 0과 Type 1로 실제 Model Zoo 모델을 빌드하며, YOLO26 검출, 세그멘테이션, 포즈 모델에 TopK 우선 최적화를 적용합니다(기본값은 n이며 s, m, l, x는 변수 하나만 바꾸면 됩니다).

<!-- cell: 6dffc67d-0a31-4b77-b975-65a83973f314 src: caf8fba665 -->
> 이 튜토리얼의 3부 중 2부입니다. 코스 맵과 전체 파트 목록은 초급 노트북에 있습니다.

<!-- cell: objectives src: ef4f57c018 -->
## 학습 목표

이 튜토리얼을 마치면 다음을 할 수 있습니다.

- 대표성 있는 캘리브레이션 데이터를 선택하고 모델 전처리를 재현한다,
- 검출 연산 중 어떤 것이 NPU의 PPU에서 실행되고 어떤 것이 호스트 CPU에 남는지 설명한다,
- 검출 헤드 아키텍처를 보고 PPU Type 0 또는 Type 1을 선택한다,
- PPU 레이어 매핑을 ONNX 그래프와 대조해 검증한다,
- Model Zoo의 YOLOv7과 YOLOX-S 모델을 하드웨어 PPU로 컴파일한다,
- PPU 적용 모델과 비적용 모델의 Model Zoo DXNN 런타임 동작을 비교한다,
- YOLO26 DXNN의 어느 부분이 호스트 CPU에서 실행되며 TopK 우선 최적화가 그 부분에서 무엇을 바꾸는지 설명한다,
- YOLO26 ONNX 모델의 헤드 텐서를 자동으로 찾는다,
- 모든 크기(n, s, m, l, x)의 YOLO26 검출, 세그멘테이션, 포즈 모델에 TopK 우선 최적화를 적용한다,
- 최적화 모델이 같은 결과를 내는지 확인하고, 절약되는 CPU 시간과 종단 간 처리량을 측정한다,
- 애플리케이션을 변경하지 않고 최적화된 DXNN을 DX-APP에서 실행한다.

이 노트북은 SDK 소스 트리를 수정하지 않습니다. 생성되는 모든 파일은 git이 무시하는 `<dx-tutorials>/notebooks/T05-DX-Compiler/workspace` 아래에 저장됩니다.

<!-- cell: 64617a25-7ca0-407c-97d2-80bcd28d505c src: 38b24ba0d7 -->
## 사전 요구 사항

- 튜토리얼 05-1 완료: `venv-dx-compiler-local`의 `dxcom`과 샘플 캘리브레이션 데이터셋.
- 4절과 6절의 `dxrun` 비교를 위해 DEEPX NPU가 있는 환경에 DX-RT 설치.
- 6.8절용: DX-RT Python 환경(`venv-dx-runtime`, 튜토리얼 01)과 DX-APP(튜토리얼 02). 이것들이 없으면 DX-APP 셀만 건너뜁니다.
- 다운로드: Model Zoo의 YOLOv7 및 YOLOX ONNX, DXNN 파일(약 290 MB)과 6절에서 쓰는 YOLO26 검출, 세그멘테이션, 포즈 ONNX 파일(기본 크기 `n` 기준 약 33 MB).
- 소요 시간: YOLOv7 640x640 컴파일은 이 시리즈에서 가장 오래 걸립니다. 노트북 PC에서는 수십 분을 예상하세요. 6절은 YOLO26 모델 여섯 개(검출, 세그멘테이션, 포즈 각각의 기준 모델과 TopK 모델)를 컴파일하며, `n`은 x86_64 데스크톱에서 모델당 약 2분 걸립니다.

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

<!-- cell: ppu-types src: 69849e3a90 -->
### 3.2 모델 아키텍처에 따라 모드 선택

| 모드 | 아키텍처 | 대표 모델 | 레이어 매핑 | 실행 위치 |
|---|---|---|---|---|
| PPU Type 0 | 앵커 기반(anchor-based) | YOLOv3, YOLOv4, YOLOv5, YOLOv7 | 검출 Conv 노드 → 앵커 수 | PPU 하드웨어 |
| PPU Type 1 | 앵커 프리(anchor-free) | YOLOX, YOLOv8–YOLOv12 | `bbox`, (있는 경우) `obj_conf`, `cls_conf` 노드 | PPU 하드웨어 |
| `pre_optimize()` | TopK 우선 ONNX 재작성 | YOLOv8, v9, v11, v12, v13 및 YOLO26(검출, 세그멘테이션, 포즈); RTMDet | 스케일별 헤드 출력 텐서 | NPU + 호스트 CPU 그래프 |

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

<!-- cell: topk-heading src: 7c5984ab3c -->
## 6. YOLO26 TopK 우선 최적화 (n / s / m / l / x)

PPU Type 0과 Type 1은 검출 후처리를 NPU 하드웨어로 옮깁니다. YOLO26은 다른 방식을 택합니다. 후처리는 호스트 CPU에 그대로 남고, DX-COM의 `pre_optimize()`가 그 CPU 부분을 재작성하여 작업량을 크게 줄입니다. 이 절에서는 작업이 어디에서 실행되는지, 재작성이 무엇을 바꾸는지, 결과가 그대로 유지되는지 확인하는 방법, 그리고 시간이 얼마나 절약되는지를 보여 줍니다. 그런 다음 같은 단계를 세그멘테이션 모델과 포즈 모델에 적용합니다.

### YOLO26 후처리가 실행되는 위치

컴파일된 YOLO26 DXNN에는 태스크가 두 개 있습니다. 6.6에서 `dxparse`로 이 태스크들을 확인합니다.

```text
image ─► npu_0 [NPU] ──────────► head tensors ─► cpu_0 [CPU, ONNX Runtime on the host] ─► [1, 300, 6]
         backbone, neck, and     box and class    Sigmoid on the class logits, box decoding,
         the head Convs          logits, 3 scales  keep the best 300 candidates
```

NPU 태스크는 각 헤드 브랜치의 마지막 Conv에서 끝납니다. 그 이후는 모두 일반 ONNX이며, DX-RT가 호스트 CPU에서 ONNX Runtime으로 실행합니다. `dxrun`에 `--use-ort`가 필요한 이유가 바로 이것입니다. 640x640에서 헤드는 후보 8,400개를 만들므로, 기준 모델의 CPU 태스크는 매 프레임마다 클래스 로짓 8,400 x 80 = 672,000개에 Sigmoid를 적용한 뒤에야 그중 300개를 남깁니다.

### `pre_optimize()`가 바꾸는 것

```text
Baseline : 8,400 candidates → Sigmoid on every class logit → decode → TopK 300
Optimized: 8,400 candidates → TopK 300 on the raw logits    → Sigmoid and decode for those 300 only
```

Sigmoid는 점수의 순서를 바꾸지 않으므로, 원시 로짓에 TopK를 적용해도 같은 후보가 선택됩니다. 출력은 `[1, 300, 6]`(박스 `x1 y1 x2 y2`, 점수, 클래스 id)으로 유지되며, 6.5절에서 값이 동일하다는 것을 보여 줍니다. 바뀌는 것은 호스트 CPU 작업량뿐입니다.

YOLO26용 패스는 `yolo_no_dfl_postprocess`입니다(YOLO26 박스는 DFL을 사용하지 않습니다). DX-COM에는 YOLOv8~v13용 `yolo_dfl_postprocess`와 RTMDet용 `rtmdet_postprocess`도 있으며, DX-COM 사용자 매뉴얼의 *Pre-Optimize API*에서 세 가지를 모두 설명합니다.

### 효과가 큰 경우

절약은 호스트 CPU에서 일어납니다. 따라서 각 프레임에서 CPU 부분이 차지하는 비중이 클 때, 즉 Arm 보드처럼 느린 호스트 CPU를 쓰거나 클래스 수가 많은 모델일 때 의미가 있습니다. 빠른 x86_64 데스크톱이나 클래스가 몇 개뿐인 모델에서는 이득이 작습니다. 6.7절에서는 CPU 부분과 종단 간 처리량을 따로 측정하므로, 사용 중인 호스트에서의 효과를 직접 판단할 수 있습니다.

### 패스를 적용하는 두 가지 방법

| 방법 | 방식 | 사용하는 경우 |
|---|---|---|
| Python API | `dx_com.pre_optimize(model, passes=...)`가 새 ONNX 파일을 작성하고, 그 파일을 컴파일 | 컴파일 전에 재작성된 ONNX를 살펴보고 확인하려는 경우(이 절) |
| 컴파일 설정 | 컴파일 JSON의 `pre_optimize` 블록. `dxcom`이 컴파일하면서 패스를 적용 | 패스 설정을 이미 확인했고 컴파일 한 단계로 끝내려는 경우(6.3의 참고 참조) |

### 절 구성

| 단계 | 할 일 | 결과 |
|---|---|---|
| 6.1 | Model Zoo의 YOLO26 검출 모델 다운로드 | 기준 ONNX와 JSON |
| 6.2 | 헤드 텐서 자동으로 찾기 | 패스 설정 |
| 6.3 | 패스 적용 | TopK ONNX |
| 6.4 | 적용 전후의 CPU 부분 비교 | 연산 및 shape 표 |
| 6.5 | 결과가 동일한지 확인 | 실제 이미지로 ONNX Runtime 비교 |
| 6.6 | 두 모델 컴파일 | DXNN 파일 두 개와 각각의 NPU/CPU 분할 |
| 6.7 | 측정 | CPU 부분 시간과 `dxrun` FPS |
| 6.8 | DX-APP에서 TopK DXNN 실행 | 결과 이미지 |
| 6.9 | 세그멘테이션과 포즈에 대해 6.1~6.8 반복 | |

6.1의 `YOLO26_VARIANT`가 Model Zoo 모델 크기를 선택하며, 아래의 모든 파일 이름, 컴파일, 벤치마크가 이 값을 따릅니다. 먼저 `n`으로 시작한 다음 변수를 바꿔 더 큰 모델로 이 절을 다시 실행해 보세요. 각 크기는 `models/`, `configs/`, `outputs/` 아래에 자체 파일을 사용하므로, 이미 컴파일한 크기로 다시 전환하면 컴파일을 건너뜁니다.

| `YOLO26_VARIANT` | 검출 ONNX 다운로드 | 모델당 컴파일 시간 (x86_64 데스크톱) |
|---|---:|---|
| `n` (기본값) | 10 MB | 약 2분 (측정값) |
| `s` | 37 MB | 약 2분 (측정값) |
| `m` | 78 MB | 측정하지 않음, 수 분 예상 |
| `l` | 95 MB | 측정하지 않음, 10분 이상 예상 |
| `x` | 213 MB | 측정하지 않음, 10분 이상 예상 |

<!-- cell: download-yolo26-heading src: 306e030b72 -->
### 6.1 기준 모델 다운로드 및 검증

`YOLO26_VARIANT`를 설정하고 셀을 실행하세요. 기준 ONNX와 그에 맞는 Q-Lite JSON은 Model Zoo에서 가져옵니다(`yolo26-<variant>_640x640.onnx`와 `.json`). 안전 다운로더는 셸 다운로드가 실패했는데도 다음 노트북 셀이 계속 진행될 때 생길 수 있는 빈 파일 오류를 방지합니다.

<!-- cell: verify-heads-heading src: 86c8f319d0 -->
### 6.2 헤드 텐서 찾기

`pre_optimize()`는 NPU 부분이 어디에서 끝나는지 알아야 합니다. 즉, 각 검출 스케일에서 각 헤드 브랜치의 마지막 Conv가 내보내는 출력 텐서입니다. 이것들은 지금 사용하는 바로 그 ONNX 파일의 텐서 이름입니다. 모델과 내보내기 도구(exporter) 버전에 따라 달라지므로 절대 다른 모델에서 복사하지 마세요.

헤드 텐서는 쉽게 알아볼 수 있습니다.

1. shape이 `[1, C, H, W]`인 `Conv`의 출력이며, 여기서 `H = 640 / stride`이고 stride는 8, 16, 32 중 하나입니다(80x80, 40x40, 20x20).
2. 뒤에 활성화 함수가 오지 않으며, `Reshape`나 `Concat`만 이 텐서를 사용합니다.
3. 채널 수 `C`가 역할을 알려 줍니다.

| 역할 | 채널 수 | task |
|---|---|---|
| `bbox` | 4 (박스 네 변까지의 거리) | 전체 |
| `cls_conf` | 클래스 수 (클래스 로짓) | 전체 |
| `mask_coeff` | 32 | `seg` |
| `kpt` | 키포인트 수 x 키포인트당 값 수 (17 x 3 = 51) | `pose` |

실제 모델에서는 두 가지 세부 사항이 더해집니다.

- **채널 수가 같은 경우.** 4클래스 모델에서는 `cls_conf`도 `bbox`처럼 채널이 4개입니다. 클래스 점수 브랜치만 곧바로 `Sigmoid`로 들어가므로, 이것으로 둘을 구분합니다.
- **헤드가 두 개인 경우.** YOLO26은 one-to-many 헤드와 one-to-one 헤드로 학습됩니다. 추론에는 텐서 이름에 `one2one_*`가 들어간 one-to-one 헤드를 사용합니다. one-to-many 헤드만 포함된 내보내기 모델은 이 패스에서 동작하지 않습니다(6.3의 `nms=False` 관련 참고 참조).

다음 셀은 이 규칙을 적용하는 작은 모듈 `yolo26_heads.py`를 작성합니다. 이 모듈은 ONNX 파일의 Ultralytics 메타데이터에서 task, 클래스 수, 키포인트 레이아웃을 읽고, 규칙이 성립하지 않으면 설명과 함께 중단합니다. `onnx` 패키지만 필요하므로 노트북과 DX-COM 환경 모두에서 가져올(import) 수 있습니다. DX-COM 사용자 매뉴얼의 *Identifying Conv Head Tensor Names*에서 같은 규칙과 Netron에서 이를 확인하는 방법을 설명합니다.

<!-- cell: apply-topk-heading src: 45a61021f4 -->
### 6.3 `yolo_no_dfl_postprocess` 적용

`optimize_yolo26.py`는 `dx_com`이 설치된 DX-COM Python 환경에서 실행합니다. 이 스크립트는 `yolo26_heads.py`로 헤드 텐서를 찾고, 패스 설정을 출력한 뒤, TopK 모델을 별도의 ONNX 파일로 작성하므로 기준 모델은 그대로 유지됩니다. 같은 스크립트를 모든 크기와 task에, 그리고 나중에는 직접 만든 모델에도 사용할 수 있습니다.

패스 설정은 다음과 같습니다.

| 키 | 값 | 비고 |
|---|---|---|
| `task` | `base`, `seg`, `pose` 중 하나 | 필수. 각 레이어에 어떤 헤드 텐서가 있어야 하는지 결정 |
| `layers` | 스케일마다 항목 하나, stride 8, 16, 32 순서 | 6.2에서 구함 |
| `num_classes` | 클래스 수 | 기본값 80 |
| `topk` | 디코딩 전에 남길 후보 수 | 기본값 300. Model Zoo의 pre-optimized 프로필은 512를 사용 |

> **DX-COM v2.5.0에서 변경됨.** 이전에 `yolo26_postprocess`라고 불리던 패스는 이제 `yolo_no_dfl_postprocess`이고, `yolo_postprocess`(YOLOv8~v13)는 이제 `yolo_dfl_postprocess`입니다. 이전 이름을 사용하면 오류가 발생합니다. 모든 YOLO 패스에는 `task` 키가 필요하며, `input_height` / `input_width`는 더 이상 허용되지 않습니다(입력 크기는 모델에서 가져옵니다).

> **직접 학습한 YOLO26 모델.** 이 스크립트는 직접 학습한 모델에도 동작합니다. 단, NMS-free 헤드로 내보낸 모델이어야 합니다: `yolo export model=best.pt format=onnx imgsz=640 batch=1 nms=False`. `nms=False` 없이 내보내면 현재 Ultralytics 릴리스(8.4.208로 테스트)는 출력이 `[1, 4 + classes, 8400]`인 one-to-many 헤드를 내보내며, `yolo26_heads.py`는 다시 내보내라는 메시지와 함께 중단합니다. `imgsz`는 고정하고 `dynamic=True`는 사용하지 마세요. `num_classes`와 포즈 모델의 `kpt_shape`는 내보낸 모델의 메타데이터에서 가져옵니다.

> **컴파일 JSON에서 패스 적용.** 스크립트는 설정을 `{"yolo_no_dfl_postprocess": {...}}` 형태로 출력합니다. `dxcom`이 원본 ONNX를 컴파일하면서 패스를 적용하게 하려면, 이 딕셔너리를 리스트에 담아 컴파일 JSON의 `pre_optimize` 아래에 넣으세요: `"pre_optimize": [{"yolo_no_dfl_postprocess": {...}}]`. 그러면 컴파일 로그에 `[pre_optimize] applying pass 'yolo_no_dfl_postprocess' from config`가 표시되고, DXNN은 ONNX 파일 이름을 그대로 유지합니다. `pre_optimize`와 `ppu`를 같은 JSON에 넣지 마세요. DX-COM 2.5.0은 `ppu`로 컴파일하고 `pre_optimize`는 버린 채 `falling back to legacy: ValueError: config 'pre_optimize' and 'ppu' are mutually exclusive`만 출력합니다.

다음 두 셀은 스크립트를 작성하고 실행합니다. 기본 `n` 변형의 경우 다음과 동일합니다.

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
python optimize_yolo26.py models/yolo26-n_640x640.onnx models/yolo26-n_640x640_topk300.onnx
```

<!-- cell: cpu-part-heading src: 94cb953e24 -->
### 6.4 적용 전후의 CPU 부분 비교

컴파일된 DXNN의 CPU 태스크는 헤드 텐서에서 모델 출력까지의 ONNX 서브그래프입니다. 다음 셀은 두 모델에서 이 서브그래프를 잘라 내고(`onnx.utils.extract_model`), 후보 수에 따라 작업량이 늘어나는 연산을 나열합니다.

살펴볼 점:

- **기준 모델:** `Sigmoid`가 모든 후보의 클래스마다 값 하나씩, 즉 `[1, 80, 8400]`에 대해 실행되고, TopK는 그 뒤에 옵니다.
- **TopK 모델:** `ReduceMax`와 `TopK`가 원시 로짓에서 먼저 후보 300개를 선택하고, `Sigmoid`는 `[1, 300, 80]`에 대해 실행됩니다. 처리하는 값이 28배 적습니다.

TopK 모델은 후보 300개를 골라내는 gather 단계가 추가되므로 노드 수가 더 많습니다. 추가된 노드는 8,400행이 아니라 300행에 대해 동작합니다. 잘라 낸 CPU 부분은 6.7에서 CPU 작업 시간을 측정할 때도 사용합니다.

<!-- cell: verify-results-heading src: 368a5b670d -->
### 6.5 결과가 동일한지 확인

그래프를 변경했다면, 컴파일할 가치가 있는지 판단하기 전에 먼저 같은 결과를 내는지 확인해야 합니다. 다음 셀들은 같은 실제 이미지로 두 ONNX 모델을 ONNX Runtime에서 실행하고, 출력을 비교하고, 각 모델의 검출 결과를 그립니다.

이미지는 Model Zoo JSON이 캘리브레이션 이미지를 준비하는 방식과 같게 준비합니다. 값 114로 패딩해 640x640 정사각형으로 만들고, BGR에서 RGB로 변환하고, 255로 나눈 뒤 NCHW로 바꿉니다. 테스트 이미지는 DX-APP 샘플에서 가져오며, DX-APP이 설치되어 있지 않으면 첫 번째 캘리브레이션 이미지를 사용합니다.

비교 대상은 DX Model Zoo가 mAP 평가에 사용하는 임계값인 0.001보다 점수가 높은 모든 후보와, 그 밖의 모든 출력(예: 세그멘테이션 프로토타입)입니다. 점수가 더 낮은 후보는 의도적으로 제외합니다. 목록 아래쪽으로 내려가면 점수가 정확히 같은 후보가 많고, TopK는 동점 후보를 어떤 순서로든 반환할 수 있습니다. 재작성된 모델은 Sigmoid 출력 대신 원시 로짓을 정렬하므로 이런 동점 후보의 순서가 달라질 수 있습니다. 이 노트북의 테스트 이미지에서는 첫 번째 동점이 점수 0.0001 미만에서 나타났습니다.

이 비교는 재작성된 ONNX 그래프를 확인하는 것입니다. 컴파일 시 각 모델을 따로 양자화하므로 두 DXNN 파일은 약간 다를 수 있습니다. 6.8에서는 두 모델을 모두 NPU에서 실행합니다.

<!-- cell: verify-results-note src: 98e55a730d -->
후보 수가 같고 최대 차이가 0이면, 이 이미지에서 재작성된 모델이 같은 결과를 낸다는 뜻입니다. 직접 만든 모델이라면 객체가 많은 이미지와 객체가 없는 이미지를 포함해, 사용하는 데이터에서 고른 여러 이미지로 같은 확인을 실행하세요.

<!-- cell: adapt-yolo26-heading src: 1cce1fabf3 -->
### 6.6 두 모델 컴파일

#### 6.6.1 Model Zoo 설정 조정

두 모델은 같은 캘리브레이션 및 전처리 설정을 사용하므로, 두 컴파일의 유일한 차이는 ONNX 그래프입니다. Model Zoo 서버를 가리키는 데이터셋 경로만 변경합니다.

<!-- cell: compile-yolo26-baseline-command src: 9ec494a9de -->
#### 6.6.2 기준 모델 컴파일

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

<!-- cell: compile-yolo26-topk-command src: 6679ececa5 -->
#### 6.6.3 TopK 모델 컴파일

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

<!-- cell: task-split-heading src: 5ea330f2c1 -->
#### 6.6.4 NPU/CPU 분할 확인

`dxparse -v`는 DXNN의 태스크와 태스크 사이에 전달되는 텐서를 나열합니다. 다음 셀은 그 부분만 출력합니다. 기준 모델에서는 `npu_0`이 6.2에서 찾은 헤드 텐서에서 끝나고, `cpu_0`이 이를 `output0`으로 변환합니다. TopK 모델에서는 NPU가 CPU 부분을 위해 헤드 텐서의 레이아웃도 바꾸며(`preopt_*_tr_*`, channels last), `cpu_0`이 `preopt_output`을 생성합니다.

<!-- cell: benchmark-heading src: 021a6c65d3 -->
### 6.7 효과 측정

두 가지 측정으로 두 가지 질문에 답합니다.

1. **패스가 CPU 작업을 얼마나 줄이는가?** 6.4의 CPU 부분을 ONNX Runtime에서 스레드 하나로 실행하며, 입력으로는 테스트 이미지의 실제 헤드 텐서를 넣습니다. 이렇게 하면 패스가 바꾸는 작업만 따로 떼어 볼 수 있습니다.
2. **전체 파이프라인이 빨라지는가?** `dxrun --use-ort -t 5`는 합성 입력으로 각 DXNN을 5초 동안 실행하며, NPU 부분과 CPU 부분을 함께 실행합니다.

DX-RT는 연속된 프레임의 NPU 태스크와 CPU 태스크를 겹쳐서 실행하므로, 종단 간 FPS는 둘 중 느린 쪽이 결정합니다. 빠른 x86_64 호스트에서 흔히 그렇듯 NPU가 더 느린 단계라면, CPU 부분이 짧아져도 FPS는 거의 바뀌지 않습니다. 호스트 CPU가 느리면 CPU 부분이 한계가 되며, 측정 1에서 절약한 시간이 FPS 향상으로 이어집니다.

기본 `n` 변형의 `dxrun` 명령은 다음과 같습니다.

```bash
dxrun -m outputs/yolo26n_baseline/yolo26-n_640x640.dxnn --use-ort -t 5
dxrun -m outputs/yolo26n_topk300/yolo26-n_640x640_topk300.dxnn --use-ort -t 5
```

두 모델을 같은 장치에서 비슷한 부하와 온도 조건으로 실행하고, 결론을 내리기 전에 측정을 반복하세요. 이 수치는 정확도나 전체 비디오 애플리케이션의 FPS를 측정한 것이 아닙니다.

<!-- cell: benchmark-results src: 7013e3c5ea -->
결과를 읽는 방법:

- **CPU 부분.** 패스가 바꾸는 것은 바로 이 부분입니다. Model Zoo 모델에서는 약 35~45퍼센트 줄어듭니다. 절대 시간은 호스트 CPU에 따라 달라지며, 같은 작업이라도 Arm 코어에서는 x86_64 데스크톱 코어보다 몇 배 더 오래 걸립니다.
- **`dxrun` FPS.** CPU 부분이 파이프라인을 제한하는 만큼만 향상됩니다. 이 노트북을 테스트한 DX-M1 장착 x86_64 데스크톱에서 YOLO26-n의 CPU 부분은 약 0.74 ms → 0.41 ms로 측정되었지만, NPU가 더 느린 단계였기 때문에 `dxrun` FPS는 약 1.06×만 향상되었습니다.
- **Arm 호스트에서는 CPU 부분이 한계입니다.** 이 절에서 만든 DXNN 파일을 그대로 DX-M1을 장착한 Raspberry Pi 5에서 측정한 결과입니다(`dxrun -m <model> --use-ort -t 10`, 기본 설정).

  | YOLO26-n | 기준 모델 FPS | TopK 300 FPS | 속도 향상 |
  |---|---:|---:|---:|
  | det | 100.79 | 179.54 | 1.78× |
  | seg | 64.13 | 95.61 | 1.49× |
  | pose | 142.48 | 265.44 | 1.86× |

  기준 모델 세 개에 대해서는 DX-RT가 `CPU TASK [cpu_0] Inference Worker - Average Input Queue Load : ...% (DXRT_DYNAMIC_CPU_THREAD: OFF) - To improve FPS, set: 'export DXRT_DYNAMIC_CPU_THREAD=ON'`도 출력했습니다. 이 메시지는 프레임이 CPU 태스크를 기다리고 있었다는 뜻입니다. TopK 모델에서는 이 메시지가 나타나지 않았습니다. 사용 중인 보드에서 이 메시지가 보이면 CPU 부분이 병목이며, 이 최적화가 도움이 될 가능성이 높습니다. `DXRT_DYNAMIC_CPU_THREAD=ON`은 CPU 태스크에 더 많은 스레드를 할당하는 별도의 설정이며, 이 수치를 측정할 때는 사용하지 않았습니다.
- **DX-COM 매뉴얼의 참고 결과**, ARM Cortex-A53 호스트의 DX-M1 기준(`DXRT_DYNAMIC_CPU_THREAD=ON dxrun -m <model> -l 1000 --use-ort`): det 184.89 → 315.73 FPS (1.7×), seg 130.82 → 195.05 FPS (1.5×), pose 242.77 → 304.38 FPS (1.3×).

자사 제품에 적용할지 결정하려면 대상 보드에서 직접 이 절을 실행하세요.

<!-- cell: dx-app-heading src: a0178a7c3f -->
### 6.8 DX-APP에서 TopK DXNN 실행

애플리케이션 코드는 바뀌지 않습니다. DX-APP의 YOLO26 후처리기는 첫 번째 출력을 위치로 읽고, DX-STREAM의 `yolo26_od`, `yolo26_seg`, `yolo26_pose` 후처리 라이브러리는 shape(`[1, 300, 6]`, 38열 이상, 57열 이상)으로 출력을 선택합니다. 따라서 출력 이름이 바뀌어도(`output0` → `preopt_output`) 수정할 필요가 없습니다. 두 경우 모두 CPU 태스크를 위해 여전히 ONNX Runtime이 필요합니다.

다음 셀은 해당 모델의 DX-APP Python 예제를 같은 이미지에 대해 각 DXNN으로 실행하고 두 결과를 보여 줍니다. 튜토리얼 01의 DX-RT Python 환경과 튜토리얼 02의 DX-APP 소스를 사용하며, 이것들이 없으면 셀을 건너뜁니다. 기본 `n` 변형의 경우 다음과 동일합니다.

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_app
<DX_ALL_SUITE_DIR>/dx-runtime/venv-dx-runtime/bin/python \
  src/python_example/object_detection/yolo26/yolo26-n_640x640/yolo26-n_640x640_sync.py \
  --model <workspace>/outputs/yolo26n_topk300/yolo26-n_640x640_topk300.dxnn \
  --image sample/img/sample_people.jpg --no-display --save --save-dir <workspace>/outputs/dx_app_results
```

이 노트북의 테스트 머신에서는 두 DXNN 파일이 같은 객체를 찾았습니다. 각 모델이 따로 양자화되기 때문에 박스 모서리는 최대 약 2픽셀, 점수는 최대 약 0.05 차이가 났습니다. 배포 전에 검증 데이터셋으로 정확도를 확인하세요. 예를 들어 DX Model Zoo 평가 도구를 사용할 수 있습니다.

<!-- cell: seg-pose-heading src: 61a29534c1 -->
### 6.9 세그멘테이션과 포즈: `task`만 바꾼 같은 패스

DX-COM v2.5.0은 `yolo_no_dfl_postprocess`를 YOLO26 세그멘테이션과 포즈 모델로 확장했습니다. 패스 이름, `topk`, `bbox`와 `cls_conf` 텐서는 6.3과 같습니다. 바뀌는 것은 두 가지로, `task` 키와 스케일마다 추가되는 헤드 텐서 하나입니다.

| `task` | Model Zoo 모델 (`n`) | 스케일별 헤드 텐서 | `num_classes` | 패스 적용 후 출력 |
|---|---|---|---:|---|
| `base` | `yolo26-n_640x640` | `bbox` (4 ch), `cls_conf` (80 ch) | 80 | `[1, 300, 6]` |
| `seg` | `yolo26-n-seg_640x640` | + `mask_coeff` (32 ch), `one2one_cv4.<scale>` | 80 | `[1, 300, 38]`와 프로토타입 `[1, 32, 160, 160]` |
| `pose` | `yolo26-n-pose_640x640` | + `kpt` (51 ch), `one2one_cv4_kpts.<scale>`, `kpt_shape` `[17, 3]` 포함 | 1 | `[1, 300, 57]` |

- **세그멘테이션.** 남은 후보 300개 각각이 마스크 계수 32개를 함께 가집니다. 마스크 프로토타입 출력은 바뀌지 않고 그대로 전달되며, 계수와 프로토타입을 결합해 마스크를 만드는 일은 여전히 호스트가 합니다.
- **포즈.** 키포인트 헤드는 Conv 하나로 되어 있어서, 텐서 이름(`/model.23/one2one_cv4_kpts.<scale>/Conv_output_0`)이 다른 브랜치의 `...<scale>.2/Conv_output_0` 패턴과 다릅니다. 6.2의 규칙은 이름에 의존하지 않으므로 `yolo26_heads.py`는 이 텐서도 찾아냅니다. `kpt_shape`는 ONNX 메타데이터에서 가져오며, 17 x 3은 키포인트 Conv의 채널 수 51과 같아야 합니다.
- **검증.** `pre_optimize`는 그래프를 수정하기 전에 헤드 텐서를 `task`와 대조합니다. `seg`는 `mask_coeff`가, `pose`는 `kpt`와 `kpt_shape`가 반드시 필요하며, 각각 상대 쪽 텐서는 거부합니다. 다만 세그멘테이션 모델을 `task="base"`로 실행한 경우는 감지하지 못합니다. 이 경우 컴파일은 되지만 마스크가 사라집니다. 그래서 `yolo26_heads.py`는 `task`를 모델 메타데이터와 비교합니다.

다음 셀들은 두 모델에 대해 6.1~6.8을 셀 하나에 한 단계씩 반복합니다. Model Zoo가 모든 크기의 `seg`와 `pose` 모델을 제공하므로 이 셀들도 `YOLO26_VARIANT`를 따릅니다.

<!-- cell: seg-pose-prepare-heading src: fcd279e22c -->
#### 6.9.1 모델 다운로드 및 헤드 텐서 찾기

<!-- cell: seg-pose-apply-heading src: 1c5e139d9d -->
#### 6.9.2 패스 적용 및 CPU 부분 비교

기본 `n` 변형의 경우 최적화 단계는 다음과 동일합니다.

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
python optimize_yolo26.py models/yolo26-n-seg_640x640.onnx models/yolo26-n-seg_640x640_topk300.onnx --task seg
python optimize_yolo26.py models/yolo26-n-pose_640x640.onnx models/yolo26-n-pose_640x640_topk300.onnx --task pose
```

스크립트가 메타데이터에서 task를 읽으므로 여기서 `--task`는 선택 사항입니다. 이 인수를 전달하면 기대하는 task를 명시하게 되며, 모델과 일치하지 않으면 스크립트가 중단합니다. 포즈 표에서는 키포인트 `Sigmoid`도 살펴보세요. `[1, 17, 1, 8400]`에서 `[1, 300, 17, 1]`로 바뀝니다.

<!-- cell: seg-pose-verify-heading src: 7657781fd1 -->
#### 6.9.3 결과가 동일한지 확인

세그멘테이션에서는 마스크 계수와 프로토타입을, 포즈에서는 키포인트를 비교합니다. 이미지에는 각 모델 자체의 출력으로 마스크와 키포인트를 그립니다.

<!-- cell: seg-pose-compile-heading src: e98f1b66b9 -->
#### 6.9.4 기준 모델과 TopK 모델 컴파일

이 셀은 각 모델의 로컬 JSON을 작성하고(6.6.1과 같은 방식) 6.6과 같은 옵션으로 모델 네 개를 컴파일합니다. 기본 `n` 변형의 경우 다음 명령과 동일합니다(포즈 모델은 `seg`를 `pose`로 바꾸세요).

```bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
cd <dx-tutorials>/notebooks/T05-DX-Compiler/workspace
dxcom -m models/yolo26-n-seg_640x640.onnx \
      -c configs/yolo26-n-seg_640x640.local.json \
      -o outputs/yolo26n_seg_baseline \
      --gen_log \
      --export_html
dxcom -m models/yolo26-n-seg_640x640_topk300.onnx \
      -c configs/yolo26-n-seg_640x640.local.json \
      -o outputs/yolo26n_seg_topk300 \
      --gen_log \
      --export_html
```

출력 디렉터리에 이미 DXNN 파일이 있으면 셀은 그 파일을 알려 주고 해당 모델의 `dxcom`을 건너뜁니다.

<!-- cell: seg-pose-benchmark-heading src: 083b768e06 -->
#### 6.9.5 검출, 세그멘테이션, 포즈 측정

이 셀은 6.7과 같은 방식으로 새 DXNN 파일 네 개의 CPU 부분과 `dxrun` FPS를 측정하고, 6.7의 검출 결과를 표에 함께 넣습니다.

<!-- cell: seg-pose-results src: 967a128494 -->
세 task 비교:

- **검출과 세그멘테이션**은 같은 곳에서 CPU 시간을 절약합니다. 바로 80개 클래스 x 후보 8,400개에 대한 클래스 점수 `Sigmoid`입니다. 세그멘테이션은 패스가 바꾸지 않는 마스크 프로토타입 출력도 유지하고 여전히 호스트가 마스크를 만들기 때문에, 상대적인 이득이 더 작습니다.
- **포즈**는 클래스가 하나뿐이라 클래스 점수 작업이 적습니다. 포즈의 절약은 키포인트 가시성 `Sigmoid`(17 x 8,400개 값)와 키포인트 디코딩에서 나오며, 이제 이 작업은 후보 300개에 대해서만 실행됩니다. 그 효과가 얼마나 큰지는 호스트에 따라 다릅니다. 위의 Raspberry Pi 5에서는 포즈의 향상 폭이 가장 컸지만(1.86×), 매뉴얼의 Cortex-A53 결과에서는 가장 작았습니다(1.3×).

<!-- cell: seg-pose-dx-app-heading src: 42492b118d -->
#### 6.9.6 DX-APP에서 세그멘테이션과 포즈 DXNN 파일 실행

<!-- cell: 6be8d246-d16c-429d-a82f-e1c45594974c src: 8bd09629d5 -->
## 7. 문제 해결

| 증상 | 표시되는 내용 | 원인 | 해결 |
|---|---|---|---|
| PPU 노드 이름을 찾을 수 없음 | `ValueError: ... not found in the ONNX graph` | PPU 설정의 `layer` 이름이 다른 내보내기 모델의 것임 | Netron에서 ONNX를 확인하고 JSON의 노드 이름을 수정 |
| `dxrun`이 장치를 찾지 못함 | `No device found` 또는 `Fail to initialize device` | 이 호스트에 DEEPX NPU가 없음 | NPU가 있는 호스트에서 비교를 실행. 컴파일 전용 단계는 그대로 동작함 |
| FPS를 읽을 수 없음 | `Could not read FPS from dxrun output` | `dxrun` 출력 형식이 변경됨 | 위에서 전체 출력을 출력하고 FPS 줄을 직접 확인 |
| JSON 변경이 반영되지 않음 | `Skip compilation: found ...` | 출력 디렉터리에 이미 DXNN이 있음 | `workspace/outputs/<dir>`를 삭제하고 다시 실행 |
| `pre_optimize`가 헤드 텐서를 거부함 | `ValueError: [yolo_no_dfl_postprocess] task='seg' requires 'mask_coeff' on every layer` 또는 `Cannot find shape for mask_coeff=...` | `task`와 헤드 텐서가 모델과 맞지 않음 | 바로 이 ONNX 파일에서 `find_yolo26_heads()`로 텐서를 찾음 |
| head finder가 task 확인에서 멈춤 | `The model metadata says task='seg', not 'base'` | task 인수가 모델과 맞지 않음 | 메시지에 표시된 task를 전달. `base`로 실행하면 세그멘테이션 모델이 마스크 없이 컴파일됨 |
| head finder가 one-to-many 헤드라고 보고함 | `This ONNX file has the one-to-many head (output [1, 4 + classes, 8400])` | `nms=False` 없이 내보낸 모델임 | `yolo export ... format=onnx nms=False`로 다시 내보냄 |
| head finder가 `cls_conf` Conv를 찾지 못함 | `expected one cls_conf head Conv with N channels, found none` | 클래스 수가 잘못되었거나 메타데이터에 없음 | `num_classes`를 전달(스크립트에서는 `--num-classes`) |
| head finder가 동적 입력이라고 보고함 | `The input height is dynamic` | `dynamic=True`로 내보낸 모델임 | 고정된 `imgsz`로 다시 내보냄 |
| TopK DXNN이 `dxrun`에서 빨라지지 않음 | 상대 처리량이 1.00×에 가까움 | 호스트 CPU가 아니라 NPU가 파이프라인을 제한함(x86_64에서 흔함), 또는 모델의 클래스 수가 적음 | 6.7의 CPU 부분 시간을 비교하고, 대상 보드에서 측정 |
| JSON의 `pre_optimize`가 적용되지 않음 | `falling back to legacy: ... 'pre_optimize' and 'ppu' are mutually exclusive` | JSON에 `ppu` 블록도 있음 | JSON에서 `ppu`를 제거 |
| 다른 클래스를 쓰는 모델인데 DX-APP이 COCO 레이블을 표시함 | 객체에 COCO 클래스 이름이 표시됨 | 레이블은 예제 자체의 `config.json`에서 가져오며, `--config`로는 바뀌지 않음 | `scripts/extract_model_package.sh --no-prune`(튜토리얼 02)로 독립 패키지를 내보내고, 그 `config.json`에서 `num_classes`와 `visualizer.kwargs.custom_labels`를 설정 |
| 다운로드가 불완전함 | `workspace/models`에 아주 작거나 0바이트인 파일 | 다운로드가 중단됨 | 파일을 삭제. 다운로드 헬퍼가 사용 전에 크기를 검증함 |

<!-- cell: summary src: a00cdd65b0 -->
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

YOLO26 det / seg / pose head
          │
          ▼
 Find head tensors → TopK-first ONNX rewrite
 (task = base / seg / pose)
          │
          ▼
 Same results, less host-CPU work
 per frame
```

### 8.2 세 가지 최적화 경로

| 주제 | 사용 모델 | 핵심 결정 | 최적화 위치 | 호스트 CPU 담당 |
|---|---|---|---|---|
| PPU Type 0 | YOLOv7 | 앵커 기반 헤드를 선택하고 Conv 노드를 매핑 | 하드웨어 PPU | NMS와 애플리케이션 로직 |
| PPU Type 1 | YOLOX-S | bbox, 객체성, 클래스 브랜치를 매핑 | 하드웨어 PPU | NMS와 애플리케이션 로직 |
| TopK 우선 최적화 | YOLO26 검출, 세그멘테이션, 포즈 (`YOLO26_VARIANT`, 기본값 n) | TopK를 Sigmoid와 디코딩 앞으로 이동하고, `task`와 헤드 텐서를 지정 | 재작성된 ONNX 그래프 (Python API 또는 컴파일 JSON) | 더 적은 후보(와 그 마스크 계수 또는 키포인트)를 디코딩 |

### 8.3 완료한 실험

| 실험 | 비교 모델 | 통제 변수 | 검증 |
|---|---|---|---|
| Type 0 PPU | YOLOv7 non-PPU vs PPU | 하드웨어 후처리 | `dxparse`, `dxrun` |
| Type 1 PPU | YOLOX-S non-PPU vs PPU | 하드웨어 후처리 | `dxparse`, `dxrun` |
| TopK 최적화 | YOLO26 검출, 세그멘테이션, 포즈: 기준 모델 vs TopK 300 (선택한 크기) | ONNX 그래프 구조 | ONNX Runtime 출력 비교, CPU 부분 시간, `dxparse`, `dxrun`, DX-APP |

### 8.4 완료 체크리스트

- [ ] 캘리브레이션 데이터와 전처리를 모델 정의의 일부로 다룸
- [ ] NPU, PPU, 호스트 CPU 처리 사이의 경계를 식별함
- [ ] YOLOv7 Type 0 Conv 노드 매핑을 확인함
- [ ] Type 0 PPU 모델을 컴파일하고 확인함
- [ ] YOLOX-S Type 1의 bbox, 객체성, 클래스 매핑을 확인함
- [ ] Type 1 PPU 모델을 컴파일하고 확인함
- [ ] 같은 벤치마크 조건에서 PPU 모델과 non-PPU 모델을 비교함
- [ ] YOLO26 DXNN의 어느 부분이 호스트 CPU에서 실행되는지 설명함
- [ ] 이름을 복사하는 대신 `find_yolo26_heads()`로 YOLO26 헤드 텐서를 찾음
- [ ] YOLO26 모델에 TopK 우선 최적화를 적용함 (두 번째 크기도 시도)
- [ ] 기준 모델과 TopK 모델이 동일한 출력을 내는지 ONNX Runtime으로 확인함
- [ ] CPU 부분 시간과 `dxrun` FPS를 측정하고, 둘이 왜 다른지 설명함
- [ ] 애플리케이션을 변경하지 않고 DX-APP에서 TopK DXNN을 실행함
- [ ] `task` = `seg`, `pose`로 YOLO26 세그멘테이션과 포즈 모델에 같은 패스를 적용함
- [ ] 실제 데이터로 정확도, 호스트 CPU 사용량, 전체 파이프라인 지연 시간을 검증함

### 8.5 선택 방법

| 모델이 다음에 해당하면... | 다음으로 시작... |
|---|---|
| 앵커 기반 YOLO 헤드 | PPU Type 0 |
| 분리된 bbox, 객체성, 클래스 브랜치 | PPU Type 1 |
| 많은 후보 집합 뒤에 TopK가 있고, CPU 부분이 FPS를 제한하는 호스트 | TopK 우선 그래프 최적화 |
| 세그멘테이션 또는 포즈 헤드 (YOLO26 또는 YOLOv8~v13) | `task` = `seg` 또는 `pose`로 TopK 우선 그래프 최적화 |
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
