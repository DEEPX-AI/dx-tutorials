<!-- i18n source: notebooks/T10-demo-paddleocr/paddleocr_v6.ipynb -->
<!-- i18n lang: ko -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: t10-v6-001 src: b22c49c292 -->
# DEEPX Tutorial 10 - DEEPX NPU에서 PP-OCRv6 실행

이 튜토리얼은 PP-OCRv6를 소개하고 DEEPX NPU에서 ONNX부터 DXNN까지 이어지는 전체 OCR 워크플로를 시연합니다. 컴파일과 수업 진행 시간을 짧게 유지할 수 있어 PP-OCRv6_tiny를 대상 티어로 선택했습니다.

워크플로는 독립적으로 확인할 수 있는 단계로 나뉩니다.

1. 원본 ONNX 모델을 다운로드하고,
2. 동적 입력 차원을 고정 형상으로 바꾸고,
3. 고정된 모델이 ONNX 결과를 유지하는지 검증하고,
4. 모델을 DXNN으로 컴파일하고,
5. 검출과 텍스트 인식을 하나의 애플리케이션으로 실행합니다.

<!-- cell: t10-v6-002 src: f4377ce97d -->
## 학습 목표

이 튜토리얼을 마치면 다음을 할 수 있습니다.

- PP-OCRv6 아키텍처를 설명하고 tiny, small, medium 티어 중에서 선택하고,
- 검출 → 인식 OCR 파이프라인을 설명하고,
- 하나의 동적 인식 모델이 왜 여섯 개의 고정 형상 DXNN 모델이 되는지 설명하고,
- 원본 ONNX 파일을 안전하게 다운로드하고 검증하고,
- 전처리가 애플리케이션과 일치하는 캘리브레이션 설정을 만들고,
- 실패를 숨기지 않고 <code>dxcom</code>으로 모델을 컴파일하고,
- 생성된 DXNN 산출물을 검사하고,
- 검토된 카메라 애플리케이션을 실행하고 실제 OCR 정확도의 한계를 파악할 수 있습니다.

<!-- cell: t10-npu-pattern src: deed40455a -->
## 이 애플리케이션의 NPU 사용 방식

| 항목 | 이 튜토리얼에서 | 배운 곳 |
|---|---|---|
| 엔진 | 여섯 개의 `InferenceEngine` 객체가 상주합니다. 검출기 하나와 인식기 다섯 개이며, 인식기는 텍스트 크롭마다 종횡비에 따라 선택됩니다 | T06-2 §3 |
| 실행 | 검출은 동기(`run()`)로 실행하고, 인식기는 한 프레임의 모든 크롭을 `run_async()`로 실행한 뒤 `wait()`로 결과를 수집합니다 | T06-2 §4, §5 |
| 태스크 그래프 | 컴파일된 각 모델은 NPU 태스크 뒤에 `cpu_0` 태스크가 이어집니다(DX-COM이 CPU에 남겨 둔 연산자를 ONNX Runtime이 실행) | T06-1 §6 |
| 요청당 입력 | DET `[1, 640, 640, 3]` UINT8 = 1.2 MB, REC 크롭은 높이 48픽셀에 너비 120~1200 | T06-3 §4 |
| 측정할 것 | 검출기와 인식기 하나의 `dxrun` 처리량(6.3절)을 애플리케이션의 프레임당 시간과 비교합니다. 단어가 많은 프레임은 REC 요청도 많으므로 지연 시간을 텍스트 수 기준으로 보고하세요 | T06-1 §5 |

<!-- cell: 659aa80e-30b9-4cf8-ae33-1ed4bf0cbd53 src: cf3775d539 -->
## 사전 요구 사항

- 튜토리얼 01 완료: DX-COM, DEEPX NPU가 장착된 DX-RT, `uv`.
- 다운로드: Hugging Face의 ONNX 모델 두 개(1.8 MB, 4.5 MB)와 `cs.deepx.ai`의 캘리브레이션 아카이브(110 MB)
- 소요 시간: 약 15분. 6절의 `dxcom` 컴파일 여섯 번은 각각 몇 초(검출기는 약 10초)가 걸립니다. 7.1절은 `workspace/.venv-ocr` 아래에 90 MB의 Python 환경을 만듭니다.
- DX-RT 3.4.2와 DX-COM 2.4.1로 검증했습니다. 모델은 여기서 직접 컴파일하므로 항상 사용 중인 컴파일러와 일치합니다.
- 8절의 카메라 애플리케이션에는 USB 카메라와 디스플레이가 필요하며, `--check` 스모크 테스트는 둘 다 없어도 실행됩니다.

다음 셀은 SDK 위치를 찾고, 이 요구 사항을 확인한 뒤 상태 표를 출력합니다. `MISSING`으로 표시된 항목에는 그것을 제공하는 단계가 함께 안내됩니다.

<!-- cell: t10-v6-overview src: fd7df8aac3 -->
## 1. PP-OCRv6 개요

### 1.1 OCR이란?

**광학 문자 인식**(OCR)은 다양한 종류의 문서(스캔한 종이 문서, PDF 파일, 디지털 카메라로 촬영한 이미지)를 편집하고 검색할 수 있는 데이터로 변환하는 기술입니다.

AI에 "눈"을 달아 주는 것이라고 생각하면 됩니다. 일반적으로 두 단계 파이프라인으로 동작합니다.
1. 텍스트 검출: 이미지에서 텍스트가 어디에 있는지 찾습니다(상자를 그립니다).
2. 텍스트 인식: 그 상자 안의 문자가 무엇인지 해독합니다.

<!-- cell: 5b35baad-56a0-4995-aaa9-9b39b7510f74 src: e4261a9a39 -->
### 1.2 PP-OCRv6란?

PaddleOCR은 Baidu가 PaddlePaddle 프레임워크를 기반으로 개발한 초경량 오픈 소스 OCR 시스템입니다.

[PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR)는 PaddleOCR 범용 OCR 모델 제품군의 최신 세대입니다. 텍스트 검출과 텍스트 인식 모두에 새로운 **PPLCNetV4** 백본을 사용하며, 엣지 장치부터 서버까지 세 가지 배포 티어를 제공합니다.

[공식 PP-OCRv6 기술 문서](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)에 설명된 주요 개선 사항은 다음과 같습니다.

- **확장 가능한 하나의 모델 제품군:** tiny, small, medium 티어가 1.5M에서 34.5M 파라미터까지 분포합니다.
- **통합 다국어 인식:** small과 medium은 하나의 모델로 50개 언어를 지원하고, tiny는 일본어를 제외한 49개 언어를 지원합니다.
- **향상된 텍스트 검출:** RepLKFPN이 넥 파라미터를 줄이면서 수용 영역을 확장합니다.
- **향상된 텍스트 인식:** EncoderWithLightSVTR이 지역 문맥과 전역 어텐션을 결합하고, CTC가 효율적인 병렬 디코딩을 제공합니다.
- **특수 장면 지원:** 공식 평가에 손글씨, 회전 및 예술적 텍스트, 디지털 디스플레이, 도트 매트릭스 문자, 타이어 각인, 기타 산업용 텍스트가 포함됩니다.

<img src="assets/ppocrv6-backbone.jpg" style="max-width: 1000px; width: 100%;" alt="PP-OCRv6 인식 및 검출을 위한 PPLCNetV4 백본 설계">

*PPLCNetV4는 태스크 적응형 다운샘플링을 사용합니다. 인식은 수평 시퀀스 정보를 보존하고, 검출은 다중 스케일 특징 맵을 생성합니다. 출처: [공식 PaddleOCR PP-OCRv6 문서](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.md).*

<!-- cell: t10-v6-tier-comparison src: 998aa1c53c -->
### 1.3 Tiny, small, medium

세 티어는 같은 PP-OCRv6 설계를 서로 다른 규모로 사용합니다. 티어가 클수록 일반적으로 어려운 사례의 정확도가 높아지지만 모델 크기와 연산 비용도 증가합니다.

| 티어 | 파라미터 | 대상 환경 | 검출 Hmean (%) | 인식 정확도 (%) | Intel Xeon OpenVINO (초/이미지) | NVIDIA A100 PaddlePaddle (초/이미지) |
|---|---:|---|---:|---:|---:|---:|
| **tiny** | 1.5M | 엣지 / IoT | 80.6 | 73.5 | **0.20** | **0.13** |
| **small** | 7.7M | 모바일 / 데스크톱 | 84.1 | 81.3 | 0.59 | 0.25 |
| **medium** | 34.5M | 서버 / 최고 정확도 | **86.2** | **83.2** | 1.40 | 0.29 |

> **표 읽는 법:** 정확도 값은 PaddleOCR의 내부 다중 시나리오 벤치마크에서 가져왔습니다. 속도는 일반 및 문서 이미지 200장에 대한 이미지당 종단 간 초 단위 시간이며 이미지 I/O, 전처리, 후처리, 추론을 포함합니다. 이 CPU/GPU 수치는 티어 간 상대적인 트레이드오프를 설명하기 위한 것이며, **DEEPX NPU 측정값이 아니고** DX-COM 컴파일 시간을 예측하지도 않습니다.

<img src="assets/ppocrv6-performance.png" style="max-width: 1100px; width: 100%;" alt="공식 PP-OCRv6 검출 및 인식 정확도 비교">

*왼쪽: 평균 텍스트 검출 Hmean. 오른쪽: 가중 평균 텍스트 인식 정확도. 출처: [공식 PaddleOCR PP-OCRv6 문서](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.md).*

<!-- cell: t10-v6-tiny-selection src: c7b6b6fe09 -->
### 1.4 이 튜토리얼이 PP-OCRv6_tiny를 선택한 이유

이 튜토리얼은 수업 중 모델 준비와 컴파일 시간을 최소화하기 위해 **tiny** 티어를 선택합니다. 1.5M 파라미터라는 작은 크기 덕분에 엣지 및 IoT 배포의 자연스러운 출발점이기도 합니다. 이는 튜토리얼 시간을 고려한 결정이지 tiny가 최고의 프로덕션 모델이라는 주장은 아닙니다. 측정된 정확도 향상이 추가 자원을 정당화한다면 small이나 medium을 선택하세요.

tiny에는 두 가지 중요한 트레이드오프가 있습니다.

1. small과 medium보다 공식 검출 및 인식 정확도가 낮습니다.
2. small과 medium은 50개 언어를 지원하는 반면, tiny는 일본어를 제외한 49개 언어를 지원합니다.

> 4절에서 공식 PP-OCRv6_tiny 검출기와 인식기 ONNX 모델을 다운로드합니다. 이어서 7.3절에서는 라이브 데모 전에 애플리케이션의 인식 사전, 전처리, 텐서 계약이 이 모델들과 일치하는지 확인합니다.

<img src="assets/ppocrv6-detection-comparison.jpg" style="max-width: 1000px; width: 100%;" alt="산업용 및 어려운 텍스트에 대한 공식 PP-OCRv6 medium 텍스트 검출 비교">

*이 공식 정성 비교 그림은 PP-OCRv6_medium을 사용해 어려운 검출 시나리오를 보여 주는 것이며, tiny 티어의 정확도 결과가 아닙니다. 출처: [공식 PaddleOCR PP-OCRv6 문서](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.md).*

<!-- cell: t10-v6-006 src: 243e5834d1 -->
## 2. 튜토리얼 환경 준비

이 노트북은 공유 튜토리얼 경로 헬퍼를 통해 <code>config.json</code>을 읽습니다. SDK가 튜토리얼 저장소 안에 설치되어 있다고 가정하지 않습니다.

<!-- cell: t10-v6-008 src: a5b14575cb -->
### 2.1 현재 uv 환경에 Python 패키지 설치

Jupyter 환경은 uv로 만들어졌으며 <code>pip</code> 모듈이 없을 수 있습니다. 따라서 다음 셀은 <code>%pip</code> 대신 <code>uv pip install --python ...</code>을 사용합니다.

셀을 다시 실행해도 안전합니다. uv는 요구 사항을 이미 충족하는 패키지를 재사용합니다.

<!-- cell: t10-v6-004 src: 33b7724658 -->
## 3. OCR 파이프라인 이해

PP-OCRv6는 두 개의 모델 단계로 실행됩니다. 검출 후 인식이라는 같은 흐름이 모든 티어(tiny, small, medium)에 적용됩니다.

| 단계 | 입력 | 출력 | 목적 |
|---|---|---|---|
| 텍스트 검출 | 전체 이미지, 640 × 640 | 텍스트 다각형 | 텍스트 영역 찾기 |
| 텍스트 인식 | 텍스트 크롭 하나, 높이 48 | 문자 시퀀스 | 픽셀을 텍스트로 변환 |

<img src="assets/ocr-workflow.jpg" style="max-width: 980px;" alt="PP-OCR 워크플로">

PaddleOCR을 DX NPU에 적용하려면 다음 4단계가 필요합니다.

1. PaddleOCR ONNX 모델 다운로드

2. 동적 입력 형상 고정

3. DX NPU용으로 ONNX를 *.dxnn으로 컴파일

4. DEEPX-SDK로 OCR 애플리케이션 구현

<!-- cell: t10-v6-010 src: 1523e85531 -->
## 4. 리소스 다운로드 및 검사

### 4.1 ONNX 모델 다운로드

이 튜토리얼은 **tiny** 검출기와 인식기를 Hugging Face의 공식 PaddlePaddle ONNX 저장소에서 직접 다운로드합니다.

- [PP-OCRv6_tiny_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx)
- [PP-OCRv6_tiny_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx)

**DET**와 **REC** ONNX 모델만 다운로드합니다. 선택 사항인 텍스트 라인 방향 분류기는 이 튜토리얼에서 사용하지 않습니다.

재현 가능한 튜토리얼 결과를 위해 각 URL은 특정 공식 저장소 리비전에 고정되어 있습니다. 다운로드는 HTTPS 인증서 검증을 사용하고, 임시 <code>.part</code> 파일에 기록한 뒤, SHA-256 체크섬을 검증하고 나서 대상 파일을 교체합니다. 일치하는 기존 파일은 재사용하고, 오래되었거나 다른 파일은 다시 다운로드합니다.

<!-- cell: t10-v6-calibration-download src: fcfdb536d1 -->
### 4.2 캘리브레이션 데이터셋 다운로드

OCR 캘리브레이션 이미지를 이 튜토리얼 디렉터리에 다운로드하고 압축을 풉니다. 아카이브는 다음 디렉터리를 생성합니다.

```text
models/
├── det_dataset/
└── rec_dataset/
```

아카이브 파일이 완료 표시 역할을 합니다. <code>ocr-dataset.tar.gz</code>가 이미 있으면 다운로드와 압축 해제를 모두 건너뜁니다.

<!-- cell: t10-v6-013 src: d18bf1f9fc -->
원본 모델에서는 배치 크기와 이미지 크기가 동적입니다.

동적 형상은 일반적인 ONNX Runtime 애플리케이션에는 유용하지만, DEEPX 컴파일은 매번 구체적인 입력 형상을 요구합니다.

<!-- cell: t10-v6-014 src: f5160e4612 -->
## 5. 동적 입력 형상 고정

다음 표는 고정 모델에 대한 단일 기준입니다. 형상은 NCHW 순서, 즉 배치, 채널, 높이, 너비를 사용합니다.

<!-- cell: b7060f4d-1c5c-4ac6-b461-2eb3a5598a95 src: e31dd6c875 -->
### 5.1 텍스트 인식 모델의 입력 형상 고정

**왜 5개의 서로 다른 모델이 필요할까요?**

NPU는 고정된 입력 형상을 요구하므로, 짧은 단어('Hi' 같은)와 긴 문장을 같은 너비 1200의 상자로 처리하면 패딩이 과도해지고 세부 정보가 손실됩니다.

서로 다른 종횡비의 '버킷'을 만들면 텍스트를 더 효율적이고 정확하게 처리할 수 있습니다. 그래서 이 튜토리얼은 종횡비가 다른 다섯 개의 개별 인식 모델을 사용합니다(검출 모델 하나를 더해 DXNN 파일은 총 여섯 개).

각 경우마다 검출된 텍스트의 비율에 가장 잘 맞는 모델을 선택해 적용합니다.

<img src="assets/ocr-ratio.png" style="max-width: 800px;">

다음 gif 애니메이션은 실제로 검출된 텍스트의 비율에 따라 어떤 텍스트 인식 모델이 선택되는지 보여 줍니다.

<img src="assets/ocr-ratio.gif" style="max-width: 800px;">

<!-- cell: t10-v6-016 src: 231600d020 -->
### 5.2 `onnxsim`으로 동적 입력 형상 고정

노트북은 현재 커널의 Python 인터프리터를 통해 ONNX Simplifier를 호출합니다. 예를 들어 첫 번째 변환은 다음과 동일합니다.

~~~bash
python -m onnxsim models/det.onnx models/det_fixed.onnx           --overwrite-input-shape x:1,3,640,640
python -m onnxsim models/rec.onnx models/rec_fixed_ratio_2_5.onnx --overwrite-input-shape x:1,3,48,120
python -m onnxsim models/rec.onnx models/rec_fixed_ratio_5.onnx   --overwrite-input-shape x:1,3,48,240
...
~~~

비어 있지 않은 기존 고정 모델은 확인 후 재사용합니다.

<!-- cell: t10-v6-018 src: 5fa1f3bd19 -->
### 5.3 형상과 수치 동등성 검증

구조 검증만으로는 충분하지 않습니다. 다음 셀들은 다음을 수행합니다.

1. 모든 고정 ONNX 모델을 검사하고,
2. 정확한 입력 형상을 확인하고,
3. 검출과 대표적인 인식 버킷 두 개에 대해 ONNX Runtime 출력을 비교합니다.

모든 인식 버킷은 같은 원본 그래프에서 나옵니다. 튜토리얼 검증을 빠르게 유지하기 위해 수치 인식 검사에는 ratio 2.5와 ratio 5 모델을 사용합니다.

<!-- cell: t10-v6-021 src: 803f189d08 -->
## 6. DX NPU용으로 ONNX를 *.dxnn으로 컴파일

### 6.1 DX-COM 캘리브레이션 설정 만들기

캘리브레이션 전처리는 애플리케이션 전처리와 일치해야 합니다. 색상 순서, 스케일링, 정규화, 레이아웃이 다르면 컴파일이 성공하더라도 정확도가 떨어질 수 있습니다.

| 모델 | 캘리브레이션 이미지 | 리사이즈 | 평균 / 표준편차 |
|---|---|---:|---|
| 검출 | <code>det_dataset</code> | 640 × 640 | ImageNet 값 |
| 인식 | 일치하는 비율 버킷 | 고정 너비 × 48 | [0.5, 0.5, 0.5] |

<!-- cell: t10-v6-024 src: b5cb9cbaae -->
컴파일하기 전에 설정을 하나 이상 확인하세요. 특히 NCHW 입력 형상, 캘리브레이션 데이터셋, 리사이즈 크기, 채널 순서, 정규화, 전치 순서를 확인합니다.

<!-- cell: t10-v6-026 src: 7df5a7156c -->
### 6.2 ONNX 모델을 DXNN으로 컴파일

각 모델은 자체 디렉터리에 기록됩니다. 유효한 기존 DXNN 파일은 건너뛰므로 수업을 반복해도 다시 컴파일하는 데 시간을 쓰지 않습니다.

셀은 모델 이름을 출력한 뒤 각 모델에 대해 정확히 다음 명령을 실행합니다.

~~~bash
source <DX_ALL_SUITE_DIR>/dx-compiler/venv-dx-compiler-local/bin/activate
dxcom -m <fixed-model.onnx> \
      -c <calibration-config.json> \
      -o <output-directory> \
      --gen_log \
      --export_html
~~~

컴파일러 출력은 계속 표시되며, 컴파일이 실패하면 조용히 계속 진행하는 대신 오류와 함께 셀이 멈춥니다.

<!-- cell: t10-v6-029 src: 9a976ceadb -->
### 6.3 컴파일된 산출물 검증

완전한 OCR 애플리케이션에는 정확히 여섯 개의 DXNN 파일이 필요합니다. 검출 하나와 인식 버킷 다섯 개입니다. 필요한 파일이 하나라도 없으면 셀은 일찍 실패합니다.

<!-- cell: t10-v6-031 src: f143ae02c1 -->
<code>dxparse</code>로 두 가지 모델 역할을 검사합니다. 인식 ratio 2.5가 대표이며, 나머지 인식 파일은 주로 고정 입력 너비만 다릅니다.

<!-- cell: t10-dxrun-baseline-md src: 97352c10c3 -->
같은 두 모델에 대해 CLI 기준값을 측정합니다. 두 그래프 모두 CPU 태스크로 끝나므로 `--use-ort`가 필요하고, `-v`는 지연 시간 분석을 추가합니다. 이 수치를 나중에 애플리케이션과 비교하세요. 카메라 루프는 검출된 단어마다 DET 요청 하나와 REC 요청 하나를 실행하므로 프레임 시간은 텍스트 양에 따라 증가합니다.

```bash
dxrun -m <workspace>/outputs/paddleocr_v6/det_fixed/det_fixed.dxnn --use-ort -t 3 -v
dxrun -m <workspace>/outputs/paddleocr_v6/rec_fixed_ratio_2_5/rec_fixed_ratio_2_5.dxnn --use-ort -t 3 -v
```

<!-- cell: t10-v6-033 src: 877fbbedf2 -->
## 7. DEEPX-SDK로 OCR 애플리케이션 테스트

재사용 가능한 Python 애플리케이션은 <code>app/</code>에 있습니다. <code>outputs/paddleocr_v6/</code>의 컴파일된 PP-OCRv6 tiny 모델을 사용하며 각 프레임을 다음과 같이 처리합니다.

<img src="assets/ppocrv6-camera-pipeline.png" alt="PP-OCRv6 카메라 추론 파이프라인" style="max-width: 1000px; width: 100%; height: auto;">

검출기와 다섯 개의 인식 엔진은 한 번만 로드되어 계속 상주합니다.

<!-- cell: t10-v6-034 src: d8ac433600 -->
### 7.1 애플리케이션 의존성 설치

애플리케이션에는 <code>dx_engine</code>, OpenCV, NumPy, Pillow가 필요합니다. 튜토리얼 06-2와 마찬가지로 공유 Jupyter 환경 대신 <code>workspace/.venv-ocr</code> 아래의 작은 환경에 설치합니다. DX-RT 패키지는 <code>/usr/share/libdxrt-bin/python</code> 아래에 사전 빌드된 <code>dx_engine</code> 휠을 제공하며, 셀은 <code>cpXY</code> 태그가 커널의 Python과 일치하는 휠을 선택합니다. <code>app/run_camera.sh</code>도 같은 환경을 사용합니다. 셀을 다시 실행해도 안전하며, 이미 있는 것은 건너뜁니다.

```bash
uv venv --python <jupyter-python> <T10>/workspace/.venv-ocr
uv pip install --python <T10>/workspace/.venv-ocr/bin/python -r app/requirements.txt \
    /usr/share/libdxrt-bin/python/dx_engine-<version>-cp<XY>-*.whl
```

<!-- cell: t10-v6-036 src: 507594f64b -->
### 7.2 애플리케이션 구조와 안전장치

노트북은 이제 별도의 실행 스크립트를 생성하는 대신 <code>app/</code> 아래에서 유지보수되는 파일을 사용합니다.

| 파일 | 역할 |
|---|---|
| <code>app/ocr_engine.py</code> | DXNN 로딩, DET 후처리, 크롭 추출, 다섯 버킷 REC 라우팅, CTC 디코딩 |
| <code>app/camera_app.py</code> | CLI 검증, 640×480 카메라 캡처, OpenCV 미리보기, BBOX 및 다국어 텍스트 오버레이 |
| <code>app/run_camera.sh</code> | <code>workspace/.venv-ocr</code>로 애플리케이션 시작(<code>PYTHON_BIN</code>으로 재정의 가능) |
| <code>app/assets/ppocrv6_tiny_dict.txt</code> | 6906 클래스 REC 출력과 일치하는 공식 tiny 사전 |

애플리케이션은 모델 누락, 잘못된 텐서 형상, 사전 불일치, 카메라 열기/읽기 실패 시 명확하게 실패합니다. 기존 DXNN 파일은 절대 수정하지 않습니다.

<!-- cell: t10-v6-037 src: 90057954f1 -->
### 7.3 카메라 없이 애플리케이션 스모크 테스트 실행

<code>--check</code> 모드는 카메라를 열거나 GUI 창을 만들지 않습니다. 패키징된 애플리케이션을 로드하고 검출기와 다섯 개의 인식기 전체에 합성 추론을 한 번 실행합니다. 이를 통해 라이브 데모 전에 NPU 런타임, 입출력 텐서 계약, tiny 사전의 클래스 수를 검증합니다.

아래에서 실행하는 명령은 명시적인 모델 및 사전 경로와 함께 <code>workspace/.venv-ocr/bin/python app/camera_app.py --check</code>를 실행하는 것과 동일합니다.

<!-- cell: t10-v6-039 src: f2bee9052c -->
### 7.4 640×480 카메라 애플리케이션 테스트

라이브 애플리케이션은 대화형 OpenCV 창을 엽니다. 노트북 커널이 막히지 않고 카메라를 깔끔하게 해제할 수 있도록 **JupyterLab 터미널**에서 실행하세요. `run_camera.sh`는 7.1에서 만든 `workspace/.venv-ocr` 환경으로 애플리케이션을 시작합니다.

기본 입력은 640×480, 15 FPS의 <code>/dev/video0</code>입니다. 미리보기 창에서 **q** 또는 **Esc**를 누르면 종료됩니다. 다른 장치를 선택하려면 <code>--camera /dev/videoN</code>을 사용하세요.

<!-- cell: t10-v6-041 src: 4c25749e1a -->
<img src="assets/sc-ocr-app.png" style="max-width: 800px;" alt="예상되는 PP-OCR 결과">

<!-- cell: t10-v6-042 src: 4c5e64d2e3 -->
## 8. 정확도 및 성능 체크리스트

컴파일 성공이 허용 가능한 OCR 정확도를 보장하지는 않습니다. 대표성 있는 이미지로 전체 파이프라인을 검증하세요.

| 확인 항목 | 중요한 이유 | 실제 조치 |
|---|---|---|
| 검출 해상도 | 작은 텍스트는 640 × 640에서 사라질 수 있음 | 예상 카메라 거리에서 재현율 비교 |
| 캘리브레이션 범위 | 양자화는 캘리브레이션 통계를 따름 | 실제 조명, 글꼴, 흐림, 배경 포함 |
| 전처리 일치 | 색상 또는 정규화 불일치는 모델 입력을 왜곡함 | 컴파일과 런타임 전처리를 동일하게 유지 |
| 인식 버킷 | 과도한 리사이즈나 패딩은 문자를 손상시킴 | 텍스트 종횡비 분포 측정 |
| 인식 신뢰도 | 임계값이 낮으면 잘못된 텍스트가 표시되고, 높으면 텍스트가 누락됨 | 레이블이 있는 검증 세트로 조정 |
| 원근과 방향 | 카메라로 찍은 문서는 항상 평평하거나 똑바르지 않음 | 필요 시 문서 방향 보정과 왜곡 보정 추가 |
| 종단 간 지연 시간 | 한 프레임에 많은 인식 크롭이 있을 수 있음 | FPS만이 아니라 텍스트 수 기준으로 지연 시간 보고 |

보기 좋은 카메라 샘플 하나에 맞춰 조정하지 마세요. 고정된 검증 세트를 유지하고, 설정을 변경할 때마다 검출 재현율, 인식 정확도, 종단 간 지연 시간을 기록하세요.

<!-- cell: t10-v6-043 src: bf21a3d708 -->
## 9. 문제 해결

| 증상 | 표시되는 내용 | 원인 | 해결 |
|---|---|---|---|
| `dxcom`이 없음 | 상태 표에 `[MISSING] dx_com` | DX-COM이 설치되지 않았거나 `config.json`이 다른 곳을 가리킴 | 튜토리얼 01의 2절을 완료하고 `python tutorial_paths.py --show` 확인 |
| 다운로드가 거부됨 | `Checksum mismatch` 또는 `Incomplete download` | 전송 중단 또는 업스트림 파일 변경 | `workspace/models`에서 해당 모델만 삭제하고 4.1절을 다시 실행 |
| 고정 모델의 형상이 잘못됨 | `AssertionError: ... expected [1, 3, 48, 240]` | 오래된 고정 ONNX 파일 | 해당 고정 ONNX 파일을 삭제하고 5절을 다시 실행 |
| 컴파일 실패 | `Error Type: DataNotFoundError` 또는 `dxcom`의 다른 `[ERROR]` 줄 | 잘못된 `dataset_path`, 또는 나열된 확장자의 이미지가 없음 | `configs_v6/<model>.json`을 6.1에서 출력된 디렉터리와 비교. 전체 로그는 `outputs/paddleocr_v6/<model>/compiler.log`에 있음 |
| 7.1에서 `dx_engine` import 실패 | `ModuleNotFoundError: No module named 'dx_engine'` 또는 휠/ABI 불일치 | 환경이 다른 Python으로 만들어졌거나 DX-RT가 업그레이드됨 | `workspace/.venv-ocr`을 삭제하고 7.1을 다시 실행. 휠은 설치된 `libdxrt-bin`과 일치해야 함 |
| `--check`가 모델 누락을 보고 | `Application test files are missing` | 여섯 개의 DXNN 파일이 모두 컴파일되지 않음 | `SELECTED_COMPILES`에 모든 모델을 넣고 6.2를 다시 실행 |
| 카메라를 열 수 없음 | `Could not open camera /dev/video0` | 카메라 없음, 다른 장치 경로, 또는 사용자가 `video` 그룹에 없음 | `v4l2-ctl --list-devices` 실행, `--camera /dev/videoN` 전달, `sudo usermod -aG video $USER` 후 다시 로그인 |
| OpenCV 디스플레이 오류 | `cannot open display` | 그래픽 세션 없음 | 데스크톱 세션(유효한 `DISPLAY`가 있는 JupyterLab 터미널)에서 실행 |
| 텍스트가 표시되지 않음 | 텍스트 없는 상자, 또는 상자 없음 | 조명, 초점, 또는 임계값 | 대표성 있는 이미지로 `--det-threshold`, `--box-threshold`, `--rec-threshold` 조정 |

<!-- cell: t10-v6-044 src: 7776b639fb -->
## 10. 요약

### 10.1 완료한 워크플로

```text
Official PP-OCRv6 tiny DET + REC ONNX
                    │
                    ▼
       Fix dynamic input dimensions
          ┌─────────┴─────────┐
          │                   │
   DET 640×640       REC width buckets ×5
          └─────────┬─────────┘
                    ▼
     Calibration configuration + DX-COM
                    │
                    ▼
           Six verified DXNN models
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 camera_app.py --check    640×480 camera
                              │
                              ▼
                   BBOX + recognized text
```

### 10.2 산출물 및 검증 대시보드

| 단계 | 산출물 또는 명령 | 검증 내용 |
|---|---|---|
| 원본 모델 | <code>det.onnx</code>, <code>rec.onnx</code> | 공식 리비전과 SHA-256 체크섬 |
| 고정 ONNX | DET 하나와 REC 모델 다섯 개 | 정적 입력 형상과 ONNX 유효성 |
| 수치 검사 | ONNX Runtime 비교 | 고정 형상 변환이 대표 출력을 보존함 |
| 캘리브레이션 | <code>configs_v6/*.json</code> | 데이터셋, 리사이즈, 색상 순서, 정규화, 레이아웃 |
| 컴파일 | <code>dxcom</code> | 로그와 리포트가 표시되는 ONNX-DXNN 변환 |
| 런타임 계약 | <code>camera_app.py --check</code> | DET와 다섯 개의 REC 모델 전체가 예상된 텐서 계약으로 실행됨 |
| 라이브 애플리케이션 | <code>app/run_camera.sh</code> | 640×480 캡처, 비율 라우팅, CTC 디코딩, BBOX, 텍스트 오버레이 |

### 10.3 런타임 모델 맵

아래 형상은 DX-RT가 런타임에 보고하는 값(`dxparse`, `get_input_tensors_info()`)으로, 배치, 높이, 너비, 채널 순서(NHWC, UINT8)입니다. 5절에서는 ONNX 입력을 NCHW float 순서로 고정했습니다. DX-COM이 레이아웃 변경과 정규화를 컴파일된 모델 안에 포함시키므로 애플리케이션은 원시 HWC 픽셀을 그대로 입력합니다.

| 역할 | 고정 입력 | 선택 규칙 |
|---|---:|---|
| DET | `[1, 640, 640, 3]` | 카메라 프레임당 한 번 |
| REC 2.5 | `[1, 48, 120, 3]` | 크롭 종횡비 ≤ 2.5 |
| REC 5 | `[1, 48, 240, 3]` | 크롭 종횡비 ≤ 5 |
| REC 10 | `[1, 48, 480, 3]` | 크롭 종횡비 ≤ 10 |
| REC 15 | `[1, 48, 720, 3]` | 크롭 종횡비 ≤ 15 |
| REC 25 | `[1, 48, 1200, 3]` | 크롭 종횡비 ≤ 25, 또는 더 넓은 크롭의 대체 모델 |

### 10.4 완료 체크리스트

- [ ] 공식 PP-OCRv6 tiny DET 및 REC ONNX 모델 다운로드
- [ ] 원본 모델 체크섬과 ONNX 구조 검증
- [ ] 고정 DET 모델 하나와 고정 REC 모델 다섯 개 생성
- [ ] 대표적인 원본 및 고정 ONNX 출력 비교
- [ ] 캘리브레이션 설정 생성 및 DXNN 모델 컴파일
- [ ] 컴파일된 산출물 검사 및 NPU에서 여섯 모델 모두 테스트
- [ ] DET-REC OpenCV 카메라 애플리케이션 준비
- [ ] 레이블이 있는 검증 세트로 검출 및 인식 정확도 측정
- [ ] 대표성 있는 카메라 이미지로 임계값 조정
- [ ] 종단 간 지연 시간, 처리량, 호스트 CPU 사용량 기록

### 10.5 small 또는 medium 티어 시도하기

더 큰 PP-OCRv6 티어로 이 워크플로를 반복하려면 해당 공식 PaddlePaddle ONNX 저장소에서 검출기와 인식기를 모두 다운로드하세요.

| 티어 | 검출 ONNX 저장소 | 인식 ONNX 저장소 |
|---|---|---|
| **small** | [PP-OCRv6_small_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx) | [PP-OCRv6_small_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx) |
| **medium** | [PP-OCRv6_medium_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx) | [PP-OCRv6_medium_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx) |

<!-- cell: 87889eeb-8768-43a7-97e9-5c0805d6cd64 src: 78b7602a47 -->
### 10.6 OCR 성능을 높이려면?

이 튜토리얼은 `Document Image Orientation Classification`, `Text Image Unwarping`, `Text Line Orientation Classification` 같은 전처리 블록을 구현하지 않습니다.

더 높은 OCR 성능이 필요하다면 누락된 AI 모델을 구현해 OCR AI 파이프라인에 적용하세요.

<img src="assets/ppocrv6-full-pipeline.png" style="max-width: 1200px;">

더 높은 정확도를 위해:

- **실제 배포 환경**의 텍스트 이미지로 캘리브레이션 데이터셋을 구성하세요.
- 실제 카메라, 거리, 텍스트 크기, 종횡비, 글꼴, 언어, 조명, 흐림, 원근을 모두 포함하세요.
- 현재 **캘리브레이션 데이터셋**에는 ratio 5, 15, 25의 넓은 REC 버킷 세 개가 있습니다. 배포 환경의 텍스트 형태가 크게 다양하다면 이 데이터셋 버킷을 더 세분화하세요.
- 이 캘리브레이션 버킷은 다섯 개의 런타임 REC 모델 경로(ratio 2.5, 5, 10, 15, 25)와는 다릅니다.
- 동적 DET 및 REC 입력을 측정된 배포 데이터에서 선택한 고정 형상으로 변환하세요.
- 캘리브레이션과 런타임 전처리를 동일하게 유지하세요.

| 배포 데이터 | 조정 |
|---|---|
| 현재 버킷 사이의 비율 | 중간 REC 버킷 추가 |
| 높거나 좁거나 긴 텍스트 | 일치하는 고정 REC 형상 사용 |
| 매우 작은 텍스트 | 더 큰 DET 입력 평가 |
| 특이한 카메라 종횡비 | DET 입력 형상을 맞춤 |
| 리사이즈 후 왜곡 | 형상과 전처리 조정 |

릴리스 전에:

- 같은 레이블 데이터셋으로 ONNX와 DXNN 정확도를 비교하세요.
- 검출 재현율, 인식 정확도, 지연 시간, 메모리, CPU 부하를 측정하세요.
- 텐서 계약, 전처리, 캘리브레이션, 인식 사전을 다시 확인하세요.
- 측정 가능한 가치가 있을 때만 형상을 추가하세요. 형상이 많아지면 빌드 시간, 저장 공간, 라우팅 복잡도가 증가합니다.

> **다음:** 튜토리얼 20으로 이동해 같은 런타임에서 C++로 멀티 채널 YOLO 애플리케이션을 빌드하고 실행해 보세요.
