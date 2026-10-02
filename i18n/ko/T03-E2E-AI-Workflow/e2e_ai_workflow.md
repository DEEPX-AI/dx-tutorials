<!-- i18n source: notebooks/T03-E2E-AI-Workflow/e2e_ai_workflow.ipynb -->
<!-- i18n lang: ko -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 2dbf2d92-815e-4a2e-b468-26f71cd8dbe3 src: e823f81eaf -->
# DEEPX Tutorial 03 - DEEPX NPU를 활용한 AI 프로젝트 워크플로

<!-- cell: 74c3b426-7d6e-4844-bf75-737a94d1055f src: 7da7d09c6b -->
이 세 번째 튜토리얼은 DEEPX 하드웨어에 AI 모델을 배포하는 전체 엔드투엔드 워크플로를 보여 줍니다.

지게차와 작업자 검출 모델을 학습하고, DX-Compiler 도구로 DXNN 형식으로 변환한 뒤, 최종 AI 애플리케이션을 DEEPX NPU에서 실행합니다. 이 과정을 통해 DEEPX NPU 개발 파이프라인의 전체 그림을 파악할 수 있습니다. 

<!-- cell: 38bde5ed-95d6-4750-a6fb-174273a70167 src: 7bfac3c725 -->
## 학습 목표

이 튜토리얼을 마치면 다음을 할 수 있습니다.

- 모델 선택부터 NPU 배포까지 완전한 AI 프로젝트를 따라 진행하기
- 직접 학습한 ONNX 모델을 `dxcom`과 Model Zoo JSON 설정으로 컴파일하기
- 관련된 두 JSON 파일(컴파일러 설정과 DX-APP 런타임 설정)을 설명하기
- 표준 DX-APP 실행 파일로 커스텀 2-클래스 검출기를 이미지와 영상에서 실행하기

<!-- cell: 445ad29e-2955-4f2e-beac-2e2a34bbe81f src: 6275be0099 -->
## 실습 프로젝트 개요

<!-- cell: e37c1d96-a612-4246-af9c-ebf798355fba src: 08e68a9f0e -->
- **검출 클래스**: Forklift, Worker
- **기반 AI 모델**: YOLOv7
- **데이터셋**: [Kaggle](https://www.kaggle.com/datasets/hakantaskiner/personforklift-dataset/data)의 Forklift & Worker 이미지 1448장
- **학습**: GPU 메모리 24 GB 이상의 NVIDIA GPU (학습에는 고성능 GPU가 필요하지만 배포에는 DEEPX NPU만 있으면 됩니다)
- **추론 NPU**: `DX-M1`
- **AI 애플리케이션**: DX-APP의 yolo 데모를 수정해 재사용
- **예상 결과**:
<img src="assets/detection-goal.jpg" style="max-width: 1200px;">

<!-- cell: e8217ad7-fc19-421b-967d-98ff2dfbc762 src: 73e0f10840 -->
## AI 워크플로 개요

<!-- cell: 955d81f9-0699-45e6-ba37-e27a481ad370 src: 2d9b410433 -->
이 다이어그램은 AI 프로젝트의 일반적인 워크플로를 설명합니다.

목표를 정의하고, 데이터를 수집·라벨링하고, 모델을 학습합니다.
DX-Compiler는 DX NPU를 위해 모델을 더 빠르고 가볍게(INT8) 만드는 데 도움을 줍니다.
마지막 단계는 DX-APP 또는 DX-STREAM을 사용해 모델을 DEEPX NPU에 배포하는 것입니다.

각 단계는 작업자 및 지게차 검출과 같은 실제 AI 솔루션을 향해 쌓여 갑니다.

  <img src="assets/workflow2.jpg" style="max-width: 1200px;">

<!-- cell: 1b3d8491-f89b-4fbf-b601-2cfc061e96a5 src: 22621a4e9d -->
## 사전 요구 사항

- 튜토리얼 01 완료: DX-COM, DX-RT, DX-APP이 설치되어 있고 NPU가 인식됨.
- 4절(컴파일)에는 DX-COM과 캘리브레이션 데이터셋이 필요합니다. 또는 옵션 B로 사전 컴파일된 모델을 다운로드할 수 있습니다.
- 5절에는 DEEPX NPU와 영상 결과를 볼 디스플레이가 필요합니다.
- 학습(3절)은 선택 사항이며 24 GB 메모리의 NVIDIA GPU가 필요합니다. Colab 노트북 링크는 해당 절에 있습니다.
- 다운로드: `cs.deepx.ai`에서 ONNX 모델(139 MB), 사전 컴파일된 DXNN(71 MB), 테스트 영상(19 MB). 선택 사항인 YOLOv7 컴파일은 10분 이상 걸립니다.
- 이 튜토리얼이 다운로드하거나 생성하는 모든 파일은 git이 무시하는 `notebooks/T03-E2E-AI-Workflow/workspace/`에 저장되므로 SDK 체크아웃은 깨끗하게 유지됩니다.

다음 셀은 SDK를 찾고, 이 요구 사항을 확인하고, 상태 표를 출력합니다. `MISSING`으로 표시된 항목에는 그것을 제공하는 단계가 함께 표시됩니다.

<!-- cell: 691e8f9d-0115-48d7-a02a-1b64c84c3b72 src: 92bb42d767 -->
## 1. AI 워크플로 - 사용 사례에 따른 모델 선택

<!-- cell: afa66140-dbee-4bf2-a9ce-82f4316a8512 src: 9e254de77f -->
AI 프로젝트를 시작하려면 사용 사례에 맞는 AI 모델을 선택해야 합니다.

이 튜토리얼의 목표는 지게차와 작업자를 검출하는 것입니다.
객체 검출 분야에서 잘 알려진 모델인 YOLOv7을 사용합니다.

- Forklift & Worker 검출을 위해 YOLOv7 선택
- YOLOv7 상세 정보: [링크](https://docs.ultralytics.com/models/yolov7/)
- YOLOv7 사용 방법: [링크](https://github.com/WongKinYiu/yolov7)

<!-- cell: 0a81b7c5-ec05-412b-bc79-338f77d02b18 src: dbe5a538f7 -->
## 2. AI 워크플로 - 데이터 준비 및 어노테이션

<!-- cell: ac53d3cc-8142-4d59-ac44-fb7f2f836e74 src: 0be5dc9c10 -->
Kaggle에서 지게차-사람 라벨링 데이터셋을 다운로드합니다.
 - 참고: [Kaggle 링크](https://www.kaggle.com/datasets/hakantaskiner/personforklift-dataset)

<!-- cell: e071e952-82bb-4cf9-8c88-6a007b150d3b src: 0aadb5e072 -->
## 3. AI 워크플로 - 학습

<!-- cell: cbd31d3c-e849-49d3-b7f4-10efefca0861 src: 88b6d45076 -->
모델을 효율적으로 학습하려면 그래픽 메모리가 24GB 이상인 GPU를 사용해야 합니다.

 - YOLOv7 학습 방법: [링크](https://colab.research.google.com/drive/1dAdjJuhXqFM_Qcd0QqAn7_AGx7abA5aX?usp=sharing)

학습은 여기가 아니라 해당 Colab 노트북에서 진행합니다. 그 결과물인 내보낸 ONNX 파일 `yolov7-forklift-person.onnx`는 4절에서 다운로드할 수 있도록 제공되므로 GPU 없이도 계속 진행할 수 있습니다. 학습에서 이어지는 세부 사항이 하나 있습니다. 데이터셋 라벨의 클래스 순서는 `0 = Forklift`, `1 = Worker`이며, 5.1절에서는 `class_names`를 정확히 이 순서로 나열해야 합니다.

<!-- cell: f7dd943a-796f-4aec-8341-a524c71c4e2f src: 9802354bde -->
## 4. AI 워크플로 - DX-Compiler를 이용한 최적화

<!-- cell: 6347d2b0-3fd8-4a96-bc0e-84c5885586eb src: 70970cc292 -->
사전 학습된 AI 모델을 DXNN 형식으로 컴파일해 보겠습니다.

전체 과정은 다음과 같습니다.
1. pytorch 프레임워크의 사전 학습된 모델 준비
2. ONNX 형식으로 변환
3. ONNX를 DXNN으로 컴파일 (DX-Compiler의 자세한 내용은 [여기](https://developer.deepx.ai/download/?id=581)의 사용자 가이드를 참고하세요
                                                                                             
> **참고:** 사용자 가이드를 다운로드하려면 먼저 https://developer.deepx.ai/ 에 로그인해야 합니다.

<img src="assets/dx-com-workflow.jpg" style="max-width: 1200px;">

<!-- cell: bdca7205-5cef-453c-863a-794925e61820 src: 0df77eb438 -->
DX-Compiler의 소스 구조는 다음과 같이 구성되어 있습니다.
```bash
dx_com
 ├── calibration_dataset   # Dataset used to optimize model accuracy
 └── sample_models         # Sample configuration file and ONNX files 
```

<!-- cell: 6655c5ab-7718-4a75-a305-54516a723d1d src: eb8bb286ef -->
### 4.1 내보낸 ONNX 모델과 YOLOv7 DX-Compiler 설정 준비

<!-- cell: a2886542-8a78-4e17-801b-121deb3b41e0 src: 9e9aa01488 -->
커스텀 YOLOv7 ONNX 모델은 직접 다운로드합니다. 이 튜토리얼에는 YOLOv7 Q-Lite JSON 설정의 사본도 포함되어 있어, 브라우저 다운로드 위치를 가정하지 않고도 노트북을 처음부터 끝까지 실행할 수 있습니다.

원본 설정은 [DEEPX Model Zoo](https://developer.deepx.ai/modelzoo/)에서 가져왔습니다: Object Detection, YOLOv7, Q-Lite JSON 다운로드.

<img src="assets/sc-modelzoo-yolov7.png" style="max-width: 1000px;">

  
  **참고:** DEEPX Model Zoo는 DEEPX가 검증한 AI 모델과 컴파일러 설정을 제공합니다.

<!-- cell: acb10634-544d-4ace-ba8a-0fd870fbf87e src: e1d203991b -->
### 4.2 사용자 환경에 맞게 YOLOv7 json 파일 수정

<!-- cell: 97b44e9e-20ad-4a1a-be68-321b3e981ec2 src: 618757da53 -->
### 4.3 json 설정을 참조해 ONNX를 DXNN으로 컴파일

<!-- cell: t03-dxcom-environment-guide src: 8e71d364a7 -->
DX-Compiler는 SDK 설치 중에 생성된 전용 Python 가상 환경을 사용합니다. JupyterLab은 별도의 `dx-tutorials` 환경에서 실행되므로, 일반적으로 노트북의 `PATH`에서는 `dxcom`을 사용할 수 없습니다.

다음 코드 셀은 `config.json`에서 DX-Compiler 경로를 구하고, DX-Compiler 환경을 활성화하고, `dx_com` 워크스페이스로 이동한 뒤, 같은 셸에서 `dxcom`을 실행합니다. 각 `!` 코드 셀은 새 셸을 시작하므로 활성화는 셀 간에 유지되지 않습니다. 그래서 `dxcom`을 실행하는 모든 노트북 셀은 활성화 명령을 반복합니다.

이 튜토리얼 밖에서는 터미널을 열고 환경을 한 번만 활성화하면 됩니다.

```bash
cd <DX_ALL_SUITE_DIR>/dx-compiler
source venv-dx-compiler-local/bin/activate
cd dx_com
dxcom -h
```

활성화된 환경은 `deactivate`를 실행하거나 터미널을 닫을 때까지 해당 터미널에서 계속 유지됩니다.

<!-- cell: 87678442-bbd9-47c0-a4cd-631987888e24 src: aed9a3d45a -->
DXNN을 얻는 두 가지 방법 중 하나를 선택하세요. 다음 셀에서 `COMPILE_OPTION`을 설정합니다. `"A"`는 ONNX를 직접 컴파일하고(10분 이상 소요), `"B"`는 DEEPX가 같은 명령으로 컴파일한 DXNN을 다운로드합니다(71 MB). 워크스페이스에 DXNN이 이미 있으면 두 셀 모두 자동으로 건너뜁니다.

<!-- cell: 39c4982e-a87b-48c0-8a3f-1ed63295c4cd src: cfeb99e9ac -->
#### 4.3.1 옵션 A: `dxcom`으로 ONNX 컴파일

튜토리얼 01의 2.2.2절과 같은 명령이며, 커스텀 ONNX와 수정한 JSON을 사용합니다. `-o`가 워크스페이스를 가리키므로 SDK 트리에는 아무것도 기록되지 않습니다.

```bash
cd <DX_ALL_SUITE_DIR>/dx-compiler
source venv-dx-compiler-local/bin/activate
cd dx_com
dxcom -m <workspace>/yolov7-forklift-person.onnx -c <workspace>/yolov7-forklift-person.json -o <workspace>/output --gen_log
```

> **참고:** 컴파일은 호스트에 따라 **10분 이상** 걸립니다. 이 셀은 `COMPILE_OPTION`이 `"A"`일 때만 실행됩니다.

<!-- cell: ddb0f169-582f-4a96-9c49-d8325df7e616 src: 6d1d5efeaf -->
#### 4.3.2 옵션 B: 사전 컴파일된 DXNN 다운로드

DEEPX가 같은 ONNX를 같은 JSON으로 컴파일해 결과를 게시해 두었습니다. 다운로드하면 몇 초 만에 동일한 파일을 얻을 수 있습니다.

```bash
cd <workspace>/output
wget -nc https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/yolov7-forklift-person.dxnn
```

<!-- cell: 6daf1b20-9182-4a77-82d0-8188d9c6e257 src: c0768f5b3c -->
## 5. AI 워크플로 - DEEPX NPU에 배포

<!-- cell: 4d54f93a-31ff-4f9c-8fd5-1de4776cc622 src: f2166618d1 -->
### 5.1 커스텀 YOLOv7 모델을 위한 DX-APP 설정

현재 DX-APP은 커스텀 클래스 수를 위해 C++ 소스를 패치할 필요가 없습니다. YOLOv7 팩토리는 런타임 설정 파일에서 `num_classes`와 `class_names`를 읽고, 후처리기는 검출 결과를 디코딩할 때 이 값을 사용합니다.

> **참고 자료:** 이 절은 [DX-APP YOLO Customizing Guide](https://github.com/DEEPX-AI/dx_app/blob/main/docs/source/docs/12_DX-APP_YOLO_Customizing_Guide.md#step-3-tune-configjson)(체크아웃의 `dx_app/docs/source/docs/12_DX-APP_YOLO_Customizing_Guide.md`)의 **Step 3, "Tune `config.json`"**을 따릅니다. 이 가이드는 YOLO 후처리기가 읽는 모든 키(`obj_threshold`, `score_threshold`, `nms_threshold`, `num_classes`, `class_names`, `anchors`, `strides`)를 나열하고, 3-클래스 모델에 대한 같은 파일을 보여 줍니다. 번들된 YOLOv7 예제는 기본값을 `src/cpp_example/object_detection/yolov7/config.json`에 담아 제공하며, 아래에서 작성하는 파일은 2-클래스 Forklift 및 Worker 모델을 위해 그 기본값을 재정의합니다.

이 튜토리얼에서 사용하는 두 JSON 설정 파일을 혼동하지 마세요.

| 설정 | 사용 주체 | 용도 |
|---|---|---|
| `yolov7-forklift-person.json` | `dxcom -c` | 모델 컴파일, 입력 형상, 전처리, 캘리브레이션 데이터를 기술합니다. |
| `yolov7-forklift-runtime.json` | DX-APP `--config` | 런타임 후처리 임계값, 클래스 수, 표시 라벨을 기술합니다. |

컴파일된 모델의 디코딩된 출력 형상은 `[1, 25200, 7]`입니다. 각 행에는 박스 값 4개, objectness 값 1개, 클래스 점수 2개가 들어 있습니다: `4 + 1 + 2 = 7`. 원시 검출 헤드는 각 헤드가 앵커 3개를 사용하므로 21개 채널을 가집니다: `3 × (5 + 2) = 21`.

<!-- cell: c8598424-1ee1-4413-9ecd-ab7f6fd6e2d6 src: 37fd0c267c -->
#### 런타임 설정 필드

| 필드 | 설명 |
|---|---|
| `obj_threshold` | objectness 점수가 이 값보다 낮은 후보를 제거합니다. |
| `score_threshold` | 최종 클래스 신뢰도가 이 값보다 낮은 검출을 제거합니다. YOLOv7의 최종 신뢰도는 objectness와 클래스 점수를 기반으로 합니다. |
| `nms_threshold` | Non-Maximum Suppression이 겹치는 검출을 제거할 때 사용하는 IoU 임계값입니다. 값이 낮을수록 겹치는 박스를 더 적극적으로 제거합니다. |
| `num_classes` | 모델이 생성하는 클래스 수입니다. 이 모델은 두 클래스를 생성하므로 값은 `2`여야 합니다. |
| `class_names` | 각 클래스 ID에 표시되는 라벨입니다. 배열 순서는 학습 시 사용한 클래스 순서와 정확히 일치해야 합니다. |

이 모델에서 클래스 ID `0`은 `Forklift`, 클래스 ID `1`은 `Worker`입니다. 라벨 텍스트만 바꾸는 것은 모델 동작을 바꾸지 않으며, 검출된 클래스 ID가 표시되는 방식만 바꿉니다. `num_classes`, `class_names`, 모델 출력이 서로 맞지 않으면 잘못된 디코딩이나 잘못된 라벨이 발생할 수 있습니다.

> **참고:** 가이드에서 경고하듯이 `num_classes`는 컴파일된 모델의 출력과 일치해야 합니다. 클래스 수는 컴파일 시점에 고정되므로, 런타임 JSON만 바꿔서는 텐서 형상 불일치를 해결할 수 없습니다.

<!-- cell: 12ba0bdd-1df3-47d2-a454-a1bc39c8768c src: d725d9b9f9 -->
### 5.2 YOLOv7 실행 파일 확인

런타임 설정은 DX-APP이 YOLOv7 후처리기를 생성하기 전에 `--config`로 로드됩니다. C++ 소스를 변경하지 않으므로 `bin/yolov7_async`가 이미 있으면 DX-APP을 다시 빌드할 필요가 없습니다.

모델이 실행되면 DX-APP은 `Config loaded: yolov7-forklift-runtime.json (4 keys)`를 로그로 출력합니다. 이 개수는 스칼라 값만 센 것이며, `class_names`는 리스트라서 별도로 저장되므로 다섯 가지 설정이 모두 적용됩니다.

실행 파일이 없으면 터미널(File > New > Terminal)에서 이 타깃만 빌드하세요.

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_app
./build.sh --target yolov7_async
```

<!-- cell: aef5fbb5-57c5-425b-bb18-be6c09f84008 src: 548c63ca47 -->
### 5.3 이미지로 커스텀 모델 실행

커스텀 DXNN과 그 런타임 설정을 표준 YOLOv7 실행 파일에 전달합니다. 공유 DX-APP 소스는 변경되지 않습니다. 이 셀은 디스플레이 없이도 동작하도록 헤드리스(`--no-display --save`)로 실행되며, 저장된 결과를 아래에 표시합니다. 대신 DX-APP 창을 열려면 `--no-display`를 제거하세요. 그러면 창이 닫힐 때까지 셀이 실행 중 상태로 유지됩니다.

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_app
./bin/yolov7_async -m <workspace>/output/yolov7-forklift-person.dxnn \
    --config <workspace>/yolov7-forklift-runtime.json \
    -i <tutorial>/assets/forklift-worker.png --no-display --save --save-dir <workspace>/outputs
```

<!-- cell: c6d86bb5-c77a-4144-8072-4c30ac0e0386 src: 2dbebc5c03 -->
### 5.4 영상으로 커스텀 모델 실행

영상 실행은 같은 명령에 `-v`를 사용합니다. 클립 길이는 80초(2001프레임)입니다. `SHOW_WINDOW = False`(기본값)이면 애플리케이션은 헤드리스로 실행되어 렌더링된 영상을 워크스페이스 아래에 기록하며, **끝날 때까지 아무것도 출력하지 않습니다**. DX-M1에서는 약 20초가 걸리므로 셀이 멈춘 것이 아닙니다. 그다음 저장된 영상이 노트북에 표시됩니다. 대신 DX-APP 창에서 실시간으로 보려면 `SHOW_WINDOW = True`로 설정하세요(창이 닫힐 때까지 셀이 실행 중 상태로 유지됩니다).

```bash
./bin/yolov7_async -m <workspace>/output/yolov7-forklift-person.dxnn \
    --config <workspace>/yolov7-forklift-runtime.json -v <workspace>/forklift-worker.mp4 \
    --no-display --save --save-dir <workspace>/outputs
```

<!-- cell: 0fefce5e-afd0-44ab-a03f-ed3a18e92272 src: 015011ee11 -->
## 6. 문제 해결

| 증상 | 표시되는 내용 | 원인 | 해결 |
|---|---|---|---|
| 다운로드 실패 | `wget: unable to resolve host` 또는 `404` | 인터넷 연결이 없거나 아카이브가 이동됨 | 연결을 확인하고 재시도. 모델은 4.3.2의 옵션 B 사용 |
| `dxcom`이 캘리브레이션 이미지를 찾지 못함 | `No images found` 또는 캘리브레이션 오류 | JSON의 `dataset_path`가 잘못됨 | 4.2의 셀을 다시 실행. `dx_com/calibration_dataset`의 절대 경로를 기록함 |
| 컴파일이 매우 오래 걸림 | 오류 없이 10분 이상 소요 | 640x640의 YOLOv7은 CPU 연산에 의존함 | 기다리거나 옵션 B(사전 컴파일된 DXNN) 사용 |
| `No DXNN at .../workspace/output/...` | 4.3.2 이후 `FileNotFoundError` | 두 옵션 모두 파일을 생성하지 않음 | `COMPILE_OPTION`을 `"A"` 또는 `"B"`로 설정하고 해당 셀을 다시 실행 |
| 5.3 또는 5.4에서 창이 뜨지 않음 | 셀이 성능 요약만 출력하고 끝남 | 셀이 기본적으로 헤드리스로 실행됨 | `workspace/outputs/` 아래의 저장된 결과를 열거나, 데스크톱 세션에서 `--no-display` 제거 / `SHOW_WINDOW = True` 설정 |
| 박스에 잘못된 라벨이 표시됨 | 지게차가 Worker로 표시되거나 그 반대 | `class_names` 순서가 학습 순서와 다름 | 런타임 JSON의 순서를 고치고 다시 실행. `num_classes`는 2로 유지 |

<!-- cell: 2df122e0-2ea3-4c2e-b294-6fee2d7c5b24 src: 4848d5487c -->
## 7. 요약

### 7.1 완료한 워크플로

```text
Choose a model (YOLOv7) and a labeled dataset
       │
       ▼
Train on a GPU (Colab notebook)
       │
       ▼
Export ONNX and compile to DXNN with dxcom
       │
       ▼
Describe classes and thresholds in a DX-APP runtime config
       │
       ▼
Run the custom model on the NPU with yolov7_async
```

### 7.2 완료 체크리스트

- [ ] 컴파일러용 ONNX 모델과 Model Zoo JSON 설정 준비
- [ ] `dataset_path`를 로컬 캘리브레이션 데이터셋으로 지정
- [ ] `dxcom`으로 모델을 컴파일하거나 사전 컴파일된 DXNN 다운로드
- [ ] `num_classes`와 `class_names`를 담은 DX-APP 런타임 설정 작성
- [ ] 이미지와 영상에서 커스텀 모델 실행

> **다음:** 튜토리얼 04로 이동해 같은 Forklift 및 Worker 검출기를 DX-STREAM으로 GStreamer 파이프라인에 구성해 보세요.
