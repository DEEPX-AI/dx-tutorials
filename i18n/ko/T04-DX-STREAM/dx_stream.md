<!-- i18n source: notebooks/T04-DX-STREAM/dx_stream.ipynb -->
<!-- i18n lang: ko -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 1726e9e4-5821-4f34-add4-37f1fb44ee77 src: c7877fefea -->
# DEEPX Tutorial 04 - DX-STREAM 워크플로

이 튜토리얼은 DX-STREAM v3.1.x를 소개하고 DEEPX NPU에서 엔드투엔드 비전 AI 파이프라인을 구축하는 방법을 보여 줍니다.

## 학습 목표

이 튜토리얼에서는 다음을 배웁니다.

- DEEPX SDK에서 DX-STREAM이 차지하는 위치 이해
- `DxPreprocess → DxInfer → DxPostprocess → DxTracker → DxOsd`로 파이프라인 구축
- DX 멀티스트림 도메인과 셀렉터 경계 학습
- 추론 백엔드, 스케일링, 변환, 메시지 브로커 요소 사용
- 사용자 정의 AI 모델에 맞게 사용자 정의 후처리 라이브러리 적용
- 메타데이터 기반 인원 수 오버레이 추가
- 메타데이터 검사와 GStreamer 파이프라인 진단

<!-- cell: 32d7336d-3f4d-4c4a-aa70-2662f4558edc src: 6645080682 -->
## 1. DX-STREAM 개요

DX-STREAM은 DEEPX NPU에서 비전 AI 파이프라인을 구성하기 위한 GStreamer 요소(element) 모음입니다.

```text
Source → Decode → DxPreprocess → DxInfer → DxPostprocess → DxTracker → DxOsd → Sink
```

![DX-STREAM 파이프라인](assets/dx-stream-pipeline.png)

| 요소 | 역할 |
|---|---|
| `dxpreprocess` | 프레임을 리사이즈·크롭하고 모델 입력 텐서로 변환 |
| `dxinfer` | 컴파일된 `.dxnn` 모델을 실행 |
| `dxpostprocess` | 출력 텐서를 디코딩하고 추론 메타데이터를 기록 |
| `dxtracker` | 검출된 객체에 지속적인 ID를 부여 |
| `dxosd` | 박스, 라벨, 포즈, 세그멘테이션 결과를 그림 |
| `dxgather` | 같은 소스에서 생성된 브랜치를 병합 |
| `dxinputselector` | 입력 스트림을 DX 멀티스트림 도메인으로 병합 |
| `dxoutputselector` | DX 멀티스트림 도메인을 스트림 ID별로 분리 |
| `dxrate` | 출력 프레임 레이트 제어 |
| `dxmsgconv` | 추론 메타데이터를 구조화된 메시지로 변환 |
| `dxmsgbroker` | MQTT 또는 Kafka로 메시지 발행 |
| `dxscale` | 프레임 해상도 변경 |
| `dxconvert` | 프레임 색상 형식 변경 |

전체 요소 및 속성 레퍼런스는 [DEEPX Developer Portal](https://developer.deepx.ai/download/?id=583)에서 제공하는 DX-STREAM User Manual을 참고하세요. 로그인이 필요합니다.

<!-- cell: 2d53a070-40a1-4658-950d-62c7c796af9b src: fe4feb49aa -->
## 2. 사전 요구 사항

- `./dx-runtime/install.sh --all`로 튜토리얼 01을 완료: DX-RT, DX-APP, DX-STREAM이 빌드되고 NPU가 인식됨. 튜토리얼 03을 권장하며, 5절은 해당 튜토리얼의 Forklift and Worker 모델이 있으면 재사용합니다.
- 창을 띄우기 위한 그래픽 데스크톱 세션. 설정 셀이 디스플레이를 감지합니다. 디스플레이가 있으면 파이프라인 셀이 DX-STREAM 창을 열고, 없으면(예: SSH 접속) `fakesink`로 **헤드리스** 실행되어 노트북이 끝까지 진행됩니다. `HEADLESS`로 어느 쪽이든 강제할 수 있습니다. 4절의 `run_demo.sh` 데모와 카메라 셀은 각각 몇 분씩 걸리므로 옵트인(`RUN_DEMOS`, `RUN_CAMERA`)입니다.
- 선택 하드웨어: 4.3용 V4L2 카메라(`RUN_CAMERA`), 4.9용 접근 가능한 RTSP 스트림, 4.11용 MQTT 또는 Kafka 브로커.
- 도구: `gstreamer1.0-tools`, `v4l-utils`, `jq`(`setup.sh`가 사용), 7.3용 `graphviz`. 5.7절은 `meson`으로 DX-STREAM을 다시 빌드하며 `sudo`가 필요합니다.
- DX-STREAM GStreamer 플러그인이 `gst-inspect-1.0`에서 보여야 합니다. `build.sh`가 `export GST_PLUGIN_PATH=...` 줄을 출력합니다. 설정 셀이 이를 확인하고, 플러그인이 기본 위치에 있으면 커널에 변수를 설정합니다.
- 다운로드: `setup.sh`가 샘플 모델 17개(약 300 MB)와 공유 샘플 영상 아카이브(약 1.1 GB, `<dx-all-suite>/workspace/res/videos`에 저장되며 DX-APP과 공유)를 받습니다. 5절은 튜토리얼 03의 모델과 영상을 재사용하거나 90 MB를 다운로드합니다.
- 이 튜토리얼이 생성하는 파일은 git이 무시하는 `notebooks/T04-DX-STREAM/workspace/`에 저장됩니다. SDK 트리 안에서 바뀌는 것은 5.6의 후처리 패치뿐이며, 그것이 해당 절의 목적입니다.

다음 셀은 SDK를 찾고 이 요구 사항을 확인한 뒤 상태 표를 출력합니다.

<!-- cell: 53136a15-c25c-4168-aa38-a07629a40996 src: 1605721eba -->
### 2.1 SDK 경로 불러오기

다음 셀은 `DX_ALL_SUITE_DIR`에서 `DX_STREAM_DIR`과 다른 SDK 경로를 도출한 뒤, `run_demo.sh`와 샘플 경로가 기대하는 DX-STREAM 저장소 루트로 노트북 작업 디렉터리를 변경합니다. 또한 아래에서 사용하는 디스플레이 플래그와 워크스페이스 경로를 정의합니다.

<!-- cell: 8d9cf255-57c7-4b99-be3b-741331f6e532 src: 75741dab3d -->
### 2.2 (선택) Intel GPU 사전 요구 사항

Intel x86_64 iGPU/dGPU 시스템에서만 [`docs/intel_gpu.md`](../../docs/intel_gpu.md)를 따르세요.

<!-- cell: e7c0a679-122a-4e4e-bae9-1c034689e281 src: 5ae0009348 -->
### 2.3 샘플 리소스 다운로드

`setup.sh`는 번들 파이프라인에 필요한 모델과 영상을 `dx_stream/samples/`에 다운로드합니다. 현재 목록은 모델 17개(약 300 MB)와 공유 샘플 영상 아카이브(약 1.1 GB. `setup.sh`가 `<dx-all-suite>/workspace/res/videos` 아래에 압축을 풀고 `dx_stream/samples/videos`로 링크합니다)입니다. 첫 실행에는 몇 분이 걸립니다. 샘플이 이미 있으면 셀은 스크립트를 건너뜁니다.

<!-- cell: fe12c34c-a475-437f-bba9-d6ed0668f6b0 src: de693a0b81 -->
### 2.4 설치된 DX-STREAM 요소 확인

플러그인은 다음 13개 요소를 제공해야 합니다.

`dxconvert`, `dxgather`, `dxinfer`, `dxinputselector`, `dxmsgbroker`, `dxmsgconv`, `dxosd`, `dxoutputselector`, `dxpostprocess`, `dxpreprocess`, `dxrate`, `dxscale`, `dxtracker`.

<!-- cell: 6a2cff9b-d3eb-48ad-be1e-c18fb204726d src: 3e2aa3dfa3 -->
## 3. 빠른 시작

### 3.1 YOLO26n 객체 검출 파이프라인 실행

이 파이프라인은 영상을 읽고, 각 프레임을 전처리하고, YOLO26n을 실행하고, 결과를 디코딩한 뒤 주석이 그려진 출력을 렌더링합니다. 데스크톱 세션(`HEADLESS = False`, 디스플레이가 감지되면 기본값)에서는 결과가 창에 열리고, 영상이 끝나거나 중지 버튼(■)을 누를 때까지 셀이 실행 중 상태로 유지됩니다. 디스플레이가 없으면(`HEADLESS = True`) 프레임이 `fakesink`로 가고 셀은 파이프라인 상태 메시지만 출력합니다. 이때의 핵심은 모든 요소가 링크되고, 모델이 로드되고, 추론이 실행된다는 점입니다. 셀은 어떤 싱크를 사용하는지 출력합니다.

대신 터미널(File > New > Terminal)에서 실행하려면 설정 셀이 출력한 DX-STREAM 디렉터리로 `cd`한 뒤, 아래 `gst-launch-1.0` 명령의 마지막 요소를 `videoconvert ! fpsdisplaysink sync=false`로 바꿔 붙여 넣으세요. `Ctrl+C`로 중지합니다.

<!-- cell: 31792737-11bd-409e-975b-6d068cf213a8 src: b9702f410a -->
### 3.2 핵심 요소 살펴보기

`gst-inspect-1.0`은 설치된 버전의 패드 템플릿, 속성, 기본값, 지원되는 열거형 값을 보여 줍니다.

<!-- cell: ce012ae3-7b15-4cde-89fa-9e7a998f985f src: e7ac256517 -->
#### 3.2.1 DxPreprocess

```bash
dxpreprocess preprocess-id=1 resize-width=640 resize-height=640
```

`preprocess-id`는 생성된 입력 텐서를 식별합니다. 다운스트림의 `dxinfer`는 같은 ID를 참조해야 합니다.

<!-- cell: 286b0cb5-2662-454e-a1a3-ae94aff066ce src: 5260c0d019 -->
#### 3.2.2 DxInfer와 추론 백엔드

```bash
dxinfer preprocess-id=1 inference-id=1 model-path=/path/to/model.dxnn backend=auto
```

- `auto`: 사용 가능한 컴파일된 백엔드를 선택
- `dxrt`: DEEPX Runtime 백엔드를 사용
- `dxvnpu`: DX-STREAM이 VNPU 지원으로 빌드된 경우 VNPU 백엔드를 사용

`inference-id`는 `dxpostprocess`가 소비하는 출력 텐서를 식별합니다. DX-STREAM v3.1.x는 내부적으로 비동기 Put/Get 백엔드 인터페이스를 사용하며, 파이프라인 문법은 변하지 않았습니다.

<!-- cell: eba1a455-60a5-4898-99b4-ab0a40a00f21 src: b9001c0bcb -->
#### 3.2.3 DxPostprocess

```bash
dxpostprocess inference-id=1 \
  library-file-path=/usr/local/share/gstdxstream/lib/libpostprocess_yolo26od.so \
  function-name=PostProcess
```

`inference-id`는 업스트림 추론 출력과 일치해야 합니다. 공유 라이브러리와 함수는 모델 아키텍처와 일치해야 합니다.

<!-- cell: 98989ca3-7203-4c23-9311-c49d59adf254 src: 42550f73c9 -->
#### 3.2.4 DxOsd

DxOsd는 추론 메타데이터를 읽어 영상 프레임 위에 시각적 결과를 오버레이합니다.

<!-- cell: 98f4122d-835f-46a0-b359-58d35c5ef222 src: 86672b6017 -->
### 3.3 영상 프레임 스케일링과 변환

`dxscale`은 해상도를, `dxconvert`는 색상 형식을 변경합니다. 지원되는 플랫폼에서는 가속 커널이 자동으로 선택되며, 지원되지 않는 조합은 소프트웨어 구현으로 폴백합니다.

```bash
... ! dxscale width=640 height=480 ! \
      dxconvert ! video/x-raw,format=RGB ! ...
```

`dxconvert`는 프레임 크기를 바꾸지 않고, `dxscale`은 최종 색상 형식을 선택하지 않습니다.

이 노트북의 파이프라인은 디스플레이 싱크 앞에 소프트웨어 `videoconvert`를 두고 끝납니다. 하드웨어 `dxconvert`가 더 빠르지만, 이 SDK 릴리스에서는 일부 클립(예: 5절의 Forklift and Worker 영상)에서 간헐적으로 전체가 녹색인 프레임이 발생했습니다. 번들 스크립트는 `VIDEOCONVERT_PIPELINE` 변수를 통해 플랫폼별로 컨버터를 선택합니다. `SINK`를 `dxconvert`로 바꿨다가 녹색 프레임이 보이면 다시 되돌리세요.

<!-- cell: a2bacdbc-f65c-4c63-804d-88241bb9f07a src: 17eb05a204 -->
## 4. 번들 데모 파이프라인 실행

`run_demo.sh`는 12가지 선택지를 제공합니다. `0`~`9`, Secondary Mode용 `-`, Depth Estimation용 `=`입니다. 10초 안에 입력이 없으면 옵션 `0`을 실행합니다. 다음 셀은 설치된 메뉴를 출력합니다.

이 데모들은 네이티브 창을 열고 파이프라인이 종료될 때까지 셀을 실행 중 상태로 유지하므로, 아래 셀들은 데스크톱 세션에서 `RUN_DEMOS = True`(설정 셀)일 때만 실행됩니다. 권장 방법은 터미널(File > New > Terminal)입니다.

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_stream     # the setup cell printed your actual path
./run_demo.sh
```

메뉴 항목을 바로 입력하고(프롬프트는 10초 뒤 타임아웃됩니다) `Ctrl+C`로 파이프라인을 중지하세요. 각 데모 뒤에는 실행되는 스크립트와 그 안에서 확인할 점을 적어 두었습니다.

<!-- cell: cdce0b1a-4fae-4368-9720-5bfb211b0776 src: 63e050e95e -->
실행 중인 파이프라인은 노트북 중지 버튼(`■`)으로 중지합니다.

### 4.1 객체 검출 - YOLO26n

![단일 객체 검출 파이프라인](assets/pipline-single-detection.png)

확인할 점: 빠른 시작과 같은 다섯 개의 DX 요소. `model-path`와 `library-file-path=.../libpostprocess_yolo26od.so`가 `dxinfer`와 `dxpostprocess`에 직접 설정되어 있습니다.

<!-- cell: f9415924-9b2c-4691-8811-46da31c7b397 src: c79045ee08 -->
### 4.2 PPU를 사용한 객체 검출 - YOLOv5s

확인할 점: `library-file-path`가 없습니다. 각 DX 요소는 `dx_stream/configs/YoloV5S_PPU/` 아래의 JSON `config-file-path`를 읽습니다. PPU는 NPU가 이미 박스를 디코딩했다는 뜻이므로, 후처리 설정은 PPU 출력을 매핑하기만 합니다.

<!-- cell: 0598f06b-a1de-40bd-8339-d6c93eff57a5 src: ef74dca18a -->
### 4.3 얼굴 검출

확인할 점: 얼굴 모델과 `libpostprocess_yolov5s_face.so`를 사용하는 4.1의 구조. `dxosd`가 얼굴 박스와 랜드마크를 그립니다.

<!-- cell: 95a894ed-f224-4e72-b6f3-b2fc702b4f9e src: 51efb7d6f3 -->
다음 예제는 PPU 출력을 사용하는 SCRFD500M 모델을 사용합니다.

확인할 점: 범용 `libpostprocess_ppu.so`. PPU 모델에서는 하나의 공유 라이브러리가 여러 아키텍처를 처리합니다.

<!-- cell: bed6a8d7-6e12-444d-85c4-72c6e7dab466 src: f29ec878ae -->
#### 카메라 소스 사용

첫 번째 예제는 raw 비디오를 요청합니다. 두 번째는 MJPEG를 요청하는데, 모든 USB 카메라가 지원하지는 않습니다. `v4l2-ctl --list-formats-ext --device=/dev/video0`에 맞게 장치와 caps를 조정하세요.

<!-- cell: db029aad-2a05-4904-946c-d7b67e8e27cc src: 9f586331fc -->
### 4.4 포즈 추정

확인할 점: `libpostprocess_yolo26pose.so`가 `DXObjectMeta`에 키포인트를 기록하고, `dxosd`가 이를 바탕으로 스켈레톤을 그립니다.

<!-- cell: 70f47140-89cd-477b-abfe-e5c5ed869042 src: b99cd2cfe3 -->
다음 예제는 PPU 출력을 사용하는 YOLOv5 Pose를 사용합니다.

확인할 점: 다시 `libpostprocess_ppu.so`입니다. 4.4와 비교하면 모델과 라이브러리만 바뀝니다.

<!-- cell: fc15ee5e-4fc9-4595-a4e2-3da2f54f5420 src: eece8565d4 -->
### 4.5 인스턴스 세그멘테이션 - YOLO26n-Seg

옵션 `6`은 세그멘테이션 데모를 실행합니다. 현재 SDK 레이아웃에서 스크립트는 `instance_segmentation` 아래에 있습니다.

확인할 점: `dxpostprocess`와 `dxosd` 사이의 `dxtracker` 요소. 세그멘테이션 데모는 마스크에도 트랙 ID를 부여합니다.

<!-- cell: ba5d8d52-5fe5-4cd2-93b8-b2db52570a1a src: 2a8e38c8f9 -->
### 4.6 다중 객체 추적

옵션 `7`은 객체 검출에 이어 추적을 실행합니다.

![단일 추적 파이프라인](assets/pipline-single-tracking.png)

확인할 점: 4.2의 검출 체인에 `dxtracker config-file-path=.../tracker_config.json`이 추가되었습니다. 트래커 설정이 알고리즘과 임계값을 선택합니다.

<!-- cell: 154b4e5c-5a1f-4036-9be6-fd8e8e1a39ab src: ea853b3cad -->
### 4.7 멀티 채널 객체 검출

옵션 `8`은 네 개의 독립적인 추론 브랜치를 실행하고 그 출력을 합성합니다.

![멀티 채널 파이프라인](assets/pipeline-multi-stream.png)

확인할 점: 각각 자체 전처리, 추론, 후처리를 가진 네 개의 `filesrc ... ! dxscale` 브랜치가 `compositor`로 병합됩니다. 모델은 브랜치마다 한 번씩 로드됩니다.

<!-- cell: 30471517-0937-4f1a-bd77-f41ac4749987 src: 3b21ecdbaf -->
### 4.8 DX 멀티스트림 도메인

DX-STREAM v3.1.0은 `application/x-dxvideoraw` caps 도메인을 도입했습니다. 이를 통해 여러 스트림이 각 스트림의 식별자, 크기, 형식, 메타데이터, 타임라인을 유지한 채 하나의 처리 체인을 공유할 수 있습니다.

```text
N × video/x-raw
       ↓
dxinputselector                 domain entry
       ↓ application/x-dxvideoraw
dxpreprocess → dxinfer → dxpostprocess → dxosd
       ↓ application/x-dxvideoraw
dxoutputselector                domain exit
       ↓
N × video/x-raw
```

`dxinputselector`는 스트림 ID를 부여하고 필요할 때 `DXFrameMeta`를 생성합니다. `dxoutputselector`는 각 버퍼를 라우팅하고 해당 스트림별 이벤트를 복원합니다.

도메인은 스트림별 `STREAM_START`, `CAPS`, `SEGMENT`, `TAG`, `EOS`, `GAP` 이벤트를 보존합니다. 따라서 느린 스트림이 모든 입력을 제한하지 않고 자체 QoS를 처리할 수 있습니다.

<!-- cell: 0ecc9dfd-6d20-4385-b4c3-f5bcc9b6bf70 src: aa2cd3fde1 -->
#### 요소 배치 규칙

| 요소 | `application/x-dxvideoraw` 내부 | 배치 참고 |
|---|---:|---|
| `dxpreprocess`, `dxinfer`, `dxpostprocess`, `dxtracker`, `dxosd`, `dxrate` | 예 | 이 요소들은 단일 스트림 모드 또는 도메인 모드로 동작합니다 |
| `dxscale`, `dxconvert` | 아니요 | `dxinputselector` 앞 또는 `dxoutputselector` 뒤에 배치합니다 |
| `dxgather` | 아니요 | 서로 다른 스트림 ID가 아니라 하나의 소스에서 나온 브랜치를 병합합니다 |
| `videoconvert`, `videoscale`, `compositor` 같은 표준 요소 | 아니요 | `application/x-dxvideoraw`가 아닌 `video/x-raw`를 받습니다 |
| `dxinputselector`, `dxoutputselector` | 경계에서만 | 도메인에 진입하고 빠져나옵니다 |

잘못된 배치는 모호한 런타임 결과를 내는 대신 caps 협상 단계에서 실패합니다.

<!-- cell: e89082d6-2e11-4491-acc9-b64f356b528a src: 76d7544217 -->
#### 하나의 추론 체인 공유

모든 채널이 같은 모델을 사용할 때 하나의 공유 추론 체인을 사용하면 채널마다 모델을 로드하지 않아도 되고 NPU 메모리 사용량이 줄어듭니다.

<img src="assets/pipeline-multi-stream-single-infer.png" style="max-width: 1400px;">

현재 예제는 `run_multi_stream_selector.sh`를 사용합니다.

확인할 점: 하나의 공유 체인을 감싸는 `dxinputselector name=in`과 `dxoutputselector name=out`. DX 요소 수를 세어 4.7과 비교해 보세요.

<!-- cell: e8f05808-0f68-475f-b0b0-d9ee7ba85240 src: c7e15732ef -->
### 4.9 멀티 채널 RTSP

옵션 `9`는 여러 RTSP 소스를 시연합니다. 라이브 파이프라인은 v3.1.x에서 수정된 지연 시간 및 QoS 보고의 이점을 얻습니다.

확인할 점: `rtspsrc` 입력을 사용하는 4.8의 셀렉터 레이아웃. 스트림 URL은 스크립트 상단의 변수이며, 사용하는 카메라에 맞게 수정하세요.

<!-- cell: 317ba344-632d-4528-964e-4411d31d7b56 src: 98c893db6e -->
### 4.10 Secondary Mode

![Secondary Mode 파이프라인](assets/pipeline-secondary.png)

- **Primary Mode**(기본 모드)는 전체 프레임을 전처리하고 추론합니다. 후처리는 보통 새 `DXObjectMeta` 객체를 생성합니다.
- **Secondary Mode**(보조 모드)는 검출된 객체 영역을 전처리합니다. 후처리는 기존 객체 메타데이터를 갱신하거나 보강합니다.

확인할 점: 두 번째 전처리·추론·후처리 3종 세트에 설정된 `secondary-mode=true`. SCRFD가 얼굴을 찾은 뒤 EfficientNet이 전체 프레임이 아닌 각 얼굴 영역을 분류합니다.

<!-- cell: c73d9a02-064d-4b78-a062-5598aa8546cc src: 6c9c479a23 -->
### 4.11 추론 결과를 MQTT 또는 Kafka로 발행

`dxmsgconv`는 사용자 정의 메시지 변환 라이브러리로 추론 메타데이터를 변환합니다. `dxmsgbroker`는 페이로드를 MQTT 또는 Kafka 브로커에 발행합니다.

```text
... → dxpostprocess → dxmsgconv → dxmsgbroker
```

v3.1.x에서는 `include-frame=true`가 현재 프레임을 Base64로 인코딩된 JPEG로 메시지에 추가합니다. 이는 JPEG 인코딩 작업, 메시지 크기, 네트워크 트래픽을 늘리므로 소비자가 이미지를 필요로 할 때만 활성화하세요.

```bash
dxmsgconv library-file-path=/path/to/libmessage_convert.so include-frame=true ! \
dxmsgbroker broker-name=mqtt conn-info=localhost:1883 topic=test
```

브로커는 파이프라인이 시작되기 전에 실행 중이어야 합니다. 번들 스크립트는 완전한 MQTT 및 Kafka 예제를 제공하지만 대화형 `run_demo.sh` 메뉴에는 포함되지 않습니다.

<!-- cell: 4f9da17d-7791-42e5-9930-16a25ace619a src: a8c8d5a58b -->
## 5. 직접 애플리케이션 작성하기

이 절에서는 튜토리얼 03에서 만든 Forklift and Worker 검출기를 통합합니다. 이 모델은 YOLOv7 출력 형식을 사용하지만 COCO 80개 클래스 대신 두 개의 클래스를 가지므로 후처리 설정을 조정해야 합니다.

![사용자 정의 파이프라인](assets/custom-pipeline.png)

<img src="assets/detection-goal.jpg" style="max-width: 1400px;">

<!-- cell: 66060d59-25bd-463a-8c02-e67713decc74 src: 423cab8027 -->
### 5.1 DX-STREAM v3.1.x 사용자 정의 라이브러리 API

이전 버전용으로 작성된 사용자 정의 전처리 및 후처리 라이브러리는 다음과 같이 업데이트해야 합니다.

- 순환 버퍼 참조를 피하기 위해 `DXFrameMeta::_buf`가 제거되었습니다.
- 사용자 정의 함수의 첫 번째 인자는 이제 `GstBuffer *buf`입니다.
- 객체 메타데이터는 `dx_acquire_obj_meta_from_pool()`로 생성합니다.
- 새 객체는 `dx_add_obj_meta_to_frame()`으로 첨부합니다.
- Primary Mode에서는 후처리가 결과 객체를 생성합니다.
- Secondary Mode에서는 후처리가 `object_meta`로 전달된 객체를 갱신합니다.

현재 YoloV7 라이브러리는 이미 v3.1.x 함수 형식을 사용합니다.

```cpp
extern "C" void PostProcess(
    GstBuffer* buf,
    std::vector<dxs::DXTensor> network_output,
    DXFrameMeta* frame_meta,
    DXObjectMeta* object_meta)
{
    DXObjectMeta* result = dx_acquire_obj_meta_from_pool();

    // Decode tensors and populate result.

    dx_add_obj_meta_to_frame(frame_meta, result);
}
```

<!-- cell: 48523baa-d664-4f24-836e-f876a6c6959e src: e73e58aa61 -->
### 5.2 메타데이터 계층 구조

추론 결과는 별도의 사이드 채널이 아니라 `GstBuffer`와 함께 이동합니다.

```text
GstBuffer
└── DXFrameMeta
    ├── stream ID, width, height, format, ROI
    ├── input tensors  (preprocess ID → tensors)
    ├── output tensors (inference ID → tensors)
    ├── frame-level classification or segmentation
    ├── DXObjectMeta[]
    │   ├── label, confidence, box, tracking ID
    │   ├── keypoints, features, OBB, face, segmentation
    │   └── DXUserMeta[]
    └── DXUserMeta[]
```

`DXUserMeta`는 애플리케이션별 데이터를 프레임이나 객체에 첨부할 수 있습니다. 사용자 메타 구현은 GStreamer가 버퍼를 복사하거나 해제할 때 메타데이터가 유효하게 유지되도록 복사 함수와 해제 함수를 모두 제공해야 합니다.

<!-- cell: 712b8026-aa30-4662-a7ec-5f7a2e79e179 src: 77b28e4653 -->
### 5.3 사용자 정의 DXNN 모델 가져오기

튜토리얼 03은 `yolov7-forklift-person.dxnn`을 컴파일(또는 다운로드)해 해당 워크스페이스에 저장했습니다. 셀은 그 파일이 있으면 복사하고, 없으면 같은 모델(71 MB)을 다운로드합니다.

```bash
wget -nc https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/yolov7-forklift-person.dxnn
```

<!-- cell: 8e29f8a5-4356-4e58-93a9-9c247bfa7c77 src: fa60065b2f -->
### 5.4 컴파일된 모델 살펴보기

후처리기를 건드리기 전에 모델이 무엇을 출력하는지 확인합니다. `dxparse -v`는 입력과 출력 텐서를 출력합니다. 사용자 정의 모델은 YOLOv7 헤드 레이아웃을 유지합니다. raw 검출 헤드는 COCO 모델의 `3 × (5 + 80) = 255` 대신 `3 × (5 + 2) = 21` 채널(앵커 3개, 박스 값 4개, objectness, 클래스 점수 2개)을 가집니다. `libpostprocess_yolov7.so`의 디코더는 이미 이 레이아웃을 처리하며, 클래스 수와 클래스 이름만 다릅니다. 5.5의 패치가 바꾸는 것이 바로 이 부분입니다.

<!-- cell: 6972d420-7312-4a41-bd63-1a01205af19a src: 0127f93eee -->
### 5.5 YoloV7 후처리 설정 조정

5.4에서 보았듯이 출력 디코더는 이미 모델 아키텍처와 일치합니다. 클래스의 수와 이름만 바꾸면 됩니다. 패치는 워크스페이스에 기록되고 5.6에서 SDK 소스에 적용됩니다.

<!-- cell: b16382e7-7b3a-45e5-88bd-9bb80cc61480 src: 1e14d10b82 -->
### 5.6 패치 적용

아래 셀은 여러 번 실행해도 안전합니다. 필요할 때 패치를 적용하고, 같은 패치가 이미 있으면 성공을 보고하며, 소스에 충돌하는 수정이 있으면 파일을 바꾸지 않고 중단합니다. 기존 작업을 초기화하거나 버리지 않습니다.

<!-- cell: 4f49345d-1354-4454-9bfe-bf1de2b099b7 src: b1a18419d0 -->
### 5.7 DX-STREAM 다시 빌드

`build.sh`는 `meson`으로 플러그인과 사용자 정의 라이브러리를 다시 빌드하고, 설정된 prefix 아래에 설치하며, 6절을 위해 `pydxs`가 포함된 `venv-dx_stream`을 생성합니다. 설치 단계와 root 소유 빌드 디렉터리 정리에는 `sudo`를 사용하므로, 튜토리얼 01과 마찬가지로 셀은 비밀번호 없는 `sudo`가 동작할 때만 여기서 빌드를 실행하고, 그렇지 않으면 터미널용 명령을 출력합니다. 전체 빌드에는 몇 분이 걸립니다. 기존 `builddir`이 다른 meson 버전으로 생성되어 증분 빌드가 실패하면, 셀은 처음부터 다시 빌드하는 `./build.sh --clean`으로 폴백합니다.

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_stream
./build.sh
```

<!-- cell: 5d801612-1e0a-44fa-8345-74636b91a421 src: 9808c3e4d1 -->
### 5.8 추론 설정 생성

`dxinfer` 속성은 파이프라인에 직접 지정하거나 JSON 파일로 제공할 수 있습니다.

| JSON 필드 | 의미 |
|---|---|
| `preprocess_id` | 같은 ID를 가진 `dxpreprocess` 요소가 생성한 입력 텐서를 선택 |
| `inference_id` | 모델 출력 텐서에 라벨을 붙임. `dxpostprocess`는 같은 ID를 사용해야 함 |
| `model_path` | 컴파일된 `.dxnn` 모델 경로. 상대 경로는 프로세스 작업 디렉터리를 기준으로 해석되므로 셀은 절대 경로를 기록 |
| `backend` | `auto`, `dxrt` 또는 컴파일된 선택 백엔드를 선택 |

여러 전처리, 추론, 후처리 요소가 포함된 파이프라인에서는 명시적인 ID 사용이 중요합니다.

<!-- cell: 1341a38d-276d-4ac6-85e3-c454925cb1a9 src: ff1c547c4c -->
### 5.9 테스트 영상 가져오기

튜토리얼 03에서 사용한 것과 같은 클립입니다. 해당 워크스페이스에 있으면 복사하고, 없으면 다운로드(19 MB)합니다.

<!-- cell: ccf95200-0466-4b6c-91fb-5dcecd0bc6cf src: f4aec25b4e -->
### 5.10 사용자 정의 파이프라인 실행

ID는 다음과 같은 관계를 이룹니다.

```text
dxpreprocess(preprocess-id=1)
        ↓
dxinfer(preprocess_id=1, inference_id=1)
        ↓
dxpostprocess(inference-id=1)
```

`library-file-path`는 5.7에서 설치한 라이브러리를 가리킵니다. 아래 경로는 기본 설치 prefix입니다. DX-STREAM을 `./build.sh --prefix=<dir>`로 빌드했다면 `<dir>/share/gstdxstream/lib/libpostprocess_yolov7.so`를 사용하세요. 출력은 `SINK`로 갑니다. 데스크톱 세션에서는 창, 그 외에는 `fakesink`입니다.

<!-- cell: 83d74722 src: 77c4166080 -->
## 6. YOLO26s 인원 카운터 애플리케이션 구축

이 실습은 각 통합 지점을 쉽게 살펴볼 수 있도록 먼저 Python으로 인원 카운팅을 구현합니다. YOLO26s 객체 검출을 실행하고, 모든 영상 프레임에 첨부된 검출 메타데이터를 읽고, COCO 클래스 `0`(`person`)을 세고, 텍스트 오버레이를 갱신합니다.

이 절을 마치면 다음을 이해하게 됩니다.

- `DXObjectMeta`가 어디에 정의되고 생성되는지
- 검출 메타데이터가 `GstBuffer`와 함께 어떻게 이동하는지
- `pydxs`가 C++ 메타데이터를 Python에 어떻게 노출하는지
- Python 문자열이 어떻게 실행 중인 GStreamer 파이프라인이 되는지
- 패드 프로브 콜백과 디스플레이 갱신이 왜 서로 다른 실행 컨텍스트를 사용하는지

<!-- cell: 08a854bd src: 2272569272 -->
### 6.1 YOLO26s 모델과 영상 준비

`yolo26-s_640x640.dxnn`은 튜토리얼 01의 3.3절에서 공유 워크스페이스에 다운로드한 모델 중 하나이므로 여기서 재사용합니다. 셀은 모델이 없을 때만 DX-APP 설정 스크립트로 다운로드합니다. DX-STREAM 샘플의 스노보드 영상은 여러 사람이 등장하기 때문에 선택했습니다.

<!-- cell: 254abcd0 src: 269af0cb87 -->
### 6.2 검출 메타데이터 이해하기

DX-STREAM은 검출 데이터를 이미지 자체에 기록하지 않고 픽셀과 추론 결과를 함께 유지합니다. 디코딩된 영상 프레임 하나는 `GstBuffer` 하나로 전달됩니다. DX-STREAM은 그 버퍼에 `DXFrameMeta` 하나를 첨부하고, 프레임 메타데이터가 검출된 객체 목록을 소유합니다.

```text
GstBuffer — one video frame moving through the pipeline
│
├── Video memory / pixels
│
└── DXFrameMeta — frame-level GstMeta
    ├── stream_id, width, height, format, frame_rate, ROI, ...
    │
    └── object_meta_list
        ├── DXObjectMeta #0
        │   ├── label          = 0
        │   ├── label_name     = "person"
        │   ├── confidence     = 0.92
        │   └── box            = [x1, y1, x2, y2]
        ├── DXObjectMeta #1
        └── ...
```

이 관계가 중요합니다. Python 콜백은 추론을 다시 실행하지 않고 이미지를 파싱하지도 않습니다. `dxpostprocess`가 이미 생성한 객체 메타데이터를 읽기만 합니다.

<!-- cell: d59665ef src: cd784d857f -->
#### 6.2.1 `DXObjectMeta`가 정의되고 생성되는 위치

SDK는 `DX_STREAM_DIR` 기준의 다음 헤더에 구조체를 정의합니다.

| 용도 | SDK 소스 |
|---|---|
| 프레임 메타데이터와 그 객체 목록 | `gst-dxstream-plugin/metadata/gst-dxframemeta.hpp` |
| 객체별 검출 메타데이터 | `gst-dxstream-plugin/metadata/gst-dxobjectmeta.hpp` |
| 두 구조체의 Python 바인딩 | `bindings/python/pydxs/src/metadata_binding.cpp` |

기본(primary) 객체 검출에서 `dxpostprocess`는 설정된 C++ `PostProcess` 함수를 호출합니다. 이 함수는 다음 생명 주기를 따라 YOLO 텐서 출력을 객체로 변환합니다.

```cpp
DXObjectMeta* object = dx_acquire_obj_meta_from_pool();
object->_label       = class_id;
object->_label_name  = class_name;
object->_confidence  = score;
object->_box         = {x1, y1, x2, y2};
dx_add_obj_meta_to_frame(frame_meta, object);
```

결과는 현재 프레임에 첨부되므로 다운스트림 요소들이 같은 검출 결과를 봅니다.

```text
dxinfer                 dxpostprocess                         downstream
tensor output ────────▶ create DXObjectMeta ────────┬──────▶ dxosd draws boxes
                                                    └──────▶ Python probe counts people
```

DX-STREAM은 메타데이터 수명을 버퍼와 함께 관리합니다. 요소가 새 버퍼를 만들고 `DXFrameMeta`를 복사하면 객체 메타데이터도 함께 복사됩니다. 그럼에도 애플리케이션 코드는 Python 메타데이터 참조를 현재 버퍼 콜백을 처리하는 동안에만 유효한 것으로 취급해야 하며, 나중에 사용하려고 저장하면 안 됩니다.

<!-- cell: b9bcdd40 src: d486fcc937 -->
#### 6.2.2 C++ 메타데이터가 Python에 나타나는 방식

`pydxs`는 C++ 멤버를 Python 친화적인 속성으로 매핑합니다.

| C++ 필드 | Python 속성 | 이 실습에서의 의미 |
|---|---|---|
| `DXFrameMeta::_object_meta_list` | `frame_meta.object_meta_list` 또는 `frame_meta` 순회 | 현재 프레임의 모든 검출 결과 |
| `DXObjectMeta::_label` | `obj_meta.label` | COCO 클래스 ID. `0`은 `person` |
| `DXObjectMeta::_label_name` | `obj_meta.label_name` | 사람이 읽을 수 있는 클래스 이름 |
| `DXObjectMeta::_confidence` | `obj_meta.confidence` | 검출 신뢰도 |
| `DXObjectMeta::_box` | `obj_meta.box` | 프레임 좌표 기준 `[left, top, right, bottom]` |
| `DXObjectMeta::_track_id` | `obj_meta.track_id` | 트래커 ID. 이 파이프라인에서는 부여되지 않음 |

콜백은 `dxpostprocess`를 막 떠나려는 것과 같은 `Gst.Buffer`를 받습니다.

```python
buffer = info.get_buffer()
frame_meta = pydxs.dx_get_frame_meta(hash(buffer))

people_count = sum(
    1 for obj_meta in frame_meta
    if obj_meta.label == PERSON_CLASS_ID
)
```

`hash(buffer)`는 바인딩이 기대하는 네이티브 `GstBuffer` 주소를 제공합니다. `pydxs`가 Python의 이터레이터 프로토콜을 통해 `_object_meta_list`를 노출하므로 `frame_meta`는 순회할 수 있습니다.

따라서 이 카운트는 **현재 프레임**에서 채택된 `person` 검출 박스의 수입니다. Python 코드는 별도의 신뢰도 임계값을 적용하지 않습니다. 신뢰도 필터링과 중복 제거를 사용한다면 객체가 첨부되기 전에 후처리기에서 수행해야 합니다. 이는 고유 방문자 수가 아니며, 시간에 따라 신원을 유지하려면 추적이 필요합니다.

<!-- cell: b962bea0 src: 2f6d8fbc24 -->
### 6.3 Python이 GStreamer 파이프라인을 임베드하는 방식 이해하기

애플리케이션은 `gst-launch-1.0`을 서브프로세스로 호출하지 않습니다. 대신 `pipeline_description`에 같은 GStreamer launch 문법이 Python 포맷 문자열로 들어 있고, `Gst.parse_launch()`가 그 텍스트를 패드로 연결된 실제 GStreamer 요소로 변환합니다.

```text
Python application
│
├── pipeline_description = f''' ... '''
│       │
│       └── Gst.parse_launch(description)
│                    │
│                    ▼
│    ┌────────┐  ┌──────────┐  ┌───────┐  ┌─────────────┐  ┌───────┐
└───▶│ source │─▶│preprocess│─▶│ infer │─▶│ postprocess │─▶│  osd  │─▶ display
     └────────┘  └──────────┘  └───────┘  └──────┬──────┘  └───────┘
                                                  │
                                    source-pad BUFFER probe
                                                  │
                                                  ▼
                                    read metadata → count people
                                                  │
                                                  ▼
                                      update `textoverlay` text
```

프로브는 버퍼를 관찰할 뿐이며, 또 다른 파이프라인 브랜치가 아니고 프레임을 복사하지도 않습니다.

| 파이프라인 구성 | 역할 |
|---|---|
| `urisourcebin ! decodebin` | 영상을 읽고 압축된 프레임을 디코딩 |
| `dxpreprocess` | 모델에 맞게 프레임을 리사이즈하고 준비 |
| `dxinfer` | YOLO26s를 실행하고 출력 텐서를 첨부 |
| `dxpostprocess name=detector_postprocess` | 텐서를 디코딩하고 `DXObjectMeta` 항목을 첨부 |
| `dxosd` | 객체 메타데이터를 읽어 박스와 라벨을 그림 |
| `videoconvert` | 표준 오버레이/디스플레이 경로에 맞게 영상 형식을 준비 |
| `textoverlay name=people_overlay` | Python이 제어하는 카운트를 표시 |
| `fpsdisplaysink` | 프레임을 표시하고 디스플레이 FPS를 측정 |

<!-- cell: 98bc52b7 src: aaf166a292 -->
#### 6.3.1 파이프라인 문자열에서 이름 있는 Python 객체로

세 줄이 파이프라인 텍스트와 애플리케이션 로직을 연결합니다.

```python
self.pipeline = Gst.parse_launch(pipeline_description)
self.people_overlay = self.pipeline.get_by_name("people_overlay")
postprocess = self.pipeline.get_by_name("detector_postprocess")
```

이름은 launch 문자열 안의 속성에서 옵니다.

```text
dxpostprocess name=detector_postprocess ...
textoverlay   name=people_overlay ...
```

그다음 애플리케이션은 후처리기의 소스 패드에 콜백을 연결합니다.

```python
src_pad = postprocess.get_static_pad("src")
src_pad.add_probe(Gst.PadProbeType.BUFFER, self._postprocess_probe)
```

`dxpostprocess`의 모든 출력 버퍼는 `dxosd`로 넘어가기 전에 `_postprocess_probe`를 호출합니다. `Gst.PadProbeReturn.OK`를 반환하면 같은 버퍼가 다운스트림으로 계속 흐릅니다.

<!-- cell: af6d6eca src: 427e4a7436 -->
#### 6.3.2 스트리밍 콜백과 GLib 메인 루프

GStreamer는 스트리밍 스레드에서 패드 프로브를 호출합니다. 느린 콜백은 이후의 모든 프레임을 지연시키므로, 프로브는 메타데이터를 가져와 객체를 세고 디스플레이 갱신을 예약하기만 합니다.

```text
GStreamer streaming thread                    GLib main-loop thread
──────────────────────────                    ─────────────────────
buffer reaches postprocess.src
          │
          ▼
_postprocess_probe()
  ├── get GstBuffer
  ├── get DXFrameMeta
  ├── count label == 0
  └── GLib.idle_add(_update_overlays, count) ───────────────┐
          │                                                 ▼
          └── return OK                           _update_overlays()
                    │                               └── set text property
                    ▼
             buffer continues                     UI remains responsive
```

버스는 파이프라인 전체 이벤트도 메인 루프에 보고합니다.

```text
GStreamer bus ── ERROR ──▶ print details and stop
              └─ EOS   ──▶ stop the main loop
```

`pipeline.set_state(Gst.State.PLAYING)`은 데이터 흐름을 시작하고, `GLib.MainLoop().run()`은 Python 애플리케이션이 idle 콜백과 버스 메시지를 처리할 수 있도록 살아 있게 유지합니다.

<!-- cell: 6ab93ab6 src: bebe7b0bf7 -->
### 6.4 Python 애플리케이션 생성

아래 전체 애플리케이션은 위에서 설명한 파이프라인, 메타데이터 프로브, 오버레이 갱신, 버스 처리, 명령줄 인자를 결합합니다.

<!-- cell: 13351f19 src: 0781fa93bf -->
### 6.5 DX-STREAM Python 환경 확인

애플리케이션은 `pydxs` 모듈과 GStreamer의 Python 바인딩을 제공하는 DX-STREAM 가상 환경을 사용해야 합니다.

<!-- cell: e6d0e88f src: 89e064ba44 -->
### 6.6 애플리케이션 실행

데스크톱 세션(`HEADLESS = False`)에서는 애플리케이션이 YOLO 박스와 `People: N` 오버레이가 있는 창을 열고, 영상이 끝나거나 중지 버튼을 누를 때까지 실행 중 상태로 유지됩니다. 디스플레이가 없으면(`HEADLESS = True`) `--no-display`로 실행됩니다. 파이프라인, 메타데이터 프로브, 카운트는 그대로 동작하고 프레임은 `fakesink`로 가며, 셀은 EOS에서 요약 한 줄과 함께 끝납니다.

`pydxs`와 GStreamer Python 바인딩이 DX-STREAM 환경에 있으므로 애플리케이션은 Jupyter 커널이 아닌 `venv-dx_stream/bin/python`을 사용해야 합니다. 터미널에서는 다음과 같이 실행합니다.

```bash
cd <DX_ALL_SUITE_DIR>/dx-runtime/dx_stream
./venv-dx_stream/bin/python <workspace>/yolo26_people_counter.py \
    --model <DX_ALL_SUITE_DIR>/workspace/res/models/yolo26-s_640x640.dxnn \
    --video dx_stream/samples/videos/snowboard.mp4
```

<!-- cell: 5c93c78a src: 946e670e83 -->
### 6.7 예상 결과와 확장 지점

일반 YOLO 박스와 라벨은 `dxosd`를 통해 계속 표시됩니다. 표준 GStreamer 텍스트 오버레이가 애플리케이션 상태를 추가합니다.

- 오른쪽 위: `People: N`, 카운트가 바뀔 때마다 갱신됨.

표시되는 숫자는 해당 프레임에서 채택된 `person` 박스 수와 일치해야 합니다. 한 사람에게서 겹치는 박스가 나온다면 Python 오버레이에서 숫자를 보정하기보다 후처리기의 신뢰도/NMS 정책을 변경하세요.

운영 환경의 점유 인원 시스템에서는 `dxtracker`를 추가하고 진입/퇴장 영역을 정의한 뒤, 프레임별 raw 검출이 아니라 안정적인 트랙 ID를 세세요.

<!-- cell: e899a740-4b78-4712-8073-a1904d5b76b2 src: 0a5e5189de -->
## 7. 디버깅

### 7.1 대상을 지정한 GStreamer 로그 사용

초기화와 상태 변경은 레벨 3으로 시작하고, 조사 중인 요소에 대해서만 레벨 4 또는 5를 활성화하세요.

```bash
# All DX-STREAM elements at INFO level
GST_DEBUG=dx*:3 ./run_demo.sh

# Inference and tracker details
GST_DEBUG=dxinfer:4,dxtracker:4 ./run_demo.sh

# Metadata lifecycle
GST_DEBUG=dxmeta:5 ./run_demo.sh

# Save logs to a file
GST_DEBUG=2,dxinfer:4 GST_DEBUG_FILE=/tmp/dxstream.log ./run_demo.sh
```

로그 출력이 타이밍에 영향을 주므로 성능 측정 중에는 상세 로깅을 비활성화하세요.

<!-- cell: 13445965-9c30-429f-80ff-3acf97fd6cee src: 771d6ca90f -->
### 7.2 v3.1.x 실패 동작 이해하기

- 호환되지 않는 링크는 caps 협상 중에 실패합니다.
- 모델 파일 누락, NPU 초기화 실패, 사용자 정의 라이브러리 오류는 `abort()`로 종료되는 대신 GStreamer 요소 오류로 보고됩니다.
- 처리 요소는 측정된 시간을 LATENCY 쿼리에 반영합니다.
- FLUSH 시 내부 상태가 초기화되어 탐색(seek)과 재생 동작이 개선되었습니다.
- EOS, FLUSH, 상태 전환 시 단일 프레임 입력을 포함해 워커 스레드가 더 안정적으로 정리됩니다.

멀티스트림 파이프라인이 링크되지 않으면, 먼저 표준 `video/x-raw` 요소가 `application/x-dxvideoraw` 도메인 안에 배치되지 않았는지 확인하세요.

<!-- cell: a340afc0-dfc5-471d-98aa-6c7e8e2e0f53 src: af81fae94c -->
### 7.3 파이프라인 그래프 내보내기

`GST_DEBUG_DUMP_DOT_DIR`이 설정되어 있으면 GStreamer는 파이프라인 상태가 바뀔 때마다 DOT 파일을 기록합니다. Graphviz는 `PAUSED_PLAYING` 파일을 실제 요소 그래프 그림으로 바꿔 줍니다. 필요하면 한 번 설치하세요.

```bash
sudo apt install graphviz
```

다음 셀은 커널에 변수를 설정하고 3.1의 빠른 시작 파이프라인을 다시 실행해(설정에 따라 헤드리스 또는 창 모드) DOT 파일이 튜토리얼 워크스페이스에 생성되도록 합니다. 동일한 터미널 명령은 다음과 같습니다.

```bash
GST_DEBUG_DUMP_DOT_DIR=<workspace>/dot gst-launch-1.0 filesrc location=... ! ... ! fakesink
```

<!-- cell: af463f4c-5ea2-4477-b67c-fcc4cad871b7 src: f4e26083b6 -->
### 7.4 DX-STREAM v3.1.x에서 달라진 점

이 튜토리얼은 DX-STREAM v3.1.2로 검증되었습니다. v3.1.x 계열은 이 튜토리얼이 의존하는 다음 런타임 동작을 도입했습니다.

- `application/x-dxvideoraw` 기반의 통합 멀티스트림 도메인
- 공유 처리 체인에서 스트림별 생명 주기와 타임라인 보존
- `dxinfer` 백엔드 추상화와 비동기 Put/Get 처리
- Base64로 인코딩된 JPEG 프레임을 위한 `dxmsgconv include-frame`
- 업데이트된 사용자 정의 라이브러리 및 메타데이터 API(5.1절)
- 개선된 지연 시간 보고, FLUSH 복구, caps 협상, 오류 보고

사용 가능한 백엔드와 요소 속성은 DX-STREAM 빌드 방식에 따라 다릅니다. `gst-inspect-1.0`으로 설치된 빌드를 확인하세요.

<!-- cell: 325d5d11-7b12-4ddb-83e2-f1cb7ce17a91 src: 9ef3c41d4e -->
## 8. 문제 해결

| 증상 | 표시되는 내용 | 원인 | 해결 |
|---|---|---|---|
| DX-STREAM 플러그인을 찾을 수 없음 | `DX-STREAM plugin not found` 또는 `no element "dxinfer"` | JupyterLab을 시작한 환경에 `GST_PLUGIN_PATH`가 설정되지 않음 | `./run-jupyter-lab.sh` 전에 `GST_PLUGIN_PATH=/usr/local/lib/x86_64-linux-gnu/gstreamer-1.0`(`build.sh`가 출력한 경로)을 export하거나 `./build.sh`로 다시 빌드 |
| 창이 나타나지 않음 | `Could not open display` 또는 파이프라인이 아무 출력 없이 멈춤 | 그래픽 세션 없음 | 데스크톱 세션에서 실행. 빠른 확인용으로는 `fpsdisplaysink`를 `fakesink`로 교체 |
| 카메라 파이프라인 실패 | `v4l2src: Cannot open device` 또는 caps 협상 오류 | 잘못된 장치 경로 또는 지원되지 않는 해상도 | `v4l2-ctl --list-formats-ext`로 확인하고 셀의 장치와 caps 수정 |
| RTSP 데모가 연결되지 않음 | `Could not connect to server` | `run_RTSP.sh`의 샘플 RTSP URL에 현재 네트워크에서 접근할 수 없음 | 접근 가능한 스트림으로 `run_RTSP.sh` 수정 |
| `build.sh`가 멈춤 | 셀 안에서 `sudo` 비밀번호 프롬프트 또는 `meson` 오류 | 빌드가 시스템 전역에 설치되어 `sudo`가 필요함 | 터미널(File > New > Terminal)에서 `./build.sh` 실행 |
| 일부 프레임이 전체 녹색 | 창이나 저장된 프레임이 간헐적으로 녹색으로 깜박임 | 디스플레이 싱크 앞의 `dxconvert`가 일부 디코딩된 버퍼를 잘못 처리 | `videoconvert ! fpsdisplaysink`(노트북 기본값) 사용. `dxconvert`는 3.3의 `dxscale`/`dxconvert` 실험에서만 유지 |
| DOT 그래프가 생성되지 않음 | `No DOT file was generated` | 파이프라인이 PLAYING에 도달하기 전에 데모가 중지됨 | 영상이 재생되기 시작한 뒤 셀을 중지 |
| 샘플 모델이 로드 시 거부됨 | `dxinfer`의 DX-RT 버전 또는 형식 오류 | 샘플 모델이 Model Zoo 릴리스 `2_4_0`용으로 컴파일되어 이전 런타임에서는 로드할 수 없음 | 이 튜토리얼이 검증된 DX-RT 버전(3.4.2)을 사용하거나 사용 중인 DX-COM으로 모델 재컴파일 |
| `setup.sh`가 비밀번호를 요구 | `jq` 설치 중 `[sudo] password for ...` | `jq`가 없어 스크립트가 `apt`로 설치함 | 터미널에서 `sudo apt install jq`를 한 번 실행한 뒤 셀 재실행 |
| `import pydxs` 실패 | `ModuleNotFoundError: No module named 'pydxs'` | DX-STREAM이 아직 빌드되지 않았거나 `venv-dx_stream` 대신 Jupyter Python을 사용함 | 5.7(빌드)을 실행한 뒤 셀과 같이 `venv-dx_stream/bin/python` 사용 |
| `build.sh`가 즉시 실패 | `Build data file ... was generated with an old version of meson` | 이전 DX-STREAM 버전이 남긴 `builddir` | `./build.sh --clean` 실행(5.7의 셀은 첫 시도가 실패하면 자동으로 수행) |

<!-- cell: 51fafd7a-3597-4c54-a355-0cad94216211 src: de9bc8a30a -->
## 9. 요약

이제 다음을 완료했습니다.

- 엔드투엔드 DX-STREAM 추론 파이프라인을 구축하고 살펴봄
- 번들 객체 검출, PPU, 얼굴 검출, 포즈, 세그멘테이션, 추적, 멀티 채널, RTSP, Secondary Mode 데모를 실행하고 그 뒤의 스크립트를 읽음
- 추론 백엔드, `dxscale`, `dxconvert`, MQTT/Kafka 메시지 브로커 스크립트를 살펴봄(브로커 데모는 실행 중인 브로커가 필요하며 메뉴에 포함되지 않음)
- `application/x-dxvideoraw` 멀티스트림 도메인과 요소 배치 규칙을 학습함
- 두 클래스 Forklift and Worker 모델에 맞게 YOLOv7 후처리기와 추론 설정을 조정함
- 검출 결과를 `GstBuffer`에서 `DXFrameMeta`, `DXObjectMeta`를 거쳐 `pydxs`로 Python까지 따라감
- GStreamer 파이프라인을 Python에 임베드하고 패드 프로브와 GLib 메인 루프로 프레임별 인원 수를 표시함
- 대상을 지정한 GStreamer 로그와 DOT 그래프로 파이프라인 동작을 진단함

### 9.1 완료 체크리스트

- [ ] `gst-inspect-1.0`으로 13개 DX-STREAM 요소 확인
- [ ] 빠른 시작 YOLO26n 파이프라인과 번들 데모 최소 3개 실행
- [ ] 멀티스트림 도메인 배치 규칙 설명
- [ ] 두 클래스 Forklift and Worker 후처리기를 패치, 재빌드, 실행
- [ ] Python 인원 카운터 애플리케이션을 실행하고 `pydxs`로 `DXObjectMeta` 읽기
- [ ] 파이프라인 DOT 그래프 내보내기 및 열기

> **다음:** 튜토리얼 05로 이동해 DX-Compiler 워크플로를 깊이 있게 배워 보세요. ONNX 검증, 캘리브레이션, PPU, 그래프 최적화, 양자화 전략을 다룹니다.
