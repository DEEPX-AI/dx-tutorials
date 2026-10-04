<!-- i18n source: notebooks/T23-demo-hand-landmarks/hand_landmarks.ipynb -->
<!-- i18n lang: ko -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: title src: 08f7a63301 -->
# DEEPX Tutorial 23 - 손 랜드마크 C++ 데모

이 노트북은 검출된 각 손에서 21개의 랜드마크를 추적하는 C++ 애플리케이션을 설명하고, 빌드하고, 실행합니다. 카메라 또는 비디오 입력을 받으며, 두 추론 단계 모두에 DEEPX NPU를 사용합니다.

<!-- cell: 169e124a src: c366966d31 -->
![손 랜드마크 데모](assets/hand-landmarks-sc.png)

<!-- cell: 99f74090-1591-4f3c-83d3-c7911115ef61 src: e791a70079 -->
## 학습 목표

이 튜토리얼을 마치면 다음을 할 수 있습니다.

- 손바닥 검출과 손 랜드마크로 이루어진 2단계 파이프라인을 설명한다.
- 데모용 모델과 샘플 비디오를 다운로드한다.
- `dxparse`로 두 DXNN 모델을 검사한다.
- Qt 애플리케이션을 빌드하고 카메라 또는 비디오로 실행한다.
- 명령줄에서 검출 임계값과 표시 옵션을 조정한다.

<!-- cell: t23-npu-pattern src: 1878462df3 -->
## 이 애플리케이션의 NPU 사용 방식

| 항목 | 이 튜토리얼에서 | 배운 곳 |
|---|---|---|
| 엔진 | 프레임마다 두 개의 `InferenceEngine` 객체가 순서대로 실행됨: 손바닥 검출기를 한 번, 그다음 검출된 손마다 랜드마크 모델을 한 번씩 | T06-2 §4 |
| 실행 | 손바닥 검출은 동기로 실행되고, 각 손의 랜드마크 요청은 `RunAsync()`로 제출한 뒤 콜백에서 프레임 좌표로 되돌리므로 여러 손이 NPU에서 겹쳐 처리됨 | T06-2 §5 |
| 태스크 그래프 | 손바닥 검출기는 `cpu_0` 태스크(ONNX Runtime을 통한 앵커 디코딩)로 끝나고, 랜드마크 모델은 NPU 전용 | T06-1 §6 |
| 요청당 입력 | `[1, 192, 192, 3]` UINT8(110 KB)과 `[1, 224, 224, 3]` UINT8(150 KB): 입력이 작으므로 호스트 측 크롭과 회전이 비용의 대부분을 차지함 | T06-3 §4 |
| 측정할 것 | 4.1절의 `dxrun` 기준값 두 개를 창에 표시되는 FPS와 비교. `--max-hands`를 사용하면 프레임당 랜드마크 요청 수가 제한됨 | T06-1 §5 |

<!-- cell: pipeline src: 148af37d15 -->
## 1. 처리 파이프라인

애플리케이션에는 손 파이프라인만 들어 있습니다.

```text
Camera or video frame
        |
        v
Palm detector, 192 x 192
        |
        v
Palm box and rotated hand region
        |
        v
Hand crop, 224 x 224 -> landmark model
        |
        v
21 landmarks and handedness -> Qt GUI
```

손바닥 검출이 먼저 손 영역을 찾습니다. 각 영역은 회전·크롭되어 랜드마크 모델로 전달됩니다. 결과에는 이미지 공간의 점 21개, 월드 랜드마크, 손 존재 신뢰도, 왼손/오른손 좌우 손 구분이 포함됩니다.

<!-- cell: prerequisites src: 21ce181bb7 -->
## 2. 사전 요구 사항

- 튜토리얼 01 완료: DX-RT가 설치되어 있고 DEEPX NPU가 `/dev/dxrt0`으로 보여야 합니다(`dxcli -s`로 확인). Qt 창을 위해 `cmake`, `g++`, `qtbase5-dev`가 필요합니다.
- 그래픽 데스크톱 세션. 카메라 데모를 위한 V4L2 카메라(선택).
- 다운로드: 약 10 MB 아카이브 하나(모델 두 개와 샘플 비디오)를 git이 무시하는 `assets/`에 받습니다.
- 소요 시간: 약 15분. 빌드는 1분 정도 걸립니다. 패키지가 없을 때 아래 `apt` 줄을 실행하는 경우를 제외하면 `sudo`는 필요하지 않습니다.
- DX-RT 3.4.2에서 검증했습니다. 모델은 DX-COM 2.3.0으로 컴파일했습니다.

C++ 빌드 및 GUI 의존성은 다음 명령으로 설치합니다.

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev qtbase5-dev ffmpeg v4l-utils
```

다음 셀은 SDK를 찾고, 위 요구 사항을 확인한 뒤 상태 표를 출력합니다. `MISSING`으로 표시된 항목에는 그것을 제공하는 단계가 함께 표시됩니다.

<!-- cell: layout-heading src: e62a57c223 -->
## 3. 프로젝트 구조

애플리케이션 코드와 스크립트는 `app/` 아래에 있습니다. 모델과 비디오는 빌드 트리 밖의 `assets/` 아래에 둡니다.

<!-- cell: resources src: 46ee82775e -->
## 4. 리소스 다운로드와 확인

두 추론 모델은 Google AI Edge가 제공하는 [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) 모델 번들에서 가져왔습니다. 그 손바닥 검출 모델과 손 랜드마크 모델을 DEEPX NPU 추론용 DXNN 형식으로 변환했습니다.

`get_resources.sh`는 리소스 아카이브를 다운로드하고, 모델과 샘플 비디오를 `assets/`에 추출한 뒤, 추출이 성공하면 다운로드한 아카이브를 삭제합니다.

```text
assets/
├── models/
│   ├── hand-detector_192x192.dxnn
│   └── HandLandmarkLite.dxnn
└── videos/
    └── hands.mp4
```

손바닥 모델 입력은 UINT8 `[1, 192, 192, 3]`이어야 합니다. 랜드마크 모델 입력은 UINT8 `[1, 224, 224, 3]`이어야 합니다.

<!-- cell: resource-download-note src: 670c65bcaf -->
리소스가 없을 때만 다음 셀을 실행하세요. 이름이 같은 기존 파일은 추출 과정에서 교체될 수 있습니다.

<!-- cell: inspect-heading src: c632a33a79 -->
### 4.1 선택: 모델 검사

`dxparse`와 모델이 있으면 다음 셀이 각 모델의 텐서 정보와 `dxrun` 기준값을 출력합니다. 손바닥 검출기는 `cpu_0` 태스크로 끝나므로(앵커가 DX-RT 안의 ONNX Runtime으로 디코딩됨) `--use-ort`가 필요하며, 랜드마크 모델은 NPU 전용입니다. 그렇지 않으면 셀은 실패하지 않고 검사를 건너뜁니다.

<!-- cell: code-heading src: 14c08de877 -->
## 5. C++ 코드 가이드

구현은 `app/hand_landmarks.cpp`에 있습니다. 다음 셀은 튜토리얼 20~22와 마찬가지로 주요 처리 구간을 소스 파일에서 직접 표시하므로, 노트북이 코드 사본을 따로 갖지 않습니다.

<!-- cell: code-explanation src: 1a18d06026 -->
### 5.1 주요 구현 단계

1. `Options`와 `parse_args`가 입력, 모델, 임계값, 표시 모드, 카메라 설정을 선택합니다.
2. `preprocess_palm_frame`이 BGR을 RGB로 변환하고 192 x 192 손바닥 검출기 입력을 만듭니다. 기본적으로 레터박스를 사용합니다.
3. `decode_palm_detections`가 원시 텐서를 손바닥 박스와 키포인트 7개로 변환한 뒤 가중 비최대 억제(NMS)를 적용합니다.
4. 손목과 중지 키포인트가 회전된 손 영역을 정의합니다. `make_landmark_input`이 이를 224 x 224로 워프합니다.
5. `run_landmark_async`가 검출된 손바닥마다 랜드마크 추론을 하나씩 제출합니다. 그 콜백이 결과를 다시 프레임 좌표로 변환합니다.
6. `draw_hand_landmarks`가 21개 점을 연결해 렌더링합니다. `FrameView`가 프레임과 성능 지표를 표시합니다.
7. `run_detection_loop`가 캡처, 두 추론 단계, 렌더링, 선택적 비디오 저장, 재생 속도 조절을 하나로 묶습니다.

<!-- cell: build-heading src: b00b1b09c7 -->
## 6. 빌드

`build.sh`는 CMake를 Release 모드로 구성하고 사용 가능한 모든 CPU 코어로 `make`를 실행합니다. 이전 빌드 디렉터리를 먼저 삭제하려면 `--clean`을 사용하세요.

<!-- cell: camera-heading src: 176345be53 -->
## 7. 카메라로 실행

스크립트는 카메라 인덱스 0을 1280 x 720, 30 FPS로 요청해 사용합니다. NPU, 카메라, 모델이 준비된 그래픽 세션에서만 `RUN_CAMERA`를 `True`로 설정하세요. 애플리케이션을 닫으려면 `Esc` 또는 `Q`를 누릅니다.

<!-- cell: video-heading src: 51c4ab12d3 -->
## 8. 비디오로 실행

`run_video.sh`는 `assets/videos/hands.mp4`를 읽어 반복 재생합니다. 모델과 비디오를 정해진 위치에 둔 뒤 `RUN_VIDEO`를 `True`로 설정하세요.

<!-- cell: custom-run src: 1a1d864d58 -->
## 9. 사용자 지정 명령

다른 카메라와 캡처 모드를 선택합니다.

```bash
cd app
./run_camera.sh --camera 2 --width 1920 --height 1080 --fps 30
```

사용자 지정 비디오를 사용하고 디버깅을 위해 손바닥 영역을 표시합니다.

```bash
./build/hand_landmarks --video /path/to/input.mp4 --loop --show-palm
```

렌더링된 출력을 저장합니다. `output-XX.mp4` 파일은 현재 디렉터리에 기록되므로, `app/`을 깨끗하게 유지하려면 튜토리얼의 `workspace/`에서 실행하세요.

```bash
mkdir -p ../workspace && cd ../workspace
../app/build/hand_landmarks --video /path/to/input.mp4 --save --landmark-only
```

종료하려면 `Esc` 또는 `Q`를, 전체 화면 모드를 전환하려면 `F`를 사용합니다.

<!-- cell: c1eb325f-0e0f-441f-979a-f3b1a4643a01 src: ecf000b5c7 -->
## 10. 모델 출처

이 데모에서 사용하는 손바닥 검출 모델과 손 랜드마크 모델은 Google AI Edge가 제공하는 [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) 모델 번들에서 가져왔습니다. 모델은 DEEPX NPU 추론용 DXNN 형식으로 변환되어 `hand-detector_192x192.dxnn`과 `HandLandmarkLite.dxnn`으로 사용됩니다.

<!-- cell: 86932e85-d1bd-4636-8069-5dd13cbb56ac src: 59326af863 -->
## 11. 유용한 옵션

- `--max-hands N`: 프레임당 최대 손 개수를 설정합니다. 기본값은 4입니다.
- `--palm-conf VALUE`: 손바닥 신뢰도 임계값을 설정합니다. 기본값은 0.2입니다.
- `--landmark-conf VALUE`: 랜드마크 존재 임계값을 설정합니다. 기본값은 0.5입니다.
- `--show-palm`: 손바닥 박스와 회전된 영역도 함께 그립니다.
- `--landmark-only`: 손 랜드마크만 그립니다.
- `--windowed`: 전체 화면 대신 1280 x 720 창을 사용합니다.
- `--save`: 렌더링 결과를 현재 디렉터리에 다음으로 사용 가능한 `output-XX.mp4` 파일로 저장합니다(`workspace/`에서 실행, 9절 참고).

전체 옵션 목록은 `./build/hand_landmarks --help`로 확인하세요. 종료하려면 `Esc` 또는 `Q`를, 전체 화면 모드를 전환하려면 `F`를 누릅니다.

<!-- cell: troubleshooting src: 22102be3a3 -->
## 12. 문제 해결

| 증상 | 표시되는 내용 | 원인 | 해결 |
|---|---|---|---|
| 모델 없음 | 리소스 확인에서 파일이 `missing`으로 표시됨 | `get_resources.sh`가 실행되지 않았거나 아카이브가 변경됨 | 리소스 셀을 실행하고 `assets/models/` 아래의 두 파일 이름을 확인 |
| 모델 거부 | 로드 시 형상 또는 dtype 오류 | 잘못된 모델 파일 | `dxparse`로 입력 형상과 UINT8 dtype을 확인 |
| 카메라를 열 수 없음 | `cannot open camera 0` | 카메라가 없거나 권한 없음 | `v4l2-ctl --list-devices`로 확인하고 `--camera`로 다른 인덱스 시도 |
| 창이 나타나지 않음 | `could not connect to display` | 그래픽 세션 없음 | 데스크톱 세션에서 실행 |
| CMake가 DXRT를 찾지 못함 | `Could not find ... dxrt` | DX-RT가 설치되지 않았거나 사용자 지정 prefix에 있음 | DX-Runtime 설치(튜토리얼 01 3절) 또는 `./build.sh -DCMAKE_PREFIX_PATH=<prefix>` 실행 |

<!-- cell: 24061ba3-6ced-4492-93d3-3508af97fdf0 src: 99d0cc5118 -->
## 13. 요약

두 DXRT 모델이 순서대로 실행됩니다. 손바닥 검출기가 손을 찾고, 랜드마크 모델이 손마다 21개의 키포인트를 예측합니다. Qt 애플리케이션은 프레임을 캡처하고, 두 모델에 비동기로 입력을 전달하고, 실시간 화면에 랜드마크를 그립니다.

### 13.1 완료 체크리스트

- [ ] `get_resources.sh`로 모델과 샘플 비디오를 다운로드함
- [ ] `dxparse`로 두 모델을 검사함
- [ ] `build.sh`로 애플리케이션을 빌드함
- [ ] 비디오 또는 카메라로 애플리케이션을 실행함

> **다음:** 튜토리얼 24로 이동해 실시간 후처리 컨트롤과 함께 PIDNet 시맨틱 세그멘테이션을 실행해 보세요.
