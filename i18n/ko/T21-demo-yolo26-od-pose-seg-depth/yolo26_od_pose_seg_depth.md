<!-- i18n source: notebooks/T21-demo-yolo26-od-pose-seg-depth/yolo26_od_pose_seg_depth.ipynb -->
<!-- i18n lang: ko -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: fa62a5df src: a9203579d1 -->
# DEEPX Tutorial 21 - YOLO26 객체 검출, 포즈, 세그멘테이션, 깊이 데모

이 튜토리얼은 DEEPX NPU로 같은 카메라 또는 비디오 스트림에서 다음 네 가지 YOLO26s 모델을 실행하는 Qt 기반 C++ 애플리케이션을 설명합니다.
- YOLO26s 객체 검출
- YOLO26s 포즈 추정
- YOLO26s 인스턴스 세그멘테이션
- YOLO26s 단안 깊이 추정

![YOLO 다중 채널 데모](assets/yolo26-od-pos-seg-depth.png)

<!-- cell: 7b3ff195 src: d25198ce78 -->
## 학습 목표

이 튜토리얼을 마치면 다음을 할 수 있습니다.

- 독립적으로 구성된 C++ 프로젝트 구조를 이해한다.
- 검출, 포즈, 세그멘테이션, 깊이 파이프라인을 구분한다.
- 입력 프레임 하나가 네 개의 비동기 워커로 분배되는 방식을 이해한다.
- Qt 애플리케이션을 Release 모드로 빌드한다.
- 카메라 또는 비디오 파일로 데모를 실행한다.

<!-- cell: t21-npu-pattern src: 0c9b4db3ff -->
## 이 애플리케이션의 NPU 사용 방식

| 항목 | 이 튜토리얼에서 | 배운 곳 |
|---|---|---|
| 엔진 | 태스크마다 하나씩 **네 개**의 `InferenceEngine` 객체를 같은 장치에서 사용하며, DX-RT가 이들의 요청을 NPU 코어에 번갈아 배정 | T06-3 §2 |
| 실행 | 각 워커는 `bufferCount`를 자신의 동시 처리 한도로 설정해 `RunAsync()`를 호출하고 완료 콜백에서 렌더링하며, 최신 프레임 큐는 지연 시간을 쌓는 대신 오래된 프레임을 버림 | T06-2 §5, §7 |
| 태스크 그래프 | 네 모델 모두 `cpu_0` 태스크(ONNX Runtime을 통한 CPU 헤드 디코딩)로 끝나므로 모델이 하나 늘어날 때마다 호스트 CPU 부하가 증가 | T06-1 §6 |
| 요청당 입력 | 세 모델은 `[1, 640, 640, 3]` UINT8(1.2 MB), 깊이 모델은 `[1, 768, 768, 3]`(1.8 MB)을 받으며, 카메라 프레임 하나가 레터박스 처리된 입력 네 개가 됨 | T06-3 §4 |
| 측정할 것 | 3.1절의 모델별 `dxrun` 기준값과 창의 FPS 레이블 네 개를 비교하며, 네 패널 속도의 합이 NPU가 어떻게 공유되는지를 보여 줌 | T06-1 §5 |

<!-- cell: t21-prerequisites src: 2ab76b757a -->
## 사전 요구 사항

- 튜토리얼 01 완료: DX-RT가 설치되어 있고 DEEPX NPU가 `/dev/dxrt0`으로 보여야 하며, Qt 창을 위한 `cmake`, `g++`, `qtbase5-dev`가 필요합니다.
- 그래픽 데스크톱 세션. 카메라 데모를 위한 V4L2 카메라(선택).
- 다운로드: 약 80 MB 아카이브 하나(깊이 모델과 샘플 비디오, 그리고 튜토리얼 01이 이미 공유 워크스페이스에 넣어 둔 YOLO26s 모델 세 개의 복사본).
- 소요 시간: 약 20분. 빌드는 1~2분이 걸립니다. 패키지가 없을 때 2절의 `apt-get` 줄을 제외하면 `sudo`는 필요하지 않습니다.
- DX-RT 3.4.2로 검증했습니다. 모델은 DX-COM 2.4.0으로 컴파일되었습니다.

다음 셀은 SDK를 찾고, 위 요구 사항을 확인한 뒤 상태 표를 출력합니다. `MISSING`으로 표시된 항목에는 그것을 제공하는 단계가 함께 표시됩니다.

<!-- cell: f8c4f71c src: bafe0141d6 -->
## 1. 튜토리얼 파일 찾기

<!-- cell: 9c6f9a02 src: fa676f3bae -->
### 1.1 프로젝트 구성

```text
T21-demo-yolo26-od-pose-seg-depth/
├── get_resources.sh
├── assets/
│   ├── models/
│   └── videos/
├── app/
│   ├── build.sh
│   ├── run_camera.sh
│   ├── run_video.sh
│   ├── CMakeLists.txt
│   ├── yolo26s_4.cpp
│   ├── common/
│   │   ├── base/
│   │   ├── processors/
│   │   └── utility/
│   ├── factory/
│   └── extern/
└── yolo26_od_pose_seg_depth.ipynb
```

<!-- cell: afe9307e src: b0dd491e5c -->
## 2. 환경 확인

Debian 또는 Ubuntu 패키지를 설치합니다.

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config qtbase5-dev libopencv-dev ffmpeg v4l-utils
```

리소스 아카이브는 약 84 MB입니다(모델 네 개와 샘플 비디오 하나).

<!-- cell: c457d2df src: 9ff0a2e16c -->
## 3. 리소스 다운로드 및 확인

애플리케이션은 `assets/` 아래에 모델 네 개와 샘플 비디오 하나가 있다고 가정합니다.

```text
assets/
├── models/
│   ├── yolo26-s_640x640.dxnn
│   ├── yolo26-s-pose_640x640.dxnn
│   ├── yolo26-s-seg_640x640.dxnn
│   └── yolo26-depth-s_768x768_q-lite.dxnn
└── videos/
    └── dance-960-540.mp4
```

검출, 포즈, 세그멘테이션 모델 세 개는 튜토리얼 01 3.3절이 공유 워크스페이스(`<DX_ALL_SUITE_DIR>/workspace/res/models`)에 다운로드한 것과 같은 파일이므로, 다음 셀은 파일이 있으면 그곳에서 링크합니다. 깊이 모델과 비디오는 이 튜토리얼의 아카이브(약 80 MB)에만 들어 있습니다. `get_resources.sh`가 아카이브를 다운로드해 `assets/`에 추출하고, 추출이 성공하면 아카이브를 삭제합니다.

<!-- cell: bbbdc0e0 src: 670c65bcaf -->
리소스가 없을 때만 다음 셀을 실행하세요. 같은 이름의 기존 파일은 추출 중에 교체될 수 있습니다.

<!-- cell: t21-baseline-md src: e5aeb7a350 -->
### 3.1 모델 네 개 검사 및 벤치마크

애플리케이션 코드를 읽기 전에 네 엔진이 실행할 모델을 먼저 살펴봅니다. `dxparse -v`는 모든 모델이 NPU 태스크 뒤에 `cpu_0` 태스크를 갖는다는 것을 보여 줍니다. 검출 헤드(그리고 깊이 모델의 출력 단계)는 DX-RT 안의 ONNX Runtime이 디코딩하므로, 의미 있는 벤치마크를 위해서는 `--use-ort`가 필요합니다. `dxrun` 수치는 단일 모델 기준값입니다. 애플리케이션은 같은 NPU에서 네 모델을 동시에 실행하므로 각 패널의 FPS는 기준값보다 낮습니다.

```bash
cd <T21>/assets/models
dxparse -m yolo26-s_640x640.dxnn -v
dxrun -m yolo26-s_640x640.dxnn --use-ort -t 3 -v
```

<!-- cell: 2deb9884 src: 1d367207ac -->
## 4. 애플리케이션 아키텍처

```text
Camera or video
       |
       v
CaptureThread
       |
       +--> Detection worker    --> Object Detection panel
       +--> Pose worker         --> Pose Estimation panel
       +--> Segmentation worker --> Instance Segmentation panel
       +--> Depth worker        --> Depth Estimation panel
```

캡처 스레드는 각 BGR 프레임을 네 개의 최신 프레임 큐에 전달합니다. 검출, 포즈, 세그멘테이션은 태스크별 팩토리를 사용합니다. `DepthWorker`는 768 x 768 레터박스 전처리, 비동기 DXRT 추론, 레터박스 제거, 원본 프레임 크기로 리사이즈, Turbo 컬러 매핑을 수행합니다. Qt는 네 가지 실시간 결과를 전체 화면 2 x 2 그리드에 표시합니다.

<!-- cell: 51d08357 src: 44c389b817 -->
## 5. C++ 코드 읽기

다음 헬퍼는 현재 소스 파일에서 선택한 부분을 표시합니다.

<!-- cell: 5e81e61e src: aa11e7c86a -->
### 5.1 빌드 설정

CMake는 C++17 실행 파일 하나를 빌드하고 Qt5 Widgets, OpenCV, DXRT를 링크합니다. `PROJECT_ROOT_DIR`은 튜토리얼 디렉터리를 가리키므로 기본 모델 경로가 `assets/models/` 아래로 해석됩니다.

<!-- cell: 02a0873a src: 4e66e3d6da -->
### 5.2 명령줄 옵션과 기본 리소스

`AppArgs`는 모델 경로 네 개, 카메라 설정, 선택적 비디오 경로, 디버깅 옵션을 정의합니다. `--model-depth`는 기본 깊이 모델을 바꿉니다. `--video`를 생략하면 카메라 모드가 선택됩니다. V4L2 장치를 선택하려면 `-c` 또는 `--camera`를, 캡처 설정을 요청하려면 `--width`, `--height`, `--fps`를 사용합니다. 이 옵션을 생략하면 기본값은 `/dev/video0`, 1280 x 720, 30 FPS입니다.

<!-- cell: 3a6d2550 src: 5104ab68bd -->
### 5.3 태스크 팩토리

각 팩토리는 하나의 태스크에 맞는 전처리, 후처리, 시각화 구성 요소를 생성합니다.

<!-- cell: 70995906 src: e8239c4efd -->
### 5.4 비동기 결과 워커

각 워커는 자체 `InferenceEngine`을 생성하고 비동기 콜백을 등록합니다. 검출, 포즈, 세그멘테이션은 태스크별 팩토리를 사용합니다. 전용 깊이 워커는 `[1, 768, 768, 3]` UINT8 입력과 `[1, 1, 768, 768]` FLOAT 출력 계약을 검증하고, 콜백이 완료될 때까지 입력 버퍼를 유지하며, 레터박스 패딩을 제거하고, 원본 프레임의 기하 구조를 복원한 뒤 OpenCV의 Turbo 컬러 맵을 적용합니다. 각 최신 프레임 큐는 모델이 입력 스트림보다 느릴 때 지연 시간을 쌓는 대신 오래된 프레임을 교체합니다.

<!-- cell: 01af8912 src: 6bab0dfd1b -->
### 5.5 카메라 및 비디오 캡처

`CaptureThread`는 OpenCV `VideoCapture`를 사용합니다. 비디오 입력은 `--no-loop-video`를 설정하지 않으면 반복 재생됩니다. 카메라 입력은 V4L2 백엔드를 사용하고 MJPG 형식을 요청합니다.

<!-- cell: 6ec88213 src: f00de1deb0 -->
### 5.6 Qt 2 x 2 창

`QuadWindow`는 실시간 패널 네 개를 만들고, 추론 워커 네 개와 캡처 스레드를 시작하고, FPS 레이블을 갱신하며, 순서대로 종료를 수행합니다.

<!-- cell: fef60649 src: 2836bd7037 -->
## 6. 애플리케이션 빌드

`build.sh`는 `app/build/`를 만들고, CMake를 Release 모드로 구성한 뒤, `nproc`이 보고하는 모든 CPU 코어로 `make`를 실행합니다. 전체 재빌드에는 `./build.sh --clean`을, DX-RT가 사용자 지정 prefix 아래에 설치된 경우에는 `./build.sh -DCMAKE_PREFIX_PATH=<prefix>`를 사용하세요.

<!-- cell: 6e6f1fe2 src: b5bfeb4275 -->
## 7. 카메라 데모 실행

기본 장치는 `/dev/video0`이며 1280 x 720, 30 FPS를 요청합니다. 다른 카메라나 캡처 설정을 선택하려면 아래 변수를 변경하세요. 실행 셀은 전체 화면 Qt 창을 열고 애플리케이션이 종료될 때까지 블로킹됩니다.

<!-- cell: c267e0de src: db9fcae673 -->
## 8. 비디오 데모 실행

`VIDEO_PATH`를 `assets/videos/` 아래의 파일로 설정하세요. `run_video.sh`는 첫 번째 인자로 비디오 경로를 받습니다.

<!-- cell: 9dee3c0f src: ca736df157 -->
## 9. 조작 방법

- `Esc` 또는 `q`: 종료
- `EXIT` 버튼: 마우스로 종료

<!-- cell: 05fc0113-d263-4d67-a4af-92c174b45b94 src: 6093c888ab -->
## 10. 문제 해결

| 증상 | 표시되는 내용 | 원인 | 해결 |
|---|---|---|---|
| CMake가 Qt5를 찾지 못함 | `Could not find a package configuration file provided by "Qt5"` | `qtbase5-dev`가 설치되지 않음 | `sudo apt install qtbase5-dev` |
| CMake가 DXRT를 찾지 못함 | `Could not find a package configuration file provided by "dxrt"` | DX-RT가 설치되지 않았거나 사용자 지정 prefix 아래에 있음 | DX-Runtime 설치(튜토리얼 01 3절) 또는 `./build.sh -DCMAKE_PREFIX_PATH=<prefix>` 실행 |
| 모델 로드 시 거부됨 | DX-RT 버전 또는 형식 오류 | DXNN이 다른 SDK 릴리스용으로 컴파일됨(이 모델들: DX-COM 2.4.0) | 이 튜토리얼을 검증한 DX-RT 버전(3.4.2)을 사용하거나 사용 중인 DX-COM으로 다시 컴파일 |
| 모델을 열 수 없음 | `assets/models` 아래 파일을 가리키는 로드 오류 | 리소스가 없음 | 3절의 리소스 셀 실행 |
| 카메라를 열 수 없음 | `cannot open /dev/video0` | 카메라가 없거나 권한이 없음 | `v4l2-ctl --list-devices` 확인. 사용자를 `video` 그룹에 추가. `--camera`로 다른 장치 지정 |
| Qt 창이 나타나지 않음 | `could not connect to display` | 그래픽 세션이 없음 | 데스크톱 세션에서 실행 |

<!-- cell: 83f9ac2c src: 18f95b8d4f -->
## 11. 요약

캡처 스레드 하나가 각 프레임을 네 개의 독립적인 비동기 DXRT 파이프라인으로 보냅니다. 태스크별 팩토리가 객체 검출, 포즈 추정, 인스턴스 세그멘테이션을 처리합니다. 전용 깊이 워커는 프레임을 레터박스 처리하고, 768 x 768 깊이 모델을 실행하고, 깊이 맵을 원본 기하 구조로 복원한 뒤 Turbo 컬러 맵을 적용합니다. Qt는 네 가지 실시간 결과를 전체 화면 2 x 2 레이아웃에 표시합니다.

### 11.1 완료 체크리스트

- [ ] 모델 네 개와 샘플 비디오를 다운로드함
- [ ] 캡처 스레드 하나가 비동기 파이프라인 네 개에 프레임을 공급하는 방식을 읽음
- [ ] Qt 애플리케이션을 빌드함
- [ ] 비디오 데모와 카메라 데모를 실행함

> **다음:** 튜토리얼 22로 이동해 CLIP으로 카메라 프레임을 텍스트 쿼리와 매칭해 보세요.
