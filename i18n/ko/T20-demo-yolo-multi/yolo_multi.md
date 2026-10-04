<!-- i18n source: notebooks/T20-demo-yolo-multi/yolo_multi.ipynb -->
<!-- i18n lang: ko -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 7bc75515 src: 7038880fdd -->
# DEEPX Tutorial 20 - YOLO 다중 채널 C++ 데모

이 튜토리얼은 C++ 애플리케이션이 DEEPX NPU로 여러 비디오 채널에서 YOLO 객체 검출을 실행하는 방식을 설명합니다. 또한 리소스를 다운로드하고, 애플리케이션을 빌드하고, 비디오 또는 카메라 데모를 실행하는 방법을 보여 줍니다.

![YOLO 다중 채널 데모](assets/yolo-multi-sc.png)

<!-- cell: 8f86f5a0 src: 6807896955 -->
## 학습 목표

이 튜토리얼을 마치면 다음을 할 수 있습니다.

- C++ 프로젝트 구조를 이해한다.
- JSON 파일의 모델, 입력, 디스플레이 설정을 읽는다.
- 비동기 다중 채널 추론 흐름을 따라간다.
- 애플리케이션을 Release 모드로 빌드한다.
- 비디오 및 카메라 예제를 실행한다.

<!-- cell: t20-npu-pattern src: 80e34194de -->
## 이 애플리케이션의 NPU 사용 방식

| 항목 | 이 튜토리얼에서 | 배운 곳 |
|---|---|---|
| 엔진 | 36개 채널 스레드 전체가 **하나의** `InferenceEngine`을 공유하며, DX-RT가 요청을 큐에 넣고 NPU에서 스케줄링함 | T06-3 §2 |
| 실행 | 모든 채널이 `RunAsync()`를 호출하고, 이미 다음 프레임을 준비하는 동안 콜백으로 박스를 받음 | T06-2 §5 |
| 태스크 그래프 | `YOLOV5S_PPU`는 단일 NPU 태스크임. PPU가 디코딩된 박스를 반환하므로 호스트는 NMS만 실행함. 36개 채널에서는 바로 여기서 PPU의 효과가 나타남(T05-2 §4와 비교) | T05-2 §3 |
| 요청당 입력 | `[1, 512, 512, 3]` UINT8 = 786 KB. 36개 채널을 30 FPS로 돌리면 H2D 트래픽이 850 MB/s에 달하므로 디스플레이 타일이 작음 | T06-3 §4 |
| 측정할 것 | 3.2절의 `dxrun` 기준값(단일 스트림, 디코딩과 디스플레이 없음)과 창 헤더에 표시되는 FPS의 비교 | T06-1 §5 |

<!-- cell: t20-prerequisites src: c5253820d0 -->
## 사전 요구 사항

- 튜토리얼 01 완료: DX-RT가 설치되어 있고 DEEPX NPU가 `/dev/dxrt0`으로 보여야 하며, `cmake`와 `g++`(`build-essential`)이 필요합니다.
- OpenCV 창을 위한 그래픽 데스크톱 세션. 카메라 데모를 위한 `/dev/video0`의 V4L2 카메라(선택).
- 다운로드: 약 134 MB의 아카이브 하나(YOLOv5s PPU 모델과 샘플 영상 45개)를 git이 무시하는 `assets/`에 받습니다.
- 소요 시간: 약 20분. 빌드는 1~2분이 걸립니다. 패키지가 없을 때 2절의 `apt-get` 줄을 제외하면 `sudo`는 필요하지 않습니다.
- DX-RT 3.4.2에서 검증되었습니다. 모델은 DX-COM 2.2.0으로 컴파일되었으며 이 런타임에서 로드됩니다.

다음 셀은 SDK를 찾고, 위 요구 사항을 확인한 뒤 상태 표를 출력합니다. `MISSING`으로 표시된 항목에는 그것을 제공하는 단계가 함께 표시됩니다.

<!-- cell: 8ea9a7fc src: a4b88669a3 -->
## 1. 튜토리얼 파일 찾기

다음 셀은 JupyterLab이 저장소 루트에서 시작되었든 이 노트북 디렉터리에서 시작되었든 튜토리얼 디렉터리를 찾습니다.

<!-- cell: 77ab9dbf src: 9a378a9fc7 -->
### 1.1 프로젝트 구성

```text
T20-demo-yolo-multi/
├── get_resources.sh
├── assets/
│   ├── models/
│   └── videos/
├── app/
│   ├── build.sh
│   ├── run_camera.sh
│   ├── run_video.sh
│   ├── config/
│   ├── include/
│   ├── src/
│   ├── lib/
│   ├── extern/
│   └── sample/
└── yolo_multi.ipynb
```

`extern/`에는 헤더 전용 cxxopts와 RapidJSON 의존성이 들어 있습니다. `sample/`에는 디스플레이 UI가 사용하는 폰트와 이미지가 들어 있습니다.

<!-- cell: 4173a889 src: 76d6941463 -->
## 2. 환경 확인

Debian 또는 Ubuntu 패키지를 설치합니다.

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config ca-certificates curl tar \
    libopencv-dev libopencv-contrib-dev libfreetype-dev ffmpeg v4l-utils \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
```

리소스 아카이브는 약 140 MB입니다(PPU 모델 하나와 샘플 영상 45개).

<!-- cell: e29c2c86 src: 4446f1b209 -->
## 3. 리소스 다운로드 및 확인

두 데모 모두 `assets/models/YOLOV5S_PPU.dxnn`과 `assets/videos/` 아래의 샘플 MP4 파일을 사용합니다. `get_resources.sh`는 아카이브 하나를 다운로드해 `assets/`에 압축을 풀고, 압축 해제가 성공하면 아카이브를 삭제합니다.

<!-- cell: d87f310e src: 670c65bcaf -->
리소스가 없을 때만 다음 셀을 실행하세요. 같은 이름의 기존 파일은 압축 해제 과정에서 덮어써질 수 있습니다.

<!-- cell: 49a25e54 src: 2ba54cf8ba -->
### 3.1 `dxparse`로 모델 검사

`dxparse`로 모델 구조, 태스크 그래프, 텐서, 메모리 사용량, 의존성을 검사합니다.

```bash
dxparse -m assets/models/YOLOV5S_PPU.dxnn -v
```

상세 출력에는 모델 입력이 다음과 같이 표시됩니다.

```text
  Inputs
     -  images, UINT8, [1, 512, 512, 3 ]
```

차원은 배치, 높이, 너비, 채널 순서입니다. 따라서 이 모델은 640 x 640이 아니라 512 x 512 크기의 3채널 입력을 기대합니다.

<!-- cell: 16d5dc87 src: 339ac4ce39 -->
### 3.2 `dxrun`으로 모델 벤치마크

자동 생성된 더미 입력으로 5초짜리 CLI 벤치마크를 실행합니다.

```bash
dxrun -m assets/models/YOLOV5S_PPU.dxnn --use-ort -t 5
```

`dxrun`은 `--single`과 `--fps` 중 어느 것도 지정하지 않으면 기본적으로 벤치마크 모드를 사용합니다. `-t 5`는 측정 시간을 5초로 설정하고, `--use-ort`는 모델 그래프의 CPU 태스크를 위해 ONNX Runtime을 활성화합니다. 보고된 결과는 다중 채널 비디오 디코딩이나 디스플레이 오버헤드가 없는 명령줄 성능 기준값이 됩니다.

<!-- cell: edb6cae2 src: 47f55de289 -->
## 4. 데모 설정 이해

애플리케이션은 모든 런타임 설정을 JSON 파일에서 읽습니다. 주요 필드는 다음과 같습니다.

- `model_path`: `app/` 기준 상대 경로로 지정한 DXNN 모델 경로
- `model_name`: 해당하는 YOLO 후처리 파라미터 선택
- `video_sources`: 입력 경로, 입력 유형, 선택적 저장 프레임 수
- `display_config`: 출력 크기, 그리드, FPS, 레이아웃 설정
- `num_devices`: 헤더에 표시되는 NPU 장치 수

<!-- cell: 3313e802 src: 0facc52619 -->
카메라 설정에는 `/dev/video0` 입력 하나와 비디오 입력 32개가 들어 있습니다. 카메라 강조 표시는 `expand_mode`와 무관하므로, 카메라는 노란색 테두리와 함께 확대된 중앙 영역에 배치됩니다.

비디오 설정에는 6 x 6 그리드의 비디오 입력 36개가 들어 있으며 `expand_mode`는 `false`로 설정되어 있습니다. 애플리케이션의 확장 레이아웃(확대된 중앙 타일 하나)은 소스가 33, 41, 61, 73개일 때만 존재하므로, 33개 소스의 카메라 설정은 이를 사용하고 36개 소스의 비디오 설정은 사용하지 않습니다.

<!-- cell: 44889672 src: 4575c7a014 -->
## 5. C++ 코드 읽기

노트북은 현재 소스 파일에서 선택한 부분을 직접 표시합니다. 이렇게 하면 C++ 코드의 사본을 노트북에 따로 유지하지 않아도 됩니다.

<!-- cell: 940a9b9a src: c69dca4723 -->
### 5.1 빌드 설정

CMake는 C++14 실행 파일 하나를 빌드하고 DXRT, OpenCV, pthread, OpenMP, 그리고 C++14에 필요한 filesystem 라이브러리를 링크합니다. OpenCV FreeType 지원은 사용 가능할 때 사용됩니다. `cmake/dxdemo.function.cmake`의 `add_dxrt_lib()` 헬퍼는 튜토리얼 06-2와 다른 데모가 사용하는 것과 같은 `find_package(dxrt)` 조회를 감싸고 임포트된 타깃 `dxrt::dxrt`를 링크합니다. 그 밖의 분기는 크로스 컴파일과 Windows 빌드에만 사용됩니다.

<!-- cell: cfc22a04 src: f378c3defe -->
### 5.2 설정 파싱

`ApplicationJsonParser()`는 JSON 필드를 검증하고 `AppConfig`를 채웁니다. 파서는 선택적 디스플레이 설정의 기본값도 제공합니다.

<!-- cell: b299367c src: 4f1996b60f -->
### 5.3 다중 채널 추론 파이프라인

```text
JSON configuration
        |
        v
DXRT InferenceEngine (shared)
        |
        +-- ObjectDetection: channel 1 --+
        +-- ObjectDetection: channel 2 --+--> output grid --> OpenCV window
        +-- ObjectDetection: channel N --+
```

`main()`은 DXRT `InferenceEngine` 하나와 입력 소스마다 `ObjectDetection` 객체 하나를 생성합니다. 각 채널은 자체 워커 스레드에서 실행됩니다.

<!-- cell: b25e522d src: 0cf1ca9c29 -->
### 5.4 채널별 비동기 추론

`ObjectDetection::threadFunc()`는 전처리된 입력 프레임을 가져와 `RunAsync()`를 호출합니다. 채널 스레드가 디스플레이 프레임을 준비하는 동안 콜백이 최신 바운딩 박스를 갱신합니다. 워커와 콜백이 공유하는 데이터는 뮤텍스로 보호합니다.

<!-- cell: 966230b7 src: 614c770d7d -->
### 5.5 YOLO 후처리

`Yolo::PostProc()`는 모델 출력 유형에 맞는 디코더를 선택합니다. 이 데모들은 PPU 출력 경로를 사용합니다. 디코딩된 후보는 클래스별로 정렬된 뒤 비최대 억제(NMS)로 전달되어 겹치는 박스가 제거됩니다.

<!-- cell: bf10922b src: 576016cf2b -->
## 6. 애플리케이션 빌드

`build.sh`는 `app/build/`를 만들고, CMake를 Release 모드로 구성한 뒤, `nproc`가 보고하는 모든 CPU 코어로 `make`를 실행합니다. 전체 재빌드가 필요하면 `./build.sh --clean`을 사용하고, DX-RT가 사용자 지정 접두사 아래에 설치되어 있다면 `./build.sh -DCMAKE_PREFIX_PATH=<prefix>`처럼 CMake 옵션을 그대로 전달하세요.

<!-- cell: a3cc6532 src: 1521315f86 -->
## 7. 36채널 비디오 데모 실행

다음 셀은 OpenCV 창을 열고 데모가 종료될 때까지 기다립니다. `Esc` 또는 `q`를 누르거나 `EXIT` 버튼을 클릭하세요.

<!-- cell: 34f09e43 src: 1e56cf0884 -->
## 8. 33채널 카메라 데모 실행

기본 설정은 `/dev/video0`에 V4L2 카메라가 있다고 가정합니다. 애플리케이션은 지원되는 카메라 모드를 자동으로 선택하고 카메라 속도를 최대 30 FPS로 제한합니다.

<!-- cell: cec0fe9e src: 0cee838170 -->
## 9. 조작 방법

- `Esc` 또는 `q`: 종료
- `t`: 검출 박스 표시 또는 숨기기
- `EXIT` 버튼: 마우스로 종료

<!-- cell: 0c8262f3-3b63-458d-a93a-f9bb5f21dc27 src: c73952716d -->
## 10. 문제 해결

| 증상 | 표시되는 내용 | 원인 | 해결 |
|---|---|---|---|
| CMake가 DXRT를 찾지 못함 | `Could not find a package configuration file provided by "dxrt"` | DX-RT가 설치되지 않았거나 사용자 지정 접두사 아래에 있음 | DX-Runtime 설치(튜토리얼 01 3절) 또는 `./build.sh -DCMAKE_PREFIX_PATH=<prefix>` 실행 |
| 모델 또는 비디오를 열 수 없음 | 애플리케이션 로그에 `[ER] ... cannot open` | 리소스가 없음 | 3절의 리소스 셀 실행 |
| 모델과 런타임 버전이 맞지 않음 | `The version of the compiled model is not compatible with the version of the runtime` | DXNN이 다른 SDK 릴리스용으로 컴파일됨(이 모델: DX-COM 2.2.0) | 이 튜토리얼이 검증된 DX-RT 버전(3.4.2)을 사용하거나, 사용 중인 DX-COM으로 모델을 다시 컴파일 |
| 카메라를 열 수 없음 | `cannot open /dev/video0` | 카메라가 없거나 사용자가 `video` 그룹에 속해 있지 않음 | `v4l2-ctl --list-devices` 확인. `sudo usermod -aG video $USER` 실행 후 다시 로그인. 다른 장치를 쓰려면 카메라 JSON 수정 |
| 창이 나타나지 않음 | `cannot open display` | 그래픽 세션이 없음 | 데스크톱 세션에서 실행 |

<!-- cell: 1da13663 src: 0ee7a9d3d1 -->
## 11. 요약

애플리케이션은 DXRT 추론 엔진 하나를 여러 채널 워커가 공유합니다. 각 워커는 비동기 추론, YOLO PPU 후처리, 프레임 렌더링을 수행합니다. 메인 루프는 채널 프레임을 설정 가능한 그리드 하나로 합치고 런타임 정보를 OpenCV 창에 표시합니다.

### 11.1 완료 체크리스트

- [ ] `get_resources.sh`로 모델과 영상을 다운로드함
- [ ] `dxparse`로 PPU 모델을 검사하고 `dxrun`으로 벤치마크함
- [ ] 채널 워커가 `InferenceEngine` 하나를 공유하는 방식을 읽음
- [ ] 애플리케이션을 빌드하고 비디오 데모를 실행함
- [ ] JSON 설정에서 그리드 또는 입력 목록을 변경하고 다시 실행함

> **다음:** 튜토리얼 21로 이동해 객체 검출, 포즈 추정, 세그멘테이션, 깊이 추정을 하나의 Qt 애플리케이션에서 함께 실행해 보세요.
