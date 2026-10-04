<!-- i18n source: notebooks/T24-demo-pidnet-cityscapes/pidnet_cityscapes.ipynb -->
<!-- i18n lang: ko -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: title src: a79a64189a -->
# DEEPX Tutorial 24 - PIDNet Cityscapes C++ 데모

이 노트북은 Qt5 C++ 시맨틱 세그멘테이션 애플리케이션을 설명하고, 빌드하고, 실행합니다. 이 애플리케이션은 DEEPX NPU에서 PIDNet으로 카메라 또는 비디오 프레임을 처리하고, 하단 컨트롤 바에 큰 입력 텐서를 강조해 표시하며, PIDNet argmax 스케일을 실시간으로 조절하는 작은 슬라이더를 제공합니다.

<!-- cell: 062ad116-becd-4b26-a4b3-475220ad077f src: cb4f28770c -->
## 학습 목표

이 튜토리얼을 마치면 다음을 할 수 있습니다.

- PIDNet 로짓이 색상이 입혀진 Cityscapes 오버레이가 되는 과정을 설명한다.
- 데모용 모델과 샘플 비디오를 다운로드한다.
- Qt 애플리케이션을 빌드하고 카메라 또는 비디오로 실행한다.
- Argmax 스케일 슬라이더로 후처리 해상도와 CPU 시간을 맞바꾼다.
- 호스트에 맞는 동시 진행 중(in-flight) 요청 수를 선택한다.

<!-- cell: t24-npu-pattern src: 0b3416de9c -->
## 이 애플리케이션의 NPU 사용 방식

| 항목 | 이 튜토리얼에서 | 배운 곳 |
|---|---|---|
| 엔진 | `InferenceEngine` 하나. 모델은 NPU 전용(CPU 태스크 없음)이므로 DX-RT는 원시 클래스 로짓을 반환하고 애플리케이션이 후처리 전체를 담당 | T06-1 §6 |
| 실행 | 캡처 스레드가 최대 `--inflight`(기본값 4)개의 `RunAsync()` 요청을 큐에 유지. NPU가 다음 프레임을 처리하는 동안 완료 콜백에서 argmax와 렌더링을 수행 | T06-2 §5, T06-3 §3 |
| 요청당 입력 | `[1, 1024, 2048, 3]` UINT8 = 6.3 MB로 이 튜토리얼 시리즈에서 가장 큰 입력. 출력은 `[1, 19, 128, 256]` FLOAT = 2.5 MB. 호스트-장치 간 전송과 장치-호스트 간 전송이 각 요청에서 눈에 띄는 비중을 차지 | T06-3 §4 |
| CPU 측 | 19개 클래스에 대한 픽셀별 argmax는 CPU 작업. Argmax 스케일 슬라이더는 NPU 속도를 그대로 유지하면서 argmax 해상도와 CPU 시간을 맞바꿈 | T06-3 §10 |
| 측정할 것 | 4.1절의 `dxrun -v`(NPU 처리 시간 대 지연 시간)와 슬라이더 위치 및 `--inflight` 값에 따른 GUI FPS 비교 | T06-1 §5 |

<!-- cell: pipeline src: 2dfd7495e4 -->
## 1. 처리 파이프라인

```text
Camera or video frame
        |
        v
Resize to model input and convert BGR to RGB
        |
        v
Asynchronous PIDNet inference -> class logits
        |
        v
Bilinear interpolation at the selected argmax scale
        |
        v
Per-pixel argmax -> Cityscapes color mask -> Qt5 GUI
```

기본 모델은 UINT8 `[1, 1024, 2048, 3]` 입력을 받아 FLOAT32 `[1, 19, 128, 256]` 로짓을 반환합니다.

<!-- cell: prerequisites src: 9341de1df6 -->
## 2. 사전 요구 사항

- 튜토리얼 01 완료: DX-RT가 설치되어 있고 DEEPX NPU가 `/dev/dxrt0`으로 보여야 합니다(`dxcli -s`로 확인). Qt 창을 위해 `cmake`, `g++`, `qtbase5-dev`가 필요합니다.
- 그래픽 데스크톱 세션. 카메라 데모를 위한 V4L2 카메라(선택).
- 다운로드: 약 20 MB 아카이브 1개(모델과 샘플 비디오)를 git이 무시하는 `assets/`에 저장합니다.
- 소요 시간: 약 15분. 빌드는 1분 정도 걸립니다. 패키지가 없을 때 실행하는 아래 `apt` 줄을 제외하면 `sudo`는 필요하지 않습니다.
- DX-RT 3.4.2에서 검증되었습니다. 모델은 DX-COM 2.3.0으로 컴파일되었습니다.

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev qtbase5-dev ffmpeg v4l-utils
```

다음 셀은 SDK를 찾고, 위 요구 사항을 확인한 뒤 상태 표를 출력합니다. `MISSING`으로 표시된 항목에는 그것을 제공하는 단계가 함께 표시됩니다.

<!-- cell: layout-heading src: 572e07788b -->
## 3. 프로젝트 구성

C++ 코드와 실행 스크립트는 `app/` 아래에 있습니다. 다운로드한 모델과 비디오는 `assets/` 아래에 둡니다.

<!-- cell: resources src: 163f686043 -->
## 4. 리소스 다운로드와 확인

이 데모는 공식 [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet) 프로젝트를 참조합니다. 해당 프로젝트의 `PIDNet_S_Cityscapes_val.pt` PyTorch 체크포인트를 DEEPX NPU 추론용 DXNN 형식으로 변환했습니다. 애플리케이션은 변환된 `pidnet_s_cityscapes_val_fixed.dxnn` 모델을 사용합니다.

`get_resources.sh`는 리소스 아카이브를 다운로드하고, 모델과 샘플 비디오를 `assets/`에 압축 해제한 뒤, 압축 해제가 정상적으로 끝나면 다운로드한 아카이브를 삭제합니다.

```text
assets/
├── models/
│   └── pidnet_s_cityscapes_val_fixed.dxnn
└── videos/
    └── pidnet.mp4
```

<!-- cell: resource-download-note src: 670c65bcaf -->
리소스가 없을 때만 다음 셀을 실행하세요. 같은 이름의 기존 파일은 압축 해제 과정에서 교체될 수 있습니다.

<!-- cell: inspect-heading src: bf1c859947 -->
### 4.1 선택: 모델 검사

`dxparse`와 모델을 사용할 수 있으면 다음 셀은 입력·출력 텐서와 `dxrun -v` 기준값을 출력합니다. NPU 처리 시간과 엔드투엔드 지연 시간을 비교해 보세요. 6.3 MB 입력 때문에 호스트-장치 간 전송이 모든 요청에서 눈에 띄는 비중을 차지하며, 비동기 동시 진행 중(in-flight) 큐가 숨기는 것이 바로 이 부분입니다. 사용할 수 없으면 셀은 실패하지 않고 검사를 건너뜁니다.

<!-- cell: scale-heading src: affd9c896d -->
## 5. `pidnet_argmax_scale` 이해

PIDNet은 저해상도 클래스 로짓을 생성합니다. 각 픽셀에서 가장 가능성이 높은 클래스를 선택하기 전에 애플리케이션은 이 로짓을 이중 선형(bilinear) 보간합니다. 스케일은 중간 보간 크기를 제어합니다.

- `0.1`: CPU 비용이 가장 낮고 경계가 거칠어집니다.
- `0.4`: 속도와 디테일의 기본 균형값입니다.
- `1.0`: 전체 프레임 해상도의 argmax이며 CPU 비용이 가장 높습니다.

GUI 슬라이더는 0.10부터 1.00까지 0.05 단위의 값을 사용합니다. 슬라이더는 원자적(atomic) 값에 기록하고, 각 비동기 완료 콜백은 프레임을 처리하기 전에 그 값을 한 번 읽습니다. 이렇게 하면 모든 프레임이 일관된 하나의 스케일을 갖게 됩니다.

<!-- cell: code-heading src: ed5809056c -->
## 6. C++ 코드 가이드

구현은 `app/pidnet_cityscapes.cpp`에 있습니다. 다음 셀은 튜토리얼 20~23과 마찬가지로 소스 파일에서 주요 부분을 직접 읽어 표시합니다.

<!-- cell: code-explanation src: f2781bc990 -->
### 6.1 주요 구현 단계

1. `Options`와 `parse_args`는 카메라 또는 비디오, 모델, 카메라 설정, 초기 스케일, 오버레이 불투명도를 선택합니다.
2. `preprocess_frame`은 BGR을 RGB로 변환하고 프레임을 모델 입력 텐서 크기로 바로 리사이즈합니다.
3. `compute_argmax_mask`는 현재 스케일로 모든 클래스 채널을 보간하고 점수가 가장 높은 클래스를 선택합니다.
4. `render_segmentation`은 클래스 ID를 Cityscapes 색상에 매핑하고 마스크를 입력 프레임과 블렌딩합니다.
5. `MainWindow`는 런타임 입력 형상, 작은 스케일 슬라이더, Exit 버튼이 있는 하단 컨트롤 바 위에 비디오를 배치합니다.
6. 캡처 스레드는 `RunAsync`로 최대 4개의 요청을 제출해 NPU 추론과 CPU 후처리가 겹쳐 실행되도록 합니다.
7. 완료 콜백은 가장 최근에 렌더링된 결과만 1프레임 메일박스에 게시합니다. Qt 타이머가 오래된 GUI 이벤트를 쌓지 않고 이를 소비합니다.
8. 프레임 단위 콜백 병렬 처리가 이미 사용 가능한 CPU 코어를 사용하므로 OpenCV 연산은 내부 스레드 하나만 사용합니다.

<!-- cell: build-heading src: 6910ddc7ad -->
## 7. 빌드

`build.sh`는 Release 빌드를 구성하고 사용 가능한 모든 CPU 코어로 `make`를 실행합니다. 기존 빌드 디렉터리를 먼저 삭제하려면 `--clean`을 사용하세요.

<!-- cell: camera-heading src: 6ce2ff2946 -->
## 8. 카메라로 실행

스크립트는 카메라 인덱스 0을 1280 x 720, 30 FPS로 요청합니다. NPU, 모델, 카메라, 그래픽 세션이 모두 준비되었을 때만 `RUN_CAMERA`를 `True`로 설정하세요.

<!-- cell: video-heading src: 85d23010af -->
## 9. 비디오로 실행

`run_video.sh`는 `assets/videos/pidnet.mp4`를 읽어 반복 재생합니다. 모델과 비디오를 예상 위치에 둔 뒤 `RUN_VIDEO`를 `True`로 설정하세요.

<!-- cell: custom-run src: 3eda8227c7 -->
## 10. 사용자 지정 명령

다른 슬라이더 값으로 시작:

```bash
cd app
./run_video.sh --pidnet-argmax-scale 0.7
```

창 모드로 다른 비디오 사용:

```bash
./build/pidnet_cityscapes --video /path/to/input.mp4 --loop --windowed
```

다른 카메라를 선택하고 마스크 불투명도 낮추기:

```bash
./run_camera.sh --camera 2 --width 1920 --height 1080 --fps 30 --alpha 0.45
```

종료하려면 `Esc`, `Q` 또는 Exit 버튼을 사용하세요. `F`를 누르면 전체 화면 모드가 전환됩니다. 기본값 `--inflight 4`는 4코어 Raspberry Pi 5를 기준으로 한 값이며, 처리량이 더 필요하면 `--inflight 6`을 테스트해 보세요.

<!-- cell: d4dfe950-47a4-4a4e-adb8-c9defa2b6703 src: 5c302aa913 -->
## 11. 모델 출처

이 데모는 공식 [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet) 프로젝트를 참조합니다. `PIDNet_S_Cityscapes_val.pt` PyTorch 체크포인트를 해당 프로젝트에서 가져와 DEEPX NPU 추론용 DXNN 형식으로 변환했습니다. 이 애플리케이션은 변환된 `pidnet_s_cityscapes_val_fixed.dxnn` 모델을 사용합니다.

<!-- cell: 508b6045-6a18-4e39-8844-ad638d890d04 src: 08a0e7b298 -->
## 12. 기타 옵션과 조작

- `--alpha VALUE`: 세그멘테이션 오버레이 불투명도를 0.0~1.0으로 설정합니다. 기본값은 0.6입니다.
- `--inflight N`: 최대 비동기 요청 수를 1~6으로 설정합니다. 기본값은 4코어 Raspberry Pi 5 기준 4입니다.
- `--no-pace`: 비디오의 원본 FPS에 맞추지 않고 가능한 한 빠르게 처리합니다.
- `--windowed`: 1280 x 800 창으로 시작합니다.
- `--full-screen`: 전체 화면 모드로 시작합니다. 스크립트의 기본값입니다.
- `Esc`, `Q` 또는 **Exit** 버튼: 애플리케이션을 종료합니다.
- `F`: 전체 화면 모드를 전환합니다.

전체 옵션 목록은 `./build/pidnet_cityscapes --help`로 확인하세요.

<!-- cell: 81d36d4a-2c61-4c7b-aa91-004b609ab880 src: 12bee04ca1 -->
## 13. 성능 설계

캡처 스레드는 `RunAsync`로 프레임을 제출합니다. NPU가 다른 프레임을 시작하는 동안 완료 콜백이 argmax와 마스크 렌더링을 수행합니다. 상한이 있는 동시 진행 중(in-flight) 제한기가 메모리의 무한 증가를 막습니다. 콜백 간에 이미 병렬 처리가 이루어지므로 OpenCV의 내부 워커 풀은 스레드 하나로 제한하며, 이렇게 하면 4코어 시스템에서 중첩된 과잉 구독(oversubscription)을 피할 수 있습니다.

완료된 프레임은 1프레임 메일박스에 기록됩니다. Qt 스레드는 항상 가장 최근 결과를 가져가므로 지연된 프레임이 GUI 큐에 쌓이지 않습니다. 덕분에 CPU 후처리가 입력 속도를 따라가지 못할 때도 상호작용이 즉각적으로 유지됩니다.

기본값 `--inflight 4`는 Raspberry Pi 5의 CPU 개수에 맞춘 값입니다. CPU 사용률과 온도가 허용 범위에 있다면 `--inflight 6`을 테스트해 DXRT 버퍼 풀 전체를 사용해 보세요.

```bash
./run_video.sh --inflight 6
```

<!-- cell: troubleshooting src: 2b346be6d4 -->
## 14. 문제 해결

| 증상 | 표시되는 내용 | 원인 | 해결 |
|---|---|---|---|
| 모델이 없음 | 리소스 확인에서 `missing`으로 표시 | `get_resources.sh`가 실행되지 않음 | 리소스 셀을 실행하고 `assets/models/` 아래의 파일 이름을 확인 |
| 비디오가 없음 | `pidnet.mp4`를 찾을 수 없음 | 리소스가 불완전함 | `pidnet.mp4`를 `assets/videos/` 아래에 두거나 `run_video.sh`에 다른 경로를 전달 |
| 카메라를 열 수 없음 | `cannot open camera 0` | 카메라가 없거나 권한이 없음 | `v4l2-ctl --list-devices`를 실행하고 다른 인덱스를 시도 |
| 창이 나타나지 않음 | `could not connect to display` | 그래픽 세션이 없음 | 데스크톱 세션에서 실행 |
| CMake가 DXRT를 찾지 못함 | `Could not find a package configuration file provided by "dxrt"` | DX-RT가 설치되지 않았거나 사용자 지정 prefix 아래에 있음 | DX-Runtime 설치(튜토리얼 01 3절) 또는 `./build.sh -DCMAKE_PREFIX_PATH=<prefix>` 실행 |
| 로드 시 모델이 거부됨 | DX-RT 버전 또는 형식 오류 | DXNN이 이전 런타임용으로 DX-COM 2.3.0으로 컴파일됨 | 이 튜토리얼이 검증된 DX-RT 버전(3.4.2)을 사용 |
| 후처리가 느림 | NPU가 유휴 상태인데 FPS가 낮음 | 전체 해상도의 CPU argmax | Argmax 스케일 슬라이더를 0.1 쪽으로 옮기거나 `--inflight`를 낮춤 |

<!-- cell: 06f03620-4bef-40dd-84a5-9fcfb0ea5fc8 src: eae16a4f91 -->
## 15. 요약

PIDNet은 NPU에서 실행되고, 애플리케이션은 CPU에서 클래스 로짓을 색상이 입혀진 Cityscapes 세그멘테이션 오버레이로 변환합니다. Argmax 스케일 슬라이더는 모델이 최고 속도로 계속 실행되는 동안 후처리 해상도와 CPU 시간을 맞바꿉니다.

### 15.1 완료 체크리스트

- [ ] `get_resources.sh`로 모델과 샘플 비디오를 다운로드함
- [ ] `dxparse`로 모델을 검사함
- [ ] `build.sh`로 애플리케이션을 빌드함
- [ ] 애플리케이션을 실행하고 Argmax 스케일을 조절함

> **다음:** 데모 시리즈를 모두 완료했습니다. 전체 튜토리얼 목록은 저장소 README를 참고하세요.
