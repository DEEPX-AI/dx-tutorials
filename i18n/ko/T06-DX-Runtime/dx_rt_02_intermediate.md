<!-- i18n source: notebooks/T06-DX-Runtime/dx_rt_02_intermediate.ipynb -->
<!-- i18n lang: ko -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 20e090c0 src: bc1aa2e5f2 -->
# DEEPX Tutorial 06-2 - DX-RT 중급

이 노트북은 CLI 검증에서 한 걸음 나아가 DX-RT Python 및 C++ API를 사용한 애플리케이션 수준의 추론을 다룹니다.

## 학습 목표

이 튜토리얼을 마치면 다음을 할 수 있습니다.

- Jupyter 커널에 맞는 사전 빌드된 <code>dx_engine</code> wheel을 설치하고,
- 텐서 메타데이터를 읽어 요구되는 shape과 dtype으로 입력을 할당하고,
- 동기 추론을 구현하고,
- 작업 ID와 <code>wait()</code>를 사용한 비동기 추론을 구현하고,
- 콜백과 버퍼 수명 규칙을 설명하고,
- 배치 API로 독립적인 샘플을 묶어 처리하고,
- <code>InferenceOption.buffer_count</code>를 의도에 맞게 사용하고,
- <code>dxrt_cxx_api.h</code>로 최소한의 C++14 애플리케이션을 빌드하고,
- 지연 시간, 처리량, 소유권 요구 사항에 따라 실행 모드를 선택할 수 있습니다.

생성되는 모든 Python, C++, 빌드, 리포트 파일은 <code>&lt;dx-tutorials&gt;/notebooks/T06-DX-Runtime/workspace</code> 아래에 유지됩니다.

<!-- cell: 550595f9 src: caf8fba665 -->
> 이 튜토리얼의 3부 중 2부입니다. 코스 맵과 전체 파트 목록은 초급 노트북에 있습니다.

<!-- cell: e054aa4b-cc8a-4145-a2f1-3087227ff739 src: d38e092018 -->
## 사전 요구 사항

- 튜토리얼 06-1 완료.
- Python 환경과 C++ 빌드를 위한 `uv`와 `cmake`(그리고 `build-essential`의 `g++`).
- 네트워크 접근: 2절은 `workspace/.venv-dxrt` 아래에 81 MB 크기의 Python 환경을 만듭니다. `dx_engine` wheel은 로컬에 있지만 `uv`가 의존성인 NumPy를 다운로드하므로, 오프라인 호스트에서는 로컬 패키지 인덱스가 필요합니다.
- 소요 시간: 읽는 데 약 10분. 다운로드가 캐시된 뒤에는 셀 실행 자체는 1분 정도 걸립니다.

다음 셀은 SDK를 찾고, 이 요구 사항을 확인하고, 상태 표를 출력합니다. `MISSING`으로 표시된 항목에는 그것을 제공하는 단계가 함께 표시됩니다.

<!-- cell: 1d7d338e src: 98620d01b8 -->
## 1. 튜토리얼 워크스페이스 초기화

<!-- cell: 0478080a src: d6685c848d -->
## 2. T06 워크스페이스에 Python 바인딩 설치

DX-RT Debian 패키지는 <code>/usr/share/libdxrt-bin/python</code> 아래에 버전별 wheel을 제공합니다. Python 3.12용으로 빌드된 wheel은 Python 3.13 커널에서 import할 수 없으므로, 이 노트북은 <code>cpXY</code> 태그가 실행 중인 커널과 일치하는 wheel을 선택합니다.

생성되는 모든 파일을 T06 안에 두기 위해, 이 튜토리얼은 <code>workspace/.venv-dxrt</code> 아래에 작은 전용 환경을 만들고 그 site-packages 디렉터리를 커널의 모듈 검색 경로 끝에 추가합니다. 앞이 아니라 뒤에 추가하는 것이 중요합니다. 이 환경에는 wheel의 의존성으로 NumPy도 들어 있는데, 커널은 자신의 NumPy를 계속 사용해야 하기 때문입니다. 커널에 없는 <code>dx_engine</code>만 로컬 환경에서 찾게 됩니다. 명령은 다음과 동일합니다.

~~~bash
uv venv --python <current-jupyter-python> <T06>/workspace/.venv-dxrt
uv pip install --python <T06>/workspace/.venv-dxrt/bin/python \
  /usr/share/libdxrt-bin/python/dx_engine-<version>-cp<XY>-*.whl
~~~

이 과정은 공유 Jupyter 환경을 수정하지 않으며 DX-RT를 다시 빌드하지도 않습니다. 셀을 다시 실행할 때 로컬 환경에 이미 일치하는 wheel이 있으면 생성과 설치를 건너뜁니다.

<!-- cell: 3d715559 src: 463872b41b -->
## 3. 메모리를 할당하기 전에 텐서 메타데이터 읽기

모델 이름으로 입력 shape이나 dtype을 추측하지 마세요. 엔진에 계약을 물어보세요. DX-RT v3.4는 정의되지 않은 동작을 막기 위해 NumPy dtype을 검증합니다.

아래 헬퍼는 <code>np.empty(...)</code>를 사용한 뒤 버퍼를 채웁니다. 이렇게 하면 실제로 쓰기 가능한 페이지가 할당되므로 DMA 고정(pinning) 중에 copy-on-write 제로 페이지 문제를 피할 수 있습니다.

<!-- cell: b3a1916a src: b039b75719 -->
### 3.1 텐서 소유권 체크리스트

| 확인 항목 | 안전한 방법 |
|---|---|
| Shape | <code>get_input_tensors_info()</code>에서 얻은 값으로 할당 |
| Dtype | 반환된 NumPy dtype을 그대로 사용 |
| 레이아웃 | 컴파일된 모델의 외부 레이아웃과 일치시킴 |
| 연속성 | C-contiguous 배열을 전달 |
| 수명 | 비동기 입력은 완료될 때까지 살려 둠 |
| 출력 범위 | 콜백 출력은 콜백 이후에도 필요할 때만 복사 |

<!-- cell: 62cafeb6 src: 658cf437ee -->
## 4. 동기 추론

<code>run()</code>은 결과가 준비될 때까지 호출 스레드를 블록합니다. 순차 처리나 지연 시간 중심의 애플리케이션에서 가장 명확한 출발점입니다.

<img src="assets/dx-rt-execution-modes.svg" style="max-width: 1100px; width: 100%;" alt="동기, 비동기, 배치 실행 모드">

<!-- cell: 3e183092 src: de13644a11 -->
세 가지 시간 측정값은 각각 다른 질문에 답합니다.

- **호스트 관측 시간**은 Python 호출과 그 주변의 호스트 측 오버헤드를 포함합니다.
- **DX-RT 지연 시간**은 런타임이 보고하는 가장 최근 요청의 지연 시간입니다.
- **NPU 추론 시간**은 NPU 실행만 다루며 종단 간 지연 시간의 일부일 뿐입니다.

NPU 시간을 카메라에서 화면까지의 지연 시간으로 표기하지 마세요.

<!-- cell: 54f7a81f src: bb3c7d9b98 -->
## 5. 작업 ID를 사용한 비동기 추론

<code>run_async()</code>는 작업 ID를 반환합니다. 이후 <code>wait(job_id)</code>가 해당 작업의 출력을 반환합니다. 이를 통해 애플리케이션은 제출, NPU 실행, 기타 작업을 겹쳐서 수행할 수 있습니다.

<img src="assets/dx-rt-buffer-lifecycle.svg" style="max-width: 1100px; width: 100%;" alt="비동기 버퍼 소유권 수명 주기">

<!-- cell: d0e724c4 src: cbe255be5a -->
### 5.1 wait와 콜백 비교

| 완료 처리 방식 | 장점 | 주요 책임 |
|---|---|---|
| <code>run_async()</code> + <code>wait(job_id)</code> | 명시적인 요청/결과 매칭 | 작업 ID를 저장하고 적절한 스레드에서 대기 |
| 등록된 콜백 | 낮은 지연 시간의 완료 처리 | 콜백을 빠르고 스레드 안전하게 유지 |
| <code>run()</code> | 가장 단순한 제어 흐름 | 호출 스레드가 블록되는 것을 감수 |

콜백 출력은 콜백 범위 안에서만 유효합니다. 후속 작업에서 이를 보관해야 한다면 콜백 안에서 필요한 데이터를 복사하고 무거운 작업은 다른 큐로 옮기세요.

<!-- cell: af9ec43d src: 35292b89d9 -->
## 6. 배치 API

DX-RT 배치 실행은 여러 독립적인 샘플을 묶어 런타임 내부에서 비동기로 스케줄링합니다. 컴파일된 모델의 배치 차원을 바꾸지는 **않으며**, DXNN 모델은 여전히 보통 배치 크기 1을 사용합니다. 현재 API는 배치 형식의 입력과 명시적인 출력 버퍼를 `run()`에 전달하는 방식이며, `run_batch()`는 사용 중단된 호환성 래퍼로만 남아 있습니다.

중첩된 Python 형식은 다음과 같습니다.

~~~python
[
    [sample_0_input_0],
    [sample_1_input_0],
    [sample_2_input_0],
]
~~~

<!-- cell: 4b77e8c1 src: 3d13107715 -->
## 7. 버퍼 개수는 파이프라인 용량

<code>InferenceOption.buffer_count</code>는 내부 추론 버퍼의 개수를 제어합니다. 값이 클수록 더 많은 작업을 동시에 진행할 수 있지만, 메모리를 더 사용하고 파이프라인이 가득 찬 뒤에는 더 이상 도움이 되지 않을 수 있습니다.

| 너무 작음 | 균형 | 너무 큼 |
|---|---|---|
| 제출 측이 버퍼를 기다림 | NPU를 계속 바쁘게 유지할 만큼의 작업 | 이득은 적고 메모리만 추가로 사용 |
| 처리량이 낮아질 수 있음 | 안정적인 처리량과 제한된 메모리 | 큐가 길어져 지연 시간이 늘어날 수 있음 |

가장 큰 값이 최선이라고 가정하지 말고 대표적인 값들을 측정하세요. 고급 튜토리얼에서 통제된 버퍼 개수 실험을 수행합니다.

<!-- cell: 485e925a src: f6386e2aa5 -->
## 8. 최소한의 C++14 애플리케이션 빌드

DX-RT v3.4는 안정적인 C ABI와 헤더 전용 C++14 래퍼를 제공합니다. 새로 작성하는 C++ 코드는 다음만 포함해야 합니다.

~~~cpp
#include <dxrt/dxrt_cxx_api.h>
~~~

<code>dxrt_api.h</code>와 <code>dxrt_cxx_api.h</code>를 같은 번역 단위에 함께 포함하지 마세요.

DX-RT는 CMake 패키지 구성 파일(<code>/usr/local/lib/cmake/dxrt/dxrtConfig.cmake</code>)을 설치하므로, 프로젝트에는 <code>find_package(dxrt REQUIRED)</code>와 임포트된 타깃 <code>dxrt::dxrt</code>만 있으면 됩니다. 이 타깃은 include 디렉터리와 <code>pthread</code> 의존성을 함께 제공합니다. 데모 튜토리얼(20~24)은 같은 조회를 작은 CMake 헬퍼로 감싸 두었으며, 아래 줄들은 그 기본 형태입니다.

다음 두 셀은 <code>T06-DX-Runtime/workspace/cpp</code> 아래에만 소스 파일을 생성합니다.

<!-- cell: 78076ec1 src: 8d9e43b968 -->
### 8.1 구성 및 빌드

이 노트북 셀들은 터미널에서 입력할 것과 같은 명령을 실행합니다.

~~~bash
cmake -S <T06>/workspace/cpp -B <T06>/workspace/cpp/build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build <T06>/workspace/cpp/build --parallel
~~~

빌드 디렉터리는 T06 안에 유지됩니다.

<!-- cell: 1d3ac8cb src: d3fa81e622 -->
### 8.2 C++ 애플리케이션 실행

동일한 터미널 명령은 다음과 같습니다.

~~~bash
<T06>/workspace/cpp/build/dxrt_sync \
  <T06>/workspace/models/resnet50_224x224.dxnn
~~~

<!-- cell: 68c92258 src: 34a1319a55 -->
## 9. 실행 방식 선택

| 요구 사항 | 권장 출발점 | 이유 |
|---|---|---|
| 가장 단순한 순차 흐름 | 동기 <code>run()</code> | 명확한 소유권과 오류 처리 |
| 단일 요청 지연 시간 최소화 연구 | 동기 <code>run()</code> | 의도적인 큐 깊이가 없음 |
| 더 높은 스트리밍 처리량 | 비동기 + <code>wait()</code> 또는 콜백 | 독립적인 파이프라인 작업을 겹쳐 수행 |
| 여러 독립 샘플을 한 번에 처리 | 배치 API | 제출과 완료를 묶어 처리 |
| 긴밀한 통합과 낮은 Python 오버헤드 | C++ API | C++14 애플리케이션에서 직접 제어 |
| 기존 Python 파이프라인 | Python API | NumPy 데이터와 빠른 통합 |

올바른 선택은 애플리케이션 전체에 따라 달라집니다. 큐가 너무 깊어지면 추론 FPS가 높아져도 사용자가 체감하는 지연 시간은 오히려 나빠질 수 있습니다.

<!-- cell: 3fd7908b-0ab7-49ad-a0a4-2513fcb1771d src: 04785bad99 -->
## 10. 문제 해결

| 증상 | 표시되는 내용 | 원인 | 해결 |
|---|---|---|---|
| `uv` 또는 `cmake`가 없음 | `[MISSING] uv` 또는 `[MISSING] cmake` | 설치되지 않음 | `curl -LsSf https://astral.sh/uv/install.sh | sh`; `sudo apt install cmake build-essential` |
| 이 Python용 wheel이 없음 | `No dx_engine wheel for cp3XY` | 커널의 Python 버전에 맞는 wheel이 `/usr/share/libdxrt-bin/python` 아래에 없음 | 지원되는 Python 버전으로 튜토리얼 `.venv`를 생성 |
| `import dx_engine` 실패 | `ImportError` 또는 버전 불일치 메시지 | `workspace/.venv-dxrt`의 wheel이 `libdxrt.so`보다 오래됨 | 2절을 다시 실행해 wheel을 재설치 |
| CMake가 DX-RT를 찾지 못함 | `Could not find a package configuration file provided by "dxrt"` | DX-RT가 사용자 지정 prefix에 설치되어 `dxrtConfig.cmake`가 기본 검색 경로에 없음 | `cmake -S ... -B ...` 줄에 `-DCMAKE_PREFIX_PATH=<prefix>`를 추가 |
| 비동기 출력이 이상함 | `wait()` 이후 쓰레기 값 | `wait()`가 반환되기 전에 입력 버퍼가 해제되거나 재사용됨 | `wait()`가 반환될 때까지 입력 배열을 살려 둠 |

<!-- cell: b69fd536 src: 8d0b95c249 -->
## 11. 요약

### 11.1 완료한 API 워크플로

**텐서 메타데이터 읽기**  
→ **정확한 dtype과 shape으로 할당**  
→ **동기 실행**  
→ **비동기 제출 및 대기**  
→ **독립 샘플 묶기**  
→ **같은 흐름을 C++로 빌드**

<img src="assets/dx-rt-execution-modes.svg" style="max-width: 1000px; width: 100%;" alt="DX-RT 실행 모드">

### 11.2 실행 대시보드

| 모드 | 제출 | 완료 | 우선 확인할 지표 |
|---|---|---|---|
| 동기 | 요청 1건 | <code>run()</code> 반환 | 요청 지연 시간 |
| 비동기 + wait | 여러 작업 ID | <code>wait(job_id)</code> | 처리량과 큐 지연 |
| 콜백 | 여러 요청 | 런타임 콜백 | 콜백 비용과 처리량 |
| 배치 | 샘플 그룹 | 출력 그룹 | 유효 샘플/초 |
| C++ | 동일한 런타임 개념 | C++ 텐서 | 종단 간 애플리케이션 비용 |

### 11.3 완료 체크리스트

- [ ] Jupyter Python ABI에 맞는 wheel 설치
- [ ] 런타임 메타데이터로 연속 텐서 할당
- [ ] 동기 호스트, 런타임, NPU 시간 측정
- [ ] <code>wait()</code>까지 비동기 입력 수명 유지
- [ ] 모델 배치 크기를 바꾸지 않고 배치 API 사용
- [ ] <code>buffer_count</code>의 메모리/처리량 트레이드오프 설명
- [ ] C++14 DX-RT 애플리케이션 빌드 및 실행
- [ ] 더미 입력을 제품 전처리로 교체하고 출력 검증

> **기억하세요:** 버퍼 소유권은 정확성의 일부입니다. 성능 튜닝은 shape, dtype, 레이아웃, 수명, 출력 매핑을 검증한 뒤에 합니다.

### 11.4 다음 단계

**고급 튜토리얼**로 이어서 장치와 코어 바인딩, 개별 단계 프로파일링, 장치 상태 모니터링, 메모리 로드 모델과 다중 입력 모델 테스트, 재현 가능한 릴리스 증빙 자료 작성을 진행하세요.
