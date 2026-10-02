<!-- i18n source: notebooks/T06-DX-Runtime/dx_rt_03_advanced.ipynb -->
<!-- i18n lang: ko -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 46c94efe src: bf00e17c30 -->
# DEEPX Tutorial 06-3 - DX-RT 고급

이 노트북은 동작하는 추론 호출을 관찰 가능하고 재현 가능한 런타임 실험으로 발전시킵니다.

## 학습 목표

이 튜토리얼을 마치면 다음을 할 수 있습니다.

- 장치 선택, NPU 코어 바인딩, ORT 사용 여부, 버퍼 수를 설정하고,
- 통제된 벤치마크로 버퍼 수를 튜닝하고,
- DX-RT 프로파일러 트레이스를 생성하고 시각화하고,
- 작업(job)별 H2D, NPU, D2H, 포맷 핸들러, CPU 태스크 지표를 읽고,
- 변동 계수(CoV)로 불안정한 타이밍을 감지하고,
- 장치 메모리, 사용률, 열 상태를 조회하고,
- 메모리 버퍼에서 DXNN 모델을 로드하고,
- 다중 입력 모델을 위한 이름 기반 입력을 준비하고,
- 런타임 이벤트 핸들러를 등록하고,
- 안정적인 C ABI, 헤더 전용 C++ 래퍼, 런타임 IPC 경계를 설명하고,
- NFH 및 CPU 연산 가속을 C++, Python, `dxrun`을 통해 설명·빌드·활성화하고 A/B 테스트하고,
- 간결한 릴리즈 검증 기록을 생성할 수 있습니다.

생성되는 모든 트레이스, 리포트, 소스 파일, 결과는 <code>&lt;dx-tutorials&gt;/notebooks/T06-DX-Runtime/workspace</code> 아래에 저장됩니다.

<!-- cell: 7bedacd4 src: 215e801d3c -->
> 이 튜토리얼의 3부 중 3부입니다. 코스 맵과 전체 파트 목록은 초급 노트북에 있습니다.

<!-- cell: ef4ee62f-6a13-4f7a-bc9d-be05af55b919 src: 5ddb64a396 -->
## 사전 요구 사항

- 튜토리얼 06-2 완료, 특히 2절(`workspace/.venv-dxrt` 환경을 여기서 재사용합니다).
- DEEPX NPU. 프로파일러와 벤치마크 절은 실제 추론을 실행합니다.
- 선택: 튜토리얼 05-3 8절의 다중 입력 모델. 없으면 해당 실습은 건너뜁니다.
- 읽는 데 약 15분이 걸립니다. 셀 자체는 1~2분이면 실행됩니다.

다음 셀은 SDK를 찾고, 이 요구 사항을 확인한 뒤 상태 표를 출력합니다. `MISSING`으로 표시된 항목에는 그것을 제공하는 단계가 함께 표시됩니다.

<!-- cell: 4c12f0dc src: be70bbc8b4 -->
## 1. 튜토리얼 워크스페이스 초기화

이 노트북은 중급 튜토리얼에서 만든 T06 전용 환경에 그에 맞는 <code>dx_engine</code> 휠이 있다고 가정합니다. 또한 <code>dxbenchmark</code>가 다중 모델 리포트를 보여 줄 수 있도록, 사용 가능한 경우 두 번째 모델을 링크합니다.

<!-- cell: b8287091 src: 431918d739 -->
## 2. 런타임 리소스 제어

<code>InferenceOption</code>은 엔진 인스턴스 하나가 어디서 어떻게 실행될지를 제어합니다.

| 옵션 | 기본값의 의미 | 변경 시점 |
|---|---|---|
| <code>devices</code> | 빈 목록이면 사용 가능한 장치를 사용 | 장치를 격리하거나 엔진을 분산할 때 |
| <code>bound_option</code> | <code>NPU_ALL</code> | 코어 하나 또는 코어 쌍을 예약할 때 |
| <code>use_ort</code> | 빌드에 따라 다름 | 모델에 CPU 태스크가 포함되어 있을 때 |
| <code>buffer_count</code> | 런타임 기본값, 보통 6 | 측정된 큐잉 또는 메모리 동작 때문에 튜닝이 필요할 때 |

장치는 물리적 가속기를 선택합니다. 바인딩 옵션은 선택된 각 장치 안의 코어를 선택합니다. 이 두 수준을 혼동하지 마세요.

<!-- cell: 01c44f7b src: b4e173b1c2 -->
### 2.1 바인딩 안전성

기존 엔진 인스턴스가 이미 개별 NPU 코어를 예약하고 있다면, <code>NPU_ALL</code>로 엔진을 하나 더 만들 때 필요한 코어가 모두 해제될 때까지 블로킹될 수 있습니다. 다중 스레드 및 다중 프로세스 서비스에서는 리소스 소유권을 명시적으로 관리하세요.

| 목표 | 일반적인 시작점 |
|---|---|
| 엔진 하나의 최대 처리량 | 장치 하나 이상, <code>NPU_ALL</code> |
| 독립적인 두 파이프라인 격리 | 서로 다른 장치 또는 겹치지 않는 코어 바인딩 할당 |
| 단일 코어 지연 시간 테스트 재현 | 장치 하나와 코어 하나 |
| 프로세스 간 하드웨어 공유 | 엔진을 만들기 전에 소유권 정의 |

바인딩은 배포 시 결정할 사항입니다. 이 노트북의 실행 실습은 장치 0에서 <code>NPU_ALL</code>로 진행합니다.

<!-- cell: 0391fabe src: 9280b3bc19 -->
## 3. 통제된 실험으로 버퍼 수 튜닝

<img src="assets/dx-rt-observability.svg" style="max-width: 1100px; width: 100%;" alt="DX-RT 관측성 및 튜닝 흐름">

아래 네 명령은 <code>--buffer-count</code>만 다릅니다. 같은 모델, 워밍업 횟수, 실행 시간, 실행 모드를 사용합니다. 처리량과 지연 시간을 비교하세요. 가장 큰 값을 자동으로 선택하지 마세요.

동일한 터미널 명령 패턴:

~~~bash
cd <T06-DX-Runtime>/workspace
dxrun -m models/resnet50_224x224.dxnn \
      --benchmark --time 3 --warmup-runs 5 --buffer-count N
~~~

<!-- cell: 2177a506 src: 4687dde300 -->
필요한 안정적인 처리량에 도달하면서 메모리나 지연 시간 한도를 넘지 않는 가장 작은 값을 선택하세요. 전처리, 후처리, 다른 프로세스가 큐 압력을 바꾸므로 실제 애플리케이션 부하에서 실험을 반복하세요.

<!-- cell: 9c4ffa77 src: 63add5ff96 -->
## 4. 런타임 타임라인 프로파일링

프로파일러는 시간을 입력 포맷 변환, 호스트→장치 전송, NPU 연산, 장치→호스트 전송, 출력 포맷 변환, CPU 태스크 등의 단계로 나눕니다.

다음 명령은 T06 프로파일러 디렉터리 안에 <code>profiler.json</code>을 생성합니다.

~~~bash
cd <T06-DX-Runtime>/workspace/profiler
dxrun -m ../models/resnet50_224x224.dxnn \
      --benchmark --time 5 --warmup-runs 5 --profiler
~~~

<!-- cell: fb34227e src: 4811beb9fe -->
### 4.1 트레이스를 이미지로 변환

SDK의 플롯 명령은 바로 아래에 있습니다. <code>--auto-select</code>는 안정적인 중앙 구간에 초점을 맞추며, 출력은 T06 아래에 유지됩니다.

동일한 터미널 명령:

~~~bash
python <DX_RT_DIR>/tool/profiler/plot.py \
  --input <T06>/workspace/profiler/profiler.json \
  --output <T06>/workspace/profiler/profiler.png \
  --auto-select
~~~

<!-- cell: 0dff6899 src: 9e9a10036d -->
### 4.2 프로파일러 이벤트 해석

| 이벤트 | 측정 대상 | 먼저 확인할 질문 |
|---|---|---|
| Buffer Wait | 사용 가능한 추론 버퍼 대기 | 큐 깊이나 CPU 압력이 너무 높은가? |
| NPU Input Format Handler | 패딩 및 레이아웃 변환 | 포맷 변환이 상당한 비중을 차지하는가? |
| PCIe Write / H2D | 호스트→장치 전송 | 입력 크기나 전송이 한계인가? |
| NPU Core | NPU 연산 | 모델이 전체 시간을 지배하는가? |
| PCIe Read / D2H | 장치→호스트 전송 | 출력이 비정상적으로 큰가? |
| NPU Output Format Handler | 출력 슬라이싱 및 레이아웃 변환 | 변환이 병목인가? |
| CPU Task Queue Wait | CPU 실행 대기 | CPU 파이프라인이 포화 상태인가? |
| cpu_N | CPU 연산자 실행 | 연산량이 많은 CPU 연산자가 지배적인가? |

전체 NPU 태스크는 포맷 처리, 전송, 연산, 출력 처리를 포함합니다. NPU 코어 연산만 따로 보는 것보다 범위가 넓습니다.

<!-- cell: 100ed5cc src: 4831be0e8a -->
## 5. Python API로 작업별 지표 읽기

DX-RT v3.4는 <code>get_job_metrics(job_id)</code>를 추가했습니다. <code>wait(job_id)</code> 직후에 호출하세요. 이전의 성능 데이터 메서드는 더 이상 사용되지 않습니다(deprecated).

아래 코드는 프로파일링을 활성화하고, 비동기 작업 20개를 실행한 뒤, 마지막 작업의 유효한 지표를 출력합니다. 반복된 작업은 5.1절의 CoV 표에 필요한 충분한 샘플도 제공합니다.

<!-- cell: 381e6586 src: 6eb64f8da6 -->
### 5.1 변동 계수(CoV)로 안정성 평가

변동 계수(Coefficient of Variation, CoV)는 표준 편차를 평균으로 나눈 값입니다.

~~~text
CoV (%) = standard deviation / mean × 100
~~~

평균 지속 시간이 서로 다른 단계 간의 상대적인 지터를 비교할 수 있습니다. 낮을수록 안정적이지만, 보편적인 릴리즈 임계값은 없습니다. 제품의 지연 시간 예산과 운영 조건에 따라 임계값을 정의하세요.

| 결과 | 해석 |
|---|---|
| 낮은 평균, 높은 CoV | 대체로 빠르지만 가끔 불안정 |
| 높은 평균, 낮은 CoV | 예측 가능하게 느림 |
| 높은 p99, 보통 수준의 평균 | 꼬리 지연 시간 조사 필요 |
| NPU는 안정, 호스트 시간은 불안정 | 호스트 큐, CPU 부하, 전처리 점검 |

<!-- cell: 9994a301 src: 646fb2cbbd -->
## 6. 장치 상태 모니터링

모니터링 서비스는 공유 상태를 약 1초에 한 번 갱신합니다. 더 빠르게 폴링해도 보통 같은 샘플이 반환됩니다.

다음 셀은 스냅샷 두 개를 기록합니다. <code>is_valid() == False</code>는 모니터링 데이터가 오래되었거나 사용할 수 없다는 뜻으로 간주하세요.

<!-- cell: c0c43234 src: 68ad41aa24 -->
지속적으로 확인하려면 애플리케이션이 실행되는 동안 별도의 터미널에서 <code>dxtop</code>을 실행하세요. 온도, 사용률, 메모리, 스로틀링 증거를 지연 시간 및 처리량과 함께 수집하세요. 식은 시스템에서 짧게 돌린 벤치마크는 운영 환경의 열 동작을 숨길 수 있습니다.

<!-- cell: 91656971 src: cf8de1ebce -->
## 7. 메모리에서 모델 로드

메모리 로딩은 모델이 암호화된 저장소, 패키지, 네트워크 서비스 또는 다른 관리형 데이터 소스에서 올 때 유용합니다. NumPy 배열은 C-연속(C-contiguous)이어야 하며 엔진이 살아 있는 동안 계속 유지되어야 합니다.

<!-- cell: bf637c62 src: 8a0360883c -->
## 8. 다중 입력 모델 계약

다중 입력 모델에서 가장 안전한 인터페이스는 정확한 텐서 이름을 키로 하는 딕셔너리입니다.

~~~python
inputs = {
    "left_image": left_tensor,
    "right_image": right_tensor,
}
outputs = engine.run_multi_input(inputs)
~~~

DX-RT는 순서가 있는 리스트와 하나로 이어 붙인 버퍼도 지원하지만, 이름 기반 입력은 순서 실수를 줄여 줍니다.

이 실습은 T05 고급 8절에서 만든 두 입력 DXNN이 있으면 재사용합니다. 모델을 컴파일하거나 T06 바깥에 파일을 만들지 않습니다. 링크는 `workspace/models/`가 아니라 `workspace/multi_input/` 아래에 두어, 11절의 디렉터리 벤치마크가 단일 입력 분류 모델만 계속 비교하도록 합니다.

<!-- cell: d7f33ae1 src: c7292df319 -->
## 9. 런타임 이벤트와 서비스 통합

<code>RuntimeEventDispatcher</code>는 장치 경고, 오류, 복구 알림, 메모리 이벤트, 스로틀링 이벤트를 한곳에서 처리합니다. 제품용 핸들러는 빠르고 스레드 안전해야 하며, 구조화된 이벤트를 애플리케이션의 로깅 또는 상태 점검 시스템으로 전달해야 합니다.

다음 셀은 핸들러를 등록하고 **합성 튜토리얼 이벤트** 하나를 디스패치해 경로를 검증합니다. 실제 하드웨어 장애를 시뮬레이션하지는 않습니다.

<!-- cell: 068ed27c src: 82e871f553 -->
### 9.1 ABI와 IPC 경계

DX-RT v3.4는 공개 통합 계층과 내부 구현을 분리합니다.

| 계층 | 목적 | 제품 적용 지침 |
|---|---|---|
| 안정적인 C ABI, <code>dxrt_c_api.h</code> | <code>libdxrt.so</code>의 버전 관리되는 C 심볼 | 언어 바인딩과 바이너리 배포에 사용 |
| 헤더 전용 C++14 래퍼, <code>dxrt_cxx_api.h</code> | C ABI 위의 모던 C++ 인터페이스 | 새 C++ 애플리케이션에 권장 |
| 레거시 브리지 헤더 | 기존 <code>dxrt_api.h</code> 소스 호환성 | 기존 제품의 빌드를 유지하되 계획적으로 마이그레이션 |
| 공유 메모리 IPC와 런타임 서비스 | 효율적인 프로세스-런타임 통신 | 애플리케이션 텐서 API가 아닌 내부 전송 계층으로 취급 |

<code>libdxrt.so</code>의 숨겨진 C++ 심볼에 의존하지 마세요. 공개 C/C++ 헤더가 지원되는 통합 경계입니다.

<!-- cell: 1dbfded8 src: 4febd54669 -->
## 10. 선택: CPU 측 가속

이 기능들은 호스트 측 런타임 작업을 가속합니다. DXNN 그래프를 바꾸거나 NPU 연산 레이어를 더 빠르게 만들지는 않습니다. 프로파일러가 해당 병목을 확인한 뒤에만 활성화하세요.

| 기능 | 가속 대상 작업 | x86_64 구현 | aarch64 구현 | 유용한 경우 |
|---|---|---|---|---|
| `NPU_FORMAT_CONVERSION_ACCELERATION` | NPU Format Handler(NFH): NPU 태스크 전후의 전치, 패딩, 슬라이싱, 장치 레이아웃 변환 | Intel IPP | ARM NEON/ASIMD | 입력 또는 출력 포맷 핸들러 시간이 상당할 때 |
| `CPU_OP_ACCELERATION` | CPU 폴백 서브그래프의 ONNX Runtime CPU 연산자 | OpenVINO Execution Provider | XNNPACK Execution Provider | ORT CPU 시간이 Conv나 MatMul 같은 연산량이 많은 연산자에 집중될 때 |

NFH 가속은 애플리케이션 전처리를 대체하지 **않습니다**. 리사이즈, 색 공간 변환, 정규화 등 모델별 작업은 여전히 모델 계약을 따릅니다. CPU 연산 가속은 CPU 연산자를 NPU로 옮기지 **않습니다**. 더 최적화된 CPU 실행 공급자(Execution Provider)를 선택할 뿐입니다.

### 10.1 두 개의 관문: 빌드 지원과 런타임 활성화

두 관문이 모두 열려 있어야 합니다.

| 관문 | 목적 | 기본값 |
|---|---|---|
| 빌드 시 CMake 옵션 | 기능과 해당 플랫폼 라이브러리를 DX-RT에 컴파일해 넣음 | `OFF` |
| 런타임 설정 | 컴파일된 기능을 해당 프로세스에서 활성화 | `OFF` |

빌드 지원이 없으면 C++ 열거형, Python 열거형, 해당 `dxrun` 옵션이 존재하지 않습니다. 런타임 활성화만으로는 빠진 구현을 추가할 수 없습니다.

### 10.2 가속 지원을 포함해 DX-RT 빌드

DX-RT 소스 트리에서 `<DX_RT_DIR>/cmake/dxrt.cfg.cmake`를 열어 다음 두 옵션만 변경합니다.

~~~cmake
option(USE_NPU_FORMAT_CONVERSION_ACCELERATION
       "Accelerate NPU data format conversion (transpose/padding)" ON)
option(USE_CPU_OP_ACCELERATION
       "Accelerate CPU-side ONNX operations" ON)
~~~

그다음 CMake가 필요한 라이브러리를 다시 감지하도록 클린 빌드를 수행합니다.

~~~bash
cd <DX_RT_DIR>
./build.sh --clean
~~~

`CPU_OP_ACCELERATION`은 최적화된 ONNX Runtime 실행 공급자를 선택하므로 ORT가 활성화된 DX-RT 빌드도 필요합니다. 클린 빌드는 플랫폼 의존성을 다운로드하거나 설치할 수 있습니다. 필요한 라이브러리나 플랫폼 요구 사항을 사용할 수 없으면 빌드는 요청된 기능을 비활성화합니다. `ON`이 성공했다고 가정하지 말고 CMake 출력을 확인하세요. 새 런타임을 설치한 뒤에는 애플리케이션이 사용하는 `dx_engine` 휠도 그에 맞게 다시 설치하세요.

<!-- cell: ac100003 src: 7d2752ab1c -->
### 10.3 설치된 런타임이 기능을 노출하는지 확인

Python 휠과 시스템 CLI를 따로 확인하세요. 같은 버전을 보고하더라도 빌드 기능은 다를 수 있습니다. 다음 셀은 읽기 전용입니다. 일관된 배포라면 제품이 사용하는 모든 인터페이스에서 필요한 기능이 노출되어야 합니다.

> **표준 설치에서 출력되는 내용:** 패키징된 `dx_engine` 휠은 두 기능을 모두 `available`로 보고하는 반면, 시스템 `dxrun`은 `No acceleration options`를 출력합니다. 이는 오류가 아니라 예상된 동작입니다. 두 바이너리는 서로 다른 CMake 옵션으로 빌드되었고(10.1절), Python 열거형 멤버는 해당 기능을 포함해 빌드된 휠에만 존재합니다. 이 셀이 드러내려는 불일치가 바로 이것입니다.

<!-- cell: ac100005 src: 324af8acdf -->
### 10.4 애플리케이션에서 기능 활성화

첫 번째 `InferenceEngine`을 생성하기 **전에** 가속을 설정하세요. DXNN에 CPU 태스크가 포함되어 있으면 ORT를 활성화된 상태로 유지하세요.

**C++**

~~~cpp
#include <dxrt/dxrt_cxx_api.h>

int main()
{
    auto& config = dxrt::Configuration::GetInstance();

#ifdef DXRT_NFH_ACCELERATION_AVAILABLE
    config.SetEnable(
        dxrt::Configuration::ITEM::NFH_ACCELERATION, true);
#endif

#ifdef DXRT_CPU_OP_ACCELERATION_AVAILABLE
    config.SetEnable(
        dxrt::Configuration::ITEM::CPU_OP_ACCELERATION, true);
#endif

    dxrt::InferenceOption option;
    option.useORT = true;  // Required only when the graph has CPU tasks.
    dxrt::InferenceEngine engine("model.dxnn", &option);
    // Prepare input and run inference.
}
~~~

전처리기 가드는 설치된 DX-RT 헤더에 가속 기능이 없어도 소스를 빌드할 수 있게 해 줍니다. 제품에서는 대신 기능이 없는 것을 설정 오류로 처리할 수도 있습니다.

**Python**

~~~python
from dx_engine import Configuration, InferenceEngine, InferenceOption

config = Configuration()
required_items = ("NFH_ACCELERATION", "CPU_OP_ACCELERATION")
missing = [name for name in required_items if not hasattr(Configuration.ITEM, name)]
if missing:
    raise RuntimeError(f"DX-RT was built without: {', '.join(missing)}")

config.set_enable(Configuration.ITEM.NFH_ACCELERATION, True)
config.set_enable(Configuration.ITEM.CPU_OP_ACCELERATION, True)

option = InferenceOption()
option.use_ort = True  # Required only when the graph has CPU tasks.
with InferenceEngine("model.dxnn", option) as engine:
    # Prepare input and run inference.
    pass
~~~

Python 휠은 새로 설치한 런타임 버전 및 Python ABI와 일치해야 합니다. 그렇지 않으면 열거형이나 네이티브 확장이 공유 라이브러리와 맞지 않을 수 있습니다.

<!-- cell: ac100006 src: 1c4141f606 -->
### 10.5 `dxrun`으로 테스트

먼저 `dxrun --help`에 `--accel-nfh`와 `--accel-cpu`가 나열되는지 확인하세요. 그다음 CPU 태스크가 포함된 DXNN 모델로 A/B 비교를 실행합니다. 모델, ORT 설정, 실행 시간, 워밍업, 버퍼 수, 장치 바인딩, 시스템 부하를 동일하게 유지하세요.

~~~bash
MODEL=/path/to/model-with-cpu-tasks.dxnn

# 1. Baseline
dxrun -m "$MODEL" --use-ort --benchmark --time 10 --warmup-runs 10 --buffer-count 6

# 2. Accelerate only NPU format conversion
dxrun -m "$MODEL" --use-ort --benchmark --time 10 --warmup-runs 10 --buffer-count 6 --accel-nfh

# 3. Accelerate only ORT CPU operators
dxrun -m "$MODEL" --use-ort --benchmark --time 10 --warmup-runs 10 --buffer-count 6 --accel-cpu

# 4. Enable both features
dxrun -m "$MODEL" --use-ort --benchmark --time 10 --warmup-runs 10 --buffer-count 6 --accel-nfh --accel-cpu
~~~

단계별 증거가 필요하면 더 짧은 진단 실행에 `--profiler`를 추가하세요. 처리량만으로는 NFH, CPU 연산자, 전송, NPU 연산 중 무엇이 바뀌었는지 알 수 없습니다.

| 결과 | 해석 |
|---|---|
| NFH 시간 감소 | 포맷 변환 가속이 동작하고 있음 |
| CPU 태스크 시간 감소 | 최적화된 ORT 실행 공급자가 도움이 되고 있음 |
| FPS 변화 없음 | 다른 단계가 병목이거나 가속된 작업이 너무 작음 |
| 지연 시간 또는 CPU 사용량 악화 | 이 워크로드에서는 기능을 비활성화하고 기준선 유지 |

`CPU_OP_ACCELERATION`은 주로 연산량이 많은 CPU 연산에 유용합니다. Reshape, Transpose, Concat은 대개 메모리 바운드라 개선이 거의 없을 수 있습니다. 두 기능 모두 성능 향상을 보장하지는 않습니다.

### 10.6 관련 옵션: 동적 CPU 스레딩

프로파일러가 비용이 큰 개별 연산자가 아니라 CPU 태스크 큐 압력을 보여 준다면, 동적 CPU 스레딩을 별도로 테스트하세요.

~~~bash
export DXRT_DYNAMIC_CPU_THREAD=ON
dxrun -m "$MODEL" --use-ort --benchmark --time 10 --warmup-runs 10
~~~

이것은 또 하나의 독립적인 A/B 실험입니다. NFH나 CPU 연산 가속을 대체하지 않습니다.

<!-- cell: f43487a0 src: a4a30ab9f0 -->
## 11. 여러 모델 벤치마크

<code>dxbenchmark</code>는 디렉터리에서 DXNN 파일을 찾아 기계가 읽을 수 있는 리포트와 시각적 리포트를 생성합니다. 결과 경로는 T06 안에 있습니다. 실행할 때마다 타임스탬프가 붙은 <code>DXBENCHMARK_&lt;date&gt;.csv/.html/.json</code> 세트가 추가되고 결과 디렉터리에 <code>profiler.json</code>도 기록되므로, 깔끔하게 비교하려면 이전 파일을 삭제하세요. 이 디렉터리에는 1절에서 링크한 단일 입력 모델만 있으며, 8절의 두 입력 모델은 의도적으로 <code>workspace/multi_input/</code>에 둡니다.

동일한 터미널 명령:

~~~bash
cd <T06>/workspace/reports/dxbenchmark
dxbenchmark --dir <T06>/workspace/models \
            --result-path <T06>/workspace/reports/dxbenchmark \
            --time 3 \
            --warmup 3 \
            --sort fps \
            --order desc
~~~

<!-- cell: 669618a2 src: 39da7d0246 -->
## 12. 릴리즈 검증 기록 생성

<img src="assets/dx-rt-release-loop.svg" style="max-width: 1100px; width: 100%;" alt="운영 런타임 검증 루프">

런타임 릴리즈 기록은 정확한 바이너리와 환경을 측정된 증거와 연결해야 합니다. 다음 셀은 T06 안에 간결한 JSON 기록을 기록합니다. 실제 릴리즈 결정 전에 제품 정확도, 엔드투엔드 지연 시간, 호스트 CPU, 메모리, 열 결과를 추가하세요.

<!-- cell: b2f25d6a-8d4f-4316-9883-6a704948cc37 src: 921089640f -->
## 13. 문제 해결

| 증상 | 표시되는 내용 | 원인 | 해결 |
|---|---|---|---|
| DX-RT Python 환경이 없음 | `Complete Intermediate Section 2 first` | 튜토리얼 06-2 2절을 실행하지 않음 | 한 번 실행. 이 노트북은 `workspace/.venv-dxrt`를 재사용 |
| `profiler.json`이 없음 | `workspace/profiler/profiler.json`에 대한 `FileNotFoundError` | `dxrun --profiler`를 실행하지 않음 | 4절을 먼저 실행 |
| 다중 입력 실습을 건너뜀 | `Multi-input execution lab skipped` | 튜토리얼 05-3 8절의 모델이 없음 | 선택 사항. 해당 절을 실행하거나 그대로 진행 |
| Python과 `dxrun`이 보고하는 가속 지원이 다름 | Python에서는 `available`, `dxrun`에서는 `No acceleration options` | 두 바이너리가 서로 다른 옵션으로 빌드됨 | 배포에 사용할 쪽을 기준으로 삼고, 필요한 옵션으로 DX-RT를 다시 빌드(10절) |
| `dxbenchmark --warmup`이 무시되는 것처럼 보임 | 워밍업 실행 횟수가 맞지 않음 | `--warmup`은 실행 횟수가 아니라 초 단위 | `--warmup <seconds>` 사용. `--warmup-runs`는 `dxrun`의 옵션 |

<!-- cell: 703286a9 src: d75aea9e39 -->
## 14. 요약

### 14.1 완료한 운영 워크플로

**계약과 워크로드 고정**  
→ **기준선 측정**  
→ **런타임 단계별 프로파일링**  
→ **리소스를 하나씩 튜닝**  
→ **안정성과 장치 상태 모니터링**  
→ **재현 가능한 릴리즈 증거 저장**

<img src="assets/dx-rt-release-loop.svg" style="max-width: 1000px; width: 100%;" alt="DX-RT 운영 검증 루프">

### 14.2 고급 제어 대시보드

| 결정 | 먼저 볼 증거 | 제어 수단 |
|---|---|---|
| 큐 깊이 | Buffer Wait, 지연 시간, 메모리 | <code>buffer_count</code> |
| 리소스 배치 | 장치/코어 사용률 | <code>devices</code>, <code>bound_option</code> |
| CPU 폴백 | 태스크 그래프와 CPU 태스크 시간 | <code>use_ort</code> |
| 포맷 변환 | 입력/출력 포맷 핸들러 시간 | 컴파일된 경우 NFH 가속 |
| CPU 연산자 비용 | CPU 태스크 유형과 지속 시간 | 컴파일된 경우 CPU 연산 가속 |
| CPU 큐 압력 | 큐 대기와 호스트 CPU | 동적 CPU 스레딩 |
| 서비스 상태 | 런타임 이벤트와 장치 스냅샷 | 이벤트 핸들러와 모니터링 정책 |

### 14.3 생성된 증거

| 산출물 | 위치 | 목적 |
|---|---|---|
| 프로파일러 JSON | <code>workspace/profiler/profiler.json</code> | 원시 이벤트 타임라인 |
| 프로파일러 이미지 | <code>workspace/profiler/profiler*.png</code> | 시각적 병목 점검 |
| 벤치마크 리포트 | <code>workspace/reports/dxbenchmark/DXBENCHMARK_*.csv/.html/.json</code>(실행당 한 세트) 및 <code>profiler.json</code> | 다중 모델 비교 |
| 릴리즈 기록 | <code>workspace/reports/release_validation.json</code> | 추적 가능한 릴리즈 체크리스트 |

### 14.4 완료 체크리스트

- [ ] 장치, 코어 바인딩, ORT, 버퍼 수 설정
- [ ] 고정된 하나의 워크로드에서 여러 버퍼 수 비교
- [ ] 프로파일러 트레이스 생성 및 시각화
- [ ] 작업별 단계 지표 읽기
- [ ] CoV를 타이밍 안정성 신호로 활용
- [ ] 메모리, 사용률, 클럭, 열 상태 조회
- [ ] 메모리에서 DXNN 모델 로드
- [ ] 이름 기반 다중 입력 추론 경로 준비
- [ ] 합성 이벤트로 런타임 이벤트 핸들러 검증
- [ ] 가속 기능의 빌드 지원과 런타임 활성화 구분
- [ ] NFH 및 CPU 연산 가속을 C++, Python, `dxrun` 제어 수단에 대응
- [ ] T06 아래에 벤치마크와 릴리즈 기록 산출물 생성
- [ ] 실제 전처리와 후처리를 포함해 같은 테스트 실행
- [ ] 대표성 있는 레이블 데이터로 태스크 정확도 검증
- [ ] 배포 호스트에서 열 소크(soak) 테스트와 꼬리 지연 시간 테스트 실행

> **기억하세요:** 프로파일러가 지목한 단계를 최적화하세요. 단독으로 측정한 NPU 결과가 빨라졌더라도 정확도, 엔드투엔드 지연 시간, CPU 부하, 메모리, 안정성, 열 동작을 함께 검증하기 전까지는 릴리즈 결과가 아닙니다.

> **다음:** 데모 튜토리얼(Python OCR 파이프라인은 튜토리얼 10, C++ 애플리케이션은 튜토리얼 20~24)로 이어서 런타임을 완전한 애플리케이션에 적용해 보세요.
