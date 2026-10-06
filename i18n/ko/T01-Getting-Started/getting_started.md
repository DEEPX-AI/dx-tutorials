<!-- i18n source: notebooks/T01-Getting-Started/getting_started.ipynb -->
<!-- i18n lang: ko -->
<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->

<!-- cell: 7561e296-41b4-44c3-ab4f-4c8582f0593f src: 14f392f9fc -->
# DEEPX Tutorial 01 - DEEPX SDK 설치 방법
이 첫 번째 튜토리얼은 DEEPX SDK를 설치하고 설치가 정상적으로 끝났는지 확인하는 방법을 설명합니다. 환경을 준비하고, SDK를 설치하고, DEEPX NPU 장치(DX-M1, DX-M1M, DX-H1 Quattro)가 시스템에서 올바르게 인식되는지 확인하는 과정을 배웁니다.

## 학습 목표

이 튜토리얼을 마치면 DX-All Suite를 성공적으로 설치하고 DEEPX NPU 장치에서 기본 흐름을 실행할 수 있습니다.

<!-- cell: 389418ec-fa11-4a60-9822-183b9d441952 src: 32aa4d182a -->
## 사전 요구 사항

<!-- cell: 9c427531-ff8d-4e74-93db-8f41d1237db6 src: 34b79239d3 -->
**참고:** 아래 요구 사항은 이 튜토리얼을 위한 것이며 **DEEPX 제품을 사용하기 위한 필수 조건은 아닙니다.**
- OS: Linux (Ubuntu 20.04/22.04/24.04/26.04, Debian 12/13)
- RAM: 8G (DX-Compiler 사용 시 16G)
- 저장 공간: 최소 40 GB
- DEEPX NPU: DX-M1, DX-M1M, DX-H1 Quattro
- CPU: x86_64에서는 DX-Compiler + DX-Runtime, aarch64에서는 DX-Runtime만 지원
- 다운로드: DX-All Suite 클론(약 600 MB), DX-COM 설치 파일(수백 MB), 3.3절의 샘플 모델 24개와 공유 샘플 영상(각 약 1.2 GB)
- 소요 시간: 30~60분. 3.1의 DX-Runtime 빌드만 10~30분이 걸리고 뒤이어 재부팅이 한 번 필요합니다
- `sudo`: 두 설치 스크립트 모두 필요합니다. 각 설치 단계는 먼저 셀 안에서 비밀번호 없는 `sudo`가 되는지 확인하고, 안 되면 터미널용 명령을 출력합니다

이 튜토리얼과 이후 튜토리얼의 터미널 명령은 DX-All Suite 디렉터리를 `<DX_ALL_SUITE_DIR>`로 적습니다. 기본값은 `~/dx-all-suite`이며, 1.1절이 이 호스트에 설정된 값을 출력합니다.

<!-- cell: 2dcfff26-61ce-46f1-b08f-5669e3ead9a8 src: d3372f2340 -->
## DXNN® - DEEPX NPU SDK 소개 (DX-AS: DX-All Suite)

<!-- cell: 52b3f559-cb24-4e41-8286-6802b4d96c4d src: 99488f0f51 -->
DX-AS(DX-All Suite)는 DEEPX 장치로 AI 모델을 컴파일하고 추론할 수 있게 해 주는 프레임워크와 도구의 통합 환경입니다. 개별 도구를 하나씩 설치해 통합 환경을 구성할 수도 있지만, DX-AS는 개별 도구의 버전을 서로 맞춰 최적의 호환성을 유지합니다.

![](https://github.com/DEEPX-AI/dx-all-suite/raw/main/docs/source/resources/dxnn_sdk_illustration.png)

DEEPX SDK는 크게 두 부분으로 나뉩니다.

첫 번째는 AI 모델 컴파일 환경으로, AI 모델을 DEEPX NPU에서 효율적으로 실행할 수 있는 최적화된 형식으로 변환합니다.

두 번째는 AI 모델 런타임 환경으로, 컴파일된 AI 모델을 실제 DEEPX NPU 하드웨어에서 실행해 결과를 생성합니다.

DX-All Suite를 사용하면 두 구성 요소를 따로 관리하지 않고도 한 번에 설정할 수 있습니다.

![](https://github.com/DEEPX-AI/dx-all-suite/raw/main/docs/source/img/dx-as.png)


![](https://github.com/DEEPX-AI/dx-all-suite/raw/main/docs/source/img/DXNN-SDK-Simple-Architecture.png)



이해를 돕기 위한 DX-SDK 소개 YouTube 영상 두 편이 있습니다.
- [Youtube - DEEPX SDK Introduction](https://www.youtube.com/watch?v=Js6Soex0WI4) | 다운로드: [EN](https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/1.DXNN_Introduce_ENG.mp4) · [中文](https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/1.DXNN_Introduce_CH.mp4) · [한국어](https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/1.DXNN_Introduce_KO.mp4)

<img src="assets/youtube-dx-sdk.png" style="max-width: 1000px;">

<!-- cell: 2af4c675-fced-4ac2-ab69-05a2a8abbeb9 src: accfc71fa0 -->
## 1. DX-All Suite 다운로드

<!-- cell: 3b0789e3-bf0c-4304-8545-bed41a13b3be src: c90ee8eafd -->
### 1.1 설치 구성

튜토리얼은 다음 순서로 DX-All Suite를 찾습니다.

1. 환경 변수 `DX_ALL_SUITE_DIR`,
2. `dx-tutorials/config.json` (이 절에서 생성되며 git으로 추적되지 않습니다),
3. `~/dx-all-suite`, `dx-tutorials` 옆의 `dx-all-suite` 폴더 등 일반적인 위치 자동 탐지,
4. 기본 위치 `~/dx-all-suite`.

다음 셀은 찾은 위치와 그 값의 출처를 출력합니다. SDK가 아직 설치되지 않았다면 기본 위치가 `MISSING`으로 표시됩니다. 클론 단계 전에는 정상적인 상태입니다.

<!-- cell: dx-all-suite-directory-heading src: b988a5d1c2 -->
#### 1.1.1 설치 디렉터리

위에 출력된 위치를 그대로 사용하려면 다음 셀을 수정하지 마세요. DX-All Suite를 다른 곳에 설치하려는 경우(또는 이미 설치한 경우)에만 `DX_ALL_SUITE_DIR`을 수정합니다. 셀을 실행하면 선택한 값이 `dx-tutorials/config.json`에 저장되어 다른 모든 튜토리얼이 같은 위치를 사용합니다.

터미널에서 언제든지 설정할 수도 있습니다(`dx-tutorials` 디렉터리에서 실행).

```bash
python tutorial_paths.py --set ~/my/dx-all-suite
```

<!-- cell: dx-all-suite-branch-heading src: 8ee70f7dfe -->
#### 1.1.2 Git 브랜치

다른 SDK 버전이 필요할 때만 아래 브랜치 또는 태그 이름을 변경하세요.
현재 기본값은 릴리즈 태그 `v2.4.2`이며, `main`은 최신 개발 상태입니다.

<!-- cell: 8d2496bb-a3a2-460c-a891-1979b3cfab1d src: 13566ea0b3 -->
### 1.2 DX-All Suite 클론

다음 셀은 설정한 위치에 **DX-All Suite가 없을 때만** 아래 명령을 실행합니다. 디렉터리와 브랜치는 위에서 저장한 설정에서 가져오며, 셀에는 `{DX_ALL_SUITE_DIR}`와 `{DX_ALL_SUITE_BRANCH}`로 표시됩니다.

```bash
git clone --depth 1 --shallow-submodules --recurse-submodules --progress \
    --branch v2.4.2 https://github.com/DEEPX-AI/dx-all-suite.git <DX_ALL_SUITE_DIR>
```

- `--branch`에는 브랜치나 태그를 쓸 수 있습니다. 튜토리얼은 기본적으로 릴리즈 태그 `v2.4.2`를 사용합니다.
- `--depth 1 --shallow-submodules`는 모든 저장소의 최신 커밋만 가져옵니다. 수 GB 대신 약 600 MB를 다운로드하고 디스크에 1.6 GB를 사용합니다.
- `--recurse-submodules`는 서브모듈(`dx-runtime`, `dx-compiler`, `dx-modelzoo` 및 그 안의 저장소)도 함께 체크아웃합니다.
- `--progress`는 출력이 터미널이 아닐 때도 진행 상황을 출력하므로 셀 안에서 진행 상황을 볼 수 있습니다.

네트워크에 따라 1~10분 정도 걸립니다. ■ 버튼으로 셀을 중지했다가 나중에 다시 실행할 수 있습니다. 완료되면 마지막 줄에 `Submodule path ... checked out`이 표시됩니다.

이전 클론이 최상위 체크아웃 이후에 중단되었다면 디렉터리에 `.git`은 있지만 일부 서브모듈이 비어 있습니다. 이 경우 셀은 다시 클론하는 대신 다음 명령을 실행합니다.

```bash
git -C <DX_ALL_SUITE_DIR> submodule update --init --recursive --depth 1 --progress
```

셀 대신 터미널(**File > New > Terminal**)에서 두 명령 중 하나를 직접 실행해도 됩니다. 아래 상태 셀은 두 경우 모두 동일하게 동작합니다.

<!-- cell: b6993bc2-9b2c-48ec-995b-4f1777a8844c src: 034c0df327 -->
### 1.3 현재 상태와 다음 단계

다음 셀은 지금까지 설치된 항목을 보여 주고, `MISSING`으로 표시된 항목에 대해서는 그것을 제공하는 정확한 단계를 알려 줍니다. 이 노트북으로 돌아올 때마다 다시 실행하세요. 반복 실행해도 안전합니다.

<!-- cell: ee6ce747-f96c-417f-917f-9cccd54be3ec src: 64b774a178 -->
### 1.4 DX-All Suite 구성

클론에는 세 개의 Git 서브모듈이 들어 있습니다.
 - `dx-compiler` (**DX-Compiler**): ONNX 모델을 DXNN으로 변환합니다. 2절에서 설치합니다.
 - `dx-runtime` (**DX-Runtime**): 컴파일된 모델을 DEEPX NPU에서 실행합니다. 3절에서 설치합니다.
 - `dx-modelzoo`: 이후 튜토리얼에서 사용하는 사전 컴파일된 모델과 컴파일러 설정입니다.

<!-- cell: 2ab0cf89-7974-40e6-8117-18165d1ef10f src: 3d336a7ce6 -->
DX-Runtime은 다음을 제공합니다.
 - `DX-APP`: 사용자 애플리케이션 관점에서 DX NPU 사용법을 보여 주는 DEEPX 애플리케이션 템플릿
 - `DX-FW`: NPU 펌웨어 바이너리
 - `DX-RT`: DX NPU를 사용한 추론 작업을 최적화해 실행하도록 설계된 프레임워크
 - `DX-NPU Driver`: DX NPU용 Linux 커널 드라이버
 - `DX-STREAM`: DX NPU용 GStreamer 기반 비전 AI 애플리케이션 개발 도구

<!-- cell: 912db406-4064-4ea1-b075-c455abeab01f src: b20e1b138c -->
## 2. DX-Compiler 설치

<!-- cell: c342f780-67f2-4914-a52d-321b5fe55b72 src: 7b700f5a1e -->
자세한 내용은 [DX-All Suite 설치 가이드](https://github.com/DEEPX-AI/dx-all-suite/blob/main/docs/source/02_Setting_Up_Environment.md)를 참고하세요.

DX-Compiler 환경은 사전 빌드된 바이너리를 제공하며 소스 코드는 포함하지 않습니다. 설치 스크립트는 각 모듈을 원격 서버에서 다운로드합니다.

| 명령 | 설치 대상 | `sudo` |
|---|---|---|
| `./dx-compiler/install.sh --target=dx_com` | 전용 Python 환경에 설치되는 DX-COM 컴파일러(`dxcom`) | 이미 설치되어 있어도 `apt-get install python3-dev python3-venv`를 실행 |
| `./dx-compiler/install.sh --target=dx_tron` | DX-TRON 모델 뷰어(`.deb` 패키지) | `apt-get`으로 패키지 설치 |
| `./dx-compiler/install.sh` | 둘 다 | 위 두 가지 모두 |

이 튜토리얼은 2.1에서 DX-COM을 설치합니다. DX-TRON은 선택 사항이며 2.3에서 다룹니다.

<!-- cell: 3bdb9b57-de73-4186-bac9-ac1effa245da src: 54b5b0324e -->
### 2.1 DX-COM 설치

실행할 명령은 다음과 같습니다.

```bash
cd <DX_ALL_SUITE_DIR>
./dx-compiler/install.sh --target=dx_com
```

`run-jupyter-lab.sh`는 튜토리얼의 Python 환경을 활성화하지 않은 채 JupyterLab을 시작하므로, 이 명령은 노트북 셀에서 실행하든 터미널에서 실행하든 시스템 `python3`를 사용합니다. 설치 스크립트는 `dx-compiler/venv-dx-compiler-local`에 자체 환경을 만들며 Jupyter 환경은 건드리지 않습니다.

*어디서* 실행할지를 결정하는 유일한 요소는 `sudo`입니다. 설치 스크립트는 `python3-dev`와 `python3-venv`를 위해 항상 `sudo apt-get`을 호출하는데, 노트북 셀은 비밀번호 프롬프트에 답할 수 없습니다. 그래서 아래 세 셀은 다음을 수행합니다.

1. **확인**: `dxcom`이 이미 설치되어 있는지, 이 환경에서 `sudo`가 비밀번호 없이 동작하는지 확인합니다.
2. **설치**: 위 명령을 실행합니다(확인 셀이 실행해도 된다고 알려 줄 때만 실행하세요).
3. **검증**: `dxcom`이 존재하는지 확인합니다.

다운로드 용량은 수백 MB이며 2~10분 정도 걸립니다. 출력 끝에 녹색 `[HINT] dx_com installation completed!` 블록이 표시되며, 참고용으로 아래에 실어 두었습니다.

<!-- cell: 891804c8-7521-4f34-a453-f8261151d1f4 src: ae07f58239 -->
설치가 정상적으로 끝나면 설치 스크립트가 아래 힌트 블록을 출력합니다. 생성된 가상 환경 `dx-compiler/venv-dx-compiler-local`은 Jupyter 환경과 분리되어 있습니다. 2.2의 셀들은 각 `!` 줄의 셸 안에서 `source venv-dx-compiler-local/bin/activate`로 이 환경을 활성화하므로 Jupyter 환경은 바뀌지 않습니다.

<pre style="color: green">
[HINT] ==================================================================== 
[HINT]   dx_com installation completed! 
[HINT]  
[HINT]   To use dx_com, activate the virtual environment first: 
[HINT]     $ source <path_to_dx_all_suite>/dx-compiler/venv-dx-compiler-local/bin/activate 
[HINT]  
[HINT]   Then you can run dxcom: 
[HINT]     $ dxcom -h 
[HINT]  
[HINT] ==================================================================== 
</pre>

다음으로 DX-Compiler가 설치한 구성 요소를 확인합니다.

<!-- cell: 63c15643-8cfc-4a6f-a6ee-67451ef0b5fb src: c2d63e41ac -->
이제 dx_com 아래의 파일과 폴더를 살펴보겠습니다.

<!-- cell: cffd0d41-fa76-48c4-af02-6a07dcf47057 src: d96c946dd1 -->
### 2.2 DX-Compiler 검증

2.1의 검증 셀에서 `dxcom`이 존재하는 것은 이미 확인했습니다. 이 절에서는 실제로 실행해 봅니다. 먼저 도움말을 출력하고, 그다음 샘플 모델을 실제로 컴파일합니다.

<!-- cell: 1803112e-39d6-4066-8abc-c1c2d61fdfbc src: 0de98f2dda -->
#### 2.2.1 `dxcom` 도움말 확인

`dxcom`은 설치 스크립트가 만든 가상 환경 안에 있으므로, 터미널에서는 호출하기 전에 그 환경을 활성화해야 합니다. 다음 셀은 정확히 아래 명령을 실행합니다.

```bash
cd <DX_ALL_SUITE_DIR>/dx-compiler
source venv-dx-compiler-local/bin/activate
dxcom -h
```

노트북의 `!` 줄은 매번 새 셸을 시작하므로 세 명령을 `&&`로 한 줄에 이어 붙였습니다. 활성화는 그 셸에만 적용되며 Jupyter 환경은 바뀌지 않습니다.

<!-- cell: 9cf00564-56e7-42cf-bbf6-873292fee8da src: 1d50e93220 -->
#### 2.2.2 `MobileNetV2-1.onnx`를 컴파일해 `MobileNetV2-1.dxnn` 생성

다음 셀은 아래 명령을 실행합니다. 컴파일러는 각 단계를 출력하고, `--gen_log` 옵션은 같은 내용을 `compiler.log`에도 기록합니다. 몇 초에서 1분 정도 걸립니다.

```bash
cd <DX_ALL_SUITE_DIR>/dx-compiler
source venv-dx-compiler-local/bin/activate
cd dx_com
dxcom -m sample_models/onnx/MobileNetV2-1.onnx \
      -c sample_models/json/MobileNetV2-1.json \
      -o output/MobileNetV2-1 \
      --gen_log
```

<!-- cell: 8233469e-d738-4c1b-84ae-a451a733a87a src: e5dd23e1a4 -->
`MobileNetV2-1.dxnn` 파일이 생성되었는지 확인합니다.

<!-- cell: 3b83c7c6-a23c-4fd3-ba83-890dfa2b3584 src: adf789f4ad -->
`--gen_log` 옵션은 컴파일 로그를 `compiler.log` 파일에 기록합니다.

저장된 로그 메시지를 확인해 보겠습니다.

<!-- cell: d93f93a6-c5f7-476e-9632-695968e6cc3c src: 2e60c3c391 -->
### 2.3 DX-Tron

**DX-TRON**은 DEEPX 툴체인으로 컴파일한 `.dxnn` 모델 파일을 살펴보는 그래픽 시각화 도구입니다. 

모델 구조를 불러와 확인하고, 색으로 구분된 그래프로 NPU와 CPU 사이의 워크로드 분배를 볼 수 있습니다. 

DX-TRON을 사용하면 모델 실행 흐름을 더 잘 이해하고 전체 성능을 개선할 수 있습니다.

> **참고:** DX-TRON은 SDK와 함께 계속 배포되지만 더 이상 적극적으로 유지보수되지 않습니다. 컴파일된 모델의 리포트로는 `dxcom --export_html`이 만드는 HTML 요약(튜토리얼 05)을 권장합니다. DX-TRON은 `.dxnn` 파일을 빠르게 시각적으로 살펴볼 때 여전히 유용합니다.

<img src="assets/sc-dxtron.png" style="max-width: 600px;">

**주요 기능:**
- **.dxnn 파일 지원**: DEEPX 툴체인으로 컴파일한 모델 파일을 불러와 시각화합니다.
- **워크로드 시각화**: 워크로드 실행을 색으로 구분해 표시합니다.
- 빨간색: NPU에서 실행되는 연산
- 파란색: CPU 또는 호스트에서 실행되는 연산
- **모델 탐색 컨트롤**: 왼쪽 아래의 뒤로 가기 화살표로 언제든 모델 개요 화면으로 돌아갈 수 있습니다.
- **대화형 노드 검사**: 그래프의 노드를 더블 클릭하면 해당 연산의 상세 정보를 볼 수 있습니다.
- 참고: DX-TRON은 DXNN을 지원하기 위해 [netron](https://netron.app/)을 기반으로 개발되었습니다.

<!-- cell: cdc69460-e92c-460b-aaa5-dc0651eba2dc src: ffa894d40f -->
#### 2.3.1 DX-TRON 설치

DX-TRON은 Debian 패키지로 배포되므로 설치 스크립트가 `sudo apt-get`을 사용합니다.

```bash
cd <DX_ALL_SUITE_DIR>
./dx-compiler/install.sh --target=dx_tron
```

2.1과 마찬가지로 세 셀이 이어집니다. **확인**(`dxtron`이 설치되어 있는지, `sudo`가 비밀번호 없이 동작하는지), **설치**(위 명령. 확인 셀이 실행해도 된다고 알려 줄 때만 실행하거나 터미널에서 실행하세요), **검증**입니다.

<!-- cell: 061c3fe9-86fa-4dc3-aa5a-bf48c2365459 src: edd3da7766 -->
#### 2.3.2 DX-TRON 실행

다음 셀은 2.2.2에서 컴파일한 MobileNetV2 모델을 DX-TRON에서 엽니다. 창을 띄우려면 데스크톱 세션이 필요하고 창이 열려 있는 동안 셀이 실행 중 상태로 유지되므로, 셀 첫 줄의 `RUN_DXTRON = True`일 때만 실행됩니다. 기본값 `False`에서는 명령만 출력합니다. 3.4.4의 두 번째 DX-TRON 셀도 같은 플래그를 사용합니다.
> **참고:** 위의 중지 버튼('■')을 클릭해 `dxtron`을 종료할 수 있습니다!

<!-- cell: e8158e1f-0476-46b1-b72a-94006d838653 src: bcd38bb9d2 -->
## 3. DX-Runtime 설치
자세한 내용은 [DX-All Suite 설치 가이드](https://github.com/DEEPX-AI/dx-all-suite/blob/main/docs/source/02_Setting_Up_Environment.md)를 참고하세요.

<!-- cell: 1dee6063-b928-4c16-8100-bbde4cc779db src: 39302d5b84 -->
### 설치 전 플랫폼별 참고 사항 (선택)

**Orange Pi 5 Plus만 해당**
 - Orange Pi 공식 이미지를 사용하는 경우 커널 헤더가 설치되어 있지 않을 수 있습니다. NPU 드라이버를 설치하려면 커널 헤더가 필요합니다.
 - [여기](../../docs/orangepi5p.md)에 링크된 문서를 참고하세요.

**Raspberry Pi 5만 해당**
 - PCIe는 기본적으로 Gen2로 설정되어 있습니다. 대역폭을 높이려면 Gen3으로 설정할 수 있습니다. 
 - [여기](../../docs/raspberrypi5.md)에 링크된 문서를 참고하세요. 

<!-- cell: 5a923f2b-8ed3-454b-b171-6162b87349ed src: d59a17ec7e -->
DX-Runtime 환경에는 각 모듈의 소스 코드가 포함되어 있습니다. 저장소는 `./dx-runtime` 아래에 Git 서브모듈(`dx_rt_npu_linux_driver`, `dx_fw`, `dx_rt`, `dx_app`, `dx_stream`)로 관리됩니다.

DX-Runtime 설치 스크립트의 모든 옵션을 살펴보겠습니다.

<!-- cell: f46c6f59-c240-40ef-beef-c391b6b02ad7 src: 1122f61d55 -->
### 3.1 DX-Runtime 설치

이후 튜토리얼에서 DX-APP과 DX-STREAM을 사용하므로 `--all`로 전부 설치합니다.

```bash
cd <DX_ALL_SUITE_DIR>
./dx-runtime/install.sh --all
```

핵심 런타임(NPU 드라이버, 펌웨어, DX-RT)만 필요하다면 `--all` 대신 `--runtime-only`를 사용하세요.

설치 스크립트는 전 과정에서 `sudo`를 사용합니다. NPU 커널 드라이버를 빌드하고 로드하며, Debian 패키지를 설치하고, `dxrt.service` 데몬을 등록합니다. 질문은 하지 않습니다. 2.1과 마찬가지로 세 셀이 이어집니다. **확인**(설치되어 있는지, `sudo`가 비밀번호 없이 동작하는지), **설치**(위 명령. 확인 셀이 실행해도 된다고 알려 줄 때만 실행하거나 터미널에서 실행하세요), **검증**입니다. 10~30분 정도 걸리며 대부분 DX-RT, DX-APP, DX-STREAM 빌드 시간입니다.

NPU 드라이버 설치 후에는 **재부팅**이 필요하며, 펌웨어 업데이트 후에는 설치 스크립트가 완전한 전원 종료를 권장합니다. 재부팅 후 `./run-jupyter-lab.sh`를 다시 시작하고, 이 노트북을 열어 첫 번째 코드 셀을 실행한 뒤 3.2부터 계속 진행하세요.

<!-- cell: 379be8a6-b287-462b-a2d7-3444b12b2b06 src: e04e936876 -->
### 3.2 설치 검증

설치 디렉터리와 Git 브랜치는 설정 셀에서 이미 `dx-tutorials/config.json`에 저장되었습니다. 이 검증은 저장소, 브랜치, 서브모듈, DX-Compiler 환경, DX-Runtime CLI, NPU 장치 노드를 확인합니다. 그다음 셀들은 PCIe 링크, 커널 드라이버, 서비스를 더 자세히 살펴봅니다.

<!-- cell: 8019bed5-064a-4d2f-b1c6-28e4608c977e src: 68dd0ab6ce -->
### 3.3 권장: 선택한 모델만 다운로드

다음 코드 셀은 `SELECTED_MODELS`를 허용 목록으로 사용해 해당 모델만 다운로드합니다. 이 튜토리얼에서 권장하는 기본 방식입니다. 다른 모델 집합이 필요하면 셀을 실행하기 전에 목록을 수정하세요.

> **경고 — 꼭 필요한 경우가 아니면 전체 다운로드 명령을 사용하지 마세요.**  
> `!cd $DX_ALL_SUITE_DIR/dx-runtime/dx_app && bash setup.sh <<< ""`를 실행하면 대화형 프롬프트에 빈 답변이 입력됩니다. 빈 답변은 모든 카테고리와 모든 모델을 선택하므로 352개 모델(`dx_app` v3.2.2 매니페스트 기준)이 전부 다운로드됩니다. 전체 집합은 많은 네트워크 트래픽과 약 29 GB의 저장 공간이 필요합니다.

권장 셀은 `SELECTED_MODELS`를 `setup.sh --models`에 전달하고 `--no-force`를 사용해 이미 있는 모델 파일은 다시 다운로드하지 않습니다. 나열된 24개 모델은 튜토리얼 02~24에서 사용하는 모델이며, 네트워크에 따라 다운로드에 몇 분 정도 걸립니다. 이후 튜토리얼은 필요한 모델만 지정해 같은 `setup.sh --models ... --no-force`를 호출하므로, 이 셀을 한 번 실행해 두면 그쪽 다운로드 셀은 모두 건너뛰고 몇 초 만에 끝납니다.

<!-- cell: 6549dfd3-3e52-45ca-b3a7-29a0c3a23134 src: de74293afb -->
### 3.4 DX-RT가 제공하는 유용한 도구

<!-- cell: f2fe6a42-d9fc-426f-9ec3-06b64fd9180b src: a05bf69c17 -->
DX-RT가 지원하는 CLI 도구를 살펴보겠습니다.

<!-- cell: dxrt-cli-compatibility-heading src: eda0521192 -->
**현재 CLI 이름과 하위 호환 이름**

DX-RT는 현재 CLI 바이너리를 `/usr/local/bin`에 설치합니다. 이전 명령 이름은 현재 바이너리를 가리키는 심볼릭 링크로 계속 사용할 수 있습니다.

| 현재 이름 | 이전 호환 이름 | 구현 |
|---|---|---|
| `dxcli` | `dxrt-cli` | `dxrt-cli -> dxcli` |
| `dxrun` | `run_model` | `run_model -> dxrun` |
| `dxparse` | `parse_model` | `parse_model -> dxparse` |

따라서 두 이름 모두 같은 바이너리를 실행하고 같은 옵션을 받습니다. 도움말에는 호출에 사용한 이름이 표시될 수 있습니다. `dxbenchmark`, `dxtop`, `dxrtd`는 기존 이름을 유지하며 독립 바이너리로 설치됩니다.

<!-- cell: 5ead4347-03ba-4d2a-8779-4333e6ddbe5d src: 688be0a7e3 -->
#### 3.4.1 dxbenchmark

`dxbenchmark`는 컴파일된 .dxnn 모델을 실행해 기능을 테스트하고 성능을 측정하는 CLI 도구입니다.

<!-- cell: 34c9aeab-e537-4c77-a570-1848c07b9e03 src: 6a7b03d552 -->
웹 브라우저를 열어 `dxbenchmark` 결과를 HTML 형식으로 확인합니다.

<!-- cell: ae8b23c7-05b3-432a-a55b-aa7fd57fcd05 src: 39068fd181 -->
#### 3.4.2 dxtop

`dxtop`은 사용률, 온도, 메모리 같은 DEEPX NPU 지표를 실시간으로 모니터링하는 htop 스타일의 도구입니다.

> dxtop의 출력 형식은 Jupyter Notebook 코드 셀에서 올바르게 표시되지 않습니다. 대신 **별도의 터미널을 열어** dxtop 명령을 실행하세요. </br>
> 별도의 터미널을 여는 방법: `File > New > Terminal`
> 
> ![](assets/open-terminal.png)

<!-- cell: 8cacae4f-7187-4e97-9c82-bbd2d4605826 src: e3b55470f3 -->
> <img src="assets/sc-dxtop.png" style="max-width: 600px;">

<!-- cell: b8469434-2b9d-466a-a5d1-2ba415a678e2 src: 277f9a2a24 -->
#### 3.4.3 dxcli

`dxcli`(호환성을 위해 `dxrt-cli`로도 사용 가능)는 DEEPX DX-RT 장치를 조회·모니터링하고 NPU 펌웨어를 관리합니다.

<!-- cell: fff91477-4f16-4cd2-b0bd-7962bacdfa4d src: af61bd9792 -->
#### 선택: NPU 펌웨어 플래시

DX-Runtime 설치 스크립트가 이 SDK 버전에 맞는 펌웨어를 이미 기록하므로 이 단계는 보통 **필요하지 않습니다**. DEEPX 지원팀이 재플래시를 요청하거나 다른 SDK 브랜치로 전환한 뒤에만 사용하세요. 다음 셀에서 `FLASH_FIRMWARE = True`로 설정하면 활성화되며, `False`이면 셀은 수행할 작업만 출력합니다.

명령은 다음과 같습니다.

```bash
sudo systemctl stop dxrt.service
dxcli -u <DX_ALL_SUITE_DIR>/dx-runtime/dx_fw/m1/latest/mdot2/fw.bin   # DX-H1은 h1/fw.bin
sleep 5
sudo systemctl start dxrt.service
```

<!-- cell: b5e02530-6e11-43f9-a4c6-12587efc5739 src: 3d4e32fb2f -->
#### 3.4.4 dxrun

`dxrun`(호환성을 위해 `run_model`로도 사용 가능)은 컴파일된 .dxnn 모델을 실행해 기능을 테스트하고 성능을 측정하는 CLI 도구입니다.

<!-- cell: d0b1d604-a0cc-4c22-bd42-c0ec1c0ff8a6 src: 3548e92d89 -->
`yolo26-s_640x640.dxnn`은 두 부분(NPU와 CPU)으로 구성됩니다.

 ![](assets/sc-yolo26s.png)

NPU가 지원하지 않는 연산자는 CPU 오프로딩을 통해 ONNX Runtime으로 CPU에서 처리됩니다. 

`dxrun`에 `--use-ort` 옵션을 추가하면 NPU와 CPU 부분을 모두 실행하므로 .dxnn AI 파이프라인 전체의 성능을 측정할 수 있습니다.

`--use-ort` 옵션을 생략하면 NPU 부분만 성능을 측정합니다.

<!-- cell: 96b76758-a19a-41ba-81f8-3b82b74d88dc src: 30f541bc2b -->
#### 3.4.5 dxparse

`dxparse`(호환성을 위해 `parse_model`로도 사용 가능)는 컴파일된 .dxnn 모델을 읽어 구조, 입출력, 메타데이터를 표시하는 명령줄 도구입니다.

<!-- cell: dae58bf1-ed01-4469-8cab-5f1ebc529039 src: dab48f2d09 -->
## 4. 문제 해결

| 증상 | 표시되는 내용 | 원인 | 해결 |
|---|---|---|---|
| 설치 셀이 `sudo`에서 멈춤 | `sudo: a terminal is required to read the password` | 비밀번호 없는 sudo가 설정되지 않아 설치 스크립트가 셀 안에서 비밀번호를 물어볼 수 없음 | 출력된 명령을 터미널(File > New > Terminal)에서 실행한 뒤 검증 셀을 실행 |
| 클론 셀 실패 | `fatal: destination path ... already exists and is not an empty directory` | 대상 위치에 git 체크아웃이 아닌 파일이 있음 | 디렉터리를 옮기거나 1.1에서 다른 위치를 선택한 뒤 셀을 다시 실행 |
| `dxcom`이 계속 MISSING | 상태 표에 `[MISSING] dx_com` | DX-COM 설치 스크립트가 끝까지 실행되지 않음 | 위로 스크롤해 설치 출력을 확인. `[HINT]` 블록이 있어야 함. 2.1을 다시 실행 |
| NPU 장치 없음 | `[MISSING] npu` 또는 `ls: cannot access '/dev/dxrt*'` | 커널 드라이버가 로드되지 않음. 보통 3.1 이후 호스트를 재부팅하지 않은 경우 | 재부팅. 그다음 `lsmod \| grep dxrt`를 확인하고, 장치가 여전히 없으면 호스트 전원을 완전히 껐다가 다시 켬 |
| `dxrt.service`가 실행되지 않음 | `systemctl status dxrt.service`에 `Active: failed` | 드라이버 또는 펌웨어 불일치 | `sudo systemctl restart dxrt.service`. 다시 실패하면 터미널에서 `./dx-runtime/install.sh --runtime-only` 실행 |
| 모델이 다운로드되지 않음 | `setup.sh`가 알 수 없는 모델 이름을 보고 | 이름이 Model Zoo 매니페스트와 다름(대소문자 구분) | `dx_app`에서 `grep name scripts/modelzoo_manifest.json`으로 정확한 이름 확인 |

<!-- cell: 2e5f3528-1de5-45e1-87db-c8018aa6c1d6 src: da34d3824a -->
## 5. 요약

<!-- cell: 7acbc00f-59ec-4386-b048-d447c0507bff src: 077c9eb1db -->
| 항목 | DX-Compiler | DX-Runtime |
|---|---|---|
| 역할 | 모델 컴파일 | 모델 추론 실행 |
| 입력 | ONNX | DXNN |
| 출력 | `.dxnn` | 추론 결과 |
| 시스템 | x86_64 전용 | x86_64, aarch64 |
| 설치 명령 | `./dx-compiler/install.sh --target=dx_com` | `./dx-runtime/install.sh --all` |
| 주요 CLI | `dxcom` (`venv-dx-compiler-local` 안) | `dxcli`, `dxrun`, `dxparse`, `dxbenchmark`, `dxtop` |

### 5.1 완료 체크리스트

- [ ] `config.json`이 DX-All Suite 디렉터리를 가리키고 1.3의 상태 셀에서 모든 항목이 `OK`로 표시됨
- [ ] `dxcom -h`가 실행되고 `MobileNetV2-1.dxnn`이 컴파일됨 (2.2)
- [ ] `dxtron`으로 `.dxnn` 파일을 열 수 있음 (2.3, 선택)
- [ ] `/dev/dxrt0`이 존재하고 `dxrt.service`가 활성 상태이며 `dxcli -s`에 NPU가 표시됨 (3.2)
- [ ] 선택한 모델이 공유 워크스페이스에 다운로드됨 (3.3)
- [ ] `dxrun`과 `dxbenchmark`로 NPU에서 모델을 실행함 (3.4)

> **다음:** 튜토리얼 02로 이동해 여기서 다운로드한 모델로 DX-APP 예제(분류, 검출, 포즈, 세그멘테이션)를 실행해 보세요.
