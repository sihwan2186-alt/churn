# ChurnRadar n8n Docker 실행 가이드

마지막 업데이트: 2026-06-01

## 목적

이 문서는 ChurnRadar 프로젝트를 n8n에서 버튼 한 번으로 재현 실행하기 위한 Docker 구성과 import 명령어를 정리한다.

생성된 n8n workflow JSON:

- `n8n_automation/churnradar_n8n_workflow.json`

이 workflow는 n8n이 직접 Python을 실행하지 않고, 별도 Python runner 컨테이너에 HTTP 요청을 보내는 방식이다. 현재 확인한 n8n 컨테이너는 Alpine 기반이고 Python이 없어서, `catboost`, `xgboost`, `python-pptx` 같은 의존성을 n8n 컨테이너 안에 직접 설치하는 방식은 피했다.

## 구성 파일

| 파일 | 역할 |
| --- | --- |
| `n8n_automation/churnradar_n8n_workflow.json` | n8n에서 Import 가능한 workflow JSON |
| `n8n_automation/churn_runner.py` | 프로젝트 스크립트를 순서대로 실행하는 Python HTTP runner |
| `n8n_automation/requirements.runner.txt` | Docker runner에서 쓰는 Python 패키지 고정 버전 |
| `n8n_automation/Dockerfile.runner` | runner 컨테이너 이미지 정의 |
| `n8n_automation/docker-compose.yml` | runner와 별도 n8n 컨테이너 실행용 compose |
| `.dockerignore` | runner 이미지 빌드 때 `.venv`, `processed/` 같은 대량 파일 제외 |

`docker-compose.yml`은 프로젝트 루트를 `/workspace`로 마운트하고, `n8n_automation/churn_runner.py`도 `/runner/churn_runner.py`로 별도 마운트한다. 그래서 runner 코드만 고친 경우에는 이미지 전체 재빌드보다 컨테이너 재생성이 빠르다.

## workflow가 실행하는 순서

1. runner 상태 확인: `GET /health`
2. 전체 재현 실행: `POST /run/full-reproduction`
3. 최종 파일 요약 수집: `GET /summary`

`/run/full-reproduction` 안에 Phase 8 통계 검정, 발표 이미지 생성, 요약 17장 PPT 생성이 이미 포함되어 있다. 그래서 n8n workflow에서는 중복 실행을 제거했다. 단독 재실행이 필요할 때만 아래의 runner 직접 테스트 명령어로 `/run/statistical-validation` 또는 `/run/ppt`를 호출하면 된다. 상세 36장 PPT는 현재 `make_detailed_ppt.py`로 별도 생성한다.

전체 재현 실행에는 아래 스크립트가 순서대로 포함된다.

```powershell
preprocess_churn.py
phase_3b_differentiation_experiments.py
phase_4_cross_validation.py
phase_5a_interpretability.py
phase_5b_business_impact.py
phase_6_extended_case_studies.py
phase_7_next_experiments.py
phase_8_statistical_validation.py
make_presentation_assets.py
make_final_ppt.py
```

## 추천 방식: 현재 실행 중인 n8n에 import

현재 Docker에서 `edurisk-n8n` 컨테이너가 `http://localhost:8081`로 실행 중인 상태를 기준으로 한다.

### 1. Python runner 컨테이너 실행

프로젝트 루트에서 실행한다.

기본 API 키는 로컬 재현용으로 `churn-radar-secret-2026`을 사용한다. 바꾸고 싶다면 runner 실행 전에 아래처럼 환경 변수를 지정한다.

```powershell
$env:CHURN_RUNNER_API_KEY = "원하는-긴-키"
```

```powershell
docker compose -f .\n8n_automation\docker-compose.yml up -d --build churn-runner
```

runner 상태 확인:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

runner 코드만 수정한 뒤 빠르게 반영:

```powershell
docker compose -f .\n8n_automation\docker-compose.yml up -d --force-recreate churn-runner
```

### 2. workflow JSON을 n8n 컨테이너로 복사

```powershell
docker cp .\n8n_automation\churnradar_n8n_workflow.json edurisk-n8n:/tmp/churnradar_n8n_workflow.json
```

### 3. n8n CLI로 workflow import

n8n 공식 CLI import 방식은 `n8n import:workflow --input=<file>`이다.

```powershell
docker exec -u node -it edurisk-n8n n8n import:workflow --input=/tmp/churnradar_n8n_workflow.json
```

같은 workflow ID로 다시 import하면 기존 `ChurnRadar Docker Reproduction Pipeline` workflow가 갱신될 수 있다. 이 프로젝트에서는 같은 자동화 파일을 최신 상태로 유지하기 위한 의도된 동작이다.

### 4. n8n UI에서 실행

브라우저에서 접속한다.

```powershell
start http://localhost:8081
```

n8n UI에서 `ChurnRadar Docker Reproduction Pipeline` workflow를 열고 `Execute Workflow`를 누른다.

## 대안 방식: ChurnRadar 전용 n8n까지 같이 실행

기존 n8n을 건드리지 않고 별도 n8n 컨테이너를 쓰려면 아래 명령어를 사용한다. 이 compose의 n8n은 충돌을 피하려고 `8082` 포트를 사용한다.

```powershell
docker compose -f .\n8n_automation\docker-compose.yml up -d --build
```

workflow import:

```powershell
docker exec -u node -it churnradar-n8n n8n import:workflow --input=/files/import/churnradar_n8n_workflow.json
```

접속:

```powershell
start http://localhost:8082
```

## runner만 직접 테스트하는 명령어

전체 재현 실행:

```powershell
$headers = @{"X-API-KEY"="churn-radar-secret-2026"}
Invoke-RestMethod -Method Post http://localhost:8000/run/full-reproduction -Headers $headers -Body '{}' -ContentType 'application/json'
```

Phase 8 통계 검정만 실행:

```powershell
$headers = @{"X-API-KEY"="churn-radar-secret-2026"}
Invoke-RestMethod -Method Post http://localhost:8000/run/statistical-validation -Headers $headers -Body '{}' -ContentType 'application/json'
```

PPT만 재생성:

```powershell
$headers = @{"X-API-KEY"="churn-radar-secret-2026"}
Invoke-RestMethod -Method Post http://localhost:8000/run/ppt -Headers $headers -Body '{}' -ContentType 'application/json'
```

드리프트 점검만 실행:

```powershell
$headers = @{"X-API-KEY"="churn-radar-secret-2026"}
Invoke-RestMethod -Method Post http://localhost:8000/run/monitor-drift -Headers $headers -Body '{}' -ContentType 'application/json'
```

최종 요약 확인:

```powershell
$headers = @{"X-API-KEY"="churn-radar-secret-2026"}
Invoke-RestMethod http://localhost:8000/summary -Headers $headers
```

## 현재 검증 결과

아래 항목은 2026-05-28에 현재 PC에서 확인했다.

- n8n workflow JSON 문법 검사: 통과
- n8n 2.12.3 임시 컨테이너 CLI import: 통과
- Docker compose 설정 검사: 통과
- `churnradar-runner` build 및 실행: 통과
- 기존 n8n 컨테이너 `edurisk-n8n`에서 `http://host.docker.internal:8000/health` 접근: 통과
- 기존 n8n 컨테이너 `edurisk-n8n`에 workflow CLI import: 통과
- runner의 `POST /run/ppt`: 통과, 요약 PPT 17장 확인
- n8n workflow 중복 단계 제거: `Health -> Full Reproduction -> Summary` 구조로 단순화
- runner의 `GET /summary`: 통과, `.venv` 제외 파일 카운트 정상 반환
- API 키 인증 추가 후 workflow JSON의 HTTP Request 노드에 `X-API-KEY` 헤더 반영

## 중지와 로그 확인

runner 로그:

```powershell
docker logs -f churnradar-runner
```

runner 중지:

```powershell
docker compose -f .\n8n_automation\docker-compose.yml stop churn-runner
```

전용 n8n까지 같이 내리기:

```powershell
docker compose -f .\n8n_automation\docker-compose.yml down
```

## 주의 사항

- workflow JSON의 runner URL은 기본적으로 `http://host.docker.internal:8000`으로 설정되어 있다. 이는 Windows/Mac의 Docker Desktop 환경용이다.
- **리눅스 서버 환경:** 리눅스에서 실행할 경우 `docker-compose.yml` 내에서 동일한 `network`를 지정하고, URL을 `http://churn-runner:8000`과 같이 서비스 이름으로 변경해야 한다.

  ```yaml
  # 리눅스용 설정 예시
  networks:
    churn-net:
      driver: bridge
  services:
    churn-runner:
      networks:
        - churn-net
    n8n:
      networks:
        - churn-net
  ```

- runner는 프로젝트 루트를 `/workspace`로 마운트한다. 따라서 실행 결과는 현재 프로젝트 폴더의 `processed/`, `presentation_assets/`, `ChurnRadar_Final_Presentation.pptx`에 바로 반영된다. `ChurnRadar_Detailed_Presentation.pptx`는 별도 상세 발표 생성 스크립트 산출물이다.
- **데이터 검증:** 실행 전 `python -m pytest -q tests/test_data_integrity.py`를 수행하여 입력 데이터의 스키마가 일치하는지 반드시 확인해야 한다.
- **API 키:** runner의 `/health`는 상태 확인용으로 열어두었고, `/summary` 및 `/run/*` endpoint는 `X-API-KEY` 헤더가 필요하다.
- 전체 재현은 시간이 걸릴 수 있다. workflow의 HTTP timeout은 긴 실행을 고려해 최대 1시간으로 설정했다.
- Import 후 workflow는 비활성 상태(`active: false`)로 들어간다. 수동 실행용이므로 n8n UI에서 직접 실행하면 된다.

## 참고한 n8n 공식 문서

- [Export and import workflows](https://docs.n8n.io/workflows/export-import/)
- [CLI import command](https://docs.n8n.io/hosting/cli-commands/)
- [Docker installation](https://docs.n8n.io/hosting/installation/docker/)
