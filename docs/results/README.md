# 측정 결과 원자료

`experiments.md`가 **판정과 해석**이라면, 이 폴더는 **그 판정을 뒷받침한 숫자 원본**이다.
손으로 옮겨 적은 표는 재현이 안 되고 무엇보다 **질의별 순위를 잃는다.**
스크립트가 직접 떨군 파일만 여기에 둔다.

| 폴더 | 단계 | 내용 |
|---|---|---|
| [`phase0-embedding/`](phase0-embedding/) | Phase 0-4 | 임베딩 모델 **6종** 비교 → 기준선 확정 |
| [`phase0-prefix-ablation/`](phase0-prefix-ablation/) | Phase 0-4 | **같은 모델, prefix만 제거** — MRR −16% |
| [`phase0-dimension/`](phase0-dimension/) | Phase 0-4 | **같은 모델, 차원만 절단**(MRL) — 1024/768/512/256 |
| [`phase1-r2a/`](phase1-r2a/) | Phase 1-b (R-2a) | **리랭커** 3종 병행 — 지연 실측 + 모델 격차 87% 흡수 |

## 파일 규격

각 폴더는 같은 구성을 갖는다.

| 파일 | 내용 | 왜 남기나 |
|---|---|---|
| `README.md` | 사람이 읽는 리포트 | 판정과 근거 |
| `metrics.json` | arm별 Hit@k · MRR · 하위 그룹 · 짝지은 검정 (`schema_version: 1`) | 기계가 읽는 요약 |
| `ranks.csv` | **질의별 정답 순위** (1행 = 질의 1건, **1열 = arm 1개**) | ★ 아래 참고 |

각 폴더의 README 끝에 **재현 명령**이 있다. 손으로 조립한 결과는 여기 두지 않는다.

### 단위는 모델이 아니라 **arm**이다

한 번의 실행이 arm 여러 개를 만든다. 같은 모델을 차원만 잘라서(`--truncate-dim 1024,768`),
또는 같은 모델을 조건만 바꿔서(prefix 있는 것/없는 것). **그래서 모델명은 키가 될 수 없다.**

| | 키 | 예 |
|---|---|---|
| `metrics.json` | `arms[].arm_id` | `E-7`, `E-7@768d`, `R-2a-A` |
| `ranks.csv` | **열 이름 = `arm_id`** | `answer_id,department,orig_is_multi,E-6,E-7,…` |
| `experiments.md` | 표의 ID | 같은 문자열을 쓴다 |

`arm_id`가 실행 조합과 무관하게 고정이라 **다른 실행의 `ranks.csv`와 `answer_id`로
join할 수 있다.** 겹치는 `arm_id`가 생기면 스크립트가 멈춘다 — 같은 이름의 열이 둘이면
join할 때 조용히 한쪽만 남기 때문이다.

### `metrics.json`에서 사람이 쓰는 칸 / 스크립트가 쓰는 칸

`title` · `question` · `verdict`만 사람이 쓴다. **재실행해도 이 세 칸은 보존된다** —
숫자를 다시 뽑았다고 해석이 사라지면 안 되기 때문이다. 나머지는 전부 스크립트가 쓴다.

⚠️ `latency_ms`의 `null`은 **"안 쟀다"는 뜻이다. 0이 아니다.** Phase 0은 지연을 재지
않았고, Phase 1 R-2(리랭커)에서 채운다.

### Δ는 `paired_tests`에만 있다

arm 안에 "기준선 대비 Δ"를 복제하지 않는다. **같은 숫자가 두 곳에 있으면 어긋났을 때
아무도 모른다.** 표의 Δ 컬럼은 `paired_tests`에서 유도한다.

`paired_tests[].base`는 **다른 실행을 지목할 수 있다** — Phase 1은 전부 이 형태다:

```bash
# R-1의 A arm을 Phase 0의 B-0(E-7)과 짝지어 비교
... 04_evaluate.py --baseline phase0-embedding:E-7 --save docs/results/phase1-r1
```

⚠️ `--baseline`을 생략하면 **이번 실행의 MRR 1위**가 기준이 된다. Phase 0은 그걸로
충분했지만 Phase 1은 "자기 모델의 B-0 대비"라 **반드시 명시해야 한다.**

### ★ `ranks.csv`가 이 폴더의 핵심이다

집계값(Hit@5, MRR)만 남기면 **나중 실험을 기준선과 짝지어 비교할 수 없다.**
질의별 순위가 있으면 Phase 0을 다시 돌리지 않고도 R-1~R-6을 같은 질의끼리 대조할 수 있다.

ΔMRR이 0.004처럼 작을 때 **평균 비교로는 실력과 잡음이 구분되지 않는다.**
실제로 Phase 0에서 평균만 봤다면 KURE-v1을 승자로 잘못 기록했을 것이다.

**보통은 손으로 할 필요가 없다** — `04_evaluate.py --baseline phase0-embedding:E-7`이
이걸 그대로 한다. 아래는 저장된 결과끼리 다시 볼 때:

```python
# 저장된 두 실행을 짝지어 비교하는 법 (열 이름 = arm_id)
import pandas as pd, numpy as np
base = pd.read_csv("docs/results/phase0-embedding/ranks.csv")[["answer_id", "E-7"]]
new  = pd.read_csv("docs/results/phase1-r1/ranks.csv")[["answer_id", "R-1-A"]]
m = base.merge(new, on="answer_id")
d = 1/m["R-1-A"] - 1/m["E-7"]                                 # 질의별 역순위 차
print(d.mean(), np.percentile(
    [d.sample(len(d), replace=True).mean() for _ in range(2000)], [2.5, 97.5]))
```

⚠️ 부트스트랩은 **비교마다 시드를 다시 잡는다**(`04_evaluate.py`). 하나의 난수열을
이어 쓰면 **arm을 하나 추가했을 뿐인데 기존 arm의 95% 구간이 바뀐다.**

## 재생성

```bash
backend/.venv/Scripts/python.exe scripts/04_evaluate.py --save docs/results/phase0-embedding
```

같은 모델을 조건만 바꿔 비교할 때는 `ranks.csv`의 열 이름에 prefix가 함께 붙어
구분된다 (예: `…-ko ['query: '/'']` vs `…-ko [없음]`).

⚠️ **평가셋(`data/normalized/evalset_colloquial.jsonl`)이 바뀌면 이전 숫자와 비교할 수 없다.**
바꿀 때는 이 폴더를 덮어쓰지 말고 새 폴더를 만들 것.
