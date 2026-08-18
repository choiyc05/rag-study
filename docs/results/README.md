# 측정 결과 원자료

`experiments.md`가 **판정과 해석**이라면, 이 폴더는 **그 판정을 뒷받침한 숫자 원본**이다.
손으로 옮겨 적은 표는 재현이 안 되고 무엇보다 **질의별 순위를 잃는다.**
스크립트가 직접 떨군 파일만 여기에 둔다.

| 폴더 | 단계 | 내용 |
|---|---|---|
| [`phase0-embedding/`](phase0-embedding/) | Phase 0-4 | 임베딩 모델 **6종** 비교 → 기준선 확정 |
| [`phase0-prefix-ablation/`](phase0-prefix-ablation/) | Phase 0-4 | **같은 모델, prefix만 제거** — MRR −16% |
| [`phase0-dimension/`](phase0-dimension/) | Phase 0-4 | **같은 모델, 차원만 절단**(MRL) — 1024/768/512/256 |

## 파일 규격

각 폴더는 같은 구성을 갖는다.

| 파일 | 내용 | 왜 남기나 |
|---|---|---|
| `README.md` | 사람이 읽는 리포트 | 판정과 근거 |
| `metrics.json` | 모델별 Hit@k · MRR · 하위 그룹 · 짝지은 검정 | 기계가 읽는 요약 |
| `ranks.csv` | **질의별 정답 순위** (1행 = 질의 1건, 1열 = 모델 1개) | ★ 아래 참고 |

### ★ `ranks.csv`가 이 폴더의 핵심이다

집계값(Hit@5, MRR)만 남기면 **나중 실험을 기준선과 짝지어 비교할 수 없다.**
질의별 순위가 있으면 Phase 0을 다시 돌리지 않고도 R-1~R-6을 같은 질의끼리 대조할 수 있다.

ΔMRR이 0.004처럼 작을 때 **평균 비교로는 실력과 잡음이 구분되지 않는다.**
실제로 Phase 0에서 평균만 봤다면 KURE-v1을 승자로 잘못 기록했을 것이다.

```python
# 이후 실험을 기준선과 짝지어 비교하는 법
import pandas as pd, numpy as np
base = pd.read_csv("docs/results/phase0-embedding/ranks.csv")
new  = pd.read_csv("docs/results/phase1-r1/ranks.csv")        # 예: R-1 결과
m = base.merge(new, on="answer_id", suffixes=("_base", "_new"))
d = 1/m["<새 모델>_new"] - 1/m["<기준 모델>_base"]             # 질의별 역순위 차
print(d.mean(), np.percentile(
    [d.sample(len(d), replace=True).mean() for _ in range(2000)], [2.5, 97.5]))
```

## 재생성

```bash
backend/.venv/Scripts/python.exe scripts/04_evaluate.py --save docs/results/phase0-embedding
```

같은 모델을 조건만 바꿔 비교할 때는 `ranks.csv`의 열 이름에 prefix가 함께 붙어
구분된다 (예: `…-ko ['query: '/'']` vs `…-ko [없음]`).

⚠️ **평가셋(`data/normalized/evalset_colloquial.jsonl`)이 바뀌면 이전 숫자와 비교할 수 없다.**
바꿀 때는 이 폴더를 덮어쓰지 말고 새 폴더를 만들 것.
