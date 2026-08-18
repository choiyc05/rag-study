# Phase 0-4 부록 — prefix 제거 대조군

**측정일** 2026-08-18 · **판정은** [../../experiments.md](../../experiments.md) Phase 0 "판정 3"

**변수 하나: `query: ` prefix의 유무.** 모델·데이터·설정이 전부 동일하다.

## 왜 쟀나

2차 후보(arctic 계열)는 질의에 `query: `를 붙이고 1차 후보(bge-m3 등)는 안 붙인다.
그래서 "arctic이 bge-m3보다 +0.073"이라는 값에 **모델 차이와 prefix 유무가 섞여 있었다.**
같은 모델을 prefix만 빼고 다시 뽑아 그 몫을 갈라냈다.

## 결과

| 구성 | Hit@1 | Hit@5 | Hit@20 | MRR |
|---|---:|---:|---:|---:|
| `arctic-l-v2.0-ko` + `query: ` (**올바른 사용**) | 0.376 | 0.572 | 0.704 | **0.468** |
| `BAAI/bge-m3` (참고) | 0.295 | 0.511 | 0.664 | 0.395 |
| `arctic-l-v2.0-ko` **prefix 없음** | 0.302 | 0.488 | 0.631 | **0.391** |

짝지은 부트스트랩 (기준 = prefix 있는 쪽):

| 상대 | ΔMRR | 95% 구간 | 이김 | 짐 | 무 | 판정 |
|---|---:|---:|---:|---:|---:|---|
| 같은 모델, prefix 없음 | −0.0772 | [−0.0852, −0.0692] | 225 | 1,342 | 832 | 기준이 우세 |

## 읽는 법

**① prefix 하나가 MRR −0.077 (상대 −16%).**
모델을 바꾼 것보다 큰 폭이다 — bge-m3 → arctic 계열 교체가 +0.073이었다.

**② ⚠️ 그리고 순위가 뒤집힌다.** prefix를 빠뜨린 arctic-ko(0.391)는
bge-m3(0.395)보다 **낮아 보인다.** 실제로는 가장 좋은 모델인데
**최하위로 기록됐을 것이고, 에러는 하나도 안 났을 것이다.**

**③ 오해하지 말 것.** "비대칭 prefix가 이득이니 bge-m3에도 붙이면 된다"가 **아니다.**
bge-m3는 prefix 없이 학습됐고 등록된 prompt도 비어 있다.
prefix는 **그 모델을 올바르게 쓰는 방법의 일부**이며, 이 표는
**모델 카드를 따라야 한다는 근거**일 뿐이다.

**④ 검사 방법이 중요하다.** `model.prompts`에 이름이 있는지로 확인하면 안 된다 —
sentence-transformers 5.x는 모델이 아무것도 등록하지 않아도
`{'query': '', 'document': ''}`를 **기본으로 채워 넣는다**(bge-m3에서 확인).
**값이 비었는지**로 검사해야 하고, `03_embed.py`가 그렇게 한다.

## 재현

```bash
# prefix 없이 (대조군)
backend/.venv/Scripts/python.exe scripts/03_embed.py \
    --model dragonkue/snowflake-arctic-embed-l-v2.0-ko \
    --out-root data/embeddings_ablation
# prefix 있는 쪽은 data/embeddings/ 의 것을 그대로 쓴다
```

## 파일

| 파일 | 내용 |
|---|---|
| `metrics.json` | 지표 · 하위 그룹 · 짝지은 검정 |
| `ranks.csv` | 질의별 순위. 열 이름에 prefix가 붙어 두 조건이 구분된다 |
