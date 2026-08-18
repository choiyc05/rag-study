# Phase 0-4 — 임베딩 모델 3종 비교

**측정일** 2026-08-18 · **판정과 해석은** [../../experiments.md](../../experiments.md) Phase 0 절

이 프로젝트의 **첫 숫자**다. 이후 모든 실험은 이 값을 기준선으로 삼는다.

1차 후보 3종(2026-08-17 선정)과 리더보드를 재조사해 고른 2차 후보 3종,
총 **6종**을 같은 조건에서 비교했다. 선정 기준은 [rag-design.md](../../rag-design.md)
"2차 후보 세트".

---

## 측정 조건

| 항목 | 값 |
|---|---|
| 지표 | 지표 1 — known-item retrieval (정답이 인덱스에 정확히 1개) |
| 인덱스 | `data/normalized/aihub_qa.jsonl` **21,606건** · 임베딩 대상은 `content`(질문) |
| 질의 | `data/normalized/evalset_colloquial.jsonl` **2,399건** (Validation 질문의 구어체 변형) |
| 검색 | numpy 브루트포스 내적 (벡터가 정규화되어 코사인과 동일). **DB 없음** |
| GPU | RTX 3050 6GB · fp16 · `max_seq_len=1024` · 길이 정렬 배치 |
| 청킹 | **없음** — 질문 p99가 628토큰이라 자를 대상이 아니다 |
| prefix | **모델이 등록한 값**을 사용 (`--query-prompt-name query`). 추측한 리터럴 아님 |

---

## 결과 (6종)

| 모델 | 차원 | prefix | Hit@1 | Hit@5 | Hit@20 | MRR | 임베딩 |
|---|---:|---|---:|---:|---:|---:|---:|
| `dragonkue/snowflake-arctic-embed-l-v2.0-ko` | 1024 | `query: ` | **0.376** | **0.572** | **0.704** | **0.468** | 275s |
| **`Snowflake/snowflake-arctic-embed-l-v2.0`** ★주 후보 | 1024 | `query: ` | 0.371 | 0.562 | 0.693 | 0.463 | 260s |
| `telepix/PIXIE-Rune-v1.0` | 1024 | `query: ` | 0.351 | 0.561 | 0.703 | 0.448 | 278s |
| `nlpai-lab/KURE-v1` | 1024 | 없음 | 0.301 | 0.509 | 0.664 | 0.399 | 275s |
| `BAAI/bge-m3` | 1024 | 없음 | 0.295 | 0.511 | 0.664 | 0.395 | 277s |
| `Alibaba-NLP/gte-multilingual-base` | 768 | 없음 | 0.251 | 0.423 | 0.581 | 0.335 | 173s |

### 짝지은 부트스트랩 (기준 = arctic-ko, 2,000회 재표집)

| 상대 | ΔMRR | 95% 구간 | 이김 | 짐 | 무 | 판정 |
|---|---:|---:|---:|---:|---:|---|
| `arctic-l-v2.0` (base) | −0.0048 | **[−0.0116, +0.0023]** | 703 | 687 | 1,009 | **차이 없음** |
| `PIXIE-Rune-v1.0` | −0.0201 | [−0.0262, −0.0140] | 640 | 714 | 1,045 | 기준이 우세 |
| `KURE-v1` | −0.0692 | [−0.0798, −0.0590] | 523 | 1,139 | 737 | 기준이 우세 |
| `bge-m3` | −0.0731 | [−0.0843, −0.0619] | 601 | 1,087 | 711 | 기준이 우세 |
| `gte-multilingual-base` | −0.1328 | [−0.1469, −0.1184] | 490 | 1,345 | 564 | 기준이 우세 |

---

## 읽는 법 — 이 표에서 실제로 결론난 것

**① 차원 1024 확정. 주 후보는 `Snowflake/snowflake-arctic-embed-l-v2.0`.**
한국어 최고점은 `-ko` 파인튜닝(0.468)이지만 base(0.463)와 **통계적 동률**이다.
동률이면 **잃는 게 없는 쪽**을 택한다 — base는 영어·교차언어 능력을 유지한다.
확장 데이터를 한글/영어 구분 없이 넣기로 한 계획에 **대가가 없다**는 뜻이다.

⚠️ **단, 모델을 여기서 하나로 줄이지 않는다.** Phase 1은 `arctic base` ·
`arctic-ko` · `bge-m3` **3종 병행**으로 돌린다. Phase 1의 변경이 모델마다 다르게
들 수 있어서다. **DB에 적재할 최종 1종은 Phase 2 진입 시** 정한다.
상세는 [../../experiments.md](../../experiments.md) "Phase 1".

**② ⚠️ 한국어 파인튜닝은 이득이 없다 — 독립 사례 2건.**

| 대조군 (base 동일, 변수는 파인튜닝 하나) | 리더보드 차 | 우리 데이터 |
|---|---:|---|
| bge-m3 → KURE-v1 | +1.46 | 차이 없음 |
| arctic-l-v2.0 → arctic-l-v2.0-**ko** | +2.14 | **차이 없음** |

**서로 다른 계열에서 반복됐다.** 리더보드 2.14점 차도 우리 평가셋에서는 안 갈렸다.

**③ 실제 개선은 계열 교체에서 왔다.** bge-m3 → arctic 계열이
**ΔMRR +0.073(상대 +18%)**, Hit@5 0.511 → 0.562. Phase 0의 유일한 실질 개선이다.

**④ ⚠️⚠️ prefix 하나가 순위를 뒤집는다.** 별도 대조군에서
같은 모델의 prefix만 빼니 MRR 0.468 → **0.391**(−16%), 앵커보다 낮아 보였다.
상세는 [../phase0-prefix-ablation/](../phase0-prefix-ablation/).

**⑤ ⚠️ 768차원 모델(gte)이 최하위지만, 그게 "768차원 탓"은 아니다.**
gte는 차원과 모델이 **같이 다르다.** 차원만 잘라 따로 재보니
같은 768차원에서 arctic은 MRR 0.460이 나왔다(gte는 0.335, **차이 0.125**).
→ **차원이 아니라 모델이 성능을 결정한다.**
차원만의 효과는 [../phase0-dimension/](../phase0-dimension/) 참고.

**⑥ 절대값이 낮은 것(Hit@5 0.56)은 고장이 아니다.**
구어체 변형이 **336자 → 34자**로 정보를 덜어내, 원문의 여러 갈래 중 한 갈래만 남은
짧은 질의로 원문을 되찾는 문제가 됐다. **평가셋이 포화되지 않았다** — 실제로
6종을 0.335~0.468로 갈랐고 prefix 유무 같은 미세한 차이까지 잡아냈다.

**⑦ 복합 질문 격차는 모델 교체로 안 풀린다.**

| 그룹 | 건수 | arctic-ko | **arctic base** | PIXIE | KURE-v1 | bge-m3 | gte |
|---|---:|---:|---:|---:|---:|---:|---:|
| 복합 질문 파생 Hit@5 | 901 | 0.502 | 0.483 | 0.482 | 0.434 | 0.451 | 0.351 |
| 단일 주제 파생 Hit@5 | 1,498 | 0.614 | 0.609 | 0.608 | 0.555 | 0.547 | 0.467 |
| **격차** | | −11.2%p | **−12.6%p** | −12.6%p | −12.1%p | −9.6%p | −11.6%p |

**6종 전부에서 반복되고, 주 후보에서 오히려 더 벌어진다** — 모델 개선분이
복합 질문에서는 절반만 나타났다. 로드맵 **R-6(긴 QA 의미 분해)**의 근거다.

**⑧ ⚠️ R-4(메타데이터 필터)의 전제는 확인되지 않았다.**
"치과 879건이 내과 11,049건에 묻힌다"고 봤으나 진료과별 Hit@5는 **평평하다**
(1차 세트 기준 0.47~0.57). **R-4는 기대 효과를 낮게 잡고 접근할 것.**

**⑨ ⚠️ 다국어 기준은 아직 미검증이다.** 주 후보를 다국어 기준으로 골랐지만
이 평가셋은 **한국어 전용**이라 그 능력을 잰 적이 없다. 검증은 roadmap
**Phase 3-0b(교차언어 평가 슬라이스)** — **영어 문서를 넣기 전에** 만들어야 한다.

---

## prefix가 "없음"인 근거

추측이 아니라 받아둔 모델 파일에서 확인했다.

추측이 아니라 모델이 등록한 값을 그대로 썼다(`--query-prompt-name query`).

| 모델 | `config_sentence_transformers.json`의 `prompts` | 적용값 |
|---|---|---|
| `Snowflake/snowflake-arctic-embed-l-v2.0` | `{"query": "query: "}` | 질의에 `query: ` |
| `dragonkue/…-l-v2.0-ko` | `{"query": "query: ", "document": ""}` | 질의에 `query: ` |
| `telepix/PIXIE-Rune-v1.0` | `{"query": "query: ", "document": ""}` | 질의에 `query: ` |
| `BAAI/bge-m3` | 키 없음 | 없음 |
| `nlpai-lab/KURE-v1` | `{}` (빈 객체) | 없음 |
| `Alibaba-NLP/gte-multilingual-base` | 파일 자체가 없음 | 없음 |

**base와 `-ko` 파인튜닝의 등록값이 같아서** 둘의 비교가 prompt 차이로 오염되지 않는다.

⚠️ 이게 **얼마나 중요한지는 [../phase0-prefix-ablation/](../phase0-prefix-ablation/)에서
숫자로 확인했다 — 빠뜨리면 MRR −16%, 최고 모델이 최하위로 보인다.**

---

## 재현

```bash
cd backend && uv sync --group ml && cd ..
E=backend/.venv/Scripts/python.exe

# prefix가 등록된 모델 — 리터럴을 적지 말고 등록 이름을 쓴다
$E scripts/03_embed.py --model Snowflake/snowflake-arctic-embed-l-v2.0    --query-prompt-name query
$E scripts/03_embed.py --model dragonkue/snowflake-arctic-embed-l-v2.0-ko --query-prompt-name query
$E scripts/03_embed.py --model telepix/PIXIE-Rune-v1.0                    --query-prompt-name query
# prefix가 없는 모델 — 옵션을 주면 값이 비어 있어 스크립트가 죽는다(의도된 동작)
$E scripts/03_embed.py --model BAAI/bge-m3
$E scripts/03_embed.py --model nlpai-lab/KURE-v1
$E scripts/03_embed.py --model Alibaba-NLP/gte-multilingual-base

$E scripts/04_evaluate.py --save docs/results/phase0-embedding
```

임베딩 산출물(`data/embeddings/`)은 `.gitignore` 대상이라 저장소에 없다.
모델당 3~5분, 6종에 약 26분이면 다시 만들어진다.

## 파일

| 파일 | 내용 |
|---|---|
| `metrics.json` | 모델별 지표 · 하위 그룹 · 진료과별 · 짝지은 검정 |
| `ranks.csv` | **질의별 정답 순위** — 2,399행 × 모델 3열. 이후 실험과 짝지어 비교하는 원자료 |
