# docs

`rag-study` 프로젝트 진행 기록. 세션이 끊겨도 여기만 읽으면 맥락을 복구할 수 있도록 정리한다.

| 문서 | 내용 |
|---|---|
| [setup.md](setup.md) | 환경 세팅 (Python 3.12, uv, 백엔드 구조, Docker, git 설정) |
| [dataset-analysis.md](dataset-analysis.md) | AI Hub 데이터셋 실측 분석 결과 |
| [rag-design.md](rag-design.md) | RAG 설계 결정과 그 근거 |

## 현재 상태 (2026-08-12 기준)

- 백엔드 / 프론트 / DB 기초 세팅 완료, `main`에 푸시됨 (`83cac7d`)
- 데이터 분석 완료 — [dataset-analysis.md](dataset-analysis.md)
- RAG 1차 시도: QA 21,606건 전체를 임베딩 → pgvector 적재 → 검색까지 확인
- **임베딩·리랭커를 다시 구성하는 중**

## 다음 할 일 (우선순위)

1. **평가 기준선 확보** — Validation 2,400건을 인덱스에서 제외하고 Recall@k 측정
   (현재는 전체를 인덱싱해서 측정 기준이 없는 상태)
2. **리랭커 도입** — 벡터 검색 top 20 → 리랭커 top 3
3. **실전 문체 검증** — Validation 질문을 구어체로 변형해 성능 낙폭 측정
4. 데이터 추가는 위 3개를 마치고 커버리지 공백이 수치로 확인될 때만
