"""Phase 0-2 — Validation 질문을 구어체로 변형해 지표 1의 질의셋을 만든다.

    backend/.venv/Scripts/python.exe scripts/02_make_evalset.py --limit 10   # 시험 생성
    backend/.venv/Scripts/python.exe scripts/02_make_evalset.py              # 전량(기본 500)

왜 원본 질문을 그대로 쓰지 않는가:
    인덱스에 그 문장이 그대로 들어 있어 자기 자신을 찾는다. Recall이 100% 근처로
    나와서 모델 A와 B가 똑같아 보이고, 지표 1이 통째로 쓸모없어진다.
    "질의가 원본과 다른 문장"이라는 게 이 지표가 성립하는 전제다.

동시에 이건 실전 문체 검증이다. AI Hub 질문은 평균 330자 정제된 문어체인데
실사용자는 그렇게 묻지 않는다. 같은 질문을 문체만 바꿔 재므로 변수가 통제된다.

⚠️ 이 작업의 대표적 실패는 LLM이 문장을 **도로 문어체로 다듬어버리는 것**이다.
   "우리 개 눈이 밤에 초록색으로 빛나는데 정상인가요"  (좋음)
   "저희 강아지의 눈이 야간에 녹색으로 빛나는 현상이 관찰되는데, 이것이
    정상적인 반응인지 궁금합니다."                      (나쁨 — 테스트 의미 상실)
   그래서 --limit 10으로 먼저 눈으로 보고 전량을 돌린다.
"""

import argparse
import json
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

IN_DEFAULT = Path("data/normalized/aihub_qa.jsonl")
OUT_DEFAULT = Path("data/normalized/evalset_colloquial.jsonl")
ENV_PATH = Path("backend/.env")

# 문어체로 되돌아간 걸 잡아내는 표지. 생성 결과에 이게 있으면 실패로 본다.
FORMAL = re.compile(r"(문의드립니다|여쭤보고자|궁금합니다|알고 싶습니다|말씀드립니다"
                    r"|관찰되는데|하고자 합니다|드리고자|바랍니다)")

# 복합 질문 선별용. 00_2_analyze_chunking.py의 B_Q와 같은 패턴을 쓴다.
B_Q = re.compile(r"(\?+|[가-힣]+(?:요|까|나요|까요|지요|습니까|합니까|가요)"
                 r"|어떻게|무엇|언제|어디|얼마나|원인|이유)")

PROMPT = """너는 잘 정제된 문어체 질문을, 실제 보호자가 챗봇 입력창에 칠 법한 짧은 구어체로 바꾼다.

규칙:
- 30~70자. 원문의 핵심 증상·궁금증 **하나만** 남긴다
- 일상 말투로. "~인데요", "~한가요", "~어떡하죠", "~괜찮을까요"
- 품종·나이·병원 방문 이력 같은 배경 설명은 과감히 버린다. 사람들은 그렇게 안 친다
- **문어체로 다듬지 마라.** "문의드립니다", "궁금합니다", "여쭤보고자" 같은 표현 금지
- 요약문이 아니라 다시 쓴 질문이다
- 변환된 질문 한 줄만 출력. 따옴표나 설명 붙이지 마라

⚠️ 원문에 서로 다른 주제가 여러 개 섞여 있으면(실제로 16%가 그렇다),
**진료과에 해당하는 주제**를 골라라. 분량이 많은 쪽이 아니라 진료과에 맞는 쪽이다.
예: 진료과가 '치과'이고 원문에 이갈이와 잘못된 음식 섭취가 같이 있으면 → 이갈이를 고른다.

예시1
원문: 진돗개의 수컷이 녹내장으로 의심됩니다. 천장 조명이나 휴대폰의 후레시로 비추어보면 왼쪽 눈에서만 초록빛이 도는 것을 확인하였습니다. 안약으로만 치료가 가능한지 궁금합니다.
출력: 우리 개 눈이 밤에 초록색으로 빛나는데 정상인가요

예시2
원문: 현재 5개월 된 말티푸를 키우고 있으며, 최근 데려온 지 11일이 경과했습니다. 제 강아지가 켁켁거리며 마치 목에 무엇인가 걸린 듯한 기침을 하고 있습니다. 병원에 방문하였더니 기관지염이라는 진단을 받았습니다. 가습기가 없어 대체할 수 있는 좋은 방법이 있을지 궁금합니다.
출력: 강아지가 기관지염인데 가습기 없으면 뭘로 대신하나요

이제 변환해라.
진료과: {dept}
원문: {q}
출력:"""


def load_key() -> str:
    m = re.search(r"^GEMINI_API_KEY=(.*)$", ENV_PATH.read_text(encoding="utf-8"), re.M)
    if not m:
        raise SystemExit(f"[!] {ENV_PATH}에 GEMINI_API_KEY가 없다.")
    return m.group(1).strip().strip("\"'")


def sample_rows(rows, n, seed):
    """진료과 비례 층화 추출. 결정적이어야 한다.

    무작위로 뽑으면 돌릴 때마다 평가셋이 바뀌어 어제 숫자와 비교할 수 없다.
    비례 층화를 쓰는 이유는 치과 129건·안과 115건이 얇아서, 단순 무작위로는
    특정 진료과가 한 건도 안 뽑히는 일이 생기기 때문이다.
    """
    by_dept = defaultdict(list)
    for r in rows:
        by_dept[r["department"]].append(r)

    total = len(rows)
    picked = []
    for dept in sorted(by_dept):
        pool = sorted(by_dept[dept], key=lambda r: r["id"])   # 입력 순서에 의존하지 않게
        k = max(1, round(n * len(pool) / total))
        picked += random.Random(seed).sample(pool, min(k, len(pool)))
    picked.sort(key=lambda r: r["id"])
    return picked[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description="Validation 질문 -> 구어체 질의셋")
    ap.add_argument("--in", dest="inp", type=Path, default=IN_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--limit", type=int, default=500, help="생성할 질의 수")
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if not args.inp.exists():
        print(f"[!] 입력이 없다: {args.inp}  — 먼저 01_prepare_data.py를 돌릴 것.", file=sys.stderr)
        return 1

    rows = [json.loads(l) for l in args.inp.open(encoding="utf-8")]
    valid = [r for r in rows if r["split"] == "Validation"]

    # Train↔Valid 누수 제외. 이 질문은 Training에도 같은 문장이 있어서
    # "정답이 인덱스에 정확히 1개"라는 지표 1의 전제가 깨진다. 채점이 애매해진다.
    train_q = {r["content"] for r in rows if r["split"] == "Training"}
    leaked = [r for r in valid if r["content"] in train_q]
    valid = [r for r in valid if r["content"] not in train_q]
    if leaked:
        print(f"[제외] Train↔Valid 누수 {len(leaked)}건")

    targets = sample_rows(valid, args.limit, args.seed)

    # 이어서 돌리기. 중간에 끊겨도 이미 만든 건 건너뛴다.
    done = {}
    if args.out.exists():
        for l in args.out.open(encoding="utf-8"):
            d = json.loads(l)
            done[d["answer_id"]] = d
    todo = [r for r in targets if r["id"] not in done]
    print(f"대상 {len(targets)}건 (기생성 {len(targets) - len(todo)}건, 신규 {len(todo)}건)"
          f" · 모델 {args.model}")
    if not todo:
        print("새로 만들 것이 없다.")
        return 0

    from google import genai
    client = genai.Client(api_key=load_key())
    lock = threading.Lock()
    results, failures = [], []

    def work(r):
        try:
            resp = client.models.generate_content(
                model=args.model,
                contents=PROMPT.format(q=r["content"], dept=r["department"]))
            q = (resp.text or "").strip().strip('"').strip("'").split("\n")[0].strip()
        except Exception as e:                       # noqa: BLE001
            with lock:
                failures.append((r["id"], f"API: {type(e).__name__}: {e}"))
            return
        if not q:
            with lock:
                failures.append((r["id"], "빈 응답"))
            return
        orig_len = len(r["content"])
        # 끝에 한 번에 쓰면 2,399건을 다 돌린 뒤 마지막에 죽었을 때 전부 잃는다.
        # '이어서 돌리기'가 의미를 가지려면 완료되는 즉시 파일에 남아야 한다.
        with lock:
            row = {
                "query": q,                    # ← 검색에 쓸 구어체 질의
                "answer_id": r["id"],          # ← 정답 행 (원본), 정확히 1개
                "original": r["content"],      # 검수용
                "department": r["department"],
                "life_cycle": r["life_cycle"],
                "orig_len": orig_len,
                "query_len": len(q),
                "orig_question_marks": r["question_marks"],
                # Phase 1-f(R-6) 가설을 이 평가셋으로 바로 검증할 수 있게 표시해둔다.
                # 복합 질문에서 파생된 질의만 따로 Hit Rate를 내면 분해의 효과가 보인다.
                #
                # 기준은 LLM 정답 40건과 대조해 고른 것이다(dataset-analysis.md).
                # 종결어미 세기는 재현율 0.30으로 나빴고, 정교한 정규식(F1 0.73)도
                # 단순 길이(F1 0.74)를 못 이겼다. 재현율을 우선해 둘을 OR로 묶는다 —
                # 오탐은 분해 단계에서 걸러지지만 미탐은 영영 못 잡는다.
                "orig_is_multi": orig_len >= 400 or len(B_Q.findall(r["content"])) >= 3,
            }
            results.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8", newline="\n") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))

    results.sort(key=lambda d: d["answer_id"])

    # ---------- 품질 점검 ----------
    print()
    print("=" * 72)
    print(f"생성 {len(results)}건" + (f" · 실패 {len(failures)}건" if failures else ""))
    print("=" * 72)
    for rid, err in failures[:5]:
        print(f"  [실패] {rid}  {err}")

    if results:
        lens = sorted(d["query_len"] for d in results)
        shrink = [d["query_len"] / d["orig_len"] for d in results]
        formal = [d for d in results if FORMAL.search(d["query"])]
        same = [d for d in results if d["query"].strip() == d["original"].strip()]
        toolong = [d for d in results if d["query_len"] > 120]

        print(f"  질의 길이  중앙 {lens[len(lens)//2]}자  (원본 중앙 "
              f"{sorted(d['orig_len'] for d in results)[len(results)//2]}자)")
        print(f"  축약률     평균 {sum(shrink)/len(shrink)*100:.0f}%  ← 낮을수록 많이 줄었다")
        print(f"  ⚠️ 문어체 잔존 {len(formal)}건 · 원본과 동일 {len(same)}건 · 120자 초과 {len(toolong)}건")
        print("  진료과:", "  ".join(f"{k} {v}" for k, v in
                                    Counter(d["department"] for d in results).most_common()))
        print()
        print("─" * 72)
        print("샘플 (원본 → 변형)")
        print("─" * 72)
        for d in results[:10]:
            print(f"\n[{d['department']}] 원본 {d['orig_len']}자 → {d['query_len']}자")
            print(f"  원본: {d['original'][:110]}{'...' if d['orig_len'] > 110 else ''}")
            print(f"  변형: {d['query']}")
            if FORMAL.search(d["query"]):
                print("        ⚠️ 문어체 표현이 남았다")

    print()
    print(f"출력: {args.out}")
    print("  다음: scripts/03_embed.py (모델 3개 임베딩)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
