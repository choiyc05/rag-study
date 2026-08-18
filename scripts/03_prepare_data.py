"""Phase 0-1 — AI Hub 라벨링데이터(QA) zip을 JSONL 한 벌로 정규화한다.

    저장소 루트에서:  backend/.venv/Scripts/python.exe scripts/03_prepare_data.py

이 스크립트가 다루는 것은 **QA 21,606건뿐이다.** 원천데이터(말뭉치) 239건은
청킹·전처리가 필요하고 투입 여부 자체가 A/B 대상이라 Phase 3에서 별도로 다룬다
(docs/roadmap.md Phase 3, experiments.md D-1).

출력 스키마는 docs/rag-design.md의 documents 테이블을 그대로 따른다.
핵심은 content/payload 분리다:

    content  = 질문(input)   ← 임베딩·검색 대상
    payload  = 답변(output)  ← LLM에 넘길 본문

질문으로 찾아 답변을 건네주므로 둘이 다르다. 이렇게 두면 나중에 청크 행이
들어와도(content=payload=청크 본문) 검색 코드가 행 종류를 몰라도 된다.

`instruction`은 버린다 — 40종 변형이 있지만 전부 같은 말이라 검색에 노이즈만 된다.
시스템 프롬프트는 Phase 2에서 새로 쓴다.
"""

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

# Windows 기본 코드페이지(cp949)로는 이 파일의 출력 문자를 못 찍고 죽는다.
# 호출부에서 PYTHONIOENCODING을 걸어주지 않아도 되도록 여기서 처리한다.
sys.stdout.reconfigure(encoding="utf-8")

DATA_ROOT = Path("data/59.반려견 성장 및 질병 관련 말뭉치 데이터/3.개방데이터/1.데이터")
OUT_DEFAULT = Path("data/normalized/aihub_qa.jsonl")

# 02_analyze_chunking.py와 같은 패턴을 쓴다. 여기서 세는 값으로 Phase 1-f(R-6)의
# 분해 대상을 고르므로, 두 스크립트가 다른 기준을 쓰면 숫자가 어긋난다.
Q_PATTERN = re.compile(r"(요\?|까\?|나요|ㄹ까요|는지요|습니까|궁금)")


def iter_qa(root: Path):
    """라벨링데이터 zip을 풀지 않고 그대로 읽어 QA 레코드를 뽑는다.

    zip 안의 JSON 파일 1개 = QA 1쌍이고, 파일명(UUID)은 18개 zip 전체에서 고유하다.
    그래서 파일명 stem을 그대로 id로 쓴다 — 순번을 새로 매기면 스크립트를 다시 돌릴
    때마다 id가 흔들려서, 평가셋이 가리키는 정답 행을 잃는다.
    """
    for zp in sorted(root.rglob("*.zip")):
        if not zp.name.startswith(("TL_", "VL_")):
            continue  # TS_/VS_ = 원천데이터. Phase 3에서 다룬다

        split = "Training" if "Training" in zp.parts else "Validation"
        dept_from_name = zp.stem.split("_")[-1]

        with zipfile.ZipFile(zp) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith(".json"))
            for name in names:
                # utf-8-sig — BOM이 붙어 있어 utf-8로 읽으면 첫 키가 '﻿meta'가 되고
                # d["meta"]가 KeyError로 죽는다. 이 데이터의 첫 번째 함정.
                doc = json.loads(z.read(name).decode("utf-8-sig"))
                meta, qa = doc.get("meta", {}), doc.get("qa", {})

                # 메타값은 반드시 strip한다. 실측에서 department가 " 치과"(앞 공백)인
                # 행이 1건 나왔다. 그대로 두면 진료과가 6종으로 집계되고,
                # WHERE department='치과' 필터가 이 행을 **에러 없이** 놓친다.
                # dataset-analysis.md가 '파일명 불일치 1건'이라 적은 것의 실제 정체다.
                def clean(v):
                    return v.strip() if isinstance(v, str) else v

                yield {
                    "id": Path(name).stem,
                    "split": split,
                    "dept_from_filename": dept_from_name,
                    "department": clean(meta.get("department")),
                    "life_cycle": clean(meta.get("lifeCycle")),
                    "disease": clean(meta.get("disease")),
                    "input": (qa.get("input") or "").strip(),
                    "output": (qa.get("output") or "").strip(),
                }


def main() -> int:
    ap = argparse.ArgumentParser(description="AI Hub QA zip -> JSONL")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    if not args.data_root.exists():
        print(f"[!] 데이터 경로가 없다: {args.data_root}", file=sys.stderr)
        print("    저장소 루트에서 실행했는지, data/를 내려받았는지 확인할 것.", file=sys.stderr)
        return 1

    records = list(iter_qa(args.data_root))
    if not records:
        print(f"[!] QA를 한 건도 못 읽었다: {args.data_root}", file=sys.stderr)
        return 1

    # 결정적 정렬. 돌릴 때마다 순서가 바뀌면 산출물 diff가 무의미해지고,
    # '어제 파일과 오늘 파일이 같은가'를 확인할 수 없다.
    records.sort(key=lambda r: (r["split"], r["department"] or "", r["id"]))

    skipped, dept_mismatch = [], []
    rows = []
    for r in records:
        # 실측상 0건이지만 방어한다. 빈 content는 임베딩하면 무의미한 벡터가 되고,
        # 그게 모든 질의에 어중간하게 걸리는 쓰레기 행이 된다.
        if not r["input"] or not r["output"]:
            skipped.append(r["id"])
            continue

        # meta.department를 정본으로 쓰고 불일치만 기록한다. 파일명은 zip 묶음 단위라
        # 개별 행의 진료과를 meta보다 더 잘 안다고 볼 근거가 없다.
        if r["department"] != r["dept_from_filename"]:
            dept_mismatch.append((r["id"], r["dept_from_filename"], r["department"]))

        content = r["input"]
        rows.append({
            "id": r["id"],
            "content": content,           # ★ 임베딩 대상 = 질문
            "payload": r["output"],       # ★ LLM에 넘길 본문 = 답변
            "source": "aihub_qa",
            # 평가셋 분리에 필수다. 지표 1은 전체를 인덱싱하고 Validation에서 질의를
            # 만들지만, 지표 2는 Validation을 인덱스에서 빼고 원본 질문으로 묻는다.
            # 이 값이 없으면 두 지표 중 하나를 못 만든다.
            "split": r["split"],
            "department": r["department"],
            "life_cycle": r["life_cycle"],
            "disease": r["disease"],
            # 아래 둘은 원본에 없는 파생값이다. Phase 1-f(R-6)에서 '복합 질문'을
            # 고를 때 쓰려고 미리 재둔다 — 그때 21,606건을 다시 파싱하지 않으려고.
            "content_len": len(content),
            "question_marks": len(Q_PATTERN.findall(content)),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---------- 요약 ----------
    n = len(rows)
    by_split = Counter(r["split"] for r in rows)
    by_dept = Counter(r["department"] for r in rows)

    print("=" * 66)
    print(f"출력: {args.out}  ({args.out.stat().st_size / 1024 / 1024:.1f} MB)")
    print("=" * 66)
    print(f"  총 {n:,}건   Training {by_split['Training']:,} / Validation {by_split['Validation']:,}")
    print("  진료과:", "  ".join(f"{k} {v:,}" for k, v in by_dept.most_common()))
    if skipped:
        print(f"  [건너뜀] 빈 input/output {len(skipped)}건")
    if dept_mismatch:
        print(f"  [주의] department 불일치 {len(dept_mismatch)}건 (meta 값을 채택)")
        for i, fn, mt in dept_mismatch[:3]:
            print(f"         {i}  파일명={fn}  meta={mt}")

    # 평가셋 전제 점검 — 여기가 깨지면 Phase 0-2를 그대로 진행하면 안 된다.
    tr = {r["content"] for r in rows if r["split"] == "Training"}
    va = {r["content"] for r in rows if r["split"] == "Validation"}
    leak = tr & va
    print()
    print("  평가셋 전제:")
    print(f"    질문 고유 {len({r['content'] for r in rows}):,} / {n:,}")
    print(f"    Train↔Valid 질문 누수 {len(leak)}건" + ("  ← 0이 아니면 0-2에서 제외할 것" if leak else ""))

    # R-6 대상 규모 — 지금 고르지는 않고, 얼마나 되는지만 알아둔다.
    multi = sum(1 for r in rows if r["question_marks"] >= 3)
    print(f"    복합 질문(질문표현 3개+) {multi:,}건 ({multi / n * 100:.1f}%)  ← Phase 1-f 대상 후보")

    print()
    print("  다음: scripts/04_make_evalset.py (구어체 변형 평가셋)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
