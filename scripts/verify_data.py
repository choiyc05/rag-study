"""docs/dataset-analysis.md 수치 재검증. zip을 풀지 않고 그대로 읽는다."""

import json
import re
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(
    "C:/Users/403/Documents/workspace/rag-study/data/"
    "59.반려견 성장 및 질병 관련 말뭉치 데이터/3.개방데이터/1.데이터"
)

rows = []          # (split, kind, dept, meta, input, output)
src_counts = Counter()

for zp in sorted(ROOT.rglob("*.zip")):
    split = "Training" if "Training" in zp.parts else "Validation"
    label = zp.name.startswith(("TL_", "VL_"))
    dept = zp.stem.split("_")[-1]

    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".json")]
        if not label:
            src_counts[(split, dept)] += len(names)
            continue
        for n in names:
            d = json.loads(z.read(n).decode("utf-8-sig"))
            qa = d.get("qa", {})
            rows.append((split, dept, d.get("meta", {}),
                         qa.get("input", ""), qa.get("output", "")))

print("=" * 62)
print("1. 라벨링데이터(QA) 건수")
print("=" * 62)
per = Counter((s, dp) for s, dp, *_ in rows)
depts = ["내과", "외과", "피부과", "안과", "치과"]
print(f"{'진료과':<8}{'Training':>10}{'Validation':>12}{'합계':>10}")
for dp in depts:
    t, v = per[("Training", dp)], per[("Validation", dp)]
    print(f"{dp:<8}{t:>10,}{v:>12,}{t + v:>10,}")
tt = sum(per[("Training", d)] for d in depts)
vv = sum(per[("Validation", d)] for d in depts)
print(f"{'합계':<8}{tt:>10,}{vv:>12,}{tt + vv:>10,}")

print()
print("=" * 62)
print("2. 무결성")
print("=" * 62)
ins = [r[3] for r in rows]
outs = [r[4] for r in rows]
print(f"빈 input/output      : {sum(1 for i, o in zip(ins, outs) if not i.strip() or not o.strip())}건")
print(f"질문 고유            : {len(set(ins)):,} / {len(ins):,}  (완전중복 {len(ins) - len(set(ins))}건)")
print(f"답변 고유            : {len(set(outs)):,} / {len(outs):,}  (완전중복 {len(outs) - len(set(outs))}건)")
tr_q = {r[3] for r in rows if r[0] == "Training"}
va_q = {r[3] for r in rows if r[0] == "Validation"}
print(f"Train↔Valid 질문 누수: {len(tr_q & va_q)}건")
mismatch = sum(1 for s, dp, m, *_ in rows if m.get("department") != dp)
print(f"department vs 파일명 : {mismatch}건 불일치")

print()
print("=" * 62)
print("3. 메타데이터")
print("=" * 62)
lc = Counter(m.get("lifeCycle") for _, _, m, *_ in rows)
for k, c in lc.most_common():
    print(f"  lifeCycle {k:<6} {c:>7,}  ({c / len(rows) * 100:.1f}%)")
dis = Counter(m.get("disease") for _, _, m, *_ in rows)
etc = dis.get("기타", 0)
print(f"  disease   {len(dis)}종, '기타' {etc:,}건 ({etc / len(rows) * 100:.0f}%)")
print("  상위:", ", ".join(f"{k} {c}" for k, c in dis.most_common(6) if k != "기타"))

print()
print("=" * 62)
print("4. 길이(문자) / output 특성")
print("=" * 62)


def stat(name, xs):
    xs = sorted(len(x) for x in xs)
    n = len(xs)
    print(f"{name:<8} 평균 {sum(xs) / n:>6.0f}  중앙 {xs[n // 2]:>5,}  "
          f"p90 {xs[int(n * 0.9)]:>5,}  최대 {xs[-1]:>7,}")


stat("input", ins)
stat("output", outs)
n = len(rows)
hosp = sum(1 for o in outs if "동물병원" in o)
vet = sum(1 for o in outs if "수의사" in o)
cost = sum(1 for o in outs if re.search(r"(비용|가격|원대|만원|\d+\s*원)", o))
qmark = sum(1 for i in ins if re.search(r"(\?|까요|나요|는지|습니까|궁금)", i))
print(f"'동물병원' 권유  {hosp:>6,} / {n:,}  ({hosp / n * 100:.0f}%)")
print(f"'수의사' 언급    {vet:>6,} / {n:,}  ({vet / n * 100:.0f}%)")
print(f"가격/비용 언급   {cost:>6,} / {n:,}  ({cost / n * 100:.0f}%)")
print(f"input 질문형태   {qmark:>6,} / {n:,}  ({qmark / n * 100:.0f}%)")

print()
print("=" * 62)
print("5. 원천데이터(말뭉치)")
print("=" * 62)
for (s, dp), c in sorted(src_counts.items(), key=lambda x: (-x[1])):
    print(f"  {s:<11}{dp:<6}{c:>5}")
print(f"  합계 {sum(src_counts.values())}건")

print()
print(f"QA 텍스트 총량: {sum(len(i) + len(o) for i, o in zip(ins, outs)):,}자")
