"""청킹이 필요한지 재검증. 문자 길이가 아니라 "잘리는가 / 여러 주제인가"를 본다.

docs/dataset-analysis.md "텍스트 길이" 절의 근거. 저장소 루트에서 실행:
    backend/.venv/Scripts/python.exe scripts/00_2_analyze_chunking.py
"""
import json, re, zipfile
from pathlib import Path

# Windows 기본 코드페이지(cp949)로는 이 파일의 출력 문자를 못 찍고 죽는다.
# 호출부에서 PYTHONIOENCODING을 걸어주지 않아도 되도록 여기서 처리한다.
import sys
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("data/59.반려견 성장 및 질병 관련 말뭉치 데이터/3.개방데이터/1.데이터")

ins, outs, srcs = [], [], []
for zp in sorted(ROOT.rglob("*.zip")):
    label = zp.name.startswith(("TL_", "VL_"))
    with zipfile.ZipFile(zp) as z:
        for n in [x for x in z.namelist() if x.lower().endswith(".json")]:
            raw = z.read(n).decode("utf-8-sig")
            if not label:
                srcs.append(len(raw))
                continue
            qa = json.loads(raw).get("qa", {})
            ins.append(qa.get("input", ""))
            outs.append(qa.get("output", ""))

def pct(xs, p):
    xs = sorted(xs)
    return xs[min(int(len(xs) * p), len(xs) - 1)]

print("=" * 70)
print("1. 길이 분포 (문자) — 꼬리까지")
print("=" * 70)
print(f"{'':<12}{'p50':>7}{'p90':>8}{'p95':>8}{'p99':>8}{'p99.9':>9}{'max':>9}")
for name, xs in [("input", [len(x) for x in ins]),
                 ("output", [len(x) for x in outs]),
                 ("in+out", [len(a) + len(b) for a, b in zip(ins, outs)])]:
    print(f"{name:<12}{pct(xs,.5):>7,}{pct(xs,.9):>8,}{pct(xs,.95):>8,}"
          f"{pct(xs,.99):>8,}{pct(xs,.999):>9,}{max(xs):>9,}")

print()
print("=" * 70)
print("2. 임계값 초과 건수 — '잘릴 위험'")
print("=" * 70)
n = len(ins)
# 한국어는 대략 1토큰 ≈ 1.3~1.8자 (모델별 상이). 보수적으로 1.3자/토큰 가정
for limit_tok, label in [(512, "512 tok (구형 BERT류)"), (8192, "8192 tok (bge-m3 계열)")]:
    lim_chars = int(limit_tok * 1.3)
    over_i = sum(1 for x in ins if len(x) > lim_chars)
    over_io = sum(1 for a, b in zip(ins, outs) if len(a) + len(b) > lim_chars)
    print(f"{label:<24} (~{lim_chars:,}자)  input {over_i:>6,}건({over_i/n*100:.2f}%)"
          f"   in+out {over_io:>6,}건({over_io/n*100:.2f}%)")

print()
print("=" * 70)
print("3. 복합 질문 여부 — '한 건에 여러 주제'인가")
print("=" * 70)
qpat = re.compile(r"(요\?|까\?|나요|ㄹ까요|는지요|습니까|궁금)")
cnt = [len(qpat.findall(x)) for x in ins]
buckets = {"1개 이하": 0, "2개": 0, "3개": 0, "4개 이상": 0}
for c in cnt:
    buckets["1개 이하" if c <= 1 else "2개" if c == 2 else "3개" if c == 3 else "4개 이상"] += 1
for k, v in buckets.items():
    print(f"  질문 표현 {k:<8} {v:>7,}건 ({v/n*100:5.1f}%)")

longs = [x for x in ins if len(x) > 1000]
lc = [len(qpat.findall(x)) for x in longs]
print(f"\n  1,000자 초과 질문 {len(longs):,}건 — 평균 질문 표현 {sum(lc)/max(len(lc),1):.1f}개")
print(f"  전체 평균 질문 표현 {sum(cnt)/n:.1f}개")

print()
print("=" * 70)
print("4. 원천데이터(말뭉치) — 청킹 필수 구간")
print("=" * 70)
print(f"  {len(srcs)}건  평균 {sum(srcs)//len(srcs):,}자  p50 {pct(srcs,.5):,}자  max {max(srcs):,}자")
print(f"  8192tok(~10,650자) 초과: {sum(1 for s in srcs if s > 10650):,}건 "
      f"({sum(1 for s in srcs if s > 10650)/len(srcs)*100:.0f}%)")

print()
print("=" * 70)
print("5. 가장 긴 질문 3건 — 실제로 쪼갤 만한가")
print("=" * 70)
for x in sorted(ins, key=len, reverse=True)[:3]:
    print(f"\n  [{len(x):,}자, 질문표현 {len(qpat.findall(x))}개]")
    print(f"  {x[:200]}...")
