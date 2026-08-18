"""청킹이 필요한지 재검증. 문자 길이가 아니라 "잘리는가 / 여러 주제인가"를 본다.

    backend/.venv/Scripts/python.exe scripts/00_2_analyze_chunking.py

docs/dataset-analysis.md "텍스트 길이" 절의 근거.

⚠️ 3번(복합 질문)의 규칙은 **선별용 추정치일 뿐 판정이 아니다.**
   LLM 정답 40건과 대조한 결과 어떤 규칙도 재현율 0.7을 못 넘었다(3번 주석 참고).
   최종 판정은 Phase 1-f에서 실제로 분해해보고 내린다.
"""
import json
import re
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("data/59.반려견 성장 및 질병 관련 말뭉치 데이터/3.개방데이터/1.데이터")

ins, outs, corpus = [], [], []
for zp in sorted(ROOT.rglob("*.zip")):
    label = zp.name.startswith(("TL_", "VL_"))
    with zipfile.ZipFile(zp) as z:
        for n in [x for x in z.namelist() if x.lower().endswith(".json")]:
            d = json.loads(z.read(n).decode("utf-8-sig"))
            if not label:
                # 원천데이터는 본문이 'disease' 키에 들어있다(필드명이 내용과 안 맞는다).
                # 예전엔 JSON 파일 문자열 길이를 재서 title/author/JSON 구문까지
                # 351자쯤 같이 셌다. 본문만 재야 청크 수 추정이 맞는다.
                corpus.append(len(d.get("disease") or ""))
                continue
            qa = d.get("qa", {})
            ins.append(qa.get("input", ""))
            outs.append(qa.get("output", ""))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(int(len(xs) * p), len(xs) - 1)]


n = len(ins)

print("=" * 74)
print("1. 길이 분포 (문자) — 꼬리까지")
print("=" * 74)
print(f"{'':<12}{'p50':>7}{'p90':>8}{'p95':>8}{'p99':>8}{'p99.9':>9}{'max':>9}")
for name, xs in [("input", [len(x) for x in ins]),
                 ("output", [len(x) for x in outs]),
                 ("in+out", [len(a) + len(b) for a, b in zip(ins, outs)])]:
    print(f"{name:<12}{pct(xs,.5):>7,}{pct(xs,.9):>8,}{pct(xs,.95):>8,}"
          f"{pct(xs,.99):>8,}{pct(xs,.999):>9,}{max(xs):>9,}")

print()
print("=" * 74)
print("2. 임계값 초과 — '잘릴 위험'")
print("=" * 74)
# 한국어는 대략 1토큰 ≈ 1.3~1.8자 (모델별 상이). 보수적으로 1.3자/토큰 가정.
# 정확한 값은 모델 확정 후 실제 토크나이저로 다시 잴 것.
for limit_tok, label in [(512, "512 tok (구형 BERT류)"), (8192, "8192 tok (bge-m3 계열)")]:
    lim = int(limit_tok * 1.3)
    oi = sum(1 for x in ins if len(x) > lim)
    oio = sum(1 for a, b in zip(ins, outs) if len(a) + len(b) > lim)
    print(f"{label:<24} (~{lim:,}자)  input {oi:>6,}건({oi/n*100:5.2f}%)"
          f"   in+out {oio:>6,}건({oio/n*100:5.2f}%)")
print("  → 후보 3종은 전부 8,192라 잘리지 않는다. 모델을 바꾸면 다시 볼 것.")

print()
print("=" * 74)
print("3. 복합 질문 — '한 건에 여러 주제'인가  ⚠️ 추정치")
print("=" * 74)

# 종결어미만 세는 방식(A)은 재현율이 낮다. 문어체 질문("~문의드립니다")은
# 종결어미가 아예 안 잡히는데도 주제가 3~4개인 경우가 많았다.
A = re.compile(r"(요\?|까\?|나요|ㄹ까요|는지요|습니까|궁금)")
# 의문사·의문명사까지 넓힌 것. 재현율이 크게 오르지만 오탐도 생긴다.
B_Q = re.compile(r"(\?+|[가-힣]+(?:요|까|나요|까요|지요|습니까|합니까|가요)\b"
                 r"|어떻게|무엇|언제|어디|얼마나|원인|이유)")
# 주제 전환 신호. 단독으로는 재현율이 낮다.
B_M = re.compile(r"(\n+|- |\*\s*|\d+\.\s*|그리고|또한|추가로|이외에도|아울러)")

RULES = [
    ("종결어미 3개+ (구버전)",       lambda x: len(A.findall(x)) >= 3),
    ("종결어미 3개+ 또는 (2개&500자+)", lambda x: (lambda c: c >= 3 or (c == 2 and len(x) >= 500))(len(A.findall(x)))),
    ("의문표현 3개+",                lambda x: len(B_Q.findall(x)) >= 3),
    ("의문표현 5개+",                lambda x: len(B_Q.findall(x)) >= 5),
    ("전환 접속사 1개+",             lambda x: len(B_M.findall(x)) >= 1),
    ("길이 400자+",                  lambda x: len(x) >= 400),
    ("길이 400자+ 또는 의문표현 3개+", lambda x: len(x) >= 400 or len(B_Q.findall(x)) >= 3),
]
print(f"  {'규칙':<32}{'해당':>9}{'비율':>8}   {'LLM 대조 F1':>12}")
# 아래 F1은 2026-08-18에 LLM 정답 40건(길이 층화 표본)과 대조해 실측한 값이다.
# 재현은 scripts 밖의 일회성 벤치마크였고, 수치는 dataset-analysis.md에 남겼다.
F1 = {"종결어미 3개+ (구버전)": "0.46", "종결어미 3개+ 또는 (2개&500자+)": "0.54",
      "의문표현 3개+": "0.73", "의문표현 5개+": "0.73", "전환 접속사 1개+": "0.65",
      "길이 400자+": "0.74", "길이 400자+ 또는 의문표현 3개+": "—"}
for name, fn in RULES:
    c = sum(1 for x in ins if fn(x))
    print(f"  {name:<32}{c:>9,}{c/n*100:>7.1f}%   {F1.get(name,'—'):>12}")

print()
print("  ⚠️ LLM 정답 40건 기준 실제 복합 비율은 68%다. 위 규칙은 전부 과소 추정한다.")
print("     종결어미만 세는 방식은 재현율 0.30으로 특히 나쁘다 —")
print("     '~문의드립니다'로 끝나는 문어체 질문에 주제가 3~4개 들어있는 경우를 통째로 놓친다.")
print("     길이 하나가 정교한 정규식보다 잘 맞는다(F1 0.74). 규칙은 선별용으로만 쓰고,")
print("     최종 판정은 Phase 1-f에서 실제로 분해해보고 내린다(분해 결과 1개면 원본 유지).")

print()
print("=" * 74)
print("4. 원천데이터(말뭉치) — 청킹 필수 구간")
print("=" * 74)
cn = len(corpus)
print(f"  {cn}건  평균 {sum(corpus)//cn:,}자  p50 {pct(corpus,.5):,}자  "
      f"p90 {pct(corpus,.9):,}자  max {max(corpus):,}자")
for limit_tok in (512, 8192):
    lim = int(limit_tok * 1.3)
    over = sum(1 for s in corpus if s > lim)
    print(f"  {limit_tok:>5} tok(~{lim:,}자) 초과: {over:>3}건 ({over/cn*100:.0f}%)")
print()
print("  ⚠️ 여기서 '초과 여부'는 청킹 필요의 **하한**일 뿐이다.")
print("     8,192를 안 넘어도 16,000자짜리 논문을 벡터 하나로 만들면 주제가 뭉개진다.")
print("     원천데이터의 진짜 문제는 길이가 아니라 품질이다 —")
print("     영어 원문 잔존 87%, 오역 52%, 표/캡션 혼입 51%. 잘라도 그대로 들어간다.")
print("     그래서 전처리가 청킹보다 먼저다(docs/rag-design.md).")
# 청크 수 추정 — 인덱스에서 원천이 차지할 비중을 가늠하려고
total = sum(corpus)
for size in (400, 800):
    print(f"     {size}자 청킹 시 약 {total//size:,}청크 → QA {n:,}건 대비 "
          f"{total//size/(total//size + n)*100:.0f}%")

print()
print("=" * 74)
print("5. 표본 눈으로 보기 — 규칙이 놓치는 것 확인용")
print("=" * 74)
print("  자동 판정이 아니다. 위 규칙들이 실제 문장과 맞는지 사람이 확인하는 단계다.")
print("  길이 상위만 보면 편향되므로, 규칙별로 '엇갈리는' 표본을 뽑는다.\n")

long_no_marker = [x for x in ins if len(x) >= 600 and len(A.findall(x)) == 0]
short_many = [x for x in ins if len(x) < 250 and len(B_Q.findall(x)) >= 3]
for title, pool in [
    ("길이는 긴데 종결어미가 0개 (구버전이 통째로 놓치던 유형)", long_no_marker),
    ("짧은데 의문표현이 많음 (길이 규칙이 놓치는 유형)", short_many),
    ("가장 긴 질문", sorted(ins, key=len, reverse=True)),
]:
    print(f"── {title}  — 해당 {len(pool):,}건")
    for x in pool[:2]:
        print(f"   [{len(x):,}자 · 종결어미 {len(A.findall(x))} · 의문표현 "
              f"{len(B_Q.findall(x))} · 전환 {len(B_M.findall(x))}]")
        print(f"   {x[:150]}...")
        print(f"   ...(끝) {x[-80:]}")
        print()
