"""Phase 1-b (R-2) — 리랭커. **정확도보다 지연 실측이 먼저다.**

    # ① 먼저 이것만 돌린다. 2초가 넘으면 정확도 비교는 의미가 없다.
    backend/.venv/Scripts/python.exe scripts/1b_rerank.py --latency-only \
        --emb-dir data/embeddings/snowflake_snowflake_arctic_embed_l_v2_0 \
        --reranker BAAI/bge-reranker-v2-m3

    # ② 지연이 받아들일 만하면 정확도
    backend/.venv/Scripts/python.exe scripts/1b_rerank.py \
        --emb-dir data/embeddings/snowflake_snowflake_arctic_embed_l_v2_0 \
        --reranker BAAI/bge-reranker-v2-m3 --arm-id R-2a-A \
        --baseline phase0-embedding:E-7 --save docs/results/phase1-r2a

왜 지연이 먼저인가 (roadmap Phase 1, experiments.md R-2):
    리랭커는 **매 요청마다** 후보 20건을 돌린다. 정확도가 올라도 응답이 3초가 되면
    서비스로 못 쓴다. 3050 6GB에서 568M × 512토큰 × 20쌍이면 2초 안팎으로 계산됐다.
    사실이면 정확도 비교를 하기 전에 구성(모델 크기·max_length·후보 수)부터 바꿔야 한다.

채점 방식 — 리랭킹은 **top-N 안에서만** 순서를 바꾼다:
    정답이 후보 안에 있으면  → 리랭킹 후 위치가 새 순위
    정답이 후보 밖에 있으면  → 벡터 검색의 순위가 그대로 남는다
                              (후보 N건은 어차피 정답보다 위에 있었다)
    → Hit@20·MRR이 B-0과 같은 척도로 유지되고, 짝지은 비교가 성립한다.

⚠️ 쌍둥이 보정과 집계는 04_evaluate.py의 것을 그대로 쓴다. 두 벌이 되면
   기준선과 채점 기준이 달라지는데 **에러 없이 Δ만 틀리게 나온다.**
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

DOCS_DEFAULT = Path("data/normalized/aihub_qa.jsonl")
EVAL_DEFAULT = Path("data/normalized/evalset_colloquial.jsonl")
# arm 하나가 11분이라 중간에 멈추면 완주한 arm까지 잃는다. data/는 gitignore 대상.
CACHE_ROOT = Path("data/rerank_cache")


def load_eval_module():
    """04_evaluate.py를 모듈로 불러온다.

    파일명이 숫자로 시작해 `import`가 안 된다. 복사해 쓰지 않는 이유는
    쌍둥이 보정·집계가 **한 곳에만** 있어야 하기 때문이다.
    """
    path = Path(__file__).with_name("04_evaluate.py")
    spec = importlib.util.spec_from_file_location("eval04", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_reranker(name, max_length, fp16):
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  [주의] CUDA를 못 찾았다. CPU 지연은 서비스 판단 근거가 못 된다.")
    kwargs = {"torch_dtype": torch.float16} if (fp16 and device == "cuda") else {}
    print(f"리랭커 로드: {name} · {device} · max_length={max_length}"
          f"{' · fp16' if kwargs else ''}")
    t0 = time.time()
    model = CrossEncoder(name, max_length=max_length, device=device, model_kwargs=kwargs)
    print(f"  로드 {time.time() - t0:.0f}s")
    return model, device


def retrieve(pack, eval_rows, n_cand):
    """벡터 검색으로 질의별 후보 N건을 뽑는다. 순위 계산은 여기서 하지 않는다."""
    import numpy as np

    kinds = np.array(pack["kinds"])
    ids = np.array(pack["ids"])
    D, doc_ids = pack["X"][kinds == "doc"], ids[kinds == "doc"]
    Q, q_ids = pack["X"][kinds == "query"], ids[kinds == "query"]

    cand, times = {}, []
    B = 256
    for s in range(0, len(Q), B):
        t0 = time.perf_counter()
        sims = Q[s:s + B] @ D.T
        # 전체 정렬은 필요 없다. 상위 N만 골라 그 안에서만 정렬한다.
        part = np.argpartition(-sims, n_cand, axis=1)[:, :n_cand]
        elapsed = (time.perf_counter() - t0) * 1000 / sims.shape[0]
        for j in range(sims.shape[0]):
            idx = part[j][np.argsort(-sims[j, part[j]])]
            cand[str(q_ids[s + j])] = [(str(doc_ids[i]), float(sims[j, i])) for i in idx]
            times.append(elapsed)
    return cand, times


def rerank_ranks(model, queries, cand, docs, gold_rank, twin_of, batch_size, chunk=16):
    """후보를 리랭킹해 정답의 새 순위를 만든다.

    ⚠️ 여기서는 질의 여러 개를 한 번에 몰아 넣는다(`chunk`). **지연 측정과 다른
    형태다** — 사용자가 겪는 지연은 measure_latency()가 한 질의씩 재고, 이쪽은
    2,399건을 끝내는 처리량이 목적이다. 둘을 같은 배치로 재면 지연이 실제보다
    좋아 보인다.
    """
    out = {}
    aids = list(cand)
    t0 = time.time()
    for s in range(0, len(aids), chunk):
        group = aids[s:s + chunk]
        pairs, spans = [], []
        for aid in group:
            start = len(pairs)
            pairs += [(queries[aid], docs.get(doc_id, "")) for doc_id, _ in cand[aid]]
            spans.append((aid, start, len(pairs)))
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        for aid, a, b in spans:
            local = scores[a:b]
            order = sorted(range(len(local)), key=lambda i: -local[i])
            ranked_ids = [cand[aid][i][0] for i in order]
            if aid in ranked_ids:
                twins = set(twin_of.get(aid, ()))
                above = ranked_ids[:ranked_ids.index(aid)]
                out[aid] = 1 + sum(1 for d in above if d not in twins)
            else:
                # 후보 밖이면 리랭킹이 건드리지 못한다. 벡터 검색 순위 그대로.
                out[aid] = gold_rank[aid]

        done = s + len(group)
        if done % (chunk * 20) == 0 or done == len(aids):
            el = time.time() - t0
            print(f"  {done:,}/{len(aids):,}  {el:.0f}s  (남은 예상 {el / done * (len(aids) - done):.0f}s)")
    return out


def measure_latency(model, queries, cand, docs, sample, batch_size):
    """**서비스 형태로** 잰다 — 한 요청 = 한 질의 × 후보 N쌍.

    정확도 실행은 질의를 몰아서 배치로 돌리는 게 빠르지만, 그 숫자는 사용자가
    겪는 지연이 아니다. 여기서는 일부러 한 번에 한 질의씩만 돌린다.
    """
    aids = list(cand)[:sample]
    warm = aids[:3]
    for aid in warm:                                   # 첫 호출은 커널 컴파일이 섞인다
        model.predict([(queries[aid], docs.get(d, "")) for d, _ in cand[aid]],
                      batch_size=batch_size, show_progress_bar=False)

    ms = []
    for aid in aids:
        pairs = [(queries[aid], docs.get(d, "")) for d, _ in cand[aid]]
        t0 = time.perf_counter()
        model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        ms.append((time.perf_counter() - t0) * 1000)
    return ms


def checkpoint_path(args, pack):
    """arm 하나의 결과를 담을 파일과, 그 결과가 어떤 조건에서 나왔는지의 지문.

    지문이 다르면 **재사용하지 않는다.** 조건이 바뀌었는데 옛 순위를 그대로 쓰면
    에러 없이 숫자만 틀린다 — 이 프로젝트에서 가장 늦게 발견되는 종류의 사고다.
    """
    key = {
        "reranker": args.reranker,
        "max_length": args.max_length,
        "candidates": args.candidates,
        "fp16": bool(args.fp16),
        "emb": pack["src_dir"],
        "eval": str(args.eval).replace("\\", "/"),
        "limit": args.limit,
    }
    h = hashlib.sha1(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    slug = pack["src_dir"].rstrip("/").split("/")[-1]
    return CACHE_ROOT / f"{slug}__{h}.json", key


def load_checkpoint(path, key):
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return d if d.get("key") == key else None


def save_checkpoint(path, key, ranks, latency):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "ranks": ranks, "latency": latency},
                               ensure_ascii=False), encoding="utf-8")


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser(description="R-2 리랭커 (지연 실측 우선)")
    ap.add_argument("--emb-dir", type=Path, action="append", required=True,
                    help="후보를 뽑을 임베딩 디렉터리(반복 가능). Phase 1은 3종 병행이다")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--candidates", type=int, default=20, help="리랭킹할 후보 수 (R-5에서 50과 비교)")
    ap.add_argument("--max-length", type=int, default=512, help="리랭커 입력 길이 (R-2c에서 320)")
    ap.add_argument("--batch", type=int, default=None,
                    help="리랭킹 배치. 기본은 후보 수 = 한 요청을 한 배치로 (서비스 형태)")
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--accuracy-batch", type=int, default=64,
                    help="정확도 실행의 리랭커 배치. **지연 측정에는 쓰지 않는다** — "
                         "지연은 항상 한 요청(=후보 수)을 한 배치로 잰다")
    ap.add_argument("--chunk", type=int, default=16,
                    help="정확도 실행에서 한 번에 몰아 넣을 질의 수(처리량용). 지연 측정과 무관하다")
    ap.add_argument("--latency-sample", type=int, default=100)
    ap.add_argument("--latency-only", action="store_true",
                    help="지연만 재고 끝낸다. **R-2는 여기서부터 시작한다**")
    ap.add_argument("--limit", type=int, default=None, help="정확도 실행 질의 수 제한(디버그)")
    ap.add_argument("--refresh", action="store_true",
                    help="체크포인트를 무시하고 다시 계산한다")
    ap.add_argument("--docs", type=Path, default=DOCS_DEFAULT)
    ap.add_argument("--eval", type=Path, default=EVAL_DEFAULT)
    # 아래는 04_evaluate.py와 같은 이름·의미
    ap.add_argument("--arm-id", action="append", default=[], metavar="패턴=ID")
    ap.add_argument("--baseline", action="append", default=[], metavar="[arm_id=]실행폴더:arm_id",
                    help="arm마다 자기 B-0을 지정한다 (예: R-2a-A=phase0-embedding:E-7). "
                         "arm_id를 빼면 모든 arm에 같은 기준선을 쓴다")
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--phase", default="1")
    ap.add_argument("--title", default=None)
    ap.add_argument("--measured-at", default=None)
    ap.add_argument("--note", default=None)
    args = ap.parse_args()

    ev = load_eval_module()
    batch = args.batch or args.candidates

    eval_rows = [json.loads(l) for l in args.eval.open(encoding="utf-8")]
    queries = {r["answer_id"]: r["query"] for r in eval_rows}
    twin_of = ev.twin_map(eval_rows)

    print(f"원천 로드: {args.docs}")
    docs = {}
    with args.docs.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs[d["id"]] = d["content"]
    print(f"  문서 {len(docs):,}건")

    # 리랭커는 arm 사이에 공유한다. 모델을 3번 로드할 이유가 없다.
    model, device = load_reranker(args.reranker, args.max_length, args.fp16)

    results, latencies = [], {}
    for emb_dir in args.emb_dir:
        pack = ev.load_model_dir(emb_dir)
        if not pack:
            print(f"[!] 임베딩이 없다: {emb_dir}", file=sys.stderr)
            return 1

        print()
        print("=" * 88)
        print(f"{pack['model']}  —  후보 {args.candidates}건")
        print("=" * 88)
        base = ev.evaluate(pack, eval_rows)       # B-0 순위 (쌍둥이 보정 포함)
        cand, search_ms = retrieve(pack, eval_rows, args.candidates)
        cand = {a: c for a, c in cand.items() if a in base["_ranks"]}
        print(f"  B-0  Hit@1 {base['Hit@1']:.3f}  Hit@5 {base['Hit@5']:.3f}  MRR {base['MRR']:.3f}")
        print(f"  검색 지연 p50 {pct(search_ms, 50):.2f}ms (브루트포스, DB 아님)")

        ck_path, ck_key = checkpoint_path(args, pack)
        cached = None if (args.refresh or args.latency_only) else load_checkpoint(ck_path, ck_key)

        if cached:
            p50, p95, n_lat = cached["latency"]["p50"], cached["latency"]["p95"], cached["latency"]["n"]
            print(f"  체크포인트 재사용 → {ck_path}  (지연 p50 {p50:.0f}ms · p95 {p95:.0f}ms, 표본 {n_lat}건)")
            ranks = cached["ranks"]
        else:
            print(f"  지연 실측 — 한 요청 = 질의 1건 × 후보 {args.candidates}쌍 · {device}"
                  f" · max_length {args.max_length} · batch {batch}")
            lat = measure_latency(model, queries, cand, docs, args.latency_sample, batch)
            p50, p95, n_lat = pct(lat, 50), pct(lat, 95), len(lat)
            print(f"    표본 {n_lat}건   p50 {p50:.0f}ms   p95 {p95:.0f}ms   "
                  f"평균 {statistics.mean(lat):.0f}ms   최대 {max(lat):.0f}ms")
            verdict = ("⚠️ 2초 넘음 — 구성을 먼저 바꿔야 한다" if p95 >= 2000 else
                       "1초 넘음 — 체감 가능. R-2b(작은 모델)·R-2c(짧은 입력)를 볼 것" if p95 >= 1000 else
                       "1초 미만 — 정확도 비교로 넘어가도 된다")
            print(f"    → {verdict}   e2e p95 ≈ {p95 + pct(search_ms, 95):.0f}ms (LLM 생성 제외)")

            if args.latency_only:
                continue

            subset = dict(list(cand.items())[:args.limit]) if args.limit else cand
            print(f"  리랭킹 — 질의 {len(subset):,}건")
            t0 = time.time()
            ranks = rerank_ranks(model, queries, subset, docs, base["_ranks"], twin_of,
                                 args.accuracy_batch, chunk=args.chunk)
            print(f"    총 {time.time() - t0:.0f}s")
            # arm이 끝나는 즉시 남긴다. 다음 arm에서 멈춰도 이건 살아남는다.
            save_checkpoint(ck_path, ck_key, ranks, {"p50": p50, "p95": p95, "n": n_lat})
            print(f"    체크포인트 저장 → {ck_path}")

        latencies[pack["src_dir"]] = (search_ms, p50, p95, n_lat)
        res = ev.summarize(ranks, eval_rows)
        res.update(
            label=f"{pack['model'].split('/')[-1]} + {args.reranker.split('/')[-1]}",
            model=pack["model"], dim=pack["dim"], qp=pack["qp"], pp=pack["pp"],
            max_seq_len=pack["max_seq_len"], src_dir=pack["src_dir"],
            truncate_dim=None, truncated=False, prefix="",
            reranker=args.reranker, top_k_candidates=args.candidates,
            rerank_max_length=args.max_length,
            latency_ms={"search_p50": round(pct(search_ms, 50), 2),
                        "rerank_p50": round(p50), "e2e_p95": round(p95 + pct(search_ms, 95))},
            _b0=base,
        )
        results.append(res)
        # ⚠️ --limit을 주면 R-2는 부분집합, B-0은 전체라 Δ가 성립하지 않는다.
        delta = (f"(B-0 대비 MRR {res['MRR'] - base['MRR']:+.4f})" if not args.limit
                 else "(--limit이라 B-0과 질의 집합이 달라 Δ를 적지 않는다)")
        print(f"    R-2  Hit@1 {res['Hit@1']:.3f}  Hit@5 {res['Hit@5']:.3f}  "
              f"Hit@20 {res['Hit@20']:.3f}  MRR {res['MRR']:.3f}   {delta}")

    if args.latency_only:
        print()
        print("  --latency-only: 정확도는 재지 않았다.")
        return 0

    overrides = []
    for spec in args.arm_id:
        pat, _, aid = spec.partition("=")
        if not aid:
            print(f"[!] --arm-id는 '패턴=ID' 형식이다: {spec!r}", file=sys.stderr)
            return 1
        overrides.append((pat, aid))
    ev.resolve_arm_ids(results, overrides, sweeping_dims=False)

    if not args.save:
        return 0

    # arm마다 자기 B-0과 비교한다. 모델끼리 섞어서 비교하지 않는다.
    bases = {}
    for spec in args.baseline:
        arm, _, ref = spec.partition("=")
        if not ref:
            arm, ref = "*", spec
        bases[arm] = ref
    if not bases:
        print("[!] --baseline이 필요하다 (예: R-2a-A=phase0-embedding:E-7)", file=sys.stderr)
        return 1

    paired, base_refs = [], []
    for res in results:
        ref = bases.get(res["arm_id"], bases.get("*"))
        if not ref:
            print(f"[!] {res['arm_id']}의 기준선이 없다. --baseline {res['arm_id']}=... 를 줄 것.",
                  file=sys.stderr)
            return 1
        b = ev.load_baseline_ranks(ref)
        paired += ev.paired_report([res], b)
        base_refs.append((b.get("run"), b["arm_id"]))

    # 세 arm이 각자 다른 기준선을 쓰면 공통 기준선이 없다 → metrics.json은 null.
    common = base_refs[0] if len(set(base_refs)) == 1 else (None, None)
    run_env = {
        "search": "brute-force inner product (normalized)",
        "bootstrap": {"n": 2000, "seed": 42},
        "truncate_sweep": None,
        "reranker": {"model": args.reranker, "device": device, "fp16": bool(args.fp16),
                     "max_length": args.max_length, "candidates": args.candidates,
                     "latency_batch": batch, "accuracy_batch": args.accuracy_batch,
                     "latency_sample": args.latency_sample,
                     "checkpoint_dir": str(CACHE_ROOT).replace("\\", "/"),
                     "latency_shape": "질의 1건씩 (서비스 형태). 정확도 실행은 질의 %d건씩 몰아서 돌린다"
                                      % args.chunk},
        "note": args.note,
    }
    ev.save_results(args.save, results, paired, eval_rows,
                    base={"run": common[0], "arm_id": common[1]}, run_env=run_env, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
