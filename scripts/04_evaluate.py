"""Phase 0-4 — 지표 1(known-item retrieval)로 모델을 비교한다. DB 없이 numpy로.

    backend/.venv/Scripts/python.exe scripts/04_evaluate.py                    # 전체 모델
    backend/.venv/Scripts/python.exe scripts/04_evaluate.py --model KURE-v1    # 하나만

19,206 × 1024 float32 = 79MB라 통째로 메모리에 올라가고 브루트포스 유사도가
1초 미만이다. **여기까지 DB가 필요 없다** — 차원이 안 정해졌는데 테이블부터
만들면 세 번 다시 만들게 된다(docs/roadmap.md Phase 0).

채점 방식:
    질의(구어체 변형)로 21,606건을 검색 → 그 질의가 나온 원본 행이 몇 위인가.
    정답이 인덱스 안에 정확히 1개라 완전 자동 채점된다. LLM도 사람도 필요 없다.

⚠️ 절대값은 과대평가된다. 질의가 정답 행에서 파생된 것이라
   "Hit@5 = 0.87"을 실력이라 믿으면 안 되고 **모델 A vs B 비교에만** 쓴다.
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

EMB_ROOT = Path("data/embeddings")
DOCS_DEFAULT = Path("data/normalized/aihub_qa.jsonl")
EVAL_DEFAULT = Path("data/normalized/evalset_colloquial.jsonl")
KS = (1, 5, 20)


def load_model_dir(d: Path):
    """조각 parquet을 모아 (kind, id, 벡터행렬)로 만든다. 메타 일관성도 검사한다."""
    import numpy as np
    import pyarrow.parquet as pq

    shards = sorted(d.glob("shard_*.parquet"))
    if not shards:
        return None
    kinds, ids, vecs, metas = [], [], [], set()
    for sh in shards:
        t = pq.read_table(sh)
        kinds += t.column("kind").to_pylist()
        ids += t.column("id").to_pylist()
        vecs.append(np.array(t.column("embedding").to_pylist(), dtype=np.float32))
        metas.add((t.column("model_name")[0].as_py(), t.column("dim")[0].as_py(),
                   t.column("query_prefix")[0].as_py(),
                   t.column("passage_prefix")[0].as_py()))

    # 적재와 질의가 다른 모델/prefix로 뽑혔으면 여기서 멈춘다.
    # 이런 사고는 에러 없이 결과만 이상해져서 발견이 늦는다.
    if len(metas) != 1:
        raise SystemExit(f"[!] {d.name}: 조각마다 메타가 다르다 → {metas}\n"
                         "    섞인 임베딩이다. 해당 디렉터리를 지우고 다시 뽑을 것.")
    model_name, dim, qp, pp = metas.pop()
    X = np.vstack(vecs)
    # 정규화 안 된 벡터가 섞이면 코사인이 아니라 내적이 되어 긴 문서가 유리해진다.
    norms = np.linalg.norm(X, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-2):
        print(f"  [주의] {d.name}: 정규화가 안 된 벡터가 있다 "
              f"(norm 최소 {norms.min():.3f} 최대 {norms.max():.3f}). 강제 정규화한다.")
        X = X / np.clip(norms, 1e-9, None)[:, None]
    return dict(model=model_name, dim=dim, qp=qp, pp=pp, kinds=kinds, ids=ids, X=X)


def evaluate(pack, eval_rows):
    import numpy as np

    kinds = np.array(pack["kinds"])
    ids = np.array(pack["ids"])
    doc_mask, q_mask = kinds == "doc", kinds == "query"

    D, doc_ids = pack["X"][doc_mask], ids[doc_mask]
    Q, q_ids = pack["X"][q_mask], ids[q_mask]
    if len(Q) == 0:
        raise SystemExit(f"[!] {pack['model']}: 질의 임베딩이 없다. 03_embed.py를 평가셋과 함께 돌릴 것.")

    pos = {d: i for i, d in enumerate(doc_ids)}          # 정답 행 -> 인덱스 위치
    meta = {r["answer_id"]: r for r in eval_rows}

    # 같은 구어체 질의가 서로 다른 원본에서 나오는 경우가 있다(실측 5건/2,399).
    # 그러면 정답이 둘인데 하나만 정답으로 치므로, 쌍둥이가 위에 오면 순위가
    # 부당하게 밀린다. 순위를 셀 때 '같은 질의를 공유하는 다른 정답'은 제외한다.
    twins = {}
    for r in eval_rows:
        twins.setdefault(r["query"], []).append(r["answer_id"])
    twin_of = {r["answer_id"]: [a for a in twins[r["query"]] if a != r["answer_id"]]
               for r in eval_rows}
    n_twin = sum(1 for v in twin_of.values() if v)
    if n_twin:
        print(f"  [보정] 질의가 겹치는 정답 {n_twin}건 — 순위 계산에서 쌍둥이 제외")

    ranks = []
    keep = []
    # 질의를 나눠서 처리한다. 2,400 × 21,606 float32 = 200MB라 한 번에도 되지만
    # 질의가 늘어나면 터지므로 처음부터 배치로 짜둔다.
    B = 256
    for s in range(0, len(Q), B):
        sims = Q[s:s + B] @ D.T                          # 정규화돼 있으므로 내적 = 코사인
        for j in range(sims.shape[0]):
            aid = q_ids[s + j]
            if aid not in pos:
                continue
            gold = pos[aid]
            # 정답보다 유사도가 높은 문서 수 + 1 = 순위. 전체 정렬보다 훨씬 싸다.
            better = int((sims[j] > sims[j, gold]).sum())
            # 쌍둥이(같은 질의를 공유하는 다른 정답)가 위에 있으면 빼준다.
            for t in twin_of.get(aid, ()):
                ti = pos.get(t)
                if ti is not None and sims[j, ti] > sims[j, gold]:
                    better -= 1
            ranks.append(better + 1)
            keep.append(meta.get(aid, {}))

    r = np.array(ranks)
    out = {f"Hit@{k}": float((r <= k).mean()) for k in KS}
    out["MRR"] = float((1.0 / r).mean())
    out["n"] = len(r)

    # 하위 그룹. 전체 평균만 보면 얇은 진료과와 복합 질문의 개선이 묻힌다.
    groups = {}
    for name, sel in [
        ("복합 질문 파생", [i for i, m in enumerate(keep) if m.get("orig_is_multi")]),
        ("단일 주제 파생", [i for i, m in enumerate(keep) if m and not m.get("orig_is_multi")]),
    ]:
        if sel:
            rr = r[sel]
            groups[name] = {"n": len(rr), "Hit@5": float((rr <= 5).mean()),
                            "MRR": float((1.0 / rr).mean())}
    depts = {}
    for m, rank in zip(keep, r):
        depts.setdefault(m.get("department", "?"), []).append(rank)
    out["_groups"], out["_depts"] = groups, depts
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="지표 1 평가 (known-item)")
    ap.add_argument("--emb-root", type=Path, default=EMB_ROOT)
    ap.add_argument("--eval", type=Path, default=EVAL_DEFAULT)
    ap.add_argument("--model", default=None, help="디렉터리명 부분 일치로 하나만 평가")
    args = ap.parse_args()

    if not args.eval.exists():
        print(f"[!] 평가셋이 없다: {args.eval} — 먼저 02_make_evalset.py", file=sys.stderr)
        return 1
    eval_rows = [json.loads(l) for l in args.eval.open(encoding="utf-8")]

    dirs = sorted(d for d in args.emb_root.glob("*") if d.is_dir())
    if args.model:
        dirs = [d for d in dirs if args.model.lower().replace("-", "_") in d.name]
    if not dirs:
        print(f"[!] 임베딩이 없다: {args.emb_root} — 먼저 03_embed.py", file=sys.stderr)
        return 1

    results = []
    for d in dirs:
        pack = load_model_dir(d)
        if not pack:
            continue
        print(f"평가 중: {pack['model']} (dim {pack['dim']})")
        res = evaluate(pack, eval_rows)
        res["model"], res["dim"] = pack["model"], pack["dim"]
        res["prefix"] = f"{pack['qp']!r}/{pack['pp']!r}" if (pack["qp"] or pack["pp"]) else "없음"
        results.append(res)

    results.sort(key=lambda r: -r["MRR"])

    print()
    print("=" * 88)
    print("지표 1 — known-item retrieval  (질의: Validation 구어체 변형, 정답: 원본 행 1개)")
    print("=" * 88)
    print(f"{'모델':<40}{'차원':>6}{'prefix':>8}{'Hit@1':>8}{'Hit@5':>8}{'Hit@20':>8}{'MRR':>8}")
    for r in results:
        print(f"{r['model']:<40}{r['dim']:>6}{r['prefix']:>8}"
              f"{r['Hit@1']:>8.3f}{r['Hit@5']:>8.3f}{r['Hit@20']:>8.3f}{r['MRR']:>8.3f}")
    print(f"\n  질의 {results[0]['n']:,}건")

    for r in results:
        print()
        print("─" * 88)
        print(f"{r['model']}  하위 그룹")
        print("─" * 88)
        for name, g in r["_groups"].items():
            print(f"  {name:<16}{g['n']:>6,}건   Hit@5 {g['Hit@5']:.3f}   MRR {g['MRR']:.3f}")
        print("  진료과별 Hit@5:", "  ".join(
            f"{k} {sum(1 for x in v if x <= 5)/len(v):.2f}({len(v)})"
            for k, v in sorted(r["_depts"].items(), key=lambda kv: -len(kv[1]))))

    print()
    print("  ⚠️ 절대값은 과대평가다(질의가 정답에서 파생됨). 모델 간 비교에만 쓸 것.")
    print("  ★ experiments.md의 Phase 0 표에 위 숫자를 옮기고 모델·차원을 확정할 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
