"""Phase 0-4 — 지표 1(known-item retrieval)로 모델을 비교한다. DB 없이 numpy로.

    backend/.venv/Scripts/python.exe scripts/04_evaluate.py                    # 전체 모델
    backend/.venv/Scripts/python.exe scripts/04_evaluate.py --model KURE-v1    # 하나만

**arm이 실험의 단위다.** 한 번의 실행이 arm 여러 개를 만들 수 있다 —
같은 모델을 차원만 잘라서(`--truncate-dim 1024,768`), 또는 같은 모델을 조건만
바꿔서(`--emb-dir`로 prefix 있는 것/없는 것을 나란히). 그래서 결과의 키는
모델명이 아니라 **`arm_id`**이고, 이 값이 experiments.md의 ID(E-7, R-2a-A)와 같다.

    # 차원 절단 스윕 — 재임베딩 없이 arm 4개
    ... 04_evaluate.py --model snowflake_snowflake --truncate-dim 1024,768,512,256 \
        --arm-id snowflake=E-7 --save docs/results/phase0-dimension

    # 같은 모델, 조건만 다른 대조군 — 디렉터리를 직접 지목한다
    ... 04_evaluate.py --emb-dir data/embeddings/dragonkue_... --arm-id dragonkue_...=E-6 \
                       --emb-dir data/embeddings_ablation/dragonkue_... --arm-id ablation=E-8

    # 다른 실행을 기준선으로 삼아 짝지어 비교 (Phase 1은 전부 이 형태다)
    ... 04_evaluate.py --baseline phase0-embedding:E-7 --save docs/results/phase1-r1

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

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

EMB_ROOT = Path("data/embeddings")
DOCS_DEFAULT = Path("data/normalized/aihub_qa.jsonl")
EVAL_DEFAULT = Path("data/normalized/evalset_colloquial.jsonl")
RESULTS_ROOT = Path("docs/results")
KS = (1, 5, 20)

# metrics.json의 형식 번호. 프런트/후속 스크립트가 이 값으로 분기한다.
SCHEMA_VERSION = 1

# 사람이 쓰는 칸. 스크립트는 이 키들을 **덮어쓰지 않고 보존한다.**
# 숫자는 스크립트만 쓰고 해석은 사람만 쓴다 — 재실행이 해석을 지우면 안 된다.
HUMAN_FIELDS = ("title", "question", "verdict")


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
        # max_seq_len은 나중에 추가된 컬럼이라 Phase 0에 뽑아둔 벡터엔 없다.
        # 없으면 None — **0으로 채우지 않는다.** "안 쟀다"와 "0이었다"는 다르다.
        msl = (t.column("max_seq_len")[0].as_py()
               if "max_seq_len" in t.column_names else None)
        metas.add((t.column("model_name")[0].as_py(), t.column("dim")[0].as_py(),
                   t.column("query_prefix")[0].as_py(),
                   t.column("passage_prefix")[0].as_py(), msl))

    # 적재와 질의가 다른 모델/prefix로 뽑혔으면 여기서 멈춘다.
    # 이런 사고는 에러 없이 결과만 이상해져서 발견이 늦는다.
    if len(metas) != 1:
        raise SystemExit(f"[!] {d.name}: 조각마다 메타가 다르다 → {metas}\n"
                         "    섞인 임베딩이다. 해당 디렉터리를 지우고 다시 뽑을 것.")
    model_name, dim, qp, pp, msl = metas.pop()
    X = np.vstack(vecs)
    # 정규화 안 된 벡터가 섞이면 코사인이 아니라 내적이 되어 긴 문서가 유리해진다.
    norms = np.linalg.norm(X, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-2):
        print(f"  [주의] {d.name}: 정규화가 안 된 벡터가 있다 "
              f"(norm 최소 {norms.min():.3f} 최대 {norms.max():.3f}). 강제 정규화한다.")
        X = X / np.clip(norms, 1e-9, None)[:, None]
    return dict(model=model_name, dim=dim, qp=qp, pp=pp, max_seq_len=msl,
                src_dir=str(d).replace("\\", "/"), kinds=kinds, ids=ids, X=X)


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
    q_ids_kept = []
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
            q_ids_kept.append(aid)

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
    # 모델 간 '짝지은' 비교용. 질의별 순위를 answer_id로 남긴다.
    # 평균만 비교하면 ΔMRR 0.004가 실력인지 잡음인지 구분이 안 된다.
    out["_ranks"] = {aid: int(rk) for aid, rk in zip(q_ids_kept, r)}
    return out


def resolve_arm_ids(results, overrides, sweeping_dims):
    """arm에 안정적인 키를 붙인다. **모델명을 키로 쓰면 안 되는 이유가 세 개다.**

    ① 같은 모델을 조건만 바꿔 비교하면 이름이 겹친다(prefix 대조군).
    ② 차원 절단은 모델이 하나인데 arm이 넷이다.
    ③ 열 이름이 실행 조합에 따라 달라지면 **ranks.csv를 다른 실행과 join할 수 없다.**

    `--arm-id 패턴=ID`로 지정하고, 패턴은 **임베딩 디렉터리 경로**에 맞춘다
    (모델명은 겹칠 수 있어도 디렉터리는 다르다). 지정이 없으면 모델명 뒤쪽을 쓴다.
    차원 스윕이면 `@1024d`가 자동으로 붙는다.
    """
    for r in results:
        base_id = None
        for pat, aid in overrides:
            if pat.lower() in r["src_dir"].lower():
                base_id = aid
                break
        if base_id is None:
            base_id = r["model"].split("/")[-1]
        r["arm_id"] = f"{base_id}@{r['dim']}d" if sweeping_dims else base_id

    dup = {a for a in (r["arm_id"] for r in results)
           if [r["arm_id"] for r in results].count(a) > 1}
    if dup:
        # 여기서 멈추지 않으면 ranks.csv에 같은 이름의 열이 둘 생기고,
        # 나중에 join할 때 **조용히 한쪽만 남는다.**
        raise SystemExit(
            f"[!] arm_id가 겹친다: {sorted(dup)}\n"
            "    --arm-id '<디렉터리 일부>=<ID>' 로 각각 지정할 것.\n"
            "    예: --arm-id embeddings/dragonkue=E-6 --arm-id ablation/dragonkue=E-8")


def load_baseline_ranks(spec):
    """다른 실행의 ranks.csv에서 기준선 arm의 질의별 순위를 읽는다.

    **Phase 1이 이걸로 돌아간다.** 각 실험은 "자기 모델의 B-0 대비"로 증감을 적는데,
    B-0은 이번 실행이 아니라 phase0-embedding에 있다. 이 함수가 없으면
    기준선을 다시 계산해야 하고, 그러면 **같은 숫자가 두 번 나온다.**
    """
    import csv
    run, _, arm = spec.partition(":")
    if not arm:
        raise SystemExit("[!] --baseline은 '<실행폴더>:<arm_id>' 형식이다. 예: phase0-embedding:E-7")
    path = Path(run) if Path(run).exists() else RESULTS_ROOT / run
    csv_path = path / "ranks.csv"
    if not csv_path.exists():
        raise SystemExit(f"[!] 기준선 원자료가 없다: {csv_path}")
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or arm not in rows[0]:
        cols = [c for c in (rows[0] if rows else {}) if c not in
                ("answer_id", "department", "orig_is_multi")]
        raise SystemExit(f"[!] {csv_path}에 arm '{arm}'이 없다. 있는 arm: {cols}")
    ranks = {r["answer_id"]: int(r[arm]) for r in rows if r.get(arm)}
    print(f"  기준선 ← {csv_path} [{arm}] {len(ranks):,}건")
    return {"run": path.name, "arm_id": arm, "label": f"{path.name}:{arm}", "ranks": ranks}


def paired_report(results, base, n_boot=2000, seed=42):
    """기준선 대비 나머지를 **같은 질의끼리** 비교한다.

    왜 필요한가: 전체 MRR 차이가 0.004면 그게 실력인지 질의 표본의 흔들림인지
    평균만 봐서는 알 수 없다. **같은 질의**에 대한 두 모델의 역순위 차를 직접
    재고, 부트스트랩으로 95% 구간을 잡는다. 구간이 0을 품으면 "차이 없음"이며,
    그때 모델은 점수가 아니라 **차원·속도·라이선스로 고르는 게 맞다.**

    질의를 다시 뽑아 쓰는 부트스트랩이라 질의셋 자체의 표본 오차를 반영한다.
    """
    import numpy as np

    print()
    print("=" * 88)
    print(f"짝지은 비교 — 기준 {base['label']} (같은 질의끼리, 부트스트랩 95% 구간)")
    print("=" * 88)
    print(f"{'상대 arm':<40}{'ΔMRR':>9}{'95% 구간':>20}{'이김':>7}{'짐':>7}{'무':>7}  판정")
    out = []
    for r in results:
        # ⚠️ RNG를 비교마다 **다시 시드한다.** 하나의 rng를 이어 쓰면 앞쪽 비교가
        # 소비한 난수만큼 뒤쪽이 밀려서, **arm을 하나 추가했을 뿐인데 기존 arm의
        # 95% 구간이 바뀐다.** 판정이 뒤집히진 않지만 표를 믿을 수 없게 된다.
        # 같은 리샘플 인덱스를 모든 비교에 쓰는 것은 공통난수법이라 오히려 낫다.
        rng = np.random.default_rng(seed)
        # 기준선 자신은 건너뛴다(같은 실행 안에서 기준을 고른 경우).
        if base.get("run") is None and r["arm_id"] == base["arm_id"]:
            continue
        common = sorted(set(base["ranks"]) & set(r["_ranks"]))
        if not common:
            raise SystemExit(f"[!] {r['arm_id']}: 기준선과 겹치는 질의가 없다. "
                             "평가셋이 서로 다른 것 아닌가?")
        b = np.array([base["ranks"][a] for a in common], dtype=np.float64)
        o = np.array([r["_ranks"][a] for a in common], dtype=np.float64)
        d = 1.0 / o - 1.0 / b                      # 상대 - 기준 (양수면 상대가 좋음)
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boot = d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        win, lose = int((o < b).sum()), int((o > b).sum())
        tie = len(d) - win - lose
        verdict = "차이 없음" if lo <= 0 <= hi else ("상대가 우세" if lo > 0 else "기준이 우세")
        print(f"{r['arm_id']:<40}{d.mean():>+9.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>20}{win:>7,}{lose:>7,}{tie:>7,}  {verdict}")
        out.append({"base": {"run": base.get("run"), "arm_id": base["arm_id"]},
                    "arm_id": r["arm_id"], "n": len(d),
                    "delta_mrr": float(d.mean()), "ci95": [float(lo), float(hi)],
                    "win": win, "lose": lose, "tie": tie, "verdict": verdict})
    print("  ※ 구간이 0을 품으면 이 평가셋으로는 우열을 못 가린다는 뜻이다.")
    print("     그때는 차원(=DB 용량)·속도·라이선스로 고른다.")
    return out


def save_results(out_dir: Path, results, paired, eval_rows, *, base, run_env, args):
    """채점 결과를 기계가 읽을 수 있는 형태로 남긴다.

    왜 파일로 남기는가: 숫자를 손으로 experiments.md에 옮기면 재현이 안 되고,
    무엇보다 **질의별 순위를 잃는다.** 순위를 남겨두면 나중에 R-1~R-6을 이
    기준선과 **짝지어** 비교할 수 있다 — Phase 0을 다시 돌리지 않고도.
    """
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {r["answer_id"]: r for r in eval_rows}
    metrics_path = out_dir / "metrics.json"

    # 사람이 써넣은 해석(title/question/verdict)은 재실행해도 살린다.
    # 이게 없으면 "숫자 다시 뽑았더니 판정이 사라졌다"가 된다.
    kept = {}
    if metrics_path.exists():
        try:
            old = json.loads(metrics_path.read_text(encoding="utf-8"))
            kept = {k: old[k] for k in HUMAN_FIELDS if old.get(k)}
        except json.JSONDecodeError:
            print("  [주의] 기존 metrics.json을 못 읽었다 — 해석 칸을 보존하지 못한다.")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": args.exp_id or out_dir.name,
        "phase": args.phase,
        "title": kept.get("title") or args.title,
        "question": kept.get("question"),
        "verdict": kept.get("verdict"),
        "measured_at": args.measured_at,
        "metric": "known-item retrieval (지표 1)",
        "run_env": run_env,
        "index": {"file": str(DOCS_DEFAULT).replace("\\", "/"), "n_docs": 21606,
                  "embedded_field": "content(질문)"},
        "queries": {"file": str(EVAL_DEFAULT).replace("\\", "/"), "n": results[0]["n"]},
        "baseline": {"run": base.get("run"), "arm_id": base["arm_id"]},
        "arms": [
            {"arm_id": r["arm_id"],
             "label": r["model"].split("/")[-1] + (f" @{r['dim']}d" if r.get("truncated") else ""),
             "model": r["model"],
             "config": {"dim": r["dim"], "query_prefix": r["qp"], "passage_prefix": r["pp"],
                        "truncate_dim": r["truncate_dim"], "max_seq_len": r["max_seq_len"],
                        "embedded_field": "content", "reranker": None,
                        "top_k_candidates": None, "src_dir": r["src_dir"]},
             "metrics": {k: r[k] for k in ("Hit@1", "Hit@5", "Hit@20", "MRR", "n")},
             # ⚠️ null은 "안 쟀다"는 뜻이다. Phase 1 R-2에서 채운다.
             "latency_ms": {"search_p50": None, "rerank_p50": None, "e2e_p95": None},
             "groups": r["_groups"],
             "depts": {k: {"n": len(v), "Hit@5": sum(1 for x in v if x <= 5) / len(v)}
                       for k, v in r["_depts"].items()}}
            for r in results
        ],
        # Δ의 유일한 출처. arm 안에 복제하지 않는다 — 같은 숫자가 두 곳에 있으면
        # 어긋났을 때 아무도 모른다.
        "paired_tests": paired,
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 질의별 순위. 이후 실험과 짝지어 비교하기 위한 원자료.
    # 열 이름은 **arm_id**다. 실행 조합과 무관하게 고정이라 다른 실행의
    # ranks.csv와 answer_id로 join할 수 있다.
    with (out_dir / "ranks.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["answer_id", "department", "orig_is_multi"] + [r["arm_id"] for r in results])
        for aid in sorted(results[0]["_ranks"]):
            m = meta.get(aid, {})
            w.writerow([aid, m.get("department", ""), int(bool(m.get("orig_is_multi")))]
                       + [r["_ranks"].get(aid, "") for r in results])
    print()
    print(f"  저장됨 → {out_dir}/metrics.json · ranks.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description="지표 1 평가 (known-item)")
    ap.add_argument("--emb-root", type=Path, default=EMB_ROOT)
    ap.add_argument("--emb-dir", type=Path, action="append", default=[],
                    help="임베딩 디렉터리를 직접 지목(반복 가능). "
                         "같은 모델의 조건 대조군처럼 서로 다른 루트를 섞을 때 쓴다")
    ap.add_argument("--eval", type=Path, default=EVAL_DEFAULT)
    ap.add_argument("--model", default=None, help="디렉터리명 부분 일치로 하나만 평가")
    ap.add_argument("--save", type=Path, default=None,
                    help="채점 결과를 metrics.json / ranks.csv로 저장할 디렉터리")
    ap.add_argument("--arm-id", action="append", default=[], metavar="패턴=ID",
                    help="임베딩 디렉터리 경로에 '패턴'이 들어가는 arm의 ID를 지정 (반복 가능)")
    ap.add_argument("--baseline", default=None, metavar="실행폴더:arm_id",
                    help="다른 실행의 arm을 짝지은 비교의 기준으로 삼는다 "
                         "(예: phase0-embedding:E-7). 생략하면 이번 실행의 MRR 1위")
    ap.add_argument("--exp-id", default=None, help="experiment_id (생략 시 --save 폴더명)")
    ap.add_argument("--phase", default="0-4")
    ap.add_argument("--title", default=None)
    ap.add_argument("--measured-at", default=None, metavar="YYYY-MM-DD",
                    help="측정일. 생략하면 null로 남는다 — 나중에 손으로 채워도 된다")
    ap.add_argument("--note", default=None, help="run_env에 남길 자유 메모(측정 조건 등)")
    # MRL(Matryoshka) 모델은 벡터 앞부분만 써도 성능이 완만하게 떨어지도록 학습된다.
    # 차원을 **변수 하나로** 재려면 다른 모델을 가져오면 안 되고(모델이 같이 바뀐다)
    # 같은 벡터를 잘라야 한다. 재임베딩이 필요 없어 거의 공짜다.
    ap.add_argument("--truncate-dim", default=None, metavar="N[,N...]",
                    help="벡터를 앞에서 N차원만 쓰고 재정규화 (MRL 절단). "
                         "쉼표로 여러 개를 주면 한 번에 arm 여러 개를 만든다")
    args = ap.parse_args()

    if not args.eval.exists():
        print(f"[!] 평가셋이 없다: {args.eval} — 먼저 02_make_evalset.py", file=sys.stderr)
        return 1
    eval_rows = [json.loads(l) for l in args.eval.open(encoding="utf-8")]

    if args.emb_dir:
        # 직접 지목한 경우엔 **준 순서를 지킨다.** 대조군은 순서가 곧 의미다.
        dirs = list(args.emb_dir)
        missing = [d for d in dirs if not d.is_dir()]
        if missing:
            print(f"[!] 없는 디렉터리: {missing}", file=sys.stderr)
            return 1
    else:
        dirs = sorted(d for d in args.emb_root.glob("*") if d.is_dir())
        if args.model:
            dirs = [d for d in dirs if args.model.lower().replace("-", "_") in d.name]
    if not dirs:
        print(f"[!] 임베딩이 없다: {args.emb_root} — 먼저 03_embed.py", file=sys.stderr)
        return 1

    dims = [int(x) for x in args.truncate_dim.split(",")] if args.truncate_dim else [None]
    overrides = []
    for spec in args.arm_id:
        pat, _, aid = spec.partition("=")
        if not aid:
            print(f"[!] --arm-id는 '패턴=ID' 형식이다: {spec!r}", file=sys.stderr)
            return 1
        overrides.append((pat, aid))

    results = []
    for d in dirs:
        pack = load_model_dir(d)
        if not pack:
            continue
        for td in dims:
            p = dict(pack)
            if td:
                import numpy as np
                if td > p["dim"]:
                    raise SystemExit(f"[!] {p['model']}: {p['dim']}차원을 {td}으로 늘릴 수 없다.")
                X = p["X"][:, :td]
                # 자르면 노름이 1이 아니게 되므로 다시 정규화해야 코사인이 된다.
                X = X / np.clip(np.linalg.norm(X, axis=1), 1e-9, None)[:, None]
                p["X"], p["dim"] = X, td
            print(f"평가 중: {p['model']} (dim {p['dim']})")
            res = evaluate(p, eval_rows)
            res.update(model=p["model"], dim=p["dim"], qp=p["qp"], pp=p["pp"],
                       max_seq_len=p["max_seq_len"], src_dir=p["src_dir"],
                       truncate_dim=td, truncated=bool(td))
            res["prefix"] = f"{p['qp']!r}/{p['pp']!r}" if (p["qp"] or p["pp"]) else "없음"
            results.append(res)

    resolve_arm_ids(results, overrides, sweeping_dims=len(dims) > 1)
    # 출력만 MRR 순으로 본다. results의 순서는 **준 순서 그대로** 둔다 —
    # ranks.csv의 열 순서가 실행마다 바뀌면 diff가 읽히지 않는다.
    ordered = sorted(results, key=lambda r: -r["MRR"])

    print()
    print("=" * 88)
    print("지표 1 — known-item retrieval  (질의: Validation 구어체 변형, 정답: 원본 행 1개)")
    print("=" * 88)
    print(f"{'arm':<12}{'모델':<40}{'차원':>6}{'prefix':>8}"
          f"{'Hit@1':>8}{'Hit@5':>8}{'Hit@20':>8}{'MRR':>8}")
    for r in ordered:
        print(f"{r['arm_id']:<12}{r['model']:<40}{r['dim']:>6}{r['prefix']:>8}"
              f"{r['Hit@1']:>8.3f}{r['Hit@5']:>8.3f}{r['Hit@20']:>8.3f}{r['MRR']:>8.3f}")
    print(f"\n  질의 {results[0]['n']:,}건")

    if args.baseline:
        base = load_baseline_ranks(args.baseline)
    else:
        # 기준선을 안 주면 이번 실행의 MRR 1위. Phase 0은 이걸로 충분했지만
        # ⚠️ Phase 1은 "자기 모델의 B-0 대비"라 **반드시 --baseline을 줘야 한다.**
        top = ordered[0]
        base = {"run": None, "arm_id": top["arm_id"], "label": top["arm_id"],
                "ranks": top["_ranks"]}
    paired = paired_report(results, base) if (args.baseline or len(results) > 1) else []

    for r in ordered:
        print()
        print("─" * 88)
        print(f"[{r['arm_id']}] {r['model']}  하위 그룹")
        print("─" * 88)
        for name, g in r["_groups"].items():
            print(f"  {name:<16}{g['n']:>6,}건   Hit@5 {g['Hit@5']:.3f}   MRR {g['MRR']:.3f}")
        print("  진료과별 Hit@5:", "  ".join(
            f"{k} {sum(1 for x in v if x <= 5)/len(v):.2f}({len(v)})"
            for k, v in sorted(r["_depts"].items(), key=lambda kv: -len(kv[1]))))

    if args.save:
        run_env = {
            "search": "brute-force inner product (normalized)",
            "bootstrap": {"n": 2000, "seed": 42},
            "truncate_sweep": [d for d in dims if d] or None,
            # ⚠️ GPU·정밀도는 임베딩 단계(03_embed.py)의 조건이라 여기선 모른다.
            #    max_seq_len은 arm의 config에 arm별로 남는다.
            "note": args.note,
        }
        save_results(args.save, results, paired, eval_rows,
                     base=base, run_env=run_env, args=args)

    print()
    print("  ⚠️ 절대값은 과대평가다(질의가 정답에서 파생됨). 모델 간 비교에만 쓸 것.")
    print("  ★ experiments.md의 Phase 0 표에 위 숫자를 옮기고 모델·차원을 확정할 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
