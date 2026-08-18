"""Phase 0-3 — 문서와 평가 질의를 임베딩해 parquet으로 떨군다.

    backend/.venv/Scripts/python.exe scripts/03_embed.py --model nlpai-lab/KURE-v1
    backend/.venv/Scripts/python.exe scripts/03_embed.py --model BAAI/bge-m3
    backend/.venv/Scripts/python.exe scripts/03_embed.py --model Alibaba-NLP/gte-multilingual-base

로직을 노트북이 아니라 이 파일에 두는 이유(docs/rag-design.md "실행 환경"):
RunPod은 SSH로 .py를 돌리는 환경이라, 노트북은 clone → install → 실행 3줄이면 된다.
모델 3개 비교도 --model만 바꿔 세 번 돌린다.

⚠️ 적재할 때와 질의할 때 **같은 모델·같은 prefix·같은 정규화**여야 한다.
   다르면 에러 없이 결과만 이상해져서 발견이 늦는다. 그래서 무엇을 썼는지
   parquet에 함께 기록하고, 04_evaluate.py가 대조한다.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DOCS_DEFAULT = Path("data/normalized/aihub_qa.jsonl")
EVAL_DEFAULT = Path("data/normalized/evalset_colloquial.jsonl")
OUT_ROOT = Path("data/embeddings")


def slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()


def load_inputs(docs_path: Path, eval_path: Path):
    """임베딩할 텍스트를 (kind, id, text)로 모은다.

    문서는 content(질문)를, 질의는 구어체 변형을 넣는다. 둘을 한 번에 처리하는
    이유는 반드시 **같은 모델**로 뽑혀야 하기 때문이다. 따로 돌리면 나중에
    모델을 헷갈려도 아무 에러가 안 난다.
    """
    items = []
    for line in docs_path.open(encoding="utf-8"):
        r = json.loads(line)
        items.append(("doc", r["id"], r["content"]))
    if eval_path.exists():
        for line in eval_path.open(encoding="utf-8"):
            r = json.loads(line)
            items.append(("query", r["answer_id"], r["query"]))
    else:
        print(f"[주의] 평가셋이 없다: {eval_path} — 문서만 임베딩한다.")
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="문서/질의 임베딩 -> parquet")
    ap.add_argument("--model", required=True)
    ap.add_argument("--docs", type=Path, default=DOCS_DEFAULT)
    ap.add_argument("--eval", type=Path, default=EVAL_DEFAULT)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="직전 시도 실측: 100은 패딩 낭비로 8보다 오히려 느렸다")
    ap.add_argument("--shard-size", type=int, default=2000, help="이 건수마다 parquet 조각으로 flush")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--fp16", action="store_true", default=True)
    # ⚠️ prefix는 모델 카드를 보고 명시적으로 넘긴다. E5 계열은 query:/passage:가
    # 필수인데 빠뜨려도 **에러가 안 나고** 성능만 떨어져서 "이 모델 별로네"라고
    # 오판하게 된다. 후보 3종(bge-m3/KURE-v1/gte)은 prefix가 없는 것으로 알려져
    # 있으나, 반드시 모델 카드로 확인하고 여기에 적을 것.
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--passage-prefix", default="")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not args.docs.exists():
        print(f"[!] 문서가 없다: {args.docs} — 먼저 01_prepare_data.py", file=sys.stderr)
        return 1

    items = load_inputs(args.docs, args.eval)
    out_dir = args.out_root / slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 이어서 돌리기. Colab은 유휴 90분에 끊기고 RunPod spot은 예고 없이 회수된다.
    # 조각 단위로 남겨두고 이미 끝난 (kind,id)는 건너뛴다.
    import pyarrow.parquet as pq
    done = set()
    shards = sorted(out_dir.glob("shard_*.parquet"))
    for sh in shards:
        t = pq.read_table(sh, columns=["kind", "id"])
        done |= set(zip(t.column("kind").to_pylist(), t.column("id").to_pylist()))
    todo = [it for it in items if (it[0], it[1]) not in done]
    print(f"모델 {args.model}")
    print(f"  대상 {len(items):,}건 (완료 {len(done):,} / 남음 {len(todo):,})  조각 {len(shards)}개")
    if not todo:
        print("  새로 뽑을 것이 없다.")
        return 0

    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}  batch={args.batch_size}  fp16={args.fp16}"
          f"  max_seq_len={args.max_seq_len}")
    if device == "cpu":
        print("  [주의] GPU를 못 찾았다. 21,606건을 CPU로 돌리면 매우 느리다.")

    model = SentenceTransformer(args.model, device=device, trust_remote_code=True)
    model.max_seq_length = args.max_seq_len
    if args.fp16 and device == "cuda":
        model = model.half()

    import pyarrow as pa

    next_idx = len(shards)
    buf_meta, buf_vec = [], []
    t0 = time.time()

    def flush():
        nonlocal next_idx, buf_meta, buf_vec
        if not buf_meta:
            return
        tbl = pa.table({
            "kind": pa.array([m[0] for m in buf_meta]),
            "id": pa.array([m[1] for m in buf_meta]),
            "embedding": pa.array(buf_vec, type=pa.list_(pa.float32())),
            # 질의 모델과 문서 모델이 어긋나는 사고를 막는 장치.
            # 04_evaluate.py가 이 값들을 대조해서 다르면 멈춘다.
            "model_name": pa.array([args.model] * len(buf_meta)),
            "dim": pa.array([len(buf_vec[0])] * len(buf_meta), type=pa.int32()),
            "query_prefix": pa.array([args.query_prefix] * len(buf_meta)),
            "passage_prefix": pa.array([args.passage_prefix] * len(buf_meta)),
        })
        pq.write_table(tbl, out_dir / f"shard_{next_idx:05d}.parquet", compression="zstd")
        next_idx += 1
        buf_meta, buf_vec = [], []

    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i:i + args.batch_size]
        texts = [(args.query_prefix if k == "query" else args.passage_prefix) + t
                 for k, _, t in chunk]
        vecs = model.encode(texts, batch_size=len(chunk),
                            normalize_embeddings=True,   # 코사인 = 내적이 되게
                            show_progress_bar=False)
        buf_meta += [(k, i_) for k, i_, _ in chunk]
        buf_vec += [v.astype("float32").tolist() for v in vecs]

        if len(buf_meta) >= args.shard_size:
            flush()
            n = i + len(chunk)
            el = time.time() - t0
            print(f"  {n:,}/{len(todo):,}  {el:.0f}s  "
                  f"({n/max(el,1):.0f}건/s, 남은 시간 ~{(len(todo)-n)/max(n/max(el,1),1e-9)/60:.0f}분)")
    flush()

    dim = None
    for sh in sorted(out_dir.glob("shard_*.parquet")):
        dim = pq.read_table(sh, columns=["dim"]).column("dim")[0].as_py()
        break
    print(f"\n완료 {time.time()-t0:.0f}s · 차원 {dim} · 출력 {out_dir}/")
    print(f"  ★ experiments.md에 차원 {dim}과 prefix"
          f"({args.query_prefix!r}/{args.passage_prefix!r})를 기록할 것")
    print("  다음: scripts/04_evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
