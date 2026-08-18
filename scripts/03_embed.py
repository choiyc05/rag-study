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

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

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
    # 길이 정렬 + 토큰 예산 배치. "배치 100이 8보다 느렸다"는 직전 시도 기록은
    # **길이 정렬을 안 했을 때** 얘기다. 정렬하면 한 배치의 문장 길이가 고르게
    # 모여 패딩 낭비가 사라지므로 큰 배치가 유리해진다(docs/rag-design.md).
    # 배치 크기를 고정하지 않고 `글자예산 / 배치최장길이`로 잡는 이유는 6GB에서
    # 긴 문장 배치만 OOM이 나기 때문이다. 짧으면 크게, 길면 작게 자동으로 잡힌다.
    ap.add_argument("--batch-size", type=int, default=32, help="배치 상한(건수)")
    ap.add_argument("--char-budget", type=int, default=8000,
                    help="배치 하나의 (건수 x 최장 글자수) 상한. 낮추면 VRAM을 덜 쓴다")
    ap.add_argument("--shard-size", type=int, default=2000, help="이 건수마다 parquet 조각으로 flush")
    ap.add_argument("--max-seq-len", type=int, default=1024,
                    help="실측 p99=628토큰, 1024 초과 0.03% — 사실상 무손실")
    ap.add_argument("--limit", type=int, default=0, help="앞 N건만 (연기 테스트용)")
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                    help="--no-fp16으로 끌 수 있다")
    # ⚠️ prefix는 이 스크립트가 가장 조용히 틀리는 지점이다. 빠뜨려도 **에러가
    # 안 나고** 성능만 떨어져서 "이 모델 별로네"라고 오판하게 된다.
    #
    # 그래서 리터럴을 손으로 적는 대신 **모델이 자기 config에 등록해둔 이름**을
    # 쓰는 걸 기본으로 한다(`--query-prompt-name query`). 리터럴을 추측하지 않게
    # 되고, 등록돼 있지 않으면 **조용히 무시되는 대신 즉시 죽는다.**
    #   arctic-ko / PIXIE-Rune : {"query": "query: "}  → --query-prompt-name query
    #   bge-m3 / KURE-v1 / gte : prompts 없음          → 아무것도 넘기지 않는다
    ap.add_argument("--query-prompt-name", default=None,
                    help="모델 config에 등록된 프롬프트 이름 (예: query)")
    ap.add_argument("--passage-prompt-name", default=None)
    # 등록돼 있지 않은 모델에 리터럴을 직접 넣어야 할 때만 쓴다.
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
    if args.limit:
        todo = todo[:args.limit]
    # 긴 것부터. 내림차순이면 **가장 무거운 배치가 맨 처음**에 오므로 OOM이
    # 90% 지점이 아니라 시작 직후에 드러난다. 순서는 결과에 영향이 없다 —
    # 재개는 (kind,id) 집합으로 하고, 문장별 임베딩은 배치 구성과 무관하다.
    todo.sort(key=lambda it: -len(it[2]))
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

    # 등록된 프롬프트를 해석한다. 이름을 줬는데 실제로는 안 붙는 상황에서
    # **여기서 죽인다** — 그냥 넘어가면 prefix 없이 임베딩된 걸 모른 채
    # "이 모델 성능이 낮네"로 끝난다.
    #
    # ⚠️ `model.prompts`를 존재 여부로 검사하면 안 된다. sentence-transformers 5.x는
    #    모델이 아무것도 등록하지 않아도 {'query': '', 'document': ''}를 채워 넣는다.
    #    (실측: bge-m3 → {'query': '', 'document': ''}). 이름은 항상 있고 값만 비어
    #    있으므로, **값이 비면 실패로 본다.**
    registered = getattr(model, "prompts", None) or {}
    def resolve(name, literal, what):
        if name is None:
            return literal
        if not registered.get(name):
            print(f"[!] {args.model}: '{name}' 프롬프트가 비어 있다"
                  f" (등록분 {registered!r}).", file=sys.stderr)
            print("    이름을 넘겼는데 실제로 붙는 문자열이 없다 ="
                  " prefix 없이 임베딩된다.", file=sys.stderr)
            print(f"    모델 카드를 확인하고 --{what}-prompt-name을 고치거나,"
                  " prefix가 없는 모델이면 이 옵션을 빼고 돌릴 것.", file=sys.stderr)
            raise SystemExit(2)
        return registered[name]
    qp = resolve(args.query_prompt_name, args.query_prefix, "query")
    pp = resolve(args.passage_prompt_name, args.passage_prefix, "passage")
    print(f"  prompt  query={qp!r}  passage={pp!r}"
          f"  (모델 등록분: {sorted(registered) or '없음'})")

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
            # 인자로 받은 이름이 아니라 **실제로 붙인 문자열**을 남긴다.
            "query_prefix": pa.array([qp] * len(buf_meta)),
            "passage_prefix": pa.array([pp] * len(buf_meta)),
        })
        pq.write_table(tbl, out_dir / f"shard_{next_idx:05d}.parquet", compression="zstd")
        next_idx += 1
        buf_meta, buf_vec = [], []

    def batches(rows):
        """길이 내림차순으로 정렬된 rows를 (건수 x 최장길이) 예산으로 묶는다."""
        cur = []
        for r in rows:
            longest = max(len(r[2]), len(cur[0][2]) if cur else 0)
            if cur and ((len(cur) + 1) * longest > args.char_budget
                        or len(cur) >= args.batch_size):
                yield cur
                cur = []
            cur.append(r)
        if cur:
            yield cur

    # kind별로 나눠 돌린다. 문서와 질의는 prompt가 다르므로 한 배치에 섞으면
    # 안 된다. 길이 정렬은 각 그룹 안에서 그대로 유지된다.
    done_n, last_report = 0, 0.0
    for kind, prefix in (("doc", pp), ("query", qp)):
        rows = [it for it in todo if it[0] == kind]
        if not rows:
            continue
        for chunk in batches(rows):
            vecs = model.encode([prefix + t for _, _, t in chunk],
                                batch_size=len(chunk),
                                normalize_embeddings=True,   # 코사인 = 내적이 되게
                                show_progress_bar=False)
            buf_meta += [(k, i_) for k, i_, _ in chunk]
            buf_vec += [v.astype("float32").tolist() for v in vecs]
            done_n += len(chunk)

            if len(buf_meta) >= args.shard_size:
                flush()
            el = time.time() - t0
            if el - last_report >= 20:                   # flush 주기와 분리
                last_report = el
                rate = done_n / max(el, 1e-9)
                print(f"  {done_n:,}/{len(todo):,}  {el:.0f}s  "
                      f"({rate:.0f}건/s, 남은 시간 ~{(len(todo)-done_n)/max(rate,1e-9)/60:.0f}분)")
    flush()

    dim = None
    for sh in sorted(out_dir.glob("shard_*.parquet")):
        dim = pq.read_table(sh, columns=["dim"]).column("dim")[0].as_py()
        break
    print(f"\n완료 {time.time()-t0:.0f}s · 차원 {dim} · 출력 {out_dir}/")
    print(f"  ★ experiments.md에 차원 {dim}과 prefix({qp!r}/{pp!r})를 기록할 것")
    print("  다음: scripts/04_evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
