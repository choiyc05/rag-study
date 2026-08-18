/**
 * docs/ 를 **읽기만** 한다. 이 프런트는 기록처가 아니라 렌더러다.
 *
 * 숫자의 출처는 `04_evaluate.py --save`가 떨군 파일 하나뿐이고, 여기서 다시
 * 계산하지 않는다. 계산을 여기서 한 번 더 하면 같은 값이 두 곳에서 나오고,
 * 어긋나도 아무도 모른다. (docs/results/README.md "Δ는 paired_tests에만 있다")
 *
 * ⚠️ `next dev` 전용이다. cwd가 frontend/라 저장소 루트를 한 단계 올라가서 찾는다.
 *    빌드해서 다른 곳에 올리면 docs/가 따라가지 않으므로 빈 목록이 된다.
 */
import { promises as fs } from "node:fs";
import path from "node:path";

const REPO_ROOT = process.env.RAG_REPO_ROOT ?? path.resolve(process.cwd(), "..");
export const DOCS_DIR = path.join(REPO_ROOT, "docs");
export const RESULTS_DIR = path.join(DOCS_DIR, "results");

/** metrics.json — schema_version 1. 스크립트가 쓰는 형식 그대로다. */
export type Arm = {
  arm_id: string;
  label: string;
  model: string;
  config: {
    dim: number;
    query_prefix: string;
    passage_prefix: string;
    truncate_dim: number | null;
    max_seq_len: number | null;
    embedded_field: string;
    reranker: string | null;
    top_k_candidates: number | null;
    rerank_max_length?: number | null;
    src_dir: string;
  };
  metrics: { "Hit@1": number; "Hit@5": number; "Hit@20": number; MRR: number; n: number };
  /** null = 안 쟀다. 0이 아니다. */
  latency_ms: { search_p50: number | null; rerank_p50: number | null; e2e_p95: number | null };
  groups: Record<string, { n: number; "Hit@5": number; MRR: number }>;
  depts: Record<string, { n: number; "Hit@5": number }>;
};

export type PairedTest = {
  base: { run: string | null; arm_id: string };
  arm_id: string;
  n: number;
  delta_mrr: number;
  ci95: [number, number];
  win: number;
  lose: number;
  tie: number;
  verdict: string;
};

export type Experiment = {
  schema_version: number;
  experiment_id: string;
  phase: string;
  title: string | null;
  question: string | null;
  verdict: string | null;
  measured_at: string | null;
  metric: string;
  run_env: { search: string; bootstrap: { n: number; seed: number }; truncate_sweep: number[] | null; note: string | null };
  index: { file: string; n_docs: number; embedded_field: string };
  queries: { file: string; n: number };
  /** arm마다 기준선이 다르면 null이다 — 그때는 paired_tests[].base를 봐야 한다. */
  baseline: { run: string | null; arm_id: string } | null;
  arms: Arm[];
  paired_tests: PairedTest[];
};

async function readJson<T>(file: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(file, "utf-8")) as T;
  } catch {
    return null;
  }
}

/** `docs/results/<실험>/metrics.json` 을 훑는다. 새 실험을 돌리면 저절로 늘어난다. */
export async function listExperiments(): Promise<Experiment[]> {
  let entries: string[] = [];
  try {
    entries = (await fs.readdir(RESULTS_DIR, { withFileTypes: true }))
      .filter((e) => e.isDirectory())
      .map((e) => e.name);
  } catch {
    return [];
  }
  const found = await Promise.all(
    entries.map((name) => readJson<Experiment>(path.join(RESULTS_DIR, name, "metrics.json"))),
  );
  return found
    .filter((x): x is Experiment => !!x && x.schema_version === 1)
    .sort((a, b) => a.experiment_id.localeCompare(b.experiment_id));
}

export async function getExperiment(id: string): Promise<Experiment | null> {
  if (!isSafeSlug(id)) return null;
  return readJson<Experiment>(path.join(RESULTS_DIR, id, "metrics.json"));
}

/** 실험 폴더의 사람이 읽는 리포트. 없으면 null. */
export async function getExperimentReadme(id: string): Promise<string | null> {
  if (!isSafeSlug(id)) return null;
  try {
    return await fs.readFile(path.join(RESULTS_DIR, id, "README.md"), "utf-8");
  } catch {
    return null;
  }
}

export type RanksTable = {
  /** answer_id · department · orig_is_multi 를 뺀 나머지 열 = arm_id */
  armIds: string[];
  rows: { answer_id: string; department: string; multi: boolean; ranks: (number | null)[] }[];
};

/**
 * ranks.csv — 질의별 정답 순위. **이 폴더의 핵심이자 프런트를 만드는 유일한 이유다.**
 * 집계값은 마크다운으로 충분하지만 2,399행의 질의별 순위는 표로 볼 수 없다.
 */
export async function getRanks(id: string): Promise<RanksTable | null> {
  if (!isSafeSlug(id)) return null;
  let text: string;
  try {
    text = await fs.readFile(path.join(RESULTS_DIR, id, "ranks.csv"), "utf-8");
  } catch {
    return null;
  }
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(",");
  const armIds = header.slice(3);
  const rows = lines.slice(1).map((line) => {
    const c = line.split(",");
    return {
      answer_id: c[0],
      department: c[1],
      multi: c[2] === "1",
      ranks: c.slice(3).map((v) => (v === "" ? null : Number(v))),
    };
  });
  return { armIds, rows };
}

/** docs/*.md — roadmap · experiments · rag-design 등. */
export async function listDocs(): Promise<string[]> {
  try {
    return (await fs.readdir(DOCS_DIR))
      .filter((f) => f.endsWith(".md"))
      .map((f) => f.replace(/\.md$/, ""))
      .sort();
  } catch {
    return [];
  }
}

export async function getDoc(slug: string): Promise<string | null> {
  if (!isSafeSlug(slug)) return null;
  try {
    return await fs.readFile(path.join(DOCS_DIR, `${slug}.md`), "utf-8");
  } catch {
    return null;
  }
}

/** 경로 조작 차단. 로컬 전용이라도 `../../.env`가 읽히는 건 곤란하다. */
function isSafeSlug(s: string): boolean {
  return /^[A-Za-z0-9._-]+$/.test(s) && !s.includes("..");
}

export const fmt = {
  n3: (x: number) => x.toFixed(3),
  d4: (x: number) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(4),
  pp: (x: number) => (x >= 0 ? "+" : "−") + Math.abs(x * 100).toFixed(1) + "%p",
};
