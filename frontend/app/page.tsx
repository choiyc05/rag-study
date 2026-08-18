import Link from "next/link";
import { listExperiments } from "@/lib/results";

export default async function Home() {
  const experiments = await listExperiments();
  const arms = experiments.reduce((s, e) => s + e.arms.length, 0);

  return (
    <div className="space-y-10">
      <section>
        <h1 className="text-2xl font-semibold tracking-tight">반려견 QA RAG</h1>
        <p className="mt-3 max-w-2xl text-[15px] leading-7 text-zinc-600 dark:text-zinc-400">
          같은 질문을 <strong className="font-semibold text-zinc-900 dark:text-zinc-100">임베딩 모델 3종</strong>에
          동시에 물어 보고, 그 3종을 어떻게 골랐는지의 근거를 같은 자리에서 본다.
          체험과 기록이 <strong className="font-semibold text-zinc-900 dark:text-zinc-100">같은 arm</strong>을 가리킨다.
        </p>
      </section>

      <section className="grid gap-4 sm:grid-cols-2">
        <Link
          href="/chat"
          className="rounded-lg border border-zinc-200 p-5 transition hover:border-zinc-400 dark:border-zinc-800 dark:hover:border-zinc-600"
        >
          <div className="text-sm font-semibold">체험 — 3종 나란히</div>
          <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
            A · K · B 세 arm에 같은 질문을 던져 검색 결과와 답변을 비교한다.
          </p>
          <p className="mt-3 text-xs text-amber-700 dark:text-amber-500">
            아직 배선 전 — Phase 2에서 실제 검색에 연결된다
          </p>
        </Link>

        <Link
          href="/lab"
          className="rounded-lg border border-zinc-200 p-5 transition hover:border-zinc-400 dark:border-zinc-800 dark:hover:border-zinc-600"
        >
          <div className="text-sm font-semibold">실험 기록 — {experiments.length}건</div>
          <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
            arm {arms}개의 Hit@k · MRR · 짝지은 검정. 질의별 순위까지 파고들 수 있다.
          </p>
          <p className="mt-3 text-xs text-zinc-500">
            출처는 <code className="font-mono">docs/results/</code> — 이 페이지는 렌더러다
          </p>
        </Link>
      </section>
    </div>
  );
}
