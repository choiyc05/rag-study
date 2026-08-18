import Link from "next/link";
import { listExperiments, listDocs, fmt } from "@/lib/results";

export default async function LabIndex() {
  const experiments = await listExperiments();
  const docs = await listDocs();

  return (
    <div className="space-y-12">
      <section>
        <h1 className="text-2xl font-semibold tracking-tight">실험 기록</h1>
        <p className="mt-3 max-w-2xl text-[15px] leading-7 text-zinc-600 dark:text-zinc-400">
          <code className="font-mono text-[13px]">docs/results/*/metrics.json</code> 을 그대로 읽는다.
          숫자를 여기에 적지 않는다 — 새 실험을 돌리면 이 목록에 저절로 생긴다.
        </p>
      </section>

      {experiments.length === 0 && (
        <p className="rounded border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
          결과를 못 찾았다. <code className="font-mono">docs/results/</code> 가 저장소 루트 기준으로
          보이는지 확인할 것 (이 페이지는 <code className="font-mono">next dev</code> 전용이다).
        </p>
      )}

      <section className="space-y-6">
        {experiments.map((e) => {
          const best = [...e.arms].sort((a, b) => b.metrics.MRR - a.metrics.MRR)[0];
          return (
            <article
              key={e.experiment_id}
              className="rounded-lg border border-zinc-200 p-5 dark:border-zinc-800"
            >
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <Link href={`/lab/${e.experiment_id}`} className="text-base font-semibold hover:underline">
                  {e.title ?? e.experiment_id}
                </Link>
                <span className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[11px] text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
                  Phase {e.phase}
                </span>
                <span className="text-xs text-zinc-500">
                  arm {e.arms.length}개 · 질의 {e.queries.n.toLocaleString()}건
                  {e.measured_at ? ` · ${e.measured_at}` : ""}
                </span>
              </div>

              {e.question && (
                <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">답하려는 질문: {e.question}</p>
              )}
              {e.verdict && <p className="mt-3 text-sm leading-6">{e.verdict}</p>}

              <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-zinc-500">
                <span>
                  최고 <span className="font-mono text-zinc-700 dark:text-zinc-300">{best.arm_id}</span> MRR{" "}
                  <span className="font-mono">{fmt.n3(best.metrics.MRR)}</span>
                </span>
                <span>
                  기준선{" "}
                  <span className="font-mono">
                    {e.baseline.run ? `${e.baseline.run}:` : ""}
                    {e.baseline.arm_id}
                  </span>
                </span>
                <Link href={`/lab/${e.experiment_id}/ranks`} className="text-blue-600 hover:underline dark:text-blue-400">
                  질의별 순위 →
                </Link>
              </div>
            </article>
          );
        })}
      </section>

      <section>
        <h2 className="text-sm font-semibold text-zinc-500">문서</h2>
        <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-2 text-sm">
          {docs.map((d) => (
            <li key={d}>
              <Link href={`/lab/docs/${d}`} className="text-blue-600 hover:underline dark:text-blue-400">
                {d}.md
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
