import Link from "next/link";
import { notFound } from "next/navigation";
import { Markdown } from "@/components/Markdown";
import { getExperiment, getExperimentReadme, listExperiments, fmt } from "@/lib/results";

export async function generateStaticParams() {
  return (await listExperiments()).map((e) => ({ experiment: e.experiment_id }));
}

export default async function ExperimentPage({
  params,
}: {
  params: Promise<{ experiment: string }>;
}) {
  const { experiment } = await params;
  const exp = await getExperiment(experiment);
  if (!exp) notFound();
  const readme = await getExperimentReadme(experiment);

  // Δ는 paired_tests에만 있다. 여기서 다시 계산하지 않고 arm에 붙이기만 한다.
  const deltaOf = new Map(exp.paired_tests.map((p) => [p.arm_id, p]));
  const arms = [...exp.arms].sort((a, b) => b.metrics.MRR - a.metrics.MRR);
  const groupNames = [...new Set(exp.arms.flatMap((a) => Object.keys(a.groups)))];
  const hasReranker = exp.arms.some((a) => a.config.reranker);
  const hasLatency = exp.arms.some((a) => a.latency_ms.e2e_p95 !== null);

  return (
    <div className="space-y-12">
      <header>
        <Link href="/lab" className="text-xs text-zinc-500 hover:underline">
          ← 실험 기록
        </Link>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight">{exp.title ?? exp.experiment_id}</h1>
        <p className="mt-2 text-xs text-zinc-500">
          Phase {exp.phase} · {exp.metric} · 인덱스 {exp.index.n_docs.toLocaleString()}건 · 질의{" "}
          {exp.queries.n.toLocaleString()}건{exp.measured_at ? ` · ${exp.measured_at}` : ""}
        </p>
        {exp.question && <p className="mt-4 text-[15px] text-zinc-600 dark:text-zinc-400">{exp.question}</p>}
        {exp.verdict && (
          <p className="mt-3 rounded border-l-2 border-zinc-900 bg-zinc-50 p-4 text-[15px] leading-7 dark:border-zinc-100 dark:bg-zinc-900">
            {exp.verdict}
          </p>
        )}
      </header>

      <section>
        <h2 className="mb-1 text-sm font-semibold">arm별 결과</h2>
        <p className="mb-3 text-xs text-zinc-500">
          {exp.baseline ? (
            <>
              Δ는 기준선{" "}
              <span className="font-mono">
                {exp.baseline.run ? `${exp.baseline.run}:` : ""}
                {exp.baseline.arm_id}
              </span>{" "}
              대비
            </>
          ) : (
            // Phase 1은 "자기 모델의 B-0 대비"라 arm마다 기준선이 다르다.
            <>Δ는 <strong>각 arm의 기준선</strong> 대비 — 모델끼리 섞어서 비교하지 않는다</>
          )}{" "}
          · 95% 구간이 0을 품으면 <strong>차이 없음</strong>
        </p>
        <div className="overflow-x-auto rounded border border-zinc-200 dark:border-zinc-800">
          <table className="w-full border-collapse text-[13px]">
            <thead className="bg-zinc-50 dark:bg-zinc-900">
              <tr>
                {[
                  "arm",
                  "모델",
                  ...(hasReranker ? ["리랭커"] : ["차원", "prefix"]),
                  "Hit@1",
                  "Hit@5",
                  "Hit@20",
                  "MRR",
                  ...(exp.baseline ? [] : ["기준"]),
                  "ΔMRR",
                  "95% 구간",
                  "판정",
                  ...(hasLatency ? ["지연 p95"] : []),
                ].map(
                  (h) => (
                    <th key={h} className="whitespace-nowrap border-b border-zinc-200 px-3 py-2 text-left font-semibold dark:border-zinc-800">
                      {h}
                    </th>
                  ),
                )}
              </tr>
            </thead>
            <tbody>
              {arms.map((a) => {
                const d = deltaOf.get(a.arm_id);
                const isBase = !!exp.baseline && !exp.baseline.run && a.arm_id === exp.baseline.arm_id;
                return (
                  <tr key={a.arm_id} className={isBase ? "bg-blue-50/50 dark:bg-blue-950/20" : undefined}>
                    <td className="whitespace-nowrap border-b border-zinc-100 px-3 py-2 font-mono font-semibold dark:border-zinc-900">
                      {a.arm_id}
                      {isBase && <span className="ml-1 text-[10px] font-normal text-blue-600 dark:text-blue-400">기준</span>}
                    </td>
                    <td className="border-b border-zinc-100 px-3 py-2 dark:border-zinc-900">{a.model}</td>
                    {hasReranker ? (
                      <td className="border-b border-zinc-100 px-3 py-2 text-[12px] dark:border-zinc-900">
                        {a.config.reranker ? (
                          <>
                            {a.config.reranker.split("/").pop()}
                            <span className="ml-1 text-zinc-400">
                              top {a.config.top_k_candidates} · len {a.config.rerank_max_length}
                            </span>
                          </>
                        ) : (
                          <span className="text-zinc-400">없음</span>
                        )}
                      </td>
                    ) : (
                      <>
                        <td className="border-b border-zinc-100 px-3 py-2 text-right tabular-nums dark:border-zinc-900">{a.config.dim}</td>
                        <td className="whitespace-nowrap border-b border-zinc-100 px-3 py-2 dark:border-zinc-900">
                          {a.config.query_prefix ? (
                            <code className="font-mono text-[11px]">{a.config.query_prefix}</code>
                          ) : (
                            <span className="text-zinc-400">없음</span>
                          )}
                        </td>
                      </>
                    )}
                    {(["Hit@1", "Hit@5", "Hit@20", "MRR"] as const).map((k) => (
                      <td key={k} className="border-b border-zinc-100 px-3 py-2 text-right font-mono tabular-nums dark:border-zinc-900">
                        {fmt.n3(a.metrics[k])}
                      </td>
                    ))}
                    {!exp.baseline && (
                      <td className="whitespace-nowrap border-b border-zinc-100 px-3 py-2 font-mono text-[11px] text-zinc-500 dark:border-zinc-900">
                        {d ? `${d.base.run ? `${d.base.run}:` : ""}${d.base.arm_id}` : "—"}
                      </td>
                    )}
                    <td className="border-b border-zinc-100 px-3 py-2 text-right font-mono tabular-nums dark:border-zinc-900">
                      {d ? fmt.d4(d.delta_mrr) : "—"}
                    </td>
                    <td className="whitespace-nowrap border-b border-zinc-100 px-3 py-2 text-right font-mono text-[11px] tabular-nums text-zinc-500 dark:border-zinc-900">
                      {d ? `[${fmt.d4(d.ci95[0])}, ${fmt.d4(d.ci95[1])}]` : "—"}
                    </td>
                    <td className="whitespace-nowrap border-b border-zinc-100 px-3 py-2 dark:border-zinc-900">
                      {d ? (
                        <span className={d.verdict === "차이 없음" ? "text-zinc-500" : "font-medium"}>{d.verdict}</span>
                      ) : (
                        "—"
                      )}
                    </td>
                    {hasLatency && (
                      <td className="border-b border-zinc-100 px-3 py-2 text-right font-mono tabular-nums dark:border-zinc-900">
                        {a.latency_ms.e2e_p95 === null ? (
                          <span className="text-zinc-400">안 쟀음</span>
                        ) : (
                          `${a.latency_ms.e2e_p95}ms`
                        )}
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-zinc-500">
          {hasLatency ? (
            <>
              지연은 <strong>한 요청 = 질의 1건 × 후보 N쌍</strong>의 서비스 형태로 쟀다(정확도 실행의
              배치와 다르다). 표의 값은 검색+리랭킹 p95이고 LLM 생성은 빠져 있다.
            </>
          ) : (
            <>
              지연(latency)은 이 실험에서 재지 않았다 — <span className="font-mono">null</span>은 0이 아니라
              &ldquo;안 쟀다&rdquo;는 뜻이다.
            </>
          )}
        </p>
      </section>

      {groupNames.length > 0 && (
        <section>
          <h2 className="mb-1 text-sm font-semibold">하위 그룹 — Hit@5</h2>
          <p className="mb-3 text-xs text-zinc-500">전체 평균만 보면 복합 질문의 손해가 묻힌다.</p>
          <div className="overflow-x-auto rounded border border-zinc-200 dark:border-zinc-800">
            <table className="w-full border-collapse text-[13px]">
              <thead className="bg-zinc-50 dark:bg-zinc-900">
                <tr>
                  <th className="border-b border-zinc-200 px-3 py-2 text-left font-semibold dark:border-zinc-800">그룹</th>
                  {arms.map((a) => (
                    <th key={a.arm_id} className="border-b border-zinc-200 px-3 py-2 text-right font-mono font-semibold dark:border-zinc-800">
                      {a.arm_id}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {groupNames.map((g) => (
                  <tr key={g}>
                    <td className="whitespace-nowrap border-b border-zinc-100 px-3 py-2 dark:border-zinc-900">
                      {g}{" "}
                      <span className="text-zinc-400">
                        {arms[0].groups[g] ? `(${arms[0].groups[g].n.toLocaleString()})` : ""}
                      </span>
                    </td>
                    {arms.map((a) => (
                      <td key={a.arm_id} className="border-b border-zinc-100 px-3 py-2 text-right font-mono tabular-nums dark:border-zinc-900">
                        {a.groups[g] ? fmt.n3(a.groups[g]["Hit@5"]) : "—"}
                      </td>
                    ))}
                  </tr>
                ))}
                <tr className="bg-zinc-50 dark:bg-zinc-900">
                  <td className="border-b border-zinc-100 px-3 py-2 font-semibold dark:border-zinc-900">격차</td>
                  {arms.map((a) => {
                    const [g1, g2] = groupNames;
                    const gap =
                      g1 && g2 && a.groups[g1] && a.groups[g2]
                        ? a.groups[g1]["Hit@5"] - a.groups[g2]["Hit@5"]
                        : null;
                    return (
                      <td key={a.arm_id} className="border-b border-zinc-100 px-3 py-2 text-right font-mono font-semibold tabular-nums dark:border-zinc-900">
                        {gap === null ? "—" : fmt.pp(gap)}
                      </td>
                    );
                  })}
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section>
        <div className="flex items-baseline justify-between">
          <h2 className="text-sm font-semibold">질의별 순위 (원자료)</h2>
          <Link href={`/lab/${experiment}/ranks`} className="text-sm text-blue-600 hover:underline dark:text-blue-400">
            탐색기 열기 →
          </Link>
        </div>
        <p className="mt-2 text-xs text-zinc-500">
          집계값은 표로 충분하지만, &ldquo;기준선보다 나빠진 질의가 무엇인가&rdquo;는 원자료를 봐야 한다.
        </p>
      </section>

      {readme && (
        <section className="border-t border-zinc-200 pt-6 dark:border-zinc-800">
          <h2 className="text-sm font-semibold text-zinc-500">리포트 — README.md</h2>
          <Markdown from="result">{readme}</Markdown>
        </section>
      )}
    </div>
  );
}
