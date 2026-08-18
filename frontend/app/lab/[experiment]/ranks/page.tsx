import Link from "next/link";
import { notFound } from "next/navigation";
import { RanksExplorer } from "@/components/RanksExplorer";
import { getExperiment, getRanks } from "@/lib/results";

export default async function RanksPage({
  params,
}: {
  params: Promise<{ experiment: string }>;
}) {
  const { experiment } = await params;
  const [exp, table] = await Promise.all([getExperiment(experiment), getRanks(experiment)]);
  if (!exp || !table) notFound();

  return (
    <div className="space-y-6">
      <header>
        <Link href={`/lab/${experiment}`} className="text-xs text-zinc-500 hover:underline">
          ← {exp.title ?? experiment}
        </Link>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight">질의별 정답 순위</h1>
        <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
          질의 {table.rows.length.toLocaleString()}건 × arm {table.armIds.length}개 ·{" "}
          <code className="font-mono text-[12px]">docs/results/{experiment}/ranks.csv</code>
        </p>
      </header>
      <RanksExplorer table={table} baselineArm={exp.baseline?.arm_id ?? table.armIds[0]} />
    </div>
  );
}
