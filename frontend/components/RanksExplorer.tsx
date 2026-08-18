"use client";

import { useMemo, useState } from "react";
import type { RanksTable } from "@/lib/results";

/**
 * 질의별 정답 순위 탐색기.
 *
 * **프런트가 마크다운보다 값을 더하는 유일한 지점이다.** 집계값(Hit@5, MRR)은
 * 표로 충분하지만, "기준선보다 나빠진 질의가 무엇인가"는 2,399행을 훑어야 나온다.
 * R-6(복합 질문 의미 분해)과 R-2(리랭커) 판정에 실제로 필요한 작업이다.
 *
 * ⚠️ 순위 차는 여기서 계산해도 되지만 **집계 지표는 계산하지 않는다.**
 *    MRR·Hit@k는 metrics.json의 값만 쓴다 — 같은 숫자가 두 곳에서 나오면 안 된다.
 */
export function RanksExplorer({ table, baselineArm }: { table: RanksTable; baselineArm: string }) {
  const { armIds, rows } = table;
  const [base, setBase] = useState(armIds.includes(baselineArm) ? baselineArm : armIds[0]);
  const [cmp, setCmp] = useState(armIds.find((a) => a !== base) ?? armIds[0]);
  const [group, setGroup] = useState<"all" | "multi" | "single">("all");
  const [dept, setDept] = useState("all");
  const [only, setOnly] = useState<"all" | "worse" | "better" | "miss">("all");

  const depts = useMemo(() => [...new Set(rows.map((r) => r.department))].sort(), [rows]);
  const bi = armIds.indexOf(base);
  const ci = armIds.indexOf(cmp);

  const filtered = useMemo(() => {
    return rows
      .filter((r) => (group === "all" ? true : group === "multi" ? r.multi : !r.multi))
      .filter((r) => dept === "all" || r.department === dept)
      .map((r) => ({ ...r, b: r.ranks[bi], c: r.ranks[ci] }))
      .filter((r) => r.b !== null && r.c !== null)
      .filter((r) => {
        if (only === "worse") return r.c! > r.b!;
        if (only === "better") return r.c! < r.b!;
        if (only === "miss") return r.c! > 5; // top-5에 못 든 질의
        return true;
      })
      .sort((x, y) => y.c! - y.b! - (x.c! - x.b!));
  }, [rows, group, dept, only, bi, ci]);

  // 이 화면 안에서만 쓰는 진단값이다. 리포트용 지표가 아니다.
  const worse = filtered.filter((r) => r.c! > r.b!).length;
  const better = filtered.filter((r) => r.c! < r.b!).length;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end gap-3 text-sm">
        <Select label="기준" value={base} onChange={setBase} options={armIds} />
        <Select label="비교" value={cmp} onChange={setCmp} options={armIds} />
        <Select
          label="질문 유형"
          value={group}
          onChange={(v) => setGroup(v as typeof group)}
          options={["all", "multi", "single"]}
          labels={{ all: "전체", multi: "복합 질문 파생", single: "단일 주제 파생" }}
        />
        <Select label="진료과" value={dept} onChange={setDept} options={["all", ...depts]} labels={{ all: "전체" }} />
        <Select
          label="필터"
          value={only}
          onChange={(v) => setOnly(v as typeof only)}
          options={["all", "worse", "better", "miss"]}
          labels={{ all: "전체", worse: "나빠진 것만", better: "좋아진 것만", miss: "비교 arm이 top-5 실패" }}
        />
      </div>

      <p className="text-xs text-zinc-500">
        {filtered.length.toLocaleString()}건 · 나빠짐 {worse.toLocaleString()} · 좋아짐 {better.toLocaleString()} ·
        무승부 {(filtered.length - worse - better).toLocaleString()}
        <span className="ml-2 text-zinc-400">(순위 차 큰 순. 화면 필터용 집계이며 리포트 지표가 아니다)</span>
      </p>

      <div className="overflow-x-auto rounded border border-zinc-200 dark:border-zinc-800">
        <table className="w-full border-collapse text-[13px]">
          <thead className="bg-zinc-50 dark:bg-zinc-900">
            <tr>
              {["answer_id", "진료과", "유형", `${base} 순위`, `${cmp} 순위`, "차이"].map((h) => (
                <th key={h} className="whitespace-nowrap border-b border-zinc-200 px-3 py-2 text-left font-semibold dark:border-zinc-800">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, 200).map((r) => {
              const diff = r.c! - r.b!;
              return (
                <tr key={r.answer_id}>
                  <td className="border-b border-zinc-100 px-3 py-1.5 font-mono text-[11px] text-zinc-500 dark:border-zinc-900">
                    {r.answer_id.slice(0, 8)}
                  </td>
                  <td className="border-b border-zinc-100 px-3 py-1.5 dark:border-zinc-900">{r.department}</td>
                  <td className="border-b border-zinc-100 px-3 py-1.5 dark:border-zinc-900">
                    {r.multi ? "복합" : "단일"}
                  </td>
                  <td className="border-b border-zinc-100 px-3 py-1.5 text-right font-mono tabular-nums dark:border-zinc-900">
                    {r.b}
                  </td>
                  <td className="border-b border-zinc-100 px-3 py-1.5 text-right font-mono tabular-nums dark:border-zinc-900">
                    {r.c}
                  </td>
                  <td
                    className={`border-b border-zinc-100 px-3 py-1.5 text-right font-mono tabular-nums dark:border-zinc-900 ${
                      diff > 0 ? "text-red-600 dark:text-red-400" : diff < 0 ? "text-emerald-600 dark:text-emerald-400" : "text-zinc-400"
                    }`}
                  >
                    {diff > 0 ? `+${diff}` : diff}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {filtered.length > 200 && (
        <p className="text-xs text-zinc-500">상위 200건만 표시 — 필터를 좁힐 것.</p>
      )}
    </div>
  );
}

function Select({
  label,
  value,
  onChange,
  options,
  labels = {},
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
  labels?: Record<string, string>;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[11px] text-zinc-500">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border border-zinc-300 bg-white px-2 py-1 text-sm dark:border-zinc-700 dark:bg-zinc-900"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {labels[o] ?? o}
          </option>
        ))}
      </select>
    </label>
  );
}
