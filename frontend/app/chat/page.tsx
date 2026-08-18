"use client";

import Link from "next/link";
import { useState } from "react";
import { API_BASE, ARMS, type ArmKey } from "@/lib/arms";

type Source = { content: string; score: number };
type Answer = { answer: string; sources: Source[] };
type State = Record<ArmKey, { loading: boolean; error: string | null; data: Answer | null }>;

const EMPTY: State = {
  A: { loading: false, error: null, data: null },
  K: { loading: false, error: null, data: null },
  B: { loading: false, error: null, data: null },
};

/**
 * 3종 나란히 체험.
 *
 * ⚠️ 백엔드가 아직 arm을 구분하지 않는다. `POST /chat`은 `{question, top_k}`만 받고
 *    `chat_service._embed()`도 더미다. 그래서 지금은 **세 칸에 같은 응답**이 온다.
 *    Phase 2에서 (1) 요청에 arm 필드 추가 (2) arm별 모델·prefix로 질의 임베딩
 *    (3) repository 실제 쿼리를 붙이면 이 화면이 그대로 살아난다.
 */
export default function ChatPage() {
  const [question, setQuestion] = useState("");
  const [state, setState] = useState<State>(EMPTY);

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setState((s) => {
      const next = { ...s };
      for (const a of ARMS) next[a.key] = { loading: true, error: null, data: null };
      return next;
    });

    await Promise.all(
      ARMS.map(async (arm) => {
        try {
          const res = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            // arm은 아직 백엔드 스키마에 없다. 추가되기 전까지는 무시된다.
            body: JSON.stringify({ question, top_k: 5, arm: arm.key }),
          });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data: Answer = await res.json();
          setState((s) => ({ ...s, [arm.key]: { loading: false, error: null, data } }));
        } catch (err) {
          setState((s) => ({
            ...s,
            [arm.key]: { loading: false, error: err instanceof Error ? err.message : "실패", data: null },
          }));
        }
      }),
    );
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">3종 나란히 체험</h1>
        <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
          같은 질문을 Phase 1 병행 3종에 동시에 던진다. 최종 1종은 Phase 2 진입 시 정한다.
        </p>
      </header>

      <p className="rounded border border-amber-300 bg-amber-50 p-3 text-xs leading-5 text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
        ⚠️ <strong>아직 배선 전이다.</strong> <code className="font-mono">POST /chat</code>은{" "}
        <code className="font-mono">{"{question, top_k}"}</code>만 받고 arm을 구분하지 않으며,{" "}
        <code className="font-mono">chat_service._embed()</code>도 더미다. 지금은 세 칸에 같은 응답이 온다 —
        Phase 2에서 arm 필드와 실제 검색을 붙이면 이 화면이 그대로 살아난다.
      </p>

      <form onSubmit={ask} className="flex gap-2">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="강아지가 밥을 안 먹고 토해요"
          className="flex-1 rounded border border-zinc-300 px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-900"
        />
        <button
          type="submit"
          className="rounded bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-700 dark:bg-zinc-100 dark:text-zinc-900"
        >
          3종에 묻기
        </button>
      </form>

      <div className="grid gap-4 lg:grid-cols-3">
        {ARMS.map((arm) => {
          const s = state[arm.key];
          return (
            <section key={arm.key} className="rounded-lg border border-zinc-200 p-4 dark:border-zinc-800">
              <div className="flex items-baseline gap-2">
                <span className="rounded bg-zinc-900 px-1.5 py-0.5 font-mono text-[11px] text-white dark:bg-zinc-100 dark:text-zinc-900">
                  {arm.key}
                </span>
                <span className="text-xs text-zinc-500">{arm.role}</span>
              </div>
              <p className="mt-2 break-all font-mono text-[11px] text-zinc-500">{arm.model}</p>
              <p className="mt-1 text-[11px] text-zinc-400">
                query prefix: {arm.queryPrefix ? <code className="font-mono">{arm.queryPrefix}</code> : "없음"}
              </p>
              <Link
                href={`/lab/phase0-embedding`}
                className="mt-1 inline-block text-[11px] text-blue-600 hover:underline dark:text-blue-400"
              >
                Phase 0 기준선 {arm.phase0ArmId} →
              </Link>

              <div className="mt-4 min-h-24 border-t border-zinc-100 pt-3 text-sm dark:border-zinc-900">
                {s.loading && <p className="text-zinc-400">묻는 중…</p>}
                {s.error && <p className="text-red-600 dark:text-red-400">에러: {s.error}</p>}
                {s.data && (
                  <>
                    <p className="leading-6">{s.data.answer}</p>
                    {s.data.sources.length > 0 && (
                      <ul className="mt-3 space-y-2">
                        {s.data.sources.map((src, i) => (
                          <li key={i} className="rounded bg-zinc-50 p-2 text-xs dark:bg-zinc-900">
                            <span className="font-mono text-[10px] text-zinc-500">score {src.score.toFixed(3)}</span>
                            <p className="mt-1 line-clamp-4 text-zinc-600 dark:text-zinc-400">{src.content}</p>
                          </li>
                        ))}
                      </ul>
                    )}
                  </>
                )}
                {!s.loading && !s.error && !s.data && <p className="text-zinc-400">질문을 입력하면 여기에 나온다.</p>}
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}
