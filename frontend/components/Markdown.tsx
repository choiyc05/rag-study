import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * docs/의 마크다운을 그대로 그린다.
 *
 * ⚠️ 문서끼리 `[rag-design.md](rag-design.md)`처럼 상대경로로 촘촘히 엮여 있어서
 *    링크를 그대로 두면 **교차 링크가 전부 404**가 된다. 여기서 라우트로 바꾼다.
 */
function rewrite(href: string | undefined, from: "doc" | "result"): string {
  if (!href) return "#";
  if (/^(https?:)?\/\/|^#|^mailto:/.test(href)) return href;

  // 결과 폴더 README에서 ../../experiments.md → /lab/docs/experiments
  const doc = href.match(/(?:^|\/)([A-Za-z0-9._-]+)\.md(#.*)?$/);
  if (doc) return `/lab/docs/${doc[1]}${doc[2] ?? ""}`;

  // docs/README.md에서 results/phase0-embedding/ → /lab/phase0-embedding
  const result = href.match(/results\/([A-Za-z0-9._-]+)\/?$/);
  if (result) return `/lab/${result[1]}`;

  // 같은 폴더의 형제 실험(결과 README 안에서) — ../phase0-dimension/
  if (from === "result") {
    const sibling = href.match(/^\.\.\/([A-Za-z0-9._-]+)\/?$/);
    if (sibling) return `/lab/${sibling[1]}`;
  }
  return href;
}

export function Markdown({ children, from = "doc" }: { children: string; from?: "doc" | "result" }) {
  return (
    <div className="max-w-none text-[15px] leading-7 text-zinc-800 dark:text-zinc-200">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: (p) => <h1 className="mt-10 mb-4 text-2xl font-semibold tracking-tight" {...p} />,
          h2: (p) => (
            <h2
              className="mt-10 mb-3 border-b border-zinc-200 pb-2 text-xl font-semibold dark:border-zinc-800"
              {...p}
            />
          ),
          h3: (p) => <h3 className="mt-8 mb-2 text-base font-semibold" {...p} />,
          h4: (p) => <h4 className="mt-6 mb-2 text-sm font-semibold" {...p} />,
          p: (p) => <p className="my-4" {...p} />,
          ul: (p) => <ul className="my-4 list-disc space-y-1 pl-6" {...p} />,
          ol: (p) => <ol className="my-4 list-decimal space-y-1 pl-6" {...p} />,
          strong: (p) => <strong className="font-semibold text-zinc-950 dark:text-white" {...p} />,
          blockquote: (p) => (
            <blockquote className="my-4 border-l-2 border-zinc-300 pl-4 text-zinc-600 dark:border-zinc-700 dark:text-zinc-400" {...p} />
          ),
          hr: () => <hr className="my-8 border-zinc-200 dark:border-zinc-800" />,
          a: ({ href, ...p }) => (
            <a
              className="text-blue-600 underline underline-offset-2 hover:text-blue-500 dark:text-blue-400"
              href={rewrite(href, from)}
              {...p}
            />
          ),
          // 표가 문서의 본체다. 넓은 표는 페이지가 아니라 **표가** 가로 스크롤돼야 한다.
          table: (p) => (
            <div className="my-6 overflow-x-auto rounded border border-zinc-200 dark:border-zinc-800">
              <table className="w-full border-collapse text-[13px]" {...p} />
            </div>
          ),
          thead: (p) => <thead className="bg-zinc-50 dark:bg-zinc-900" {...p} />,
          th: (p) => (
            <th className="whitespace-nowrap border-b border-zinc-200 px-3 py-2 text-left font-semibold dark:border-zinc-800" {...p} />
          ),
          td: (p) => (
            <td className="border-b border-zinc-100 px-3 py-2 align-top dark:border-zinc-900" {...p} />
          ),
          code: ({ className, children, ...p }) => {
            const block = /language-/.test(className ?? "");
            return block ? (
              <code className="block font-mono text-[12.5px] leading-6" {...p}>
                {children}
              </code>
            ) : (
              <code
                className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[0.87em] text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100"
                {...p}
              >
                {children}
              </code>
            );
          },
          pre: (p) => (
            <pre
              className="my-5 overflow-x-auto rounded border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-900"
              {...p}
            />
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
