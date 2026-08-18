import Link from "next/link";
import { notFound } from "next/navigation";
import { Markdown } from "@/components/Markdown";
import { getDoc, listDocs } from "@/lib/results";

export async function generateStaticParams() {
  return (await listDocs()).map((slug) => ({ slug }));
}

export default async function DocPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const md = await getDoc(slug);
  if (!md) notFound();
  const docs = await listDocs();

  return (
    <div className="space-y-6">
      <nav className="flex flex-wrap gap-x-4 gap-y-1 border-b border-zinc-200 pb-3 text-xs dark:border-zinc-800">
        {docs.map((d) => (
          <Link
            key={d}
            href={`/lab/docs/${d}`}
            className={
              d === slug
                ? "font-semibold text-zinc-900 dark:text-zinc-100"
                : "text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100"
            }
          >
            {d}
          </Link>
        ))}
      </nav>
      <Markdown from="doc">{md}</Markdown>
    </div>
  );
}
