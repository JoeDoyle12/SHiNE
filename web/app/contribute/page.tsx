"use client";

import { useEffect, useRef, useState } from "react";

function useFadeInOut() {
  const ref = useRef<HTMLDivElement | null>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      (entries) => entries.forEach((e) => setVisible(e.isIntersecting)),
      { rootMargin: "-15% 0px -20% 0px", threshold: [0, 0.1, 0.25, 0.5, 0.75, 1] }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return { ref, visible };
}

function Card({ id, children }: { id: string; children: React.ReactNode }) {
  const { ref, visible } = useFadeInOut();
  return (
    <section
      id={id}
      ref={ref}
      className={`rounded-2xl bg-black/35 backdrop-blur-sm p-6 md:p-8 will-change-[opacity,transform]
        transition-all duration-700 ease-out
        ${visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"}`}
    >
      <div className="prose prose-invert prose-lg md:prose-xl max-w-none leading-relaxed">
        {children}
      </div>
    </section>
  );
}

const OWNER  = process.env.NEXT_PUBLIC_GH_OWNER  ?? "owner";
const REPO   = process.env.NEXT_PUBLIC_GH_REPO   ?? "repo";
const BRANCH = process.env.NEXT_PUBLIC_GH_BRANCH ?? "main";
const PATH   = "community_data";

const UPLOADS = `https://github.com/${encodeURIComponent(OWNER)}/${encodeURIComponent(REPO)}/upload/${encodeURIComponent(BRANCH)}/${PATH}`;

export default function Contribute() {
  return (
    <div className="mx-auto max-w-5xl space-y-10 md:space-y-14 text-center">
      <Card id="intro">
        <h1 className="text-3xl md:text-5xl font-bold !mb-3">Contribute Data! </h1>
        <p className="!mt-0 text-neutral-300">
          Help us by appending your own filtered datasets in the
          <code className="mx-1">community_data</code> folder! 
        </p>
      </Card>

      <Card id="how">
        <h2 className="text-2xl md:text-3xl font-semibold !mt-0">How To Contribute!</h2>

        <div className="mx-auto max-w-3xl text-left">
          <ol className="list-decimal ml-6 space-y-3">
            <li>Organize Your Data: Use CSV'S and please provide a short README! </li>
            <li>In the README, include source, license, and brief column descriptions.</li>
            <li>Access And View the folder below to help expand our datasets! </li>
          </ol>

          {/* Single white CTA (folder link removed) */}
          <div className="mt-8 flex justify-center">
            <a
              href={UPLOADS}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center rounded-xl px-5 py-3 text-base font-semibold bg-white text-black hover:opacity-90"
            >
              Contribute Data!
            </a>
          </div>
        </div>
      </Card>
    </div>
  );
}
