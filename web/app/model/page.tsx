/* eslint-disable react/no-unescaped-entities */
"use client";

import Image from "next/image";
import { useEffect, useRef, useState } from "react";

/* ── small reveal hook for fade-in on scroll ── */
function useReveal() {
  const ref = useRef<HTMLDivElement | null>(null);
  const [show, setShow] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            setShow(true);
            obs.unobserve(e.target);
          }
        });
      },
      { threshold: 0.15 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);
  return { ref, show };
}

/* ── generic card ── */
function Card({ id, children }: { id: string; children: React.ReactNode }) {
  const { ref, show } = useReveal();
  return (
    <section
      id={id}
      ref={ref}
      className={`rounded-2xl bg-black/35 backdrop-blur-sm p-6 md:p-8 transition-all duration-700
      ${show ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}
    >
      <div className="prose prose-invert prose-lg md:prose-xl max-w-none leading-relaxed">
        {children}
      </div>
    </section>
  );
}

/* ── text + image side-by-side ── */
function SideBySide({
  id,
  title,
  imgSrc,
  imgAlt,
  children,
  reverse = false,
}: {
  id: string;
  title: string;
  imgSrc: string;
  imgAlt: string;
  children: React.ReactNode;
  reverse?: boolean;
}) {
  const { ref, show } = useReveal();
  return (
    <section
      id={id}
      ref={ref}
      className={`rounded-2xl bg-black/35 backdrop-blur-sm p-6 md:p-8 transition-all duration-700
      ${show ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}
    >
      <div
        className={`grid items-center gap-6 md:gap-10 md:grid-cols-2 ${
          reverse ? "md:[&>div:first-child]:order-2" : ""
        }`}
      >
        <div>
          <h2 className="text-2xl md:text-3xl font-semibold !mt-0">{title}</h2>
          <div className="prose prose-invert prose-lg md:prose-xl max-w-none leading-relaxed">
            {children}
          </div>
        </div>
        <div className="relative rounded-xl overflow-hidden border border-white/10">
          <Image
            src={imgSrc}
            alt={imgAlt}
            width={1200}
            height={900}
            className="w-full h-auto object-cover"
            priority={false}
          />
        </div>
      </div>
    </section>
  );
}

/* ── page ── */
export default function Model() {
  return (
    <div className="mx-auto max-w-5xl space-y-10 md:space-y-14">
      {/* Intro */}
      <Card id="what-is-shines">
        <h1 className="text-3xl md:text-5xl font-bold !mb-3">What is SHiNES?</h1>
        <p className="!mt-0">
          <strong>SHiNES:</strong> <strong>S</strong>earches For <strong>H</strong>igh-<strong>I</strong>nclination <strong>N</strong>on Transiting <strong>E</strong>xoplanet <strong>S</strong>ystems. In these multi planet systems not all the planets transit. Thus, we rely more on timing and duration variations rather than radial velocity methods and photometry. 

        </p>
      </Card>

      {/* TTV block */}
      <SideBySide
        id="ttv"
        title="Transit Timing Variations (TTV)"
        imgSrc="/transitdepth_illustrations.png"
        imgAlt="Transit-depth and light-curve schematic"
      >
        <ul>
          <li>
            <strong>TTV</strong> are variations in the timing of a planetary transit caused by the gravitational tugs of another "non transiting, hidden" exoplanet. 
          </li>
          <li>We feed normalized TTV sequences into the model alongside TDV Data.</li>
        </ul>
      </SideBySide>

      {/* Architecture */}
      <SideBySide
        id="architecture"
        title="Model Architecture (Big Picture)"
        imgSrc="/model_illustration.png"
        imgAlt="SHiNES model architecture diagram"
        reverse
      >
        <ul>
          <li><strong>Bi-LSTM</strong> encodes TTV/TDV sequences (context from both directions).</li>
          <li>
            <strong>Transformer</strong> with self-attention; multi-pooling (CLS / mean / max) builds a
            robust sequence summary.
          </li>
          <li>
            <strong>MLP head</strong> outputs the probability that a system hosts a high-inclination,
            non-transiting companion.
          </li>
        </ul>
      </SideBySide>

      {/* TDV block — right after architecture with your precession picture */}
      <SideBySide
        id="tdv"
        title="Transit Duration Variations (TDV)"
        imgSrc="/precessionpic.png"
        imgAlt="Nodal precession driving slow transit-duration drifts"
      >
        <ul>
          <li>
            <strong>TDV</strong> are slow changes in how long the transit lasts as the transit chord
            shifts due to nodal precession from mutual inclination.
          </li>
          <li>
            Over multi-year baselines, a gentle duration drift + periodic structure can point to
            tilted companions.
          </li>
          <li>TDV'S can complement TTV data, helping us determine whether a system is high inclination or not!</li>
        </ul>
      </SideBySide>

      {/* Short + sweet remainder */}
      <Card id="training">
        <h2>Training &amp; Data</h2>
        <ul>
          <li>Split: <strong>80-10-10</strong>; features normalized per channel (TTV, TDV).</li>
          <li>Loss: Cross-Entropy; Optimizer: AdamW; regularization via dropout + weight decay.</li>
          <li>Threshold chosen on validation to balance precision/recall (ROC-AUC &amp; PR-AUC tracked).</li>
        </ul>
      </Card>

      <Card id="simulation">
        <h2>Simulation (for training)</h2>
        <ul>
          <li>Sample plausible systems; integrate orbits over multi-year baselines.</li>
          <li>Extract synthetic transit centers &amp; durations → compute TTV/TDV sequences.</li>
          <li>Add survey-like cadence/noise so the model learns realistic signals.</li>
        </ul>
      </Card>

      <Card id="validation">
        <h2>Validation &amp; Follow-up</h2>
        <ul>
          <li>Use model probability to rank candidates.</li>
          <li>Check with TTV tools; attempt dynamical fits (e.g., N-body + MCMC).</li>
          <li>Seek RV / added photometry for confirmation when feasible.</li>
        </ul>
      </Card>
    </div>
  );
}
