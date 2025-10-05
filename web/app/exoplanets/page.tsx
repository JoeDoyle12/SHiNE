/* eslint-disable react/no-unescaped-entities */
"use client";

import Image from "next/image";
import { useEffect, useRef, useState } from "react";

function useFadeInOut() {
  const ref = useRef<HTMLDivElement | null>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            setVisible(true);
          } else {
            setVisible(false);
          }
        });
      },
      // Trigger a bit before/after center for a pleasant feel
      { rootMargin: "-15% 0px -20% 0px", threshold: [0, 0.1, 0.25, 0.5, 0.75, 1] }
    );

    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return { ref, visible };
}

/* ── reusable card ── */
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

/* ── text + image side-by-side ── */
function SideBySide({
  id,
  title,
  imgSrc,
  imgAlt,
  gif = false,
  reverse = false,
  children,
}: {
  id: string;
  title: string;
  imgSrc: string;
  imgAlt: string;
  gif?: boolean;
  reverse?: boolean;
  children: React.ReactNode;
}) {
  const { ref, visible } = useFadeInOut();
  return (
    <section
      id={id}
      ref={ref}
      className={`rounded-2xl bg-black/35 backdrop-blur-sm p-6 md:p-8 will-change-[opacity,transform]
        transition-all duration-700 ease-out
        ${visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"}`}
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
            {...(gif ? { unoptimized: true } : {})}
          />
        </div>
      </div>
    </section>
  );
}

/* ── page ── */
export default function Exoplanets() {
  return (
    <div className="mx-auto max-w-5xl space-y-10 md:space-y-14">
      <Card id="intro">
        <h1 className="text-3xl md:text-5xl font-bold !mb-3">Methods Of Exoplanet Detection</h1>
        <p className="!mt-0">
          Astronomers use several complementary techniques to find and characterize planets around
          other stars. Below are the three <strong>big</strong> ones: <em>Direct Imaging</em>,
          <em> Radial Velocity</em>, and <em>Transit Photometry</em>—what each measures, what it’s
          best at, and its main trade-offs.
        </p>
      </Card>

      {/* 1) Direct Imaging */}
      <SideBySide
        id="direct-imaging"
        title="Direct Imaging"
        imgSrc="/direct_exa.png"     // use .gif here if you have the animated version
        imgAlt="Directly imaged exoplanet system orbiting over time"
        gif={false}                  // set to true if you switch to /direct_exa.gif
      >
        <ul>
          <li>
            Captures actual light from the planet by blocking starlight with a <strong>coronagraph</strong>
            and using advanced image processing.
          </li>
          <li>
            Best for <strong>young, hot, wide-orbit</strong> giants (tens of AU) where the planet glows
            in the infrared and is well separated from its star.
          </li>
          <li>
            Gives <strong>astrometry</strong> (on-sky position vs. time), crude spectra (atmospheric features), and
            <strong> true brightness</strong> → constraints on temperature and size.
          </li>
          <li>Trade-off: difficult for small/old/cool or close-in planets.</li>
        </ul>
      </SideBySide>

      {/* 2) Radial Velocity */}
      <SideBySide
        id="radial-velocity"
        title="Radial Velocity (RV)"
        imgSrc="/radial_exa.png"
        imgAlt="Doppler wobble: star’s spectrum shifts due to unseen planet"
        reverse
      >
        <ul>
          <li>
            Measures the star’s <strong>line-of-sight wobble</strong> from Doppler shifts of spectral lines as the
            planet orbits.
          </li>
          <li>
            Yields the planet’s <strong>minimum mass</strong> (<code>M&nbsp;sin&nbsp;i</code>) and <strong>orbital period</strong>;
            most sensitive to <strong>massive, short-period</strong> planets.
          </li>
          <li>
            Combine with a transit (which gives inclination) to upgrade min-mass to <strong>true mass</strong>, completing density.
          </li>
          <li>Main biases: favors close-in, heavier planets; stellar activity can add noise.</li>
        </ul>
      </SideBySide>

      {/* 3) Transit Photometry */}
      <SideBySide
        id="transit-photometry"
        title="Transit Photometry"
        imgSrc="/transit_exa.png"
        imgAlt="Transit geometry and light-curve depth schematic"
      >
        <ul>
          <li>
            Watches a star’s brightness and looks for periodic <strong>dips</strong> when a planet crosses the
            star’s disk.
          </li>
          <li>
            Dip depth gives <strong>planet radius</strong> (relative to the star) and timing gives <strong>period</strong>;
            multi-planet systems can show <strong>TTV/TDV</strong> fingerprints.
          </li>
          <li>
            Enormously productive (e.g., Kepler/TESS) but requires fortuitous alignment—so it misses
            many inclined or wide-orbit planets.
          </li>
        </ul>
      </SideBySide>

      {/* Quick comparison */}
      <Card id="compare">
        <h2>What Each Method Tells You (at a glance)</h2>
        <ul>
          <li>
            <strong>Direct Imaging:</strong> brightness, spectra, on-sky motion → atmosphere & wide orbits.
          </li>
          <li>
            <strong>Radial Velocity:</strong> <strong>mass (M sin i)</strong> and period → complements transits for <strong>true mass</strong>.
          </li>
          <li>
            <strong>Transit Photometry:</strong> <strong>radius</strong> and period → with RV → <strong>density</strong>; TTV/TDV hint at
            additional planets/inclinations.
          </li>
        </ul>
      </Card>
    </div>
  );
}
