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

function Card({ id, children, center = false }:{
  id: string; children: React.ReactNode; center?: boolean;
}) {
  const { ref, visible } = useFadeInOut();
  return (
    <section
      id={id}
      ref={ref}
      className={`rounded-2xl bg-black/35 backdrop-blur-sm p-6 md:p-8 will-change-[opacity,transform]
      transition-all duration-700 ease-out ${visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"}
      ${center ? "text-center" : ""}`}
    >
      <div className="prose prose-invert prose-lg md:prose-xl max-w-none leading-relaxed">
        {children}
      </div>
    </section>
  );
}

function ProfileCard({
  name, href, children,
}: { name: string; href: string; children: React.ReactNode }) {
  const { ref, visible } = useFadeInOut();
  return (
    <div
      ref={ref}
      className={`rounded-2xl bg-black/35 backdrop-blur-sm p-6 md:p-7 h-full transition-all duration-700
      ${visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"}`}
    >
      <h3 className="text-xl md:text-2xl font-semibold !mt-0">
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="text-sky-300 hover:text-sky-200 hover:underline"
        >
          {name}
        </a>
      </h3>
      <p className="text-neutral-300">{children}</p>
    </div>
  );
}

export default function About() {
  return (
    <div className="mx-auto max-w-5xl space-y-10 md:space-y-14">
      {/* Mission */}
      <Card id="mission" center>
        <h1 className="text-3xl md:text-5xl font-bold !mb-3">About Us</h1>
        <p className="!mt-0">
          Three high school seniors who are passionate about Astrophysics and want to find a meaningful and unique way to contribute to the exploration of new exoplanets! 
          
        </p>
      </Card>

      {/* Team — left to right: Andy, Joe, Michael */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8">
        <ProfileCard
          name="Andy Long"
          href="https://www.linkedin.com/in/andy-long-68606b344/"
        >
          Andy Long is a high school senior at Seven Lakes (Katy, TX) who&apos;s passionate about
          observational astrophysics and theoretical physics!
        </ProfileCard>

        <ProfileCard
          name="Joe Doyle"
          href="https://www.linkedin.com/in/joseph-c-doyle/"
        >
          Joe Doyle is a high school senior at Phillips Academy Andover from New York City
          interested in physics, astrophysics, and mathematics!
        </ProfileCard>

        <ProfileCard
          name="Michael Telesco"
          href="https://www.astrotelesco.com/"
        >
          Michael Telesco is a high school senior at New Canaan High School interested in
          astrophysics, exoplanet research, and observational astronomy!
        </ProfileCard>
      </div>
    </div>
  );
}
