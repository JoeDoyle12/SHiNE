  "use client";

  export default function Home() {
    return (
      <section className="text-center mx-auto max-w-6xl space-y-8 py-8 md:py-12">
        <h1 className="text-3xl md:text-4xl font-bold">A World Away: Hunting For Exoplanets With AI!</h1>
        <p className="text-neutral-300 max-w-3xl mx-auto">
          Here you can explore our code, contribute to the community datasets, and see how we came
          up with our model!
        </p>

  <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-12 place-items-center">
          <Feature title="What Is SHINES?" href="/model" />
          <Feature title="How Do We Detect Exoplanets?" href="/exoplanets" />
          <Feature title="Contribute!" href="/contribute" />
        </div>
      </section>
    );
  }

  function Feature({ title, href }: { title: string; href: string }) {
    return (
      <a
        href={href}
        className="block rounded-2xl p-5 bg-white/5 hover:bg-white/10 text-center w-full md:w-[360px]"
      >
        <div className="text-lg font-semibold">{title}</div>
      </a>
    );
  }
