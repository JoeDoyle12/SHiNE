import "./globals.css";
import { Montserrat } from "next/font/google";
import Link from "next/link";
export const metadata = { title: "Exoplanet AI", description: "NASA Space Apps 2025" };

const montserrat = Montserrat({
  subsets: ["latin"],
  weight: ["400", "600", "700"], // adjust if you need more/less
  display: "swap",
});

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      
<body className={`${montserrat.className} relative text-neutral-100`}>
  {/* EVERYTHING BELOW stays the same – no background divs here */}
  <div className="relative z-10">
    <header className="border-b border-neutral-800">
      <nav className="mx-auto max-w-6xl p-4 flex items-center">
        <Link
          href="/"
          className="text-lg font-semibold tracking-wide select-none no-underline hover:opacity-90"
          aria-label="Go to homepage"
        >
          SHiNES
        </Link>
        <div className="ml-auto">
          <a href="/about" className="text-sm font-medium hover:opacity-80 transition">About</a>
        </div>
      </nav>
    </header>

    <main className="mx-auto max-w-6xl px-6 pt-4 md:pt-6 pb-12">{children}</main>

    <footer className="mx-auto max-w-6xl p-6 text-sm text-neutral-300">
      © {new Date().getFullYear()} SHiNES
    </footer>
  </div>
</body>

    </html>
  );
}
