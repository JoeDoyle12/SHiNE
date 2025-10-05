import "./globals.css";
import { Montserrat } from "next/font/google";
import Link from "next/link";
export const metadata = { title: "Exoplanet AI", description: "NASA Space Apps 2025" };

const OWNER = process.env.NEXT_PUBLIC_GH_OWNER ?? "owner";
const REPO  = process.env.NEXT_PUBLIC_GH_REPO  ?? "repo";
const REPOURL = `https://github.com/${OWNER}/${REPO}`;

const montserrat = Montserrat({
  subsets: ["latin"],
  weight: ["400", "600", "700"], // adjust if you need more/less
  display: "swap",
});

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      
<body className={`${montserrat.className} relative text-neutral-100`}>

  <div className="relative z-10">    
   <header className="border-b border-neutral-800">
  {/* full-bleed nav with responsive side padding */}
  <nav className="w-full py-4 px-4 sm:px-6 lg:px-8 flex items-center">
    <Link
      href="/"
      className="text-lg font-semibold tracking-wide select-none no-underline hover:opacity-90"
      aria-label="Go to homepage"
    >
      SHiNES
    </Link>

    {/* Right side links */}
    <div className="ml-auto flex items-center gap-6">
      <a href="/about" className="text-sm font-medium hover:opacity-80 transition">
        About
      </a>
      <a
        href={REPOURL}
        target="_blank"
        rel="noopener noreferrer"
        className="text-sm font-medium hover:opacity-80 transition underline-offset-4 hover:underline"
      >
        Github
      </a>
    </div>
  </nav>
</header>


<main className="px-4 sm:px-6 lg:px-8 mx-auto max-w-screen-xl pt-4 md:pt-6 pb-12">
  {children} </main>
  </div>
</body>

    </html>
  );
}
