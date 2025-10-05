const OWNER  = process.env.NEXT_PUBLIC_GH_OWNER  ?? "owner";
const REPO   = process.env.NEXT_PUBLIC_GH_REPO   ?? "repo";
const BRANCH = process.env.NEXT_PUBLIC_GH_BRANCH ?? "main";
const PATH   = "community_data";

const NEW_FILE = `https://github.com/${OWNER}/${REPO}/new/${BRANCH}/${PATH}`;
const UPLOADS  = `https://github.com/${OWNER}/${REPO}/upload/${BRANCH}/${PATH}`;

export default function Contribute() {
  return (
    <section className="space-y-6">
      <h1 className="text-2xl font-bold">Contribute Data</h1>
      <ol className="list-decimal ml-5 space-y-3 text-neutral-300">
        <li>Prepare your files (CSV/JSON/ZIP) + a short README (source, license, columns).</li>
        <li><a className="underline" href={UPLOADS}>Upload to <code>community_data/</code> via Pull Request</a>.</li>
        <li>We’ll review and merge. Your data never touches our simulation outputs.</li>
      </ol>
      <p className="text-sm text-neutral-400">
        Or <a className="underline" href={NEW_FILE}>create a new file</a> directly in <code>community_data/</code>.
      </p>
    </section>
  );
}
