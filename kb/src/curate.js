// Curation of the customer-call Q&A and per-service summaries, from kb/seed/qa_overrides.json.
//
//  - suppress: qa rows that must no longer be served (wrong or unverified price, timing, address, offer, a call note,
//    or an answer to a different question). The page is set inactive, never deleted, so the decision is reversible
//    and the row stays in the review CSV. The bot's fast path (/api/knowledge/faq) and search only read active pages.
//  - fix: the answer / English line (and optionally the question list) replaced with the website's own facts.
//  - add: Q&A rows written from the website's text (human_bot/persona/acr_service_kb.json), url = the service page
//    with a #faq-n fragment, page_type "qa" so the fast path serves them.
//  - service_pages: one summary page per service (page_type "kb", url = service page + "#service-summary"), with the
//    Hinglish / Devanagari words customers use, so "AC thanda nahi kar raha" finds the AC page.
//
// The running server picks the changes up without a restart (PRAGMA data_version invalidates its caches; the
// bot reloads the fast path every 30 min or on its own restart).
//
//   node src/curate.js [--website auto_car_repair] [--dry] [--no-gemini]
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { ROOT_DIR, settings, websites } from "./config.js";
import { embedTexts, embeddingInfo } from "./embed/gemini.js";
import { LOCAL_MODEL, embedTextsLocal, loadLocal } from "./embed/local.js";
import { log } from "./log.js";
import { KnowledgeStore } from "./store/sqlite.js";

export const OVERRIDES_FILE = resolve(ROOT_DIR, "seed", "qa_overrides.json");
const sha = (s) => createHash("sha256").update(s).digest("hex").slice(0, 32);
const qaId = (url) => String(url).split("#")[1] ?? "";

/** {suppress:{id:reason}, fix:{id:{answer,answer_en,questions?,reason}}, add:[...], service_pages:[...]} (empty when absent). */
export function loadOverrides(path = OVERRIDES_FILE) {
  const empty = { suppress: {}, fix: {}, add: [], service_pages: [], site_rules: { exclude_pages: [], drop_chunks: [], strip: [] } };
  if (!existsSync(path)) return empty;
  const o = JSON.parse(readFileSync(path, "utf8").replace(/^\uFEFF/, ""));
  const r = o.site_rules ?? {};
  return { suppress: o.suppress ?? {}, fix: o.fix ?? {}, add: o.add ?? [], service_pages: o.service_pages ?? [],
    site_rules: { exclude_pages: r.exclude_pages ?? [], drop_chunks: r.drop_chunks ?? [], strip: r.strip ?? [] } };
}

/** A qa chunk the way ingest_qa.js writes it: first line = Neha's answer, then the English line. */
export function qaChunk({ questions, answer, answerEn, source }) {
  const lines = [answer];
  if (answerEn) lines.push(`English: ${answerEn}`);
  if (source) lines.push(`Source: ${source}`);
  const text = lines.join("\n");
  const heading = [...new Set(questions.map((q) => String(q).trim()).filter(Boolean))].join(" | ");
  return { heading, text };
}

/** Pages (page + chunks, no vectors) this file defines, for the live DB or ingest_qa.js. */
export function curatedPages(ov, domain = "autocarrepair.in") {
  const base = `https://${domain}`;
  const out = [];
  for (const a of ov.add) {
    const { heading, text } = qaChunk({ questions: a.questions, answer: a.answer, answerEn: a.answer_en, source: a.source_url });
    const url = `${base}${a.service}#faq-${a.id}`;
    out.push({ page: { url, canonical: url, title: a.questions[0], description: a.questions[1] ?? a.questions[0], pageType: "qa", lang: "hi-Latn", contentHash: sha(heading + "\n" + text) },
      chunks: [{ index: 0, heading, text, hash: sha(text) }] });
  }
  for (const s of ov.service_pages) {
    const url = `${base}${s.path}#service-summary`;
    const chunks = [{ index: 0, heading: `${s.title} | service summary`, text: s.summary }];
    if (s.faq) chunks.push({ index: 1, heading: `${s.title} | common questions`, text: s.faq });
    for (const c of chunks) c.hash = sha(c.text);
    out.push({ page: { url, canonical: url, title: `${s.title} (service summary)`, description: s.title, pageType: "kb", lang: "en",
      contentHash: sha(chunks.map((c) => c.heading + "\n" + c.text).join("\n")) }, chunks });
  }
  return out;
}

/** Apply a fix to a CSV row (ingest_qa.js): the fixed answer replaces the call note, the price note and the extras. */
export function fixRow(row, fix) {
  return { ...row, answer: fix.answer, answer_en: fix.answer_en ?? "", answer_extra: "", has_price: "0",
    question: fix.questions?.[0] ?? row.question, question_en: fix.questions ? (fix.questions[1] ?? "") : row.question_en,
    aliases: fix.questions ? fix.questions.slice(2).join(" || ") : row.aliases, curated: true };
}

/** Set the suppressed qa pages inactive (never deletes). Returns how many changed. */
export function suppressPages(store, siteId, ov) {
  const ids = new Set(Object.keys(ov.suppress));
  if (!ids.size) return 0;
  const rows = store.db.prepare("SELECT id, url, status FROM pages WHERE website_id = ? AND page_type = 'qa'").all(siteId);
  const upd = store.db.prepare("UPDATE pages SET status = 'inactive', error = ? WHERE id = ?");
  let n = 0;
  store.db.exec("BEGIN");
  try {
    for (const r of rows) {
      if (r.status === "active" && ids.has(qaId(r.url))) { upd.run("curated: " + String(ov.suppress[qaId(r.url)]).slice(0, 400), r.id); n++; }
    }
    store.db.exec("COMMIT");
  } catch (e) { store.db.exec("ROLLBACK"); throw e; }
  if (n) store.bump(siteId);
  return n;
}

// ---------------------------------------------------------------- website content rules (crawled pages)
// The website itself has mistakes (a page showing another service's text, expired offers, template FAQ lines,
// duplicate reviews, price text). Neha must not repeat them, so these rules keep that text out of retrieval:
//   exclude_pages: [{url, reason}]                      page set inactive (url = exact page URL)
//   drop_chunks:   [{url?, heading?, text?, reason}]    chunk deleted; url/heading/text are regexes, all given must match
//   strip:         [{url?, pattern, reason}]            matching text removed from the chunk (regex, global, case-insens.)
// Enforced by `node src/curate.js` and by the crawl itself (pipeline.js calls enforceSite after every page), so a
// re-crawl cannot bring the text back. qa and kb pages (ours) are never touched by these rules.
const rx = (s, f = "i") => (s ? new RegExp(s, f) : null);
export function compileRules(rules = {}) {
  return {
    exclude: new Map((rules.exclude_pages ?? []).map((e) => [e.url, e.reason])),
    drop: (rules.drop_chunks ?? []).map((d) => ({ url: rx(d.url), heading: rx(d.heading), text: rx(d.text), reason: d.reason })),
    strip: (rules.strip ?? []).map((s) => ({ url: rx(s.url), pattern: new RegExp(s.pattern, "gi"), reason: s.reason })),
  };
}

/** What happens to one chunk: {drop: reason} | {text: newText} | null (unchanged). Pure, for tests. */
export function ruleChunk(c, url, heading, text) {
  for (const d of c.drop) {
    if ((!d.url || d.url.test(url)) && (!d.heading || d.heading.test(heading ?? "")) && (!d.text || d.text.test(text))) return { drop: d.reason };
  }
  let out = text;
  for (const s of c.strip) if (!s.url || s.url.test(url)) out = out.replace(s.pattern, "");
  if (out === text) return null;
  out = out.replace(/[ \t]{2,}/g, " ").replace(/\n{3,}/g, "\n\n").trim();
  const body = out.split(String(heading ?? "")).join("").replace(/[\s#*>|-]+/g, " ").trim();
  return body.length < 60 ? { drop: "nothing left after removing excluded text" } : { text: out };
}

/** Apply the website rules to the store (all crawled pages, or only `urls`). Returns counts. */
export async function enforceSite(store, siteId, ov, { urls = null } = {}) {
  const c = compileRules(ov.site_rules);
  if (!c.exclude.size && !c.drop.length && !c.strip.length) return { excluded: 0, dropped: 0, stripped: 0 };
  const only = urls ? new Set(urls) : null;
  let excluded = 0, dropped = 0;
  const pages = store.db.prepare("SELECT id, url, status FROM pages WHERE website_id = ? AND COALESCE(page_type,'') NOT IN ('qa','kb')").all(siteId)
    .filter((p) => !only || only.has(p.url));
  const setInactive = store.db.prepare("UPDATE pages SET status = 'inactive', error = ? WHERE id = ?");
  const delChunk = store.db.prepare("DELETE FROM chunks WHERE id = ?");
  const listChunks = store.db.prepare("SELECT id, heading, text FROM chunks WHERE page_id = ?");
  const edits = [];
  store.db.exec("BEGIN");
  try {
    for (const p of pages) {
      if (c.exclude.has(p.url)) {
        if (p.status === "active") { setInactive.run("curated: " + String(c.exclude.get(p.url)).slice(0, 400), p.id); excluded++; }
        continue;
      }
      if (p.status !== "active") continue;
      for (const ch of listChunks.all(p.id)) {
        const r = ruleChunk(c, p.url, ch.heading, ch.text);
        if (!r) continue;
        if (r.drop) { delChunk.run(ch.id); dropped++; } else edits.push({ id: ch.id, title: ch.heading, text: r.text });
      }
    }
    store.db.exec("COMMIT");
  } catch (e) { store.db.exec("ROLLBACK"); throw e; }
  if (edits.length) {
    if (!(await loadLocal())) throw new Error("local embedding model unavailable (needed to re-embed edited chunks)");
    const vecs = await embedTextsLocal(edits.map((e) => ({ text: e.text, title: String(e.title).slice(0, 200) })));
    const upd = store.db.prepare("UPDATE chunks SET text = ?, content_hash = ? WHERE id = ?");
    store.db.exec("BEGIN");
    try {
      edits.forEach((e, i) => {
        upd.run(e.text, sha(e.text), e.id);   // FTS follows by trigger; the Gemini vector is kept (fallback path only)
        store.q.insertLocal.run(e.id, LOCAL_MODEL, vecs[i].length, Buffer.from(vecs[i].buffer, vecs[i].byteOffset, vecs[i].byteLength));
      });
      store.db.exec("COMMIT");
    } catch (e) { store.db.exec("ROLLBACK"); throw e; }
  }
  if (excluded || dropped || edits.length) store.bump(siteId);
  return { excluded, dropped, stripped: edits.length };
}

async function vectors(items, { gemini }) {
  // items: [{text, title}] -> {gem: Float32Array[]|null, local: Float32Array[]}
  const local = await embedTextsLocal(items);
  let gem = null;
  if (gemini) {
    try { gem = await embedTexts(items, { retries: 2 }); } catch (err) { log.warn("gemini embedding failed; local vectors only", { err: String(err.message).slice(0, 160) }); }
  }
  return { gem, local };
}

/** Write pages (page + chunks without vectors) with local + Gemini vectors; unchanged pages are skipped. */
export async function writePages(store, siteId, pages, { gemini = true } = {}) {
  const { model, dim } = embeddingInfo();
  const todo = pages.filter((p) => { const ex = store.getPage(siteId, p.page.url); return !ex || ex.status !== "active" || ex.content_hash.replace(/:pending-gemini$/, "") !== p.page.contentHash; });
  for (let i = 0; i < todo.length; i += 8) {
    const slice = todo.slice(i, i + 8);
    const flat = slice.flatMap((p) => p.chunks.map((c) => ({ text: c.text, title: c.heading.slice(0, 200) })));
    const { gem, local } = await vectors(flat, { gemini });
    let k = 0;
    for (const p of slice) {
      const localVecs = [];
      for (const c of p.chunks) { c.vec = gem ? gem[k] : new Float32Array(dim); localVecs.push(local[k]); k++; }
      const page = gem ? p.page : { ...p.page, contentHash: p.page.contentHash + ":pending-gemini" };
      store.replacePage({ websiteId: siteId, page, chunks: p.chunks, model, dim, jobId: null, httpStatus: 200, localModel: LOCAL_MODEL, localVecs });
    }
  }
  return todo.length;
}

async function main() {
  const args = process.argv.slice(2);
  const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const siteId = opt("--website", "auto_car_repair");
  const site = websites[siteId];
  if (!site) throw new Error(`unknown website ${siteId}`);
  const ov = loadOverrides();
  const store = new KnowledgeStore(settings.dbPath);
  try {
    const qa = store.db.prepare("SELECT p.id, p.url, p.title, p.description, p.status, c.heading, c.text FROM pages p JOIN chunks c ON c.page_id = p.id AND c.chunk_index = 0 WHERE p.website_id = ? AND p.page_type = 'qa'").all(siteId);
    const byId = new Map(qa.map((r) => [qaId(r.url), r]));
    const missing = [...Object.keys(ov.suppress), ...Object.keys(ov.fix)].filter((id) => !byId.has(id));
    log.info("curation", { suppress: Object.keys(ov.suppress).length, fix: Object.keys(ov.fix).length, add: ov.add.length, service_pages: ov.service_pages.length, unknown_ids: missing.length });
    // fixed rows keep their page (url, title); answer, English line and optionally the questions change
    const fixed = [];
    for (const [id, f] of Object.entries(ov.fix)) {
      const r = byId.get(id);
      if (!r || ov.suppress[id]) continue;
      const questions = f.questions ?? String(r.heading).split("|").map((q) => q.trim()).filter(Boolean);
      const { heading, text } = qaChunk({ questions, answer: f.answer, answerEn: f.answer_en, source: "autocarrepair.in (checked 2026-11-25)" });
      fixed.push({ page: { url: r.url, canonical: r.url, title: questions[0], description: questions[1] ?? questions[0], pageType: "qa", lang: "hi-Latn", contentHash: sha(heading + "\n" + text) },
        chunks: [{ index: 0, heading, text, hash: sha(text) }] });
    }
    const pages = [...fixed, ...curatedPages(ov, site.domain)];
    if (args.includes("--dry")) {
      const c = compileRules(ov.site_rules);
      const hit = { drop: 0, strip: 0 };
      for (const ch of store.db.prepare("SELECT p.url, c.heading, c.text FROM chunks c JOIN pages p ON p.id = c.page_id WHERE p.website_id = ? AND p.status = 'active' AND COALESCE(p.page_type,'') NOT IN ('qa','kb')").all(siteId)) {
        if (c.exclude.has(ch.url)) continue;
        const r = ruleChunk(c, ch.url, ch.heading, ch.text); if (r) hit[r.drop ? "drop" : "strip"]++;
      }
      console.log(JSON.stringify({ exclude_pages: c.exclude.size, chunks_to_drop: hit.drop, chunks_to_strip: hit.strip }));
      console.log(JSON.stringify({ would_suppress: Object.keys(ov.suppress).filter((id) => byId.get(id)?.status === "active").length, fix: fixed.length, curated: pages.length - fixed.length, unknown_ids: missing }, null, 1));
      return;
    }
    if (!(await loadLocal())) throw new Error("local embedding model unavailable");
    const n = suppressPages(store, siteId, ov);
    log.info("website rules applied", await enforceSite(store, siteId, ov));
    const w = await writePages(store, siteId, pages, { gemini: !args.includes("--no-gemini") });
    // curated pages that left the file leave the index too (only ours: #faq-site-* and #service-summary)
    const keep = new Set(pages.map((p) => p.page.url));
    const stale = store.db.prepare("SELECT id, url FROM pages WHERE website_id = ? AND (url LIKE '%#faq-site-%' OR url LIKE '%#service-summary')").all(siteId).filter((p) => !keep.has(p.url));
    if (stale.length) { const del = store.db.prepare("DELETE FROM pages WHERE id = ?"); for (const p of stale) del.run(p.id); store.bump(siteId); }
    store.checkpoint();
    const active = store.db.prepare("SELECT COUNT(*) n FROM pages WHERE website_id = ? AND page_type = 'qa' AND status = 'active'").get(siteId).n;
    log.info("curation applied", { suppressed_now: n, written: w, stale_removed: stale.length, active_qa: active, local: store.localCoverage(siteId, LOCAL_MODEL) });
  } finally {
    store.close();
  }
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  main().catch((err) => { log.error("curation failed", { err }); process.exit(1); });
}
