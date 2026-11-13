// One rendered page in -> chunks + embeddings in the store, or a cheap "unchanged" touch.
import { settings } from "./config.js";
import { chunkMarkdown } from "./chunk/chunk.js";
import { chunkMarkdownDynamic } from "./chunk/dynamic.js";
import { embedTextsLocal, loadLocal, LOCAL_MODEL } from "./embed/local.js";
import { embedTexts, embeddingInfo } from "./embed/gemini.js";
import { extractPage, sha256 } from "./extract/extract.js";
import { log } from "./log.js";

const ERROR_VIEW_RE = /(something (broke|went wrong)|unexpected error|error boundary|unable to preload css|page not found|404\b[^\n]{0,40}not found|reload (the )?page to recover|chunkloaderror)/i;

/** Fixed-size chunks, or H-Net-style dynamic ones (KB_CHUNKER=dynamic) when the local model is usable. */
export async function chunkPage(page) {
  const c = settings.chunk;
  if (c.mode === "dynamic" && LOCAL_MODEL && (await loadLocal())) {
    try {
      return await chunkMarkdownDynamic(page.markdown, {
        title: page.title, targetChars: c.targetChars, minChars: c.minChars, maxChars: c.maxChars, overlapRatio: c.overlapRatio,
        embed: (texts) => embedTextsLocal(texts.map((text) => ({ text }))),
      });
    } catch (err) {
      log.warn("dynamic chunking failed, using fixed", { url: page.url, err: String(err.message).slice(0, 160) });
    }
  }
  return chunkMarkdown(page.markdown, { size: c.size, overlap: c.overlap, title: page.title });
}

/** The chunking settings a page was indexed with: part of its stored hash, so changing them (mode, size,
 *  overlap) re-chunks every page on the next ordinary crawl. --reindex is not needed (it would also wipe the
 *  customer-call Q&A pages). */
export function chunkerSignature() {
  const c = settings.chunk;
  return c.mode === "dynamic" ? `dyn:${c.targetChars}:${c.minChars}:${c.maxChars}:${c.overlapRatio}` : `fixed:${c.size}:${c.overlap}`;
}

/** Short pages dominated by a client-side error / not-found view are not knowledge. */
export function looksLikeErrorPage(page) {
  return page.text.length < 1500 && ERROR_VIEW_RE.test(page.text);
}

/**
 * @returns {{outcome:'indexed'|'unchanged'|'thin'|'failed', chunks:number, url:string}}
 */
export async function processPage({ store, site, url, html, httpStatus, jobId }) {
  const t0 = Date.now();
  const page = extractPage(html, url);
  const finalUrl = page.url ?? url;

  if (!page.text || page.text.length < settings.chunk.minPageChars) {
    store.recordFailure(site.id, finalUrl, jobId, httpStatus, "thin: no meaningful content after extraction");
    log.info("page skipped (thin)", { website: site.id, url: finalUrl, chars: page.text?.length ?? 0 });
    return { outcome: "thin", chunks: 0, url: finalUrl };
  }
  if (looksLikeErrorPage(page)) {
    // existing chunks (if any) are kept; the page is only flagged
    store.recordFailure(site.id, finalUrl, jobId, httpStatus, "spa-error: page rendered an error boundary / not-found view");
    log.warn("page skipped (error view)", { website: site.id, url: finalUrl, chars: page.text.length });
    return { outcome: "thin", chunks: 0, url: finalUrl };
  }

  page.contentHash = sha256(page.contentHash + "|" + chunkerSignature());
  const existing = store.getPage(site.id, finalUrl);
  if (existing && existing.status === "active" && existing.content_hash === page.contentHash) {
    store.touchPage(existing.id, jobId, httpStatus);
    log.debug("page unchanged", { website: site.id, url: finalUrl });
    return { outcome: "unchanged", chunks: existing.chunk_count, url: finalUrl };
  }

  const { model, dim } = embeddingInfo();
  store.assertModelCompatible(site.id, model, dim, LOCAL_MODEL ? [LOCAL_MODEL] : []);

  const chunks = await chunkPage(page);
  if (!chunks.length) {
    store.recordFailure(site.id, finalUrl, jobId, httpStatus, "thin: produced no chunks");
    return { outcome: "thin", chunks: 0, url: finalUrl };
  }

  // the embedding sees "title / heading" context so short chunks are not orphaned
  const items = chunks.map((c) => ({ text: c.text, title: page.title }));
  let vecs;
  let vecModel = model;
  let vecDim = dim;
  let localVecs = null;
  try {
    vecs = await embedTexts(items);
  } catch (err) {
    // Gemini quota / key: index the page with the local CPU vectors instead of throwing it away. A failed
    // page is knowledge missing exactly when a customer asks; hybrid-local is the default query path, so the
    // page answers immediately, and a later crawl overwrites it with Gemini vectors once the quota is back.
    if (!LOCAL_MODEL || !(await loadLocal())) throw err;
    vecs = await embedTextsLocal(items);
    vecModel = LOCAL_MODEL;
    vecDim = vecs[0]?.length ?? 0;
    log.warn("gemini embedding unavailable: page indexed with local vectors", {
      website: site.id, url: finalUrl, chunks: chunks.length, err: String(err.message).slice(0, 160),
    });
  }
  if (vecModel !== LOCAL_MODEL && LOCAL_MODEL && (await loadLocal())) {
    // every chunk also gets a local vector, so a mixed index (Gemini + local) still leaves the fast
    // hybrid-local path fully covered instead of silently dropping to the slower Gemini path
    try { localVecs = await embedTextsLocal(items); } catch (e) { log.warn("local vectors failed for page", { url: finalUrl, err: String(e.message).slice(0, 120) }); }
  }
  chunks.forEach((c, i) => { c.vec = vecs[i]; });

  store.replacePage({
    websiteId: site.id, page, chunks, model: vecModel, dim: vecDim, jobId, httpStatus,
    localModel: LOCAL_MODEL || "", localVecs,
  });
  log.info(existing ? "page updated" : "page indexed", {
    website: site.id, url: finalUrl, type: page.pageType, chunks: chunks.length, model: vecModel, ms: Date.now() - t0,
  });
  return { outcome: "indexed", chunks: chunks.length, url: finalUrl, changed: Boolean(existing) };
}
