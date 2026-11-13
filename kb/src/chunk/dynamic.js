// Dynamic (content-defined) chunking, after H-Net: Hwang, Wang, Gu, "Dynamic Chunking for End-to-End
// Hierarchical Sequence Modeling", arXiv 2507.07955.
//
// H-Net's routing module puts a chunk boundary where adjacent representations stop looking alike:
//     p_t = ½ · (1 − cos(q_t, k_{t−1}))        b_t = 1 if p_t ≥ 0.5
// with q/k learned projections of byte-level states, and a ratio loss that keeps the number of boundaries
// near a target compression ratio. We do not train anything here (no GPU, and retrieval is the job, not
// language modelling), so the same rule runs one level up:
//   * units are sentences (tables whole), their states are e5 sentence embeddings (embed/local.js, CPU);
//   * p_t is the same ½(1 − cos) between neighbouring sentences: a topic shift scores high;
//   * the ratio target becomes a size target: the threshold is set per section so the expected chunk count
//     is total/targetChars, and min/max sizes bound every chunk (a learned ratio loss does the same softly).
// Headings stay hard boundaries. Output has the same shape as chunkMarkdown(), so the store and search
// do not change. Fixed-size chunking (chunk.js) stays the default; KB_CHUNKER=dynamic opts in.
import { sha256 } from "../extract/extract.js";
import { sectionsOf } from "./chunk.js";

const SENTENCE_RE = /(?<=[.!?।])\s+(?=[A-Z0-9ऀ-ॿ"'(])/;

function dot(a, b) {
  let s = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { s += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na && nb ? s / Math.sqrt(na * nb) : 0;
}

/** H-Net boundary probabilities: p[0] = 1 (a section always starts a chunk), p[t] = ½(1 − cos(v_t, v_{t−1})). */
export function boundaryProbs(vecs) {
  return vecs.map((v, t) => (t === 0 ? 1 : Math.min(1, Math.max(0, 0.5 * (1 - dot(v, vecs[t - 1]))))));
}

/**
 * Choose chunk starts from boundary probabilities under size bounds. Pure.
 * @param {number[]} lens  characters per unit
 * @param {number[]} probs boundaryProbs()
 * @returns {number[]} indices where a chunk starts (always includes 0)
 */
export function selectBoundaries(lens, probs, { targetChars = 900, minChars = 300, maxChars = 1400 } = {}) {
  const n = lens.length;
  if (!n) return [];
  const total = lens.reduce((a, b) => a + b, 0);
  const want = Math.max(1, Math.round(total / targetChars));        // the "compression ratio" target
  // threshold = the (want-1)-th largest interior probability; with want = 1 nothing passes on score alone
  const interior = probs.slice(1).map((p, i) => [p, i + 1]).sort((a, b) => b[0] - a[0]);
  const thr = want > 1 && interior.length ? interior[Math.min(want - 2, interior.length - 1)][0] : Infinity;
  const starts = [0];
  let cur = lens[0];
  for (let t = 1; t < n; t++) {
    const forced = cur + lens[t] + 1 > maxChars;
    const chosen = probs[t] >= thr && cur >= minChars;
    if (forced || chosen) { starts.push(t); cur = lens[t]; } else cur += lens[t] + 1;
  }
  // a tail far below minChars rides with the chunk before it (if that still fits)
  if (starts.length > 1) {
    const last = starts[starts.length - 1];
    const tail = lens.slice(last).reduce((a, b) => a + b + 1, 0);
    const prev = lens.slice(starts[starts.length - 2], last).reduce((a, b) => a + b + 1, 0);
    if (tail < minChars / 2 && prev + tail <= maxChars) starts.pop();
  }
  return starts;
}

/** Hard wrap a unit that alone exceeds maxChars (a run-on list) on word boundaries. */
function wrap(text, maxChars) {
  if (text.length <= maxChars) return [text];
  const out = [];
  let w = "";
  for (const word of text.split(/\s+/)) {
    if (w && (w + " " + word).length > maxChars) { out.push(w); w = word; } else w = w ? w + " " + word : word;
  }
  if (w) out.push(w);
  return out;
}

/**
 * @param {string} markdown cleaned page markdown
 * @param {{title?:string, embed:(texts:string[])=>Promise<ArrayLike<number>[]>, targetChars?:number, minChars?:number, maxChars?:number}} opts
 *   embed: sentence embeddings for boundary scoring (normalized or not); e.g. texts => embedTextsLocal(texts.map(text => ({text})))
 * @returns {Promise<{index:number, heading:string, text:string, hash:string, chars:number, boundary?:number}[]>}
 */
/**
 * Overlap: the sentences that end the previous chunk and start this one, about `ratio` of this chunk's length
 * (at least one sentence when it is short enough), so a fact cut by a boundary is whole in one of the two. Pure.
 * @param {string[]} prev units of the previous chunk  @param {number} curChars  @param {number} ratio
 */
export function overlapUnits(prev, curChars, ratio) {
  if (!ratio || !prev.length) return [];
  const budget = curChars * ratio;
  const out = [];
  let n = 0;
  for (let i = prev.length - 1; i >= 0; i--) {
    const len = prev[i].length + 1;
    if (n + len > budget * 1.25 && (out.length || len > curChars * ratio * 1.75)) break;
    out.unshift(prev[i]); n += len;
    if (n >= budget) break;
  }
  return out;
}

export async function chunkMarkdownDynamic(markdown, { title = "", embed, targetChars = 900, minChars = 300, maxChars = 1400, overlapRatio = 0 } = {}) {
  if (typeof embed !== "function") throw new Error("chunkMarkdownDynamic needs an embed function");
  const sections = sectionsOf(markdown, title);
  const chunks = [];
  const push = (heading, body, boundary) => {
    const text = ((heading ? `${heading}\n\n` : "") + body).trim();
    if (text.replace(/\s+/g, "").length < 40) return; // noise, same rule as chunk.js
    chunks.push({ index: chunks.length, heading, text, hash: sha256(text), chars: text.length, boundary });
  };
  for (const sec of sections) {
    // units in reading order; a table interrupts the prose run and travels alone (as in chunk.js)
    let run = [];
    const flushRun = async () => {
      if (!run.length) return;
      const units = run; run = [];
      const vecs = units.length > 1 ? await embed(units) : [[1]];
      const probs = boundaryProbs(vecs);
      const starts = selectBoundaries(units.map((u) => u.length), probs, { targetChars, minChars, maxChars });
      for (let k = 0; k < starts.length; k++) {
        const a = starts[k], b = k + 1 < starts.length ? starts[k + 1] : units.length;
        const own = units.slice(a, b);
        const lead = k > 0 ? overlapUnits(units.slice(starts[k - 1], a), own.join(" ").length, overlapRatio) : [];
        push(sec.heading, [...lead, ...own].join(" "), Number(probs[a].toFixed(3)));
      }
    };
    for (const p of sec.paragraphs) {
      if (p.table) {
        await flushRun();
        if (p.text.length <= maxChars * 2) push(sec.heading, p.text, 1);
        else {
          const rows = p.text.split("\n"), head = rows.slice(0, 2).join("\n");
          let part = head;
          for (const r of rows.slice(2)) {
            if ((part + "\n" + r).length > maxChars * 2) { push(sec.heading, part, 1); part = head + "\n" + r; } else part += "\n" + r;
          }
          push(sec.heading, part, 1);
        }
        continue;
      }
      for (const s of p.text.split(SENTENCE_RE)) run.push(...wrap(s.trim(), maxChars).filter(Boolean));
    }
    await flushRun();
  }
  return chunks;
}
