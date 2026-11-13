/**
 * Listener cues: Neha's "hmm" / "haan ji" *while the customer is still talking*.
 *
 * Idea from Moshi (Défossez et al., "Moshi: a speech-text foundation model for real-time dialogue",
 * arXiv 2410.00037): real conversation is full duplex. The listener's stream is never silent by rule;
 * it carries short backchannels in the speaker's pauses without taking the turn. Moshi learns that from
 * two-channel audio; we cannot run Moshi here (GPU, English-only fixed voices, no strict-KB grounding),
 * so this is the same behaviour as a small policy on top of the half-duplex pipeline we already have:
 *
 *   mic gate "early" (a pause) -> early transcript -> looksComplete() says "mid-thought"
 *   -> this policy decides whether a cue fits -> a warm server clip plays softly, the turn stays open.
 *
 * A finished thought never gets a cue (that pause is the end of the turn, and Neha will answer).
 * Pure and clock-injected, so it is unit-tested without a browser.
 */

/** Cues safe to play over someone mid-sentence: short, closed or near-closed mouth, no content. */
export const DEFAULT_CUES = Object.freeze(["hmm", "haan ji", "acha"]);

// Words that carry a complaint; a cheerful "acha" over "brake kharab ho gaya…" sounds wrong, "hmm" does not.
const CONCERN = /\b(problem|issue|dikkat|kharab|awaaz|noise|leak|mehnga|mehenga|expensive|accident|claim|damage|toot|tut)\w*/i;

/**
 * @param {{cues?: string[], minTalkMs?: number, minWords?: number, gapMs?: number, maxPerTurn?: number}} opts
 *   minTalkMs  the customer must have been talking this long (a 1 s question needs no "hmm")
 *   minWords   …and said at least this many words
 *   gapMs      at least this long between two cues (persona conversation.backchannel_interval_s)
 *   maxPerTurn at most this many cues in one customer turn
 */
export function createListenerCues({ cues = DEFAULT_CUES, minTalkMs = 2500, minWords = 5, gapMs = 4000, maxPerTurn = 2 } = {}) {
  const list = (cues || []).map((c) => String(c || "").trim()).filter(Boolean);
  let lastAt = -Infinity, inTurn = 0, i = 0;
  return {
    /** A new customer utterance started: the per-turn budget resets, the spacing clock does not. */
    newTurn() { inTurn = 0; },
    /**
     * The early transcript of a pause arrived. Returns the cue to play, or null.
     * @param {{text: string, complete: boolean, talkMs: number, now: number}} p
     */
    next({ text, complete, talkMs, now }) {
      if (complete || !list.length) return null;
      const words = String(text || "").trim().split(/\s+/).filter(Boolean).length;
      if (words < minWords || talkMs < minTalkMs) return null;
      if (inTurn >= maxPerTurn || now - lastAt < gapMs) return null;
      lastAt = now; inTurn++;
      if (CONCERN.test(text) && list.includes("hmm")) return "hmm";
      return list[i++ % list.length];
    },
  };
}

/**
 * Which persona backchannels work as mid-sentence cues: one or two words, no "samajh gayi" (that one
 * claims understanding of a thought that is not finished yet).
 */
export function cueListFrom(backchannels) {
  const ok = (backchannels || []).filter((b) => typeof b === "string" && b.trim()
    && b.trim().split(/\s+/).length <= 2 && !/^samajh|^theek|^right$|^okay$/i.test(b.trim()));
  return ok.length ? ok : [...DEFAULT_CUES];
}
