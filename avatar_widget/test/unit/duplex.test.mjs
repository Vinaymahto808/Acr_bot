import test from "node:test";
import assert from "node:assert/strict";
import { createListenerCues, cueListFrom, DEFAULT_CUES } from "../../src/duplex.js";

const long = "mera gaadi ka brake thoda sa awaaz kar raha hai aur";
const base = { text: "kal main service ke liye aana chahta tha lekin", complete: false, talkMs: 3000 };

test("a mid-thought pause in a long turn gets a cue; a finished thought never does", () => {
  const c = createListenerCues();
  assert.equal(c.next({ ...base, complete: true, now: 10_000 }), null);
  assert.equal(c.next({ ...base, now: 10_000 }), DEFAULT_CUES[0]);
});

test("short turns and few words stay silent", () => {
  const c = createListenerCues();
  assert.equal(c.next({ ...base, talkMs: 1200, now: 10_000 }), null);
  assert.equal(c.next({ ...base, text: "haan woh", now: 10_000 }), null);
});

test("cues are spaced by gapMs and capped per turn; newTurn resets only the cap", () => {
  const c = createListenerCues({ gapMs: 4000, maxPerTurn: 2 });
  assert.ok(c.next({ ...base, now: 10_000 }));
  assert.equal(c.next({ ...base, now: 12_000 }), null);          // too soon
  assert.ok(c.next({ ...base, now: 14_500 }));
  assert.equal(c.next({ ...base, now: 30_000 }), null);          // turn budget used
  c.newTurn();
  assert.ok(c.next({ ...base, now: 30_000 }));
});

test("a complaint gets 'hmm', not a cheerful 'acha'", () => {
  const c = createListenerCues({ cues: ["acha", "hmm"] });
  assert.equal(c.next({ ...base, text: long, now: 10_000 }), "hmm");
});

test("cueListFrom keeps short neutral backchannels and drops claims of understanding", () => {
  assert.deepEqual(cueListFrom(["haan ji", "hmm", "right", "acha", "samajh gayi", "theek hai ji"]), ["haan ji", "hmm", "acha"]);
  assert.deepEqual(cueListFrom([]), [...DEFAULT_CUES]);
});

test("an empty cue list disables cues", () => {
  assert.equal(createListenerCues({ cues: [] }).next({ ...base, now: 10_000 }), null);
});
