import { test } from "node:test";
import assert from "node:assert/strict";
import { boundaryProbs, selectBoundaries, chunkMarkdownDynamic } from "../src/chunk/dynamic.js";
import { overlapUnits } from "../src/chunk/dynamic.js";

// Toy "embedding": one axis per topic word, so sentences on the same topic are parallel and a topic shift is orthogonal.
const TOPICS = ["ac", "brake", "battery"];
const fakeEmbed = async (texts) => texts.map((t) => TOPICS.map((w) => (t.toLowerCase().includes(w) ? 1 : 0.01)));

test("boundaryProbs is ½(1 − cos): 0 for the same direction, ½ for orthogonal, 1 for opposite", () => {
  const p = boundaryProbs([[1, 0], [1, 0], [0, 1], [0, -1]]);
  assert.equal(p[0], 1);
  assert.equal(p[1], 0);
  assert.equal(p[2], 0.5);
  assert.equal(p[3], 1);
});

test("selectBoundaries cuts at the highest-probability shifts, within min/max size", () => {
  const lens = [200, 200, 200, 200, 200, 200];
  const probs = [1, 0.05, 0.9, 0.04, 0.8, 0.03];
  assert.deepEqual(selectBoundaries(lens, probs, { targetChars: 400, minChars: 150, maxChars: 900 }), [0, 2, 4]);
});

test("maxChars forces a cut even when the topic does not change; one target chunk means no score cuts", () => {
  const lens = [500, 500, 500];
  assert.deepEqual(selectBoundaries(lens, [1, 0, 0], { targetChars: 3000, minChars: 100, maxChars: 1100 }), [0, 2]);
  assert.deepEqual(selectBoundaries([100, 100], [1, 0.99], { targetChars: 900 }), [0]);
});

test("a tiny tail joins the chunk before it", () => {
  assert.deepEqual(selectBoundaries([400, 400, 60], [1, 0.9, 0.95], { targetChars: 300, minChars: 300, maxChars: 1400 }), [0, 1]);
});

test("chunkMarkdownDynamic splits a mixed section where the topic changes, keeps headings and tables", async () => {
  const ac = "Car AC gas top-up takes about one hour at our workshop. The AC compressor is checked for leaks first. AC cooling coils are cleaned with foam. ";
  const brake = "Brake pads are inspected on every service visit. Worn brake discs are skimmed or replaced. Brake fluid is changed every two years. ";
  const md = `# Services\n\n${ac.repeat(2)}${brake.repeat(2)}\n\n| Service | Price |\n|---|---|\n| AC gas | Rs 1,499 |`;
  const chunks = await chunkMarkdownDynamic(md, { embed: fakeEmbed, targetChars: 300, minChars: 200, maxChars: 900 });
  const prose = chunks.filter((c) => !c.text.includes("|"));
  assert.ok(prose.length >= 2, "topic shift produces at least two prose chunks");
  assert.ok(prose.every((c) => c.text.startsWith("Services")), "heading prefix kept");
  // no prose chunk mixes the two topics: the cut fell on the AC -> brake shift
  assert.ok(prose.every((c) => !(c.text.includes("AC ") && c.text.toLowerCase().includes("brake"))));
  assert.ok(chunks.some((c) => c.text.includes("| AC gas | Rs 1,499 |")), "table travels alone");
  assert.ok(chunks.every((c) => c.chars <= 900 + "Services\n\n".length));
});

test("overlap: the previous chunk's last sentences lead the next one, about 20% of it", () => {
  const prev = ["Pickup free hai 15 km tak.", "Uske baad charge lagta hai.", "Cashless claim sab insurers ke saath hota hai."];
  const lead = overlapUnits(prev, 240, 0.2);
  assert.deepEqual(lead, ["Cashless claim sab insurers ke saath hota hai."]);   // ~48 chars of a 240-char chunk
  assert.deepEqual(overlapUnits(prev, 240, 0), []);
  assert.deepEqual(overlapUnits([], 240, 0.2), []);
  // one very long sentence is not dragged along (it would double the chunk)
  assert.deepEqual(overlapUnits(["x".repeat(400)], 240, 0.2), []);
});
