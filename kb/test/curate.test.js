import { test } from "node:test";
import assert from "node:assert/strict";
import { KnowledgeStore } from "../src/store/sqlite.js";
import { websites } from "../src/config.js";
import { createApp } from "../src/server.js";
import { curatedPages, fixRow, qaChunk, suppressPages } from "../src/curate.js";

const vec = () => Float32Array.from([1, 0, 0]);
const qaPage = (id, heading, text) => ({
  page: { url: `https://autocarrepair.in/faq/customer-calls#${id}`, canonical: "", title: heading.split("|")[0], description: "", pageType: "qa", lang: "hi-Latn", contentHash: id },
  chunks: [{ index: 0, heading, text, hash: id, vec: vec() }],
});

test("curate: suppressed qa rows leave the fast-path feed, are kept (inactive) and survive a crawl's retire step", async () => {
  const s = new KnowledgeStore(":memory:");
  s.upsertWebsite(websites.auto_car_repair);
  for (const p of [qaPage("qa-a", "AC gas refill bhi hota hai? | Do you refill AC gas?", "AC gas refill ke charges 1 thousand rupees.\nEnglish: AC gas refill costs 1,000."),
    qaPage("qa-b", "Pickup free hai? | Is pickup free?", "Haan, complimentary hai.\nEnglish: Yes, it is complimentary.")]) {
    s.replacePage({ websiteId: "auto_car_repair", ...p, model: "m", dim: 3, jobId: null, httpStatus: 200 });
  }
  const n = suppressPages(s, "auto_car_repair", { suppress: { "qa-a": "states a price not in acr_prices.json" } });
  assert.equal(n, 1);
  assert.equal(suppressPages(s, "auto_car_repair", { suppress: { "qa-a": "again" } }), 0, "idempotent");
  const row = s.getPage("auto_car_repair", "https://autocarrepair.in/faq/customer-calls#qa-a");
  assert.equal(row.status, "inactive");
  assert.match(row.error, /^curated: states a price/);

  const app = createApp({ store: s, jobs: { status: () => ({ state: "idle" }) } });
  const srv = app.listen(0, "127.0.0.1");
  await new Promise((r) => srv.once("listening", r));
  try {
    const body = await (await fetch(`http://127.0.0.1:${srv.address().port}/api/knowledge/faq?website=auto_car_repair`)).json();
    assert.deepEqual(body.rows.map((r) => r.url.split("#")[1]), ["qa-b"]);
  } finally { srv.close(); }

  // curated summary pages (page_type kb) are not crawled, so a completed crawl must not retire them
  const [summary] = curatedPages({ add: [], service_pages: [{ path: "/category/car-ac-service-repair", title: "AC Service and Repair", summary: "Customers say: AC thanda nahi kar raha", faq: "" }] });
  assert.equal(summary.page.pageType, "kb");
  s.replacePage({ websiteId: "auto_car_repair", page: summary.page, chunks: summary.chunks.map((c) => ({ ...c, vec: vec() })), model: "m", dim: 3, jobId: null, httpStatus: 200 });
  const job = s.startJob("auto_car_repair", "crawl");
  s.replacePage({ websiteId: "auto_car_repair", page: { url: "https://autocarrepair.in/x", canonical: "", title: "x", description: "", pageType: "service", lang: "en", contentHash: "x" }, chunks: [{ index: 0, heading: "x", text: "x", hash: "x", vec: vec() }], model: "m", dim: 3, jobId: job, httpStatus: 200 });
  s.deactivateUnseen("auto_car_repair", job, 1, 0);
  assert.equal(s.getPage("auto_car_repair", summary.page.url).status, "active");
  s.close();
});

test("curate: fixed rows drop the call note, the price note and the voice rewrite; added rows point at the service page", () => {
  const row = { id: "qa-x", question: "Workshop kitne baje khulta hai?", question_en: "What are the timings?", aliases: "timing kya hai", answer: "9 se 6, saato din.",
    answer_extra: "Sunday bhi", answer_en: "9 to 6, all week.", has_price: "1" };
  const f = fixRow(row, { answer: "Monday se Saturday 9 se 7.", answer_en: "Mon-Sat 9 to 7.", questions: ["Workshop kab khulta hai?", "When do you open?"] });
  assert.equal(f.answer, "Monday se Saturday 9 se 7.");
  assert.equal(f.answer_extra, "");
  assert.equal(f.has_price, "0");
  assert.equal(f.curated, true);
  assert.equal(f.aliases, "");
  assert.equal(f.question, "Workshop kab khulta hai?");
  const kept = fixRow(row, { answer: "a", answer_en: "b" });
  assert.equal(kept.question, row.question);
  assert.equal(kept.aliases, row.aliases);

  const c = qaChunk({ questions: ["a?", "b?", "a?"], answer: "hi", answerEn: "en", source: "src" });
  assert.equal(c.heading, "a? | b?");
  assert.equal(c.text.split("\n")[0], "hi");
  assert.match(c.text, /^English: en$/m);

  const [p] = curatedPages({ add: [{ id: "site-ac-1", service: "/category/car-ac-service-repair", questions: ["AC thanda nahi?", "AC not cooling?"], answer: "Leak check.", answer_en: "Leak check.", source_url: "u" }], service_pages: [] });
  assert.equal(p.page.url, "https://autocarrepair.in/category/car-ac-service-repair#faq-site-ac-1");
  assert.equal(p.page.pageType, "qa");
});

test("curate: website rules drop copied/review/link-list chunks, strip template and price text, keep the rest", async () => {
  const { compileRules, ruleChunk, enforceSite } = await import("../src/curate.js");
  const c = compileRules({
    drop_chunks: [{ heading: "CUSTOMER REVIEWS", reason: "reviews" }, { url: "/rubbing-polish$", heading: "COMMONLY ASKED", reason: "no answers" }],
    strip: [{ pattern: "(?:How long does [^?]{1,80}\\?\\s*)?Most [a-z &]+ jobs are completed the same day\\.", reason: "template" },
      { pattern: "[^.?!\\n]*(?:₹\\s?\\d[\\d,]*(?![\\d,]| coin)|\\bRs\\.?\\s?\\d)[^.?!\\n]*[.?!]?", reason: "price" }],
  });
  assert.equal(ruleChunk(c, "https://x/a", "A > CUSTOMER REVIEWS", "great").drop, "reviews");
  assert.equal(ruleChunk(c, "https://x/rubbing-polish", "R > COMMONLY ASKED QUESTIONS", "q?").drop, "no answers");
  assert.equal(ruleChunk(c, "https://x/wiper", "W > COMMONLY ASKED QUESTIONS", "Wiper blades typically last around 6 to 12 months, depending on use and conditions here. How long does Wiper take? Most maintenance tips jobs are completed the same day.").text,
    "Wiper blades typically last around 6 to 12 months, depending on use and conditions here.");
  const p = ruleChunk(c, "https://x/foam", "Foam", "Snow-foam touchless pre-wash and a two-bucket hand wash that prevents swirls on the paint. From Rs 800. ACR Detailing Studio.");
  assert.ok(!/Rs/.test(p.text) && /Snow-foam/.test(p.text));
  assert.equal(ruleChunk(c, "https://x/glass", "G", "A chip smaller than a ₹10 coin can be repaired."), null, "a coin is not a price");

  const s = new KnowledgeStore(":memory:");
  s.upsertWebsite(websites.auto_car_repair);
  const pg = (url) => ({ url, canonical: url, title: "t", description: "", pageType: "offers", lang: "en", contentHash: url });
  s.replacePage({ websiteId: "auto_car_repair", page: pg("https://autocarrepair.in/offers"), chunks: [{ index: 0, heading: "O", text: "Rs 6,375 this October", hash: "o", vec: vec() }], model: "m", dim: 3, jobId: null, httpStatus: 200 });
  s.replacePage({ websiteId: "auto_car_repair", page: pg("https://autocarrepair.in/x"), chunks: [{ index: 0, heading: "X > CUSTOMER REVIEWS", text: "five stars", hash: "r", vec: vec() }, { index: 1, heading: "X", text: "real text", hash: "t", vec: vec() }], model: "m", dim: 3, jobId: null, httpStatus: 200 });
  const res = await enforceSite(s, "auto_car_repair", { site_rules: { exclude_pages: [{ url: "https://autocarrepair.in/offers", reason: "expired" }], drop_chunks: [{ heading: "CUSTOMER REVIEWS", reason: "reviews" }] } });
  assert.deepEqual(res, { excluded: 1, dropped: 1, stripped: 0 });
  assert.equal(s.getPage("auto_car_repair", "https://autocarrepair.in/offers").status, "inactive");
  assert.equal(s.stats("auto_car_repair").chunks, 2);
  s.close();
});
