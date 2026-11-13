/*!
 * Human Bot — embeddable video sales widget.
 * Drop this one line before </body>:
 *   <script src="http://127.0.0.1:4002/widget.js" data-persona="hans-hyundai" defer></script>
 *
 * Behaviour and assets are original to this project. No third-party avatar art,
 * voice samples or prompt text is copied.
 */
(function () {
  "use strict";
  var tag = document.currentScript || (function () {
    var all = document.getElementsByTagName("script");
    return all[all.length - 1];
  })();
  var CFG = window.__HHBOT_CONFIG || {};
  var BASE = CFG.base || (tag && tag.src ? tag.src.replace(/\/widget\.js.*$/, "") : "");
  if (!BASE) return;
  /* the card's origin: postMessage is sent to it and only accepted from it */
  var ORIGIN = "";
  try { ORIGIN = new URL(BASE, location.href).origin; } catch (e) {}
  if (!ORIGIN || ORIGIN === "null") return;

  /* data-* attributes come from the host page's markup: text is escaped before it goes into innerHTML and
     colours must be a hex value or a plain CSS colour name (no url(), no ;, no quotes) */
  function escHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; });
  }
  function safeColor(v, dflt) {
    v = String(v || "").trim();
    return (/^#[0-9a-f]{3,8}$/i.test(v) || /^[a-z]{3,20}$/i.test(v)) ? v : dflt;
  }
  var PERSONA = CFG.persona || (tag && tag.getAttribute("data-persona")) || "";
  var LABEL = escHtml((tag && tag.getAttribute("data-label")) || "Talk to sales");
  var ACCENT = safeColor(tag && tag.getAttribute("data-accent"), "#0b2f6b");
  var SIDE = ((tag && tag.getAttribute("data-position")) || "right").toLowerCase() === "left" ? "left" : "right";
  /* "bubble" (default): round photo with a glowing ring, like a live video presence.
     "pill": the older photo + text capsule. */
  var LAUNCHER = ((tag && tag.getAttribute("data-launcher")) || "bubble").toLowerCase() === "pill" ? "pill" : "bubble";
  var RING = safeColor(tag && tag.getAttribute("data-ring"), "#2f8bff");
  if (window.__hhBotLoaded) return;
  window.__hhBotLoaded = true;

  /* page context: which model page is the visitor on? */
  var MODELS = {
    "hyundai-creta": "Hyundai Creta", "hyundai-creta-n-line": "Creta N Line",
    "hyundai-creta-electric": "Creta Electric", "hyundai-venue": "Hyundai Venue",
    "hyundai-venue-n-line": "Venue N Line", "hyundai-exter": "Hyundai Exter",
    "hyundai-verna": "Hyundai Verna", "hyundai-alcazar": "Hyundai Alcazar",
    "hyundai-aura": "Hyundai Aura", "hyundai-grand-i10-nios": "Grand i10 NIOS",
    "hyundai-all-new-i20": "Hyundai i20", "hyundai-i20-n-line": "i20 N Line"
  };
  function pageContext() {
    var path = (location.pathname || "").replace(/^\/+|\/+$/g, "").toLowerCase();
    var slug = path.split("/").pop() || "";
    var model = MODELS[slug] || MODELS["hyundai-" + slug] || "";
    if (!model) {
      var t = (document.title || "").toLowerCase();
      for (var k in MODELS) {
        var short = MODELS[k].replace(/^hyundai\s+/i, "").toLowerCase();
        if (short && t.indexOf(short) !== -1) { model = MODELS[k]; break; }
      }
    }
    return {
      page: {
        url: location.href,
        title: document.title,
        model: model,
        source: "website_widget"
      }
    };
  }
  var ctx = pageContext();

  var host = document.createElement("div");
  host.id = "hh-human-bot";
  host.style.cssText = "position:fixed;z-index:2147483000;inset:auto 0 0 auto;width:0;height:0;";
  document.body.appendChild(host);
  var root = host.attachShadow ? host.attachShadow({ mode: "open" }) : host;

  var style = document.createElement("style");
  style.textContent = [
    ":host{all:initial}",
    "*{box-sizing:border-box;font-family:'Segoe UI',Inter,system-ui,-apple-system,sans-serif}",
    ".wrap{position:fixed;bottom:20px;" + SIDE + ":20px;display:flex;flex-direction:column;align-items:" + (SIDE === "right" ? "flex-end" : "flex-start") + ";gap:12px}"   /* "flex-right" is not CSS: on phones (full-width wrap) the launcher fell to the left */,
    /* portrait video card: the iframe draws its own header/controls, so no bar here */
    ".panel{width:320px;height:min(480px,calc(100vh - 60px));height:min(480px,calc(100dvh - 60px));background:#000;border-radius:18px;overflow:hidden;",
    "  display:none;flex-direction:column;",
    "  transform:translateX(24px) scale(.98);opacity:0;transition:transform .22s cubic-bezier(.2,.8,.2,1),opacity .18s ease,width .25s ease,height .25s ease}",
    ".panel.open{display:flex;transform:none;opacity:1}",
    /* mini: the card's minimize icon turns it into a passport-size photo of Neha; the call keeps going.
       Size follows the screen: ~12% of the viewport width, never smaller than 120 px or larger than 170 px. */
    ".panel.mini{width:clamp(120px,12vw,170px);height:auto;aspect-ratio:150/190;border-radius:16px}",
    /* tablets and small laptops: a little wider card, bounded by the viewport height */
    "@media(min-width:521px) and (max-width:1024px){.panel{width:min(340px,calc(100vw - 40px));height:min(540px,calc(100dvh - 48px))}}",
    /* large screens: the card may use more height, the width stays readable */
    "@media(min-width:1440px){.panel{width:360px;height:min(560px,calc(100dvh - 60px))}}",
    /* landscape phones (short viewport): full height, keep the photo small */
    "@media(max-height:500px) and (max-width:1024px){.wrap{bottom:8px}.panel{height:calc(100dvh - 16px);max-height:calc(100dvh - 16px)}.panel.mini{width:110px}}",
    ".panel .bar{display:none}",
    ".bar{display:flex;align-items:center;gap:8px;padding:8px 10px 8px 14px;background:" + ACCENT + ";color:#fff;font-size:12px;font-weight:600;letter-spacing:.2px}",
    ".bar .dot{width:7px;height:7px;border-radius:50%;background:#57e0a5;box-shadow:0 0 0 3px rgba(87,224,165,.22)}",
    ".bar .x{margin-left:auto;background:rgba(255,255,255,.14);border:0;color:#fff;width:26px;height:26px;",
    "  border-radius:8px;cursor:pointer;font-size:14px;line-height:1}",
    ".bar .x:hover{background:rgba(255,255,255,.26)}",
    "iframe{flex:1;width:100%;border:0;display:block;background:#000;color-scheme:normal;outline:none}"   /* focused on open: the ring showed round the see-through card */,
    ".launcher{display:flex;align-items:center;gap:10px;background:#fff;border:1px solid #e3e8ef;border-radius:999px;",
    "  padding:6px 13px 6px 6px;cursor:pointer;box-shadow:0 10px 30px rgba(8,18,34,.18);transition:transform .15s ease,box-shadow .15s ease}",
    ".launcher:hover{transform:translateY(-1px);box-shadow:0 14px 34px rgba(8,18,34,.24)}",
    ".launcher .pic{width:36px;height:36px;border-radius:50%;object-fit:cover;background:#0b1522;display:block}",
    ".launcher .txt{display:flex;flex-direction:column;align-items:flex-start;line-height:1.15}",
    ".launcher .txt b{font-size:12px;color:#0d1520;font-weight:650}",
    ".launcher .txt span{font-size:10.5px;color:#66748a}",
    ".launcher .pulse{width:8px;height:8px;border-radius:50%;background:#15b77a;margin-left:2px;",
    "  box-shadow:0 0 0 0 rgba(21,183,122,.5);animation:p 2.2s infinite}",
    "@keyframes p{0%{box-shadow:0 0 0 0 rgba(21,183,122,.55)}70%{box-shadow:0 0 0 9px rgba(21,183,122,0)}100%{box-shadow:0 0 0 0 rgba(21,183,122,0)}}",
    ".teaser{max-width:230px;background:#fff;border:1px solid #e3e8ef;border-radius:12px;border-bottom-" + SIDE + "-radius:5px;",
    "  padding:9px 12px;font-size:12px;color:#28323f;box-shadow:0 10px 26px rgba(8,18,34,.14);display:none;position:relative}",
    ".teaser.show{display:block}",
    ".teaser b{color:" + ACCENT + "}",
    ".teaser .close{position:absolute;top:4px;right:6px;border:0;background:none;color:#95a1b1;cursor:pointer;font-size:13px}",
    /* bubble launcher: dark disc, photo, glowing ring that slowly rotates, live dot, name tag on hover */
    ".bubble{position:relative;width:68px;height:68px;border:0;padding:0;background:none;cursor:pointer;outline:none}",
    ".bubble .ring{position:absolute;inset:0;border-radius:50%;padding:3px;",
    "  background:conic-gradient(from 0deg," + RING + " 0 62%,rgba(47,139,255,.15) 62% 78%," + RING + " 78% 100%);",
    "  -webkit-mask:radial-gradient(farthest-side,transparent calc(100% - 3.5px),#000 calc(100% - 3px));mask:radial-gradient(farthest-side,transparent calc(100% - 3.5px),#000 calc(100% - 3px));",
    "  animation:spin 6s linear infinite}",
    ".bubble .glow{position:absolute;inset:-6px;border-radius:50%;box-shadow:0 0 0 0 rgba(47,139,255,.45),0 12px 34px rgba(8,18,34,.35);animation:breathe 2.6s ease-out infinite}",
    ".bubble .disc{position:absolute;inset:5px;border-radius:50%;background:#0b1522;overflow:hidden;box-shadow:inset 0 0 0 1px rgba(255,255,255,.08)}",
    ".bubble .disc img{width:100%;height:100%;object-fit:cover;object-position:50% 18%;display:block}",
    ".bubble .orbit{position:absolute;inset:0;border-radius:50%;pointer-events:none;animation:orbit 6s linear infinite}",
    ".bubble .live{position:absolute;top:-7px;left:50%;margin-left:-8px;width:16px;height:16px;border-radius:50%;background:#15b77a;border:3px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.3)}",
    ".bubble .tag{position:absolute;bottom:50%;" + (SIDE === "right" ? "right" : "left") + ":calc(100% + 12px);transform:translateY(50%);white-space:nowrap;",
    "  background:#0d1520;color:#fff;font-size:12px;font-weight:600;padding:7px 11px;border-radius:9px;opacity:0;pointer-events:none;transition:opacity .15s ease;box-shadow:0 8px 20px rgba(8,18,34,.25)}",
    ".bubble .tag:after{content:'';position:absolute;top:50%;" + (SIDE === "right" ? "right:-5px" : "left:-5px") + ";width:10px;height:10px;background:#0d1520;transform:translateY(-50%) rotate(45deg)}",
    ".bubble:hover .tag,.bubble:focus-visible .tag{opacity:1}",
    ".bubble:hover .glow{box-shadow:0 0 0 7px rgba(47,139,255,.18),0 16px 40px rgba(8,18,34,.4)}",
    "@keyframes spin{to{transform:rotate(360deg)}}",
    "@keyframes orbit{from{transform:rotate(135deg)}to{transform:rotate(495deg)}}",
    "@keyframes breathe{0%{box-shadow:0 0 0 0 rgba(47,139,255,.45),0 12px 34px rgba(8,18,34,.35)}70%{box-shadow:0 0 0 14px rgba(47,139,255,0),0 12px 34px rgba(8,18,34,.35)}100%{box-shadow:0 0 0 0 rgba(47,139,255,0),0 12px 34px rgba(8,18,34,.35)}}",
    "@media(max-width:520px){.wrap{" + SIDE + ":10px;bottom:12px;width:calc(100vw - 20px)}.panel{width:100%;height:calc(100dvh - 24px);max-height:calc(100dvh - 24px);border-radius:18px}",
    "  .panel.mini{width:clamp(110px,32vw,140px);height:auto;max-height:none;aspect-ratio:150/190;border-radius:14px;margin-" + (SIDE === "right" ? "left" : "right") + ":auto}",
    "  .launcher{padding:5px 11px 5px 5px}.launcher .pic{width:34px;height:34px}.bubble{width:62px;height:62px}}"
  ].join("\n");

  var wrap = document.createElement("div");
  wrap.className = "wrap";
  wrap.innerHTML =
    '<div class="teaser" id="teaser"><button class="close" aria-label="dismiss">✕</button>' +
    '<b id="tname">Hi!</b> <span id="ttext">Creta ke offers ya test drive ke liye baat karein?</span></div>' +
    '<div class="panel" id="panel"><div class="bar"><span class="dot"></span>' +
    '<span id="barname">Sales</span><button class="x" id="close" aria-label="close">✕</button></div>' +
    '<iframe id="frame" title="Sales consultant" allow="microphone; autoplay; clipboard-write"></iframe></div>' +
    (LAUNCHER === "bubble"
      ? '<button class="bubble" id="open" aria-label="' + LABEL + '">' +
        '<span class="glow"></span><span class="ring"></span>' +
        '<span class="disc"><img src="' + BASE + '/avatar.jpg" alt=""></span>' +
        '<span class="orbit" aria-hidden="true"><span class="live"></span></span>' +
        '<span class="tag" id="lname">' + LABEL + '</span></button>'
      : '<button class="launcher" id="open" aria-label="' + LABEL + '">' +
        '<img class="pic" src="' + BASE + '/avatar.jpg" alt="">' +
        '<span class="txt"><b id="lname">' + LABEL + '</b><span>Video · replies in seconds</span></span>' +
        '<i class="pulse"></i></button>');

  root.appendChild(style);
  root.appendChild(wrap);

  var panel = wrap.querySelector("#panel");
  var frame = wrap.querySelector("#frame");
  var teaser = wrap.querySelector("#teaser");
  var loaded = false;
  var open = false;

  function frameUrl() {
    var p = new URLSearchParams();
    if (PERSONA) p.set("persona", PERSONA);
    p.set("accent", ACCENT);
    p.set("ctx", JSON.stringify(ctx));
    p.set("autostart", "1");   // the click on the launcher is the user gesture: the card greets at once, no start button
    p.set("host", location.origin);   // the card pins its postMessage traffic to this page's origin
    /* data-clips="neha" (default) | "sample" — which reaction-clip set the card plays */
    var clips = tag && tag.getAttribute("data-clips");
    if (clips) p.set("clips", clips);
    return BASE + "/embed?" + p.toString();
  }

  function toFrame(msg) {
    try { frame.contentWindow.postMessage(msg, ORIGIN); } catch (e) {}
  }
  function show() {
    open = true;
    if (!loaded) {
      ctx = pageContext();
      if (siteFormButton()) ctx.page.site_form = true;   // the server lets Neha fill this page's own booking form
      frame.src = frameUrl();
      loaded = true;
      frame.addEventListener("load", function () { toFrame({ type: "hhbot:ctx", ctx: ctx }); });
    } else {
      toFrame({ type: "hhbot:open" });   // re-opened: the card re-arms its mic and idle motion
    }
    panel.classList.add("open");
    panel.classList.remove("mini");   // the launcher always opens the full card
    teaser.classList.remove("show");
    wrap.querySelector("#open").style.display = "none";
    try { frame.contentWindow.focus(); } catch (e) {}
    try {
      var b = JSON.parse(sessionStorage.getItem("hhbot_seen") || "{}");
      b.open = 1; sessionStorage.setItem("hhbot_seen", JSON.stringify(b));
    } catch (e) {}
  }
  function hide() {
    open = false;
    panel.classList.remove("open");
    wrap.querySelector("#open").style.display = LAUNCHER === "bubble" ? "block" : "flex";
    if (loaded) toFrame({ type: "hhbot:close" });   // hidden card: mic off, speech cut
  }
  function toggle() { open ? hide() : show(); }

  wrap.querySelector("#open").addEventListener("click", show);
  wrap.querySelector("#close").addEventListener("click", hide);
  teaser.querySelector(".close").addEventListener("click", function (e) {
    e.stopPropagation(); teaser.classList.remove("show");
  });
  teaser.addEventListener("click", show);
  document.addEventListener("keydown", function (e) { if (e.key === "Escape" && open) hide(); });
  window.addEventListener("message", function (e) {
    if (e.origin !== ORIGIN || e.source !== frame.contentWindow) return;   // only our own card may collapse the panel
    var d = e.data || {};
    if (d.type === "hhbot:close") hide();
    if (d.type === "hhbot:size") panel.classList.toggle("mini", d.size === "mini");
    if (d.type === "hhbot:navigate") { navigate(d.path); if (isPhone()) goMini(); }
    if (d.type === "hhbot:siteform") fillSiteForm(d.data || {}).then(function (r) {
      toFrame({ type: "hhbot:siteform_result", ok: r.ok, ref: r.ref || "", error: r.error || "" });
    });
  });

  /* Phones: the card fills the screen. When the customer scrolls the website (or Neha opens a page / fills the
     site's form) the card shrinks to her photo in the corner; the call keeps going and she keeps talking from it.
     Tapping the photo brings the full card back (embed.html). */
  function isPhone() { try { return window.matchMedia("(max-width:520px)").matches; } catch (e) { return false; } }
  function goMini() {
    if (!open || panel.classList.contains("mini")) return;
    panel.classList.add("mini");
    toFrame({ type: "hhbot:size", size: "mini" });
  }
  var scrollFrom = null;
  window.addEventListener("scroll", function () {
    if (!open || !isPhone() || panel.classList.contains("mini")) { scrollFrom = null; return; }
    var y = window.pageYOffset || 0;
    if (scrollFrom === null) scrollFrom = y;
    if (Math.abs(y - scrollFrom) > 40) { scrollFrom = null; goMini(); }
  }, { passive: true });
  document.addEventListener("touchmove", function (e) {
    if (open && isPhone() && !panel.classList.contains("mini") && e.target !== host) goMini();
  }, { passive: true });

  /* ---- the website's own "Book a Service" form (ACR, autocarrepair.in) ----
     Neha's server sends the confirmed details (crm/site_form.py); this opens the site's 3-step form, types them in
     the way a person would (React sees real input/change events), and presses Submit Inquiry. The steps are found
     by their placeholders / option texts below: if the site changes its form, update SITE_FORM. Any step that does
     not work ends with {ok:false} and the bot's server files the lead itself instead (nothing is lost).
     data-site-form="off" on the script tag turns it off. */
  var SITE_FORM = {
    open: /^\s*book a service\b/i,
    name: "John Doe", phone: "10-digit number", email: "john@example.com", reg: "HR 26 XX 0000", year: "e.g. 2022",
    desc: "Describe the problem...",
    location: /moti nagar/i, brand: /select brand/i, model: /select model/i, service: /select service type/i,
    next: /next step/i, submit: /submit inquiry/i, done: /request received|thank you/i
  };
  function vis(e) { var r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0 && getComputedStyle(e).visibility !== "hidden"; }
  function siteFormButton() {
    if (tag && tag.getAttribute("data-site-form") === "off") return null;
    var all = document.querySelectorAll("button,a,[role=button]"), i;
    for (i = 0; i < all.length; i++) if (SITE_FORM.open.test(all[i].innerText || "") && vis(all[i])) return all[i];
    for (i = 0; i < all.length; i++) if (SITE_FORM.open.test(all[i].textContent || "")) return all[i];
    return null;
  }
  /* every lookup is scoped to the form's popup: the page has look-alike controls of its own (a service-centre
     picker in the header matched "Moti Nagar" and was changed instead of the form's location, 2026-11-13 test) */
  var FORM_ROOT = document;
  function popupOf(el) {
    var c = el;
    while (c && c !== document.body && getComputedStyle(c).position !== "fixed") c = c.parentElement;
    return c && c !== document.body ? c : document;
  }
  function formRoot() { if (FORM_ROOT !== document && !FORM_ROOT.isConnected) FORM_ROOT = document; return FORM_ROOT; }
  function byPh(ph) {
    var all = formRoot().querySelectorAll("input,textarea");
    for (var i = 0; i < all.length; i++) if (all[i].placeholder === ph && vis(all[i])) return all[i];
    return null;
  }
  function selectWith(rx) {
    var all = formRoot().querySelectorAll("select");
    for (var i = 0; i < all.length; i++) {
      if (!vis(all[i])) continue;
      for (var j = 0; j < all[i].options.length; j++) if (rx.test(all[i].options[j].text)) return all[i];
    }
    return null;
  }
  function button(rx) {
    var all = formRoot().querySelectorAll("button");
    for (var i = 0; i < all.length; i++) if (rx.test(all[i].innerText || "") && vis(all[i])) return all[i];
    return null;
  }
  function setVal(el, v) {
    var proto = el.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : el.tagName === "SELECT" ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, "value").set.call(el, v);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }
  function norm(t) { return String(t || "").toLowerCase().replace(/[^a-z0-9]/g, ""); }
  /* the option whose text matches best: equal, then starts with, then contains (closest length wins); never a placeholder */
  function pick(sel, want) {
    var w = norm(want), best = null, score = 0;
    if (!sel || !w) return false;
    for (var i = 0; i < sel.options.length; i++) {
      var o = sel.options[i], t = norm(o.text);
      if (!o.value || !t || /^select/.test(t)) continue;
      var sc = t === w ? 1000 : t.indexOf(w) === 0 ? 500 - t.length : (t.indexOf(w) >= 0 || w.indexOf(t) >= 0) ? 200 - Math.abs(t.length - w.length) : 0;
      if (sc > score) { score = sc; best = o; }
    }
    if (!best) return false;
    setVal(sel, best.value);
    return true;
  }
  function wait(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function waitFor(fn, ms) {
    var t0 = Date.now();
    return new Promise(function (res) {
      (function tick() { var v = fn(); if (v) return res(v); if (Date.now() - t0 > ms) return res(null); setTimeout(tick, 120); })();
    });
  }
  function siteDone() {
    var all = document.querySelectorAll("h1,h2,h3,h4,p,div");
    for (var i = 0; i < all.length; i++) if (all[i].children.length < 3 && SITE_FORM.done.test(all[i].innerText || "") && vis(all[i])) return all[i];
    return null;
  }
  function fillSiteForm(d) {
    var step = "open";
    function fail(why) { return { ok: false, error: step + ": " + why }; }
    if (isPhone()) goMini();
    var b = siteFormButton();
    if (!b) return Promise.resolve(fail("no Book a Service button"));
    try { b.scrollIntoView({ block: "center" }); } catch (e) {}
    FORM_ROOT = document;
    b.click();
    return waitFor(function () { return byPh(SITE_FORM.name); }, 6000).then(function (nameEl) {
      if (!nameEl) return fail("form did not open");
      FORM_ROOT = popupOf(nameEl);
      step = "details";
      setVal(nameEl, d.name || "");
      return wait(250).then(function () {
        var ph = byPh(SITE_FORM.phone); if (ph) setVal(ph, d.phone || "");
        var em = byPh(SITE_FORM.email); if (em && d.email) setVal(em, d.email);
        var loc = selectWith(SITE_FORM.location); if (loc && d.location) pick(loc, d.location);
        return wait(500);
      }).then(function () {
        var n = button(SITE_FORM.next); if (!n) return fail("no Next button");
        n.click();
        return waitFor(function () { return selectWith(SITE_FORM.brand); }, 4000).then(function (brandSel) {
          if (!brandSel) return fail("step 1 not accepted");
          step = "vehicle";
          var reg = byPh(SITE_FORM.reg); if (reg && d.registration) setVal(reg, d.registration);
          if (!pick(brandSel, d.brand)) return fail("brand " + (d.brand || "missing"));
          return waitFor(function () { var m = selectWith(SITE_FORM.model); return m && m.options.length > 1 ? m : null; }, 4000).then(function (modelSel) {
            if (!modelSel || !pick(modelSel, d.model)) return fail("model " + (d.model || "missing"));
            var yr = byPh(SITE_FORM.year); if (yr && d.year) setVal(yr, d.year);
            return wait(500).then(function () {
              var n2 = button(SITE_FORM.next); if (!n2) return fail("no Next button");
              n2.click();
              return waitFor(function () { return selectWith(SITE_FORM.service); }, 4000).then(function (svcSel) {
                if (!svcSel) return fail("step 2 not accepted");
                step = "service";
                if (!pick(svcSel, d.service)) return fail("service " + d.service);
                var ds = byPh(SITE_FORM.desc); if (ds && d.description) setVal(ds, d.description);
                return wait(700).then(function () {
                  var sb = button(SITE_FORM.submit); if (!sb) return fail("no Submit button");
                  step = "submit";
                  sb.click();
                  return waitFor(siteDone, 15000).then(function (ok) {
                    if (!ok) return fail("no confirmation from the site");
                    var box = ok.parentElement || ok, m = /reference\s*(?:no\.?|number|id)?\s*[:#]?\s*([A-Z0-9-]{4,})/i.exec(box.innerText || "");
                    return { ok: true, ref: m ? m[1] : "" };
                  });
                });
              });
            });
          });
        });
      });
    }).catch(function (e) { return fail(String((e && e.message) || e).slice(0, 80)); });
  }

  /* "insurance page dikhao": open a page of THIS site. Only a same-origin path is accepted. A single-page site
     (React Router & co. listen to popstate) changes the page in place, so the card and the call keep running;
     if nothing changed after 1.5 s the page is loaded normally and the card re-opens itself after the load.
     data-navigate="off" on the script tag turns it off. */
  function pageText() {
    var m = document.getElementById("root") || document.querySelector("main") || document.body;
    return m ? (m.innerText || "").slice(0, 600) : "";
  }
  function navigate(path) {
    if (tag && tag.getAttribute("data-navigate") === "off") return;
    if (typeof path !== "string" || path.charAt(0) !== "/" || path.charAt(1) === "/") return;
    var url;
    try { url = new URL(path, location.origin); } catch (e) { return; }
    if (url.origin !== location.origin || url.pathname === location.pathname) return;
    var before = pageText();
    try {
      history.pushState({}, "", url.pathname + url.search);
      window.dispatchEvent(new PopStateEvent("popstate", { state: {} }));
      window.scrollTo(0, 0);
    } catch (e) { reload(url.href); return; }
    setTimeout(function () {
      if (pageText() === before) { reload(url.href); return; }   // not a client-side router: a real page load
      ctx = pageContext();
      toFrame({ type: "hhbot:ctx", ctx: ctx });
    }, 1500);
  }
  function reload(href) {
    try { sessionStorage.setItem("hhbot_reopen", "1"); } catch (e) {}
    location.assign(href);
  }
  try {
    if (sessionStorage.getItem("hhbot_reopen")) { sessionStorage.removeItem("hhbot_reopen"); setTimeout(show, 600); }
  } catch (e) {}

  /* name + teaser copy come from the persona registry, so the widget stays brand-correct */
  fetch(BASE + "/personas").then(function (r) { return r.json(); }).then(function (d) {
    var p = (d.personas || []).filter(function (x) { return x.id === (PERSONA || d.default); })[0];
    if (!p) return;
    wrap.querySelector("#lname").textContent = "Chat with " + p.name;
    wrap.querySelector("#barname").textContent = p.name + " · " + p.company;
    wrap.querySelector("#tname").textContent = p.name + ":";
    var m = ctx.page && ctx.page.model;
    var service = p.business_type === "service";
    wrap.querySelector("#ttext").textContent = service
      ? (m ? m + " ki service ya repair ke liye baat karein? Free pickup available hai."
           : "Car service ya repair chahiye? Main quote aur pickup arrange kar deti hoon.")
      : (m ? m + " ke price, offers ya test drive ke liye baat karein?"
           : "Kaunsi car dekh rahe hain? Main help kar deti hoon.");
  }).catch(function () {});

  setTimeout(function () {
    if (open) return;
    try { if ((JSON.parse(sessionStorage.getItem("hhbot_seen") || "{}").open)) return; } catch (e) {}
    teaser.classList.add("show");
  }, 7000);

  window.HansHyundaiBot = { open: show, close: hide, toggle: toggle, context: ctx };
})();
