/*!
 * Human Bot — embeddable video receptionist widget (Neha on autocarrepair.in).
 * Drop this one line before </body>, on every page (BOT-HOST = the bot server's public HTTPS origin):
 *   <script src="https://BOT-HOST/widget.js" data-persona="auto-car-repair" defer></script>
 * Install guide: human_bot/web/INSTALL_ON_WEBSITE.md. Options and API: human_bot/web/EMBED.md.
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
    /* --lift: how far the widget stands above the site's own bottom-corner things (WhatsApp/chat buttons, the cookie
       banner, a sticky "Check prices" bar), measured by avoidCorner() below. The panel heights subtract it too. */
    ".wrap{position:fixed;bottom:calc(20px + var(--lift,0px));" + SIDE + ":20px;display:flex;flex-direction:column;align-items:" + (SIDE === "right" ? "flex-end" : "flex-start") + ";gap:12px;transition:bottom .2s ease}"   /* "flex-right" is not CSS: on phones (full-width wrap) the launcher fell to the left */,
    /* the wrap is a see-through box (full width on phones): only its children take taps, so the site's cookie
       buttons and sticky bars under it stay usable (live test 2026-11-25: Reject/Accept and "Check prices" were dead) */
    ".wrap{pointer-events:none}.wrap>*{pointer-events:auto}",
    /* portrait video card: the iframe draws its own header/controls, so no bar here */
    ".panel{width:320px;height:min(480px,calc(100vh - 60px - var(--lift,0px)));height:min(480px,calc(100dvh - 60px - var(--lift,0px)));background:#000;border-radius:18px;overflow:hidden;",
    "  box-shadow:0 18px 48px rgba(8,18,34,.34),0 3px 10px rgba(8,18,34,.22);",   /* lifts the dark card off a white site (it looked pasted on) */
    "  display:none;flex-direction:column;",
    "  transform:translateX(24px) scale(.98);opacity:0;transition:transform .22s cubic-bezier(.2,.8,.2,1),opacity .18s ease,width .25s ease,height .25s ease}",
    ".panel.open{display:flex;transform:none;opacity:1}",
    /* mini: the card's minimize icon turns it into a passport-size photo of Neha; the call keeps going.
       Size follows the screen: ~12% of the viewport width, never smaller than 120 px or larger than 170 px. */
    ".panel.mini{width:clamp(120px,12vw,170px);height:auto;aspect-ratio:150/190;border-radius:16px}",
    /* tablets and small laptops: a little wider card, bounded by the viewport height */
    "@media(min-width:521px) and (max-width:1024px){.panel{width:min(340px,calc(100vw - 40px));height:min(540px,calc(100dvh - 48px - var(--lift,0px)))}}",
    /* large screens: the card may use more height, the width stays readable */
    "@media(min-width:1440px){.panel{width:360px;height:min(560px,calc(100dvh - 60px - var(--lift,0px)))}}",
    /* landscape phones (short viewport): full height, keep the photo small */
    "@media(max-height:500px) and (max-width:1024px){.wrap{bottom:calc(8px + var(--lift,0px))}.panel{height:calc(100dvh - 16px - var(--lift,0px));max-height:calc(100dvh - 16px - var(--lift,0px))}.panel.mini{width:110px}}",
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
    /* teaser: a small white card above the bubble, one short line, a quiet close cross (2026-11-24 redesign) */
    ".teaser{max-width:240px;background:#fff;border:0;border-radius:12px;",
    "  padding:10px 32px 10px 14px;font-size:14px;line-height:1.3;color:#1f2733;box-shadow:0 8px 24px rgba(8,18,34,.22),0 1px 3px rgba(8,18,34,.12);",
    "  display:none;position:relative;cursor:pointer;margin-" + SIDE + ":4px;transform-origin:bottom " + SIDE + "}",
    ".teaser.show{display:block;animation:pop .28s cubic-bezier(.2,.9,.3,1.2)}",
    ".teaser b{color:#1f2733;font-weight:500}",
    ".teaser .sub{display:none;font-size:12px;color:#6a7587;margin-top:2px}",   /* one short line; data-teaser-sub="on" shows the second */
    ".teaser.two .sub{display:block}",
    ".teaser .close{position:absolute;top:7px;right:8px;width:20px;height:20px;border:0;border-radius:50%;background:none;color:#7d8899;cursor:pointer;font-size:12px;line-height:20px;padding:0}",
    ".teaser .close:hover{background:#eef1f5;color:#2b3442}",
    "@keyframes pop{from{opacity:0;transform:translateY(6px) scale(.96)}to{opacity:1;transform:none}}",
    /* bubble launcher: Neha's living portrait (a muted 12 s loop over the still photo) in a thin steady ring,
       a small green "online" dot, a soft breathing shadow, name tag on hover */
    ".bubble{position:relative;width:80px;height:80px;border:0;padding:0;background:none;cursor:pointer;outline:none;transition:transform .18s ease}",
    ".bubble:hover{transform:scale(1.05)}",
    ".bubble .ring{position:absolute;inset:0;border-radius:50%;border:2px solid " + RING + "}",
    ".bubble .glow{position:absolute;inset:0;border-radius:50%;box-shadow:0 10px 28px rgba(8,18,34,.42),0 2px 6px rgba(8,18,34,.25)}",   /* a soft shadow, no halo */
    ".bubble .disc{position:absolute;inset:3px;border-radius:50%;background:#e9edf2;overflow:hidden}",
    ".bubble .disc img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:50% 0%;display:block}",   /* head and shoulders, like the video */
    ".bubble .disc video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block}",
    ".bubble .disc video{opacity:0;transition:opacity .4s ease}",
    ".bubble .disc video.on{opacity:1}",
    ".bubble .orbit{position:absolute;inset:0;border-radius:50%;pointer-events:none}",
    ".bubble .live{position:absolute;bottom:3px;" + (SIDE === "right" ? "left" : "right") + ":3px;width:15px;height:15px;border-radius:50%;background:#16c47f;border:2.5px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.3)}",
    "@media(prefers-reduced-motion:reduce){.bubble .glow{animation:none}.teaser.show{animation:none}}",
    ".bubble .tag{position:absolute;bottom:50%;" + (SIDE === "right" ? "right" : "left") + ":calc(100% + 12px);transform:translateY(50%);white-space:nowrap;",
    "  background:#0d1520;color:#fff;font-size:12px;font-weight:600;padding:7px 11px;border-radius:9px;opacity:0;pointer-events:none;transition:opacity .15s ease;box-shadow:0 8px 20px rgba(8,18,34,.25)}",
    ".bubble .tag:after{content:'';position:absolute;top:50%;" + (SIDE === "right" ? "right:-5px" : "left:-5px") + ";width:10px;height:10px;background:#0d1520;transform:translateY(-50%) rotate(45deg)}",
    ".bubble:hover .tag,.bubble:focus-visible .tag{opacity:1}",
    ".bubble:hover .glow{box-shadow:0 0 0 6px rgba(47,139,255,.16),0 16px 40px rgba(8,18,34,.42)}",
    "@keyframes spin{to{transform:rotate(360deg)}}",
    "@keyframes orbit{from{transform:rotate(135deg)}to{transform:rotate(495deg)}}",
    "@keyframes breathe{0%{box-shadow:0 0 0 0 rgba(47,139,255,.45),0 12px 34px rgba(8,18,34,.35)}70%{box-shadow:0 0 0 14px rgba(47,139,255,0),0 12px 34px rgba(8,18,34,.35)}100%{box-shadow:0 0 0 0 rgba(47,139,255,0),0 12px 34px rgba(8,18,34,.35)}}",
    "@media(max-width:520px){.wrap{" + SIDE + ":10px;bottom:calc(12px + var(--lift,0px));width:calc(100vw - 20px)}.panel{width:100%;height:calc(100dvh - 24px - var(--lift,0px));max-height:calc(100dvh - 24px - var(--lift,0px));border-radius:18px}",
    /* the open card is a true full-screen sheet on phones (like WhatsApp / Intercom): no border, no radius, no lift;
       the notch and home bar stay clear when the host page uses viewport-fit=cover. fitKeyboard() pins it above the keyboard. */
    "  .panel.open:not(.mini){position:fixed;top:0;left:0;right:0;bottom:0;width:100%;height:100vh;height:100dvh;max-height:none;border-radius:0;box-shadow:none;",
    "    padding:env(safe-area-inset-top,0px) env(safe-area-inset-right,0px) env(safe-area-inset-bottom,0px) env(safe-area-inset-left,0px);background:#050608}",
    "  .panel.open:not(.mini) iframe{background:#050608}",
    "  .panel.mini{width:clamp(110px,32vw,140px);height:auto;max-height:none;aspect-ratio:150/190;border-radius:14px;margin-" + (SIDE === "right" ? "left" : "right") + ":auto}",
    "  .launcher{padding:5px 11px 5px 5px}.launcher .pic{width:34px;height:34px}.bubble{width:68px;height:68px}.teaser{max-width:210px;font-size:13px}}"
  ].join("\n");

  var wrap = document.createElement("div");
  wrap.className = "wrap";
  wrap.innerHTML =
    '<div class="teaser" id="teaser"><button class="close" aria-label="dismiss">✕</button>' +
    '<b id="tname">Talk to us!</b><span class="sub" id="ttext"></span></div>' +
    '<div class="panel" id="panel"><div class="bar"><span class="dot"></span>' +
    '<span id="barname">Sales</span><button class="x" id="close" aria-label="close">✕</button></div>' +
    '<iframe id="frame" title="Sales consultant" allow="microphone; autoplay; clipboard-write"></iframe></div>' +
    (LAUNCHER === "bubble"
      ? '<button class="bubble" id="open" aria-label="' + LABEL + '">' +
        '<span class="glow"></span><span class="ring"></span>' +
        '<span class="disc"><img src="' + BASE + '/avatar.jpg" alt=""></span>' +   // the video joins in startBubbleVideo()
        '<span class="orbit" aria-hidden="true"><span class="live"></span></span>' +
        '<span class="tag" id="lname">' + LABEL + '</span></button>'
      : '<button class="launcher" id="open" aria-label="' + LABEL + '">' +
        '<img class="pic" src="' + BASE + '/avatar.jpg" alt="">' +
        '<span class="txt"><b id="lname">' + LABEL + '</b><span>Video · replies in seconds</span></span>' +
        '<i class="pulse"></i></button>');

  root.appendChild(style);
  root.appendChild(wrap);

  /* The bubble's living portrait: a muted, looping 200x200 clip of Neha over the still photo. It fades in only once
     it really plays; a 404, a decode error, a blocked autoplay or "reduce motion" leave the still photo. Paused
     while the card is open (the card has the full face) and while the tab is hidden. */
  var bubbleVid = null;
  function startBubbleVideo() {
    var disc = wrap.querySelector(".bubble .disc");
    if (!disc || (window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches)) return;
    if (tag && tag.getAttribute("data-bubble-video") === "off") return;
    var v = document.createElement("video");
    v.muted = true; v.defaultMuted = true; v.loop = true; v.playsInline = true; v.autoplay = true; v.preload = "auto";
    v.setAttribute("muted", ""); v.setAttribute("playsinline", ""); v.setAttribute("aria-hidden", "true");
    v.addEventListener("playing", function () { v.classList.add("on"); });
    v.addEventListener("error", function () { if (v.parentNode) v.parentNode.removeChild(v); bubbleVid = null; });
    v.src = BASE + "/avatar_bubble.mp4";
    disc.appendChild(v);
    bubbleVid = v;
    var p = v.play();
    if (p && p.catch) p.catch(function () { /* autoplay refused: the still photo stays */ });
  }
  function bubbleVideo(on) {
    if (!bubbleVid) return;
    try { if (on && !document.hidden) { var p = bubbleVid.play(); if (p && p.catch) p.catch(function () {}); } else bubbleVid.pause(); } catch (e) {}
  }
  document.addEventListener("visibilitychange", function () { bubbleVideo(!open && !document.hidden); });
  if (LAUNCHER === "bubble") startBubbleVideo();

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
    bubbleVideo(false);
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
    avoidCorner();                    // the open card steps over full-width bars only: re-measure now, not in a second
    try { frame.contentWindow.focus(); } catch (e) {}
    try {
      var b = JSON.parse(sessionStorage.getItem("hhbot_seen") || "{}");
      b.open = 1; sessionStorage.setItem("hhbot_seen", JSON.stringify(b));
    } catch (e) {}
  }
  function hide() {
    open = false;
    bubbleVideo(true);
    setTimeout(function () { try { wrap.querySelector("#open").focus({ preventScroll: true }); } catch (e) {} }, 0);   // keyboard users land back on the bubble
    panel.classList.remove("open");
    wrap.querySelector("#open").style.display = LAUNCHER === "bubble" ? "block" : "flex";
    if (loaded) toFrame({ type: "hhbot:close" });   // hidden card: mic off, speech cut
    avoidCorner();
    fitKeyboard();
  }
  function toggle() { open ? hide() : show(); }

  wrap.querySelector("#open").addEventListener("click", show);
  wrap.querySelector("#close").addEventListener("click", hide);
  teaser.querySelector(".close").addEventListener("click", function (e) {
    e.stopPropagation(); teaser.classList.remove("show");
    try {   // dismissed stays dismissed for this tab (a reload used to bring it back)
      var sn = JSON.parse(sessionStorage.getItem("hhbot_seen") || "{}");
      sn.dismissed = 1; sessionStorage.setItem("hhbot_seen", JSON.stringify(sn));
    } catch (x) {}
  });
  teaser.addEventListener("click", show);
  document.addEventListener("keydown", function (e) { if (e.key === "Escape" && open && e.isTrusted) hide(); });
  window.addEventListener("message", function (e) {
    if (e.origin !== ORIGIN || e.source !== frame.contentWindow) return;   // only our own card may collapse the panel
    var d = e.data || {};
    if (d.type === "hhbot:close") hide();
    if (d.type === "hhbot:size") { panel.classList.toggle("mini", d.size === "mini"); fitKeyboard(); avoidCorner(); }
    if (d.type === "hhbot:navigate") { navigate(d.path); if (isPhone()) goMini(); }
    if (d.type === "hhbot:siteform_open") requestSiteForm(d.data || {});
    if (d.type === "hhbot:siteform") {
      if (FILLING) return;                 // one fill at a time: a second request would click the same popup
      FILLING = true;
      fillSiteForm(d.data || {}).then(function (r) {
        if (!r.ok && r.opened) closeSiteForm();   // never leave the popup this fill opened stuck half-filled
        FORM_ROOT = document;
        FILLING = false;
        toFrame({ type: "hhbot:siteform_result", ok: r.ok, ref: r.ref || "", error: r.error || "" });
      });
    }
  });

  /* Phones: the card fills the screen. When the customer scrolls the website (or Neha opens a page / fills the
     site's form) the card shrinks to her photo in the corner; the call keeps going and she keeps talking from it.
     Tapping the photo brings the full card back (embed.html). */
  function isPhone() { try { return window.matchMedia("(max-width:520px)").matches; } catch (e) { return false; } }
  function goMini() {
    if (!open || panel.classList.contains("mini")) return;
    panel.classList.add("mini");
    toFrame({ type: "hhbot:size", size: "mini" });
    fitKeyboard();
    avoidCorner();
  }
  var scrollFrom = null;
  window.addEventListener("scroll", function () {
    if (!open || !isPhone() || panel.classList.contains("mini") || kbOpen()) { scrollFrom = null; return; }   // the keyboard opening pans the page: not a scroll
    var y = window.pageYOffset || 0;
    if (scrollFrom === null) scrollFrom = y;
    if (Math.abs(y - scrollFrom) > 40) { scrollFrom = null; goMini(); }
  }, { passive: true });
  document.addEventListener("touchmove", function (e) {
    if (open && isPhone() && !panel.classList.contains("mini") && e.target !== host) goMini();
  }, { passive: true });

  /* Phones, typing: the on-screen keyboard covers the bottom of the layout viewport, where the card's text box is
     (Chrome Android and iOS Safari shrink only the visual viewport). While the keyboard is up, the full card is
     pinned to the visible area above it; it goes back to normal when the keyboard closes. */
  var VV = window.visualViewport || null;
  function kbOpen() { return !!(VV && isPhone() && Math.abs((VV.scale || 1) - 1) < 0.05 && VV.height < window.innerHeight - 120); }   // not a pinch-zoom
  var pinned = false;
  function fitKeyboard() {
    if (!VV) return;
    if (open && kbOpen() && !panel.classList.contains("mini")) {
      // the phone sheet is position:fixed over the whole layout viewport: move it onto the visible part
      pinned = true;
      panel.style.top = Math.round(VV.offsetTop) + "px";
      panel.style.bottom = "auto";
      panel.style.height = panel.style.maxHeight = Math.max(200, Math.round(VV.height)) + "px";
    } else if (pinned) {
      pinned = false;
      panel.style.top = panel.style.bottom = panel.style.height = panel.style.maxHeight = "";
    }
  }
  if (VV) { VV.addEventListener("resize", fitKeyboard); VV.addEventListener("scroll", fitKeyboard); }

  /* The site's own bottom-corner furniture: a WhatsApp / "Need help? Chat with us" button, the cookie banner, a
     sticky "Check prices" bar on phones. autocarrepair.in has all three right where the bubble sits (live test
     2026-11-25: the bubble covered the WhatsApp button and the cookie banner's Accept). The bubble and its teaser
     stand just above anything fixed in that corner; the open card (and Neha's photo) only above full-width bars,
     so a cookie banner stays clickable. Hit-testing a few points in the corner once a second is cheap and follows
     the banner going away after "Accept". data-avoid="off" turns it off; data-offset-bottom="NN" adds NN px. */
  var AVOID = !(tag && tag.getAttribute("data-avoid") === "off");
  var EXTRA = Math.max(0, Math.min(400, parseInt((tag && tag.getAttribute("data-offset-bottom")) || "0", 10) || 0));
  var liftNow = -1;
  /* 2026-11-26 speed: a phone CPU profile put ~30% of the page's scripting time in this check (elementsFromPoint +
     getComputedStyle on every ancestor, ~63 points a second). Now: one style lookup per element per measurement
     (memo), half the sample rows, never while the page scrolls, and only in idle time. */
  var fixedMemo = null;
  function fixedOf(el) {
    var chain = [];
    for (var e = el; e && e.nodeType === 1 && e !== document.body && e !== document.documentElement; e = e.parentElement) {
      if (fixedMemo && fixedMemo.has(e)) { var hit = fixedMemo.get(e); for (var k = 0; k < chain.length; k++) fixedMemo.set(chain[k], hit); return hit; }
      chain.push(e);
      if (getComputedStyle(e).position === "fixed") { if (fixedMemo) for (var m = 0; m < chain.length; m++) fixedMemo.set(chain[m], e); return e; }
    }
    if (fixedMemo) for (var n = 0; n < chain.length; n++) fixedMemo.set(chain[n], null);
    return null;
  }
  function measureLift() {
    if (!AVOID || !document.elementsFromPoint) return 0;
    var vw = window.innerWidth, vh = window.innerHeight, phone = isPhone();
    var mini = open && panel.classList.contains("mini");
    var base = phone ? 12 : (vh <= 500 ? 8 : 20), wideOnly = open && !mini, top = vh, modal = false, seen = [];   // Neha's photo is launcher-sized: it steps over everything too
    // sample the strip the widget really uses: the bubble and its teaser when closed, the card when open
    var reach = 200;
    if (mini) reach = (panel.getBoundingClientRect().width || 140) + 10;
    else if (!open) {
      var bw = wrap.querySelector("#open").getBoundingClientRect().width || 80;
      var tw = teaser.classList.contains("show") ? teaser.getBoundingClientRect().width + 6 : 0;
      reach = Math.max(bw, tw) + (phone ? 10 : 20);
    }
    var xs = [24, Math.round(reach / 2), Math.round(reach - 6)];
    fixedMemo = typeof Map === "function" ? new Map() : null;
    for (var y = vh - 4; y > vh * 0.45; y -= 44) {
      for (var i = 0; i < xs.length; i++) {
        var x = SIDE === "right" ? vw - xs[i] : xs[i];
        var list = document.elementsFromPoint(x, y);
        for (var j = 0; j < list.length; j++) {
          var el = list[j];
          if (el === host) continue;
          var f = fixedOf(el);
          if (!f) break;                                   // the page itself: nothing fixed above it at this point
          if (seen.indexOf(f) >= 0) continue;
          seen.push(f);
          var r = f.getBoundingClientRect(), cs = getComputedStyle(f);
          if (r.height > vh * 0.45) { modal = true; break; }  // a popup or its backdrop: what lies under it is hidden anyway
          if (cs.visibility === "hidden" || parseFloat(cs.opacity) < 0.1 || r.bottom < vh - 230 || r.width < 8 || r.height < 8) continue;
          if (wideOnly && r.width < vw * 0.75) continue;      // ACR's phone chat pill is 62% wide: not a bar
          if (r.top < top) top = r.top;
        }
      }
    }
    // A site popup is open on a phone: Neha's photo stays in the corner instead of climbing onto the popup's buttons.
    if (modal && phone && open) return EXTRA;
    var lift = top < vh ? Math.max(0, vh - top + 10 - base) : 0;
    return Math.round(Math.min(lift, vh * 0.4)) + EXTRA;
  }
  var scrolling = false, scrollEndT = 0;
  function avoidNow() {
    if (document.hidden || scrolling) return;
    var l = 0;
    try { l = measureLift(); } catch (e) { l = EXTRA; }
    fixedMemo = null;
    if (Math.abs(l - liftNow) < 2) return;
    liftNow = l;
    wrap.style.setProperty("--lift", l + "px");
  }
  function avoidCorner() {   // in idle time, so it never competes with scrolling or the face animation
    if (document.hidden || scrolling) return;
    if (window.requestIdleCallback) window.requestIdleCallback(avoidNow, { timeout: 700 }); else avoidNow();
  }
  // fixed bars do not move while the page scrolls: measure again once it settles
  window.addEventListener("scroll", function () {
    scrolling = true; clearTimeout(scrollEndT);
    scrollEndT = setTimeout(function () { scrolling = false; avoidCorner(); }, 250);
  }, { passive: true });
  avoidNow();
  setInterval(avoidCorner, 1000);
  window.addEventListener("resize", avoidCorner);

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
  var FORM_ROOT = document, FILLING = false, NAV_PENDING = null, PENDING_FORM = null;
  /* "book karna hai": the service's page opens first, then the booking form on top of it, once the page is there */
  function requestSiteForm(data) {
    if (FILLING) return;
    PENDING_FORM = data;
    FILLING = true;
    (NAV_PENDING || Promise.resolve()).then(function () { return wait(NAV_PENDING === null ? 400 : 0); }).then(function () {
      return openSiteForm(data);
    }).then(function () { PENDING_FORM = null; FILLING = false; }, function () { PENDING_FORM = null; FILLING = false; });
  }
  function popupOf(el) {
    var c = el;
    while (c && c !== document.body && !/fixed|absolute/.test(getComputedStyle(c).position) &&
           !(c.matches && c.matches("[role=dialog],[aria-modal=true],dialog"))) c = c.parentElement;
    return c && c !== document.body ? c : document;
  }
  function formRoot() { if (FORM_ROOT !== document && !FORM_ROOT.isConnected) FORM_ROOT = document; return FORM_ROOT; }
  function byPh(ph) {
    var all = formRoot().querySelectorAll("input,textarea");
    for (var i = 0; i < all.length; i++) if (all[i].placeholder === ph && vis(all[i])) return all[i];
    return null;
  }
  function newField(ph, before) {
    var all = document.querySelectorAll("input,textarea");
    for (var i = 0; i < all.length; i++) {
      var e = all[i];
      if (e.placeholder === ph && vis(e) && before.indexOf(e) < 0 && popupOf(e) !== document) return e;
    }
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
    var w = norm(want), best = null, score = 0, tie = false;
    if (!sel || !w) return false;
    for (var i = 0; i < sel.options.length; i++) {
      var o = sel.options[i], t = norm(o.text);
      if (!o.value || !t || /^select/.test(t)) continue;
      var sc = t === w ? 1000 : t.indexOf(w) === 0 ? 500 - t.length : (t.indexOf(w) >= 0 || w.indexOf(t) >= 0) ? 200 - Math.abs(t.length - w.length) : 0;
      if (sc > score) { score = sc; best = o; tie = false; }
      else if (sc && sc === score) {
        // "swift dzire", "grand i10": the later word names the model. Equal partial matches go to the option found
        // later in what was said; still equal -> ambiguous (below)
        var a = w.lastIndexOf(t), bpos = w.lastIndexOf(norm(best.text));
        if (a > bpos) { best = o; tie = false; } else if (a === bpos) tie = true;
      }
    }
    if (!best || (tie && score < 500)) return false;   // two equally good partial matches: do not submit the wrong car
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
  function doneTexts() {
    var out = [], all = formRoot().querySelectorAll("h1,h2,h3,h4,p,div");
    for (var i = 0; i < all.length; i++) if (all[i].children.length < 3 && SITE_FORM.done.test(all[i].innerText || "") && vis(all[i])) out.push(all[i]);
    return out;
  }
  function siteDone(before) {
    var now = doneTexts();
    for (var i = 0; i < now.length; i++) if (!before || before.indexOf(now[i]) < 0) return now[i];
    return null;
  }
  /* A fill that stopped part-way closes the site's popup again (its X button). Only the popup this script opened
     (FORM_ROOT), never anything else on the page. Without an X it sends Escape to the popup; Neha's own panel only
     reacts to a real (trusted) Escape key, so it stays open. */
  function closeSiteForm() {
    var root = formRoot();
    if (root === document) return;
    var all = root.querySelectorAll("button,[role=button]"), i, b;
    for (i = 0; i < all.length && !b; i++) {
      var lab = (all[i].getAttribute("aria-label") || all[i].getAttribute("title") || "") + " " + (all[i].innerText || "").trim();
      if (/close|band karein|^\s*(×|✕|x)\s*$/i.test(lab) && vis(all[i])) b = all[i];
    }
    for (i = 0; i < all.length && !b; i++) {   // an icon-only button (svg X) in the popup's top part
      var r = all[i].getBoundingClientRect(), top = root.getBoundingClientRect().top;
      if (!(all[i].innerText || "").trim() && all[i].querySelector("svg") && r.top - top < 90 && vis(all[i])) b = all[i];
    }
    if (b) b.click();
    else root.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", code: "Escape", keyCode: 27, bubbles: true }));
    FORM_ROOT = document;
  }
  /* the site's booking popup if it is open already (Neha opened it on "book karna hai", or the customer did) */
  function openPopupName() { return newField(SITE_FORM.name, []); }
  /* "book karna hai": open the website's own booking form and type in what Neha already knows (step 1 only: name,
     phone, centre). The customer finishes it on the site, or tells Neha and she fills the rest after the read-back. */
  function openSiteForm(d) {
    if (isPhone()) goMini();
    var already = openPopupName();
    var start;
    if (already) start = Promise.resolve(already);
    else {
      var b = siteFormButton();
      if (!b) return Promise.resolve({ ok: false, error: "no Book a Service button" });
      try { b.scrollIntoView({ block: "center" }); } catch (e) {}
      var before = [].slice.call(document.querySelectorAll("input,textarea")).filter(vis);
      b.click();
      start = waitFor(function () { return newField(SITE_FORM.name, before); }, 6000);
    }
    return start.then(function (nameEl) {
      if (!nameEl) return { ok: false, error: "form did not open" };
      FORM_ROOT = popupOf(nameEl);
      if (d.name && !nameEl.value) setVal(nameEl, d.name);
      var ph = byPh(SITE_FORM.phone); if (ph && d.phone && !ph.value) setVal(ph, d.phone);
      var em = byPh(SITE_FORM.email); if (em && d.email && !em.value) setVal(em, d.email);
      var loc = selectWith(SITE_FORM.location); if (loc && d.location && !loc.value) pick(loc, d.location);
      FORM_ROOT = document;
      return { ok: true };
    }).catch(function (e) { return { ok: false, error: String((e && e.message) || e).slice(0, 80) }; });
  }
  function fillSiteForm(d) {
    var step = "open";
    var opened = false;
    function fail(why) { return { ok: false, error: step + ": " + why, opened: opened }; }
    // brand + model are required selects on the site's step 2: without them "Next Step" stays disabled and the
    // popup would be left half-filled in front of the customer (2026-11-24). Do not open it; the server files the lead.
    var need = ["name", "phone", "brand", "model"].filter(function (k) { return !String(d[k] || "").trim(); });
    if (need.length) return Promise.resolve(fail("missing " + need.join(",")));
    if (isPhone()) goMini();
    FORM_ROOT = document;
    var already = openPopupName(), waitName;
    if (already) {                         // opened earlier on "book karna hai": fill that one
      opened = true;
      waitName = Promise.resolve(already);
    } else {
      var b = siteFormButton();
      if (!b) return Promise.resolve(fail("no Book a Service button"));
      try { b.scrollIntoView({ block: "center" }); } catch (e) {}
      var before = [].slice.call(document.querySelectorAll("input,textarea")).filter(vis);   // look-alikes already on the page
      b.click();
      opened = true;
      waitName = waitFor(function () { return newField(SITE_FORM.name, before); }, 6000);
    }
    return waitName.then(function (nameEl) {
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
          // the model list loads after the brand (sometimes with a "Loading..." option first): keep trying to pick
          return waitFor(function () { var m = selectWith(SITE_FORM.model); return m && pick(m, d.model) ? m : null; }, 6000).then(function (modelSel) {
            if (!modelSel) return fail("model " + (d.model || "missing"));
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
                  var doneBefore = doneTexts();
                  sb.click();
                  return waitFor(function () { return siteDone(doneBefore); }, 15000).then(function (ok) {
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
    // Wait for the site's router to paint the new page: its route code can take a few seconds to load (seen 2026-11-24:
    // /services -> /category/car-battery took > 1.5 s and the old fixed 1.5 s check forced two needless reloads).
    // Reload only when nothing changed after 8 s (not a client-side router) or the router shows its own 404 (a page
    // served only by the site's backend). Anything waiting on the page (the booking form) runs once it has settled.
    var t0 = Date.now(), last = null, stableSince = 0;
    NAV_PENDING = new Promise(function (done) {
      (function check() {
        var txt = pageText(), h1 = document.querySelector("h1");
        if (/page not found|^\s*404\b/i.test((h1 && h1.innerText) || "") || /page not found/i.test(txt)) { done(); reload(url.href); return; }
        if (txt !== before && txt.length > 80) {
          if (txt === last) {
            if (Date.now() - stableSince >= 500) {     // changed, and unchanged for half a second: the page is there
              ctx = pageContext();
              toFrame({ type: "hhbot:ctx", ctx: ctx });
              NAV_PENDING = null;
              return done();
            }
          } else { last = txt; stableSince = Date.now(); }
        }
        // 8 s, not 5: on a busy phone the site's route chunk took > 5 s (live test 2026-11-25) and the forced reload
        // restarts the conversation (the card keeps no session across a page load)
        if (Date.now() - t0 > 8000) { done(); if (txt === before) reload(url.href); NAV_PENDING = null; return; }
        setTimeout(check, 200);
      })();
    });
  }
  function reload(href) {
    try { sessionStorage.setItem("hhbot_reopen", "1"); } catch (e) {}
    try { if (PENDING_FORM) sessionStorage.setItem("hhbot_form", JSON.stringify(PENDING_FORM)); } catch (e) {}
    location.assign(href);
  }
  try {
    if (sessionStorage.getItem("hhbot_reopen")) { sessionStorage.removeItem("hhbot_reopen"); setTimeout(show, 600); }
    var pf = sessionStorage.getItem("hhbot_form");   // a booking form that was asked for just before a full reload
    if (pf) {
      sessionStorage.removeItem("hhbot_form");
      setTimeout(function () { requestSiteForm(JSON.parse(pf)); }, 2500);
    }
  } catch (e) {}

  /* name + teaser copy come from the persona registry, so the widget stays brand-correct */
  fetch(BASE + "/personas").then(function (r) { return r.json(); }).then(function (d) {
    var p = (d.personas || []).filter(function (x) { return x.id === (PERSONA || d.default); })[0];
    if (!p) return;
    wrap.querySelector("#lname").textContent = "Chat with " + p.name;
    wrap.querySelector("#barname").textContent = p.name + " · " + p.company;
    // one short headline (like a person waving), one quiet sub-line; data-teaser="..." on the tag overrides the headline
    var custom = tag && tag.getAttribute("data-teaser");
    wrap.querySelector("#tname").textContent = custom || ("Talk to " + p.name + "!");
    if (tag && tag.getAttribute("data-teaser-sub") === "on") teaser.classList.add("two");
    var m = ctx.page && ctx.page.model;
    var service = p.business_type === "service";
    wrap.querySelector("#ttext").textContent = service
      ? (m ? m + " ki service? Free pickup." : "Service, repair, quote · free pickup")
      : (m ? m + " ke offers aur test drive" : "Price, offers, test drive");
  }).catch(function () {});

  setTimeout(function () {
    if (open) return;
    try { var sn = JSON.parse(sessionStorage.getItem("hhbot_seen") || "{}"); if (sn.open || sn.dismissed) return; } catch (e) {}
    teaser.classList.add("show");
  }, 2500);   // was 7 s: the bubble greets like a person who has just noticed you

  // HumanBot is the neutral name for the same object; HansHyundaiBot stays for pages that already call it
  window.HumanBot = window.HansHyundaiBot = { open: show, close: hide, toggle: toggle, context: ctx };
})();
