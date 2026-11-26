# Embedding the bot (widget.js)

One script tag. The widget is a floating video bubble; the card is our `/embed` page inside an
iframe, so the host site's CSS can never break it and the bot's CSS can never leak out
(shadow DOM + iframe).

For autocarrepair.in, use the step-by-step guide for the web developer instead:
**[INSTALL_ON_WEBSITE.md](INSTALL_ON_WEBSITE.md)**. This file is the reference.

```html
<!-- before </body>, on every page -->
<script src="https://BOT-HOST/widget.js"
        data-persona="auto-car-repair"
        data-position="left" defer></script>
```

Replace `BOT-HOST` with the public HTTPS origin of the bot server (see **Hosting** below).
`http://127.0.0.1:4001` only works for local development: Chrome blocks a public HTTPS page
from loading a private-network script (mixed content + Local Network Access).

## Script tag options

| Attribute | Default | What it does |
|---|---|---|
| `data-persona` | server default | Persona id from `/personas` (`auto-car-repair` = Neha). Name and company come from the persona JSON. |
| `data-position` | `right` | `left` or `right` bottom corner. |
| `data-launcher` | `bubble` | `bubble` (round living portrait) or `pill` (photo + text capsule). |
| `data-teaser` | `Talk to <name>!` | Text of the small nudge above the bubble. |
| `data-teaser-sub` | off | `on` adds a second, quieter line to the teaser. |
| `data-label` | `Chat with <name>` | Accessible label / hover tag of the bubble. |
| `data-accent` | `#0b2f6b` | Accent colour passed to the card (hex or a plain colour name). |
| `data-ring` | `#2f8bff` | Colour of the bubble's ring. |
| `data-bubble-video` | on | `off` keeps the still photo in the bubble (no looping clip). |
| `data-clips` | `neha` | Reaction-clip set the card plays. |
| `data-navigate` | on | `off`: Neha never opens pages of the site. |
| `data-site-form` | on | `off`: Neha never opens or fills the site's own "Book a Service" form (the server files the request instead). |
| `data-avoid` | on | `off`: do not move away from the site's own corner widgets (see below). |
| `data-offset-bottom` | `0` | Extra px between the bottom of the screen and the widget (0 to 400). |

`window.__HHBOT_CONFIG = { base, persona }` set before the tag overrides the script origin and the persona.

## What it does

| Thing | Behaviour |
|---|---|
| Page context | Sends the page URL and title (and a model name if the URL or title names one) as `ctx` so Neha knows where the visitor is. |
| Teaser | 2.5 s after load an unopened widget shows a one-line nudge; closing it keeps it closed for the tab. |
| Open | A click on the bubble or the teaser opens the card. That click is the user gesture: Neha greets with audio at once. |
| Corner avoidance | Once a second the widget hit-tests its corner. It stands just above anything the site has fixed there (WhatsApp/chat button, cookie banner, sticky "Check prices" bar). The open card only steps over full-width bars, so a cookie banner stays clickable. While a site popup is open on a phone, Neha's photo stays in the corner. The banner going away after "Accept" brings the widget back down. |
| Tap-through | The widget's container is see-through to taps; only the bubble, teaser and card take clicks. |
| Phones (< 520 px) | The card is full screen (minus the lifted area). Scrolling the site, or Neha opening a page or the booking form, shrinks it to her photo; a tap on the photo brings it back. |
| Phone keyboard | While the on-screen keyboard is up (visual viewport smaller than the layout viewport), the card is pinned to the visible area above it, so the text box stays in view. |
| Mini | The minimize icon in the card header shrinks the card to a passport-size photo while the call keeps running hands-free. The card sends `hhbot:size` (`mini`/`normal`). |
| Navigation | "battery ki servicing" → Neha sends `hhbot:navigate` with a same-origin path. The widget does `history.pushState` + `popstate` (React Router changes the page in place, the call keeps going). If nothing changed after 8 s, or the router shows its 404, the page is loaded normally and the card re-opens itself (that starts a new conversation: the card keeps no session across a page load). |
| Site booking form | On "book karna hai" the card sends `hhbot:siteform_open`: the widget opens the site's own "Book a Service" popup. After the read-back and "haan" it sends `hhbot:siteform`: the widget fills the 3 steps (name, phone, location → brand, model, registration, year → service, description) and presses **Submit Inquiry**, then answers `hhbot:siteform_result`. Any step that fails closes the popup again and the server files the request itself. The steps are found by placeholders and button texts (`SITE_FORM` in widget.js): if the site changes its form, update them. |
| Keyboard | `Esc` closes the card. |
| Isolation | Shadow DOM for the launcher, iframe for the app, `z-index: 2147483000`. postMessage is pinned to the bot's origin both ways. |

## JavaScript API

```js
HumanBot.open();     // open the card   (HansHyundaiBot is the same object, kept for older pages)
HumanBot.close();    // close it
HumanBot.toggle();
HumanBot.context;    // what was sent to the bot ({page:{url,title,model,source}})
```

Use it from your own buttons:

```html
<button onclick="HumanBot.open()">Neha se baat karein</button>
```

## Data sent to the bot

```json
{"page": {"url": "https://autocarrepair.in/category/car-battery",
          "title": "(the page's <title>)",
          "model": "",
          "source": "website_widget",
          "site_form": true}}
```

The widget reads the page's own booking form only to fill it in when the customer says "haan"; it never
reads cookies, other form fields or the site's storage.

## Hosting

The bot server sends `Content-Security-Policy: frame-ancestors …` (from `HUMAN_BOT_ALLOWED_ORIGINS`, `*` by
default) and `Cross-Origin-Resource-Policy: cross-origin`, and reflects allowed origins in CORS. Set
`HUMAN_BOT_ALLOWED_ORIGINS=https://autocarrepair.in` in production. It needs a public **HTTPS** origin, because:

- `https://autocarrepair.in` + `http://.../widget.js` = mixed content, blocked by the browser;
- public site → `127.0.0.1` = blocked by Chrome's Local Network Access checks.

Ways to get HTTPS: Railway (`deploy/railway/README.md`, the current plan), a reverse proxy (Caddy/nginx) in
front of the server, or a Cloudflare Tunnel for a short demo. The app is stdlib Python + `edge-tts`:
`python human_bot/server/app.py --host 0.0.0.0 --port $PORT --persona auto-car-repair`.

If the site ever sets a `Content-Security-Policy`, allow the bot origin in `script-src`, `frame-src`,
`img-src`, `media-src` and `connect-src`. If it sets a `Permissions-Policy`, allow
`microphone=(self "https://BOT-HOST")` and `autoplay=(self "https://BOT-HOST")`.
