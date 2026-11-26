# Put Neha on autocarrepair.in

For the website developer. Two parts: **(A)** the bot server must be online on its own HTTPS address, and
**(B)** one line goes into the website. Do A first; B without A shows nothing.

> Hindi note: Pehle server (A) live karein, phir website par ek line (B) lagayein. Sirf B karne se kuch nahi dikhega.

---

## A. The bot server (one time)

Neha is a small Python server (`human_bot/server/app.py`, Python 3.12, only `edge-tts` as an extra package).
It must run on a **public HTTPS address**, for example `https://neha.autocarrepair.in` or the address Railway
gives (`https://<something>.up.railway.app`). Below this address is called **BOT-HOST**.

Why HTTPS: the website is HTTPS, so the browser blocks a script from `http://…`, and Chrome blocks a public
site from calling `127.0.0.1` or any local address.

The ready-made deployment is in `deploy/railway/` (`bot.Dockerfile` + `README.md`, step by step). The container
starts with:

```
python human_bot/server/app.py --host 0.0.0.0 --port ${PORT:-8080} --persona auto-car-repair
```

It also needs the knowledge-base service (`deploy/railway/kb.Dockerfile`), reached through `KB_URL`.

### Environment variables

Only these names are read by the server. Set them in the hosting panel, never in the website code.

**Must set**

| Variable | Value | Why |
|---|---|---|
| `GROQ_API_KEY` | your Groq key | the main AI model and speech-to-text (Whisper) |
| `GEMINI_API_KEY` | your Gemini key | backup AI model when Groq is busy |
| `HUMAN_BOT_LLM_ORDER` | `groq,gemini` | which AI to try first (skips the unused OpenAI-compatible hop) |
| `KB_URL` | e.g. `http://kb.railway.internal:4100` | the knowledge-base service (prices, services, FAQs) |
| `HUMAN_BOT_ALLOWED_ORIGINS` | `https://autocarrepair.in` | **CORS + iframe allow-list**, see below |
| `HUMAN_BOT_BEHIND_PROXY` | `1` | Railway / a proxy terminates HTTPS; rate limits use the real visitor IP |
| `HUMAN_BOT_SITE_MIRROR` | `0` | the bot's `/` shows its demo page, not a copy of the website |
| `HUMAN_BOT_PUBLIC_URL` | `https://BOT-HOST` | used for the recording links written into Bitrix24 leads |
| `HUMAN_BOT_ADMIN_KEY` | a random 32+ character string | protects `/conversations` (saved chats) on the internet |

**Leads (CRM)**

| Variable | Value | Why |
|---|---|---|
| `BITRIX24_WEBHOOK_URL` | the Bitrix24 inbound webhook URL | every useful conversation becomes a Bitrix24 lead. Without it, conversations wait in a local queue and are sent once it is set. |
| `BITRIX24_LEAD_STATUS` | default `NEW` | status of the new Bitrix24 lead |
| `BITRIX24_ASSIGNED_BY` | a Bitrix24 user id (optional) | who gets the lead |
| `CRM_MIN_TURNS` | default `3` | a chat with no name/phone becomes a lead only after this many customer messages |
| `ACR_LEADS_MODE` | `dry` (default) or `live` | the server's **own** fallback POST to `https://autocarrepair.in/api/v1/leads`, used only when Neha could not fill the website's form. `dry` = only logged and queued for Bitrix24. Set `live` only after you confirm the field names. |
| `ACR_LEADS_KEY` | optional | if your `/api/v1/leads` wants a key from the bot server |
| `ACR_LEADS_SOURCE` | default `neha_bot` | the `source` value of those server-side leads |

**Voice, avatar, recording**

| Variable | Value | Why |
|---|---|---|
| `HUMAN_BOT_TTS_PROVIDER` | `edge` (default) | Neha's voice (Microsoft Edge voices, no key needed) |
| `HUMAN_BOT_VOICE_STYLE` | `hindi` (default) | Hindi voice for Hinglish replies |
| `HUMAN_BOT_AVATAR_PROVIDER` | `puppet` | the animated photo of Neha (no extra service) |
| `HUMAN_BOT_RECORD` | `1` (default) or `0` | `1` stores each conversation's transcript and audio on the server (`human_bot/data`). See the consent note. |
| `HUMAN_BOT_SPOKEN_DISCLOSURE` | `0` (default) or `1` | `1` makes Neha also **say** that the call is recorded |
| `DEEPGRAM_API_KEY` | optional | a second speech-to-text provider (tried before Groq when set) |

**Optional switches**

| Variable | Default | Why |
|---|---|---|
| `HUMAN_BOT_SITE_FORM` | `1` | `0`: Neha never fills the website's "Book a Service" form |
| `HUMAN_BOT_NAV` | `1` | `0`: Neha never opens pages of the site |
| `HUMAN_BOT_RATE_PER_MIN` | `240` | API calls per visitor IP per minute |
| `HUMAN_BOT_SESSIONS_PER_MIN` | `20` | new chats per visitor IP per minute |

Without a storage volume, recordings and the lead queue are lost on every redeploy: mount one at
`/app/human_bot/data` (see `deploy/railway/README.md`, step 6).

### Recording and consent (HUMAN_BOT_RECORD)

With `HUMAN_BOT_RECORD=1` the server keeps the text and the voice of every conversation. The card always shows:
*"AI service advisor. Yeh baat-cheet record hoti hai — say "human" any time for a call back."*
Before going live:

- add a line to the website's **Privacy Policy**: chats with the AI assistant (text and voice) are recorded and
  stored to handle the service request, how long they are kept, and whom to contact to delete them (India's DPDP
  Act 2023 expects a clear notice and purpose);
- if you want a spoken notice too, set `HUMAN_BOT_SPOKEN_DISCLOSURE=1`;
- if you do not want to store voice and text at all, set `HUMAN_BOT_RECORD=0` (leads still reach Bitrix24 only
  if recording is on: the lead queue lives in the same storage).

> Hindi note: Recording chalu ho to Privacy Policy mein likhna zaroori hai ki AI chat record hoti hai.

### CORS and allowed origins

`HUMAN_BOT_ALLOWED_ORIGINS` is a comma-separated list of the websites that may use the bot:

```
HUMAN_BOT_ALLOWED_ORIGINS=https://autocarrepair.in
```

- It sets the CORS header on the bot's API and `Content-Security-Policy: frame-ancestors` on the card, so only
  these sites can show Neha. Exact origins only: scheme + host, no path, no trailing slash.
- `https://www.autocarrepair.in` redirects to `https://autocarrepair.in`, so the one entry is enough. Add more
  entries only if Neha must also run on another domain (a staging site, for example
  `https://autocarrepair.in,https://staging.autocarrepair.in`).
- The default `*` (any site) is for development only; the server prints a warning when it runs that way.

### Check the server

Open `https://BOT-HOST/health` in a browser. You want `"ok": true`, `"persona": "Neha"`, and a `kb` block
without `last_error`. `https://BOT-HOST/widget.js` must download a JavaScript file.

---

## B. The website (one line)

### The exact script tag

```html
<script src="https://BOT-HOST/widget.js" data-persona="auto-car-repair" data-position="left" defer></script>
```

Replace `BOT-HOST` with the real address from part A, for example:

```html
<script src="https://neha.autocarrepair.in/widget.js" data-persona="auto-car-repair" data-position="left" defer></script>
```

`data-position="left"`: autocarrepair.in already has its WhatsApp button and the "Need help? Chat with us" pill
in the bottom-right corner. On the left, Neha's bubble sits in free space. With `right` it still works: the
bubble moves itself above those buttons, but the right corner gets crowded, most of all on phones.

### Where to paste it

- **All pages.** The site is a single-page React app, so paste it once in the main `index.html`, just before
  `</body>`. Every page then has Neha, and she stays on screen while the visitor moves between pages.
- If the site is managed through Google Tag Manager instead, a *Custom HTML* tag with the same line, firing on
  *All Pages*, also works. A direct paste is better: it loads earlier and keeps the `data-…` settings exactly.
- Do not paste it twice. A second copy is ignored, but it costs a download.

> Hindi note: `index.html` mein `</body>` se theek pehle yeh ek line paste karein. Har page par Neha aa jayegi.

### Do not change on the website without telling us

Neha opens and fills the site's own **"BOOK A SERVICE"** popup. She finds it by these texts and placeholders:
the button text "Book a Service", the fields "John Doe", "10-digit number", "john@example.com",
"HR 26 XX 0000", "e.g. 2022", "Describe the problem...", the selects "Select Brand", "Select Model",
"Select Service Type", the location option "Moti Nagar", the buttons "Next Step" and "Submit Inquiry", and the
"Request Received" screen. If you rename any of these, tell us first (they live in `SITE_FORM` in `widget.js`).
If the form cannot be filled, nothing breaks: Neha closes the popup and the bot server files the request
itself.

Leads that Neha submits through the form arrive in your normal `/api/v1/leads` flow, like a customer's, with the
note *"Via Neha (AI service advisor)"*.

### Security headers (only if you add them later)

Today the site sends no `Content-Security-Policy` (except `upgrade-insecure-requests`) and no
`Permissions-Policy`, so nothing else is needed. If you add them later:

- `Content-Security-Policy`: add `https://BOT-HOST` to `script-src`, `frame-src`, `img-src`, `media-src` and
  `connect-src`;
- `Permissions-Policy`: allow the microphone and autoplay for the bot, e.g.
  `microphone=(self "https://BOT-HOST"), autoplay=(self "https://BOT-HOST")`. Without it the customer can only
  type, and Neha may start silent.

---

## How to check it works (5 minutes)

On a computer (Chrome) and on a phone:

1. Open https://autocarrepair.in in a private window. Neha's round photo appears in the bottom corner, and after
   2–3 seconds a small "Talk to Neha!" note above it. It must not cover the WhatsApp button, the cookie banner or the
   "BOOK A SERVICE" button.
2. On the phone, tap **Reject** / **Accept all** on the cookie banner and the "CHECK PRICES FOR FREE" bar: they
   must react (Neha must not block them).
3. Tap Neha. The card opens and she greets you **with voice** ("Namaste! Main Neha, Auto Car Repair se…").
   Allow the microphone when the browser asks.
4. Tap the keyboard icon and type `battery ki servicing karani hai`. The site moves to the Car Battery page and
   the card stays (on a phone it shrinks to her photo; tap it to bring it back).
5. Type `book kar do`, then a car and number, e.g. `Swift DL3CAB1234`: the site's booking popup opens.
6. **Use a test lead only.** Give a mobile and a name, listen to the read-back, say `haan`. The form is filled and
   submitted; the site shows "Request Received". Delete that test lead from the lead list afterwards.
7. Close the card with ✕: the bubble comes back.

Browser console (F12): there must be no red errors from `widget.js` or `BOT-HOST`. (The site's own
`/api/v1/user/profile 401` for a logged-out visitor is normal and not from Neha.)

---

## How to turn Neha off quickly

Fastest first:

1. **Stop or pause the bot service** in the hosting panel. `widget.js` then cannot load and the bubble does not
   appear on any page; the website keeps working normally. (The server sends `widget.js` with
   `Cache-Control: no-cache`, so browsers check it on every page load.)
2. **Remove the script line** from `index.html` and redeploy the website.
3. Partial switches, without removing her:
   - stop form filling: `data-site-form="off"` on the tag, or `HUMAN_BOT_SITE_FORM=0` on the server;
   - stop page navigation: `data-navigate="off"`, or `HUMAN_BOT_NAV=0`;
   - stop storing chats: `HUMAN_BOT_RECORD=0`.

> Hindi note: Turant band karna ho to hosting panel mein bot service ko stop kar dein — website par Neha dikhna
> band ho jayegi, baaki site normal chalegi.

---

## Known limits

- **Needs the bot server online.** If it is down, the bubble does not show (or the card cannot connect if it
  goes down during a chat). Free hosting plans may sleep or hit limits; use a paid plan for the live site.
- **AI providers can be busy.** On the free Groq tier a burst of visitors can get `429 Too Many Requests`; the
  server switches to Gemini for a minute. A paid Groq tier or a second key avoids it.
- **Voice input** needs the customer to allow the microphone. Without it (or on old browsers) they type and Neha
  still speaks. Some phones block sound until the first tap; the tap on the bubble counts, so the greeting plays.
- **The booking form is matched by its texts** (see "Do not change"). A redesign of the form needs a matching
  update in `widget.js`.
- **The popup is filled at the end.** When it opens on "book kar do" its fields are still empty; Neha fills
  name, mobile, car and service after the customer confirms the read-back with "haan".
- **Navigation** works for pages of autocarrepair.in only, and only the services Neha knows. The page changes in
  place, so the chat goes on. On a very slow connection (the new page takes more than 8 seconds to appear) the
  page is loaded normally instead; the card re-opens by itself but the conversation starts again.
- **Phones:** while the on-screen keyboard is up, the card moves above it. On a very small screen the card covers
  most of the page; scrolling the site shrinks it to her photo.
- **Recording storage** is lost on redeploy unless a volume is mounted (part A).
- Neha is an AI: she says so in the card, and hands over to a person when the customer asks for "human".
