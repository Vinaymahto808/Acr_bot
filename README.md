# Auto Car Repair: Neha AI video service advisor (standalone project)

The AI video chatbot for **autocarrepair.in**, split out of the multi-brand VIDEO project on 2026-11-13.
This folder holds only Auto Car Repair (ACR): the Galaxy Toyota and Hans Hyundai personas, websites and
training sets are not part of it. It is its own git repository (the parent VIDEO repo ignores `acr/`).

The folder layout is the same as the original project, so the code runs unchanged:

| Folder | What |
|---|---|
| `human_bot/` | Python bot server (`server/app.py`): session, chat (SSE), TTS, STT, Simli/puppet avatar, CRM outbox. Persona: `persona/persona_auto_car_repair.json` only |
| `human_bot/data/` | Conversation records (`conversations.sqlite`: 141 conversations, 79 queued leads), call audio, turn-model data. **Private, git-ignored** |
| `human_bot/turn/` | End-of-turn model (weights + trainer) |
| `human_bot/train/` | `build_acr_sft.py`, `eval_acr.py`, `lora_sft.py` |
| `kb/` | Node knowledge base service: crawler, chunking, local e5 + FTS5 hybrid search. Website registry: `auto_car_repair` only |
| `kb/data/` | Live `kb.sqlite` (676 pages, 1,359 chunks, all vectors), pre-dynamic-chunking backup, local model. **Git-ignored** |
| `kb/seed/` | `kb.sqlite` crawl snapshot (committed, seeds a diskless host) + call Q&A CSVs (git-ignored) |
| `avatar_widget/` | Embeddable voice widget (loader + panel) |
| `tran_moment/` | Neha's face: portrait, idle video, reaction/talk clips, puppet layers, motion tracks, the scripts that make them |
| `voice_clone/` | `.env` with the API keys (**git-ignored**), voice consent, voice samples |
| `training/acr_sft/`, `ft_data.jsonl`, `fine_tune_neha.ipynb` | ACR fine-tuning data (git-ignored) and notebook |
| `deploy/`, `render.yaml`, `vercel.json` | Hosting: Render services `acr-neha-bot` + `acr-neha-kb`, Caddy/systemd files for a VPS |

## Run locally

```
cd kb && node src/server.js                                              # :4100
python human_bot/server/app.py --port 4001 --persona auto-car-repair     # :4001, open http://127.0.0.1:4001
```

The original VIDEO project uses the same default ports: stop it first, or pick others
(`KB_PORT=4600 node src/server.js` and `KB_URL=http://127.0.0.1:4600 python human_bot/server/app.py --port 4501`).

## Tests (all green on 2026-11-13)

```
cd human_bot && python -m unittest discover -s tests      # 56
cd kb && npm test                                          # 22
node --test avatar_widget/test/unit/*.test.mjs             # 76
```

## What changed against the original code

- `kb/src/config.js`: only the `auto_car_repair` website; `galaxy_toyota` / `hans_hyundai` rows removed from both `kb.sqlite` files (they held no pages).
- Tests: the kb isolation test uses a neutral second site; the prompt tests expect exactly the ACR persona.
- `human_bot/turn/synth.py`: synthetic-turn topics and prompt are workshop-only.
- `human_bot/llm/client.py`, `persona/prompting.py`, `server/app.py`: default names point at ACR.
- `render.yaml`: services renamed `acr-neha-bot` / `acr-neha-kb`, so the new project deploys next to the old one.
- Left out: other brands' personas and SFT data, `research/`, vendor recon (`avatar_widget/evidence`),
  face experiments `tran_moment/path_a`, `path_c`, `brands`, and the voice-cloning notebooks.

## Deploying

1. Create a **private** GitHub repository for this folder and push it (customer data stays out: check `git status` first).
2. Render: New, Blueprint, this repository. Fill the keys marked `sync: false`.
3. The original `neha-bot` / `neha-kb` services keep running from the old repo until you switch the site over.

## Not included

- `tools/turn-venv` (the PyTorch venv for retraining the turn model): create one with `python -m venv tools/turn-venv` and `pip install torch numpy`.
- Shared Neha media: the face and voice are also used by the other brands; this is a copy.

## Website booking form auto-fill (2026-11-13)
On a page with ACR's own "Book a Service" form, `widget.js` reports `site_form` and Neha fills that form herself once
the customer has confirmed the details (spoken "haan" or Submit in her small form): 3 steps, then "Submit Inquiry".
The site sends its usual lead; the bot's server does **not** post a second copy (`crm/site_form.py`). If the site
form cannot be completed (layout changed), the card reports it on `POST /lead/site` and the server files the lead
itself. Switches: `HUMAN_BOT_SITE_FORM=0` (server) or `data-site-form="off"` (script tag). Field selectors:
`SITE_FORM` in `human_bot/web/widget.js`. Phones: scrolling the site shrinks the card to Neha's photo; she keeps talking.
