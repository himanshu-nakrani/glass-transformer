# The Glass Transformer

**A real GPT with glass walls — every number on screen is genuinely computed, live, in your browser.**

This is an interactive explainer of how transformers work. Instead of diagrams *about* the architecture, it runs an actual micro-GPT — **2 layers, 2 attention heads, 16 dimensions, ~7,500 parameters** — implemented from scratch in JavaScript, and makes every intermediate tensor inspectable: click any attention cell and see the exact q·k dot product that produced it, element by element.

## Features

- **🗺 The Map** — the full architecture with animated pulses riding the residual stream; every block is a door
- **🔤 Tokenize → Embed** — real vocabulary IDs, real 16-dim embeddings (hover any cell for its exact value), sinusoidal position codes
- **🔍 Attention microscopes** — a live heatmap per layer/head, a **dot-product microscope** (all 8 q·k element products, summed → ÷√d → + relative bias → softmax), and a **softmax microscope** showing all four steps
- **🧠 Real head mechanisms** — a T5/ALiBi-style **position head**, a **content head** that routes by grammatical category, a pure ALiBi **recency head**, and one untrained head for contrast
- **🔇 Ablation** — mute any head and watch the output distribution shift
- **🌊 The Residual Stream + Logit Lens** — one token's vector at every checkpoint, plus what the model *would* predict at each depth — watch it change its mind layer by layer
- **🎲 Generation** — real logits → temperature → top-k → sample; sampled words fly into the sentence; auto-write loop
- **🎬 Cinematic guided tour** — a 14-beat narrated walkthrough that drives the app itself: spotlights panels, switches heads, demonstrates ablation, and ends by sampling a word live
- **📖 Intuition layer** — hover glossary for every term (softmax, logit, layernorm…), plain-English "what am I looking at" lines, self-dismissing hints

## Honesty

The model is **untrained** — three heads are hand-crafted with real published mechanisms (T5 relative attention bias / ALiBi), the rest is seeded random. A toggleable grammar prior keeps auto-write readable; switch it off to see exactly what an untrained transformer believes. Nothing else is faked: the forward pass (layernorm → QKV → causal softmax attention → residual → GELU MLP → residual → tied unembedding) runs in full on every keystroke.

## Run it

```bash
npm install
npm run dev        # hot-reload dev server
npm run build      # produces dist/index.html — a single self-contained file
```

The build output is **one double-clickable HTML file** (via `vite-plugin-singlefile`) — no server needed.

## Stack

React 18 · Vite 5 · Framer Motion 11 · no other runtime dependencies. The model core (`src/model.js`) is pure JavaScript with zero imports.

## Architecture

```
src/
├── model.js       # the micro-GPT: forward pass, attention, MLP, sampling — pure functions
├── App.jsx        # state, generation loop, keyboard, stage router
├── stages.jsx     # the 7 dissection stages
├── hero.jsx       # scroll-driven landing with a constellation derived from real model weights
├── tour.jsx       # guided-tour state machine + rAF-tracked spotlight overlay
├── tourScript.jsx # the 14 tour beats
├── fx.jsx         # flying sampled tokens, stage-transition ghosts
├── glossary.jsx   # hover glossary + affordance hints
└── ui.jsx         # shared primitives (NumberTicker, Switch, vector strips…)
```

## Keyboard

`←` `→` change stages · `Space` sample a token · during the tour: `Space` pause, `Esc` exit
