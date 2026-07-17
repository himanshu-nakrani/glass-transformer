/* The 14-beat cinematic tour. `enter`/`exit` receive the controller built in
   App (all state setters + sampleStep + after(ms, fn) for tracked timeouts). */

export const TOUR_SENTENCE = 'the cat sat on the'

export const BEATS = [
  {
    id: 'welcome', stage: 0, anchor: 'archmap', dur: 9000,
    title: 'This machine is real',
    body: 'You are looking at an actual GPT — 2 layers, 2 attention heads, 16 dimensions — running live in your browser. Those pulses are your sentence riding the residual stream from bottom to top. In the next two minutes we\'ll follow one sentence all the way through.',
    enter: ctl => { ctl.setSelToken(null); ctl.setProbe(null); ctl.setLayer(0); ctl.setHead(0) },
  },
  {
    id: 'tokenize', stage: 1, anchor: 'tokens', dur: 8000,
    title: 'Step 1 — words become numbers',
    body: 'A model can\'t read. So each word is swapped for an ID from a vocabulary — "cat" is just a number to the machine. The colors show grammatical categories. Remember them: one attention head will secretly use exactly this structure.',
    enter: ctl => ctl.setSelToken(1),
  },
  {
    id: 'embed', stage: 2, anchor: 'emb-rows', dur: 8000,
    title: 'Step 2 — every ID becomes a vector',
    body: 'Each ID looks up a row of 16 numbers — the word\'s coordinates in meaning-space. Teal is positive, rose is negative. Look at "cat": dims 8–15 carry its noun-ness. These numbers are the only thing the model will ever know about the word.',
    enter: ctl => ctl.setSelToken(1),
  },
  {
    id: 'position', stage: 2, anchor: 'pos-rows', dur: 7000,
    title: '…plus a fingerprint for position',
    body: 'Attention has no built-in sense of order — "dog bites man" would equal "man bites dog". So a unique sin/cos wave pattern for each slot is added on top. Word + place, fused into one vector. This enters the machine.',
  },
  {
    id: 'attn-pos', stage: 3, anchor: 'heatmap', dur: 9000,
    title: 'Step 3 — attention: tokens look at each other',
    body: 'Each row is a token deciding where to look. This head has a built-in bias for the previous token — see the bright stripe just below the diagonal? That\'s the same mechanism T5 and ALiBi use in real models. The dark corner is the causal mask: no peeking at the future.',
    enter: ctl => { ctl.setLayer(0); ctl.setHead(0); ctl.setSelToken(null); ctl.setProbe(null) },
  },
  {
    id: 'attn-content', stage: 3, anchor: 'heatmap', dur: 8000,
    title: 'The other head hunts for meaning',
    body: 'Same sentence, different head, different pattern. This one lights up when two tokens share a grammatical category — watch the second "the" find the first "the". No position information at all. Two heads, two strategies, running in parallel.',
    enter: ctl => ctl.setHead(1),
  },
  {
    id: 'microscope', stage: 3, anchor: 'scope-dot', dur: 9000,
    title: 'No magic — just multiply and add',
    body: 'Where does an attention weight come from? We opened one cell: "the" asking about "the". Eight little multiplications between a query and a key, summed, scaled, softmaxed. Every cell in that heatmap is exactly this. That\'s the whole trick.',
    enter: ctl => ctl.setProbe({ i: 4, j: 0 }),
  },
  {
    id: 'ablate', stage: 3, anchor: 'ablate', dur: 8000,
    title: 'Prove it: mute a head',
    body: 'How do we know a head matters? Silence it. We just zeroed the content head\'s output — the model now runs without it, and the final predictions will shift. This is ablation: the standard scalpel of interpretability research. (We\'ll un-mute it before moving on.)',
    enter: ctl => ctl.setAblate([[false, true], [false, false]]),
    exit: ctl => ctl.setAblate([[false, false], [false, false]]),
  },
  {
    id: 'mlp', stage: 4, anchor: 'neurons', dur: 8000,
    title: 'Step 4 — the MLP thinks in private',
    body: 'Attention moved information between tokens; now each token is processed alone. Its 16 dims fan out to 64 neurons, pass a soft gate called GELU, and squeeze back down. In trained models these neurons become detectors — for quotes, years, languages. This is where knowledge lives.',
    enter: ctl => ctl.setSelToken(4),
  },
  {
    id: 'stream', stage: 5, anchor: 'stream', dur: 8000,
    title: 'Step 5 — the residual stream',
    body: 'The single best mental model for transformers: one vector per token, flowing upward, and every block only adds to it. Nothing is overwritten. Watch "the"\'s vector evolve checkpoint by checkpoint — Δ shows how loudly each block wrote into the stream.',
    enter: ctl => ctl.setSelToken(4),
  },
  {
    id: 'lens', stage: 5, anchor: 'lens', dur: 9000,
    title: 'Watch the model change its mind',
    body: 'The logit lens: decode the stream early, at every depth, and see what the model would predict if it stopped there. Early rows are noise; each layer nudges the guess. Researchers use this exact trick on GPT-scale models to watch facts surface at specific layers.',
  },
  {
    id: 'temp', stage: 6, anchor: 'sliders', dur: 8000,
    title: 'Step 6 — from scores to a choice',
    body: 'The last token\'s vector is compared against every word in the vocabulary — one dot product each — giving logits. Temperature reshapes them: low = confident and repetitive, high = chaotic. We just cooled it to 0.5. Top-k throws away the long tail.',
    enter: ctl => { ctl.setGenTemp(0.5); ctl.setTopK(5); ctl.setPriorOn(true) },
  },
  {
    id: 'sample', stage: 6, anchor: 'predrows', dur: 9000,
    title: 'The dice roll',
    body: '"the cat sat on the" — the bars are the model\'s honest odds for what comes next. Now we sample: one weighted dice roll… watch the winner fly into the sentence. This loop — predict, sample, append, repeat — is all "AI writing" has ever been.',
    enter: ctl => ctl.after(2200, () => ctl.sampleStep()),
  },
  {
    id: 'outro', stage: 0, anchor: 'archmap', dur: 12000,
    title: 'Now it\'s yours',
    body: 'That\'s the entire machine — no step was skipped, no number was faked. Type your own sentence, click any cell to see its math, mute heads, break things. Everything you just watched is still running, live, under glass.',
    enter: ctl => { ctl.setSelToken(null); ctl.setProbe(null) },
  },
]
