import { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { forward, tokenize, distribution, paramCount, VOCAB } from './model.js'
import { STAGES, FlowMini, NumberTicker } from './ui.jsx'
import { MapStage, TokenizeStage, EmbedStage, AttentionStage, MlpStage, StreamStage, PredictStage } from './stages.jsx'
import Hero from './hero.jsx'
import { createFxBus, FxLayer } from './fx.jsx'
import { useTour, TourOverlay } from './tour.jsx'
import { BEATS, TOUR_SENTENCE } from './tourScript.jsx'

const EXAMPLES = [
  'the cat sat on the mat',
  'she opened the door because it was cold',
  'robots dream of electric sheep',
  'attention is all you need',
]

const STAGE_META = {
  map: {
    title: 'The Machine, End to End',
    plain: "You're looking at: the whole machine — your sentence enters at the top box and exits as a prediction at the bottom.",
    meta: <>This is the whole architecture — and it's <b>live</b>: the pulses trace your sentence's actual path. Every box is clickable and opens the stage where that computation is dissected.</>,
  },
  tokenize: {
    title: 'Tokenize',
    plain: "You're looking at: your sentence, chopped into pieces the model can count.",
    meta: <>Text → integer IDs from a real <b>vocabulary</b> that grows as you type. Colors show grammatical category — watch how the model's <b>content head</b> uses exactly this structure later. Click to spotlight, hover × to delete.</>,
  },
  embed: {
    title: 'Embed + Position',
    plain: "You're looking at: each word as 16 numbers — the form it travels in from here on.",
    meta: <>Each ID looks up a real <b>16-dimensional vector</b>. Hover any cell for its exact value. Dims <b>8–15</b> carry a category signature — dims <b>0–7</b> are the position band. Then a sinusoidal position code is <b>added</b>.</>,
  },
  attention: {
    title: 'Attention, Under the Microscope',
    plain: "You're looking at: who is looking at whom. Each row is one token distributing 100% of its attention.",
    meta: <>Real attention weights from the live forward pass. <b>Click any cell</b> to expand the actual q·k dot product that produced it — element products, summed, scaled, biased, softmaxed. Try <b>muting a head</b> and check Predict.</>,
  },
  mlp: {
    title: 'The MLP — Thinking Alone',
    plain: "You're looking at: one token's 64 neurons — the model's private scratchpad between attention steps.",
    meta: <>After attention gathers context, each token is processed <b>independently</b>: 16 dims expand to <b>64 neurons</b>, pass through GELU, and compress back. Pick a token; every circle is a real activation.</>,
  },
  stream: {
    title: 'The Residual Stream + Logit Lens',
    plain: "You're looking at: one token's vector at every checkpoint, and what the model would predict at each depth.",
    meta: <>The single most clarifying view in interpretability. Left: one token's vector at every checkpoint — each block only <b>adds</b> to it. Right: the <b>logit lens</b> — decode the stream early at every depth and watch the prediction <em>form</em>.</>,
  },
  predict: {
    title: 'Unembed, Sample, Repeat',
    plain: "You're looking at: the model's actual bet on the next word — and the dice you roll to pick one.",
    meta: <>The last token's final vector is dotted against <b>every word's embedding</b> (tied unembedding) → real logits → temperature → top-k → softmax → sample. The loop that writes.</>,
  },
}

const appendTo = (cur, w) => {
  const c = cur.replace(/\s+$/, '')
  return /[.,!?]/.test(w) ? c + w : c + ' ' + w
}

export default function App() {
  const [stage, setStage] = useState(0)
  const [sentence, setSentence] = useState('the cat sat on the mat')
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)
  const [ablate, setAblate] = useState([[false, false], [false, false]])
  const [selToken, setSelToken] = useState(null)
  const [probe, setProbe] = useState(null)
  const [priorOn, setPriorOn] = useState(true)
  const [genTemp, setGenTemp] = useState(0.8)
  const [topK, setTopK] = useState(10)
  const [auto, setAuto] = useState(false)
  const [landing, setLanding] = useState(null)   // word currently in flight toward the ticker
  const [popN, setPopN] = useState(0)            // increments when a flight lands → pop keyframe
  const [hasSampled, setHasSampled] = useState(false)
  const [seen, setSeen] = useState(() => {
    try { return typeof window !== 'undefined' && localStorage.getItem('gt-seen') === '1' } catch { return false }
  })

  const instrumentRef = useRef(null)
  const fx = useRef(null)
  if (!fx.current) fx.current = createFxBus()

  const tokens = useMemo(() => tokenize(sentence), [sentence])
  const F = useMemo(() => forward(tokens, ablate), [tokens, ablate])

  const sel = selToken !== null && selToken < F.n ? selToken : null
  useEffect(() => { if (probe && (probe.i >= F.n || probe.j >= F.n)) setProbe(null) }, [F.n, probe])

  const spotlight = i => setSelToken(s => (s === i ? null : i))
  const toggleAblate = (L, h) =>
    setAblate(a => a.map((row, li) => row.map((v, hi) => (li === L && hi === h ? !v : v))))

  const appendWord = w => setSentence(s => appendTo(s, w))

  /* refs so sampling & tour always see fresh state without re-subscribing */
  const cfg = useRef({})
  cfg.current = { priorOn, genTemp, topK, ablate }
  const sentenceRef = useRef(sentence)
  sentenceRef.current = sentence

  const sampleStep = useCallback(() => {
    const s = sentenceRef.current
    const t = tokenize(s)
    if (t.length >= 12) return
    const { priorOn, genTemp, topK, ablate } = cfg.current
    const dist = distribution(forward(t, ablate), { prior: priorOn, temp: genTemp, topk: topK })
    const r = Math.random()
    let acc = 0, pick = dist[0].word
    for (const d of dist) { acc += d.p; if (r <= acc) { pick = d.word; break } }
    setLanding(pick)
    fx.current.emit({ type: 'fly', word: pick })
    setSentence(cur => appendTo(cur, pick))
    setHasSampled(true)
  }, [])
  const sampleStepRef = useRef(sampleStep)
  sampleStepRef.current = sampleStep

  const onLand = useCallback(() => { setLanding(null); setPopN(n => n + 1) }, [])

  useEffect(() => {
    if (!auto) return
    const id = setInterval(() => {
      if (tokenize(sentenceRef.current).length >= 12) { setAuto(false); return }
      sampleStepRef.current()
    }, 900)
    return () => clearInterval(id)
  }, [auto])

  /* ---------------- tour ---------------- */
  const snapRef = useRef({})
  snapRef.current = { sentence, stage, layer, head, ablate, selToken, probe, genTemp, topK, priorOn }

  const markSeen = useCallback(() => {
    setSeen(true)
    try { localStorage.setItem('gt-seen', '1') } catch { /* private mode */ }
  }, [])

  const ctl = useMemo(() => ({
    setStage, setSentence, setSelToken, setProbe, setLayer, setHead,
    setAblate, setGenTemp, setTopK, setPriorOn,
    sampleStep: () => sampleStepRef.current(),
    prepare: () => {
      setAuto(false); setSentence(TOUR_SENTENCE)
      setAblate([[false, false], [false, false]])
      setProbe(null); setSelToken(null); setPriorOn(true)
    },
    onTourDone: markSeen,
  }), [markSeen])

  const { tour, start: startTour, next, prev, exit: exitTour, togglePause } = useTour(ctl, BEATS, snapRef)
  const tourRef = useRef(null)
  tourRef.current = { tour, next, prev, exitTour, togglePause }

  /* first-visit hints auto-expire */
  useEffect(() => {
    if (seen) return
    const id = setTimeout(markSeen, 90000)
    return () => clearTimeout(id)
  }, [seen, markSeen])

  /* stage-transition token ghosts */
  const firstStage = useRef(true)
  useEffect(() => {
    if (firstStage.current) { firstStage.current = false; return }
    fx.current.emit({ type: 'ghost', tokens: tokenize(sentenceRef.current) })
  }, [stage])

  /* keyboard — tour gets priority */
  useEffect(() => {
    const onKey = e => {
      if (e.target.tagName === 'INPUT') return
      const t = tourRef.current
      if (t.tour) {
        if (e.key === ' ') { e.preventDefault(); t.togglePause() }
        if (e.key === 'ArrowRight') t.next()
        if (e.key === 'ArrowLeft') t.prev()
        if (e.key === 'Escape') t.exitTour()
        return
      }
      if (e.key === 'ArrowRight') setStage(s => Math.min(s + 1, STAGES.length - 1))
      if (e.key === 'ArrowLeft') setStage(s => Math.max(s - 1, 0))
      if (e.key === ' ') { e.preventDefault(); setStage(STAGES.length - 1); sampleStepRef.current() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const scrollToInstrument = useCallback(() => {
    instrumentRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])
  const go = i => { setStage(i); scrollToInstrument() }
  const playTour = () => { startTour(); scrollToInstrument() }

  const s = STAGES[stage]
  const meta = STAGE_META[s.id]
  const muted = ablate.flat().filter(Boolean).length
  const hints = !seen && !tour

  const stageProps = {
    F, layer, head, setLayer, setHead, ablate, toggleAblate,
    selToken: sel, spotlight, probe, setProbe, go, hints,
  }

  return (
    <>
      <div className="aurora"><div className="blob b1" /><div className="blob b2" /><div className="blob b3" /></div>
      <div className="grain" />
      <Hero onEnter={scrollToInstrument} onPlayTour={playTour} />
      <div className="wrap" id="instrument" ref={instrumentRef}>

        <header className="mast-slim">
          <span className="wordmark"><span className="dot" /> The Glass <em>Transformer</em></span>
          <span className="mast-stats">
            a real GPT under glass · <NumberTicker value={paramCount()} /> params, live
          </span>
          <span className="mast-actions">
            {!seen && !tour && <span className="tournudge">2-minute guided tour — watch it think</span>}
            <button type="button" className="tourbtn" onClick={playTour}>▶ Play the machine</button>
          </span>
        </header>

        <nav className="stepper">
          {STAGES.map((st, i) => (
            <button key={st.id} className={'step' + (i === stage ? ' active' : '')} onClick={() => setStage(i)}>
              {i === stage && <motion.span layoutId="stepGlow" className="stepGlow" transition={{ type: 'spring', stiffness: 300, damping: 30 }} />}
              <span className="num">STAGE {st.num}</span>
              <span className="lbl">{st.lbl}</span>
            </button>
          ))}
          <motion.span className="stepprog"
            animate={{ width: `${((stage + 1) / STAGES.length) * 100}%` }}
            transition={{ type: 'spring', stiffness: 120, damping: 24 }} />
        </nav>

        <div className="console">
          <label>Input&nbsp;Sequence</label>
          <input value={sentence} spellCheck="false" autoComplete="off" onChange={e => setSentence(e.target.value)} />
          <div className="chips">
            {EXAMPLES.map(ex => (
              <button key={ex} className="chip" onClick={() => { setSentence(ex); setSelToken(null); setProbe(null) }}>{ex}</button>
            ))}
          </div>
        </div>

        <div className="ticker" data-tour="ticker">
          <span className="tlabel">Live&nbsp;Sequence</span>
          {F.tokens.map((w, i) => {
            const isLast = i === F.n - 1
            return (
              <motion.span key={`${i}-${w}${isLast ? '-' + popN : ''}`} layout="position"
                className={'ttok' + (isLast && landing ? ' landing' : '') + (isLast && popN > 0 && !landing ? ' pop' : '')}
                style={{ color: i === sel ? 'var(--amber)' : 'var(--paper)' }}>
                {w}
              </motion.span>
            )
          })}
          <span className="caret" data-fx="caret" />
          <span className="stat">
            <b><NumberTicker value={F.n} /></b> tokens · <b><NumberTicker value={VOCAB.length} /></b> vocab · <b><NumberTicker value={paramCount()} /></b> params
            {muted > 0 && <span className="muted-note"> · {muted} head{muted > 1 ? 's' : ''} muted</span>}
          </span>
        </div>

        <section className="stage">
          <div className="stage-head">
            <div>
              <h2><span className="snum">{s.num}</span> · {meta.title}</h2>
              <div className="plainline">{meta.plain}</div>
            </div>
            <div className="meta">{meta.meta}</div>
            <div className="watermark">{s.num}</div>
          </div>
          <div className="stage-body" data-tour="stage-body">
            <AnimatePresence mode="wait">
              <motion.div key={s.id}
                initial={{ opacity: 0, y: 24, filter: 'blur(6px)' }}
                animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
                exit={{ opacity: 0, y: -18, filter: 'blur(6px)' }}
                transition={{ duration: 0.35, ease: [0.2, 0.9, 0.2, 1] }}>
                {s.id === 'map' && <MapStage {...stageProps} />}
                {s.id === 'tokenize' && (
                  <TokenizeStage {...stageProps}
                    killToken={i => { const arr = tokenize(sentence); arr.splice(i, 1); setSentence(arr.join(' ') || 'the'); setSelToken(null) }} />
                )}
                {s.id === 'embed' && <EmbedStage {...stageProps} />}
                {s.id === 'attention' && <AttentionStage {...stageProps} />}
                {s.id === 'mlp' && <MlpStage {...stageProps} />}
                {s.id === 'stream' && <StreamStage {...stageProps} />}
                {s.id === 'predict' && (
                  <PredictStage {...stageProps}
                    genTemp={genTemp} setGenTemp={setGenTemp} topK={topK} setTopK={setTopK}
                    priorOn={priorOn} setPriorOn={setPriorOn} auto={auto} setAuto={setAuto}
                    sampleStep={sampleStep} appendWord={appendWord} hasSampled={hasSampled}
                    undo={() => { const a = tokenize(sentence); a.pop(); setSentence(a.join(' ') || 'the') }}
                    reset={() => { setAuto(false); setSentence('the cat sat on the') }} />
                )}
                <FlowMini here={s.id} go={go} />
              </motion.div>
            </AnimatePresence>
          </div>
        </section>

        <div className="navbtns">
          <button className="navbtn" disabled={stage === 0} onClick={() => setStage(stage - 1)}>
            <span className="narrow">←</span> Previous stage
          </button>
          <button className="navbtn" disabled={stage === STAGES.length - 1} onClick={() => setStage(stage + 1)}>
            Next stage <span className="narrow">→</span>
          </button>
        </div>

        <footer>
          <span>
            THE GLASS TRANSFORMER · a real micro-GPT (2 layers · 2 heads · d=16 · ~7k params) forward-passing live in
            React.<br />It is <b>untrained</b> — three heads are hand-crafted with real mechanisms (a T5/ALiBi-style
            position head, a content-matching head, a pure ALiBi recency head); the rest is seeded random. A toggleable
            grammar prior keeps generation readable; switch it off to see the raw model.
          </span>
          <span>← → STAGES · SPACE = SAMPLE · CLICK CELLS TO DISSECT</span>
        </footer>
      </div>

      <FxLayer bus={fx.current} onLand={onLand} />
      <TourOverlay tour={tour} beats={BEATS} onNext={next} onPrev={prev} onExit={exitTour} onTogglePause={togglePause} />
    </>
  )
}
