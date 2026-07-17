import { useMemo, useRef, useEffect } from 'react'
import { motion, useScroll, useSpring, useTransform, useInView, useReducedMotion } from 'framer-motion'
import { forward, tokenize, VOCAB, emb, hash, norm, catOf, CATCOLOR, paramCount } from './model.js'
import { NumberTicker } from './ui.jsx'

/* Entrance choreography plays once per JS lifetime; scrolling back up later
   should not replay the reveal. */
let heroPlayed = false

const HERO_SENTENCE = 'the cat sat on the mat'

function Constellation() {
  const art = useMemo(() => {
    // vocab constellation — deterministic polar layout from word hashes
    const dots = VOCAB.slice(0, 44).map(w => {
      const a = ((hash(w) % 3600) / 10) * (Math.PI / 180)
      const r = 150 + ((hash(w) >>> 8) % 170)
      return {
        w, x: 820 + Math.cos(a) * r, y: 300 + Math.sin(a) * r * 0.72,
        r: 1.5 + norm(emb(w)) * 0.5, c: CATCOLOR[catOf(w)],
      }
    }).filter(d => d.x > 380 && d.x < 1180 && d.y > 30 && d.y < 560)
    // live attention arcs from a real forward pass
    const toks = tokenize(HERO_SENTENCE)
    const F = forward(toks, [[false, false], [false, false]])
    const xs = toks.map((_, i) => 180 + i * 150)
    const yBase = 660
    const arcs = []
    for (const h of [0, 1]) {
      const A = F.layers[0].heads[h].A
      for (let i = 0; i < F.n; i++) for (let j = 0; j < i; j++) {
        if (A[i][j] < 0.12) continue
        const x1 = xs[i], x2 = xs[j], apex = yBase - 40 - Math.abs(i - j) * 46
        arcs.push({
          d: `M${x1} ${yBase} Q ${(x1 + x2) / 2} ${apex}, ${x2} ${yBase}`,
          w: 1 + 3 * A[i][j], o: 0.22 + 0.5 * A[i][j],
          c: h === 0 ? 'var(--amber)' : 'var(--cyan)',
        })
      }
    }
    return { dots, toks, xs, yBase, arcs }
  }, [])

  const reduced = useReducedMotion()
  return (
    <svg className="heroArt" viewBox="0 0 1200 800" preserveAspectRatio="xMidYMid slice">
      <g className="heroDrift">
        {art.dots.map((d, i) => (
          <circle key={i} className="heroDot" cx={d.x} cy={d.y} r={d.r} fill={d.c}
            style={{ animationDelay: `${0.6 + i * 0.04}s` }} />
        ))}
      </g>
      <g className="heroBreathe">
        {art.arcs.map((a, i) => (
          <motion.path key={i} d={a.d} fill="none" stroke={a.c} strokeWidth={a.w}
            strokeLinecap="round" opacity={a.o}
            initial={heroPlayed || reduced ? false : { pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ delay: 0.8 + i * 0.06, duration: 0.9, ease: 'easeOut' }} />
        ))}
        {art.toks.map((t, i) => (
          <g key={i}>
            <circle cx={art.xs[i]} cy={art.yBase} r="3.5" fill="var(--paper)" opacity=".8" />
            <text x={art.xs[i]} y={art.yBase + 26} textAnchor="middle" className="heroTokLbl">{t}</text>
          </g>
        ))}
      </g>
    </svg>
  )
}

export default function Hero({ onEnter, onPlayTour }) {
  const wrapRef = useRef(null)
  const inView = useInView(wrapRef)
  const reduced = useReducedMotion()
  const { scrollYProgress } = useScroll({ target: wrapRef, offset: ['start start', 'end end'] })
  const titleScaleRaw = useTransform(scrollYProgress, [0, 0.6], [1, 0.94])
  const titleOpacity = useTransform(scrollYProgress, [0, 0.55], [1, 0])
  const titleScale = useSpring(titleScaleRaw, { stiffness: 120, damping: 26 })
  const artY = useTransform(scrollYProgress, [0, 0.8], [0, -80])
  const artOpacity = useTransform(scrollYProgress, [0, 0.8], [1, 0.15])
  const cueOpacity = useTransform(scrollYProgress, [0, 0.08], [1, 0])

  const playedRef = useRef(heroPlayed)   // captured once per mount — stable across re-renders
  useEffect(() => { heroPlayed = true }, [])
  const played = playedRef.current
  const seq = (delay, extra = {}) => played || reduced
    ? {}
    : { initial: { opacity: 0, y: 18, ...extra }, animate: { opacity: 1, y: 0 }, transition: { delay, duration: 0.7, ease: [0.16, 1, 0.3, 1] } }

  return (
    <div className={'heroWrap' + (inView ? '' : ' offstage')} ref={wrapRef}>
      <div className="heroSticky">
        <motion.div className="heroArtWrap" style={{ y: artY, opacity: artOpacity }}>
          <Constellation />
        </motion.div>
        <motion.div className="heroTitleBlock" style={{ scale: titleScale, opacity: titleOpacity }}>
          <motion.div className="kicker" {...seq(0.1)}>
            <span className="dot" /> A Transformer With Glass Walls · Live Model
          </motion.div>
          <h1 className="heroTitle">
            <span className="hmask"><motion.span className="hline"
              initial={played || reduced ? false : { y: '108%' }} animate={{ y: 0 }}
              transition={{ delay: 0.3, duration: 0.9, ease: [0.16, 1, 0.3, 1] }}>The Glass</motion.span></span>
            <span className="hmask"><motion.span className="hline glow"
              initial={played || reduced ? false : { y: '108%' }} animate={{ y: 0 }}
              transition={{ delay: 0.42, duration: 0.9, ease: [0.16, 1, 0.3, 1] }}><em>Transformer</em></motion.span></span>
          </h1>
          <motion.p className="heroSub" {...seq(1.1)}>
            <span className="live">● RUNNING LIVE:</span> a real GPT executing in your browser. Every number on this
            page is genuinely computed — every matrix multiply, layernorm, softmax and GELU. Nothing is faked.
          </motion.p>
          <motion.div className="heroStats" {...seq(1.4)}>
            2 LAYERS · 2 HEADS · D=16 · <NumberTicker value={paramCount()} className="heroStatNum" /> PARAMS · COMPUTED IN YOUR BROWSER
          </motion.div>
          <motion.div className="heroCtas" {...seq(1.7)}>
            <motion.button type="button" className="gbtn primary" whileTap={{ scale: 0.95 }} onClick={onPlayTour}>
              ▶ Play the machine
            </motion.button>
            <motion.button type="button" className="gbtn ghost" whileTap={{ scale: 0.95 }} onClick={onEnter}>
              Explore on my own ↓
            </motion.button>
          </motion.div>
        </motion.div>
        <motion.div className="scrollCue" style={{ opacity: cueOpacity }}>
          <span>scroll</span><span className="chev">⌄</span>
        </motion.div>
      </div>
    </div>
  )
}
