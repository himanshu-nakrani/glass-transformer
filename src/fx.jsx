import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

/* Tiny event bus — created once in App via useRef, no context needed. */
export function createFxBus() {
  const subs = new Set()
  return {
    emit(evt) { subs.forEach(f => f(evt)) },
    on(f) { subs.add(f); return () => subs.delete(f) },
  }
}

let flightId = 0
const reducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

/* Fixed full-viewport layer that renders in-flight words and stage-transition
   token ghosts. Mounts empty; all measurement happens inside event handlers
   (client-only by construction — SSR renders nothing). */
export function FxLayer({ bus, onLand }) {
  const [flights, setFlights] = useState([])
  const [ghosts, setGhosts] = useState([])

  useEffect(() => bus.on(evt => {
    if (evt.type === 'fly') {
      if (reducedMotion()) { onLand?.() ; return }
      let src = null
      try { src = document.querySelector(`[data-fx-word="${CSS.escape(evt.word)}"]`) } catch { /* older browsers */ }
      if (!src) src = document.querySelector('[data-fx="sample-btn"]')
      const dst = document.querySelector('[data-fx="caret"]')
      if (!src || !dst) { onLand?.(); return }
      const a = src.getBoundingClientRect(), b = dst.getBoundingClientRect()
      setFlights(f => [...f, {
        id: ++flightId, word: evt.word,
        from: { x: a.left, y: a.top },
        to: { x: b.left + 6, y: Math.max(8, b.top - 2) },
      }])
    }
    if (evt.type === 'ghost') {
      if (reducedMotion()) return
      const panel = document.querySelector('[data-tour="stage-body"]')
      if (!panel) return
      const r = panel.getBoundingClientRect()
      if (r.top > window.innerHeight || r.bottom < 0) return
      const words = evt.tokens.slice(0, 6)
      setGhosts(words.map((w, i) => ({
        id: ++flightId, word: w,
        x: r.left + r.width * (0.12 + 0.14 * i),
        y: Math.max(60, r.top + 6),
        delay: i * 0.06,
      })))
    }
  }), [bus, onLand])

  return (
    <div className="fxlayer" aria-hidden="true">
      {flights.map(f => (
        <motion.span key={f.id} className="fly-word"
          initial={{ x: f.from.x, y: f.from.y, scale: 1.2, opacity: 1 }}
          animate={{ x: f.to.x, y: f.to.y, scale: 0.9, opacity: 1 }}
          transition={{ type: 'spring', stiffness: 90, damping: 16, mass: 0.8 }}
          onAnimationComplete={() => { setFlights(fl => fl.filter(x => x.id !== f.id)); onLand?.() }}>
          {f.word}
        </motion.span>
      ))}
      <AnimatePresence>
        {ghosts.map(g => (
          <motion.span key={g.id} className="ghost-word" style={{ left: g.x, top: g.y }}
            initial={{ y: 0, opacity: 0 }}
            animate={{ y: 64, opacity: [0, 0.45, 0] }}
            transition={{ duration: 0.7, delay: g.delay, ease: 'easeIn' }}
            onAnimationComplete={() => setGhosts(gs => gs.filter(x => x.id !== g.id))}>
            {g.word}
          </motion.span>
        ))}
      </AnimatePresence>
    </div>
  )
}
