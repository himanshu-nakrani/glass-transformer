import { useEffect, useRef, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

/* rAF-poll until a selector exists — AnimatePresence mode="wait" means anchors
   appear ~400ms after a stage jump. */
function waitForEl(sel, timeout = 1800) {
  return new Promise(res => {
    const t0 = performance.now()
    const tick = () => {
      const el = document.querySelector(sel)
      if (el) return res(el)
      if (performance.now() - t0 > timeout) return res(null)
      requestAnimationFrame(tick)
    }
    tick()
  })
}

/* Tour state machine. `ctl` = stable controller of app setters; `snapRef`
   holds a fresh snapshot of restorable state (updated by App each render). */
export function useTour(ctl, beats, snapRef) {
  const [tour, setTour] = useState(null)   // null | { beat, playing }
  const advanceTimer = useRef(null)
  const fxTimer = useRef(null)
  const savedRef = useRef(null)
  const lastBeatRef = useRef(-1)

  const clearTimers = () => {
    clearTimeout(advanceTimer.current)
    clearTimeout(fxTimer.current)
  }

  // controller extension: tracked timeout for in-beat effects
  const ctlRef = useRef(null)
  if (!ctlRef.current) {
    ctlRef.current = { ...ctl, after: (ms, fn) => { clearTimeout(fxTimer.current); fxTimer.current = setTimeout(fn, ms) } }
  }

  const start = useCallback(() => {
    savedRef.current = { ...snapRef.current }
    lastBeatRef.current = -1
    ctl.prepare()
    setTour({ beat: 0, playing: true })
  }, [])   // eslint-disable-line react-hooks/exhaustive-deps

  const finish = useCallback(() => {   // natural completion — keep the written sentence
    clearTimers()
    setTour(null)
    lastBeatRef.current = -1
    ctl.onTourDone?.()
  }, [])   // eslint-disable-line react-hooks/exhaustive-deps

  const exit = useCallback(() => {     // user bail — restore everything
    clearTimers()
    const cur = lastBeatRef.current
    if (cur >= 0) beats[cur]?.exit?.(ctlRef.current)
    const s = savedRef.current
    if (s) {
      ctl.setSentence(s.sentence); ctl.setStage(s.stage)
      ctl.setLayer(s.layer); ctl.setHead(s.head); ctl.setAblate(s.ablate)
      ctl.setSelToken(s.selToken); ctl.setProbe(s.probe)
      ctl.setGenTemp(s.genTemp); ctl.setTopK(s.topK); ctl.setPriorOn(s.priorOn)
    }
    setTour(null)
    lastBeatRef.current = -1
  }, [beats])   // eslint-disable-line react-hooks/exhaustive-deps

  const tourValRef = useRef(null)
  tourValRef.current = tour

  const next = useCallback(() => {
    const t = tourValRef.current
    if (!t) return
    if (t.beat + 1 >= beats.length) { finish(); return }
    setTour({ ...t, beat: t.beat + 1 })
  }, [beats, finish])

  const prev = useCallback(() => setTour(t => (t && t.beat > 0 ? { ...t, beat: t.beat - 1 } : t)), [])
  const togglePause = useCallback(() => setTour(t => (t ? { ...t, playing: !t.playing } : t)), [])

  // beat transitions: run prev exit, jump stage, run enter
  useEffect(() => {
    if (!tour) return
    const prevIdx = lastBeatRef.current
    if (prevIdx >= 0 && prevIdx !== tour.beat) beats[prevIdx]?.exit?.(ctlRef.current)
    lastBeatRef.current = tour.beat
    clearTimeout(fxTimer.current)
    const b = beats[tour.beat]
    ctl.setStage(b.stage)
    b.enter?.(ctlRef.current)
  }, [tour?.beat])   // eslint-disable-line react-hooks/exhaustive-deps

  // auto-advance
  useEffect(() => {
    if (!tour || !tour.playing) return
    advanceTimer.current = setTimeout(next, beats[tour.beat].dur)
    return () => clearTimeout(advanceTimer.current)
  }, [tour?.beat, tour?.playing])   // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => clearTimers, [])

  return { tour, start, next, prev, exit, togglePause }
}

/* Spotlight + caption card. Renders null when the tour is off (SSR-safe).
   Spotlight geometry is written imperatively from a rAF loop — resilient to
   stage transitions, smooth scrolling, resize, and token-count changes. */
export function TourOverlay({ tour, beats, onNext, onPrev, onExit, onTogglePause }) {
  const spotRef = useRef(null)
  const anchorElRef = useRef(null)
  const beat = tour ? beats[tour.beat] : null

  // find + scroll to the anchor on each beat
  useEffect(() => {
    if (!tour) return
    let alive = true
    anchorElRef.current = null
    ;(async () => {
      const el = (await waitForEl(`[data-tour="${beat.anchor}"]`)) ||
                 (await waitForEl('[data-tour="stage-body"]', 600))
      if (!alive || !el) return
      anchorElRef.current = el
      el.scrollIntoView({ block: 'center', behavior: 'smooth' })
    })()
    return () => { alive = false }
  }, [tour?.beat])   // eslint-disable-line react-hooks/exhaustive-deps

  // continuous geometry tracking
  useEffect(() => {
    if (!tour) return
    let raf
    const loop = () => {
      const el = anchorElRef.current, spot = spotRef.current
      if (el && spot) {
        const r = el.getBoundingClientRect()
        const pad = 10
        spot.style.opacity = '1'
        spot.style.transform = `translate(${r.left - pad}px, ${r.top - pad}px)`
        spot.style.width = `${r.width + pad * 2}px`
        spot.style.height = `${r.height + pad * 2}px`
      } else if (spot) {
        spot.style.opacity = '0'
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [tour !== null])   // eslint-disable-line react-hooks/exhaustive-deps

  if (!tour) return null
  const n = beats.length
  return (
    <div className="tourlayer">
      <div ref={spotRef} className="spotlight" />
      <AnimatePresence mode="wait">
        <motion.div key={tour.beat} className="tourcard"
          initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 12 }}
          transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}>
          <div className="tourprog-track">
            <span key={tour.beat} className="tourprog"
              style={{ animationDuration: `${beat.dur}ms`, animationPlayState: tour.playing ? 'running' : 'paused' }} />
          </div>
          <div className="tourhead">
            <h3>{beat.title}</h3>
            <span className="tourcount">{String(tour.beat + 1).padStart(2, '0')} / {String(n).padStart(2, '0')}</span>
          </div>
          <p>{beat.body}</p>
          <div className="tourctls">
            <button type="button" className="tbtn" onClick={onPrev} disabled={tour.beat === 0}>‹</button>
            <button type="button" className="tbtn" onClick={onTogglePause}>{tour.playing ? '⏸' : '▶'}</button>
            <button type="button" className="tbtn" onClick={onNext}>›</button>
            <span className="tourhint">space = pause · esc = exit</span>
            <button type="button" className="tbtn texit" onClick={onExit}>✕ Exit tour</button>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
