import { useEffect, useRef } from 'react'
import { motion, animate } from 'framer-motion'
import { cellColor, HEADNAMES, PALETTE, NLAYER, NHEAD } from './model.js'

export const STAGES = [
  { id: 'map', num: '00', lbl: 'The Map' },
  { id: 'tokenize', num: '01', lbl: 'Tokenize' },
  { id: 'embed', num: '02', lbl: 'Embed + Pos' },
  { id: 'attention', num: '03', lbl: 'Attention' },
  { id: 'mlp', num: '04', lbl: 'MLP' },
  { id: 'stream', num: '05', lbl: 'The Stream' },
  { id: 'predict', num: '06', lbl: 'Predict' },
]

export const fmt = (v, d = 2) => (v >= 0 ? '+' : '') + v.toFixed(d)

export const rise = {
  hidden: { opacity: 0, y: 14 },
  show: (i = 0) => ({ opacity: 1, y: 0, transition: { delay: i * 0.05, type: 'spring', stiffness: 220, damping: 24 } }),
}

/* Rolling numeric readout. SSR renders the static value; on change, framer's
   imperative animate() writes textContent directly — zero React re-renders,
   interruptible, and skipped for sub-threshold deltas. */
export function NumberTicker({ value, decimals = 0, suffix = '', prefix = '', className }) {
  const ref = useRef(null)
  const prev = useRef(value)
  useEffect(() => {
    const from = prev.current
    prev.current = value
    if (!ref.current) return
    const text = v => prefix + v.toFixed(decimals) + suffix
    if (Math.abs(value - from) < Math.pow(10, -decimals) / 2) { ref.current.textContent = text(value); return }
    const ctrl = animate(from, value, {
      duration: 0.35, ease: 'easeOut',
      onUpdate: v => { if (ref.current) ref.current.textContent = text(v) },
    })
    return () => ctrl.stop()
  }, [value, decimals, suffix, prefix])
  return (
    <span ref={ref} className={className} style={{ fontVariantNumeric: 'tabular-nums' }}>
      {prefix + value.toFixed(decimals) + suffix}
    </span>
  )
}

/* Physical toggle switch — knob slides via layout spring when justify flips. */
export function Switch({ on, onToggle, tone = 'lime', children }) {
  return (
    <button type="button" className={`switch tone-${tone}` + (on ? ' on' : '')} onClick={onToggle} aria-pressed={on}>
      <span className="sw-track">
        <motion.span className="sw-knob" layout transition={{ type: 'spring', stiffness: 500, damping: 30 }} />
      </span>
      {children && <span className="sw-lbl">{children}</span>}
    </button>
  )
}

export function Strip({ vec, scale = 1.5 }) {
  return (
    <div className="vecbar">
      {vec.map((v, i) => (
        <div key={i} className="vcell" style={{ background: cellColor(v, scale), '--hd': `${i * 10}ms` }}
          title={`dim ${i}: ${v.toFixed(4)}`} />
      ))}
    </div>
  )
}

export function SecLbl({ children }) {
  return <div className="seclbl">{children}</div>
}

export function FlowMini({ here, go }) {
  return (
    <div className="flowmini">
      {STAGES.map((s, i) => (
        <span key={s.id} style={{ display: 'contents' }}>
          <button type="button" className={'fnode' + (s.id === here ? ' here' : '')} onClick={() => go(i)}>{s.lbl}</button>
          {i < STAGES.length - 1 && <span className="arr">→</span>}
        </span>
      ))}
    </div>
  )
}

export function PickerRow({ tokens, sel, onPick }) {
  return (
    <div className="qtoks">
      {tokens.map((t, i) => (
        <motion.button type="button" key={i} className={'qtok' + (i === sel ? ' sel' : '')} onClick={() => onPick(i)}
          whileHover={{ y: -2 }} whileTap={{ scale: 0.95 }}>
          {t}<sub> {i}</sub>
        </motion.button>
      ))}
    </div>
  )
}

export function LayerHeadBar({ layer, head, setLayer, setHead, ablate, toggleAblate, showAblate = true }) {
  return (
    <div className="lhbtns" data-tour="lh-bar">
      {Array.from({ length: NLAYER }, (_, L) => (
        <motion.button type="button" key={L} className={'lh-btn' + (L === layer ? ' on' : '')} whileTap={{ scale: 0.95 }}
          onClick={() => setLayer(L)}>
          {L === layer && <motion.span layoutId="layerPill" className="lh-pill" style={{ background: 'var(--violet)' }}
            transition={{ type: 'spring', stiffness: 400, damping: 32 }} />}
          <span className="lh-txt">LAYER {L + 1}</span>
        </motion.button>
      ))}
      <span className="lh-spacer" />
      {Array.from({ length: NHEAD }, (_, h) => (
        <motion.button type="button" key={h} whileTap={{ scale: 0.95 }}
          className={'lh-btn' + (h === head ? ' on' : '') + (ablate[layer][h] ? ' mute' : '')}
          onClick={() => setHead(h)}>
          {h === head && <motion.span layoutId="headPill" className="lh-pill" style={{ background: PALETTE[h] }}
            transition={{ type: 'spring', stiffness: 400, damping: 32 }} />}
          <span className="lh-txt">{HEADNAMES[layer][h]}</span>
        </motion.button>
      ))}
      {showAblate && <span className="lh-flex" />}
      {showAblate && (
        <span className="abl-group" data-tour="ablate">
          {Array.from({ length: NHEAD }, (_, h) => (
            <Switch key={h} tone="rose" on={ablate[layer][h]} onToggle={() => toggleAblate(layer, h)}>
              mute {HEADNAMES[layer][h].toLowerCase()}
            </Switch>
          ))}
        </span>
      )}
    </div>
  )
}
