import { useEffect, useRef, useState } from 'react'

export const GLOSSARY = {
  token: 'A word or punctuation mark, converted to an integer ID. The only unit the model sees.',
  embedding: 'The vector of numbers that stands in for a token — its coordinates in meaning-space.',
  logit: 'A raw, unnormalized score for one vocabulary word. Bigger = the model likes it more. Softmax turns logits into probabilities.',
  softmax: "Turns any list of scores into probabilities: exponentiate everything (so it's positive), then divide by the total (so it sums to 1).",
  layernorm: 'Rescales a vector to zero mean and unit variance before each block — keeps the numbers in a range the math behaves in.',
  'residual stream': 'The running vector each token carries through the model. Every block adds its output to it; nothing is overwritten.',
  attention: 'The mechanism that lets tokens copy information from earlier tokens, weighted by relevance.',
  'query · key': 'Each token emits a query ("what am I looking for?") and a key ("what do I contain?"). Their dot product decides the attention weight.',
  gelu: 'A soft on/off gate: negative inputs are squashed toward zero, positive ones pass through. The non-linearity that makes depth useful.',
  temperature: 'Divides the logits before softmax. Low = sharp and confident. High = flat and random.',
  'top-k': 'Keep only the k highest-probability words, discard the rest, renormalize. A crude but effective safety rail.',
  ablation: "Zeroing a component's output to see what breaks — the standard way to ask what a head actually does.",
  'logit lens': 'Decoding the residual stream early, at every layer, to watch the prediction form before the model is done.',
  'causal mask': 'The rule that a token may only attend to itself and earlier tokens — no peeking at the future it must predict.',
}

/* Hover/focus tooltip. The tip only mounts while open, so SSR output is just the
   plain underlined span — no measurement, no window access at render time. */
export function Term({ k, children }) {
  const [open, setOpen] = useState(false)
  const [flip, setFlip] = useState('')
  const tipRef = useRef(null)
  useEffect(() => {
    if (!open || !tipRef.current) return
    const r = tipRef.current.getBoundingClientRect()
    if (r.left < 8) setFlip('flip-right')
    else if (r.right > window.innerWidth - 8) setFlip('flip-left')
  }, [open])
  const show = () => setOpen(true)
  const hide = () => { setOpen(false); setFlip('') }
  return (
    <span className="term" tabIndex={0}
      onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide}>
      {children || k}
      {open && <span ref={tipRef} className={`termtip ${flip}`} role="tooltip">{GLOSSARY[k]}</span>}
    </span>
  )
}

/* Pulsing affordance pill. State-derived: parent decides `show`. */
export function Hint({ show = true, children }) {
  if (!show) return null
  return <span className="hintpill" aria-hidden="true">{children}</span>
}
