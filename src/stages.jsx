import { useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  D, DH, DFF, VOCAB, VIDX, catOf, CATCOLOR, emb, norm, gelu, heatColor, lerpColor,
  distribution, lensTop, paramCount, PALETTE, HEADNAMES,
} from './model.js'
import { Strip, SecLbl, PickerRow, LayerHeadBar, NumberTicker, Switch, fmt, rise } from './ui.jsx'
import { Term, Hint } from './glossary.jsx'

/* ---------------- 00 THE MAP ---------------- */
function MNode({ x, y, w, h, main, sub, anno, hi, onClick }) {
  return (
    <g>
      <g className="mapnode" transform={`translate(${x},${y})`} onClick={onClick}>
        <rect width={w} height={h} rx="3" />
        <text x="14" y={h / 2 - 2}>{main}</text>
        <text x="14" y={h / 2 + 14} className="sub-t">{sub}</text>
      </g>
      <text className={'mapanno' + (hi ? ' hi' : '')} x={x + w + 16} y={y + h / 2 + 3}>{anno}</text>
    </g>
  )
}

export function MapStage({ F, go, hints }) {
  const cx = 180, W2 = 300
  const spine = `M${cx + W2 / 2} 30 L${cx + W2 / 2} 662`
  const nodes = [
    [40, 44, 1, 'TOKENIZE', `"${F.tokens.slice(0, 4).join(' ')}${F.n > 4 ? ' …' : ''}"`, `n = ${F.n} tokens`, 1],
    [110, 44, 2, 'EMBED + POSITION', 'lookup Wₑ · add sinusoids', `${F.n} × ${D} matrix`, 0],
    [186, 50, 3, 'BLOCK 1 · ATTENTION', 'LN → QKV → softmax(QKᵀ/√d + b)·V', '2 heads: position + content', 1],
    [258, 50, 4, 'BLOCK 1 · MLP', 'LN → 16→64 GELU → 64→16', `${DFF} neurons / token`, 0],
    [330, 50, 3, 'BLOCK 2 · ATTENTION', 'same machinery, deeper features', '2 heads: recency + untrained', 0],
    [402, 50, 4, 'BLOCK 2 · MLP', 'refine again', `${DFF} neurons / token`, 0],
    [478, 44, 5, 'FINAL LAYERNORM', 'normalize the stream', 'logit lens lives here', 0],
    [548, 44, 6, `UNEMBED  h · Wₑᵀ`, `project onto all ${VOCAB.length} words`, `${VOCAB.length} logits`, 1],
    [618, 44, 6, 'SOFTMAX → SAMPLE', 'probabilities → one token', 'loop back to top ↺', 0],
  ]
  return (
    <div data-tour="archmap">
      {hints && <Hint>every box is a door — click one</Hint>}
      <svg className="archmap" viewBox="0 0 860 700">
        <path d={spine} stroke="var(--line)" strokeWidth="1.5" fill="none" />
        {[150, 244, 388, 482].map(y => (
          <path key={y} className="skipArc" d={`M${cx - 14} ${y} C ${cx - 64} ${y + 22}, ${cx - 64} ${y + 52}, ${cx - 14} ${y + 74}`}
            stroke="var(--violet)" strokeWidth="1.2" fill="none" opacity=".6" strokeDasharray="3 3" />
        ))}
        <text className="mapanno" x="34" y="330" fill="var(--violet)" transform="rotate(-90 40 330)">
          residual stream — the highway every block writes into
        </text>
        {nodes.map(([y, h, g, main, sub, anno, hi], i) => (
          <MNode key={i} x={cx} y={y} w={W2} h={h} main={main} sub={sub} anno={anno} hi={hi} onClick={() => go(g)} />
        ))}
        {[0, -1.7, -3.4].map(b => (
          <circle key={b} r="4" fill="var(--amber)">
            <animateMotion dur="5s" begin={`${b}s`} repeatCount="indefinite" path={spine} />
          </circle>
        ))}
        {[-0.35, -2.05, -3.75].map(b => (
          <circle key={b} r="2.5" fill="var(--cyan)" opacity=".8">
            <animateMotion dur="5s" begin={`${b}s`} repeatCount="indefinite" path={spine} />
          </circle>
        ))}
      </svg>
      <p className="note">
        <b>The one idea that organizes everything:</b> a token's vector rides a <Term k="residual stream">residual
        stream</Term> from bottom to top. <Term k="attention">Attention</Term> blocks <em>move information between
        tokens</em>; MLP blocks <em>process each token alone</em>. Each block only <b>adds</b> its contribution to the
        stream (the dashed violet skips) — nothing is overwritten. The final prediction is just the last token's stream
        vector, matched against every word in the vocabulary.
      </p>
    </div>
  )
}

/* ---------------- 01 TOKENIZE ---------------- */
export function TokenizeStage({ F, selToken, spotlight, killToken, hints }) {
  return (
    <div>
      {hints && <Hint show={selToken === null}>click a token to spotlight it everywhere</Hint>}
      <div className="tokrow" data-tour="tokens">
        <AnimatePresence mode="popLayout">
          {F.tokens.map((t, i) => {
            const c = CATCOLOR[catOf(t)]
            return (
              <motion.span key={`${t}-${i}`} layout className={'token' + (i === selToken ? ' sel' : '')}
                custom={i} variants={rise} initial="hidden" animate="show"
                exit={{ opacity: 0, scale: 0.8 }}
                whileHover={{ y: -3, scale: 1.03 }} whileTap={{ scale: 0.95 }}
                style={{ borderColor: i === selToken ? 'var(--amber)' : c + '66' }}
                onClick={() => spotlight(i)}>
                <span className="kill" onClick={e => { e.stopPropagation(); killToken(i) }}>×</span>
                {t}
                <span className="tid">#{VIDX.get(t)}</span>
                <span className="cat" style={{ color: c }}>{catOf(t)}</span>
              </motion.span>
            )
          })}
        </AnimatePresence>
      </div>
      <SecLbl>the actual vocabulary (first 28 of {VOCAB.length})</SecLbl>
      <p className="note vocabpeek" style={{ marginTop: 4 }}>
        {VOCAB.slice(0, 28).map((w, i) => (
          <span key={w}>
            <span className="vw" style={{ color: CATCOLOR[catOf(w)] }}>{w}</span>
            <span style={{ opacity: .35 }}>#{VIDX.get(w)}</span>
            {i < 27 && ' · '}
          </span>
        ))} …
      </p>
      <p className="note">
        <b>These IDs are real</b> — they index rows of the embedding matrix Wₑ on the next stage. Every <Term
        k="token">token</Term> the model has never seen gets appended to the vocabulary live (its <Term
        k="embedding">embedding</Term> is generated from a hash — a stand-in for what training would learn).
      </p>
    </div>
  )
}

/* ---------------- 02 EMBED + POSITION ---------------- */
export function EmbedStage({ F, selToken, spotlight }) {
  const Rows = ({ X, lbl, scale }) => (
    <div className="vecgrid">
      {F.tokens.map((t, i) => (
        <motion.div key={i} className={'vecrow' + (i === selToken ? ' sel' : '')}
          custom={i} variants={rise} initial="hidden" animate="show">
          <button type="button" className="vlbl" onClick={() => spotlight(i)}>{lbl(t, i)}</button>
          <Strip vec={X[i]} scale={scale} />
          <div className="vnorm">‖x‖=<NumberTicker value={norm(X[i])} decimals={2} /></div>
        </motion.div>
      ))}
    </div>
  )
  return (
    <div>
      <SecLbl>token embeddings — rows of <b>Wₑ</b> (hover cells for exact values)</SecLbl>
      <div className="dimruler"><span className="dr-band pos-band">dims 0–7 · position band</span><span className="dr-band cat-band">dims 8–15 · category + texture</span></div>
      <div data-tour="emb-rows"><Rows X={F.embRows} lbl={t => t} scale={1.4} /></div>
      <div className="legend">
        <span><span className="swatch" style={{ background: '#5ad1c8' }} />positive</span>
        <span><span className="swatch" style={{ background: '#ff6b8a' }} />negative</span>
        <span>each row is one <Term k="embedding">embedding</Term> — spot the bright column per part-of-speech</span>
      </div>
      <SecLbl>positional encoding — <b>sin/cos waves</b>, unique fingerprint per slot</SecLbl>
      <div data-tour="pos-rows"><Rows X={F.posRows} lbl={(t, i) => `pos ${i}`} scale={0.7} /></div>
      <SecLbl>the sum — <b>x₀ = Wₑ[id] + PE(pos)</b> — this enters the residual stream</SecLbl>
      <div data-tour="sum-rows"><Rows X={F.checkpoints[1].X} lbl={(t, i) => `${t} @ ${i}`} scale={1.6} /></div>
      <p className="note">
        <b>Why sinusoids?</b> Every position gets a unique wave fingerprint, and shifting by one position is a
        fixed <em>rotation</em> of each sin/cos pair — position becomes geometry. Modern models push this
        further: <b>RoPE</b> rotates q and k directly, and <b>T5/ALiBi</b> skip absolute encodings
        for <em>relative</em> biases on attention scores — which is exactly what this model's position head uses on the
        next stage.
      </p>
    </div>
  )
}

/* ---------------- 03 ATTENTION ---------------- */
export function AttentionStage({ F, layer, head, setLayer, setHead, ablate, toggleAblate, selToken, spotlight, probe, setProbe, hints }) {
  const n = F.n
  const H = F.layers[layer].heads[head]
  const size = Math.min(44, Math.max(24, Math.floor(340 / n)))
  const pi = probe ? probe.i : (selToken ?? n - 1)
  const pj = probe ? probe.j : Math.max(0, pi - 1)

  /* crosshair — imperative, zero re-renders */
  const rowBar = useRef(null), colBar = useRef(null)
  const onHeatMove = e => {
    const cell = e.target.closest('.hcell')
    if (!cell || !rowBar.current) return
    const i = +cell.dataset.i, j = +cell.dataset.j
    rowBar.current.style.opacity = colBar.current.style.opacity = '1'
    rowBar.current.style.transform = `translateY(${i * (size + 3)}px)`
    colBar.current.style.transform = `translateX(${j * (size + 3)}px)`
  }
  const onHeatLeave = () => {
    if (rowBar.current) rowBar.current.style.opacity = colBar.current.style.opacity = '0'
  }

  /* microscope data */
  const q = H.q[pi], k = H.k[pj]
  const prods = q.map((qq, d2) => qq * k[d2])
  const mx = Math.max(...prods.map(Math.abs), 1e-6)
  const sum = H.raw[pi][pj], bias = H.bias[pi - pj], scaled = H.scaled[pi][pj], aWeight = H.A[pi][pj]

  /* softmax microscope rows */
  const rowRaw = [], rowScaled = []
  for (let j = 0; j <= pi; j++) { rowRaw.push(H.raw[pi][j]); rowScaled.push(H.scaled[pi][j]) }
  const mxs = Math.max(...rowScaled)
  const rowExp = rowScaled.map(v => Math.exp(v - mxs))
  const sumExp = rowExp.reduce((a, b) => a + b, 0)
  const rowP = rowExp.map(e => e / sumExp)

  const StepRow = ({ lbl, sub, vals, color }) => {
    const m = Math.max(...vals.map(Math.abs), 1e-6)
    return (
      <div className="smaxrow">
        <div className="slbl"><b>{lbl}</b>{sub}</div>
        <div className="smaxbars">
          {vals.map((v, i) => (
            <motion.div key={i} className="sb"
              animate={{ height: `${Math.max(4, Math.abs(v) / m * 100)}%` }}
              transition={{ type: 'spring', stiffness: 120, damping: 20, delay: i * 0.02 }}
              style={{ background: color, opacity: v < 0 ? .45 : 1 }}>
              <span className="sv">{v.toFixed(v < 10 ? 2 : 1)}</span>
            </motion.div>
          ))}
        </div>
      </div>
    )
  }

  const headExplain = layer === 0
    ? (head === 0
      ? <>This head carries a <b>relative-position bias</b> b[i−j] peaked at distance 1 — the exact mechanism of <b>T5's attention bias and ALiBi</b> (BLOOM, MPT). The bright sub-diagonal is that bias; the q·k part is untrained noise on top. Click a cell and read the "+ rel-bias" step in the microscope.</>
      : <>This head's W<sub>q</sub>, W<sub>k</sub> both read the category dims (8–15), so q·k is big when two tokens share a <b>grammatical category</b> — nouns find nouns, "the" finds "the". Pure content-based routing, zero position info.</>)
    : (head === 0
      ? <>A pure <b>ALiBi head</b>: no crafted q·k at all, just a linear distance penalty b[i−j] = −0.5·(i−j). Watch attention fade smoothly with distance — recency as architecture.</>
      : <>Fully random (untrained) weights — this is what attention looks like <b>before training carves structure into it</b>. Compare with the other three heads.</>)

  return (
    <div>
      <LayerHeadBar layer={layer} head={head} setLayer={setLayer} setHead={setHead} ablate={ablate} toggleAblate={toggleAblate} />
      {hints && <Hint show={probe === null}>click any cell to open the math</Hint>}
      <div className="attwrap">
        <div>
          <div className="heatlayout" data-tour="heatmap">
            <div className="axis-left" style={{ gridTemplateRows: `repeat(${n},${size}px)` }}>
              {F.tokens.map((t, i) => (
                <button type="button" key={i} className={i === selToken ? 'hot' : ''} onClick={() => spotlight(i)}>{t}</button>
              ))}
            </div>
            <div>
              <div className="axis-top" style={{ gridTemplateColumns: `repeat(${n},${size}px)` }}>
                {F.tokens.map((t, i) => <span key={i}>{t}</span>)}
              </div>
              <div className="heatbox" onMouseMove={onHeatMove} onMouseLeave={onHeatLeave}>
                <div ref={rowBar} className="xhair xh-row" style={{ height: size }} />
                <div ref={colBar} className="xhair xh-col" style={{ width: size }} />
                <div className="heat" key={n} style={{ gridTemplateColumns: `repeat(${n},${size}px)` }}>
                  {F.tokens.flatMap((_, i) => F.tokens.map((_, j) => {
                    const causal = j <= i, v = H.A[i][j]
                    const probed = probe && probe.i === i && probe.j === j
                    return (
                      <div key={`${i}-${j}`} data-i={i} data-j={j}
                        className={'hcell' + (v > .28 ? ' showval' : '') + (probed ? ' probed' : '')}
                        style={{
                          background: causal ? heatColor(v) : '#0d0e13', opacity: causal ? 1 : .3,
                          '--hd': `${(i + j) * 14}ms`,
                        }}
                        title={`A[${i}][${j}] = ${v.toFixed(4)}   (raw q·k = ${causal ? H.raw[i][j].toFixed(3) : 'masked'})`}
                        onClick={() => causal && setProbe({ i, j })}>
                        {causal && v > .28 ? Math.round(v * 100) : ''}
                      </div>
                    )
                  }))}
                </div>
              </div>
            </div>
          </div>
          <div className="legend">
            <span>rows = <b>from</b> · cols = <b>to</b></span>
            <span>dark = <Term k="causal mask">causal mask</Term></span>
            <span>hover = exact value</span>
          </div>
          <p className="note">{headExplain}</p>
        </div>
        <div>
          {pj <= pi && (
            <motion.div className="scope" data-tour="scope-dot" layout initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
              <h4>Dot-product microscope — <b>
                <AnimatePresence mode="wait" initial={false}>
                  <motion.span key={`${pi}-${pj}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: .15 }}>
                    q("{F.tokens[pi]}") · k("{F.tokens[pj]}")
                  </motion.span>
                </AnimatePresence>
              </b> — click any heat cell to retarget</h4>
              <div className="scope-legend">
                <span style={{ color: 'var(--rose)' }}>■ <Term k="query · key">query</Term> vector of "{F.tokens[pi]}"</span>
                <span style={{ color: 'var(--cyan)' }}>■ key vector of "{F.tokens[pj]}"</span>
              </div>
              {prods.map((p, d2) => {
                const w = Math.abs(p) / mx * 50
                return (
                  <div className="prodrow" key={d2}>
                    <span className="pdim">d{d2}</span>
                    <span className="pq">{fmt(q[d2])}</span><span className="px">×</span>
                    <span className="pk">{fmt(k[d2])}</span><span className="px">=</span>
                    <div className="prodbar">
                      <motion.div className="pb" animate={{ width: `${p >= 0 ? w : 0}%` }}
                        transition={{ type: 'spring', stiffness: 170, damping: 26 }} style={{ left: '50%', background: 'var(--cyan)' }} />
                      <motion.div className="pb" animate={{ width: `${p < 0 ? w : 0}%` }}
                        transition={{ type: 'spring', stiffness: 170, damping: 26 }} style={{ right: '50%', background: 'var(--rose)' }} />
                    </div>
                    <span className="pv" style={{ color: p >= 0 ? 'var(--cyan)' : 'var(--rose)' }}>{fmt(p)}</span>
                  </div>
                )
              })}
              <div className="sumline">
                <span>Σ products = <b><NumberTicker value={sum} decimals={3} /></b></span>
                <span>÷ √{DH} = <b><NumberTicker value={sum / Math.sqrt(DH)} decimals={3} /></b></span>
                <span>+ rel-bias b[{pi - pj}] = <b><NumberTicker value={bias} decimals={2} /></b></span>
                <span><Term k="softmax">softmax</Term> → weight <b><NumberTicker value={aWeight * 100} decimals={1} suffix="%" /></b></span>
              </div>
            </motion.div>
          )}
        </div>
      </div>
      <div className="scope" data-tour="scope-softmax">
        <h4>Softmax microscope — row "<b>{F.tokens[pi]}</b>" step by step</h4>
        <div className="smaxsteps">
          <StepRow lbl="1 · raw q·k" sub="scores" vals={rowRaw} color="#7aa2ff" />
          <StepRow lbl="2 · ÷ √d + b[i−j]" sub="scaled + rel-bias" vals={rowScaled} color="#b18cff" />
          <StepRow lbl="3 · exp( )" sub="all positive" vals={rowExp} color="#ff8c42" />
          <StepRow lbl="4 · normalize" sub="sums to 1" vals={rowP} color="#ffb347" />
        </div>
        <div className="smaxlbls">{F.tokens.slice(0, pi + 1).map((t, i) => <span key={i}>{t}</span>)}</div>
        <p className="note" style={{ marginTop: 12 }}>
          Step 2 is why the formula divides by √d: without it, dot products grow with dimension, exp() explodes, and
          attention collapses onto one token before training can shape it.
        </p>
      </div>
      <p className="note">
        <b>Why it matters:</b> in "the cat sat because <em>it</em> was tired", the word <code>it</code> learns to attend
        back to <code>cat</code>. That's how a transformer resolves references — no rules, just learned <Term
        k="attention">attention</Term>.
      </p>
    </div>
  )
}

/* ---------------- 04 MLP ---------------- */
export function MlpStage({ F, layer, head, setLayer, setHead, ablate, selToken, spotlight, hints }) {
  const t = selToken ?? F.n - 1
  const Ly = F.layers[layer]
  const pre = Ly.pre[t], act = Ly.act[t]
  const mxa = Math.max(...act.map(Math.abs), 1e-6)
  const W2 = 560, H2 = 220, x0 = -4, x1 = 4
  const px = x => ((x - x0) / (x1 - x0)) * W2
  const py = y => H2 - ((y + 1) / 5) * H2
  let path = 'M'
  for (let x = x0; x <= x1; x += .1) path += `${px(x).toFixed(1)} ${py(gelu(x)).toFixed(1)} L`
  path = path.slice(0, -1)
  const fired = act.filter(a => a > .05).length
  const top3 = [...act.keys()].sort((a, b) => Math.abs(act[b]) - Math.abs(act[a])).slice(0, 3)
  return (
    <div>
      <LayerHeadBar layer={layer} head={head} setLayer={setLayer} setHead={setHead} ablate={ablate} toggleAblate={() => {}} showAblate={false} />
      <SecLbl>choose a token — its 64 neurons light up</SecLbl>
      {hints && <Hint show={selToken === null}>pick a token below</Hint>}
      <PickerRow tokens={F.tokens} sel={t} onPick={spotlight} />
      <div className="mlpgrid">
        <div>
          <SecLbl>all {DFF} neuron activations for "<b>{F.tokens[t]}</b>" (layer {layer + 1})</SecLbl>
          <div className="neurons" data-tour="neurons">
            {act.map((a, ni) => {
              const v = a / mxa
              const col = v >= 0 ? lerpColor('#141620', '#c3e88d', Math.abs(v)) : lerpColor('#141620', '#ff6b8a', Math.abs(v))
              return (
                <div key={ni} className="neuron"
                  style={{ background: col, boxShadow: `0 0 ${Math.abs(v) * 10}px ${col}`, '--hd': `${ni * 6}ms` }}
                  title={`neuron ${ni}: pre=${pre[ni].toFixed(3)}  gelu→ ${a.toFixed(3)}`} />
              )
            })}
          </div>
          <div className="legend">
            <span><span className="swatch" style={{ background: '#c3e88d' }} />firing</span>
            <span><span className="swatch" style={{ background: '#ff6b8a' }} />suppressed</span>
            <span><b><NumberTicker value={fired} />/{DFF}</b> neurons active — sparsity is the point</span>
          </div>
          <p className="note">
            Strongest: {top3.map(ni => <code key={ni}>n{ni} → {act[ni].toFixed(2)} </code>)} . In trained models
            individual neurons become detectors — "this token is inside a quote", "this is a year", "French text". The
            MLP is where a transformer stores most of its <b>knowledge</b>; attention only routes it.
          </p>
        </div>
        <div data-tour="gelu">
          <SecLbl>the <Term k="gelu">GELU</Term> gate — every dot is one of this token's neurons</SecLbl>
          <svg className="gelu" viewBox={`0 0 ${W2} ${H2}`}>
            <line x1="0" y1={py(0)} x2={W2} y2={py(0)} stroke="var(--line)" strokeWidth="1" />
            <line x1={px(0)} y1="0" x2={px(0)} y2={H2} stroke="var(--line)" strokeWidth="1" />
            <motion.path d={path} fill="none" stroke="var(--amber)" strokeWidth="2"
              initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }} />
            {pre.map((p, ni) => (
              <circle key={ni} cx="0" cy="0" r="3.4" className="geludot"
                style={{ transform: `translate(${px(Math.max(x0, Math.min(x1, p)))}px, ${py(gelu(p))}px)` }}
                fill={act[ni] >= 0.02 ? '#c3e88d' : '#ff6b8a'} opacity=".85">
                <title>{`neuron ${ni}: x=${p.toFixed(2)} → ${act[ni].toFixed(2)}`}</title>
              </circle>
            ))}
            <text x={W2 - 120} y={py(3.2)} fill="var(--amber)" fontSize="11" fontFamily="var(--mono)">gelu(x)</text>
          </svg>
          <p className="note">
            GELU ≈ a soft on/off switch: negative inputs are squashed toward zero (neuron stays quiet), positives pass
            through. This <b>non-linearity</b> is what lets stacked layers compute things a single matrix never could.
          </p>
        </div>
      </div>
    </div>
  )
}

/* ---------------- 05 THE STREAM ---------------- */
export function StreamStage({ F, selToken, spotlight, hints }) {
  const t = selToken ?? F.n - 1
  const cks = F.checkpoints
  const [hoverCk, setHoverCk] = useState(null)
  return (
    <div>
      <SecLbl>follow one token through the machine</SecLbl>
      {hints && <Hint show={selToken === null}>pick a token below</Hint>}
      <PickerRow tokens={F.tokens} sel={t} onPick={spotlight} />
      <div className="streamwrap">
        <div data-tour="stream">
          <SecLbl>residual stream of "<b>{F.tokens[t]}</b>" — same 16 dims, evolving (Δ = how much each block wrote)</SecLbl>
          <div className="vecgrid">
            {cks.map((ck, c) => {
              const x = ck.X[t]
              const d = c > 0 ? norm(x.map((v, i) => v - cks[c - 1].X[t][i])) : null
              return (
                <motion.div key={c} className={'vecrow linkable' + (hoverCk === c ? ' linked' : '')}
                  onMouseEnter={() => setHoverCk(c)} onMouseLeave={() => setHoverCk(null)}
                  custom={c} variants={rise} initial="hidden" animate="show">
                  <div className="vlbl asdiv">{ck.name}</div>
                  <Strip vec={x} scale={2.2} />
                  <div className="vnorm">
                    ‖x‖=<NumberTicker value={norm(x)} decimals={2} />{' '}
                    {d !== null && <span className="dnorm">Δ <NumberTicker value={d} decimals={2} /></span>}
                  </div>
                </motion.div>
              )
            })}
          </div>
          <p className="note">
            Attention deltas move information <b>between</b> tokens; MLP deltas <b>refine in place</b>. The vector that
            exits is the same object that entered — plus everything the network chose to write into it. That's why it's
            called a <Term k="residual stream">stream</Term>.
          </p>
        </div>
        <div>
          <SecLbl><Term k="logit lens">logit lens</Term> — decoding the <b>last position</b> at every depth</SecLbl>
          <div className="lens" data-tour="lens">
            <table>
              <tbody>
                <tr><th>checkpoint</th><th>if we stopped here, the model would predict…</th></tr>
                {cks.map((ck, c) => (
                  <tr key={c} className={hoverCk === c ? 'linked' : ''}
                    onMouseEnter={() => setHoverCk(c)} onMouseLeave={() => setHoverCk(null)}>
                    <td className="ck">{ck.name}</td>
                    <td>
                      {lensTop(F, ck.X).map((o, r) => (
                        <span key={r} className={'cand' + (r === 0 ? ' top' : '')}>
                          {r === 0 ? <b key={o.word} className="topword">{o.word}</b> : <b>{o.word}</b>}
                          {' '}<span className="pp">{(o.p * 100).toFixed(0)}%</span>
                        </span>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note">
            <b>This is the model changing its mind in real time.</b> Early rows decode near-random guesses; each block's
            contribution nudges the last position's vector toward the final answer. In GPT-scale models you can watch a
            fact ("Paris") surface at a specific layer — same technique, same math.
          </p>
        </div>
      </div>
    </div>
  )
}

/* ---------------- 06 PREDICT ---------------- */
export function PredictStage({ F, genTemp, setGenTemp, topK, setTopK, priorOn, setPriorOn, auto, setAuto, sampleStep, appendWord, undo, reset, ablate, hints, hasSampled }) {
  const dist = distribution(F, { prior: priorOn, temp: genTemp, topk: Math.max(topK, 10) })
  const shown = dist.slice(0, 10)
  const maxP = Math.max(...shown.map(d => d.p))
  const muted = ablate.flat().filter(Boolean).length
  const tempPct = ((genTemp * 100 - 5) / (180 - 5)) * 100
  const topkPct = ((topK - 1) / 9) * 100
  return (
    <div>
      <div className="flowchain">
        <span className="fc">h<sub>last</sub> ∈ ℝ<sup>16</sup></span>→
        <span className="fc">· Wₑᵀ → <b>{VOCAB.length} <Term k="logit">logits</Term></b></span>→
        <span className="fc">÷ T=<b><NumberTicker value={genTemp} decimals={2} /></b></span>→
        <span className="fc"><Term k="top-k">top-k</Term>=<b><NumberTicker value={topK} /></b></span>→
        <span className="fc"><Term k="softmax">softmax</Term></span>→
        <span className="fc"><b>sample</b></span>
      </div>
      {hints && <Hint show={!hasSampled}>drag temperature, then hit Sample</Hint>}
      <div className="slider" data-tour="sliders">
        <span><Term k="temperature">Temperature</Term></span>
        <input type="range" min="5" max="180" value={Math.round(genTemp * 100)}
          style={{ '--fill': `${tempPct}%` }}
          onChange={e => setGenTemp(+e.target.value / 100)} />
        <span className="sval"><NumberTicker value={genTemp} decimals={2} /></span>
        <span>Top-k</span>
        <input type="range" min="1" max="10" value={topK}
          style={{ '--fill': `${topkPct}%` }}
          onChange={e => setTopK(+e.target.value)} />
        <span className="sval"><NumberTicker value={topK} /></span>
        <Switch tone="lime" on={priorOn} onToggle={() => setPriorOn(!priorOn)}>
          grammar prior {priorOn ? 'ON' : 'OFF'}
        </Switch>
      </div>
      <div data-tour="predrows">
        {shown.map((d, i) => (
          <motion.div key={d.word} layout="position" className={'predrow' + (i === 0 ? ' win' : '') + (i >= topK ? ' cut' : '')}
            initial={{ opacity: 0, x: -16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * .04 }}>
            <div className="word" data-fx-word={d.word} onClick={() => appendWord(d.word)}>{d.word}</div>
            <div className="lgt" title={`raw model logit ${d.raw.toFixed(2)}${priorOn ? ' + prior' : ''}`}>{d.logit.toFixed(2)}</div>
            <div className="track">
              <motion.div className="barfill"
                animate={{ width: `${(d.p / maxP * 100).toFixed(0)}%` }}
                transition={{ type: 'spring', stiffness: 90, damping: 20 }}
                style={{ background: PALETTE[i % PALETTE.length] }}>
                <NumberTicker value={d.p * 100} decimals={1} suffix="%" />
              </motion.div>
            </div>
          </motion.div>
        ))}
      </div>
      <div className="genbtns" data-tour="genbtns">
        <motion.button type="button" className="gbtn primary" data-fx="sample-btn" whileTap={{ scale: .95 }} onClick={sampleStep}>
          <span className="dice">🎲</span> Sample next token →
        </motion.button>
        <motion.button type="button" className={'gbtn' + (auto ? ' live' : '')} whileTap={{ scale: .95 }} onClick={() => setAuto(!auto)}>
          <AnimatePresence mode="wait" initial={false}>
            <motion.span key={auto ? 'stop' : 'go'} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: .12 }}>
              {auto ? '■ Stop auto-write' : '▶ Auto-write'}
            </motion.span>
          </AnimatePresence>
        </motion.button>
        <button type="button" className="gbtn" onClick={undo}>↶ Undo</button>
        <button type="button" className="gbtn" onClick={reset}>⟲ Reset</button>
      </div>
      <AnimatePresence>
        {muted > 0 && (
          <motion.div className="softnote warn" initial={{ opacity: 0, y: 10, height: 0 }}
            animate={{ opacity: 1, y: 0, height: 'auto' }} exit={{ opacity: 0, y: -6, height: 0 }}>
            <b className="w">⚠ {muted} attention head{muted > 1 ? 's' : ''} muted.</b> These probabilities are computed
            with that head's output zeroed — go back to Attention and un-mute to compare. This is <Term
            k="ablation">ablation</Term>, the standard tool for asking "what does this head actually do?"
          </motion.div>
        )}
      </AnimatePresence>
      <div className="softnote">
        <b>Honesty box:</b> the logits column is the real model output (dot products against tied embeddings). Since
        this micro-model is <b>untrained</b>, its raw preferences mostly reflect category geometry — the toggleable
        grammar prior adds bigram-style boosts so auto-write reads like English. Turn it <b>OFF</b> to see exactly what
        an untrained transformer believes. Training is nothing more than nudging all
        ~{paramCount().toLocaleString()} numbers until this page's bars rank human text highly.
      </div>
    </div>
  )
}
