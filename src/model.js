/* ================================================================
   THE GLASS TRANSFORMER — model core
   A real micro-GPT implemented from scratch. d=16, 2 heads (dh=8),
   2 layers, MLP 16→64→16 (GELU), tied unembedding, causal attention.
   Three heads use real published mechanisms:
     L1H1 — relative-position bias peaked at distance 1 (T5 / ALiBi)
     L1H2 — content head: Wq, Wk read the category subspace
     L2H1 — pure ALiBi recency head (linear distance penalty)
     L2H2 — untrained, for contrast.
   Every intermediate tensor is cached and inspectable in the UI.
   ================================================================ */

/* ---------- deterministic RNG ---------- */
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function gaussOf(r){let u=0,v=0;while(!u)u=r();while(!v)v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
export function hash(s){let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}return h>>>0}

/* ---------- math ---------- */
export const D=16, DH=8, NHEAD=2, NLAYER=2, DFF=64, PE_AMP=0.6, PE_BASE=100;
const zeros=(r,c)=>Array.from({length:r},()=>new Array(c).fill(0));
function matvec(M,v){const o=new Array(M.length);for(let r=0;r<M.length;r++){let s=0;const row=M[r];for(let c=0;c<row.length;c++)s+=row[c]*v[c];o[r]=s}return o}
function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s}
function addv(a,b){return a.map((x,i)=>x+b[i])}
export function norm(v){return Math.sqrt(dot(v,v))}
function layernorm(v){const m=v.reduce((a,b)=>a+b,0)/v.length;const va=v.reduce((a,b)=>a+(b-m)*(b-m),0)/v.length;const s=Math.sqrt(va+1e-5);return v.map(x=>(x-m)/s)}
export function gelu(x){return .5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+.044715*x*x*x)))}
function softmax(arr,t=1){const mx=Math.max(...arr);const e=arr.map(v=>Math.exp((v-mx)/t));const s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s)}

/* ---------- vocabulary & embeddings ---------- */
const LEX={
  det:['the','a','an','my','his','her'],
  noun:['cat','dog','mat','door','house','bird','fish','moon','sun','tree','robot','robots','sheep','child','king','sea','night','rain','light','room','floor'],
  verb:['sat','ran','sleeps','sang','jumped','opened','closed','saw','loved','ate','was','is','dream','dreams','fell','shines','need'],
  adj:['big','small','cold','warm','dark','bright','quiet','old','electric','tired','soft'],
  prep:['on','in','under','over','near','with','of','to'],
  pron:['it','she','he','they','we','i','you'],
  conj:['and','but','because','so','then'],
  punct:['.',',','!','?']
};
export const CATS=Object.keys(LEX);
export const CATCOLOR={det:'#7aa2ff',noun:'#5ad1c8',verb:'#ff6b8a',adj:'#c3e88d',prep:'#ffb347',pron:'#b18cff',conj:'#ffd166',punct:'#a49f92'};
const WORDCAT=new Map();
for(const c of CATS)for(const w of LEX[c])WORDCAT.set(w,c);
export function catOf(w){
  if(WORDCAT.has(w))return WORDCAT.get(w);
  if(/(ed|ing|s)$/.test(w)&&w.length>3)return 'verb';
  if(/(ly)$/.test(w))return 'adj';
  return 'noun';
}
export const VOCAB=[],VIDX=new Map();
for(const c of CATS)for(const w of LEX[c]){VIDX.set(w,VOCAB.length);VOCAB.push(w)}
function ensureWord(w){if(!VIDX.has(w)){VIDX.set(w,VOCAB.length);VOCAB.push(w)}}

const EMB=new Map();
export function emb(w){
  if(EMB.has(w))return EMB.get(w);
  const r=mulberry32(hash(w));
  const v=new Array(D).fill(0);
  for(let i=0;i<8;i++)v[i]=gaussOf(r)*0.03;              // dims 0..7  : near-silent — reserved for the position signal
  const ci=CATS.indexOf(catOf(w));
  for(let i=8;i<16;i++)v[i]=gaussOf(r)*0.35;             // dims 8..15 : category one-hot + word texture
  v[8+ci]+=1.2;
  // normalize so every token has the same magnitude — keeps LayerNorm's 1/σ
  // roughly constant across tokens, so attention scores compare cleanly
  const nv=Math.sqrt(v.reduce((a,b)=>a+b*b,0));
  for(let i=0;i<D;i++)v[i]*=1.35/nv;
  EMB.set(w,v);return v;
}
export function posVec(i){
  const v=new Array(D);
  for(let p=0;p<8;p++){const w=1/Math.pow(PE_BASE,p/8);v[2*p]=Math.sin(i*w)*PE_AMP;v[2*p+1]=Math.cos(i*w)*PE_AMP}
  return v;
}

/* ---------- weights (seeded; three heads crafted with real mechanisms) ---------- */
const grng=mulberry32(20240717);
function randmat(r,c,s){const M=zeros(r,c);for(let i=0;i<r;i++)for(let j=0;j<c;j++)M[i][j]=gaussOf(grng)*s;return M}
function addNoise(M,s){for(const row of M)for(let j=0;j<row.length;j++)row[j]+=gaussOf(grng)*s}

function buildWeights(){
  const layers=[];
  for(let L=0;L<NLAYER;L++){
    const Wq=[],Wk=[],Wv=[],relBias=[];
    for(let h=0;h<NHEAD;h++){
      if(L===0&&h===0){
        /* POSITION HEAD — a relative-position bias b[i-j] added to the scores,
           exactly the mechanism of T5's relative attention bias / ALiBi (BLOOM, MPT).
           Peaked at distance 1 → the head attends to the previous token. */
        Wq.push(randmat(DH,D,.22));Wk.push(randmat(DH,D,.22));
        relBias.push([0.6,2.8,0.5,0.1,-0.2,-0.5,-0.8,-1.1,-1.4,-1.7,-2.0,-2.3,-2.6,-2.9,-3.2,-3.5]);
      } else if(L===0&&h===1){
        /* CONTENT HEAD — Wq and Wk both read the category dims 8..15,
           so q·k is large when two tokens share a grammatical category */
        const q=zeros(DH,D),k=zeros(DH,D);
        for(let r2=0;r2<8;r2++){q[r2][8+r2]=1.5;k[r2][8+r2]=1.5}
        addNoise(q,.07);addNoise(k,.07);Wq.push(q);Wk.push(k);
        relBias.push(new Array(16).fill(0));
      } else if(L===1&&h===0){
        /* RECENCY HEAD — pure ALiBi: a linear penalty on distance, nothing else crafted */
        Wq.push(randmat(DH,D,.3));Wk.push(randmat(DH,D,.3));
        relBias.push(Array.from({length:16},(_,d2)=>-0.5*d2));
      } else {
        Wq.push(randmat(DH,D,.4));Wk.push(randmat(DH,D,.4));
        relBias.push(new Array(16).fill(0));
      }
      Wv.push(randmat(DH,D,.45));
    }
    layers.push({Wq,Wk,Wv,relBias,Wo:randmat(D,D,.3),
      W1:randmat(DFF,D,.35),b1:new Array(DFF).fill(0).map(()=>gaussOf(grng)*.05),
      W2:randmat(D,DFF,.25),b2:new Array(D).fill(0).map(()=>gaussOf(grng)*.05)});
  }
  return layers;
}
const WEIGHTS=buildWeights();
export function paramCount(){
  return VOCAB.length*D + (NLAYER*(3*DH*D*NHEAD + D*D + DFF*D+DFF + D*DFF+D));
}

/* ---------- grammar prior (disclosed, toggleable) ---------- */
const PRIOR={det:{noun:2.4,adj:1.1},noun:{verb:2.2,prep:1.1,punct:.8,conj:.5},
  verb:{det:1.9,prep:1.5,pron:.5,adj:.4},adj:{noun:2.4,punct:.5},
  prep:{det:2.4,pron:.7,noun:.7},pron:{verb:2.4},conj:{det:1.5,pron:1.2,noun:.5},
  punct:{det:1.7,pron:1.4,conj:.4}};
const BIGRAM={'sat':{on:2},'cat':{sat:1.6},'dream':{of:2.4},'dreams':{of:2.4},
  'of':{electric:1.8},'electric':{sheep:2.6},'opened':{the:1.6},'robots':{dream:2.4},
  'attention':{is:2},'is':{all:1.6},'all':{you:2.2},'you':{need:2.4}};

/* ---------- THE FORWARD PASS (everything cached) ---------- */
export function forward(tokens,ablate){
  tokens.forEach(ensureWord);
  const n=tokens.length;
  const embRows=tokens.map(t=>emb(t).slice());
  const posRows=tokens.map((_,i)=>posVec(i));
  let X=embRows.map((e,i)=>addv(e,posRows[i]));
  const checkpoints=[{name:'embedding',X:embRows.map(v=>v.slice())},{name:'+ position',X:X.map(v=>v.slice())}];
  const layers=[];
  for(let L=0;L<NLAYER;L++){
    const W=WEIGHTS[L];
    const A_in=X.map(layernorm);
    const heads=[];
    for(let h=0;h<NHEAD;h++){
      const q=A_in.map(a=>matvec(W.Wq[h],a));
      const k=A_in.map(a=>matvec(W.Wk[h],a));
      const v=A_in.map(a=>matvec(W.Wv[h],a));
      const raw=zeros(n,n),scaled=zeros(n,n),att=[];
      const bias=W.relBias[h];
      for(let i=0;i<n;i++){
        const row=[];
        for(let j=0;j<=i;j++){raw[i][j]=dot(q[i],k[j]);scaled[i][j]=raw[i][j]/Math.sqrt(DH)+bias[i-j];row.push(scaled[i][j])}
        const sm=softmax(row);
        att.push(sm.concat(new Array(n-i-1).fill(0)));
      }
      const z=[];
      for(let i=0;i<n;i++){
        const zi=new Array(DH).fill(0);
        for(let j=0;j<=i;j++)for(let d2=0;d2<DH;d2++)zi[d2]+=att[i][j]*v[j][d2];
        z.push(ablate[L][h]?zi.map(()=>0):zi);
      }
      heads.push({q,k,v,raw,scaled,bias,A:att,z});
    }
    const attnOut=[];
    for(let i=0;i<n;i++){
      const zcat=heads[0].z[i].concat(heads[1].z[i]);
      attnOut.push(matvec(W.Wo,zcat));
    }
    const X1=X.map((x,i)=>addv(x,attnOut[i]));
    checkpoints.push({name:`L${L+1} · after attention`,X:X1.map(v=>v.slice())});
    const B=X1.map(layernorm);
    const pre=B.map(b=>addv(matvec(W.W1,b),W.b1));
    const act=pre.map(p=>p.map(gelu));
    const mlpOut=act.map(a=>addv(matvec(W.W2,a),W.b2));
    const X2=X1.map((x,i)=>addv(x,mlpOut[i]));
    checkpoints.push({name:`L${L+1} · after MLP`,X:X2.map(v=>v.slice())});
    layers.push({A_in,heads,attnOut,X1,B,pre,act,mlpOut,X2});
    X=X2;
  }
  const final=X.map(layernorm);
  checkpoints.push({name:'final layernorm',X:final.map(v=>v.slice())});
  const rawLogits=VOCAB.map(w=>dot(final[n-1],emb(w)));
  return {tokens,n,embRows,posRows,checkpoints,layers,final,rawLogits};
}

/* logits -> distribution with options */
export function distribution(F,{prior=true,temp=1,topk=10}={}){
  const last=F.tokens[F.n-1],lastCat=catOf(last);
  const logits=F.rawLogits.map((l,vi)=>{
    let x=l;const w=VOCAB[vi];
    if(prior){
      x+=(PRIOR[lastCat]?.[catOf(w)]||0);
      x+=(BIGRAM[last]?.[w]||0)*1.2;
    }
    if(w===last||w===F.tokens[F.n-2])x-=2.5;   // repetition penalty
    return x;
  });
  const idx=[...logits.keys()].sort((a,b)=>logits[b]-logits[a]);
  const kept=idx.slice(0,topk);
  const probsAll=softmax(kept.map(i=>logits[i]),Math.max(.05,temp));
  return kept.map((vi,r)=>({word:VOCAB[vi],logit:logits[vi],raw:F.rawLogits[vi],p:probsAll[r]}));
}
export function lensTop(F,ckX,k=3){
  const v=layernorm(ckX[F.n-1]);
  const lg=VOCAB.map(w=>dot(v,emb(w)));
  const idx=[...lg.keys()].sort((a,b)=>lg[b]-lg[a]).slice(0,k);
  const ps=softmax(idx.map(i=>lg[i]));
  return idx.map((vi,r)=>({word:VOCAB[vi],p:ps[r]}));
}

export function tokenize(t){return (t.trim().toLowerCase().match(/[a-z']+|[.,!?;]/g)||['the']).slice(0,12)}

/* ---------- shared UI constants ---------- */
export const PALETTE=['#ffb347','#5ad1c8','#b18cff','#ff6b8a','#c3e88d','#ff8c42','#7aa2ff','#ffd166'];
export const HEADNAMES=[['POSITION HEAD','CONTENT HEAD'],['RECENCY (ALiBi)','UNTRAINED']];

export function lerpColor(a,b,t){
  const pa=[parseInt(a.slice(1,3),16),parseInt(a.slice(3,5),16),parseInt(a.slice(5,7),16)];
  const pb=[parseInt(b.slice(1,3),16),parseInt(b.slice(3,5),16),parseInt(b.slice(5,7),16)];
  return `rgb(${Math.round(pa[0]+(pb[0]-pa[0])*t)},${Math.round(pa[1]+(pb[1]-pa[1])*t)},${Math.round(pa[2]+(pb[2]-pa[2])*t)})`;
}
export function cellColor(v,scale=1.5){const t=Math.max(-1,Math.min(1,v/scale));return t>=0?lerpColor('#141620','#5ad1c8',t):lerpColor('#141620','#ff6b8a',-t)}
export function heatColor(v){return v<.5?lerpColor('#141620','#5a3d1a',v/.5):lerpColor('#5a3d1a','#ffb347',(v-.5)/.5)}
