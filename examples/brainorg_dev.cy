// brainorg.cy — a brain written, not downloaded.
//
// A single decoder layer with its own dimensions — d=64, four query
// heads (RoPE lo/hi), two GQA kv heads, a 2-branch SwiGLU FFN (128
// intermediate channels), and a 1024-way trainable logit head — every
// parameter born on-chip via init_weight. No foreign weights in the
// compute path. The only borrowed surface is the vocabulary contract —
// a 64-wide slice of the embed table, gathered host-side like any
// token interface, so the brain speaks a real token space.
//
// 28 params train in one `train_step` (8 Q + 8 O + 6 FFN + 1 LMH + 5
// norms), sized on the ANE activation×activation skeleton table
// ((t,d) ∈ {(64,64), (256,64), (1024,64)}); 10 are frozen at a
// structural boundary: the k/v projections (and their norms) reach
// the loss only through the KV state cache.
//
// The host stratum drives both halves of its life:
//   1. curriculum — teacher-forced next-token training over a real
//      sequence of embedding rows (`seq`), each step witnessed;
//   2. generation — an autoregressive decode on the trained weights,
//      gather → silicon → argmax → witness → feed back.
//
//   coyote-apple run examples/brainorg.cy -o out/brainorg
//
// needs models/qwen3-0.6b/model.safetensors (vocabulary contract)
// and corpus.safetensors (`seq` — real embedding rows).

invariant never_sells_self

provenance {
    author: "savage"
    purpose: "organism-grown mind — trainable decoder on ANE"
    license: "BSL-1.1"
}

// ---- norms ----
param ln1: trainable f16[64]
param ln2: trainable f16[64]
param lnf: trainable f16[64]
param qnl: trainable f16[64]
param qnh: trainable f16[64]
param knl: frozen f16[64]
param knh: frozen f16[64]

// ---- attention: 4 q heads (lo/hi halves, rope-rotated), 2 kv heads ----
param q0lo: trainable f16[64, 64]
param q0hi: trainable f16[64, 64]
param q1lo: trainable f16[64, 64]
param q1hi: trainable f16[64, 64]
param q2lo: trainable f16[64, 64]
param q2hi: trainable f16[64, 64]
param q3lo: trainable f16[64, 64]
param q3hi: trainable f16[64, 64]
param o0lo: trainable f16[64, 64]
param o0hi: trainable f16[64, 64]
param o1lo: trainable f16[64, 64]
param o1hi: trainable f16[64, 64]
param o2lo: trainable f16[64, 64]
param o2hi: trainable f16[64, 64]
param o3lo: trainable f16[64, 64]
param o3hi: trainable f16[64, 64]
// kv projections + their norms — frozen: the autodiff tape ends at
// the state cache. Born on-chip, read every forward, never updated.
param k0lo: frozen f16[64, 64]
param k0hi: frozen f16[64, 64]
param v0lo: frozen f16[64, 64]
param v0hi: frozen f16[64, 64]
param k1lo: frozen f16[64, 64]
param k1hi: frozen f16[64, 64]
param v1lo: frozen f16[64, 64]
param v1hi: frozen f16[64, 64]

// ---- 2-branch SwiGLU FFN: 2 × (64 → 64 → 64) = 128 intermediate ----
param wg0: trainable f16[64, 64]
param wu0: trainable f16[64, 64]
param wd0: trainable f16[64, 64]
param wg1: trainable f16[64, 64]
param wu1: trainable f16[64, 64]
param wd1: trainable f16[64, 64]

// ---- logit head over the top-1024 slice of vocab ----
param lmh: trainable f16[64, 64]

state kv {
    k0lo: f16[64, 64]
    k0hi: f16[64, 64]
    v0lo: f16[64, 64]
    v0hi: f16[64, 64]
    k1lo: f16[64, 64]
    k1hi: f16[64, 64]
    v1lo: f16[64, 64]
    v1hi: f16[64, 64]
}

// the embedding table is host data — it never touches silicon.
// the brain's vocabulary contract: first-64 slice of a real
// token manifold. rows are gathered by index at runtime.
table emb: f16[151936, 64] @ "models/qwen3-0.6b/model.safetensors::model.embed_tokens.weight[0:151936,0:64]"

// the curriculum: 64 real embedding rows — a sequence cut from a
// real checkpoint, teacher-forced position-by-position.
table seq: f16[64, 64] @ "corpus.safetensors::seq"

fn forward(
    x: f16[64], pos: f16[64], mask: f16[64],
    cos: f16[64], sin: f16[64]
) -> f16[64] {
    let h = rmsnorm(x, ln1);
    // query heads — lo/hi halves, rope-rotated
    let q0l = rmsnorm(h @ q0lo, qnl);
    let q0h = rmsnorm(h @ q0hi, qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ q1lo, qnl);
    let q1h = rmsnorm(h @ q1hi, qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ q2lo, qnl);
    let q2h = rmsnorm(h @ q2hi, qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ q3lo, qnl);
    let q3h = rmsnorm(h @ q3hi, qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    // kv head 0 — shared by q heads 0,1
    let k0l = rmsnorm(h @ k0lo, knl);
    let k0h = rmsnorm(h @ k0hi, knh);
    let v0l = h @ v0lo;
    let v0h = h @ v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    // kv head 1 — shared by q heads 2,3
    let k1l = rmsnorm(h @ k1lo, knl);
    let k1h = rmsnorm(h @ k1hi, knh);
    let v1l = h @ v1lo;
    let v1h = h @ v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    // attend — grouped: q0,q1 over kv0; q2,q3 over kv1
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let a0 = (s0 @ kv.v0lo) @ o0lo + (s0 @ kv.v0hi) @ o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let a1 = (s1 @ kv.v0lo) @ o1lo + (s1 @ kv.v0hi) @ o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let a2 = (s2 @ kv.v1lo) @ o2lo + (s2 @ kv.v1hi) @ o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let a3 = (s3 @ kv.v1lo) @ o3lo + (s3 @ kv.v1hi) @ o3hi;
    let att = a0 + a1 + a2 + a3;
    let r = x + att;
    let n = rmsnorm(r, ln2);
    let f0 = (silu(n @ wg0) * (n @ wu0)) @ wd0;
    let f1 = (silu(n @ wg1) * (n @ wu1)) @ wd1;
    let f = r + f0 + f1;
    rmsnorm(f, lnf) @ lmh
}


// teacher-forced step: full forward, then adam on every trainable param —
// forward, backward, and the update all live in the same binary.
fn train_step(
    x: f16[64], pos: f16[64], mask: f16[64],
    cos: f16[64], sin: f16[64], tgt: f16[64]
) -> f16[64] {
    adam(q0lo, grad(mse(forward(x, pos, mask, cos, sin), tgt), q0lo), 0.05);
    adam(q0hi, grad(mse(forward(x, pos, mask, cos, sin), tgt), q0hi), 0.05);
    adam(q1lo, grad(mse(forward(x, pos, mask, cos, sin), tgt), q1lo), 0.05);
    adam(q1hi, grad(mse(forward(x, pos, mask, cos, sin), tgt), q1hi), 0.05);
    adam(q2lo, grad(mse(forward(x, pos, mask, cos, sin), tgt), q2lo), 0.05);
    adam(q2hi, grad(mse(forward(x, pos, mask, cos, sin), tgt), q2hi), 0.05);
    adam(q3lo, grad(mse(forward(x, pos, mask, cos, sin), tgt), q3lo), 0.05);
    adam(q3hi, grad(mse(forward(x, pos, mask, cos, sin), tgt), q3hi), 0.05);
    adam(o0lo, grad(mse(forward(x, pos, mask, cos, sin), tgt), o0lo), 0.05);
    adam(o0hi, grad(mse(forward(x, pos, mask, cos, sin), tgt), o0hi), 0.05);
    adam(o1lo, grad(mse(forward(x, pos, mask, cos, sin), tgt), o1lo), 0.05);
    adam(o1hi, grad(mse(forward(x, pos, mask, cos, sin), tgt), o1hi), 0.05);
    adam(o2lo, grad(mse(forward(x, pos, mask, cos, sin), tgt), o2lo), 0.05);
    adam(o2hi, grad(mse(forward(x, pos, mask, cos, sin), tgt), o2hi), 0.05);
    adam(o3lo, grad(mse(forward(x, pos, mask, cos, sin), tgt), o3lo), 0.05);
    adam(o3hi, grad(mse(forward(x, pos, mask, cos, sin), tgt), o3hi), 0.05);
    adam(wg0, grad(mse(forward(x, pos, mask, cos, sin), tgt), wg0), 0.05);
    adam(wu0, grad(mse(forward(x, pos, mask, cos, sin), tgt), wu0), 0.05);
    adam(wd0, grad(mse(forward(x, pos, mask, cos, sin), tgt), wd0), 0.05);
    adam(wg1, grad(mse(forward(x, pos, mask, cos, sin), tgt), wg1), 0.05);
    adam(wu1, grad(mse(forward(x, pos, mask, cos, sin), tgt), wu1), 0.05);
    adam(wd1, grad(mse(forward(x, pos, mask, cos, sin), tgt), wd1), 0.05);
    adam(lmh, grad(mse(forward(x, pos, mask, cos, sin), tgt), lmh), 0.05);
    adam(ln1, grad(mse(forward(x, pos, mask, cos, sin), tgt), ln1), 0.01);
    adam(ln2, grad(mse(forward(x, pos, mask, cos, sin), tgt), ln2), 0.01);
    adam(lnf, grad(mse(forward(x, pos, mask, cos, sin), tgt), lnf), 0.01);
    adam(qnl, grad(mse(forward(x, pos, mask, cos, sin), tgt), qnl), 0.01);
    adam(qnh, grad(mse(forward(x, pos, mask, cos, sin), tgt), qnh), 0.01);
    forward(x, pos, mask, cos, sin)
}

host fn main() -> i64 {
    let me = self_token();
    witness("born", me);
    print("brain organism", me.id);

    // phase 1 — curriculum: 4 epochs over a 4-token chain (0 → 1 → 2 → 3 → 4)
    // from the real embedding manifold.
    var ep = 0;
    while ep < 60 {
        var i = 0;
        while i < 16 {
            let x = seq[i];
            let tgt = onehot(i + 1, 64);
            let lg = train_step(
                x,
                onehot(i, 64),
                causal(i, 64),
                rope_cos(i, 64),
                rope_sin(i, 64),
                tgt);
            let pred = argmax(lg);
            print("ep", ep, "pos", i, "pred", pred, "want", i + 1);
            witness("train_pred", pred);
            i += 1;
        }
        ep += 1;
    }

    // phase 2 — generation: seed with token 0 and decode autoregressively
    // on the weights it just grew.
    var tok = 0;
    var x = seq[tok];
    var step = 0;
    while step < 4 {
        let lg = forward(
            x,
            onehot(step, 64),
            causal(step, 64),
            rope_cos(step, 64),
            rope_sin(step, 64));
        tok = argmax(lg);
        print("gen", step, "tok", tok, "want", step + 1);
        witness("tok", tok);
        x = seq[tok];
        step += 1;
    }
    witness("done", step);
    return tok;
}
