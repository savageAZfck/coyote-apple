// std/nn — neural building blocks, the library layer. `import "std/nn"`
// resolves through CYO_PATH (coyote-apple ships it automatically).
//
// Same convention as std/lora: weights arrive as *args* — the
// importing file owns the params, these fns compose the math.
// Generic dims inline into the caller's program; concrete-64 helpers
// keep the public emitter's mat constraints.
//
// Dual-backend by construction: every helper is plain tensor math
// (matmul / elementwise / softmax / rmsnorm / gather), so a program
// built on std/nn compiles on `fn` (ane) and `gpu fn` (metal) alike —
// with one capability split: the ane emitter requires matmul's rhs to
// be a `param`, so `attn`/`attn_v`/`embed_row` (activation rhs) are
// gpu-only today, while `ffn_*`/`mlp`/`res_norm` run on both.

// ---- attention ----

// Single-head attention over an [L, D] activation. `q` must arrive
// pre-scaled by 1/sqrt(d) — the language has no const-dim arithmetic,
// so the scale lives in the caller's weight init. `rsoftmax`
// normalizes each query row over the keys — the flat `softmax`
// would mix probability mass across rows.
fn attn<L>(q: f16[L, 64], k: f16[L, 64], v: f16[L, 64]) -> f16[L, 64] {
    rsoftmax(q @ k.T) @ v
}

// Vector query variant — scores are a length-L distribution over
// positions; out is the [D] mixture. Same pre-scaled-q convention.
fn attn_v<L>(q: f16[64], k: f16[L, 64], v: f16[L, 64]) -> f16[64] {
    softmax(q @ k.T) @ v
}

// ---- feed-forwards (concrete 64→256→64) ----

// SwiGLU FFN — the modern transformer MLP: (silu(x·w1) ⊙ (x·w3)) · w2.
fn ffn_swiglu(
    x: f16[64],
    w1: f16[64, 256],
    w2: f16[256, 64],
    w3: f16[64, 256]
) -> f16[64] {
    (silu(x @ w1) * (x @ w3)) @ w2
}

// Classic GELU FFN: gelu(x·w1) · w2.
fn ffn_gelu(x: f16[64], w1: f16[64, 256], w2: f16[256, 64]) -> f16[64] {
    gelu(x @ w1) @ w2
}

// Plain two-layer MLP with relu.
fn mlp(x: f16[64], w1: f16[64, 256], w2: f16[256, 64]) -> f16[64] {
    relu(x @ w1) @ w2
}

// ---- block glue ----

// Pre-norm residual: rmsnorm(x + sub, w). The transformer "add & norm"
// in one call.
fn res_norm(x: f16[64], sub: f16[64], w: f16[64]) -> f16[64] {
    rmsnorm(x + sub, w)
}

// ---- embedding ----

// Row-select: `one` is a length-V one-hot (or mixture), `mat` is
// [V, D] — returns the [D] row. `embed` lowers to the one-hot matmul.
fn embed_row<V>(one: f16[V], mat: f16[V, 64]) -> f16[64] {
    embed(mat, one)
}
