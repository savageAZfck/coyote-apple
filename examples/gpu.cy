// gpu.cy — the third stratum: `gpu fn` lowers the same typed tensor
// language to a Metal kernel. Params, io, lets, reductions — identical
// source; a different backend. One thread per output element, weights
// resident in shared memory — unified memory means staging IS the buffer.
//
// An organism can mix backends in one file: `fn` → ANE, `gpu fn` → GPU,
// both callable as first-class `ane(...) -> ...` values from host code.

param w1: f16[64, 64]
param w2: f16[64, 64]
param wq: f16[64, 64]
param wk: f16[64, 64]
param rw: f16[64]
param fb: f16[64]

state kv {
    k: f16[8, 64],
    v: f16[8, 64],
}

// a real transformer block on the GPU: norm → attend → FFN
gpu fn block(x: f16[64], pos: f16[8]) -> f16[64] {
    let q = x @ wq;
    let scores = softmax(q @ kv.k.T + pos * 0.125);
    let att = scores @ kv.v;
    rmsnorm(x + att, rw)
}

gpu fn ffn(h: f16[64]) -> f16[64] {
    gelu(h @ w1) @ w2 + fb
}

host fn main() -> i64 {
    let x = zeros(64);
    let p = zeros(8);
    let h = block(x, p);
    // gpu programs are first-class — pass them like ane fns
    let f = ffn;
    let y = f(h);
    print("block argmax", argmax(h));
    print("ffn   argmax", argmax(y));
    witness("gpu-block", argmax(y));
    return 0;
}
