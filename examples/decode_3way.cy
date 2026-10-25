// decode_3way.cy — one decode step, three backends, one source.
//
// The same attention decode step is declared twice: `fn` lowers to
// the ANE through mil_infer/Espresso, `gpu fn` lowers to Metal. The
// CPU evaluator is the semantic reference every run is diffed
// against. Shared params, shared state shape, identical math —
// `coyote-apple prof` times all three.
//
// D=64, T=256: the ANE native channel width (per-head dim) with a
// 256-deep cache — the shape the hwx skeleton table covers.
//
//   coyote-apple prof examples/decode_3way.cy --runs 50
//   coyote-apple run  examples/decode_3way.cy

param wq: f16[64, 64]
param wk: f16[64, 64]
param wv: f16[64, 64]
param wo: f16[64, 64]

state kv {
    k: f16[256, 64]
    v: f16[256, 64]
}

fn layer_ane(x: f16[64], pos: f16[256], mask: f16[256]) -> f16[64] {
    let q = x @ wq;
    kv.k[pos] = x @ wk;
    kv.v[pos] = x @ wv;
    let s = softmax(q @ kv.k.T * 0.125 + mask);
    let a = s @ kv.v @ wo;
    x + a
}

gpu fn layer_gpu(x: f16[64], pos: f16[256], mask: f16[256]) -> f16[64] {
    let q = x @ wq;
    kv.k[pos] = x @ wk;
    kv.v[pos] = x @ wv;
    let s = softmax(q @ kv.k.T * 0.125 + mask);
    let a = s @ kv.v @ wo;
    x + a
}

host fn main() -> i64 {
    let x = zeros(64);
    let pos = onehot(0, 256);
    let mask = zeros(256);
    let ya = layer_ane(x, pos, mask);
    let yg = layer_gpu(x, pos, mask);
    print("ane argmax", argmax(ya));
    print("gpu argmax", argmax(yg));
    witness("3way", argmax(ya));
    return 0;
}
