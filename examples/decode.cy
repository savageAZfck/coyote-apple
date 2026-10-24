// decode.cy — kv-cache inference, the organism form.
//
// `state kv` fields live resident on the GPU; each `step` dispatch
// scatters the new k/v rows in at the token's position (one-hot
// `pos`), then attends over the whole cache. Between calls the VM
// reads state_outs back into the host store and restages them — the
// cache persists across dispatches exactly like eval's env insert.
//
//   coyote-apple run examples/decode.cy
//   coyote-apple oracle examples/decode.cy   # cpu↔gpu parity

param wq: f16[16, 16]
param wk: f16[16, 16]
param wv: f16[16, 16]
param wo: f16[16, 16]

state kv {
    k: f16[8, 16],
    v: f16[8, 16],
}

gpu fn step(x: f16[16], pos: f16[8]) -> f16[16] {
    kv.k[pos] = x @ wk;
    kv.v[pos] = x @ wv;
    let q = x @ wq;
    let s = q @ kv.k.T;
    (softmax(s) @ kv.v) @ wo
}

host fn main() -> i64 {
    // three token steps — embeddings from rope tables, positions
    // as one-hot scatters.
    let y1 = step(rope_cos(0, 16), onehot(0, 8));
    let y2 = step(rope_cos(1, 16), onehot(1, 8));
    let y3 = step(rope_cos(2, 16), onehot(2, 8));
    print("tok0", argmax(y1));
    print("tok1", argmax(y2));
    print("tok2", argmax(y3));
    witness("decode", argmax(y3));
    return 0;
}
