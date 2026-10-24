param w: trainable f32[8, 8]
io x: in f16[8]
io t: in f16[8]
io y: out f16[8]
fn loss(x: f16[8], t: f16[8], w: f32[8, 8]) -> f32 {
    mse(x @ w, t)
}
gpu fn main(x: f16[8], t: f16[8]) -> f16[8] {
    w = w * 0.5;
    w += w * 0.1;
    let s = x @ w;
    w -= grad(loss(x, t, w), w) * 0.1;
    s
}
