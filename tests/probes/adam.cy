param w1: trainable f16[8, 8]
io x: in f16[8]
io t: in f16[8]
io y: out f16[8]
fn adapt(x: f16[8], w1: f16[8, 8]) -> f16[8] {
    x @ w1
}
fn loss(x: f16[8], t: f16[8], w1: f16[8, 8]) -> f32 {
    mse(adapt(x, w1), t)
}
gpu fn main(x: f16[8], t: f16[8]) -> f16[8] {
    adam(w1, grad(loss(x, t, w1), w1), 0.01);
    adapt(x, w1)
}
