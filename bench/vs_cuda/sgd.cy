io x: in f16[64]
io y: in f16[64]
io o: out f16[64]
param w1: trainable f16[64, 256]
param w2: trainable f16[256, 64]

gpu fn main(x: f16[64], y: f16[64]) -> f16[64] {
    w1 -= 0.001 * grad(mse(gelu(x @ w1) @ w2, y), w1);
    w2 -= 0.001 * grad(mse(gelu(x @ w1) @ w2, y), w2);
    gelu(x @ w1) @ w2
}
