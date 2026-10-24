param w: trainable f16[8, 8]
io x: in f16[4, 8]
io y: out f16[4, 8]
gpu fn main(x: f16[4,8]) -> f16[4,8] {
    w -= 0.1 * grad(mse(rsoftmax(x @ w), x), w)
    rsoftmax(x @ w)
}
