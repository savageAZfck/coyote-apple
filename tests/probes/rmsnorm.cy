param w: f16[64]
io x: in f16[64]
io y: out f16[64]
gpu fn main(x: f16[64]) -> f16[64] {
    rmsnorm(x, w)
}
