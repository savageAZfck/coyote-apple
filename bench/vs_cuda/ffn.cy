io x: in f16[256]
io o: out f16[256]
param w1: f16[256, 1024]
param w2: f16[1024, 256]
param w3: f16[256, 1024]

gpu fn main(x: f16[256]) -> f16[256] {
    let h = silu(x @ w1) * (x @ w3);
    h @ w2
}
