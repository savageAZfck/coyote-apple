io q: in f16[64, 64]
io k: in f16[64, 64]
io v: in f16[64, 64]
io o: out f16[64, 64]

gpu fn main(q: f16[64,64], k: f16[64,64], v: f16[64,64]) -> f16[64,64] {
    (q @ k.T) @ v
}
