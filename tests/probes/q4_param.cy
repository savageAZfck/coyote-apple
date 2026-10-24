param w: q4[8, 8]
io x: in f16[8]
io y: out f16[8]
gpu fn main(x: f16[8]) -> f16[8] {
    x @ w
}
