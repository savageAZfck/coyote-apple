state kv {
    k: f16[8, 8]
}
io x: in f16[8, 8]
io pos: in f16[8]
io y: out f16[8]
gpu fn main(x: f16[8, 8], pos: f16[8]) -> f16[8] {
    kv.k += x;
    kv.k[pos] = x[0];
    kv.k[0]
}
