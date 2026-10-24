io x: in f16[8]
io y: out f16[8]
gpu fn main(x: f16[8]) -> f16[8] {
    let a = x + 1.0;
    a = a * 2.0;
    a += a - x;
    a
}
