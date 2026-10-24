io x: in f16[4, 8]
io y: out f16[4, 8]
gpu fn main(x: f16[4,8]) -> f16[4,8] {
    rsoftmax(x)
}
