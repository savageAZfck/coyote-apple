io a: in f16[512, 512]
io b: in f16[512, 512]
io o: out f16[512, 512]

gpu fn main(a: f16[512,512], b: f16[512,512]) -> f16[512,512] {
    a @ b
}
