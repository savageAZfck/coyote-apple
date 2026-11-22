// revdemo — the whole reversible-computing claim in one file.
struct Pair { x: i64, y: i64 }

rev host fn arx8(st: Pair, key: i64) -> Pair {
    var x = st.x;
    var y = st.y;
    let k0 = bitxor(key, 2654435769);
    let k1 = bitxor(k0, rotl(key, 11));
    let k2 = bitxor(k1, shr(key, 13));
    let k3 = bitxor(k2, rotl(k0, 17));
    let k4 = bitxor(k3, shr(k1, 19));
    let k5 = bitxor(k4, rotl(k2, 23));
    let k6 = bitxor(k5, shr(k3, 29));
    let k7 = bitxor(k6, rotl(k4, 31));
    x += y; x ^= rotl(y, 7);  x ^= k0;
    y += x; y ^= ror(x, 11); y ^= k1;
    x += y; x ^= rotl(y, 13); x ^= k2;
    y += x; y ^= ror(x, 17); y ^= k3;
    x += y; x ^= rotl(y, 19); x ^= k4;
    y += x; y ^= ror(x, 23); y ^= k5;
    x += y; x ^= rotl(y, 29); x ^= k6;
    y += x; y ^= ror(x, 31); y ^= k7;
    return Pair { x: x, y: y };
}

// a tensor rev fn — same primitive, ANE stratum
rev fn step(st: f16[4], k: f16[4], j: f16[4]) -> f16[4] {
    var acc = st;
    acc ^= k;
    acc ^= j;
    acc ^= k;
    acc
}

host fn main() -> i64 {
    let msg = Pair { x: 1234567890123456789, y: -9876543210987 };
    let key = 777666555444333 % 9223372036854775807;
    let ct = arx8(msg, key);
    let pt = uncall arx8(ct, key);
    print("PLAINTEXT =", msg.x, msg.y);
    print("CIPHERTEXT=", ct.x, ct.y);
    print("RECOVERED =", pt.x, pt.y);
    var ok = 0;
    if pt.x == msg.x && pt.y == msg.y { ok = 1; }
    print("roundtrip =", ok);
    return ok;
}
