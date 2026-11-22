struct Pair { x: i64, y: i64 }

// self-mixing ARX — impossible under single-var v1: x's updates read
// y and y's updates read x. Per-var Janus rule + Bennett ordering
// make it invertible; the struct return carries BOTH final values
// so the inverse can re-seed both threads.
rev host fn arx(st: Pair, k: i64) -> Pair {
    var x = st.x;
    var y = st.y;
    x += y;
    x ^= rotl(y, 7);
    y += bitxor(x, k);
    y ^= shr(x, 11);
    x += bitxor(y, rotl(k, 13));
    y ^= rotl(x, 19);
    return Pair { x: x, y: y };
}
host fn main() -> i64 {
    let p = Pair { x: 123456789, y: -987654321 };
    let k = 555444333;
    let c = arx(p, k);
    let m = uncall arx(c, k);
    print("cipher  =", c.x, c.y);
    print("plain   =", m.x, m.y);
    var ok = 0;
    if m.x == p.x && m.y == p.y { ok = 1; }
    print("arx roundtrip ok =", ok);
    return ok;
}
