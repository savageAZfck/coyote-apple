// seal — keyed reversible permutation on i64.
// v1 single-state-thread: deltas come from key-derived lets, so this
// is an ARX-*op* permutation (add/rotate/xor), not yet self-mixing
// ARX — that needs multi-var state (Phase 5+). Every step is a
// bijection in `v` for fixed ctx, so `uncall seal` is the exact
// inverse by construction.
rev host fn seal(st: i64, key: i64, nonce: i64) -> i64 {
    var v = st;
    // key schedule — pure lets over ctx args (no state capture)
    let k0 = bitxor(key, nonce);
    let k1 = bitxor(k0, rotl(nonce, 13));
    let k2 = bitxor(k1, shr(key, 7));
    let k3 = bitxor(k2, rotl(key, 29));
    let k4 = bitxor(k3, shr(nonce, 17));
    let k5 = bitxor(k4, rotl(k0, 5));
    let k6 = bitxor(k5, shr(k1, 23));
    let k7 = bitxor(k6, rotl(k2, 41));
    // 8 ARX rounds — xor subkey, add rotated subkey, xor shifted subkey
    v ^= k0;  v += rotl(k1, 7);   v ^= shr(k2, 11);
    v ^= k3;  v += rotl(k4, 17);  v ^= shr(k5, 19);
    v ^= k6;  v += rotl(k7, 23);  v ^= shr(k0, 29);
    v ^= k1;  v += rotl(k2, 31);  v ^= shr(k3, 3);
    v ^= k4;  v += rotl(k5, 37);  v ^= shr(k6, 43);
    v ^= k7;  v += rotl(k0, 47);  v ^= shr(k1, 53);
    v ^= k2;  v += rotl(k3, 59);  v ^= shr(k4, 61);
    v ^= k5;  v += rotl(k6, 11);  v ^= shr(k7, 13);
    return v;
}
host fn main() -> i64 {
    let msg = 1234567890123456789;
    let key = 9876543210987654321 % 9223372036854775807;
    let nonce = 555444333222111;
    let ct = seal(msg, key, nonce);
    let pt = uncall seal(ct, key, nonce);
    print("sealed =", ct);
    print("opened =", pt);
    var ok = 0;
    if pt == msg { ok = 1; }
    print("seal roundtrip ok =", ok);
    return ok;
}
