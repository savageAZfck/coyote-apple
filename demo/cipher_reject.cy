// cipher_reject.cy — the honest constraint.
// Leveled schemes do linear ops without a bootstrap; the compiler
// refuses nonlinear ops at emit — a circuit that can't be honest
// never gets built.

enc fn bad(x: f16[8]) -> f16[8] {
    softmax(x)
}

host fn main() -> i64 {
    let x = onehot(0, 8);
    let y = bad(x);
    return 0;
}
