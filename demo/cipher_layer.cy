// cipher_layer.cy — the fourth substrate.
// Same program shape as any fn: a linear layer over an input.
// `enc fn` encrypts the input at exec time, evaluates the matvec
// homomorphically against plaintext weights, and decrypts on the way
// back — the client never sees the weights, the server never sees
// the data, and the oracle still proves it's the same math.

param w: f16[8, 8]

enc fn project(x: f16[8]) -> f16[8] {
    x @ w
}

host fn main() -> i64 {
    let x = onehot(0, 8);
    let y = project(x);
    print("cipher out", y);
    return 0;
}
