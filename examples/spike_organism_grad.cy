// spike_organism_grad.cy — does grad/adam compile inside an organism?
// host main calls a compiled fn that trains a trainable param on ANE.

invariant never_sells_self

provenance {
    author: "savage"
    purpose: "spike — trainable param inside host-driven organism"
}

param w: trainable f16[64, 64]

fn learn(x: f16[64], t: f16[64]) -> f16[64] {
    adam(w, grad(mse(x @ w, t), w), 0.05);
    x @ w
}

host fn main() -> i64 {
    let me = self_token();
    witness("born", me);
    var step = 0;
    while step < 6 {
        let x = onehot(0, 64);
        let t = onehot(1, 64);
        let y = learn(x, t);
        print("step", step, "pred", argmax(y), "y1", i64(y[1] * 1000.0));
        step += 1;
    }
    witness("done", step);
    return step;
}
