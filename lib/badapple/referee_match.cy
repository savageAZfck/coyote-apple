// referee_match.cy — an N-step match between two provers on the referee's clock.
//
// `fn` lowers to the Neural Engine, `gpu fn` lowers to Metal. The host
// evaluates every step on the CPU and reads each prover's ballot
// (argmax over its returned tensor). One source, three silicons, eight
// committed rounds — the substrate a refereed-verification dispute
// bisects when two provers disagree.
//
//   coyote-apple run coyote/referee_match.cy
//   coyote-apple run coyote/referee_match.cy --cpu

param wq: f16[64, 64]

fn step_ane(x: f16[64]) -> f16[64] {
    softmax(x @ wq * 0.125) + x
}

gpu fn step_gpu(x: f16[64]) -> f16[64] {
    softmax(x @ wq * 0.125) + x
}

host fn main() -> i64 {
    var x = zeros(64);
    var i = 0;
    for i in 0..8 {
        let ya = step_ane(x);
        let yg = step_gpu(x);
        print("referee_step", i, "ane", argmax(ya), "gpu", argmax(yg));
        witness("step", argmax(ya));
        x = ya;
    }
    return 0;
}
