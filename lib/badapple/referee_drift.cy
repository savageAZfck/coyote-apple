// referee_drift.cy — a computation engineered to die in fp16.
//
// Each step multiplies the state by 0.001. In f64 the trajectory stays
// alive for ~250 iterations; in fp16 the state collapses to exact zeros
// within ~4 steps and argmax falls to index 0. Same source, same ops —
// the Neural Engine physically loses precision the CPU evaluator does
// not. This is a REAL substrate divergence the referee must isolate:
// not forged, not simulated — silicon numerics.
//
//   badapple-referee dispute coyote/referee_drift.cy

param wq: f16[64, 64]

fn decay_ane(x: f16[64]) -> f16[64] {
    x @ wq * 0.001
}

host fn main() -> i64 {
    var x = onehot(0, 64);
    var i = 0;
    for i in 0..12 {
        x = decay_ane(x);
        witness("u", argmax(x));
        print("referee_step", i, "ane", argmax(x));
    }
    return 0;
}
