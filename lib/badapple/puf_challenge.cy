// puf_challenge.cy — embodiment probe for the organism's silicon body.
//
// A fixed, deterministic tensor workload declared twice: `fn` lowers to
// the ANE, `gpu fn` lowers to Metal. Same math, same shapes, every run —
// so the only thing that varies between probes is *which physical chip*
// executed it. `coyote-apple prof` / `bench` timing lanes on this file
// are the physical-unclonable-function signal: semiconductor process
// variation makes per-device dispatch latency a fingerprint.
//
// Workload is deliberately boring — a matmul chain — because a PUF
// challenge wants repeatability, not cleverness.

param w1: f16[64, 64]
param w2: f16[64, 64]
param w3: f16[64, 64]
param w4: f16[64, 64]

fn probe_ane(x: f16[64]) -> f16[64] {
    ((x @ w1) @ w2) @ w3 @ w4
}

gpu fn probe_gpu(x: f16[64]) -> f16[64] {
    ((x @ w1) @ w2) @ w3 @ w4
}

host fn main() -> i64 {
    let x = zeros(64);
    let a = probe_ane(x);
    let g = probe_gpu(x);
    print("ane argmax", argmax(a));
    print("gpu argmax", argmax(g));
    witness("puf", argmax(a));
    return 0;
}
