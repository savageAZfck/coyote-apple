// std/lora — masked-LoRA helpers. `import "std/lora"` (CYO_PATH ships
// coyote-apple's lib/ automatically).
//
// The skeleton table wants dim ≥ 64 on adapter matmuls, so rank lives
// in data: `sel` (from `coyote-apple genmask sel:<r>:64`) zeroes the
// bottleneck coordinates ≥ r and the dead weights receive exactly
// zero gradient. Concrete-64 for the public emitter table.

fn lora_adapt(
    x: f16[64], w: f16[64, 64], la: f16[64, 64],
    lb: f16[64, 64], sel: f16[64]
) -> f16[64] {
    (x @ w) + (relu((x @ la) * sel) @ lb)
}

fn lora_loss(
    x: f16[64], t: f16[64], w: f16[64, 64],
    la: f16[64, 64], lb: f16[64, 64], sel: f16[64]
) -> f32 {
    mse(lora_adapt(x, w, la, lb, sel), t)
}
