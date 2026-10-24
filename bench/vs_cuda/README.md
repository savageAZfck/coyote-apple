# Coyote vs CUDA — matched-workload head-to-head

Same math, same init bytes, same dispatch granularity on both sides.
Coyote numbers measured on this machine (Apple Silicon, Metal,
`coyote-apple prof --runs 100`, GPU-side kernel timestamps via
`GPUStartTime/GPUEndTime`). CUDA numbers: **not run — no NVIDIA GPU
on this machine.** `run_cuda.sh` is ready for any CUDA host.

## Workloads

| file | math |
|---|---|
| `gemm` | C = A·B^T, 512³ — Coyote's `[N,K]` rhs convention replicated in CUDA (`b[n*K+k]`) |
| `attn` | softmax(Q·K^T)·V, L=64 D=64 |
| `ffn`  | swiglu FFN 256→1024→256 |
| `sgd`  | 2-layer gelu MLP, MSE, both weights updated per step — Coyote: 4 lines + `grad`; CUDA: ~100 lines of hand-written backward |

Coyote init is replicated exactly in `init.h` (xorshift64, same name
seeds) — both sides compute on identical bytes.

## Coyote results (measured)

| workload | device ms (min) | parity vs eval | kernels emitted |
|---|---|---|---|
| gemm 512³ | 0.53–0.75 | 0.000003 | 1 (tiled GEMM) |
| attn 64×64 | 0.079 | 0.000000 | 4 (matmul, fused softmax, matmul, out) |
| ffn 256→1024 | 0.10–0.18 | 0.000020 | 3 (auto-materialized) |
| sgd step | 0.050–0.065 | 0.000002 | 4 (fwd scratch, 2 updates, out) |

Device time is kernel-only (GPU timestamps). This machine's GPU is
shared — readings swing ~3× under contention, so mins are the honest
kernel numbers and ranges reflect observed bests across sessions.

Emitter notes:

- `gemm`: cooperative-tile matmul — 32×32 blocks, 8×8 threadgroups,
  4×4 register micro-tile per thread, threadgroup-staged A/B slices,
  +1 stride padding against bank conflicts. ~3–4× over the naive
  one-thread-per-element kernel on this GPU. `CYO_SIMD=1` swaps in a
  `simdgroup_matrix` variant — measured slower on this silicon
  (emulated ops), kept for matrix-hardware machines.
- `attn`: `softmax` is a fused single-threadgroup kernel — parallel
  tree-max, parallel exp into scratch, serial sum in eval's exact
  accumulation order, parallel normalize. The order-sensitive part
  stays serial on purpose: it's what keeps parity at 0.000000.
- `sgd`: `grad` lowers reverse-mode to Metal; `w -= lr·grad` emits
  per-param update kernels in statement order. `adam` rides along
  the same path (persistent moment buffers, post-kernel apply).
- `CYO_HALF=1` stores `f16` params as `half` buffers (eval already
  f16-quantizes param init, so it's parity-free); ~2× device time
  on param-bound kernels.
- `CYO_PKT=1` adds per-kernel GPU timestamps to `prof` — the
  breakdown table that caught the O(n²) softmax.

## CUDA results

**Not run** — this machine has no NVIDIA GPU/toolchain.
`./run_cuda.sh` on any CUDA host builds all four (`gemm` includes a
cuBLAS sgemm column for the maturity-gap reference).

## What to expect (pre-registered, so results can't be spun)

- **gemm**: Coyote's tile kernel vs the naive `.cu` — Coyote should
  win on structure alone (register-blocked threadgroup tiling vs
  one-thread-per-element). cuBLAS will still beat both: that's the
  real gap, and it's a tuning-library gap, not a language gap.
- **attn/ffn/sgd**: at these sizes both sides are dispatch-bound;
  kernel times will look similar, host overhead decides the average.
- **DX asymmetry** is already visible in the source: `sgd.cy` is 4
  statements; `sgd.cu` is ~100 lines of manual forward+backward+update
  kernels with Coyote's sequential-statement ordering reproduced by
  hand.
- **Parity**: Coyote diffs every run against the spec evaluator
  automatically. The `.cu` side prints samples for manual comparison
  only — CUDA has no equivalent semantic gate.

## Methodology notes

- Coyote's `attn` emits 4 kernels — scores matmul (tiled), fused
  single-TG softmax, attn·V matmul (tiled), out — reduction
  subexpressions materialize automatically. The CUDA port fuses the
  same math into one kernel by hand; both are honest.
- `sgd.cu` reproduces Coyote's *statement-ordered* semantics: the
  `w2` gradient is evaluated after `w1` is already updated. Standard
  frameworks compute both grads from one forward — Coyote's is the
  eval-faithful version.
- `parity` is max |gpu−eval| over all outputs.
