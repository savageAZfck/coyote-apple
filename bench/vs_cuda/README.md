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

| workload | device ms (min) | device ms (avg) | host dispatch ms (avg) | parity vs eval |
|---|---|---|---|---|
| gemm 512³ | 1.694 | 2.032 | 2.415 | 0.000003 |
| attn 64×64 | 0.246 | 0.404 | 0.679 | 0.000000 |
| ffn 256→1024 | 0.144 | 0.966 | 8.410 | 0.000020 |
| sgd step | 0.050 | 0.322 | 0.627 | 0.000002 |

Device time is kernel-only; host dispatch includes buffer staging +
encoder overhead. Small workloads are dispatch-bound — the honest
read is that Coyote's runtime hasn't amortized staging yet.

## CUDA results

**Not run** — this machine has no NVIDIA GPU/toolchain.
`./run_cuda.sh` on any CUDA host builds all four (`gemm` includes a
cuBLAS sgemm column for the maturity-gap reference).

## What to expect (pre-registered, so results can't be spun)

- **gemm**: naive-vs-naive should be within ~2× either way — both are
  one-thread-per-element, memory-bound. cuBLAS will be 10–50× faster
  than either: that's the real gap, and it's a tuning-library gap,
  not a language gap.
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

- Coyote's `attn` emitted 3 kernels (matmul scratch, softmax scratch,
  out matmul) — the emitter materializes reduction subexpressions
  automatically rather than recomputing per element. The CUDA port
  fuses the same math into one kernel by hand; both are honest.
- `sgd.cu` reproduces Coyote's *statement-ordered* semantics: the
  `w2` gradient is evaluated after `w1` is already updated. Standard
  frameworks compute both grads from one forward — Coyote's is the
  eval-faithful version.
- `parity` is max |gpu−eval| over all outputs.
