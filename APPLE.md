# coyote-apple — the AI programming language

Coyote is the language: `.cy` source → typed HIR → MIL → HWX → resident
VM → wasm. `coyote-apple` is the language *as a distribution* — one
binary that owns the whole lifecycle: scaffold, write, check, test,
verify, train, package, audit, ship. Every public verb passes through
to `coyote::cli` unchanged (no forked logic); the apple verbs add the
surface a language needs to be *used*, not just compiled.

This is a separate repo by design: the public language stays MIT and
clean-room, the distribution carries the org surface — and it's the
unit that becomes private the moment first-party interfaces exist.

## verbs

```
## authoring
coyote-apple new      <name> [--program]   scaffold a project (organism or graph)
coyote-apple doc      <file.cy> [-o md]    generated API markdown
coyote-apple test     <file.cy> [--cpu]    host fn test_*() runner
coyote-apple spec     <file.cy>            capability + structure matrix
coyote-apple watch    <file.cy> [--run]    rebuild on save (import-aware)
coyote-apple gold     record|check <f.cy>  golden-output regression snapshots

## checkpoint tooling
coyote-apple weights  inspect <file.st>    tensor inventory (dtype/shape/bytes)
coyote-apple weights  slice <f> <t>[a:b]   cut a tensor → bindable .safetensors
coyote-apple genmask  <n>:<r>:<d> [-o f]   rank-selector mask → safetensors

## verification + provenance
coyote-apple oracle   <file.cy> [--tol X]  semantic parity gate (silicon vs spec)
coyote-apple audit    <build_dir>          hash-chained build provenance
coyote-apple diff     <dirA> <dirB>        build-vs-build artifact delta
coyote-apple bench    <file.cy> [--runs N] e2e + organism-exec timing
coyote-apple prof     <file.cy> [--runs N] per-dispatch device timing (GPU kernel ms)

## distribution
coyote-apple package  <file.cy> [-o pkg]   single-file .cypack distributable
coyote-apple verify   <pkg.cypack>         integrity check a package
coyote-apple unpack   <pkg.cypack> -o dir  extract a package
coyote-apple emit     <file.cy> --swift    typed Swift bridge over libcoyote

coyote-apple <any coyote verb>             passthrough — check/build/run/repl
```

## the gpu stratum — first-party compute

`gpu fn` is parsed and typechecked by the public language, but its
*engine* lives only here. The public crate exposes a backend seam —
`coyote::backend::register_backend(target, impl Backend)` — and this
distribution registers `MetalBackend` at startup:

```
gpu fn block(x: f16[64]) -> f16[64] {
    rmsnorm(x + x @ wq, rw)
}
```

- `src/gpu_emit.rs` lowers the typed AST to Metal Shading Language —
  one kernel per program, one thread per output element, elementwise
  fused inline, matmul/softmax/rmsnorm/gather as per-thread loops.
- `src/gpu_rt.rs` compiles the MSL via the `metal` crate and dispatches
  over `StorageModeShared` buffers — unified memory means the staging
  write IS the device buffer. Weights stay resident across calls.
- `net.msl` + `coyote_build.json` land in the build dir, so `audit`,
  `package`, `diff`, and `gold` cover gpu builds exactly like ane
  builds. `oracle` gates them automatically.
- Reduction subexpressions materialize automatically: a tensor whose
  subtree contains matmul/softmax/rmsnorm/gather gets its own prelude
  kernel + scratch buffer in discovery order (let-bound reductions
  materialize at bind position, matching eval's eager `let`). Without
  this, `softmax(q@k.T)@v` inlined to 1G MACs — with it, ~1M.
- Bare `coyote` parses `gpu fn` but fails at emit — "no backend
  registered". The capability is the distribution's.
- `grad(loss, w)` lowers reverse-mode on-device — same rule set and
  `mse`-terminal scope as the evaluator — and `w op= s*grad` folds to
  an update kernel dispatched before the out kernel (eval's
  stmt→ret order).
- `adam(w, grad(loss, w), lr)` trains on-device: moments + step live
  in persistent zero-init scratch; the apply kernel runs AFTER the
  out kernel because eval leaves the param untouched during `main`
  (the driver applies `adam_step` between calls — same math, same
  fp16 writeback rounding).
- State writes (`s.f op= e`, one-hot scatters `s.f[p] op= e`) emit
  stmt-position kernels against the resident state buffer — the
  kv-cache story. `w op= e` on params writes at stmt position;
  `a op= e` on lets shadow-binds the scope. Every `TStmt` lowers —
  no honest-error statement left.
- Matmuls emit cooperative-tile kernels (32×32 blocks, 8×8 groups,
  4×4 register micro-tile, padded threadgroup staging); `CYO_SIMD=1`
  swaps in the `simdgroup_matrix` path — correct but slower on this
  silicon, gated for newer hardware. `CYO_TILE`/`CYO_TM`/`CYO_BK`
  override tile geometry for autotuning sweeps — 32/4/16 is the
  measured sweet spot on dev silicon.
- Vector-lhs matmuls (`x @ w`, `q @ kv.k.T`, `s @ kv.v` — the decode
  shape) emit a dedicated GEMV kernel: one thread per output, serial-k
  accumulator in eval's exact order, contraction row consumed as
  `float4` loads — a quarter of the load instructions. Column-
  contraction cases keep one-thread-per-output serial-t. On the
  D=1024/T=256 decode probe: device 1.617 → 0.762 ms, dominant
  matvecs 0.30 → 0.10 ms each, parity unchanged at 0.000027.
  (A cooperative threadgroup-staging variant measured 2.4× slower on
  this silicon — barriers expose latency where sequential rows
  already prefetch — `CYO_NOGEMV=1` falls back for A/B work.)
- `softmax`/`rmsnorm` emit fused single-threadgroup kernels — the
  order-free half (max) tree-reduces in parallel, the
  order-sensitive sum stays serial in eval's exact accumulation
  order, so parity holds at 0.000000 while work stays linear.
- `rsoftmax(x[R,C])` — per-row softmax, one thread per row (serial
  within the row = eval's order, parallel across rows). This is what
  attention actually means: attn at L=256 went 1.8ms → 0.22ms
  device. Global `softmax` remains for 1-D distributions.
- `grad` through a batched `[M,K]` lhs accumulates `dW` over batch
  rows and `dx` per row — matching the eval fix for what used to be
  a panic on any 2-D matmul in a loss.
- `CYO_HALF=1` stages `f16` params as real `half` buffers — compute
  stays fp32, reads promote, writes round (eval's f16 param-init is
  bit-identical; SGD-update programs see ~1e-3 rounding drift, adam
  stays exact). Measured ~2× on the param-heavy FFN bench.
- `CYO_PKT=1` makes `prof` report per-kernel GPU timestamps — one
  command buffer per kernel — the breakdown that caught both the
  O(n²) softmax and the staging overhead. `prof --dump` reads
  trained weights back vs deterministic init.

## mind — intelligence as a language primitive

`lib/mind.cy` + `src/substrate.rs` + `src/mind.rs` give organisms a
checked, cap-gated path to the resident mind — the same daemon the
rest of the platform uses, reached through its SLICKS socket:

- `substrate.rs` is a self-contained SLICKS client: Hello →
  Challenge → Execute over `/var/run/badapple/substrate.sock`, v2
  (identity-agent Secure Enclave signatures, TOFU-pinned server key)
  when the identity agent is up, v1 shared-key HMAC otherwise — same
  auto-resolution as the reference client. Nothing links back into
  bad_apple; the contract is the wire protocol.
- `mind_infer` reaches the daemon's `inference` agent method;
  `mind_call` covers the whole agent surface (`list_models`,
  `invoke_tool`, `run_agent_task`, `audit_event`, …); `mind_witness`
  lands events on the organism's hash-chained ledger; `mind_task`/
  `mind_tasks`/`mind_tool` cover delegation and tools.
- `mind_gen` takes the plain-prompt route instead of the agent
  envelope — same generation path as the CLI, but Done frames carry
  telemetry the envelope drops: `decode_tps`, `total_tps`, `tier`,
  `prefix_cache`, `speculative`, `draft_accept_pct` — the
  cross-silicon draft/verify surface, observable from a `.cy` program.
- Language-side verbs in `lib/mind.cy`: `think`, `ask`, `gen`,
  `timed` (telemetry), `models`, `remember`, `delegate` — every call
  is `stoch` + `cap mind`, checked by the same constitution as every
  other capability.
- `run` is intercepted in the distribution so every Vm it builds gets
  `mind::bind_all`; `bench`, `test`, and `oracle` bind on their Vm
  paths too (`repl` doesn't — its Vm is built inside the public cli).
- Verified live: `tests/probes/mind_probe.cy` — status → models → a
  real generation → a ledger witness, all through the socket.
- The VM skips param/state staging when the host tensor handle is
  unchanged (`staged_ext` in public `vm.rs`) — after `read`-back the
  mirror holds device values, so a re-dispatch doesn't round-trip
  identical bytes or redo the [K,N]→[N,K] transpose.
- `tests/run_probes.sh` — the gpu regression suite: 11 probes gated
  on parity, one shell command, no framework.

Same-file mixing is the point: `fn` → ANE Mach-O nodes, `gpu fn` →
Metal kernels, `host fn` → bytecode VM — one type system, one witness
chain, all three silicons.

## the stdlib — `lib/std/`

The distribution ships a standard library. `import "std/lora"` (or
`std/expect`) resolves through `CYO_PATH`, which coyote-apple extends
with its own `lib/` automatically — user paths win, distribution
paths are the floor. Modules splice at parse time; the manifest hash
covers every imported file, so `audit` fingerprints the stdlib too.

- **`std/lora`** — masked-LoRA helpers (`lora_adapt`, `lora_loss`):
  the rank-in-data adapter pattern, ready to `grad`.
- **`std/expect`** — test assertions (`expect_eq`, `expect_true`,
  `expect_ne`, `expect_near`): return 0/1 semantics for `test_*` fns.
- **`std/nn`** — transformer building blocks (`attn`, `attn_v`,
  `ffn_swiglu`, `ffn_gelu`, `mlp`, `res_norm`, `embed_row`): weights
  arrive as args, dims inline generically. Bit-exact on Metal; the
  `ffn_*`/`mlp`/`res_norm` helpers also run on ANE — activation-rhs
  matmul (`attn*`, `embed_row`) is gpu-only while the ane emitter
  requires a param rhs.

## what each gives an org

- **new** — `coyote-apple new probe` produces a working organism in one
  command: invariant, provenance, a tensor program, `host fn main`,
  and a `test_*` fn that passes `coyote-apple test` on first contact.
- **doc** — markdown rendered from the *checked* program: io/param/
  state tables, host signatures, invariants. API docs that cannot
  drift from the artifact.
- **test** — any zero-arg `host fn test_*` is a test case: returns
  0/unit/true → pass, nonzero or raised error → fail. Runs inside a
  resident VM on the selected backend, so tests exercise `ane(...)`
  program calls through the same surface the organism uses.
- **weights** — `inspect` inventories a checkpoint (311 tensors on
  Qwen3-0.6B, `--grep` filtered); `slice` cuts `tensor[a:b,c:d]` into
  a new safetensors that `param ... @ "file::name"` binds directly.
  Checkpoint-driven authoring becomes toolable, not hand-written.
- **spec** — what the typechecker *proved*: every program's io, params
  (trainable/frozen/sealed + shapes), state surfaces, host fns with
  signatures, invariants, caps, provenance. A build's constitution as
  data, not documentation.
- **audit** — every artifact in a build dir chained
  `H = sha256(H_prev ‖ name ‖ sha256(file))`. `--verify` recomputes:
  tampered blobs, reordered nodes, stale plans all break the head hash.
  Provenance you can file, not believe.
- **package/verify/unpack** — `.cypack`: `CYPK1` header + per-member
  sha256 + payload. One file moves a compiled program between machines
  with integrity checked at the boundary.
- **bench** — pipeline time (emit+compile+exec, the honest e2e number)
  and, for organisms, `host fn main` hot-path latency inside a resident
  VM on both backends. `--json` makes the last stdout line the
  machine-readable report — CI greps one line, humans read the rest.
- **oracle** — the semantic gate. `coyote run` already diffs every
  output against the CPU reference evaluator; `oracle` harvests those
  parity numbers into a verdict. Tensor programs: worst |ane−cpu|
  across all steps vs `--tol`, exit 1 on breach. Organisms: dual
  resident VMs (ANE backend + CPU backend), `main` return and witness
  count must match. `--json` for the gate record.
- **genmask** — rank-selector masks as safetensors. The skeleton table
  forces dim ≥ 64 on adapter matmuls, but rank lives in data, not
  shape: `((x @ la) * sel) @ lb` zeroes coordinates ≥ r and the dead
  weights get exactly zero gradient — true low-rank LoRA inside the
  shape the compiler wants. `examples/lora8.cy` trains rank-8 on-chip.
- **diff** — recursive per-file sha256 delta between two build dirs:
  added/removed/changed, `~` lines carry both hash prefixes. The
  receipt for "what did this edit actually change."

## integration map

Nothing here is gated, stubbed, or "coming." Where internal capability
plugs in later, it plugs into existing seams:

- **spec backends** — the emitter library lives behind
  `mil_infer::graph::compile_graph`; a first-party spec widens shape
  coverage without touching the language.
- **residency** — `GraphRuntime::load` already owns program lifetime;
  entitlement-backed residency is a load flag, not a redesign.
- **signing** — `audit` produces the chain; a platform signature wraps
  the head hash. The manifest format is already the birth certificate.
