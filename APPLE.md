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

## distribution
coyote-apple package  <file.cy> [-o pkg]   single-file .cypack distributable
coyote-apple verify   <pkg.cypack>         integrity check a package
coyote-apple unpack   <pkg.cypack> -o dir  extract a package

coyote-apple <any coyote verb>             passthrough — check/build/run/repl
```

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
