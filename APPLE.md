# coyote-apple — the internal profile

Coyote is the language: `.cy` source → typed HIR → MIL → HWX → resident
VM → wasm. `coyote-apple` is the same pipeline with the org surface an
internal toolchain needs — it shares every stage with the public binary
(no forked logic, no second implementation) and adds four things a team
runs against artifacts instead of source:

## verbs

```
coyote-apple spec     <file.cy>            capability + structure matrix
coyote-apple audit    <build_dir>          hash-chained build provenance
coyote-apple package  <file.cy> [-o pkg]   single-file .cypack distributable
coyote-apple verify   <pkg.cypack>         integrity check a package
coyote-apple unpack   <pkg.cypack> -o dir  extract a package
coyote-apple bench    <file.cy> [--runs N] e2e + organism-exec timing
coyote-apple oracle   <file.cy> [--tol X]  semantic parity gate (silicon vs spec)
coyote-apple genmask  <n>:<r>:<d> [-o f]   rank-selector mask → safetensors
coyote-apple diff     <dirA> <dirB>        build-vs-build artifact delta
coyote-apple <any coyote verb>             passthrough — check/build/run/repl
```

## what each gives an org

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
