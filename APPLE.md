# coyote-apple — the AI programming language

Coyote is the language: `.cy` source → typed HIR → MIL → HWX → resident
VM → wasm. `coyote-apple` is the language *as a distribution* — one
binary that owns the whole lifecycle: scaffold, write, check, test,
verify, train, package, audit, ship. Shared verbs still pass through
to `coyote::cli`. The grammar it builds on — `enc fn`, namespaced
imports (`import "x" as m`), dtype generics (`fn f<D>(x: D[64])`), and
`?` propagation — lives upstream in public `coyote`. The apple verbs
add the surface a language needs to be *used*, not just compiled.

This is a separate repo by design: the public language stays MIT and
clean-room, the distribution carries the org surface — source-available
under [LICENSE](LICENSE), the same terms as Bad Apple.

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
coyote-apple inspect  <path|url>           evidence-grade dissection —
                                           real parsers over dozens of
                                           formats; --dir audits a tree,
                                           --world audits a fleet plan,
                                           URLs are fetched (bounded)
                                           then dissected by the same engine

## distribution
coyote-apple package  <file.cy> [-o pkg]   single-file .cypack distributable
coyote-apple verify   <pkg.cypack>         integrity check a package
coyote-apple unpack   <pkg.cypack> -o dir  extract a package
coyote-apple emit     <file.cy> --swift    typed Swift bridge over libcoyote
coyote-apple shardize <file.cy> -o <dir>   fn programs → resident ANE
                                           pipeline shards (.mlmodelc)

## fleet — sites, ranks, serving
coyote-apple meshnode  --listen h:p        treaty'd peer execution site
coyote-apple pccnode   --listen h:p        attested remote-compute site
coyote-apple peernode  --listen h:p --treaty t.json [--programs d]
                                           treaty-gated program peer
coyote-apple ranknode  --shard <dir>       weight-carrying ANE-resident rank
coyote-apple serve     [opts]              session-managed inference daemon
coyote-apple ask       <prompt>            client into the serving tier
coyote-apple mind                          mind_* extern bindings/status
coyote-apple fleet     up|down|status --world <w>   fleet lifecycle
coyote-apple fleetcheck --world <w>        plan-vs-live-ranks conformance
coyote-apple replay    --journal <f>       re-execute committed witness
                                           entries (serve-request receipts)

# verification layer — refereed delegation as toolchain verbs
coyote-apple referee  <file.cy> [--replay] RDoC bisection over the committed
                                           witness chain — catches a lying
                                           backend in log(n) challenge rounds
coyote-apple certify  <file.cy> [-o cert]  proof-carrying certificate:
                                           compiles/parity/replay determinism
coyote-apple certify  --check <cert>       re-verify a certificate
coyote-apple conform  <f> --spec s.cyproto session conformance over the chain
coyote-apple caps     <f> [--deny/--allow/--pure]  capability policy gate

# product surface — language tooling people live in
coyote-apple fmt      <file.cy> [--check]  canonical formatter (idempotent)
coyote-apple pkg      install|list|remove|search   package registry —
                                           ~/.coyote/pkg rides CYO_PATH
coyote-apple lsp                           full language server: diagnostics,
                                           completion, hover, def, semtoks

coyote-apple <any coyote verb>             passthrough — check/build (run/repl
                                           are intercepted: the mind externs
                                           bind on every VM this binary builds)
```

## the gpu/enc strata — first-party compute

`gpu fn` parses through the local grammar and its *engine* lives only
here. The crate exposes a backend seam —
`coyote::backend::register_backend(target, impl Backend)` — and this
distribution registers `MetalBackend` (and `EncBackend` for `enc fn`)
at startup. Strata are grammar in upstream `coyote` (`enc fn`, `?`,
`import "x" as m`, dtype generics `fn f<D>` all parse in the public
crate); engines are this repo's property — `enc`, the Metal backend,
and the mesh/pcc execution sites live only here.

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
  `timed` (telemetry), `models`, `remember`, `delegate`, `cortex`
  (the working live-mind query — the language's builtin `cortex_ask`
  attests deterministically when no socket is wired; `mind_cortex`
  always reaches the daemon), `embed`/`cosine`/`recall_key` —
  every mind call is `stoch` + `cap mind`, checked by the same
  constitution as every other capability.
- `mind_embed` is deterministic and offline: 64-dim FNV-seeded
  token rows, mean-pooled, L2-normalized — no `stoch` marker, so
  `det` contexts can embed too. Same {ok, vec} shape the daemon's
  internal semantic_embedding returns; when the daemon exposes an
  embedding RPC this extern can switch to it without touching the
  library surface.
- Binding is per-Vm in this distribution's own code — `run`,
  `bench`, `test`, `oracle` (both parity VMs), `watch --run`, and
  the apple `repl` each call `mind::bind_all` before executing.
  `src/repl.rs` is the distribution's shell: same two-machine model
  (live + scratch VMs) but every Vm is bound and scratch eval
  resolves `import` through `parse_src_at`, so `mind_status()` and
  `models()` answer live at the `cy>` prompt.
- Verified live: `tests/probes/mind_probe.cy` — status → models →
  a real generation → telemetry → cortex → embed determinism → a
  ledger witness, all through the socket. `test` runs
  `test_embed_det` against the offline embedder.
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

## the sites — substrate × execution location

Compute targets answer *what* runs the math; **sites** answer *where*.
`Site` is a second dimension on every tensor `fn` — the same typed
program, the same io plan, the same witness semantics — dispatched by
execution location through registered site backends:

| site | syntax | where it runs | posture |
|---|---|---|---|
| local | `fn`, `gpu fn`, `enc fn` | this machine | resident backends |
| mesh | `mesh fn`, `mesh gpu fn` | a treaty'd peer | mesh wire: HMAC `mb-c`/`mb-s` auth, AES-256-GCM packets |
| pcc | `pcc fn`, `pcc enc fn` | an attested node | attest-before-secrets, signed results |

The grammar carries `Site` on `Fn`/`TProgram`; the `emit` output is a
dispatch manifest plus the program source; `load` binds to a node over
`wire.rs` — a byte-exact port of the Bad Apple mesh-brain protocol
(`[u32 BE jsonLen][json][payload]`, sealed = `nonce‖GCM(inner)‖tag`,
secret ≥32B, AES key = SHA-256(secret)). A `coyote-apple` node and a
Bad Apple rank server speak the same packets.

- **meshnode** — serves `bind`/`exec`/`ping`/`reset` against the
  resident evaluator. `exec` payload = declared ins concatenated f32;
  `result` returns the output + `witness` entries.
- **pccnode** — adds `attest`: the client receives a nonce-bound
  manifest (`node`, `build`, `caps`, `sig` = HMAC(secret,
  `pcc-attest:nonce`)) and verifies it **before** the program or any
  tensor crosses — verify-before-secrets. Every `result` carries
  `sig` = HMAC(secret, `pcc-result:nonce:sha256(payload)`), checked
  client-side. Loopback attestation class is software-grade by design;
  the manifest shape is what a hardware quote plugs into later.
- **enc×pcc** — `pcc enc fn` encrypts inputs client-side (LWE), ships
  Ct quads (4×u64 LE per lane: `a‖b‖noise‖scale`), the node evaluates
  through `enc_ckks::run_ciphertext` in the ciphertext domain, and the
  client decrypts. The node never sees plaintext; the client never
  sees remote weights. This is the CHET posture as a language stratum.
- **witness across the wire** — `BackendExec::drain_remote()` hands the
  node's per-exec witness entries to the VM, which appends them to its
  own hash-chained log in dispatch order. One computation, N nodes,
  one ledger — the chain a `referee`/`certify` replay covers is the
  whole computation, not just the local half.
- **treaty/peer** — `treaty name { allow …; deny … }` is a program
  decl; `peer` is a handle type; `peer_connect` performs the mesh
  handshake then the *treaty exchange*: both sides ship allow/deny
  sets and the connection binds to the intersection minus either
  side's denies. `peer_call` re-gates the verb against that bound law
  before a byte moves. All `peer_*` externs carry `cap treaty`.
- **shardize --world N** — contiguous layer split across N ranks:
  `rank{i}/` dirs (layer shards + head iff last rank + embedding iff
  rank 0), per-rank `shard.json` in the bad_apple `ShardSpec` schema
  (rank/world/host/next_host/layer bounds/ownership flags), and a
  top-level `world.json` naming the wire (`badapple-mesh/1`) and auth.
- **mesh fn vs. world** — the two mesh surfaces are different layers:
  `mesh fn` ships a *program* to a peer's evaluator; `--world` shards
  *weights* into bad_apple-compatible rank dirs a mesh serves natively.

`demo/five_strata.sh` runs all five strata live: three nodes, one
organism, one hash chain that records what every machine did.

## the serving tier — `serve` / `ask`

`coyote-apple serve --world world.json` fronts a rank pipeline as a
session-managed inference daemon — the multi-client layer above the
single-stream substrate. Clients speak the same wire as ranks (auth +
AES-GCM); sessions pin KV **lanes** on the ranks via `frame.lane`, the
lane pool *is* admission control (exhausted → `busy`), rank-0 transport
failover retries onto the next candidate, and every request/response
lands on a hash-chained witness journal tagged `session:lane:bytes`.
`coyote-apple ask` is the client.

### `ranknode` — a weight-carrying rank on the Neural Engine

`coyote-apple ranknode --shard <dir>` is the rank the mesh was built
for: it loads a `shardize --world` rank directory through
`mil_infer::ane_pipeline::Pipeline` — the same `model.mil` graphs,
compiled onto the ANE. KV lives in the graphs' resident state buffers;
`--lanes N` spawns N independent pipelines so concurrent sessions hold
isolated decode state. `generate` takes an i32 token-id payload *or*
prompt text (auto-discovers `tokenizer.json` in the shard dir, or
`--tokenizer`); results carry both the id payload and decoded `text`.
Every step/generate writes rank-side witness entries (`rank-step` /
`rank-generate` with rank:lane:pos) that land on the caller's witness
chain — the serving path is as auditable as the exec path. Verified:
two ranks decoding real tokens across the wire, sessions on separate
lanes, LRU lane reaping, admission `busy` on true exhaustion.

## the organ library — `lib/badapple/`

Bad Apple's organ layer, in the language: 30 `.cy` ports folded in from
the organism repo — `inspect.cy` (the dissection primitives as
organism calls), `mesh_brain.cy` (wire client + shard tools),
`p2p.cy`, `cert.cy`, `ledger.cy`, `protocol.cy`, `config.cy`,
`scavenger.cy`, `slicks.cy`, `vault.cy`, `ify.cy`, `firewall.cy`,
`cage.cy`, `sentinel.cy`, `watcher.cy`, `mcp.cy`, `cli.cy`, `sys.cy`,
`strlib.cy`, `blueprint.cy`, `strategy.cy`, `layer.cy`, `mil.cy`,
`org_policy.cy`, `puf_challenge.cy`, `referee_drift.cy`,
`referee_match.cy`, `benchmark.cy`, `mesh_sync.cy`, and
`badapple_organs.cy` — the five-organ fusion (axon reflex, akicita
marshal, macuahuitl firewall, tokala council, camazotz erasure)
compiled into one organism whose witness chain *is* the audit ledger.
All check-clean; `import "badapple/x.cy"` resolves through `lib/`.

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
- **attestation** — `pcc` nodes answer `attest` with a nonce-bound,
  treaty-signed manifest today; a hardware quote plugs into the same
  `attestation()` seam without touching the language or the wire.
