# BRAIN — his own brain, born and trained on Apple silicon

Status: plan. Phase 0 (convergence) is the active blocker.

## What already exists (verified, not assumed)

**Two ANE pipelines:**

| path | source | proof |
|---|---|---|
| `.cy` native | coyote-apple emitters → mil_spec `.hwx` → resident pipeline → ranknode shards | Qwen3-0.6B, 28 layers, d=1024, 32 tok/s, CPU parity, fleet across 2 ranknodes |
| `badapple-aneconvert` | HF safetensors → `.mlmodelc` shards | Qwen3-4B resident: 36 layers, hidden 2560, seq_len 2048, int8/fp16, vocab-split head, ~1.5 GiB footprint |

**Language-native capabilities (coyote-apple):**
- Four substrates in one grammar: `fn`→ANE, `gpu fn`→Metal/MSL, `enc fn`→LWE ciphertext, host→CPU. `mesh fn`/`pcc fn`/`pcc enc fn` for treaty'd federation. Demoed on video.
- `state<>` KV rings: `pos` unbounded, `slot = pos % seq_len` — context is compile-time `seq_len`, attention span slides. 128k–1M+ tier is the paged/sequence-parallel design across lanes (weights shard across ranks; context shards the same way).
- Training primitives shipped: `grad(loss, w)`, `adam(w, g, lr)`, `trainable`/`frozen`/`sealed` params, `genmask` masked-rank LoRA. Forward+backward+update compile into the same binary.
- Provenance: name-seeded deterministic init (`init_weight`, lower.rs:149), hash-chained witness per run, `param lineage` compile-checks provenance decls, SEP-signed identity.
- `shardize --world`: splits a `.cy` model across ANE-resident ranknodes.

**Emitter law (mil_spec/hwx.rs):** everything tiles to verified skeletons —
matmul N∈{16,32,64,128,256}, conv C∈{16,32,64,128}, `attn_v` at
(64|256|1024 × 32|64|128), `avw` parametric, elementwise chains, softmax,
rmsnorm, 4-D head-partition matmuls. d=1024 runs as 64-wide slices (see
qwen28.cy: every proj is `q{head}{lo,hi}` [1024,64] tiles).

## The hard constraints that shape everything

1. **ANE residency is the training ceiling.** One d=64 transformer layer's
   `train_step` = 7,248 program nodes and already pressures `aned`. A
   d=1024 layer is ~16× the tiles. Full-model backprop in ONE graph is out —
   inference scales (28 layers resident), training graphs don't.
2. **fp16 everywhere.** Gradients through a 1024-way MSE are ~1e-3; fp16
   denormal flush + Adam epsilon can silently kill learning. Suspect #1.
3. **No BPTT through `StateUpdate`.** KV-cache-adjacent params can't take
   gradients — the tape doesn't cross the state boundary. Cache projections
   stay `frozen` until state-aware autodiff exists.
4. **`init_weight` is uniform ±0.5, no fan-in scaling.** At d≥256 activations
   explode through layers. Suspect #2 for non-convergence.
5. **CPU backend is broken for the brain**: eval.rs:116 assumes rank-2
   params (rank-1 RMSNorm gammas panic); state-field matmul backward
   unverified on CPU. Needed for ANE-vs-CPU gradient A/B.

## Target definition — "his own brain" (clarified)

**Goal: his own foundation-class models — "my own Qwen, my own Gemma,
my own DeepSeek" — made just for Apple, running flawlessly across
GPU/CPU/ANE.**

- **Physics constraint (non-negotiable):** from-scratch pretraining of a
  Qwen-class model is datacenter compute — not possible on an M4. The
  viable path to *his* weights is **distillation + on-device growth**:
  the organism's existing brains (8B MLX GPU, 4B ANE) teach the student
  on his corpus/ledger, and `grad`/`adam` on the ANE does the learning.
  This is how real small models are made (Phi, Gemma distills) — and it
  makes the provenance story stronger, not weaker: the weights cannot
  exist without his silicon and his data.
- **Architecture**: his design — own dims, head counts, FFN shape,
  possibly novel topology — constrained to emitter-verified ops, and
  co-designed for this silicon (that's the "made just for Apple" part:
  ops chosen because they map perfectly to the verified routes).
- **Weights**: born on-chip via name-seeded init; every update
  witness-chained; `param lineage` verifiable.
- **Body**: one model split across substrates — the cross-substrate
  partitioner is the missing piece (per-program parity proven by
  `qwen_layer_3way`/`decode_3way` + oracle; split-program scheduling is
  new work, building on the evolve/embodiment dispatch pattern already
  running ANE+GPU lanes).
- **Scale path**: 0.6B-class `.cy` decoder → N layers via a parameterized
  model generator → fleet via `shardize --world`/ranknode → 4B-class via
  anefconvert-compatible export if the architecture maps.

## Phases

### Phase 0 — convergence at d=64 — **SOLVED 2025-XX**
The brain learns. `examples/brainorg_dev.cy` (d=64, 4 RoPE query heads,
2 KV heads w/ state cache, SwiGLU FFN, 64-way head — every transformer
param `trainable`, born on-chip) trained on CPU: 16-position curriculum
over the real `seq` embedding slice, one-hot 64-way targets — predictions
went from all-wrong at ep 0 to ~13/16 exact by ep 59, monotonic descent.

Root causes found (all curriculum/toolchain, not architecture):
1. **Toolchain autodiff was silently broken for deep graphs** — fixed:
   - `eval.rs`: tape adjoints now accumulate per let-bound def and drain
     once in reverse topo order (was exponential tree re-walk → stack
     overflow); `needs_grad` memoized with cycle guard; `defs` memo
     cleared per `grad()` and per `ParamUpdate` (stale forward values
     were silently dropping whole gradient paths → "does not depend"
     errors and zero grads); rank-1 params sized via `numel()`;
     state-field matmul backward (`x @ kv.field[.T]`) added.
   - `lower.rs`: same adjoint-accumulation drain; `needs_grad` depth cap
     32→100k + visiting-set cycle guard (the cap was pruning every
     gradient path deeper than 32 exprs — i.e. ALL transformer paths);
     state-field `w_name` resolution for backward matvecs; param-adjoint
     matvecs routed through `mat_t`/`mat` storage refs.
   - `vm.rs`: organism readback handles 1-D params; host-side `adam_step`
     actually applied; ANE residency evicts on program switch (7,248-node
     train_step was starving `forward` of ANE resources).
4. **Pointer-keyed `defs` memo collided across cloned tape defs** —
   THE deep bug, found via d=128 (`eval.rs` panic "len 64 index 64"):
   - `grad()` stores each let's defining expr on the tape via
     `expr.clone()`; the drain walks `self.tape.get(name).cloned()` —
     a fresh Typed tree per let, dropped after its `back()`.
   - `defs: HashMap<*const Typed, Val>` memoizes forward values by node
     ADDRESS. Dropped clones free their nodes; the allocator reuses the
     addresses for the next let's clone → `fwd()` returns the PREVIOUS
     let's cached value for a structurALLY DIFFERENT node.
   - At d=64 every tensor was 64-wide so stale hits were shape-compatible
     → silently wrong gradients (likely THE source of the run-to-run
     trajectory nondeterminism — allocator layout differs per process).
     At d=128 the collision produced 64-vs-128 length mismatches → panic.
   - Fix: `self.defs.clear()` before each drain `back()` in BOTH
     `eval.rs` and `lower.rs` (the compiled/ANE path had the identical
     bug — ptr→mil-tensor-name map — meaning ANE backward graphs were
     wired to wrong intermediates; this likely explains the noisy ANE
     parity). Each clone subtree is independent, so per-def clearing
     loses nothing.
   - Verified: d=128 smoke (train_l0 26 grads + train_head) runs clean,
     zero length-mismatch diagnostics.
2. **Curriculum was the real convergence blocker** — the original
   `brainorg.cy` used `emb[0..3]` (vocab rows = meaningless special
   tokens, not a sequence), only 4 positions in a 64-slot KV window
   (softmax saturated → attention grads ≈ 0 → collapse to
   input-independent output — confirmed: 200-epoch run froze at fixed
   preds), and 1024-dim one-hot targets (ill-conditioned MSE).
   Fix: `seq[i]` real consecutive embeddings, 16 positions, 64-way head.
3. Verified regression-clean: `finetune.cy` still converges
   MSE 1.66→0.057 with ANE-CPU parity ~1e-3 after all toolchain changes.

Remaining Phase-0 tail: ANE-parity run of the fixed curriculum.
Early parity is strong: ANE ep0 preds bit-identical to CPU (48, 30),
learning trajectory visible by ep 3 before the run was paused to free
the ANE for compile tests. `finetune.cy` already proves ANE-CPU
gradient parity ~1e-3 on a training graph.

### Phase 1 started — the generator exists and depth works
`src/bin/gen_brain.rs` (pure Rust — `tools/gen_brain.py` deleted, zero
Python in the project) emits parameterized multi-layer decoder organisms:
`--layers --dmodel --qheads --kvheads --ffn-branches --ffn-dim
--vocab --seq-len --positions --epochs`. Matmul convention verified
against eval.rs (`x @ w`, w declared f16[in,out]). The generator now
also emits a companion `<name>.corpus.safetensors` — `seq [S,D]` F32,
spec-seeded (FNV+splitmix64, ~U(-0.1,0.1) matching the original corpus
scale) — so any dmodel≤1024 (qwen3-0.6b embed slice bound) is self-
contained; output is deterministic modulo the `--out` path embedded in
the table binding.

Verified:
- 1-layer default spec → checks clean, same shape as brainorg_dev.
- **d=128 1-layer (38 params) trains on CPU** post-defs-fix — smoke
  4ep×8pos clean through all 26 layer grads + head grads.
- **2-layer d=64 (74 params) trains on CPU** — 10 epochs × 8 pos,
  monotonic descent, generation produces coherent ascending trajectory.
- **2-layer compiles + runs on ANE**: forward 373 nodes; train_step
  **29,673 nodes in 32.5s**; 8 gen steps clean, witness written.
  (An earlier "no ANE resources" failure was contention with the
  parity run holding the ANE — not a residency wall. vm.rs eviction
  on program switch works at 29k nodes.)

Open scaling questions (measure, don't assume):
- **d_model=128 TRAINS ON ANE** (2025-11-05): five new skeleton tiles
  captured via `milc ane compile` (same capture path as all existing
  skeletons — execution still 100% his emitter/GraphRuntime):
  `outer64x128`, `outer128`, `attnv_128x64`, `add4d64128`,
  `add4d12864`. Key discovery: ANEC rejects `y=[1,D,1,1]` strided at
  d>64 but accepts contiguous `[1,1,1,D]` — the contiguous spelling is
  the only route. `skel_tail_off()` walks load commands for the
  ANEC-version cmd (0x8) — hardcoded tail offsets broke on captures
  with different segment mixes.
- **GraphRuntime mid-run surface liveness** (`graph.rs::run`): node
  output IOSurfaces now recycle when their last consumer executes —
  previously every node output lived until `release_all`, so a
  7,325-node d=128 graph materialized ~7k surfaces and died at
  `createIOSurface failed` (node ~5570). Liveness + pool reuse +
  `release_residents()` on any pass that recycled (stale-binding
  safety). d=64 regression-clean.
- d=128/hd=64 verified: forward 193 nodes, train_head 589,
  train_l0 7,325 — full 4ep×4pos curriculum + generation + witness
  on ANE. ANE vs CPU preds diverge slightly at init (fp16) — expected.
- **d=256 TRAINS ON ANE** (same session): added `outer64x256`,
  `attnv_64x256`, `add4d64256`, `add4d25664` — full tiny curriculum +
  generation + witness verified. Tile recipe per new d_model: probe
  compile → capture the named (t,d) via `milc ane compile` with
  contiguous-y spelling → `skel_tail_off` handles layout → table entry.
  d=512/1024 need their own outer/add4d/attnv rows — mechanical, same
  loop; watch `emit_avw` lane mask (b<32) and ANEC DRAM budgets.
- Open: 60ep ANE parity is ~4min/step → rationed curriculum or local
  per-layer objectives; real-text curriculum (byte/token embedding
  path — needs `embed [vocab,d]` param + gather); GPU+ANE partitioner;
  mil_spec changes are LOCAL via `[patch]` in coyote-apple/Cargo.toml
  — commit upstream + drop patch when ready.
- **`--head-dim` decouples head width from d_model** (gen_brain): q/k/v
  projections `[d,hd]`, o `[hd,d]`, state `[sq,hd]`, qn/kn `[hd]`,
  RoPE `[hd]`, scale 1/√hd. hd=64 keeps every state op on verified
  (64,64) tiles while d_model scales the residual stream via parametric
  avw matvecs. Default hd=dmodel = old behavior.
- ANE ceiling for joint train_step node count — 1L≈7.3k, 2L≈29.7k;
  4L per-layer compile still >60min CPU (killed — compile-time is the
  wall before residency; local per-layer objectives is the designed fix).
- d_model 64→128→256 tile coverage on the act×act skeleton table.
- **ANE step latency is the curriculum constraint**: ~4min/train-step
  observed on the 7.3k-node d=64 train_step → 60ep×16pos ≈ 60h. Parity
  proof needs a RATIONED curriculum (few epochs/positions), not the full
  sweep; layer-wise greedy pretraining is the designed answer.
- **Nondeterminism — ROOT-CAUSED**: it was the defs pointer-collision
  above, not HashMap ordering or sha. Clone addresses differ per
  process → different stale-cache hits → different (corrupted) gradient
  paths per run. Post-fix same-file double-run should be bit-identical;
  verify before claiming reproducible birth.

### Per-layer training architecture (replaces joint train_step)
Joint `train_step` re-emits forward once per param: 1L≈7.3k nodes
(2.3s compile), 2L≈29.7k (32.5s), 4L joint emit >17min — killed,
superlinear wall. `gen_brain` now emits `train_l{l}` per layer +
`train_head`; host dispatches round-robin `(ep*P+i) % (L+1)` — block
coordinate descent. Each fn still backprops through the full-depth
forward (correct — upstream layers affect the loss downstream), so
compile cost per fn is ~26-35 grads × full graph; 4L first fn still
>5min emit. Next lever if too slow: local per-layer objectives
(greedy pretraining — each layer trains on its own reconstruction
target, shallow graph) — already in the Phase-1 plan.

ANE 8-epoch probe (brain_ane8.cy): completed — learning but slower
than CPU (ep7 ≈ 2-4/16 hits vs CPU ~13/16 by ep59; only 8 epochs +
fp16 + one unlucky trajectory). Full 60-epoch ANE run needed for the
real parity verdict (~7h at observed ~20-60s/step).

### Phase 1 — the real brain v1 (inference-first, then learning)
- Scale brainorg along verified tiles: d_model 64→128→256→512→1024,
  layers 1→4→8. Forward must stay resident (shardize proves 28 fit).
- Architecture is his: pick head counts, FFN ratio, depth deliberately —
  document why each choice maps to a verified route.
- Training structure (the honest answer to constraint #1):
  - **Layer-wise greedy pretraining**: each layer trains against a local
    objective (next-token prediction from its output, or reconstruction)
    in its own `train_step` program. Real gradients, real learning,
    graphs that fit. This is a real technique, not a hack.
  - **Top-of-stack joint training**: last K layers + head in one graph
    where residency allows — that's where LM signal concentrates.
  - **Adapter tiers** for continued growth post-pretraining (lora8 pattern,
    genmask-proven).
- Distillation option: curriculum targets from the 8B/4B she already runs.

### Phase 2 — four-substrate body
- Route ops off the ANE skeleton table to `gpu fn` (some attn/embed routes
  are already gpu-only in the emitter).
- `enc fn` for anything crossing peer boundaries; `pcc`/`mesh` for
  federation with treaty + witness on the same chain.
- CPU stratum owns control flow, curriculum, witness — already true.

### Phase 3 — fleet and context
- `shardize --world` the brain across ranknodes (weights shard across
  ranks; the 4B proves 36 layers + split head resident).
- Ring-buffer lanes already give unbounded `pos`; the 128k–1M tier is the
  paged/sequence-parallel path — context shards across lanes the way
  weights shard across ranks.
- Optional: export trained weights → `badapple-aneconvert`-compatible
  safetensors → the proven `.mlmodelc` 4B deployment path carries HIS
  weights at d2560 scale.

### Phase 4 — the artifact
- Benchmarks vs the Qwen resident path (32 tok/s reference), oracle
  ANE-vs-CPU parity, witness-chain audit.
- Demo: born on-chip → trained on ANE → generating → federating →
  every weight update hash-chained. That is the Apple CoreAI artifact.

## Immediate next action
Phase 0d in flight: post-fix ANE + CPU parity runs of the d=64
curriculum (`dev_seq_ane`, `dev_seq_cpu2`) and first d=128 60-epoch
CPU run (`d128_run`) — all on the defs-collision-fixed binaries, so
parity numbers are now meaningful for the first time. Then Phase 1:
d_model 128→256 tiles, multi-layer per-layer training, cross-substrate
partition. Nothing toy stays as the baseline.

## 2025-11-05 — text curriculum + residency findings (session 3)

### Real curriculum shipped (`gen_brain --text <file>`)
- Byte-level tokenizer (vocab=256, no tokenizer dependency). Corpus
  safetensors gains `cemb [256,d]` (frozen seeded embedding),
  `seq [n,d]` = cemb[tok[j]] per position, `tgts [n,256]` next-byte
  one-hots. Host slides `(ep*pos+i) % n` across real text; generation
  feeds `cemb[argmax]` back — true autoregressive byte decoding.
- Curriculum source: the repo's own BRAIN.md — the brain learns its
  own documentation. Deterministic per (spec, text-content) seed.
- Verified: compiles+runs on ANE (d=64, d=128) and CPU; 60-epoch CPU
  run shows next-byte hit rate climbing 0→~10% (random=0.4%),
  generation emits real byte fragments (" iss  ").
- vocab=256 needed 4 new tiles: attnv_256x128, attnv_128x128,
  outer256x128, outer128x128 — captured via `milc ane compile`,
  committed in mil_spec 32fc6da.

### ANE residency cap measured: ~128 programs/runtime
- run() instrumentation: 7325-node train_l0 = 6848 loads, ensure
  20.2s vs eval 1.3s — **program load is 94% of step time**.
- After resident-rebind (adopt = retain+wrap, ~µs), steady-state
  still ~6738 loads/call: only ~128 of 7325 programs fit ANE
  residency, so ~98% reload every pass regardless of streak length.
- eval itself is fast (1.3s/step) — the ANE computes; it's the
  per-node program-load path that costs.

### Speed architecture (item 3 resolution)
- CPU does throughput: same brain, --cpu, ~25ms/train-step at
  d=128 L=1 (16-step run in 0.42s). ~760x faster than ANE path.
- ANE is the sovereign-silicon proof substrate: witness-chained
  on-chip training + generation. CPU-pretrain → ANE-finetune/proof
  is the honest split until graph fusion lands.
- Fusion (multi-op HWX programs) is the real ANE-throughput fix —
  deferred; node-granularity emission is the current design.

### Chunked block-coordinate descent (`--rounds N`, default 6)
- Host loop: `rounds` alternations of phase streaks; each streak =
  `chunk=ceil(epochs/rounds)` epochs through ONE train program.
- Preserves block co-adaptation (pure phase-major dropped hit rate
  ~10x — blocks starved against frozen partners) while giving each
  program residency streaks (12 switches/run for L=1 vs 960 before).
- Each phase branch is self-contained (trainer→argmax→print→
  witness) — no dead standalone forward() call per step.

### Tile ladder status
- d_model 64/128/256/512/1024 ALL compile+run train_head/train_lN
  on ANE (hd=64 keeps state ops on verified tiles).
- gen_brain caps dmodel<=1024 (qwen embed-slice diagnostic is stale —
  revisit when scaling past).
