# REVERSIBLE — `rev fn`: verified inversion as a Coyote-Apple primitive

> Status: **war plan**. Phase 0 (research) complete. Phases 1+ pending.
>
> Mission: make Coyote-Apple the first language/runtime where
> **reversibility is a compile-time, machine-verified property of the
> program** — not a convention, not a hope, not a paper claim. Every
> `rev fn` ships with a compiler-derived, oracle-verified inverse.
> `uncall f` runs any reversible program backwards. The witness chain of a
> `rev` organism is itself reversible — replay it in reverse and you
> uncompute its history.
>
> Scope: **coyote core** (AST/typeck/parser/eval — the language) +
> **coyote-apple** (strata lowering, oracle/certify/replay integration,
> gate emission — the runtime). Both repos, one primitive.

---

## 0. Why this and why now

Three independent lines converge:

1. **The theory is 50 years settled.** Bennett (1973): computation can be
   logically *and* thermodynamically reversible — in principle zero energy per
   step. Landauer's limit is physics, not engineering. Every computer ever
   built pays the erasure tax; reversible computing is the only known way out.
2. **The hardware just escaped the lab.** Vaire Computing's *Ice River* test
   chip (Sept 2025) demonstrated ~30% energy recovery with reversible logic +
   adiabatic power on 22nm CMOS; first reversible compute chip taped out,
   AI-inference processor on the ~2027 roadmap (IEEE Spectrum, Science News).
   Superconducting RQFP reversible logic demonstrated (2018). The silicon is
   coming — **what doesn't exist is the software stack.** No ISA, no
   language, no programming model. A language defined today IS the spec the
   hardware gets built against.
3. **Coyote-Apple already owns every hard subsystem this needs:**
   - `coyote/src/hostck.rs` — compile-time safety enforcement. Reversibility
     is the *same shape* of constraint as stoch→det provenance: track a
     property through the program, make violations unrepresentable.
   - `coyote/src/lower.rs` — tape-based autodiff records forward, walks
     backward. Compute→copy→uncompute is literally Bennett's construction.
   - `coyote-apple/src/enc_ckks.rs` — fixed-point arithmetic with scale
     tracking. Floats aren't invertible; fixed-point is. The discipline
     already exists.
   - `coyote/src/eval.rs` + `coyote-apple/src/oracle.rs` — CPU reference +
     parity gates. Inversion verification (`inv∘f = id`) rides this for free.
   - `coyote-apple/src/{certify,referee,replay,conform}.rs` — the verification
     ladder. `rev-verified` claims slot into the existing claim taxonomy.
   - `coyote-apple/src/{gpu_emit,shardize}.rs` — backend emitters. A
     `rev_emit` gate-netlist backend is the same pattern pointed at
     TOF/FRE primitives.
   - Witness chains (`coyote/src/vm.rs`) — a reversible program's history is
     itself reversible. Replay backwards = uncompute.

Nobody else has the toolchain *and* the verification machinery *and* the
will to attempt it. This is the moonshot the whole stack was secretly for.

---

## 1. Research foundation (verified, cited)

### Languages — the semantic prior art

- **Janus** (Lutz & Derby 1982; Yokoyama & Glück): the canonical clean
  reversible language. Constructs we adopt:
  - Reversible updates only: `+=`, `-=`, `^=` — **the LHS variable may not
    appear in the RHS expression** (the bijectivity rule).
  - `x <=> y` swap — trivially reversible.
  - Dual-branch conditional: `if test then S else T fi assert` — test must
    hold on branch entry, assertion on exit; inversion swaps their roles.
    (Coyote's `If` has a single cond — rev needs the dual form.)
  - `from e do S loop T until e` — counted loop, reversible counter.
  - `call f` / `uncall f` — **uncall executes the procedure's inverse
    automatically**: reverse statement order, invert each op. We productize
    this — but *verify* the derived inverse instead of trusting it.
- **Theseus** (James & Sabry): strongly-typed functional reversible language
  via type isomorphisms — "all well-typed programs are reversible." Proof
  that reversibility can be a *type-system* property — same mechanism as
  `det`/`stoch`.
- **R, SyReC, RedLang**: surveyed; Janus is the imperative reference,
  Theseus the type-theoretic one.

### Gates — the hardware target

- Generalized Toffoli `TOF(x₁…xₙ; xₙ₊₁)`: `target ^= AND(controls)` —
  self-inverse; Toffoli+NOT universal for all permutations with one ancilla.
- Fredkin `FRE`: controlled-SWAP, preserves Hamming weight.
- Libraries: NCT, Peres, MCT — RevLib: 457 benchmark realizations.
- Aaronson–Grier–Schaeffer (2016): complete classification of reversible gate
  sets; any n-bit reversible circuit compresses to ≤2ⁿ·poly(n) gates + O(1)
  ancilla.

### Garbage management — Bennett's trick

- **Compute → Copy-out → Uncompute**: forward (garbage on ancilla),
  CNOT-copy outputs to fresh wires, run the inverse sequence in reverse —
  every ancilla returns to zero *by construction*.
- ~2× gate cost; naive version needs ~#gates memory. Knill's pebble game
  (1995): poly-log space at polynomial time overhead — expressible directly
  in `state`/snapshot machinery.
- Janus achieves *clean* garbage-free computation natively via update
  discipline — no Bennett needed for well-formed programs.

### Arithmetic — the reversible ALU

- Cuccaro adder: MAJ/UMA subcircuits, in-place, single ancilla, O(n) Toffoli.
- Garbage-free multipliers: Mealy-machine constructions, O(n²) gates.
- **ARX insight (the demo unlock):** add-rotate-XOR ciphers — ChaCha, Speck,
  Simon — are already near-reversible. A reversible ARX cipher is the
  perfect first `rev fn` demo: real utility, invertible by design.

### Hardware status

- Vaire Ice River (22nm CMOS, Sept 2025): ~30% energy recovery demonstrated;
  first reversible compute chip taped out; AI-inference processor ~2027
  roadmap, ~4000× efficiency claim (10-15yr horizon).
- RQFP adiabatic superconducting logic: reversible full adder (2018).
- HKI-inductor power recycling for cryogenic reversible circuits (2025).
- **Implication:** interim target is CPU/GPU/ANE *simulation* with energy
  accounting; the language/IR defined now IS the spec the silicon gets built
  against.

---

## 2. The primitive — syntax and semantics

### 2.1 `rev fn` — a new compute stratum

`Target::Rev` joins `Target::{Ane, Gpu, Enc}` in `coyote/src/ast.rs` —
orthogonal to `Site` exactly like the others, so `mesh rev fn` /
`pcc rev fn` come free: reversible programs can execute on treaty'd peers
and attested enclaves.

```cy
rev fn chacha_round(state: u32x16, key: u32x8) -> u32x16
```

A `rev fn` body is constrained to the **reversible statement set** (checked
by `revck.rs`, Phase 2). Tensor `rev fn`s manipulate integer/fixed-point
values — floats excluded v1 (not invertible; `enc_ckks` discipline applies).

### 2.2 `uncall` — compiler-derived, verified inverse

```cy
rev fn step(s, x) { ... }        // forward
uncall step(s', y)               // runs the derived inverse
```

Janus semantics, hardened: the compiler *derives* the inverse by reversing
statement order and inverting each op (swap↔swap, `+=e`↔`-=e`, `^=e`↔`^=e`,
dual-branch if swaps test/assertion roles). Where auto-derivation fails, an
explicit `inv name = ...` clause satisfies the checker — and the oracle
verifies `inv∘f = id` numerically either way.
**No language before Coyote-Apple ships verified program inversion.**

### 2.3 Reversible statement set (revck whitelist)

Adopted from Janus, mapped onto `coyote/src/ast.rs` `Stmt`:

| Coyote today | rev form | rule |
|---|---|---|
| `let mut x = e` | `let x = e` only | single-assignment; `mutable` rejected except via update ops |
| `x = e` (Set) | **rejected** | overwrite is erasure — irreversible |
| `x += e`, `x -= e` | allowed | LHS ∉ free-vars(e) — bijectivity rule |
| — | `x ^= e` | new `AssignOp::Xor` — self-inverse |
| — | `x <=> y` | new `Stmt::Swap` — trivially reversible |
| `if c {} else {}` | `if c then {} else {} fi a` | dual test/assertion — new AST form |
| `while`/`for`/`match` | restricted forms | `from n do S loop T until c` counted loop; match → nested dual-ifs or rejected v1 |
| `break`/`continue` | **rejected** | non-local exits destroy invertibility |
| `state k { ... }` writes | bijective updates only | `Store` ops restricted like `Assign` |
| `return` | tail-position only | mid-body returns rejected |

### 2.4 The checker — `revck.rs`

New module in `coyote/src/` parallel to `hostck.rs` (~same size class, same
diag style). Enforces, in order:

1. **SSA discipline** — no rebinding; `mutable` lets rejected except
   rev-update targets.
2. **Update bijectivity** — `x ⊕= e`: `x` ∉ free-vars(e); tensor `Store`:
   index exprs contain no written vars.
3. **Control-flow reversibility** — dual-branch ifs well-formed; loops have
   reversible counters + exit assertions; no break/continue/match v1.
4. **Op whitelist** — only invertible primitives: xor/bitwise family,
   integer/fixed-point add/sub, swap, reversible comparisons, declared
   `rev fn` calls, `uncall`.
5. **Fanout accounting** — every bound value consumed exactly once per
   reversible path (or via explicit `dup` into fresh bindings —
   Theseus-style information preservation).
6. **Termination** — loops require a decreasing measure the checker can
   verify (or explicit `decreases e` clause).

### 2.5 Inversion verification — rides existing machinery

- `coyote/src/eval.rs`: rev programs execute forward *and* backward
  natively — inverse ops are the same ops in reverse order. `uncall` in
  eval is free correctness checking.
- `coyote-apple/src/oracle.rs` + `certify.rs`: emit `rev-verified` claims —
  "∃ inv: inv∘f = id on domain D" — re-derived by `--check` like every
  other claim.
- **Witness chains are reversible**: a `rev` organism's witness log
  replays backwards to uncompute its own history. `die()`/`ckpt()`/
  `migrate` on rev organisms inherit reversibility — a migrated organism
  can *un-migrate*.
- `coyote-apple/src/replay.rs`: rev witness chains replay in *both*
  directions — forward re-executes, backward uncomputes.

---

## 3. Phases

### Phase 1 — Spec + AST (coyote core)
- `REVERSIBLE.md` (this doc) → `REV_SPEC.md` formal semantics in
  coyote-apple docs.
- `coyote/src/ast.rs`: `Target::Rev`; `AssignOp::Xor`; `Stmt::Swap`;
  dual-branch `If` node (or `Fi` desugar); `uncall` stmt/expr form; `inv`
  clause on `Fn`.
- `coyote/src/lexer.rs`/`parser.rs`: `rev`, `uncall`, `<=>`, `^=`, `fi`
  tokens; `rev fn` decl path through `fn_decl_target`.
- `coyote/src/typeck.rs`: rev programs shape-checked like `fn`
  (integer/fixed-point dtypes only — reject fp16/f32 in rev bodies v1).

### Phase 2 — `revck.rs` checker (coyote core)
- Port the six rule families from §2.4; diagnostics match `hostck.rs` style.
- Self-test corpus: Janus examples (fib pairs, factorial, hanoi) +
  deliberately-irreversible programs that must be rejected.

### Phase 3 — Reversible eval semantics (coyote core)
- `eval.rs`: rev ops evaluate bidirectionally; `uncall` executes the
  derived inverse; `Store` writes verified bijective at runtime too
  (defense in depth — checker proves, runtime confirms).
- `rev_lower.rs` (sibling to `lower.rs`): inverse-derivation pass —
  statement reversal + op inversion table.

### Phase 4 — Verification ladder integration (coyote-apple)
- **Landed:** `revcheck` (`src/revcheck.rs`) — discovers every
  `rev fn`/`{name}_inv` pair (explicit `inv` or synthesized), feeds
  deterministic test tensors through `Evaluator` forward+inverse,
  reports max|Δ|. Wired as `coyote-apple revcheck <file.cy>`.
  Proven live: `roundtrip.cy` passes max|Δ|=0; `v1.cy` (mixed
  f32 `+=`/`^=`/`-=`) **fails with max|Δ|=8.0** — the gate caught
  real non-bit-exactness that static revck cannot see.
- **Landed:** `rev host fn` pairs verify through the VM —
  `host_rev_pairs`/`host_pairs_check`/`host_roundtrip` synthesize
  i64/f64/bool/str args, call forward + `{name}_inv` via
  `Vm::call_fn`, demand exact state recovery. `revcheck` and
  `certify`/`certify --check` cover both strata.
- **Landed:** `certify` emits a `reversible` claim
  (`held`/`pairs`/`max_diff`/`tol`); a failed roundtrip fails
  certification; `certify --check` re-derives the claim and
  invalidates on source change or broken inversion. Verified on
  `seal.cy`: CERTIFIED, all claims re-derived held.
- Still open: `oracle.rs` rev forward+backward parity, `replay.rs`/
  `conform.rs` backward witness replay, `referee.rs` rev bisection.

### Phase 5 — Reversible stdlib (`revlib`)
- `swap`, reversible add/sub (Cuccaro-style MAJ/UMA decomposition for
  bit-vectors), Bennett combinators as library patterns:
  `compute_copy_uncompute(f, copy_out)` — the garbage-management idiom
  made a first-class combinator.
- Pebble-game state allocator sketch: `state` blocks sized by Knill's
  recursion — the plan for when rev programs get big.
- **Prereq landed:** pure bitwise natives — `bitand`, `bitor`,
  `bitxor`, `shl`, `shr` (logical), `rotl`, `ror` on `(i64,i64)`.
  Appended to `NATIVE_NAMES` (indices stable), typed in
  `native_ret`, executed in `vm::native`. `PURE_NATIVE_NAMES`
  marks them the only calls legal inside `rev host fn` bodies —
  they are the ARX primitive surface.

### Phase 6 — Demo: reversible cipher — LANDED

Two demos in `examples/`:

- **`seal.cy`** — single-thread keyed permutation (v1 discipline).
- **`arx.cy`** — `rev host fn arx8(st: Pair, key: i64) -> Pair`:
  8-round ARX-Feistel on a 2-word struct state. Each half-round
  mixes the other word via add/rotate/xor (`x += y; x ^= rotl(y,7);
  x ^= k` then `y += x; y ^= ror(x,11); y ^= k`), round keys from
  an 8-let key schedule. `uncall arx8` is the exact inverse —
  ciphertext IS the state, no workspace, no Bennett garbage.
  `arx_mv.cy` shows the minimal self-mixing core.

Verified end-to-end on `arx.cy`: encrypt → witnessed run →
`uncall` → byte-identical plaintext; `revcheck` max|Δ|=0;
`certify` → CERTIFIED; `certify --check` re-derives all claims.

Key enabler discovered during implementation: `rotl`/`ror` are
pure natives taking `(i64, i64)` — `x ^= ror(y, 7)` rotates the
*other* var legally, so real rotation-heavy ARX needs no
workspace decomposition at all. Rotation-of-self
(`x += rotl(x, r)`) remains banned by the per-var Janus rule —
Speck's exact `x = (ror(x,α)+y)⊕k` shape still needs the
workspace-field pattern (caller-seeded scratch struct fields).

### Phase 7 — Gate-level lowering (the hardware spec, coyote-apple)
- `rev_emit` (sibling to `gpu_emit.rs`/`shardize.rs`): decompose rev ops
  to TOF/FRE/CNOT/NOT netlists — the artifact a reversible chip consumes.
  Emit as netlist IR data, not runnable silicon — **this IS the ISA spec
  Vaire-class hardware needs.**
- Energy accounting on CPU sim: count "erasure events" (must be zero for
  clean rev programs) — the metric reversible hardware will optimize.

### Phase 8 — Ship it
- `coyote` + `coyote-apple` crate publishes with `rev` feature; examples/
  (fib, cipher, Bennett demo); LANGUAGE.md section; the claim: *first
  language+runtime with compiler-verified program inversion.*
- Optional capstone: `rev` organism — a CY4B-class mind whose every
  transition is uncomputable — "a mind you can run backwards."

---

## 4. Risks and honest constraints

- **No reversible hardware yet.** Mitigation: the primitive is *semantic* —
  CPU eval verifies inversion today; value (verified bijectivity,
  backwards execution, uncomputation for secrets/debugging) lands
  immediately. Gate lowering is the spec-for-future, not a blocker.
- **Floats aren't invertible.** v1 restricts rev bodies to integer/
  fixed-point tensors. ARX-style integer math covers ciphers, hashing,
  sorting networks, reversible interpreters — plenty for the demo tier.
- **Auto-inversion isn't total.** Some approved ops can't auto-invert;
  `inv` clause + oracle check is the escape hatch, same trust model as
  `param` bindings.
- **Scope discipline.** v1 = integer/fixed-point tensor `rev fn` + host-side
  `uncall` where derivable. `match`, `break`, floats, GPU/ANE rev
  lowerings explicitly out of scope until Phase 6 lands.
- **Two-repo coordination.** Core language changes (`coyote`) and runtime
  integration (`coyote-apple`) must land in lockstep — `Target::Rev`
  needs both a parser path and a registered backend or it won't compile.

---

## 5. The claim this buys

When Phase 6 lands: **the first programming language in history where every
reversible program carries a compiler-derived, oracle-verified inverse —
and its execution history can be replayed backwards to uncompute itself.**

That is not a feature. That is a new category of computation — and the
software spec for the only computing paradigm that can beat Landauer's limit.

## Phase 3 design notes (locked before implementation)

Pipeline facts (verified):
- `typeck::check_all` → `collect()` puts every `Decl::Fn` in `ck.fns` →
  `program_for(f)` → `self.body(&f.v.body, env, dims, 0, letmap)` produces
  `(Vec<TStmt>, Option<Typed>)`. Tensor fns are straight-line: TStmt has
  Let/Assign/ParamUpdate/AdamUpdate/StateUpdate/Expr only — no control flow.
- `eval.rs`: `Evaluator::run(inputs)` at ~L120; TStmt eval at ~L1279
  (`Assign` arm matches `AssignOp`, `Xor` already implemented as bitwise
  xor on f32 bits); `needs_grad_d` traversal at ~L1219 needs arms for new
  TStmt variants.
- Tensor dtypes: `f16 f32 i32 q4` only. `i64` is HOST-only — rev tensor
  ints are `i32`. Eval stores values as f32 → certified i32 exact to ±2²⁴
  until the value repr widens (REV_SPEC §6 caveat).

Inverse derivation (`rev_lower.rs`, coyote core — dirty-local, never pushed):
- Input: `TProgram.body: Vec<TStmt>` of a `Target::Rev` fn.
- Output: inverted `Vec<TStmt>` — Bennett order: reverse sequence,
  invert each state update (`Add`↔`Sub`, `Xor`→`Xor`, `Set`/`Mul`/`Div`
  impossible — revck already rejected them).
- Pure `let`s are effects-free: keep them evaluable in the inverse stream
  (they recompute from preserved bindings; Janus rule guarantees RHS
  independence). Simplest correct v1: emit inverse = reverse of
  state-updating stmts only, with lets re-emitted lazily if needed.
- `TStmt::Swap { a, b }` — new variant; tensor-level exchange of two
  named bindings. Ripples: typeck body() arm (replace placeholder for
  Swap only), eval arm (env entry swap), needs_grad arm, lower.rs +
  gpu_emit.rs + wasm.rs + vm.rs TStmt matches (lower/gpu_emit may error
  "swap not lowerable" until rev_emit exists — CPU eval is the semantic
  reference anyway).
- `RevIf`/`Uncall` on TENSOR stratum: defer — tensor fns are
  straight-line; control-flow reversal belongs to host-stratum rev
  (Phase 4+). revck should eventually reject them inside `rev fn` bodies
  OR typeck maps tensor `uncall` to an expr-level inverse call. Decide
  when TStmt::Call inverse form is designed.
- `Evaluator::run_inverse(inputs)` — executes the derived inverse program;
  oracle gate: `run_inverse(run(x)) == x` sampled over domain.

Verification (Phase 4, coyote-apple BSL side):
- `oracle`: sample domain, assert inv∘f = id bit-exact on i32 fragment.
- `certify`/`replay`: rev witness chains replay backwards.
- `referee`/`conform`: CPU-reference vs strata parity = `rev-verified`
  claim only when bit-exact.

## Phase 3 design CORRECTION (deeper analysis — supersedes notes above)

Naive "reverse the TStmt list" is NOT executable for tensor rev fns.
Verified by working the semantics:

- Tensor rev fn is functional: `args → ret`. Vars are internal scratch.
- Var-inits (`var acc = st`) seed from ARGS. In the inverse direction
  args are OUTPUTS — unknown until the end. You cannot seed the
  trajectory backwards from nothing.
- Lets snapshot values. A let reading an arg or an evolving var captures
  a point-in-time value the inverse cannot always reconstruct.

What DOES invert cleanly (proven):

- Reversed updates: `x op= e` → `x op⁻¹= e`. The Janus rule (LHS ∉ RHS)
  guarantees e evaluates identically at the mirrored point — every other
  var holds its forward-time value because later updates are already
  undone (Bennett ordering).
- Lets reading PARAMS (constants): time-invariant, safe anywhere.
- Lets reading VARS: safe IF re-emitted at the mirrored position — vars
  are at exactly S_j when the inverse reaches that point.
- Lets/var-inits reading ARGS: UNSAFE — args are outputs in reverse.

V1 signature discipline (to encode in revck/rev_lower):

- `uncall f(out, ctx...)` — the caller supplies the output PLUS any
  context args (keys/nonces) the forward call used. The inverse program's
  io-ins = {out, ctx...}, io-outs = the bijective state args.
- Var-inits that read state args become inverse SEEDS: `var acc = st`
  inverts to `var acc = out` when `acc` is the returned trajectory —
  i.e. v1 requires the rev fn to return the final mutable state (single
  accumulator thread or state encoded in ret).
- revck must reject: let/var-init RHS referencing bijective-state args
  (except the seed position), update RHSs referencing lets that capture
  args, multi-trajectory returns that don't encode all final var state.
- Params (param decls) are constants known to both directions — always
  safe.

This is exactly why Janus uses pass-by-reference cells: all state is
explicit, nothing hides in args. Coyote's tensor functional form needs
the seed discipline above to match.

Implementation order:
1. TStmt::Swap — DONE (eval/lower/enc_ckks/gpu_emit arms all land).
2. revck: add arg-capture rules — DONE, stricter than sketched: v1
   enforces `rev fn f(state: T, ctx...) -> T` — args[0] is the state
   thread, args[1..] are ctx passed through to the inverse; exactly one
   `var` seeded by identity `var v = <state arg>`; lets may capture
   only params/ctx/other lets (no state arg, no var); every update
   targets the var and its RHS may not read the state arg; `<=>` and
   `if..fi` rejected until multi-thread rev fns land; tail must be the
   var. Consequence (documented, honest): v1 tensor rev bodies are
   constant-delta updates — `v op= f(params, ctx, lets)` — a
   keystream-xor/affine class. Var-var interaction (ARX quarter
   rounds) needs either multi-var state encoding or the host stratum —
   Phase 4+.
3. rev_lower — DONE: `src/rev_lower.rs` derives the inverse `Fn` on
   the AST (reversed updates with inverted ops, lets re-emitted in
   forward order since they are time-invariant, seed `var v = out`,
   tail `v`), and `check_all` synthesizes it as `Decl::Fn` so it
   compiles through the normal pipeline as `{name}_inv` (or the
   `inv name` clause's target — hand-written inverses skip derivation).
   Inverse fns carry `Target::Rev`.
4. RevBackend — DONE (`coyote-apple/src/rev_backend.rs`): emit produces
   the standard io plan + `rev_manifest.json`; load returns an
   `Evaluator`-backed `BackendExec` — the reference reversible engine.
   Registered for `Target::Rev` in main.rs alongside Gpu/Enc.
5. EXECUTABLE PROOF — DONE: `/tmp/revtests/roundtrip.cy` (xor chain)
   and `roundtrip_arith.cy` (+=/-= chain) run `step` forward then
   `step_inv` and recover the input bit-exact — `inv∘f = id` executes
   on the reference engine, not just checks.

Known semantic caveat (documented in REV_SPEC): eval stores f32, so
`i32` rev programs are exact only to ±2²⁴, and mixing `^=` with
`+=`/`-=` on the same thread can round — xor chains and small-int
arithmetic chains each round-trip exactly; mixed chains need the
i64/host stratum or a fixed-point dtype (Phase 5 revlib decision).

## Phase 3b — uncall + host reversible surface (DONE)

- `Expr::Uncall(name, args)` — expression-position `uncall f(args)`.
  Parser handles it in `atom()`; hostck resolves through `rev_invs`
  (built from `Decl::Fn` targets): `inv name` clause → declared name,
  else derived `{name}_inv`. Compiles via the normal `call()` path →
  `HE::CallAne` dispatch. Non-rev targets rejected: "`uncall x` —
  `x` is not a `rev fn`".
- `Stmt::Uncall` — statement position dispatches the inverse and
  discards the result (Janus `uncall`).
- Tensor stratum rejects `Expr::Uncall` explicitly — rev fns are
  self-contained; composition happens at the host layer.
- `Stmt::RevIf` on host — compiles to `HS::If` with the assertion
  appended to both branches as `if assertion {} else { die() }`.
  Forward semantics verified: true assertion passes, false assertion
  kills the organism with a witnessed `die`. Inverse-routing (swap
  test/assertion roles) lands when host rev fns exist.
- `Stmt::Swap` on host — desugars to `var t = x; x = y; y = t`
  through an always-true `HS::If` guard (no HS::Seq node yet);
  same-type mutable vars only. Verified self-inverse.
- Verified live: `uncall step(res,k,j)` roundtrip ok=1; `inv mystep`
  clause resolves the hand-written inverse (no `step_inv` derived);
  stmt+expr uncall both dispatch; RevIf assert-fail dies witnessed.

### Phase 3c — `rev host fn` (landed)

Bijective host functions on scalar host types (i64/f64/bool/str),
same signature discipline as tensor `rev fn`:

```cy
rev host fn xorshift(st: i64, k: i64) -> i64 {
    var v = st;
    v ^= k;
    v += k;
    v ^= k;
    v -= k;
    return v;
}
```

- `HFn.rev` flag; parser accepts `rev host fn` (distinct from
  `rev fn` — tensor stratum).
- `revck::check_hfn` enforces: args[0] is the state thread (ret type
  must equal it), exactly one `var` seeded `var v = <state arg>`,
  updates target only `v` with `+=`/`-=`/`^=`, lets may not capture
  state arg or `v`, **all calls banned** (host fns have effects —
  no constant-delta escape hatch like tensor rev fns), `RevIf`
  recurses with branch-local env scoping, only legal return is
  `return <v>` as the last statement.
- `rev_lower::inverse_hfn` derives `{name}_inv` as a synthesized
  `Decl::HostFn`: re-seeds `var v = out`, hoists pure lets, reverses
  statements with inverted ops, emits `return v`.
- `RevIf` inverse (Janus, verified against the spec):
  `if t {S} else {T} fi a` → `if a {inv(S)} else {inv(T)} fi t`.
  **Branches do NOT swap** — `a` is true iff the then-branch ran,
  so inverse-then undoes S. (First implementation swapped them;
  caught by re-reading REV_SPEC.md §4 against the Janus semantics.)
- hostck RevIf lowering corrected to true Janus semantics: the
  assertion must hold after the then-branch and **not hold** after
  the else-branch (`if a {} else {die}` / `if a {die} else {}`).
  Previously it asserted true after both — wrong discriminator.
- New `Op::XorI` in the VM (`i64 ^`), emitted for `^=` on i64,
  wasm parity via `i64.xor` (0x85); `^=` restricted to i64 in
  hostck. This is the ARX primitive — i64 xor was missing entirely.
- `uncall` resolves `rev host fn` → `{name}_inv` through the same
  `rev_invs` map; host→host calls dispatch normally.

Verified live (`/tmp/revtests/host_rev.cy`, `host_rev_if.cy`):
i64 `^=`/`+=`/`-=` roundtrip bit-exact; `RevIf` roundtrips on BOTH
branches through `uncall`; negatives reject (impure calls,
state-capturing lets, wrong return, `=` overwrite).

Update: pure bitwise natives (`bitand`/`bitor`/`bitxor`/`shl`/
`shr`/`rotl`/`ror`) are now legal inside `rev host fn` bodies —
the only call form revck admits. `examples/seal.cy` exercises
them end-to-end.

### Multi-var state (landed)

`rev host fn` on a **struct state** unlocks real ARX:

```cy
struct Pair { x: i64, y: i64 }
rev host fn arx(st: Pair, k: i64) -> Pair {
    var x = st.x;          // one var per field, seeded st.<field>
    var y = st.y;
    x += y;                // RHS may read OTHER state vars —
    x ^= rotl(y, 7);       // per-var Janus rule, not global
    y += bitxor(x, k);     // self-mixing is now legal
    y ^= shr(x, 11);
    return Pair { x: x, y: y };   // packs every state var
}
```

- revck: var↔field bijection (every field exactly one var, every
  var state-seeded), per-var LHS∉RHS, lets may not capture ANY
  state var or `st`, `fi` predicates may read vars (time-slices
  match across the inverse) but never `st`, `<=>` allowed between
  state vars (self-inverse — the Feistel exchange / rotation
  landing pad), branch-local `var`s banned, return must pack all
  fields bound to their seeded vars.
- `inverse_hfn` re-seeds `var x = out.<field>` per var, hoists
  pure lets (including inside `fi` branches — fixed a latent
  inverse bug where branch lets were dropped), reverses updates,
  repacks `S { field: var }`.
- ctx-seeded mutable vars stay banned — finals unrecoverable
  (Bennett garbage problem). Workspace patterns that need scratch
  state must be struct fields the caller seeds (e.g. 0).
- revcheck `synth_val`/`val_diff` handle `RTy::Struct` recursively.

Verified: `arx.cy` (cross-mixing), `ok_swap.cy` (`<=>` on state
vars), negatives reject (ctx-seeded var, state-capturing let,
missing/wrong return field, `st.` read in update, branch var,
self-rotation `x += rotl(x,7)`).

Still open: host `uncall`/user-fn calls inside rev host bodies
(banned v1 — needs callee-inverse synthesis), tensor multi-var
state, Bennett uncomputation scheduling for workspace-carrying
bodies, `<=>` on tensor stratum beyond the v1 single-thread ban.
