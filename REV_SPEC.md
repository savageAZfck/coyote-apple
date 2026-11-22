# REV_SPEC — the reversible stratum of Coyote-Apple

Status: Phase 1 landed (syntax + AST). Semantics land in Phases 2–3.
Parent plan: `REVERSIBLE.md`.

Reversibility is a **compile-time property**: a `rev fn` is bijective by
construction, carries a compiler-derived inverse, and every call is
oracle-verified `inv∘f = id`. This is the same shape as `hostck` — a
checker that makes irreversible programs unrepresentable.

## 1. Strata

`rev` is a **Target**, orthogonal to **Site** — exactly like `gpu`/`enc`:

```cy
rev fn f(...)                 // local reversible program
mesh rev fn g(...)            // reversible program on treaty-verified peers
pcc  rev fn h(...)            // reversible program on attested remote hw
```

Site dispatch (`mesh`/`pcc`) is unchanged; the reversible semantics are
the payload, the site is the wire.

## 2. Declarations

```cy
rev fn step(st: i32[8], inp: i32[8]) -> i32[8] inv step_inv {
    ...
}
```

- `rev fn` — declares a bijective tensor program.
- `inv name` — optional clause naming a hand-written inverse. When
  absent, the compiler **derives** the inverse by reversing the body and
  inverting each operation. When present, the named function is checked
  against the derived semantics by the oracle: `inv∘f = id` must hold on
  the declared domain or the program does not certify.

## 3. Statements

| Form | Semantics | Inverse |
|---|---|---|
| `let x = e;` | bind once, never rebind | unchanged (SSA discipline) |
| `var x = e;` then `x op= e;` | mutable cell | `op` restricted to reversible set |
| `x += e;` | `x := x + e` | `x -= e;` |
| `x -= e;` | `x := x - e` | `x += e;` |
| `x ^= e;` | `x := x ⊕ e` | `x ^= e;` (self-inverse) |
| `x <=> y;` | exchange two values | itself |
| `if t {S} else {T} fi a;` | dual-branch conditional | swaps entry/exit predicates |
| `uncall f(args);` | run derived inverse of `f` | `call f(args)` |
| `while`/`for` | reversible loops only | requires decreasing measure |

**Irreversible forms are rejected at compile time** (`revck`, Phase 2):

- `x = e;` overwrite (`Set`) — destroys information
- `x *= e;`, `x /= e;` — not invertible in the value domain
- `x` occurring in `e` of `x op= e` — update must be bijective in `x`
  for fixed `e` (Janus rule; `+=`/`^=` satisfy it, arbitrary `e(x)` does not)
- implicit fanout/duplication of non-reproducible values
- `stoch`, `param` mutation, `StateUpdate`, `ParamUpdate`, `AdamUpdate`
- `return` mid-body (reversible fns are single-exit)

## 4. Dual-branch `if`

Janus semantics, hardened:

```cy
if x < k { A } else { B } fi x >= 0;
```

Janus semantics (verified against the Janus/RL papers):

- `x < k` is the **entry test** — selects the branch.
- `x >= 0` is the **exit assertion** — its value on exit must **match
  the branch taken**: true after the then-branch, false after the
  else-branch. Mismatch = undefined (abnormal stop). The assertion is
  what makes the conditional backward-deterministic: in reverse it
  becomes the test that identifies which branch ran.
- Forward: evaluate test, run branch, assert exit condition.
- Inverse: assertion becomes the entry test, test becomes the exit
  assertion, branches execute inverted.

A single-branch `if` (no `else`) is illegal in `rev fn`: both directions
must be defined.

## 5. `uncall`

`uncall f(args)` executes the inverse of `f`. The compiler derives it;
the runtime **verifies** it:

```
uncall f(x)  ≡  { y = f(x); assert f⁻¹(y) == x; return f⁻¹(y) }
```

On CPU/reference eval this check is exact. On tensor backends it becomes
a witnessed parity claim (`rev-verified`) rather than a silent trust.

## 6. Arithmetic domain

v1 restricts `rev fn` to **integer/fixed-point** tensors. Note the
tensor dtype set is `f16`, `f32`, `i32`, `q4` — `i64` is a *host* type
and cannot appear in tensor signatures:

- `i32[N]` — the integer tensor type; xor/add/sub exact in eval
- `f16`/`f32` elementwise `+`/`-` are accepted but **not certified
  invertible** (fp rounding is not a bijection) — revck warns and the
  oracle's `inv∘f = id` check is the gate that decides.
- `mul`/`div`/matmul in *update position* are rejected; inside pure
  expressions they are permitted (expressions never mutate state, so
  they cannot destroy information — the oracle still gates fp exactness).
- Caveat: eval stores tensor values as `f32`, so `i32` elements are
  exact only to 2²⁴ — Phase 3 must either widen the rev value
  representation or restrict certified `i32` rev programs to the
  ±2²⁴ domain. This is the fixed-point discipline `enc_ckks` already
  practices, applied to reversibility.

This mirrors `enc_ckks`'s fixed-point discipline: reversibility wants
exact arithmetic, and the type system is where that gets enforced.

## 7. Verification ladder

1. **revck** (Phase 2) — static: SSA discipline, bijective updates,
   dual-branch shape, op whitelist, termination measures.
2. **oracle** — dynamic: `inv∘f = id` sampled over the declared domain;
   failure = compile error, not warning.
3. **certify/replay** — the witness chain of a rev program replays
   backwards: `uncall` of the whole trace uncomputes it.
4. **referee/conform** — cross-stratum parity: CPU reference vs ANE/GPU
   vs `enc` vs `mesh`/`pcc` must agree bit-exact on the reversible
   fragment, or the claim is `parity-unproven`.

## 8. What this is NOT (yet)

- Not zero-energy computing — that's the hardware horizon (Vaire/Ice
  River class). This spec defines the *semantics* that hardware would
  implement natively.
- Not a garbage-free guarantee on physical memory — `uncall` proves
  intermediates are uncomputed *in the model*; DRAM remanence is a
  separate problem.
- Not YALM/Theseus — Coyote's tensor-first syntax, site orthogonality,
  and witness infrastructure are the differentiators.
