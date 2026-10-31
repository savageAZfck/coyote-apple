# Coyote — the Apple Profile

A single-source language whose unit of compilation is a **verified
cross-silicon program**. One `.cy` file runs compute on the Neural
Engine, Metal, the CPU, and ciphertext — with parity enforced across
all of them and a committed witness trail behind every run.

## The strata

```cy
fn step(x: f16[64]) -> f16[64] {      // ANE — the Neural Engine
    softmax(x @ wq * 0.125) + x
}

gpu fn step_g(x: f16[64]) -> f16[64] {   // Metal — the GPU
    softmax(x @ wq * 0.125) + x
}

enc fn score(x: f16[4]) -> f16[4] {      // ciphertext — LWE-encrypted
    x + x * 2.0                          //   inputs, plaintext weights
}

host fn main() -> i64 {                // CPU — the organism layer
    let y = step(onehot(0, 64));
    print("argmax", argmax(y));
    return 0;
}
```

`fn` lowers to the Neural Engine through MIL/HWX. `gpu fn` lowers to
Metal. `enc fn` executes on ciphertexts — inputs are encrypted on
entry, the body evaluates homomorphically against plaintext params,
the result decrypts on exit. `host fn` is the CPU orchestration layer —
the organism's nervous system. `det`/`stoch` mark determinism contracts
the checker enforces; `reflex` marks always-on host fns.

## The constitution

```cy
param wq: frozen f16[64, 64]          // weight — sealed/trainable/frozen
state kv { k: f16[256, 64], len: i32 } // persistent tensor state
io x: in f16[64]                      // function boundary
cap fs, net::send                     // declared capabilities
invariant sealed_keys_unnameable      // constitutional clause
provenance { author: "bad_apple" }    // signed into the artifact
```

`cap` is a gate the checker enforces: an organ compiled `--pure`
touches nothing; `caps file.cy --deny net` proves it can't exfiltrate.
`invariant` and `provenance` ride into every `.cypack` signature.

## Modules

```cy
import "math.cy" as m                 // namespaced splice

fn transform(x: f16[4]) -> f16[4] {
    m::add_one(x @ w)                 // qualified call
}
```

`import "x"` splices a module's decls into the flat namespace.
`import "x" as m` namespaces them: top-level names become `m::name`,
and the module's own unqualified internal references are renamed to
match — scope-aware, so locals that shadow a decl stay locals.

## Packages

```bash
coyote-apple package prog.cy -o prog.cypack   # build + pack
coyote-apple pkg install prog.cypack          # → ~/.coyote/pkg/prog
coyote-apple pkg list                          # installed packages
```

A `.cypack` carries the compiled artifacts *and* `mod.cy` — the
combined source. `pkg install` drops it in the registry root, which
the toolchain puts on `CYO_PATH`, so `import "prog"` resolves
installed packages with no extra wiring.

## The verification verbs

```bash
coyote-apple referee prog.cy            # refereed divergence bisection
coyote-apple referee prog.cy --replay   # re-derive a verdict
coyote-apple certify prog.cy -o cert.json   # proof-carrying certificate
coyote-apple conform prog.cy --spec s.cyproto  # protocol conformance
coyote-apple caps organ.cy --pure       # capability policy gate
```

The referee runs a program on two strata (silicon + CPU reference),
compares committed witness chains, and bisects divergence in log(n)
challenge rounds — the Canetti refereed-delegation protocol as a
toolchain verb. `certify` emits a machine-checkable certificate:
compiles, parity, replay determinism, source hash. `conform` checks
the witness chain against a `.cyproto` session spec.

## Toolchain

```
check build run repl test bench prof oracle gold diff spec audit
package verify unpack weights doc watch emit shardize genmask
referee certify conform caps fmt pkg lsp new
```

- `oracle` — semantic parity gate across strata
- `prof` — per-op device timing
- `gold`/`diff` — golden regression + build diffs
- `audit`/`verify` — build provenance + package integrity
- `emit --swift` — Swift bridge generation
- `shardize`/`weights` — safetensors → ANE-shard conversion
- `fmt` — canonical formatter (idempotent, lossless)
- `lsp` — full language server: diagnostics, completion, hover,
  go-to-definition, semantic tokens

## The type surface

Tensor types are `dtype[dims]`: `f16[64, 64]`, `f32[K]`, `i32`,
`q4[...]`. Dims may be literal or generic (`fn f<M, N>(x: f16[M, N])` —
checked at call sites). Host types: `i64`, `f64`, `bool`, `str`,
`token`, `dfa`, plus `struct`, `enum`, `match`, `table` (host-only
data source), `extern` fns with capability bounds.
