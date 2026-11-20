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

## The sites

Strata say *what* runs the math; sites say *where*. `fn` is local —
`mesh fn` is the same typed program whose execution site is a treaty'd
peer; `pcc fn` is an attested remote node. A site may stack an engine:
`pcc enc fn` is ciphertext evaluation on the attested node, `mesh gpu
fn` is Metal math on a peer.

```cy
mesh fn shard(x: f16[64]) -> f16[64] {   // runs on a treaty'd peer —
    x * 2.0                              // Bad Apple mesh wire: HMAC
}                                        // auth + AES-256-GCM packets

pcc fn relay(x: f16[64]) -> f16[64] {    // attested remote — the client
    x + x                                // verifies the node's manifest
}                                        // BEFORE the program crosses,
                                         // and every result is signed.

pcc enc fn boxd(x: f16[64]) -> f16[64] { // the sovereignty square:
    x @ w                                // client encrypts (LWE), the
}                                        // attested node evals the
                                         // ciphertext domain, client
                                         // decrypts — node never sees
                                         // plaintext, client never sees
                                         // remote weights.
```

Dispatch is capability-held: no site backend registered → no remote
execution, at `load` time. `--cpu` forces local evaluation for parity
checking. Remote nodes append their own witness entries to the frames
they answer; the local VM lands them on its hash chain — one
computation across N nodes produces one verifiable log.

## The federation

```cy
cap treaty

treaty fleet {                          // the law of a connection:
    allow status, ask                   // verbs this program will speak
    deny reset, kill                    // verbs it refuses outright
}

extern fn peer_connect(addr: str, tname: str) -> peer cap treaty;
extern fn peer_call(p: peer, verb: str, arg: str) -> str cap treaty;
extern fn peer_eval(p: peer, src: str, arg: str) -> str cap treaty;
extern fn peer_terms(p: peer) -> str cap treaty;

host det fn main() -> i64 cap treaty {
    let p = peer_connect("10.0.0.7:9733", "fleet");
    // handshake → treaty exchange → the connection binds to the
    // intersection of both sides' allow sets, minus either side's
    // denies. A `call` on a verb outside that ∩ never reaches the wire.
    print(peer_call(p, "status", ""));
    // `eval` ships source: the peer typechecks it, requires its
    // declared caps ⊆ the negotiated law, then runs `main(arg)`.
    // Mobile code with a compile-verified contract — the treaty is
    // the capability ceiling.
    print(peer_eval(p, "cap compute\nhost fn main(a: f64) -> f64 { return a * 2.0; }", "21"));
    return 0;
}
```

`peer` is a first-class handle type; `treaty` is a program-level decl
the host checker carries into `HProgram.treaties`; every `peer_*`
extern requires `cap treaty`, so a program that can't declare
federation can't open a socket. `import "std/treaty"` ships canonical
terms (`federation`, `observer`) plus the extern surface.

## Determinism flow — `det T` params and `verify()`

`det`/`stoch` on *functions* declares their contract; `det` on a
*parameter* declares the provenance it demands. The checker tracks how
values derive: any value produced by a `stoch` fn (directly or
transitively through the call graph) is stoch-provenanced, and a
stoch value at a `det`-marked parameter is a compile error:

```cy
host stoch fn sample_thought(prompt: str) -> str cap cortex { … }
host det fn ledger_write(e: det str) { … }

ledger_write(sample_thought(q));        // ✗ compile: stoch into det
ledger_write(verify(sample_thought(q))); // ✓ verify() is the one
                                        // sanctioned bridge — zero-cost,
                                        // it asserts the value was checked
ledger_write("constant");               // ✓ literals are det
```

Sampled output can never silently reach a ledger, a reflex input, or a
treaty argument — the hallucination-reaches-the-record bug class is a
type error.

## Weight lineage — `param … lineage <key>`

A `param` may cite a key from the program's `provenance { }` block; the
typechecker rejects lineage the program doesn't attest. Weights carry
passports — lineage is a load-bearing compile check, not a model card:

```cy
provenance { dataset: "med_corpus_v3", vetted_by: "clinical_review" }
param w: frozen f16[4, 4] lineage dataset      // ✓ declared
param v: frozen f16[4, 4] lineage secret_pipe  // ✗ compile error
```

## Replayable receipts — `coyote-apple replay`

Serve-side witness entries are re-executable objects: `serve-request`
receipts carry the full prompt + sampling params, `serve-result` a
content hash, `serve-resume` the id checkpoint. `replay` verifies the
journal's hash chain, restores the lane's recorded pre-state (`shared`
prefix resume or reset), re-executes against the live fleet, and diffs
the recomputed output hash against the receipt:

```
✓ replay seq 92 identical — 40 toks, thash e0f02a… matches receipt
```

Deterministic (temp 0) executions replay byte-exact; sampled requests
are flagged (byte-equality isn't a stoch contract); mid-stream rewinds
expose the documented ANE scratch residual (KV rebuilds exactly —
`hidden` ping-pong buffers need an upstream `mil_infer` hook to
serialize). A divergence report is a real finding, not a failure.

## Live minds — `coyote-apple mind`

KV state is a pure function of the token stream, so a lane's id history
IS its checkpoint:

```
mind snapshot --to h:p --lane N -o mind.bin    # the mind as a value
mind resume   --to h:p --lane N -i mind.bin    # rebuild KV on a lane
mind fork     --to h:p --from-lane A --to-lane B   # one thought, two
mind migrate  --from h:p --lane A --to h:p2 --lane B  # moves machines
```

Resume replays the id stream — downstream ranks rebuild KV as frames
flow; the whole pipeline resumes. Verified deterministic under argmax:
same checkpoint → same mind → same continuation.

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
package verify unpack weights doc watch emit shardize genmask inspect
referee certify conform caps fmt pkg lsp new
serve ask mind fleet fleetcheck ranknode meshnode pccnode peernode replay
```

- `oracle` — semantic parity gate across strata
- `prof` — per-op device timing
- `gold`/`diff` — golden regression + build diffs
- `audit`/`verify` — build provenance + package integrity
- `inspect` — universal dissection: real parsers over dozens of
  formats, dir/world audits, live URL targets; every call witnessed
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
