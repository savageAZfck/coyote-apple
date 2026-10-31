# coyote-apple

**One language, four substrates.** The same `.cy` source compiles to CPU,
Metal/GPU, the Apple Neural Engine — and **ciphertext** (`enc fn` —
homomorphic eval against plaintext weights) — no CoreMLTools, no Apple
code generation — and the compiler can prove all of them compute the
same math.

This is the private Apple edition of
[Coyote](https://github.com/savageAZfck/coyote): the public language stays
MIT and clean-room, this distribution carries the first-party engines
(`gpu fn` → Metal, `fn` → MIL/Espresso, `enc fn` → LWE ciphertext) and
the org surface (oracle, audit, witness chains, package signing).

## the headline: resident ANE decode

`coyote-apple shardize` emits a full transformer as resident-pipeline
shards — per-layer `state<>` ports, ping-pong hidden buffers, vocab-split
logit heads — that `mil_infer::ane_pipeline` executes as **one
`execute_sync` per token**. The whole 28-layer model, KV cache, and head
live on the engine; nothing crosses the dispatch boundary mid-decode.

Same `examples/qwen28.cy` (Qwen3-0.6B), four execution paths, measured on
a fanless MacBook Air:

| backend | ms/token | tok/s | structure |
|---|---|---|---|
| CPU evaluator | ~4,300 | 0.23 | f32 reference |
| GPU — Metal | ~140 | ~7 | one Metal program per layer |
| ANE — per-op | ~48,000 | 0.02 | ~19k kernel dispatches per token |
| **ANE — resident** | **31** | **32** | one `execute_sync` per token |

The resident decode is **token-identical to the CPU reference**
(`14582, 271, 40, 1184, 311, 1855, 264, 13027` on both — coherent
English). Verified by `coyote-apple oracle --resident`.

## the bug worth reading

The first resident decode produced garbage while the same source ran
clean on CPU. A differential probe (`src/bin/diffprobe.rs`) bisected it
to a single op: `silu` on 3072-wide tensors. The shared lowerer lowers
`sigmoid`/`silu` on >1024-element tensors to a clamped degree-13 Horner
polynomial — whose last two coefficients (~4e-8, ~3e-10) exist only to
cancel the lower terms at the clamp boundary. In fp16 those coefficients
quantize to different values, the cancellation breaks, and `σ(6)`
evaluates to **−8.83** instead of 0.994. `silu(9.94)` came out as −87.8.

`shardize` now detects the chain and collapses it to a real `sigmoid`
op, which the e5 compiler accepts at these widths. Layer-level diff:
97.75 → 0.017.

## verbs

```
coyote-apple shardize <file.cy> -o <dir>       resident-ANE shard directory
coyote-apple oracle <f.cy> --resident <dir>    CPU-vs-resident parity gate
coyote-apple run <file.cy> [--cpu]             organism decode (witness chain)
diffprobe <file.cy> <shard_dir> [steps]        tensor-level CPU-vs-ANE diff
anepush <shard_dir> <tok,ids> [n]              resident decode smoke
```

the verification layer — the refereed-delegation stack as toolchain
verbs:

```
coyote-apple referee <f.cy> [--replay]         RDoC bisection over the
                                               committed witness chain
coyote-apple certify <f.cy> [-o cert.json]     proof-carrying certificate:
                                               compiles/parity/replay
coyote-apple certify --check cert.json         re-verify a certificate
coyote-apple conform <f.cy> --spec s.cyproto   session conformance over
                                               the witness chain
coyote-apple caps <f.cy> [--deny/--allow/--pure]  capability policy gate
```

the product surface — the parts that make it a language people use:

```
coyote-apple fmt <f.cy> [--check]              canonical formatter
coyote-apple pkg install|list|remove|search    the package registry
                                               (~/.coyote/pkg on CYO_PATH)
coyote-apple lsp                               full language server:
                                               diagnostics, completion,
                                               hover, go-to-def, semtoks
```

plus the full distribution surface — `new doc test spec watch gold
weights genmask oracle audit diff bench prof package verify unpack emit`
— documented in [APPLE.md](APPLE.md). Language guide: [LANGUAGE.md](LANGUAGE.md).

## language

```cy
import "math.cy" as m            // namespaced modules — m::name()

fn act<D>(x: D[64]) -> D[64] {   // dtype generics — D binds per callsite
    x @ w * 0.125
}

enum R { Ok(i64), Err(str) }
host fn chain() -> R {
    let a = fallible()?;          // ? — Ok binds, Err propagates
    return R::Ok(a);
}
```

Four strata — `fn` → ANE, `gpu fn` → Metal, `enc fn` → ciphertext,
`host fn` → CPU — with `reflex`, `det`/`stoch`, `param`/`state`/`io`,
`cap`/`invariant`/`provenance` constitutions, and `witness` committed
journals. `?`, `import as`, and dtype generics are the semantics that
make organs composable.

## how a shard directory works

- `layer_NN.mlmodelc/model.mil` — strict-shape MIL: packed coyote
  conventions redeclared as batch-(T,D) forms, `rms_norm` expanded to
  reduce_mean/rsqrt, one-hot KV scatters as additive `state + mask ⊗ row`
- `state<>` ports → private per-layer buffers; live inputs (`x`,
  `attn_mask`, `rope_cos/sin`, `kv_write_mask`) → shared buffers
- `head_NN.mlmodelc` — vocab-range slices of the lm_head (4×37984,
  Apple's own shard geometry), logits concatenated in order so index =
  token id
- `conversion_manifest.json` + `embedding.f16` → `ane_decode::Decoder`
  owns the token loop: embedding rows, RoPE tables, causal mask, KV
  one-hot, `push()` per position

## status

Working end-to-end on Qwen3-0.6B (28 layers, GQA, RoPE, SwiGLU, 64-token
resident context): coherent decode, CPU parity, 32 tok/s. Known edges:
fp16 only, single-stream, per-op path fragile at this scale (that's the
point of the resident path), GPU resident decode exceeds 8GB on this
machine (~18 programs).

The grammar lives in the local `~/coyote` checkout — a private fork that
carries `enc fn`, `import "x" as m`, dtype generics (`fn f<D>`), and the
`?` propagate operator, none of which exists upstream. The engines and
the toolchain surface live here. `import "name"` resolves installed
packages from `~/.coyote/pkg`.
