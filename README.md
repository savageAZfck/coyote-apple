# coyote-apple

**One language, four substrates — and every site they can run.**
The same `.cy` source compiles to CPU, Metal/GPU, the Apple Neural
Engine, and **ciphertext** (`enc fn`) — and now executes across **sites**:
`mesh fn` dispatches to a treaty'd peer, `pcc fn` to an attested remote
node, `pcc enc fn` is ciphertext evaluation on that node — over the Bad
Apple mesh wire (HMAC mutual auth, AES-256-GCM sealed packets),
verify-before-secrets, signed results, and one hash-chained witness
log that spans every machine a computation touches.

This is the source-available Apple edition of
[Coyote](https://github.com/savageAZfck/coyote): the public language stays
MIT and clean-room, this distribution carries the first-party engines
(`gpu fn` → Metal, `fn` → MIL/Espresso, `enc fn` → LWE ciphertext) and
the org surface (oracle, audit, witness chains, package signing,
universal dissection). Licensed under the
[SOURCE-AVAILABLE LICENSE](LICENSE) — read, audit, and build freely;
commercial use requires consent.

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

the federation surface — the five strata made literal:

```
coyote-apple meshnode --listen h:p          serve bound programs (mesh site)
coyote-apple pccnode --listen h:p           attested node (attest → bind →
                                            signed exec; enc×pcc Ct-quad eval)
coyote-apple peernode --listen h:p --treaty t.json [--programs dir]
                                            treaty exchange ∩, then `call
                                            <verb> <arg>` executes the bound
                                            .cy program of that name — host
                                            fns run in a staged Vm, program
                                            fns through the evaluator —
                                            every call re-checked against
                                            the intersection
coyote-apple shardize <f.cy> -o d --world N [--hosts a,b,...]
                                            per-rank dirs + shard.json
                                            (bad_apple ShardSpec) + world.json
coyote-apple fleet up --world world.json --serve h:p [--lanes N]
                                            the fleet as a unit: spawn every
                                            ranknode (deepest first, waiting
                                            for listeners) + the serving
                                            tier; pids in <dir>/fleet.pid
coyote-apple fleet down|status --world <dir>
                                            lifecycle on the pidfile
coyote-apple fleetcheck --world world.json
                                            conformance over the fleet:
                                            coverage contiguity, rank
                                            indices, auth — busy ranks
                                            report up (conn held), not down
coyote-apple serve --ranks h1[,h2...] --listen h:p [--lanes N]
                                            the serving tier: multi-client
                                            sessions pinned to KV lanes,
                                            lane-pool admission + LRU reap,
                                            rank-0 failover, hash-chained
                                            journal; waits for cold ranks
coyote-apple ranknode --shard <dir> --listen h:p [--lanes N] [--device ane]
                                            [--temp T] [--top-p P]
                                            a weight-carrying rank: the
                                            shardize rank dir resident on the
                                            ANE (layers+heads+KV), serving
                                            step/vstep/reset/rewind/generate
                                            (ids or text — tokenizer auto-
                                            discovered or --tokenizer)
                                            on the byte-exact mesh wire.
                                            generate streams token frames;
                                            prefix cache rewinds shared
                                            context; all lanes share one
                                            downstream conn (frame.lane
                                            selects the remote lane)
coyote-apple ask "prompt" --to h:p [--session s] [--max-tokens n]
                                            [--temp T] [--top-p P] | --ids "1,2,3"
                                            serving-tier client — text or ids,
                                            per-token streaming, per-request
                                            sampling carried on the frame
```

`demo/five_strata.sh` runs the whole stack live: three nodes, one
organism, five substrate×site strata, one witness chain.

`lib/badapple/` carries the Bad Apple organ layer — 29 `.cy` ports
(mesh wire, p2p, cert, ledger, protocol, config, scavenger, the
five-organ fusion `badapple_organs.cy`) — importable as
`import "badapple/x.cy"`, check-clean under the fork. The organism's
organs are library code now.

plus the full distribution surface — `new doc test spec watch gold
weights genmask oracle audit diff bench prof package verify unpack
inspect emit`
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
`host fn` → CPU — **× three sites** — `fn` local, `mesh fn` treaty'd
peer, `pcc fn` attested remote — with `reflex`, `det`/`stoch`,
`param`/`state`/`io`, `cap`/`invariant`/`provenance` constitutions,
`treaty` decls + `peer` handles (`cap treaty`-gated), and `witness`
committed journals that follow the computation across machines. `?`,
`import as`, and dtype generics are the semantics that make organs
composable.

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
resident context): coherent decode, CPU parity, 32 tok/s.

**The fleet is live-verified:** a real 28-layer model `shardize`d across
two ANE-resident ranknodes (layers 0–14 embed-side, 14–28 head-side),
fronted by `serve`, driven by `fleet up`/`fleetcheck`/`fleet down`, asked
through `ask` — streamed text in, streamed text out, per-request
`--temp`/`--top-p` reaching the head rank over forwarded `vstep` frames,
sessions on isolated lanes with prefix-cache rewinds, every request and
rank execution hash-chained into one witness journal. Peer verbs dispatch
into real `.cy` programs under treaty (`double(21) = 42.0` over the wire,
denied verbs refused before eval).

**Context is a ring, not a wall.** Lanes run sliding-window attention:
`pos` is the absolute stream position (unbounded — verified: `pos90`
witness entries past a 64-slot window), `slot = pos % seq_len`, RoPE on
the slot. Streams never die to "context full"; the attention span stays
`seq_len` wide (Mistral-style SWA). The resident span itself is a
compile-time property of the shard's `state<>` dims — bigger windows
re-emit through the model generator; the 128k–1M+ tier is the
paged/sequence-parallel path (context sharded across lanes the way
weights shard across ranks — the same law twice). Known edges: fp16
only, single-stream, argmax-grade output under base weights without a
chat template, per-op path fragile at this scale (that's the point of
the resident path), GPU resident decode exceeds 8GB on this machine
(~18 programs).

The grammar lives in the local `~/coyote` checkout — a private fork that
carries `enc fn`, `import "x" as m`, dtype generics (`fn f<D>`), and the
`?` propagate operator, none of which exists upstream. The engines and
the toolchain surface live here. `import "name"` resolves installed
packages from `~/.coyote/pkg`.

## license

Source-available, proprietary — see [LICENSE](LICENSE). Personal,
educational, research, audit, and security-review use is granted to
everyone; commercial use requires the Owner's written consent. The
public `coyote` language remains MIT.
