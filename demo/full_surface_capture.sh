#!/bin/bash
# full_surface_capture.sh — one-take staged run for the full-surface video.
# Every segment runs REAL commands; pauses mark clean cut points.
# Prereqs: fleet up on /tmp/realworld/world.json, both binaries built.
# The mesh/pcc/enc×pcc block self-stages via demo/five_strata.sh.
set -u
cd "$(dirname "$0")/.." || exit 1
export COYOTE_MESH_SECRET="${COYOTE_MESH_SECRET:-test-secret-42}"
export COYOTE_MESH_KEY="${COYOTE_MESH_KEY:-five-strata-demo-key-32bytes-minimum!!}"
export COYOTE_MESH_ENC=1
CA=./target/debug/coyote-apple
CY=/Users/savag3/coyote/target/debug/coyote
SERVE=127.0.0.1:9744
J=/tmp/realworld/serve_witness.jsonl
D=/tmp/fullsurface
P=/tmp/peerdemo

beat() { echo; echo "── $* ──"; sleep "${PAUSE:-3}"; }
run()  { echo "$ $*"; "$@"; sleep "${BEAT:-2}"; }

mkdir -p "$D" "$P/progs"

# ── 1. the fleet is real ──────────────────────────────────────────────
beat "fleet conformance — 28 layers, ANE-resident, 2 ranks"
run "$CA" fleetcheck --world /tmp/realworld/world.json
beat "live streamed decode through the authenticated wire"
run "$CA" ask "The capital of France is" --serve "$SERVE" --max-tokens 12 --temp 0

# ── 2. one source, the whole machine — the ciphertext substrate ───────
beat "the compiler sees the enc fn declaration"
run "$CA" spec demo/cipher_layer.cy || true
beat "substrate check — plaintext domain"
run "$CA" run demo/cipher_layer.cy --cpu
beat "same source, ciphertext domain — homomorphic eval"
run "$CA" run demo/cipher_layer.cy
beat "the honest constraint — a circuit it can't encrypt never builds"
run "$CA" run demo/cipher_reject.cy || true

# ── 3. one program, the whole fleet — shardize --world ────────────────
beat "one .cy source partitioned into ranks — the fleet is emitted"
run "$CA" shardize examples/qwen1x2.cy -o "$D/world" --world 2 \
    --hosts "127.0.0.1:9710,127.0.0.1:9711" || true
run cat "$D/world/world.json"

# ── 4. det/stoch flow — hallucination can't reach the record ──────────
cat > "$D/reject.cy" <<'EOF'
host stoch fn sample() -> i64 {
    return i64(rand() * 100.0);
}
host det fn ledger_write(e: det i64) -> i64 {
    return e;
}
host fn main() -> i64 {
    ledger_write(sample());
    return 0;
}
EOF
cat > "$D/accept.cy" <<'EOF'
host stoch fn sample() -> i64 {
    return i64(rand() * 100.0);
}
host det fn ledger_write(e: det i64) -> i64 {
    return e;
}
host fn main() -> i64 {
    ledger_write(verify(sample()));
    ledger_write(42);
    return 0;
}
EOF
beat "stoch value into a det param — refused by the compiler"
run "$CY" run "$D/reject.cy" || true
beat "verify() is the one sanctioned bridge"
run "$CY" run "$D/accept.cy"

# ── 5. lineage — weights carry passports ──────────────────────────────
cat > "$D/lineage_bad.cy" <<'EOF'
param w: frozen f16[64, 64] lineage secret_pipeline
io x: in f16[64]
io y: out f16[64]
fn main(x: f16[64]) -> f16[64] {
    x @ w
}
EOF
cat > "$D/lineage_ok.cy" <<'EOF'
provenance { dataset: "med_corpus_v3", vetted_by: "clinical_review" }
param w: frozen f16[64, 64] lineage dataset
io x: in f16[64]
io y: out f16[64]
fn main(x: f16[64]) -> f16[64] {
    x @ w
}
EOF
beat "undeclared lineage — refused"
run "$CY" run "$D/lineage_bad.cy" || true
beat "attested lineage — lowers to MIL"
run "$CY" run "$D/lineage_ok.cy" || true

# ── 6. mind — the thought moves ───────────────────────────────────────
beat "snapshot lane 1 — the mind as a value"
run "$CA" mind snapshot --to "$SERVE" --lane 1 -o /tmp/mind_video.bin
beat "resume onto lane 2 — fork: one mind, two continuations"
run "$CA" mind resume --to "$SERVE" --lane 2 -i /tmp/mind_video.bin
beat "two asks, one mind — identical continuations under argmax"
run "$CA" ask "List three numbers" --serve "$SERVE" --lane 1 --max-tokens 10 --temp 0
run "$CA" ask "List three numbers" --serve "$SERVE" --lane 2 --max-tokens 10 --temp 0

# ── 7. replay — the receipt is the reproduction ───────────────────────
beat "reset lane 3 — the receipt will record a clean start (shared=0)"
: > /tmp/empty.bin
run "$CA" mind resume --to "$SERVE" --lane 3 -i /tmp/empty.bin
beat "a fresh deterministic ask lands a replayable receipt"
run "$CA" ask "The capital of Japan is" --serve "$SERVE" --lane 3 --max-tokens 10 --temp 0
SEQ=$(tail -3 "$J" | python3 -c "import json,sys; [print(json.loads(l)['seq']) for l in sys.stdin if json.loads(l)['tag']=='serve-request']" | tail -1)
echo "  journaling seq $SEQ"
sleep "$BEAT"
beat "replay seq $SEQ — chain verify, pre-state restore, hash diff"
run "$CA" replay --journal "$J" --seq "$SEQ" --serve "$SERVE"

# ── 8. eval + treaty ∩ — code that travels under contract ─────────────
cat > "$P/treaty.json" <<'EOF'
{ "t": { "allow": ["double", "status", "eval", "compute"], "deny": ["patch"] } }
EOF
cat > "$P/caller_eval.cy" <<'EOF'
cap treaty
import "std/treaty.cy"
treaty t {
    allow double, status, eval, compute
    deny patch
}
host fn main() -> i64 {
    let p = peer_connect("127.0.0.1:9850", "t");
    print(peer_terms(p));
    let prog = "cap compute\nhost fn main(a: f64) -> f64 {\n  return a * 2.0;\n}\n";
    print(peer_eval(p, prog, "21"));
    let bad = "cap patch\nhost fn main(a: f64) -> f64 {\n  return a;\n}\n";
    print(peer_eval(p, bad, "5"));
    return 0;
}
EOF
pgrep -f "coyote-apple peernode" >/dev/null || {
    nohup "$CA" peernode --listen 127.0.0.1:9850 --treaty "$P/treaty.json" \
        --programs "$P/progs" > "$P/node.log" 2>&1 &
    sleep 1
}
beat "peer_terms — the negotiated law bound at connect, then peer_eval"
run "$CA" run "$P/caller_eval.cy" || true

# ── 9-10. the strata — mesh fn · pcc fn · witness wire · enc×pcc ──────
beat "five strata — remote execution, one ledger, the sovereignty square"
bash demo/five_strata.sh

beat "done — cut here"
echo "  every receipt is in $J (hash-chained) and /tmp/five_strata/*/witness.jsonl"
