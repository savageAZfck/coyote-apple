#!/bin/bash
# escalation_capture.sh — one-take staged run for the delivery-2 video.
# Every segment runs REAL commands; pauses mark clean cut points.
# Prereqs: fleet up on /tmp/realworld/world.json, both binaries built.
set -u
export COYOTE_MESH_SECRET="${COYOTE_MESH_SECRET:-test-secret-42}"
CA=/Users/savag3/coyote-apple/target/debug/coyote-apple
CY=/Users/savag3/coyote/target/debug/coyote
SERVE=127.0.0.1:9744
J=/tmp/realworld/serve_witness.jsonl
D=/tmp/detflow
P=/tmp/peerdemo

beat() { echo; echo "── $* ──"; sleep "${PAUSE:-3}"; }
run()  { echo "$ $*"; "$@"; sleep "${BEAT:-2}"; }

mkdir -p "$D" "$P/progs"

# ── 1. the fleet is real ──────────────────────────────────────────────
beat "fleet conformance — 28 layers, ANE-resident, 2 ranks"
run "$CA" fleetcheck --world /tmp/realworld/world.json
beat "live streamed decode through the authenticated wire"
run "$CA" ask "The capital of France is" --serve "$SERVE" --max-tokens 12 --temp 0

# ── 2. det/stoch flow — hallucination can't reach the record ──────────
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

# ── 3. replay — the receipt is the reproduction ───────────────────────
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

# ── 4. eval — mobile code under treaty ────────────────────────────────
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
    let prog = "cap compute\nhost fn main(a: f64) -> f64 {\n  return a * 2.0;\n}\n";
    print(peer_eval(p, prog, "21"));
    let bad = "cap patch\nhost fn main(a: f64) -> f64 {\n  return a;\n}\n";
    print(peer_eval(p, bad, "5"));
    return 0;
}
EOF
pgrep -f "coyote-apple peernode" >/dev/null || {
    nohup "$CA" peernode --listen 127.0.0.1:9850 --treaty "$P/treaty.json" \
        --program "$P/progs" > "$P/node.log" 2>&1 &
    sleep 1
}
beat "peer_eval — shipped source, treaty-gated caps, real execution"
run "$CA" run "$P/caller_eval.cy" || true

# ── 5. mind — the thought moves ───────────────────────────────────────
beat "snapshot lane 1 — the mind as a value"
run "$CA" mind snapshot --to "$SERVE" --lane 1 -o /tmp/mind_video.bin
beat "resume onto lane 2 — fork: one mind, two continuations"
run "$CA" mind resume --to "$SERVE" --lane 2 -i /tmp/mind_video.bin
beat "two asks, one mind — identical continuations under argmax"
run "$CA" ask "List three numbers" --serve "$SERVE" --lane 1 --max-tokens 10 --temp 0
run "$CA" ask "List three numbers" --serve "$SERVE" --lane 2 --max-tokens 10 --temp 0

# ── 6. lineage — weights carry passports ──────────────────────────────
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

beat "done — cut here"
echo "  every receipt is in $J (hash-chained)"
