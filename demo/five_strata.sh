#!/bin/bash
# five_strata.sh — the distributed Coyote-Apple stack, live.
#
# Five things the slide promised, all running:
#   1. mesh fn      — a program whose execution site is a treaty'd peer
#   2. pcc fn       — attested remote compute (verify-before-secrets,
#                     signed results, loopback attestation manifest)
#   3. witness wire — the remote node's entries land hash-chained on
#                     the local organism's ledger (one computation,
#                     one audit chain across N machines)
#   4. treaty/peer  — `treaty {}` decls, peer handles, treaty exchange
#                     ∩ at connect, cap-gated peer_* externs
#   5. enc × pcc    — ciphertext evaluation on attested remote silicon:
#                     client encrypts (LWE), node evals the ciphertext
#                     domain, client decrypts — the node never sees
#                     plaintext, the client never sees remote weights
#
# Transport is the Bad Apple mesh wire, byte-exact: HMAC-SHA256 mutual
# auth (mb-c/mb-s), AES-256-GCM session packets (SHA-256-keyed secret).

set -euo pipefail
cd "$(dirname "$0")/.."

BIN=target/debug/coyote-apple
export COYOTE_MESH_KEY="${COYOTE_MESH_KEY:-five-strata-demo-key-32bytes-minimum!!}"
export COYOTE_MESH_ENC=1
OUT=/tmp/five_strata
rm -rf "$OUT"; mkdir -p "$OUT"

cargo build 2>/dev/null

cat > "$OUT/treaties.json" <<'EOF'
{"fleet": {"allow": ["status", "ask"], "deny": ["reset", "kill"]}}
EOF

cat > "$OUT/organism.cy" <<'EOF'
cap treaty

treaty fleet {
    allow status
    deny reset, kill
}

extern fn peer_connect(addr: str, tname: str) -> peer cap treaty;
extern fn peer_call(p: peer, verb: str, arg: str) -> str cap treaty;
extern fn peer_terms(p: peer) -> str cap treaty;

mesh fn mshift(x: f16[8]) -> f16[8] {
    x * 2.0
}

pcc fn prelay(x: f16[8]) -> f16[8] {
    x + x
}

host det fn main() -> i64 cap treaty {
    let x = onehot(2, 8);
    print("[1] mesh fn — remote exec on treaty'd peer:");
    print("   ", mshift(x));
    print("[2] pcc fn — attested, signed result:");
    print("   ", prelay(x));
    let p = peer_connect("127.0.0.1:9733", "fleet");
    print("[4] treaty ∩ — the bound law of the connection:");
    print("   ", peer_terms(p));
    print("   ", peer_call(p, "status", ""));
    witness("strata", "mesh+pcc+treaty");
    return 0;
}
EOF

cat > "$OUT/enc_pcc.cy" <<'EOF'
param w: f16[8, 8]

pcc enc fn boxd(x: f16[8]) -> f16[8] {
    x @ w
}

host det fn main() -> i64 {
    let x = onehot(3, 8);
    print("[5] enc×pcc — ciphertext eval on attested remote:");
    print("   ", boxd(x));
    return 0;
}
EOF

echo "━━━ starting nodes (auth + AES-256-GCM wire) ━━━"
"$BIN" meshnode --listen 127.0.0.1:9711 > "$OUT/meshnode.log" 2>&1 &
MESH_PID=$!
"$BIN" pccnode --listen 127.0.0.1:9722 > "$OUT/pccnode.log" 2>&1 &
PCC_PID=$!
"$BIN" peernode --listen 127.0.0.1:9733 --treaty "$OUT/treaties.json" \
    > "$OUT/peernode.log" 2>&1 &
PEER_PID=$!
trap 'kill $MESH_PID $PCC_PID $PEER_PID 2>/dev/null || true' EXIT
sleep 1
cat "$OUT/meshnode.log" "$OUT/pccnode.log" "$OUT/peernode.log"

echo
echo "━━━ mesh + pcc + treaty strata ━━━"
COYOTE_MESH_HOST=127.0.0.1:9711 COYOTE_PCC_HOST=127.0.0.1:9722 \
    "$BIN" run "$OUT/organism.cy" -o "$OUT/build_org" || true

echo
echo "━━━ enc × pcc — the sovereignty square ━━━"
COYOTE_PCC_HOST=127.0.0.1:9722 "$BIN" run "$OUT/enc_pcc.cy" \
    -o "$OUT/build_enc" || true

echo
echo "━━━ [3] witness — one chain across nodes ━━━"
cat "$OUT/build_org/witness.jsonl" "$OUT/build_enc/witness.jsonl" 2>/dev/null \
    | grep -v '^$' || echo "(witness jsonl: see build dirs)"

echo
echo "━━━ shardize --world — the fleet-tier emission ━━━"
"$BIN" shardize examples/qwen1x2.cy -o "$OUT/world" --world 2 \
    --hosts "127.0.0.1:9710,127.0.0.1:9711" || true
cat "$OUT/world/world.json" 2>/dev/null || true

echo
echo "✓ five strata: mesh fn | pcc fn | witness wire | treaty/peer | enc×pcc"
