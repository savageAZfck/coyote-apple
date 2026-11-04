#!/bin/bash
# four_substrates.sh — one source, four substrates, same math.
# Every command is real. Drip-fed so the recording scrolls legibly.
cd "$(dirname "$0")/.." || exit 1
CA=./target/release/coyote-apple
AP=./target/release/anepush

H() { printf '\n\033[1;36m── %s ──\033[0m\n' "$1"; sleep 0.7; }
slow() { while IFS= read -r l; do printf '%s\n' "$l"; sleep 0.05; done; }
C='cut -c1-115'

printf '\033[1;36m'
cat <<'EOF'
  coyote-apple — one .cy file, four substrates
  cpu · metal/gpu · neural engine · ciphertext
  the compiler proves they all compute the same math
EOF
printf '\033[0m'
sleep 2.5

H "one source file — examples/qwen28.cy (a 28-layer transformer)"
{ head -8 examples/qwen28.cy; wc -l examples/qwen28.cy; } | slow

H "the compiler sees real weights bound into the language"
$CA spec examples/qwen28.cy 2>/dev/null | head -8 | $C | slow

H "substrate 1+2 — the resident ANE pipeline"
{ ls out/qwen28 | head -8; echo "… 28 layer shards + vocab-split head + embedding"; } | slow
$AP out/qwen28 "791, 6864, 315, 9822, 374" 16 2>&1 | slow

H "and the compiler proves the silicon computes the same math"
$CA oracle examples/qwen28.cy --resident out/qwen28 --steps 4 2>&1 | $C | slow

H "substrate 3 — Metal/GPU, same .cy source"
$CA prof examples/decode_3way.cy --runs 5 2>&1 | slow

H "substrate 4 — ciphertext"
$CA run demo/cipher_layer.cy --cpu 2>&1 | tail -3 | slow
$CA run demo/cipher_layer.cy 2>&1 | tail -4 | slow
$CA run demo/cipher_reject.cy 2>&1 | tail -3 | slow

H "the receipts"
$CA certify --check demo/decode_3way.cert.json 2>&1 | tail -7 | slow
tail -2 coyote_out/witness.jsonl | $C | slow

sleep 1
printf '\n\033[1;36m  one language. four substrates. every run proves the same math.\033[0m\n'
sleep 2.5
