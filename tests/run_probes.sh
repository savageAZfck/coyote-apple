#!/bin/sh
# run_probes.sh — gpu-backend regression suite.
#
# Each probe runs through `coyote-apple prof` and asserts the parity
# line (max |gpu − eval| over outputs) stays under the threshold that
# probe's math actually permits. Thresholds are honest: bit-exact
# where the ops have a single fp ordering, looser where eval itself
# reorders (or nondeterministically orders, e.g. HashMap param scans).
#
#   ./tests/run_probes.sh            # all probes
#   ./tests/run_probes.sh rsoftmax   # name filter

cd "$(dirname "$0")/.." || exit 1
BIN=./target/release/coyote-apple
[ -x "$BIN" ] || { echo "build first: cargo build --release"; exit 2; }

filter="$1"
fail=0

# probe_name:threshold — parity gate per probe.
for entry in \
    "adam:0.0" \
    "let_assign:0.0" \
    "nested_matmul:0.00001" \
    "param_assign:0.0" \
    "q4_param:0.0" \
    "rmsnorm:0.0" \
    "rsoftmax:0.0" \
    "rsoftmax_grad:0.0" \
    "sgd_two_layer:0.0001" \
    "state_scatter:0.0"
do
    name="${entry%%:*}"; thresh="${entry##*:}"
    [ -n "$filter" ] && [ "$name" != "$filter" ] && continue
    f="tests/probes/$name.cy"
    out=$($BIN prof "$f" --runs 10 2>&1)
    parity=$(echo "$out" | awk '/parity/ {print $2; exit}')
    if [ -z "$parity" ]; then
        echo "FAIL $name — prof produced no parity line"
        echo "$out" | tail -3
        fail=1
        continue
    fi
    ok=$(awk -v p="$parity" -v t="$thresh" 'BEGIN{print (p <= t) ? 1 : 0}')
    if [ "$ok" = 1 ]; then
        dev=$(echo "$out" | awk '/device/ {print $2; exit}')
        echo "pass $name   parity $parity   device ${dev}ms"
    else
        echo "FAIL $name — parity $parity > $thresh"
        fail=1
    fi
done

# organism-level: decode cache persistence + host feedback, both
# backends must agree.
if [ -z "$filter" ] || [ "$filter" = decode ]; then
    out=$($BIN oracle examples/decode.cy 2>&1)
    if echo "$out" | grep -q "pass"; then
        echo "pass decode  (organism oracle)"
    else
        echo "FAIL decode — oracle"
        echo "$out" | tail -4
        fail=1
    fi
fi

[ "$fail" = 0 ] && echo "all probes pass" || echo "FAILURES above"
exit $fail
