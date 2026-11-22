#!/bin/bash
# REV_DEMO recorder — runs the demo arc with narration pauses.
# Start screen recording FIRST, then run:  bash REV_DEMO.sh
# Match each pause to the VO lines in REV_DEMO.md.
set -e
cd "$(dirname "$0")"
CA=target/debug/coyote-apple

echo "=== BEAT 1: the source (pause ~20s while scrolling revdemo.cy) ==="
sleep 20

echo "=== BEAT 2: encrypt ==="
$CA run examples/revdemo.cy
sleep 8

echo "=== BEAT 3: uncall — the inverse was never written ==="
sleep 10

echo "=== BEAT 4: revcheck — the execution gate ==="
$CA revcheck examples/revdemo.cy
sleep 8

echo "=== BEAT 5: certify + --check ==="
$CA certify examples/revdemo.cy -o /tmp/revdemo.cert
sleep 4
$CA certify --check /tmp/revdemo.cert
sleep 8

echo "=== BEAT 6-7: done — end card ==="
sleep 10
