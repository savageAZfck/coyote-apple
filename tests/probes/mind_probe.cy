// mind_probe.cy — live-substrate smoke for the mind extern layer.
// Requires a running Bad Apple daemon; exits 0 only when the substrate
// answers and a real generation comes back.

cap mind

import "mind.cy";

host fn main() -> i64 {
    match mind_status() {
        Json::Obj(o) => {
            if !jbool(o, "ok") {
                print("substrate: down");
                return 2;
            }
            print("substrate: up");
        }
        _ => { return 3; }
    }
    match models() {
        Json::Obj(_) => { print("models: ok"); }
        _ => { print("models: unexpected"); }
    }
    let text = ask("Reply with exactly the word: coyote-mind-ok",
                   "You are a test harness. Reply with exactly the requested word, nothing else.");
    if len(text) == 0 {
        print("infer: empty response");
        return 4;
    }
    print("infer: " + slice_str(text, 0, 80));
    match timed("Say the word telemetry", 32) {
        Json::Obj(t) => {
            if jstr(t, "status") != "ok" {
                print("gen: not ok");
                return 5;
            }
            var line = "gen: ok tps=" + str(jnumf(t, "decode_tps"));
            if has(t, "draft_accept_pct") {
                line = line + " accept=" + str(jnumf(t, "draft_accept_pct")) + "%";
            }
            if has(t, "tier") {
                line = line + " tier=" + jstr(t, "tier");
            }
            print(line);
        }
        _ => { return 6; }
    }
    // cortex — the live-mind bridge over the working socket path
    let creply = cortex("Reply with exactly the word: cortex-ok");
    if len(creply) == 0 {
        print("cortex: empty response");
        return 7;
    }
    print("cortex: " + slice_str(creply, 0, 40));

    // embed — deterministic 64-dim vector, offline, det-safe
    let v1 = embed("coyote mind");
    if len(v1) != 64 {
        print("embed: wrong dims " + str(len(v1)));
        return 8;
    }
    let v2 = embed("coyote mind");
    // determinism is the contract — same text, same vector
    var same = true;
    for i in 0..64 {
        if v1[i] != v2[i] { same = false; }
    }
    if !same {
        print("embed: nondeterministic");
        return 9;
    }
    // unit-norm and self-similarity
    let c = cosine(v1, v2);
    if c < 0.999999 {
        print("embed: self-cosine " + str(c));
        return 10;
    }
    print("embed: ok dims=64 cosine=" + str(c));

    if !remember("mind_probe: infer+gen+cortex+embed verified") {
        print("witness: ledger unreachable");
    }
    return 0;
}

// test-verb surface — exercises the local (offline) externs through
// `coyote-apple test`, no daemon dependency.
host fn test_embed_det() -> i64 {
    let a = embed("probe vector");
    let b = embed("probe vector");
    if len(a) != 64 { return 1; }
    for i in 0..64 {
        if a[i] != b[i] { return 2; }
    }
    if cosine(a, b) < 0.999999 { return 3; }
    let other = embed("something else entirely");
    if cosine(a, other) > 0.9999 { return 4; }
    return 0;
}
