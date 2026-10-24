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
    if !remember("mind_probe: infer verified") {
        print("witness: ledger unreachable");
    }
    return 0;
}
