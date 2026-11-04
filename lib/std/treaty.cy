// std/treaty — the federation vocabulary as a library.
// `import "std/treaty"` (CYO_PATH ships coyote-apple's lib/).
//
// A `treaty name { allow …; deny … }` decl lives in the program, not
// the library — it's a constitutional clause. This module ships the
// extern surface every treaty'd peer uses plus two canonical terms.

cap treaty

extern fn peer_connect(addr: str, tname: str) -> peer cap treaty;
extern fn peer_call(p: peer, verb: str, arg: str) -> str cap treaty;
extern fn peer_eval(p: peer, src: str, arg: str) -> str cap treaty;
extern fn peer_terms(p: peer) -> str cap treaty;
extern fn peer_close(p: peer) cap treaty;

// canonical terms — the verbs every organism answers for.
// `status` is the liveness verb; `ask` delegates an inference call.
// Deny is explicit: lifecycle verbs never cross a treaty boundary.
treaty federation {
    allow status, ask
    deny reset, kill, die, migrate
}

// introspection-only terms — the read-only treaty.
treaty observer {
    allow status
    deny reset, kill, die, migrate, ask, exec
}
