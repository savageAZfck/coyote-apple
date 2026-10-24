// mind.cy — intelligence as a language primitive.
//
// The organism's path to the resident mind. Every extern here is bound
// by coyote-apple's substrate client: prompts travel the daemon's
// SLICKS socket and land on whatever brain is resident — GPU, ANE, or
// the bicameral pair — without this file knowing which.
//
// Convention: everything is `stoch` (model output and socket IO are
// nondeterministic by nature) and everything carries `cap mind`, so a
// program declares `cap mind` in its header and the constitution
// checker enforces which contexts may think at all. Failures come
// back in-band — {status:"unavailable"|"err"} or {ok:false,err} —
// externs never trap the VM.

enum Json { Null, Bool(bool), Num(f64), Str(str), Arr([Json]), Obj({str: Json}), Bytes(str) }

// ── the extern surface ───────────────────────────────────────────────

// Is the substrate reachable? → Json {ok: bool, socket: str}
extern stoch fn mind_status() -> Json cap mind;

// Raw generation on the resident brain. system="" uses the daemon's
// normal path; brain="auto" lets the daemon pick the chamber.
// → Json {status: "ok"|"unavailable"|"err", text: str, err?: str}
extern stoch fn mind_infer(prompt: str, system: str, max_tokens: i64, temperature: f64, brain: str) -> Json cap mind;

// Generation through the plain-prompt path — same route the CLI uses,
// which unlike the agent envelope returns Done-frame telemetry:
// → Json {status, text, tokens?, decode_tps?, total_tps?,
//         draft_accept_pct?, tier?, prefix_cache?, speculative?}
extern stoch fn mind_gen(prompt: str, max_tokens: i64) -> Json cap mind;

// The generic agent surface — any daemon method + Json params:
// "list_models", "model_info", "runtime_status", "audit_tail",
// "switch_main_model", "invoke_tool", … → Json (the daemon's result)
extern stoch fn mind_call(method: str, params: Json) -> Json cap mind;

// Append an event to the organism's hash-chained audit ledger through
// the daemon → bool (false when the substrate is away)
extern stoch fn mind_witness(event: str) -> bool cap mind;

// Submit a bounded agent task → Json {task_id, status, goal} | {error}
extern stoch fn mind_task(goal: str, max_steps: i64) -> Json cap mind;

// → Json {tasks: [...]}
extern stoch fn mind_tasks() -> Json cap mind;

// invoke_tool on the daemon's registry → Json {tool, result}
extern stoch fn mind_tool(name: str, params: Json) -> Json cap mind;

// Deterministic 64-dim embedding — token → FNV-hashed seeded row →
// mean-pool → L2-normalize. Pure + offline, so it carries no `stoch`
// and det contexts can embed too. Same shape as the brain's
// semantic_embedding ({ok, vec}) — different table, same semantics.
extern det fn mind_embed(text: str) -> Json cap mind;

// The live-mind query — same primitive shape as the language's
// builtin `cortex_ask`, but bound to this distribution's working
// SLICKS path. Returns the reply text, "" when the mind is away.
extern stoch fn mind_cortex(prompt: str) -> str cap mind;

// ── reading Json results ─────────────────────────────────────────────

host det fn jstr(o: {str: Json}, k: str) -> str {
    if has(o, k) {
        match o[k] { Json::Str(s) => { return s; } _ => { } }
    }
    return "";
}

host det fn jbool(o: {str: Json}, k: str) -> bool {
    if has(o, k) {
        match o[k] { Json::Bool(b) => { return b; } _ => { } }
    }
    return false;
}

host det fn jnum(o: {str: Json}, k: str) -> i64 {
    if has(o, k) {
        match o[k] { Json::Num(n) => { return i64(n); } _ => { } }
    }
    return 0;
}

host det fn jnumf(o: {str: Json}, k: str) -> f64 {
    if has(o, k) {
        match o[k] { Json::Num(n) => { return n; } _ => { } }
    }
    return 0.0;
}

host det fn jobj(o: {str: Json}, k: str) -> {str: Json} {
    if has(o, k) {
        match o[k] { Json::Obj(v) => { return v; } _ => { } }
    }
    var e: {str: Json} = {};
    return e;
}

// ── the verbs an organism actually speaks ────────────────────────────

// think(prompt) — one shot on the resident brain, text back. "" on
// unavailable/err — the caller decides whether silence is acceptable.
host fn think(prompt: str) -> str cap mind {
    match mind_infer(prompt, "", 512, 0.0, "auto") {
        Json::Obj(r) => {
            if jstr(r, "status") == "ok" { return jstr(r, "text"); }
            return "";
        }
        _ => { return ""; }
    }
}

// ask(prompt, system) — with a system prompt, still one shot.
host fn ask(prompt: str, system: str) -> str cap mind {
    match mind_infer(prompt, system, 512, 0.0, "auto") {
        Json::Obj(r) => {
            if jstr(r, "status") == "ok" { return jstr(r, "text"); }
            return "";
        }
        _ => { return ""; }
    }
}

// gen — the full envelope when the caller needs status + err too.
host fn gen(prompt: str, system: str, max_tokens: i64) -> Json cap mind {
    return mind_infer(prompt, system, max_tokens, 0.0, "auto");
}

// models — which brains are resident/known → Json
host fn models() -> Json cap mind {
    return mind_call("list_models", Json::Null);
}

// witness — the organism speaks onto its own audit chain.
host fn remember(event: str) -> bool cap mind {
    return mind_witness(event);
}

// delegate — hand the daemon a bounded goal → Json {task_id, status}
host fn delegate(goal: str, max_steps: i64) -> Json cap mind {
    return mind_task(goal, max_steps);
}

// timed — generate + telemetry: {status, text, decode_tps, tier,
// draft_accept_pct, speculative, prefix_cache} — the cross-silicon
// performance surface (draft-on-ANE verify-on-GPU shows up here).
host fn timed(prompt: str, max_tokens: i64) -> Json cap mind {
    return mind_gen(prompt, max_tokens);
}

// cortex — the working bridge: prompt → reply text over the
// substrate socket. The builtin `cortex_ask` still attests
// deterministically when nothing is wired; this one thinks.
host fn cortex(prompt: str) -> str cap mind {
    return mind_cortex(prompt);
}

// embed — text → 64-dim unit vector, deterministic and offline.
// Returns [] when the embedding fails (never today — total function).
host det fn embed(text: str) -> [f64] cap mind {
    var empty: [f64] = [];
    match mind_embed(text) {
        Json::Obj(r) => {
            if jbool(r, "ok") {
                match r["vec"] {
                    Json::Arr(v) => {
                        var vec: [f64] = [];
                        for i in 0..len(v) {
                            match v[i] {
                                Json::Num(n) => { push(vec, n); }
                                _ => { }
                            }
                        }
                        return vec;
                    }
                    _ => { return empty; }
                }
            }
            return empty;
        }
        _ => { return empty; }
    }
}

// isqrt — Newton iteration; 32 rounds is past f64 convergence for
// the norm ranges embeddings produce.
host det fn isqrt(x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    var g = x;
    if g < 1.0 { g = 1.0; }
    for _i in 0..32 { g = 0.5 * (g + x / g); }
    return g;
}

// cosine similarity between two same-dim embeddings — pure math, det.
host det fn cosine(a: [f64], b: [f64]) -> f64 {
    var dot = 0.0;
    var na = 0.0;
    var nb = 0.0;
    let n = len(a);
    var i = 0;
    while i < n {
        if i >= len(b) { break; }
        dot = dot + a[i] * b[i];
        na = na + a[i] * a[i];
        nb = nb + b[i] * b[i];
        i = i + 1;
    }
    let d = na * nb;
    if d <= 0.0 { return 0.0; }
    return dot / isqrt(d);
}

// remember_vec — embed then witness a recall key, the organism's
// semantic-memory write path.
host det fn recall_key(text: str) -> str cap mind {
    let v = embed(text);
    // a stable fingerprint of the embedding for keyed recall.
    var h = 0.0;
    for i in 0..len(v) { h = h + v[i] * 7919.0; }
    return "k" + str(i64(h * 1000000.0));
}
