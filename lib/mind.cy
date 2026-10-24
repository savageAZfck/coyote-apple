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
