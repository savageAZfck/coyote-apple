// inspect.cy — the organism's sense of sight, in Coyote.
//
// Thin constitutional wrappers over the inspection natives: every call
// is `det`, `cap dissect`-gated at compile time, and self-witnessed —
// the sha256 of everything the organism looks at lands in the hash
// chain. Evidence comes back as parsed Json; helpers extract the fields
// organisms actually branch on. Flags are observations with evidence,
// never verdicts — same discipline as haruspex and Touchstone.
//
// Read-only by construction: this file holds `dissect` and nothing
// else. `dyld_carve` writes to disk — call the native directly under
// `cap dissect, fs`.

cap dissect

import "sys.cy";
import "strlib.cy";

// ── Json field helpers (same shape as benchmark.cy's) ─────────────────

host det fn jstr_of(j: Json) -> str {
    match j { Json::Str(x) => { return x; } _ => { } }
    return "";
}

host det fn jnum(j: Json) -> f64 {
    match j { Json::Num(n) => { return n; } _ => { } }
    return 0.0;
}

host det fn jbool(j: Json) -> bool {
    match j { Json::Bool(b) => { return b; } _ => { } }
    return false;
}

// evidence object → its "findings" sub-object (empty map when absent)
host det fn findings_of(ev: Json) -> {str: Json} {
    match ev {
        Json::Obj(o) => {
            match o["findings"] {
                Json::Obj(f) => { return f; }
                _ => { }
            }
        }
        _ => { }
    }
    var empty: {str: Json} = {};
    return empty;
}

host det fn finding(ev: Json, key: str) -> Json {
    let f = findings_of(ev);
    if has(f, key) { return f[key]; }
    return Json::Null;
}

// top-level field of a Json object — Null when absent or not an object
host det fn jget(ev: Json, key: str) -> Json {
    match ev {
        Json::Obj(o) => {
            if has(o, key) { return o[key]; }
        }
        _ => { }
    }
    return Json::Null;
}

// ── core evidence ─────────────────────────────────────────────────────

// full evidence document for any supported artifact
host det fn inspect_evidence(path: str) -> Json {
    return json_parse(inspect(path));
}

host det fn inspect_format(path: str) -> str {
    return jstr_of(jget(inspect_evidence(path), "format"));
}

host det fn inspect_sha256(path: str) -> str {
    return jstr_of(jget(inspect_evidence(path), "sha256"));
}

host det fn inspect_confidence(path: str) -> str {
    return jstr_of(jget(inspect_evidence(path), "confidence"));
}

host det fn inspect_size(path: str) -> f64 {
    return jnum(jget(inspect_evidence(path), "size"));
}

// does the engine recognize this artifact at all?
host det fn inspect_known(path: str) -> bool {
    return inspect_format(path) != "unknown";
}

// does it claim to be `fmt`? ("mach-o", "safetensors", "zip", ...)
host det fn inspect_is(path: str, fmt: str) -> bool {
    return inspect_format(path) == fmt;
}

// ── haruspex deep slice (Mach-O / dyld) ────────────────────────────────

host det fn dissect_audit(path: str) -> Json {
    return json_parse(dissect(path));
}

// offset+text strings — "which function emits this" starts here
host det fn hunt_strings(path: str, needle: str) -> [str] {
    return hunt(path, needle);
}

host det fn strings_in(path: str, min: i64) -> [str] {
    return mach_strings(path, min);
}

host det fn dyld_images(grep: str) -> [str] {
    return dyld_list(grep);
}

host det fn disasm_lines(path: str, addr: str) -> [str] {
    return disasm_at(path, addr);
}

// ── comparison ────────────────────────────────────────────────────────

host det fn diff_report(old: str, new: str) -> Json {
    return json_parse(bin_diff(old, new));
}

host det fn binaries_differ(old: str, new: str) -> bool {
    let d = diff_report(old, new);
    return jstr_of(jget(d, "old_sha256")) != jstr_of(jget(d, "new_sha256"));
}

// ── bundle / container summaries ──────────────────────────────────────

// member names of a zip/bundle evidence doc
host det fn member_names(ev: Json) -> [str] {
    var acc: [str] = [];
    match finding(ev, "names") {
        Json::Arr(items) => {
            for i in 0..len(items) {
                match items[i] {
                    Json::Obj(m) => {
                        if has(m, "name") { push(acc, jstr_of(m["name"])); }
                    }
                    _ => { }
                }
            }
        }
        _ => { }
    }
    return acc;
}

// nested member evidence inside a container report
host det fn member_evidence(ev: Json) -> [Json] {
    var acc: [Json] = [];
    match finding(ev, "members") {
        Json::Arr(items) => {
            for i in 0..len(items) {
                match items[i] {
                    Json::Obj(m) => {
                        if has(m, "evidence") { push(acc, m["evidence"]); }
                    }
                    _ => { }
                }
            }
        }
        _ => { }
    }
    return acc;
}
