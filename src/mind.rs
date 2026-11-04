//! mind.rs — the intelligence extern layer.
//!
//! Binds the `mind_*` externs declared in `lib/mind.cy` to the resident
//! substrate over the daemon socket. Every primitive is `stoch` —
//! model output and socket IO are nondeterministic by nature — and
//! every one carries `cap mind`, so the constitution checker gates
//! which organism contexts may reach the mind at all.
//!
//! Failures come back in-band (`{ok:false, err}` / `status:"err"`)
//! like the reference shell — externs never trap the VM.

use crate::substrate;
use coyote::vm::{Val, Vm};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::rc::Rc;

// ── Json ↔ Val ─────────────────────────────────────────────────────────
// Same shape as the Json enum declared in mind.cy:
//   Null, Bool(bool), Num(f64), Str(str), Arr([Json]), Obj({str:Json}), Bytes(str)

struct JsonIdx {
    en: u32,
    null: u32,
    boolv: u32,
    num: u32,
    strv: u32,
    arr: u32,
    obj: u32,
}

fn json_idx(vm: &Vm) -> Result<JsonIdx, String> {
    let var = |v: &str| {
        vm.enum_variant("Json", v)
            .ok_or_else(|| format!("mind.cy enum Json missing variant `{v}`"))
    };
    let null = var("Null")?;
    Ok(JsonIdx {
        en: null.0,
        null: null.1,
        boolv: var("Bool")?.1,
        num: var("Num")?.1,
        strv: var("Str")?.1,
        arr: var("Arr")?.1,
        obj: var("Obj")?.1,
    })
}

fn jnull(j: &JsonIdx) -> Val {
    Val::En(j.en, j.null, Rc::new(Vec::new()))
}

fn json_to_val(v: &Value, j: &JsonIdx) -> Val {
    let en = |idx: u32, vals: Vec<Val>| Val::En(j.en, idx, Rc::new(vals));
    match v {
        Value::Null => jnull(j),
        Value::Bool(b) => en(j.boolv, vec![Val::B(*b)]),
        Value::Number(n) => en(j.num, vec![Val::F(n.as_f64().unwrap_or(0.0))]),
        Value::String(s) => en(j.strv, vec![Val::S(Rc::from(s.as_str()))]),
        Value::Array(a) => en(
            j.arr,
            vec![Val::V(Rc::new(RefCell::new(
                a.iter().map(|x| json_to_val(x, j)).collect(),
            )))],
        ),
        Value::Object(m) => {
            let mut pairs: Vec<(Val, Val)> = m
                .iter()
                .map(|(k, v)| (Val::S(Rc::from(k.as_str())), json_to_val(v, j)))
                .collect();
            pairs.sort_by(|a, b| {
                if let (Val::S(x), Val::S(y)) = (&a.0, &b.0) {
                    x.cmp(y)
                } else {
                    std::cmp::Ordering::Equal
                }
            });
            en(j.obj, vec![Val::M(Rc::new(RefCell::new(pairs)))])
        }
    }
}

fn json_val(v: &Val) -> Option<Value> {
    match v {
        Val::En(..) => {
            // variant index is p? no — En(en, variant, payload): variant
            // is the second field; use it to decode.
            Some(json_en_val(v))
        }
        Val::S(s) => Some(Value::String(s.to_string())),
        Val::I(i) => Some(json!(*i)),
        Val::F(f) => Some(json!(*f)),
        Val::B(b) => Some(json!(*b)),
        _ => None,
    }
}

fn json_en_val(v: &Val) -> Value {
    let Val::En(_, variant, payload) = v else {
        return Value::Null;
    };
    // Variant order in mind.cy's Json enum:
    // 0 Null, 1 Bool, 2 Num, 3 Str, 4 Arr, 5 Obj, 6 Bytes
    match (*variant, payload.as_slice()) {
        (1, [Val::B(b)]) => json!(b),
        (2, [Val::F(f)]) => json!(f),
        (2, [Val::I(i)]) => json!(i),
        (3, [Val::S(s)]) => Value::String(s.to_string()),
        (4, [Val::V(items)]) => Value::Array(items.borrow().iter().map(json_en_val).collect()),
        (5, [Val::M(pairs)]) => {
            let mut m = serde_json::Map::new();
            for (k, val) in pairs.borrow().iter() {
                if let Val::S(key) = k {
                    m.insert(key.to_string(), json_en_val(val));
                }
            }
            Value::Object(m)
        }
        (6, [Val::S(s)]) => json!({ "__bytes": s.to_string() }),
        _ => Value::Null,
    }
}

// ── arg helpers ─────────────────────────────────────────────────────────

fn arg_str(args: &[Val], i: usize) -> Result<String, String> {
    match args.get(i) {
        Some(Val::S(s)) => Ok(s.to_string()),
        _ => Err(format!("arg {i}: expected str, got wrong type")),
    }
}

fn arg_i64(args: &[Val], i: usize) -> Result<i64, String> {
    match args.get(i) {
        Some(Val::I(v)) => Ok(*v),
        Some(Val::F(v)) => Ok(*v as i64),
        _ => Err(format!("arg {i}: expected i64, got wrong type")),
    }
}

fn arg_f64(args: &[Val], i: usize) -> Result<f64, String> {
    match args.get(i) {
        Some(Val::F(v)) => Ok(*v),
        Some(Val::I(v)) => Ok(*v as f64),
        _ => Err(format!("arg {i}: expected f64, got wrong type")),
    }
}

fn err_json(j: &JsonIdx, e: impl std::fmt::Display) -> Val {
    json_to_val(&json!({"status": "err", "err": e.to_string()}), j)
}

// ── deterministic embedding ─────────────────────────────────────────────
// Same shape as the reference semantic_embedding: token → deterministic
// 64-dim row → mean-pool → L2-normalize. Ours keys rows by FNV-1a hash
// (seeded SplitMix64 stream) instead of a trained BPE vocab, so it runs
// fully offline and — being deterministic — stays `det`-safe: organisms
// can embed inside deterministic contexts.

fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// text → deterministic 64-dim unit vector.
pub fn embed(text: &str) -> Vec<f64> {
    let mut out = [0.0f64; 64];
    let mut n = 0usize;
    for tok in text
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
    {
        let mut seed = fnv1a64(tok);
        for v in out.iter_mut() {
            // [-1, 1) per component — same range as the reference table.
            *v += (splitmix64(&mut seed) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
        }
        n += 1;
    }
    if n > 0 {
        for v in out.iter_mut() {
            *v /= n as f64;
        }
        let mag = out.iter().map(|x| x * x).sum::<f64>().sqrt();
        if mag > 0.0 {
            for v in out.iter_mut() {
                *v /= mag;
            }
        }
    }
    out.to_vec()
}

// ── the bindings ────────────────────────────────────────────────────────

/// Register every `mind_*` extern on `vm`. Call after `load_vm` /
/// `Vm::new`, before `run_main`.
pub fn bind_all(vm: &mut Vm) {
    let Ok(j) = json_idx(vm) else {
        return; // program doesn't import mind.cy — nothing to bind
    };

    // mind_status() -> Json {ok, socket}
    {
        let j = JsonIdx {
            en: j.en,
            null: j.null,
            boolv: j.boolv,
            num: j.num,
            strv: j.strv,
            arr: j.arr,
            obj: j.obj,
        };
        vm.bind_extern("mind_status", move |_| {
            Ok(json_to_val(
                &json!({
                    "ok": substrate::available(),
                    "socket": "/var/run/badapple/substrate.sock",
                }),
                &j,
            ))
        });
    }

    // mind_infer(prompt, system, max_tokens, temperature, brain) -> Json
    //   {status:"ok"|"unavailable"|"err", text, metrics?}
    {
        let j = JsonIdx {
            en: j.en,
            null: j.null,
            boolv: j.boolv,
            num: j.num,
            strv: j.strv,
            arr: j.arr,
            obj: j.obj,
        };
        vm.bind_extern("mind_infer", move |args| {
            let prompt = arg_str(args, 0)?;
            let system = arg_str(args, 1)?;
            let max_tokens = arg_i64(args, 2)?.max(1) as usize;
            let temperature = arg_f64(args, 3).unwrap_or(0.0);
            let brain = arg_str(args, 4).unwrap_or_else(|_| "auto".into());
            if !substrate::available() {
                return Ok(json_to_val(
                    &json!({"status": "unavailable", "text": ""}),
                    &j,
                ));
            }
            match substrate::infer(&prompt, &system, max_tokens, temperature, &brain) {
                Ok(v) => {
                    // daemon answers {text: "..."} or an error envelope
                    let text = v
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    if v.get("error").is_some() {
                        return Ok(json_to_val(
                            &json!({"status": "err", "text": "",
                                    "err": v.get("error").and_then(Value::as_str).unwrap_or("")}),
                            &j,
                        ));
                    }
                    Ok(json_to_val(&json!({"status": "ok", "text": text}), &j))
                }
                Err(e) => Ok(err_json(&j, e)),
            }
        });
    }

    // mind_call(method, params) -> Json — the generic agent surface.
    {
        let j = JsonIdx {
            en: j.en,
            null: j.null,
            boolv: j.boolv,
            num: j.num,
            strv: j.strv,
            arr: j.arr,
            obj: j.obj,
        };
        vm.bind_extern("mind_call", move |args| {
            let method = arg_str(args, 0)?;
            let params = args.get(1).and_then(json_val);
            if !substrate::available() {
                return Ok(json_to_val(
                    &json!({"ok": false, "err": "substrate unavailable"}),
                    &j,
                ));
            }
            match substrate::call_agent(&method, params, 4096) {
                Ok(v) => Ok(json_to_val(&v, &j)),
                Err(e) => Ok(json_to_val(&json!({"ok": false, "err": e}), &j)),
            }
        });
    }

    // mind_gen(prompt, max_tokens) -> Json — the plain-prompt path:
    // same generation as the CLI query route, which unlike the agent
    // envelope returns Done metrics (tier, decode_tps, draft_accept_pct,
    // speculative) — the cross-silicon telemetry surface.
    {
        let j = JsonIdx {
            en: j.en,
            null: j.null,
            boolv: j.boolv,
            num: j.num,
            strv: j.strv,
            arr: j.arr,
            obj: j.obj,
        };
        vm.bind_extern("mind_gen", move |args| {
            let prompt = arg_str(args, 0)?;
            let max_tokens = arg_i64(args, 1)?.max(1) as usize;
            if !substrate::available() {
                return Ok(json_to_val(
                    &json!({"status": "unavailable", "text": ""}),
                    &j,
                ));
            }
            match substrate::query(&prompt, max_tokens) {
                Ok((text, metrics)) => {
                    let mut o = serde_json::Map::new();
                    o.insert("status".into(), json!("ok"));
                    o.insert("text".into(), json!(text));
                    if let Some(m) = metrics {
                        for k in [
                            "tokens",
                            "decode_tps",
                            "total_tps",
                            "draft_accept_pct",
                            "tier",
                            "prefix_cache",
                            "speculative",
                        ] {
                            if let Some(v) = m.get(k) {
                                o.insert(k.into(), v.clone());
                            }
                        }
                    }
                    Ok(json_to_val(&Value::Object(o), &j))
                }
                Err(e) => Ok(err_json(&j, e)),
            }
        });
    }

    // mind_embed(text) -> Json {ok, vec:[f64×64]} — deterministic,
    // offline, `det`-safe (registered stoch-free so det callers can use
    // it; the extern itself is pure).
    {
        let j = JsonIdx {
            en: j.en,
            null: j.null,
            boolv: j.boolv,
            num: j.num,
            strv: j.strv,
            arr: j.arr,
            obj: j.obj,
        };
        vm.bind_extern("mind_embed", move |args| {
            let text = arg_str(args, 0)?;
            let vec = embed(&text);
            Ok(json_to_val(&json!({"ok": !vec.is_empty(), "vec": vec}), &j))
        });
    }

    // mind_cortex(prompt) -> str — the live-mind query. The language's
    // builtin `cortex_ask` attests locally when no mind is wired; this
    // extern is this distribution's working socket path — "" on failure.
    vm.bind_extern("mind_cortex", move |args| {
        let prompt = arg_str(args, 0)?;
        Ok(Val::S(Rc::from(substrate::cortex_ask(&prompt).as_str())))
    });

    // mind_witness(event) -> bool — drop an audit_event onto the
    // organism's hash-chained ledger through the daemon.
    vm.bind_extern("mind_witness", move |args| {
        let event = arg_str(args, 0)?;
        if !substrate::available() {
            return Ok(Val::B(false));
        }
        Ok(Val::B(
            substrate::call_agent(
                "audit_event",
                Some(json!({"type": "coyote_organism", "data": {"event": event}})),
                64,
            )
            .is_ok(),
        ))
    });

    // mind_task(goal, max_steps) -> Json — submit an agent task on the
    // daemon ({task_id, status, goal} or {error}).
    {
        let j = JsonIdx {
            en: j.en,
            null: j.null,
            boolv: j.boolv,
            num: j.num,
            strv: j.strv,
            arr: j.arr,
            obj: j.obj,
        };
        vm.bind_extern("mind_task", move |args| {
            let goal = arg_str(args, 0)?;
            let max_steps = arg_i64(args, 1).unwrap_or(10).max(1);
            if !substrate::available() {
                return Ok(err_json(&j, "substrate unavailable"));
            }
            match substrate::call_agent(
                "run_agent_task",
                Some(json!({"goal": goal, "max_steps": max_steps})),
                4096,
            ) {
                Ok(v) => Ok(json_to_val(&v, &j)),
                Err(e) => Ok(err_json(&j, e)),
            }
        });
    }

    // mind_tasks() -> Json — list agent tasks ({tasks: [...]}).
    {
        let j = JsonIdx {
            en: j.en,
            null: j.null,
            boolv: j.boolv,
            num: j.num,
            strv: j.strv,
            arr: j.arr,
            obj: j.obj,
        };
        vm.bind_extern("mind_tasks", move |_| {
            if !substrate::available() {
                return Ok(err_json(&j, "substrate unavailable"));
            }
            match substrate::call_agent("list_agent_tasks", None, 4096) {
                Ok(v) => Ok(json_to_val(&v, &j)),
                Err(e) => Ok(err_json(&j, e)),
            }
        });
    }

    // mind_tool(name, params) -> Json — invoke_tool on the daemon's
    // tool registry.
    {
        let j = JsonIdx {
            en: j.en,
            null: j.null,
            boolv: j.boolv,
            num: j.num,
            strv: j.strv,
            arr: j.arr,
            obj: j.obj,
        };
        vm.bind_extern("mind_tool", move |args| {
            let name = arg_str(args, 0)?;
            let params = args.get(1).and_then(json_val);
            if !substrate::available() {
                return Ok(err_json(&j, "substrate unavailable"));
            }
            // daemon wants {name, args:{k:v}} — non-object params
            // travel under a single "value" key.
            let args_obj = match params {
                Some(Value::Object(m)) => Value::Object(m),
                Some(other) => json!({"value": other}),
                None => json!({}),
            };
            match substrate::call_agent(
                "invoke_tool",
                Some(json!({"name": name, "args": args_obj})),
                4096,
            ) {
                Ok(v) => Ok(json_to_val(&v, &j)),
                Err(e) => Ok(err_json(&j, e)),
            }
        });
    }
}

// ── live-mind mobility ─────────────────────────────────────────────────
// The OTHER sense of `mind`: a lane's KV state is a pure function of
// its token stream, so the id history IS the checkpoint. `snapshot`
// pulls the stream; `resume` replays it — on the same rank (a fork:
// one thought, two continuations) or another host (a migration: the
// mind physically moves machines). Downstream ranks rebuild KV as the
// replayed frames flow.

use crate::wire::{self, Frame};
use std::path::PathBuf;

fn mget(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn mind_snapshot(host: &str, lane: i64, secret: &[u8], enc: bool) -> Result<Vec<u8>, String> {
    let (mut s, c) = wire::connect(host, secret, enc, 600)?;
    let mut f = Frame::new("snapshot");
    f.lane = Some(lane as usize);
    wire::write_packet(&mut s, &f, &[], c.as_ref())?;
    let (r, payload) = wire::read_packet(&mut s, c.as_ref())?;
    if let Some(e) = r.error {
        return Err(e);
    }
    eprintln!(
        "  snapshot {host} lane {lane}: {} ids (pos {:?})",
        payload.len() / 4,
        r.token
    );
    Ok(payload)
}

fn mind_resume(host: &str, lane: i64, ids: &[u8], secret: &[u8], enc: bool) -> Result<(), String> {
    let (mut s, c) = wire::connect(host, secret, enc, 600)?;
    let mut f = Frame::new("resume");
    f.lane = Some(lane as usize);
    wire::write_packet(&mut s, &f, ids, c.as_ref())?;
    let (r, _) = wire::read_packet(&mut s, c.as_ref())?;
    if let Some(e) = r.error {
        return Err(e);
    }
    eprintln!("  resume {host} lane {lane}: pos {:?}", r.token);
    if let Some(w) = &r.witness {
        for (tag, val) in w {
            eprintln!("    witness {tag}: {val}");
        }
    }
    Ok(())
}

/// `coyote-apple mind snapshot|resume|fork|migrate` — the live mind as
/// a movable value: fork is resume-on-another-lane, migrate is
/// resume-on-another-host. Replay is the checkpoint.
pub fn mind(args: &[String]) -> Result<(), String> {
    let op = args.first().map(String::as_str).unwrap_or("");
    let secret = wire::resolve_secret()?;
    let enc = std::env::var("COYOTE_MESH_ENC").ok().as_deref() != Some("0");
    let lane =
        |name: &str, d: i64| -> i64 { mget(args, name).and_then(|s| s.parse().ok()).unwrap_or(d) };
    match op {
        "snapshot" => {
            let to = mget(args, "--to").ok_or("mind snapshot --to h:p --lane N -o file")?;
            let ids = mind_snapshot(&to, lane("--lane", 0), &secret, enc)?;
            let out = mget(args, "-o")
                .map(PathBuf::from)
                .unwrap_or_else(|| "mind.bin".into());
            std::fs::write(&out, &ids).map_err(|e| e.to_string())?;
            println!("{} ids → {}", ids.len() / 4, out.display());
        }
        "resume" => {
            let to = mget(args, "--to").ok_or("mind resume --to h:p --lane N -i file")?;
            let inp = mget(args, "-i")
                .map(PathBuf::from)
                .unwrap_or_else(|| "mind.bin".into());
            let ids = std::fs::read(&inp).map_err(|e| format!("{e}"))?;
            mind_resume(&to, lane("--lane", 0), &ids, &secret, enc)?;
        }
        "fork" => {
            let to = mget(args, "--to").ok_or("mind fork --to h:p --from-lane A --to-lane B")?;
            let ids = mind_snapshot(&to, lane("--from-lane", 0), &secret, enc)?;
            mind_resume(&to, lane("--to-lane", 1), &ids, &secret, enc)?;
            println!(
                "forked lane {} → lane {}",
                lane("--from-lane", 0),
                lane("--to-lane", 1)
            );
        }
        "migrate" => {
            let from = mget(args, "--from")
                .ok_or("mind migrate --from h:p --lane A --to h:p2 --lane B")?;
            let to =
                mget(args, "--to").ok_or("mind migrate --from h:p --lane A --to h:p2 --lane B")?;
            let ids = mind_snapshot(&from, lane("--lane", 0), &secret, enc)?;
            mind_resume(&to, lane("--to-lane", 0), &ids, &secret, enc)?;
            println!(
                "migrated {from} lane {} → {to} lane {}",
                lane("--lane", 0),
                lane("--to-lane", 0)
            );
        }
        _ => {
            return Err(
                "usage: mind snapshot|resume|fork|migrate — the live mind as a movable value"
                    .into(),
            )
        }
    }
    Ok(())
}
