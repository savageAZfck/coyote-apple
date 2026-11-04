//! peer.rs — treaty'd peers as a language primitive.
//!
//! `treaty name { allow …; deny … }` declares what a program will say
//! to another organism and what it refuses. `peer_connect(addr, treaty)`
//! performs the mesh handshake, then the *treaty exchange*: both sides
//! ship their allow/deny sets, the connection is bound to the
//! intersection — a verb neither side permits can never execute. Every
//! `peer_call(p, verb, arg)` re-checks the bound treaty before a byte
//! moves; the externs all carry `cap treaty`, so a program without the
//! capability fails to compile.
//!
//! Wire ops: `treaty` (exchange/intersect), `call` (verb + arg → text
//! result), `ping`, `bye`. Same Frame + auth + cipher as mesh/pcc —
//! a `peer` is just a mesh connection with a law attached.

use crate::wire;
use coyote::vm::{Val, Vm};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};

struct PeerConn {
    stream: TcpStream,
    cipher: Option<wire::SessionCipher>,
    addr: String,
    /// intersected allow set — every call gates on this.
    allow: Vec<String>,
    deny: Vec<String>,
}

type PeerTable = Arc<Mutex<Vec<PeerConn>>>;

fn vkind(v: &Val) -> &'static str {
    match v {
        Val::Unit => "unit",
        Val::I(_) => "i64",
        Val::F(_) => "f64",
        Val::B(_) => "bool",
        Val::S(_) => "str",
        Val::V(_) => "vec",
        Val::M(_) => "map",
        Val::St(..) => "struct",
        Val::En(..) => "enum",
        Val::Dfa(_) => "dfa",
        Val::T(_) => "tensor",
        Val::Ane(_) => "ane",
        Val::Tok { .. } => "token",
        Val::Peer(_) => "peer",
    }
}

/// `treaty A ∩ treaty B`: verbs allowed by both sides, minus anything
/// either side denies. The intersection is the law of the connection.
fn intersect(
    a_allow: &[String],
    a_deny: &[String],
    b_allow: &[String],
    b_deny: &[String],
) -> Vec<String> {
    a_allow
        .iter()
        .filter(|v| {
            b_allow.iter().any(|w| w == *v || w.as_str() == "*")
                && !a_deny.contains(v)
                && !b_deny.contains(v)
        })
        .cloned()
        .collect()
}

/// Register the `peer_*` externs on `vm` — call after `load_vm`.
pub fn bind_all(vm: &mut Vm) {
    let treaties = vm.treaties().clone();
    let table: PeerTable = Arc::new(Mutex::new(Vec::new()));

    // peer_connect(addr: str, treaty: str) -> peer
    {
        let table = table.clone();
        let treaties = treaties.clone();
        vm.bind_extern("peer_connect", move |args| {
            let addr = match &args[0] {
                Val::S(s) => s.to_string(),
                v => return Err(format!("peer_connect addr must be str, found {}", vkind(v))),
            };
            let tname = match &args[1] {
                Val::S(s) => s.to_string(),
                v => {
                    return Err(format!(
                        "peer_connect treaty must be str, found {}",
                        vkind(v)
                    ))
                }
            };
            let (allow, deny) = treaties
                .get(&tname)
                .cloned()
                .ok_or_else(|| format!("no treaty `{tname}` declared in this program"))?;
            let secret = wire::resolve_secret()?;
            let enc = std::env::var("COYOTE_MESH_ENC").ok().as_deref() != Some("0");
            let (mut s, cipher) = wire::connect(&addr, &secret, enc, 60)?;
            // treaty exchange — offer ours, learn theirs, bind the ∩.
            let offer = serde_json::json!({ "allow": allow, "deny": deny });
            let mut f = wire::Frame::new("treaty");
            f.text = Some(offer.to_string());
            wire::write_packet(&mut s, &f, &[], cipher.as_ref())?;
            let (resp, _) = wire::read_packet(&mut s, cipher.as_ref())?;
            match resp.op.as_str() {
                "treaty-ok" => {}
                "error" => {
                    return Err(format!(
                        "peer treaty refused: {}",
                        resp.error.unwrap_or_default()
                    ))
                }
                other => return Err(format!("expected treaty-ok, got {other}")),
            }
            let theirs: serde_json::Value =
                serde_json::from_str(resp.text.as_deref().unwrap_or("{}"))
                    .map_err(|e| format!("treaty reply json: {e}"))?;
            let their_allow: Vec<String> = theirs["allow"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let their_deny: Vec<String> = theirs["deny"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let law = intersect(&allow, &deny, &their_allow, &their_deny);
            let mut t = table.lock().unwrap();
            t.push(PeerConn {
                stream: s,
                cipher,
                addr,
                allow: law,
                deny: deny.into_iter().chain(their_deny).collect(),
            });
            Ok(Val::Peer((t.len() - 1) as u32))
        });
    }

    // peer_call(p: peer, verb: str, arg: str) -> str
    {
        let table = table.clone();
        vm.bind_extern("peer_call", move |args| {
            let h = match &args[0] {
                Val::Peer(h) => *h as usize,
                v => return Err(format!("peer_call wants peer, found {}", vkind(v))),
            };
            let verb = match &args[1] {
                Val::S(s) => s.to_string(),
                v => return Err(format!("peer_call verb must be str, found {}", vkind(v))),
            };
            let arg = match &args[2] {
                Val::S(s) => s.to_string(),
                v => return Err(format!("peer_call arg must be str, found {}", vkind(v))),
            };
            let mut t = table.lock().unwrap();
            let conn = t
                .get_mut(h)
                .ok_or_else(|| format!("peer handle {h} invalid"))?;
            // the treaty is the law — a verb outside the intersection
            // never reaches the wire.
            if conn.deny.iter().any(|v| v == &verb) {
                return Err(format!(
                    "treaty violation: `{verb}` is denied on {}",
                    conn.addr
                ));
            }
            if !conn.allow.iter().any(|v| v == &verb || v == "*") {
                return Err(format!(
                    "treaty violation: `{verb}` not in treaty for {}",
                    conn.addr
                ));
            }
            let mut f = wire::Frame::new("call");
            f.prompt = Some(verb);
            f.text = Some(arg);
            wire::write_packet(&mut conn.stream, &f, &[], conn.cipher.as_ref())?;
            let (resp, _) = wire::read_packet(&mut conn.stream, conn.cipher.as_ref())?;
            match resp.op.as_str() {
                "result" => Ok(Val::S(std::rc::Rc::from(
                    resp.text.unwrap_or_default().as_str(),
                ))),
                "error" => Err(format!(
                    "peer {}: {}",
                    conn.addr,
                    resp.error.unwrap_or_default()
                )),
                other => Err(format!("peer call: expected result, got {other}")),
            }
        });
    }

    // peer_eval(p: peer, src: str, arg: str) -> str — sendable verified
    // computation: ship program source; the peer typechecks it against
    // the treaty before running `main(arg)`. `eval` must be in the law.
    {
        let table = table.clone();
        vm.bind_extern("peer_eval", move |args| {
            let h = match &args[0] {
                Val::Peer(h) => *h as usize,
                v => return Err(format!("peer_eval wants peer, found {}", vkind(v))),
            };
            let src = match &args[1] {
                Val::S(s) => s.to_string(),
                v => return Err(format!("peer_eval src must be str, found {}", vkind(v))),
            };
            let arg = match &args[2] {
                Val::S(s) => s.to_string(),
                v => return Err(format!("peer_eval arg must be str, found {}", vkind(v))),
            };
            let mut t = table.lock().unwrap();
            let conn = t
                .get_mut(h)
                .ok_or_else(|| format!("peer handle {h} invalid"))?;
            if conn.deny.iter().any(|v| v == "eval") {
                return Err(format!(
                    "treaty violation: `eval` is denied on {}",
                    conn.addr
                ));
            }
            if !conn.allow.iter().any(|v| v == "eval" || v == "*") {
                return Err(format!(
                    "treaty violation: `eval` not in treaty for {}",
                    conn.addr
                ));
            }
            let mut f = wire::Frame::new("eval");
            f.prompt = Some(src);
            f.text = Some(arg);
            wire::write_packet(&mut conn.stream, &f, &[], conn.cipher.as_ref())?;
            let (resp, _) = wire::read_packet(&mut conn.stream, conn.cipher.as_ref())?;
            match resp.op.as_str() {
                "result" => Ok(Val::S(std::rc::Rc::from(
                    resp.text.unwrap_or_default().as_str(),
                ))),
                "error" => Err(format!(
                    "peer {}: {}",
                    conn.addr,
                    resp.error.unwrap_or_default()
                )),
                other => Err(format!("peer eval: expected result, got {other}")),
            }
        });
    }

    // peer_terms(p: peer) -> str — the bound law, as json.
    {
        let table = table.clone();
        vm.bind_extern("peer_terms", move |args| {
            let h = match &args[0] {
                Val::Peer(h) => *h as usize,
                v => return Err(format!("peer_terms wants peer, found {}", vkind(v))),
            };
            let t = table.lock().unwrap();
            let conn = t.get(h).ok_or_else(|| format!("peer handle {h} invalid"))?;
            Ok(Val::S(std::rc::Rc::from(
                serde_json::json!({
                    "addr": conn.addr,
                    "allow": conn.allow,
                    "deny": conn.deny,
                })
                .to_string()
                .as_str(),
            )))
        });
    }

    // peer_close(p: peer)
    {
        let table = table.clone();
        vm.bind_extern("peer_close", move |args| {
            let h = match &args[0] {
                Val::Peer(h) => *h as usize,
                v => return Err(format!("peer_close wants peer, found {}", vkind(v))),
            };
            let mut t = table.lock().unwrap();
            if let Some(conn) = t.get_mut(h) {
                let f = wire::Frame::new("bye");
                let _ = wire::write_packet(&mut conn.stream, &f, &[], conn.cipher.as_ref());
            }
            Ok(Val::Unit)
        });
    }
}

// ---------------------------------------------------------------------------
// peernode — the answering end. Loads treaties from a `--treaty` JSON file
/// (name → {allow, deny}), intersects on connect, serves `call` verbs.
/// Verbs are dispatched to bound .cy programs the same way mesh exec
/// works — `call <verb>` runs the bound fn of that name.
// ---------------------------------------------------------------------------

/// `coyote-apple peernode --listen h:p --treaty treaties.json --program dir`
pub fn serve(listen: &str, treaty_file: &str, program_dir: Option<&str>) -> Result<(), String> {
    let secret = wire::resolve_secret()?;
    let enc = std::env::var("COYOTE_MESH_ENC").ok().as_deref() != Some("0");
    let treaties: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(treaty_file)
            .map_err(|e| format!("treaty file {treaty_file}: {e}"))?,
    )
    .map_err(|e| format!("treaty json: {e}"))?;
    let programs: Vec<(String, String)> = match program_dir {
        Some(d) => {
            let mut v = Vec::new();
            for e in std::fs::read_dir(d).map_err(|e| e.to_string())? {
                let e = e.map_err(|e| e.to_string())?;
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) == Some("cy") {
                    let stem = p.file_stem().unwrap().to_string_lossy().to_string();
                    let src = std::fs::read_to_string(&p).map_err(|e| e.to_string())?;
                    v.push((stem, src));
                }
            }
            v
        }
        None => Vec::new(),
    };
    let listener =
        std::net::TcpListener::bind(listen).map_err(|e| format!("bind {listen}: {e}"))?;
    println!(
        "peernode listening on {listen} (enc={enc}, treaties={}, programs={})",
        treaties.as_object().map(|o| o.len()).unwrap_or(0),
        programs.len()
    );
    for conn in listener.incoming() {
        let mut s = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept: {e}");
                continue;
            }
        };
        let (secret, treaties, programs) = (secret.clone(), treaties.clone(), programs.clone());
        std::thread::spawn(move || {
            if let Err(e) = serve_conn(&mut s, &secret, enc, &treaties, &programs) {
                let mut f = wire::Frame::new("error");
                f.error = Some(e);
                let _ = wire::write_frame(&mut s, &f);
            }
        });
    }
    Ok(())
}

fn serve_conn(
    s: &mut TcpStream,
    secret: &[u8],
    enc: bool,
    treaties: &serde_json::Value,
    programs: &[(String, String)],
) -> Result<(), String> {
    wire::server_handshake(s, secret)?;
    let cipher = enc.then(|| wire::SessionCipher::new(secret));
    // the law of THIS connection — set by the treaty exchange.
    let mut law: Vec<String> = Vec::new();
    let mut denied: Vec<String> = Vec::new();
    loop {
        let (frame, _payload) = match wire::read_packet(s, cipher.as_ref()) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        match frame.op.as_str() {
            "treaty" => {
                let theirs: serde_json::Value =
                    serde_json::from_str(frame.text.as_deref().unwrap_or("{}"))
                        .map_err(|e| format!("treaty offer json: {e}"))?;
                let their_allow: Vec<String> = theirs["allow"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let their_deny: Vec<String> = theirs["deny"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                // our side is the union of every declared treaty's terms —
                // the intersection still decides per-verb.
                let our_allow: Vec<String> = treaties
                    .as_object()
                    .map(|o| {
                        o.values()
                            .flat_map(|t| {
                                t["allow"]
                                    .as_array()
                                    .map(|a| {
                                        a.iter()
                                            .filter_map(|v| v.as_str().map(String::from))
                                            .collect::<Vec<String>>()
                                    })
                                    .unwrap_or_default()
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let our_deny: Vec<String> = treaties
                    .as_object()
                    .map(|o| {
                        o.values()
                            .flat_map(|t| {
                                t["deny"]
                                    .as_array()
                                    .map(|a| {
                                        a.iter()
                                            .filter_map(|v| v.as_str().map(String::from))
                                            .collect::<Vec<String>>()
                                    })
                                    .unwrap_or_default()
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                law = intersect(&our_allow, &our_deny, &their_allow, &their_deny);
                denied = our_deny.iter().cloned().chain(their_deny).collect();
                let mut r = wire::Frame::new("treaty-ok");
                r.text =
                    Some(serde_json::json!({ "allow": our_allow, "deny": our_deny }).to_string());
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            "call" => {
                let verb = frame.prompt.clone().unwrap_or_default();
                let arg = frame.text.clone().unwrap_or_default();
                if denied.iter().any(|v| v == &verb) || !law.iter().any(|v| v == &verb || v == "*")
                {
                    let mut r = wire::Frame::new("error");
                    r.error = Some(format!("treaty violation: `{verb}` not permitted"));
                    wire::write_packet(s, &r, &[], cipher.as_ref())?;
                    continue;
                }
                match dispatch(&verb, &arg, programs) {
                    Ok(text) => {
                        let mut r = wire::Frame::new("result");
                        r.text = Some(text);
                        wire::write_packet(s, &r, &[], cipher.as_ref())?;
                    }
                    Err(e) => {
                        let mut r = wire::Frame::new("error");
                        r.error = Some(e);
                        wire::write_packet(s, &r, &[], cipher.as_ref())?;
                    }
                }
            }
            "eval" => {
                // sendable verified computation: the caller ships program
                // SOURCE; it is parsed, typechecked, and its declared caps
                // are gated against this connection's treaty law before a
                // single instruction runs.
                if denied.iter().any(|v| v == "eval")
                    || !law.iter().any(|v| v == "eval" || v == "*")
                {
                    let mut r = wire::Frame::new("error");
                    r.error = Some("treaty violation: `eval` not permitted".to_string());
                    wire::write_packet(s, &r, &[], cipher.as_ref())?;
                    continue;
                }
                let src = frame.prompt.clone().unwrap_or_default();
                let arg = frame.text.clone().unwrap_or_default();
                let res = (|| -> Result<String, String> {
                    let ast = coyote::parser::parse(&src).map_err(|d| d.to_string())?;
                    // declared caps must fit inside the negotiated law —
                    // the treaty is the capability ceiling for mobile code.
                    for d in &ast.decls {
                        if let coyote::ast::Decl::Cap(caps) = d {
                            for c in caps {
                                if !law.iter().any(|v| v == &c.v || v == "*") {
                                    return Err(format!(
                                        "treaty violation: shipped program declares cap `{}` not in law",
                                        c.v
                                    ));
                                }
                            }
                        }
                    }
                    let checked = coyote::typeck::check_all(&ast).map_err(|d| d.to_string())?;
                    drop(checked);
                    exec_source("main", &src, &arg)
                })();
                match res {
                    Ok(text) => {
                        let mut r = wire::Frame::new("result");
                        r.text = Some(text);
                        wire::write_packet(s, &r, &[], cipher.as_ref())?;
                    }
                    Err(e) => {
                        let mut r = wire::Frame::new("error");
                        r.error = Some(e);
                        wire::write_packet(s, &r, &[], cipher.as_ref())?;
                    }
                }
            }
            "ping" => {
                let mut r = wire::Frame::new("result");
                r.token = Some(law.len() as i64);
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            "bye" => return Ok(()),
            other => {
                let mut r = wire::Frame::new("error");
                r.error = Some(format!("unknown op {other}"));
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
        }
    }
}

/// verbs dispatch to bound `.cy` programs — `call ask "…"` runs the fn
/// `ask` if a program of that name is bound, else `call status`-style
/// introspection verbs answer from the node's own state.
fn dispatch(verb: &str, arg: &str, programs: &[(String, String)]) -> Result<String, String> {
    match verb {
        "status" => Ok(serde_json::json!({
            "node": format!("coyote-apple/peernode {}", env!("CARGO_PKG_VERSION")),
            "programs": programs.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        })
        .to_string()),
        _ => {
            // a bound program named <verb> answers the call — it runs
            // here, treaty-gated. `fn` programs eval through the program
            // evaluator (f32 io); `host fn` runs in the VM via call_fn.
            let Some((name, src)) = programs.iter().find(|(n, _)| n == verb) else {
                return Err(format!("no verb `{verb}` bound on this peer"));
            };
            let ast = coyote::parser::parse(src).map_err(|d| d.to_string())?;
            let checked = coyote::typeck::check_all(&ast).map_err(|d| d.to_string())?;
            // arg: comma/space-separated scalars.
            let argvals: Vec<f64> = arg
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter(|s| !s.is_empty())
                .filter_map(|s| s.parse::<f64>().ok())
                .collect();
            if let Some((_, tp)) = checked.ane.iter().find(|(n, _)| n == name) {
                // program fn: f32 io marshalled across declared ins in order.
                let argv: Vec<f32> = argvals.iter().map(|v| *v as f32).collect();
                let mut ins: std::collections::HashMap<String, Vec<f32>> =
                    std::collections::HashMap::new();
                let mut off = 0usize;
                for (pname, dir, ty) in tp.ios.iter() {
                    if !matches!(dir, coyote::ast::IoDir::In) {
                        continue;
                    }
                    let elems: usize = ty.dims.iter().map(|d| (*d).max(0) as usize).product();
                    ins.insert(
                        pname.clone(),
                        argv[off..(off + elems).min(argv.len())].to_vec(),
                    );
                    off = (off + elems).min(argv.len());
                }
                let mut ev = coyote::eval::Evaluator::new(tp);
                let out = ev.run(&ins).map_err(|d| format!("eval `{name}`: {d}"))?;
                return Ok(format!(
                    "{name}({arg}) = {:?}",
                    out.iter().take(8).collect::<Vec<_>>()
                ));
            }
            exec_source(name, src, arg)
        }
    }
}

/// run fn `name` in program `src` with scalar arg `arg` — the shared
/// execution tail behind `call` (bound program) and `eval` (shipped
/// source). Program fns eval through the ANE evaluator; host fns run in
/// a real `Vm` via call_fn.
fn exec_source(name: &str, src: &str, arg: &str) -> Result<String, String> {
    let argvals: Vec<f64> = arg
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<f64>().ok())
        .collect();
    let stage = std::env::temp_dir().join(format!("peernode_{}_{name}.cy", std::process::id()));
    std::fs::write(&stage, src).map_err(|e| e.to_string())?;
    let out_dir = std::env::temp_dir().join(format!("peernode_{}_{name}.out", std::process::id()));
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    let mut vm = coyote::load_vm(&stage, &out_dir, true, false, true).map_err(|d| d.to_string())?;
    bind_all(&mut vm);
    let hargs: Vec<coyote::vm::Val> = if argvals.is_empty() && !arg.trim().is_empty() {
        vec![coyote::vm::Val::S(arg.trim().into())]
    } else {
        argvals.iter().map(|v| coyote::vm::Val::F(*v)).collect()
    };
    let out = vm
        .call_fn(name, hargs)
        .map_err(|d| format!("call `{name}`: {d}"))?;
    let rendered = vm.stringify(&out);
    std::fs::remove_file(&stage).ok();
    Ok(format!("{name}({arg}) = {rendered}"))
}
