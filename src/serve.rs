//! serve.rs — the fleet's serving tier.
//!
//! `coyote-apple serve` is a session-managed inference daemon sitting in
//! front of a rank pipeline. Clients speak the same wire as ranks
//! (wire.rs — HMAC auth, AES-GCM packets); the daemon multiplexes their
//! sessions over the pipeline upstream.
//!
//! Semantics today (v1, honest):
//!   - world plan from `world.json` (`--world`) or `--ranks a,b,...`
//!   - one upstream conn to rank 0, mutex-serialized — the pipeline is
//!     single-stream hardware; sessions queue at the door, each holds
//!     its own history string, and a `reset` between turns keeps KV
//!     from bleeding across sessions
//!   - `session` op binds a session id to history; `generate` appends
//!     and runs; `rewind` to a client-supplied position for prefix reuse
//!   - every request lands on a hash-chained witness jsonl — the same
//!     audit posture the rest of the stack carries
//!
//! The prefix-reuse seam: `rewind target` already exists on the wire;
//! when a rank reports its position the session layer can rewind to a
//! shared prefix instead of resetting — same optimization bad_apple's
//! driver uses for its ~4k system prompt.

use crate::wire;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

/// hash-chained witness journal — seq/tag/val/hash, sha256 over
/// prev_hash‖tag‖val, same construction the VM's chain uses.
struct Journal {
    path: String,
    seq: u64,
    tip: String,
}

impl Journal {
    fn open(path: &str) -> Self {
        let (seq, tip) = std::fs::read_to_string(path)
            .ok()
            .and_then(|t| {
                let last = t.lines().last()?.to_string();
                let v: serde_json::Value = serde_json::from_str(&last).ok()?;
                Some((v["seq"].as_u64()? + 1, v["hash"].as_str()?.to_string()))
            })
            .unwrap_or((0, "genesis".into()));
        Journal {
            path: path.into(),
            seq,
            tip,
        }
    }

    fn wit(&mut self, tag: &str, val: &str) {
        let h = Sha256::digest(format!("{}:{}:{}", self.tip, tag, val).as_bytes());
        let hash: String = h.iter().map(|b| format!("{b:02x}")).collect();
        let line = serde_json::json!({
            "seq": self.seq, "tag": tag, "val": val, "hash": hash,
        })
        .to_string();
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            use std::io::Write;
            let _ = writeln!(f, "{line}");
        }
        self.tip = hash;
        self.seq += 1;
    }
}

struct Session {
    history: String,
    turns: u64,
    lane: usize,
    last: std::time::Instant,
}

/// Upstream = the rank-0 conn plus its failover candidates. On transport
/// failure the next candidate is tried and the conn is replaced — a dead
/// rank costs one retry, not the request.
struct Upstream {
    hosts: Vec<String>,
    cur: usize,
    secret: Vec<u8>,
    enc: bool,
    stream: TcpStream,
    cipher: Option<wire::SessionCipher>,
}

impl Upstream {
    fn connect(hosts: &[String], secret: &[u8], enc: bool) -> Result<Self, String> {
        // ranks come up slow — ANE compile precedes bind. Keep walking
        // candidates until one answers or the patience budget runs dry.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
        loop {
            for (i, h) in hosts.iter().enumerate() {
                if let Ok((stream, cipher)) = wire::connect(h, secret, enc, 600) {
                    return Ok(Upstream {
                        hosts: hosts.to_vec(),
                        cur: i,
                        secret: secret.to_vec(),
                        enc,
                        stream,
                        cipher,
                    });
                }
            }
            if std::time::Instant::now() > deadline {
                return Err(format!("no rank-0 host reachable: {}", hosts.join(", ")));
            }
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    }

    fn reconnect(&mut self) -> Result<(), String> {
        for off in 1..=self.hosts.len() {
            let i = (self.cur + off) % self.hosts.len();
            if let Ok((s, c)) = wire::connect(&self.hosts[i], &self.secret, self.enc, 600) {
                eprintln!(
                    "  upstream failover: {} → {}",
                    self.hosts[self.cur], self.hosts[i]
                );
                self.cur = i;
                self.stream = s;
                self.cipher = c;
                return Ok(());
            }
        }
        Err("all rank-0 candidates unreachable".into())
    }
}

/// Forward a frame+payload to rank 0 — retry once after failover.
fn rank_call(
    up: &Mutex<Upstream>,
    frame: &wire::Frame,
    payload: &[u8],
) -> Result<(wire::Frame, Vec<u8>), String> {
    let mut u = up.lock().map_err(|_| "upstream lock poisoned")?;
    let try_io = |u: &mut Upstream| {
        let Upstream { stream, cipher, .. } = u;
        wire::write_packet(stream, frame, payload, cipher.as_ref())
            .and_then(|_| wire::read_packet(stream, cipher.as_ref()))
    };
    match try_io(&mut u) {
        Ok(r) => Ok(r),
        Err(e) => {
            u.reconnect().map_err(|f| format!("{e}; failover: {f}"))?;
            try_io(&mut u)
        }
    }
}

/// Streaming call: write once, invoke `on` per response frame until
/// `done` — the serve tier forwards each partial to its own client.
fn rank_stream(
    up: &Mutex<Upstream>,
    frame: &wire::Frame,
    payload: &[u8],
    mut on: impl FnMut(&wire::Frame, &[u8]) -> Result<(), String>,
) -> Result<(), String> {
    let mut u = up.lock().map_err(|_| "upstream lock poisoned")?;
    {
        let Upstream { stream, cipher, .. } = &mut *u;
        wire::write_packet(stream, frame, payload, cipher.as_ref())?;
    }
    loop {
        let (r, p) = {
            let Upstream { stream, cipher, .. } = &mut *u;
            wire::read_packet(stream, cipher.as_ref())
        }
        .map_err(|e| e)?;
        let done = r.done.unwrap_or(false) || r.error.is_some();
        on(&r, &p)?;
        if done {
            return Ok(());
        }
    }
}

/// `coyote-apple serve --world world.json [--listen h:p] [--journal path]`
/// or `--ranks h1,h2` for a host list without a plan file.
pub fn serve(args: &[String]) -> Result<(), String> {
    let get = |flag: &str| -> Option<String> {
        args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
    };
    let listen = get("--listen").unwrap_or_else(|| "127.0.0.1:9744".into());
    let journal_path = get("--journal").unwrap_or_else(|| "/tmp/coyote_serve_witness.jsonl".into());
    let ranks: Vec<String> = if let Some(w) = get("--world") {
        let v: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&w).map_err(|e| format!("world {w}: {e}"))?,
        )
        .map_err(|e| format!("world json: {e}"))?;
        v["ranks"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|r| r["host"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    } else {
        get("--ranks")
            .map(|s| s.split(',').map(|h| h.trim().to_string()).collect())
            .unwrap_or_default()
    };
    let rank0 = ranks
        .first()
        .cloned()
        .ok_or_else(|| "serve needs --world <world.json> or --ranks h1,h2".to_string())?;

    let secret = wire::resolve_secret()?;
    let enc = std::env::var("COYOTE_MESH_ENC").ok().as_deref() != Some("0");
    // admission: --lanes N caps concurrent sessions; each session pins a
    // KV lane on the ranks (frame.lane rides the wire).
    let lane_max: usize = get("--lanes")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
        .max(1);

    // upstream — rank-0 candidates, first reachable wins; failover on error.
    let upstream = Arc::new(Mutex::new(Upstream::connect(&ranks, &secret, enc)?));

    // boot health — ping through the upstream conn itself: rank-0 is a
    // single-conn server, so a *second* conn just queues behind ours and
    // times out. The conn that serves requests is the one worth checking.
    match rank_call(&upstream, &wire::Frame::new("ping"), &[]) {
        Ok((r, _)) => println!(
            "  rank {} up — layers {:?}..{:?}",
            r.token.unwrap_or(-1),
            r.layer_start,
            r.layer_end
        ),
        Err(e) => eprintln!("  rank0 health: {e}"),
    }

    let sessions: Arc<Mutex<HashMap<String, Session>>> = Arc::new(Mutex::new(HashMap::new()));
    let free_lanes: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new((0..lane_max).collect()));
    let journal = Arc::new(Mutex::new(Journal::open(&journal_path)));
    let listener = TcpListener::bind(&listen).map_err(|e| format!("bind {listen}: {e}"))?;
    println!(
        "coyote-apple serve — {listen} → rank0 {rank0} ({} rank(s), enc={enc}, journal={journal_path})",
        ranks.len()
    );
    for conn in listener.incoming() {
        let mut s = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept: {e}");
                continue;
            }
        };
        let (secret, up, sess, fl, jrnl) = (
            secret.clone(),
            upstream.clone(),
            sessions.clone(),
            free_lanes.clone(),
            journal.clone(),
        );
        std::thread::spawn(move || {
            if let Err(e) = client_conn(&mut s, &secret, enc, up, sess, fl, jrnl) {
                let mut f = wire::Frame::new("error");
                f.error = Some(e);
                let _ = wire::write_frame(&mut s, &f);
            }
        });
    }
    Ok(())
}

fn client_conn(
    s: &mut TcpStream,
    secret: &[u8],
    enc: bool,
    upstream: Arc<Mutex<Upstream>>,
    sessions: Arc<Mutex<HashMap<String, Session>>>,
    free_lanes: Arc<Mutex<Vec<usize>>>,
    journal: Arc<Mutex<Journal>>,
) -> Result<(), String> {
    wire::server_handshake(s, secret)?;
    let cipher = enc.then(|| wire::SessionCipher::new(secret));
    // the session this connection is bound to ("" = stateless lane 0).
    let mut sid = String::new();
    loop {
        let (frame, payload) = match wire::read_packet(s, cipher.as_ref()) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        match frame.op.as_str() {
            "session" => {
                sid = frame.text.clone().unwrap_or_default();
                let mut r = wire::Frame::new("result");
                let lane = {
                    let mut fl = free_lanes.lock().unwrap();
                    fl.pop().or_else(|| {
                        // pool dry — evict the least-recently-active session,
                        // hand its lane to the newcomer. LRU reaping keeps
                        // admission honest without a fixed ceiling on life.
                        let mut sess = sessions.lock().unwrap();
                        sess.iter()
                            .max_by_key(|(_, se)| se.last.elapsed())
                            .map(|(old_sid, _)| old_sid.clone())
                            .and_then(|old_sid| sess.remove(&old_sid))
                            .map(|evicted| evicted.lane)
                    })
                };
                match lane {
                    Some(lane) => {
                        sessions.lock().unwrap().insert(
                            sid.clone(),
                            Session {
                                history: String::new(),
                                turns: 0,
                                lane,
                                last: std::time::Instant::now(),
                            },
                        );
                        r.text = Some(format!("session `{sid}` lane {lane}"));
                    }
                    None => {
                        r.op = "error".into();
                        r.error = Some("busy: lane pool exhausted".into());
                    }
                }
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            "generate" | "ask" => {
                let prompt = frame.prompt.clone().unwrap_or_default();
                let max = frame.max_tokens.unwrap_or(64);
                let (full_prompt, lane) = {
                    let mut sess = sessions.lock().unwrap();
                    if sid.is_empty() {
                        // sessionless asks honor an explicit frame.lane —
                        // mind verbs and replay address lanes directly.
                        (prompt.clone(), frame.lane.unwrap_or(0))
                    } else {
                        let se = sess.entry(sid.clone()).or_insert_with(|| Session {
                            history: String::new(),
                            turns: 0,
                            lane: 0,
                            last: std::time::Instant::now(),
                        });
                        se.turns += 1;
                        se.last = std::time::Instant::now();
                        if se.history.is_empty() {
                            se.history = prompt.clone();
                        } else {
                            se.history = format!("{}\nuser: {}\nassistant:", se.history, prompt);
                        }
                        (se.history.clone(), se.lane)
                    }
                };
                // replayable receipt: the request as an object — the
                // prompt the model actually saw (session-composed), the
                // raw user turn, params, and any id payload. `replay`
                // re-executes the frame from this entry alone.
                let ids: Vec<u32> = payload
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let req_val = serde_json::json!({
                    "sid": sid, "lane": lane, "max": max,
                    "temp": frame.temp, "top_p": frame.top_p,
                    "prompt": full_prompt, "raw": prompt,
                    "ids": if ids.is_empty() { serde_json::Value::Null } else { ids.clone().into() },
                });
                journal
                    .lock()
                    .unwrap()
                    .wit("serve-request", &req_val.to_string());
                let mut g = wire::Frame::new("generate");
                g.temp = frame.temp;
                g.top_p = frame.top_p;
                g.prompt = Some(full_prompt);
                g.max_tokens = Some(max);
                g.lane = Some(lane);
                let mut full_text = String::new();
                let stream_res = rank_stream(&upstream, &g, &payload, |rf, rp| {
                    // rank-side witness entries land on the same chain.
                    if let Some(entries) = rf.witness.as_ref() {
                        let mut j = journal.lock().unwrap();
                        for (tag, val) in entries {
                            j.wit(tag, val);
                        }
                    }
                    if let Some(e) = rf.error.as_ref() {
                        let mut r = wire::Frame::new("error");
                        r.error = Some(e.clone());
                        return wire::write_packet(s, &r, &[], cipher.as_ref());
                    }
                    if rf.done == Some(true) {
                        // result receipt carries a content hash — replay
                        // diffs recomputed output against this digest.
                        let thash = rf.text.as_deref().map(|t| {
                            let h = Sha256::digest(t.as_bytes());
                            h.iter().map(|b| format!("{b:02x}")).collect::<String>()
                        });
                        let res_val = serde_json::json!({
                            "sid": sid, "lane": lane, "toks": rp.len(), "thash": thash,
                        });
                        journal
                            .lock()
                            .unwrap()
                            .wit("serve-result", &res_val.to_string());
                        // terminal frame's text is the authoritative full
                        // detok — partials were only for streaming.
                        if let Some(t) = rf.text.as_deref() {
                            full_text = t.to_string();
                        }
                    } else if let Some(t) = rf.text.as_deref() {
                        full_text.push_str(t);
                    }
                    wire::write_packet(s, rf, rp, cipher.as_ref())
                });
                if !sid.is_empty() && !full_text.is_empty() {
                    let mut sess = sessions.lock().unwrap();
                    if let Some(se) = sess.get_mut(&sid) {
                        se.history = format!("{} {}", se.history, full_text);
                    }
                }
                if let Err(e) = stream_res {
                    let mut r = wire::Frame::new("error");
                    r.error = Some(e);
                    wire::write_packet(s, &r, &[], cipher.as_ref())?;
                }
            }
            "snapshot" | "resume" => {
                // the live mind through the tier: forward to rank0 on the
                // caller's session lane (stateless callers get lane 0).
                let lane = frame.lane.unwrap_or_else(|| {
                    if sid.is_empty() {
                        0usize
                    } else {
                        sessions
                            .lock()
                            .unwrap()
                            .get(&sid)
                            .map(|se| se.lane)
                            .unwrap_or(0)
                    }
                });
                let mut g = wire::Frame::new(&frame.op);
                g.lane = Some(lane);
                // resume receipts carry the id stream — the checkpoint
                // itself — so replay can rebuild the same mind.
                let ids: Vec<u32> = payload
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let ops_val = serde_json::json!({
                    "sid": sid, "lane": lane,
                    "ids": if frame.op == "resume" && !ids.is_empty() {
                        serde_json::Value::from(ids)
                    } else { serde_json::Value::Null },
                });
                journal
                    .lock()
                    .unwrap()
                    .wit(&format!("serve-{}", frame.op), &ops_val.to_string());
                let mut up = upstream.lock().unwrap();
                let Upstream {
                    stream, cipher: c, ..
                } = &mut *up;
                let res = wire::write_packet(stream, &g, &payload, c.as_ref())
                    .and_then(|_| wire::read_packet(stream, c.as_ref()));
                drop(up);
                match res {
                    Ok((r, rp)) => {
                        if let Some(entries) = r.witness.as_ref() {
                            let mut j = journal.lock().unwrap();
                            for (tag, val) in entries {
                                j.wit(tag, val);
                            }
                        }
                        wire::write_packet(s, &r, &rp, cipher.as_ref())?;
                    }
                    Err(e) => {
                        let _ = upstream.lock().unwrap().reconnect();
                        let mut r = wire::Frame::new("error");
                        r.error = Some(e);
                        wire::write_packet(s, &r, &[], cipher.as_ref())?;
                    }
                }
            }
            "ping" => {
                let mut r = wire::Frame::new("result");
                r.token = Some(sessions.lock().unwrap().len() as i64);
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            "status" => {
                let mut r = wire::Frame::new("result");
                r.text = Some(
                    serde_json::json!({
                        "node": format!("coyote-apple/serve {}", env!("CARGO_PKG_VERSION")),
                        "rank0": upstream.lock().unwrap().hosts[upstream.lock().unwrap().cur],
                        "sessions": sessions.lock().unwrap().len(),
                    })
                    .to_string(),
                );
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

/// `coyote-apple ask "prompt" --to h:p [--session s] [--max-tokens n]`
/// or `ask --ids "1,2,3"` for token-id prompts (real-decode path).
/// — client side of the serving tier.
pub fn ask(args: &[String]) -> Result<(), String> {
    let get = |flag: &str| -> Option<String> {
        args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
    };
    let host = get("--to").unwrap_or_else(|| "127.0.0.1:9744".into());
    let session = get("--session");
    let ids_arg = get("--ids");
    let max_tokens: usize = get("--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let (prompt, id_payload): (Option<String>, Vec<u8>) = if let Some(ids) = ids_arg {
        let mut p = Vec::new();
        for t in ids.split(',') {
            let v: i32 = t.trim().parse().map_err(|_| format!("bad id {t}"))?;
            p.extend_from_slice(&v.to_le_bytes());
        }
        (None, p)
    } else {
        // first non-flag arg that isn't a flag's value.
        let flags: Vec<&str> = vec![
            "--to",
            "--session",
            "--max-tokens",
            "--temp",
            "--top-p",
            "--lane",
            "--serve",
        ];
        let mut prompt = None;
        let mut skip = false;
        for a in args {
            if skip {
                skip = false;
                continue;
            }
            if a.starts_with('-') {
                if flags.contains(&a.as_str()) {
                    skip = true;
                }
                continue;
            }
            prompt = Some(a.clone());
            break;
        }
        let prompt = prompt.ok_or_else(|| {
            "usage: coyote-apple ask \"prompt\" --to h:p [--session s] | --ids \"1,2,3\""
                .to_string()
        })?;
        (Some(prompt), Vec::new())
    };
    let secret = wire::resolve_secret()?;
    let enc = std::env::var("COYOTE_MESH_ENC").ok().as_deref() != Some("0");
    let (mut s, cipher) = wire::connect(&host, &secret, enc, 600)?;
    if let Some(sid) = &session {
        let mut f = wire::Frame::new("session");
        f.text = Some(sid.clone());
        wire::write_packet(&mut s, &f, &[], cipher.as_ref())?;
        let (r, _) = wire::read_packet(&mut s, cipher.as_ref())?;
        if let Some(e) = r.error {
            return Err(format!("session: {e}"));
        }
    }
    let mut f = wire::Frame::new("generate");
    f.prompt = prompt;
    f.max_tokens = Some(max_tokens);
    f.temp = get("--temp").and_then(|s| s.parse().ok());
    f.top_p = get("--top-p").and_then(|s| s.parse().ok());
    // explicit lane addressing — mind forks & replay address lanes
    // directly; serve honors frame.lane when set.
    f.lane = get("--lane")
        .or_else(|| get("--serve").filter(|_| false))
        .and_then(|s| s.parse().ok());
    wire::write_packet(&mut s, &f, &id_payload, cipher.as_ref())?;
    // stream: partial frames until done — print pieces as they land.
    loop {
        let (r, rp) = wire::read_packet(&mut s, cipher.as_ref())?;
        if let Some(e) = r.error {
            return Err(format!("serve: {e}"));
        }
        if r.done == Some(true) {
            if id_payload.is_empty() && r.text.is_none() && !rp.is_empty() {
                let ids: Vec<i32> = rp
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                println!("tokens: {ids:?}");
            } else {
                println!();
            }
            return Ok(());
        }
        if let Some(t) = r.text.as_deref() {
            use std::io::Write;
            print!("{t}");
            let _ = std::io::stdout().flush();
        }
    }
}
