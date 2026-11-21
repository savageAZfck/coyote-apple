//! ranknode.rs — a weight-carrying mesh rank, ANE-resident.
//!
//! `coyote-apple ranknode --shard <dir>` loads a `shardize --world` rank
//! directory through `mil_infer::ane_pipeline::Pipeline` — the same
//! model.mil graphs, compiled onto the Neural Engine. KV state lives in
//! the graphs' resident state buffers; `--lanes N` spawns N independent
//! pipelines (own KV, own downstream conn) so concurrent sessions get
//! isolated decode state — the `lane` wire field selects the lane.
//!
//! Ops: ping status reset rewind step vstep generate(ids payload).
//! Single-conn at a time — same discipline as bad_apple's serve_ane_tail;
//! lanes multiplex inside that stream.

use crate::wire::{self, Frame, SessionCipher};
use mil_infer::ane_decode::Decoder;
use mil_infer::ane_pipeline::Pipeline;
use mil_infer::e5rt::DeviceMask;
use serde::Deserialize;
use std::io::{BufRead, BufReader};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct Shard {
    rank: usize,
    world: usize,
    #[serde(default)]
    next_host: String,
    #[serde(default)]
    owns_embed: bool,
    #[serde(default)]
    owns_head: bool,
    #[serde(default)]
    layer_start: usize,
    #[serde(default)]
    layer_end: usize,
}

struct Manifest {
    hidden: usize,
    seq_len: usize,
    rope_dim: usize,
    rope_base: f64,
    head_starts: Vec<usize>,
}

/// One decode lane: a resident pipeline (layers + heads + KV states)
/// and its position. Downstream is one shared conn on the rank —
/// `frame.lane` routes the frame to the right lane on the next rank.
struct Lane {
    pipe: Pipeline,
    n_layers: usize,
    pos: usize,
    /// ids this lane last processed — the prefill prefix cache: a new
    /// request sharing a prefix rewinds instead of replaying.
    last_ids: Vec<i64>,
}

struct Rank {
    shard: Shard,
    man: Manifest,
    embed: Vec<u8>, // raw f16 [vocab × hidden]
    eos: i64,
    secret: Vec<u8>,
    enc: bool,
    tok: Option<Tokenizer>,
    temp: f64,
    top_p: f64,
    /// one conn to next_host shared by all lanes — downstream ranks are
    /// single-connection, and `frame.lane` selects the lane over there.
    down: std::cell::RefCell<Option<(TcpStream, Option<SessionCipher>)>>,
}

fn resolve_tokenizer(dir: &Path, flag: Option<String>) -> Option<Tokenizer> {
    let candidates: Vec<PathBuf> = flag
        .map(PathBuf::from)
        .into_iter()
        .chain([
            dir.join("tokenizer.json"),
            dir.parent().unwrap_or(dir).join("tokenizer.json"),
        ])
        .collect();
    candidates
        .iter()
        .find_map(|p| Tokenizer::from_file(p).ok().map(|t| (p.clone(), t)))
        .map(|(p, t)| {
            eprintln!("  tokenizer: {}", p.display());
            t
        })
}

fn load_manifest(dir: &Path) -> Result<Manifest, String> {
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("conversion_manifest.json"))
            .map_err(|e| format!("manifest: {e}"))?,
    )
    .map_err(|e| format!("manifest json: {e}"))?;
    let m = &v["model"];
    let mut head_starts: Vec<usize> = v["shared"]["lm_head_shards"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|h| h["vocab_start"].as_u64().map(|x| x as usize))
                .collect()
        })
        .unwrap_or_default();
    head_starts.sort_unstable();
    Ok(Manifest {
        hidden: m["hidden_size"].as_u64().unwrap_or(1024) as usize,
        seq_len: m["seq_len"].as_u64().unwrap_or(64) as usize,
        rope_dim: m["rope_dim"].as_u64().unwrap_or(128) as usize,
        rope_base: m["rope_freq_base"].as_f64().unwrap_or(1e6),
        head_starts,
    })
}

fn payload_to_f16(p: &[u8]) -> Vec<u16> {
    p.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn f16_to_payload(v: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn f16v(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

/// Write hidden (f16 bits) into the pipeline's x slot, broadcast the
/// positional inputs, run one step, return the output hidden.
/// Write hidden (f16 bits) into the pipeline's x slot, broadcast the
/// positional inputs, run one step, return the output hidden.
///
/// KV is a **ring over `seq_len` slots**: `pos` is the absolute stream
/// position (unbounded — there is no "context full"), `slot = pos % seq`
/// is where this token lands and the angle it rotates by. While the ring
/// fills the mask is causal; once it wraps, every live slot holds one of
/// the last `seq` positions and the whole window attends. Attention span
/// stays `seq_len`; stream length doesn't.
fn lane_forward(lane: &mut Lane, man: &Manifest, h: &[u16]) -> Result<Vec<u16>, String> {
    let (pos, seq, rd, base) = (lane.pos, man.seq_len, man.rope_dim / 2, man.rope_base);
    let slot_pos = pos % seq;
    let (slot, elems) = lane
        .pipe
        .embedding_slot()
        .map_err(|e| format!("embedding_slot: {e}"))?;
    if h.len() as u64 > elems {
        return Err(format!("hidden {} > slot {}", h.len(), elems));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(h.as_ptr(), slot, h.len());
    }
    let mut kw = vec![0u16; seq];
    kw[slot_pos] = f16v(1.0);
    let attn: Vec<u16> = if pos < seq {
        (0..seq)
            .map(|j| if j <= slot_pos { 0 } else { f16v(-65504.0) })
            .collect()
    } else {
        vec![0u16; seq]
    };
    let angles: Vec<f32> = (0..rd)
        .map(|i| (base.powf(-2.0 * i as f64 / (rd * 2) as f64) * slot_pos as f64) as f32)
        .collect();
    let cos: Vec<u16> = angles.iter().map(|a| f16v(a.cos())).collect();
    let sin: Vec<u16> = angles.iter().map(|a| f16v(a.sin())).collect();
    lane.pipe
        .set_shared("kv_write_mask", &kw)
        .map_err(|e| format!("kv_write_mask: {e}"))?;
    lane.pipe
        .set_shared("attn_mask", &attn)
        .map_err(|e| format!("attn_mask: {e}"))?;
    lane.pipe
        .set_shared("rope_cos", &cos)
        .map_err(|e| format!("rope_cos: {e}"))?;
    lane.pipe
        .set_shared("rope_sin", &sin)
        .map_err(|e| format!("rope_sin: {e}"))?;
    lane.pipe.step().map_err(|e| format!("step: {e}"))?;
    lane.pos += 1;
    lane.pipe
        .last_hidden(lane.n_layers)
        .ok_or_else(|| "no hidden out".to_string())
}

/// logits as (vocab_id, f32) pairs across the vocab-split heads.
fn scored_vocab(lane: &Lane, man: &Manifest) -> Vec<(i64, f32)> {
    let logits = lane.pipe.logits();
    if logits.is_empty() {
        return Vec::new();
    }
    let n_heads = man.head_starts.len().max(1);
    let chunk = logits.len() / n_heads;
    let mut out = Vec::with_capacity(logits.len());
    for (i, seg) in logits.chunks(chunk.max(1)).enumerate() {
        let start = man.head_starts.get(i).copied().unwrap_or(0);
        for (j, b) in seg.iter().enumerate() {
            out.push(((start + j) as i64, half::f16::from_bits(*b).to_f32()));
        }
    }
    out
}

/// argmax over concatenated vocab-split head logits.
fn argmax(lane: &Lane, man: &Manifest) -> Option<i64> {
    scored_vocab(lane, man)
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(t, _)| t)
}

/// xorshift64 — no dep, seeded once per process.
fn rand_f() -> f64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static S: AtomicU64 = AtomicU64::new(0);
    let mut x = S.load(Ordering::Relaxed);
    if x == 0 {
        x = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9e3779b97f4a7c15)
            | 1;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    S.store(x, Ordering::Relaxed);
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// temperature + top-p sampling over the scored vocab.
fn sample(lane: &Lane, man: &Manifest, temp: f64, top_p: f64) -> Option<i64> {
    let mut v = scored_vocab(lane, man);
    if v.is_empty() {
        return None;
    }
    if temp <= 1e-6 {
        return v
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(t, _)| *t);
    }
    // softmax over logit/temp
    let mx = v.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
    let mut ws: Vec<(i64, f64)> = v
        .drain(..)
        .map(|(t, s)| (t, ((s - mx) as f64 / temp).exp()))
        .collect();
    ws.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let total: f64 = ws.iter().map(|(_, w)| *w).sum();
    // nucleus
    let mut acc = 0.0;
    let mut cutoff = ws.len();
    for (i, (_, w)) in ws.iter().enumerate() {
        acc += *w / total;
        if acc >= top_p {
            cutoff = i + 1;
            break;
        }
    }
    let sub: f64 = ws[..cutoff].iter().map(|(_, w)| *w).sum();
    let mut r = rand_f() * sub;
    for (t, w) in &ws[..cutoff] {
        r -= *w;
        if r <= 0.0 {
            return Some(*t);
        }
    }
    ws.first().map(|(t, _)| *t)
}

fn embed_row(rank: &Rank, id: i64) -> Vec<u16> {
    let h = rank.man.hidden;
    let off = id.max(0) as usize * h * 2;
    rank.embed
        .get(off..off + h * 2)
        .map(|b| {
            b.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        })
        .unwrap_or_else(|| vec![0u16; h])
}

/// forward a frame+payload to next_host on the rank's shared conn.
/// The frame's `lane` field tells the downstream rank which of its own
/// pipelines services it — one conn carries every lane's traffic.
fn downstream(rank: &Rank, frame: &Frame, payload: &[u8]) -> Result<(Frame, Vec<u8>), String> {
    let mut down = rank.down.borrow_mut();
    if down.is_none() {
        let (s, c) = wire::connect(&rank.shard.next_host, &rank.secret, rank.enc, 600)?;
        *down = Some((s, c));
    }
    let (s, c) = down.as_mut().unwrap();
    match wire::write_packet(s, frame, payload, c.as_ref())
        .and_then(|_| wire::read_packet(s, c.as_ref()))
    {
        Ok(r) => Ok(r),
        Err(_) => {
            // transport broke — reconnect once and retry; rank failover.
            drop(down);
            let (mut s2, c2) = wire::connect(&rank.shard.next_host, &rank.secret, rank.enc, 600)?;
            wire::write_packet(&mut s2, frame, payload, c2.as_ref())
                .and_then(|_| wire::read_packet(&mut s2, c2.as_ref()))
                .map(|r| {
                    *rank.down.borrow_mut() = Some((s2, c2));
                    r
                })
        }
    }
}

fn lane_at(lanes: &mut [Lane], id: usize) -> Option<&mut Lane> {
    // lanes are addressed by their own id — out of range is an error,
    // never a silent alias onto another lane's KV.
    if id == usize::MAX {
        return None;
    }
    lanes.get_mut(id)
}

pub fn ranknode(args: &[String]) -> Result<(), String> {
    let get = |flag: &str| -> Option<String> {
        args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
    };
    if args.iter().any(|a| a == "--cy4b") {
        return ranknode_cy4b(args);
    }
    if args.iter().any(|a| a == "--cy") {
        return ranknode_cy(args);
    }
    let dir = PathBuf::from(
        get("--shard").ok_or("usage: coyote-apple ranknode --shard <dir> [--listen h:p] [--lanes N] [--eos id] [--device ane|gpu|cpu|auto]")?,
    );
    let listen = get("--listen").unwrap_or_else(|| "127.0.0.1:9710".into());
    let lanes_n: usize = get("--lanes")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    let eos = get("--eos")
        .and_then(|s| s.parse().ok())
        .unwrap_or(151645i64);
    let temp: f64 = get("--temp").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let top_p: f64 = get("--top-p").and_then(|s| s.parse().ok()).unwrap_or(0.9);
    let tok = resolve_tokenizer(&dir, get("--tokenizer"));
    let mask = match get("--device").as_deref().unwrap_or("ane") {
        "ane" => DeviceMask::Ane,
        "gpu" => DeviceMask::Gpu,
        "cpu" => DeviceMask::Cpu,
        _ => DeviceMask::Auto,
    };

    let shard: Shard = serde_json::from_str(
        &std::fs::read_to_string(dir.join("shard.json")).map_err(|e| format!("shard.json: {e}"))?,
    )
    .map_err(|e| format!("shard.json: {e}"))?;
    let man = load_manifest(&dir)?;
    let embed = if shard.owns_embed {
        std::fs::read(dir.join("embedding.f16")).map_err(|e| format!("embedding.f16: {e}"))?
    } else {
        Vec::new()
    };
    let secret = wire::resolve_secret()?;
    let enc = std::env::var("COYOTE_MESH_ENC").ok().as_deref() != Some("0");
    let rank = Rank {
        shard,
        man,
        embed,
        eos,
        secret,
        enc,
        tok,
        temp,
        top_p,
        down: std::cell::RefCell::new(None),
    };

    let mut lanes: Vec<Lane> = Vec::new();
    let n_layers = mil_infer::ane_pipeline::shard_paths(&dir)
        .map(|(l, _)| l.len())
        .unwrap_or(0);
    for li in 0..lanes_n {
        let pipe =
            Pipeline::open_dir(&dir, mask).map_err(|e| format!("lane {li} open_dir: {e}"))?;
        lanes.push(Lane {
            pipe,
            n_layers,
            pos: 0,
            last_ids: Vec::new(),
        });
        eprintln!("  lane {li}: pipeline resident ({n_layers} layer shard(s))");
    }

    let listener = TcpListener::bind(&listen).map_err(|e| format!("bind {listen}: {e}"))?;
    println!(
        "ranknode {}/{} — layers {}..{} ({} lane(s), embed={}, head={}) on {listen}",
        rank.shard.rank,
        rank.shard.world,
        rank.shard.layer_start,
        rank.shard.layer_end,
        lanes_n,
        rank.shard.owns_embed,
        rank.shard.owns_head
    );
    for conn in listener.incoming() {
        let mut s = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept: {e}");
                continue;
            }
        };
        let _ = s.set_nodelay(true);
        if let Err(e) = serve_conn(
            &mut s,
            &rank.secret,
            enc,
            &mut lanes,
            &dir,
            eos,
            &rank.tok,
            temp,
            top_p,
        ) {
            let mut f = Frame::new("error");
            f.error = Some(e);
            let _ = wire::write_frame(&mut s, &f);
        }
    }
    Ok(())
}

fn serve_conn(
    s: &mut TcpStream,
    secret: &[u8],
    enc: bool,
    lanes: &mut Vec<Lane>,
    dir: &Path,
    eos: i64,
    tok: &Option<Tokenizer>,
    temp: f64,
    top_p: f64,
) -> Result<(), String> {
    wire::server_handshake(s, secret)?;
    let cipher = enc.then(|| wire::SessionCipher::new(secret));
    let shard: Shard = serde_json::from_str(
        &std::fs::read_to_string(dir.join("shard.json")).map_err(|e| format!("{e}"))?,
    )
    .map_err(|e| format!("{e}"))?;
    let man = load_manifest(dir)?;
    let rank = Rank {
        shard,
        man,
        embed: Vec::new(),
        eos,
        secret: secret.to_vec(),
        enc,
        tok: tok.clone(),
        temp,
        top_p,
        down: std::cell::RefCell::new(None),
    };
    let embed: Vec<u8> = if rank.shard.owns_embed {
        std::fs::read(dir.join("embedding.f16")).map_err(|e| format!("{e}"))?
    } else {
        Vec::new()
    };
    let rank = Rank { embed, ..rank };

    loop {
        let (frame, payload) = match wire::read_packet(s, cipher.as_ref()) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        let lane_id = frame.lane.unwrap_or(0);
        match frame.op.as_str() {
            "ping" => {
                let mut r = Frame::new("result");
                r.token = Some(rank.shard.rank as i64);
                r.layer_start = Some(rank.shard.layer_start);
                r.layer_end = Some(rank.shard.layer_end);
                r.done = Some(false);
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            "status" => {
                let mut r = Frame::new("result");
                r.text = Some(
                    serde_json::json!({
                        "node": format!("coyote-apple/ranknode {}", env!("CARGO_PKG_VERSION")),
                        "rank": rank.shard.rank, "world": rank.shard.world,
                        "layers": [rank.shard.layer_start, rank.shard.layer_end],
                        "lanes": lanes.len(),
                    })
                    .to_string(),
                );
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            "reset" => {
                let mut r = Frame::new("result");
                if let Some(l) = lane_at(lanes, lane_id) {
                    if let Err(e) = l.pipe.reset_state() {
                        r.error = Some(format!("reset_state lane {lane_id}: {e}"));
                    } else {
                        l.pos = 0;
                        l.last_ids.clear();
                    }
                }
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            "snapshot" => {
                // the mind as a value: KV is a pure function of the token
                // stream, so the lane's id history IS the checkpoint.
                let mut r = Frame::new("result");
                if let Some(l) = lane_at(lanes, lane_id) {
                    let mut p = Vec::with_capacity(l.last_ids.len() * 4);
                    for id in &l.last_ids {
                        p.extend_from_slice(&(*id as i32).to_le_bytes());
                    }
                    r.token = Some(l.pos as i64);
                    wire::write_packet(s, &r, &p, cipher.as_ref())?;
                } else {
                    r.error = Some(format!("lane {lane_id} out of range"));
                    wire::write_packet(s, &r, &[], cipher.as_ref())?;
                }
            }
            "resume" => {
                // replay the checkpoint: reset, forward reset downstream,
                // then re-step the id stream — the ring rebuilds itself
                // here and on every downstream rank.
                let ids: Vec<i64> = payload
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                    .collect();
                let res = (|| -> Result<(), String> {
                    let Some(l) = lane_at(lanes, lane_id) else {
                        return Err(format!("lane {lane_id} out of range"));
                    };
                    if !rank.shard.next_host.is_empty() {
                        let mut sync = Frame::new("reset");
                        sync.lane = frame.lane;
                        downstream(&rank, &sync, &[])?;
                    }
                    l.pipe
                        .reset_state()
                        .map_err(|e| format!("resume reset lane {lane_id}: {e}"))?;
                    l.pos = 0;
                    l.last_ids.clear();
                    for &id in &ids {
                        let out = lane_forward(l, &rank.man, &embed_row(&rank, id))?;
                        if !rank.shard.next_host.is_empty() {
                            let mut fwd = Frame::new("step");
                            fwd.lane = frame.lane;
                            downstream(&rank, &fwd, &f16_to_payload(&out))?;
                        }
                        l.last_ids.push(id);
                    }
                    Ok(())
                })();
                let mut r = Frame::new("result");
                if let Err(e) = res {
                    r.error = Some(e);
                } else {
                    let l = lane_at(lanes, lane_id).unwrap();
                    r.token = Some(l.pos as i64);
                    r.witness = Some(vec![(
                        "rank-resume".into(),
                        format!(
                            "rank{}:lane{}:pos{}:ids{}",
                            rank.shard.rank,
                            lane_id,
                            l.pos,
                            ids.len()
                        ),
                    )]);
                }
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            "rewind" => {
                let mut r = Frame::new("result");
                if let Some(l) = lane_at(lanes, lane_id) {
                    l.pos = frame.token.unwrap_or(0).max(0) as usize;
                    l.last_ids.truncate(l.pos);
                    r.token = Some(l.pos as i64);
                } else {
                    r.error = Some("no lane".into());
                }
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            "step" | "vstep" => {
                let h = payload_to_f16(&payload);
                let is_v = frame.op == "vstep";
                let Some(l) = lane_at(lanes, lane_id) else {
                    let mut r = Frame::new("error");
                    r.error = Some(format!("lane {lane_id} out of range"));
                    wire::write_packet(s, &r, &[], cipher.as_ref())?;
                    continue;
                };
                match lane_forward(l, &rank.man, &h) {
                    Ok(out) => {
                        if !rank.shard.next_host.is_empty() {
                            let mut fwd = Frame::new(&frame.op);
                            fwd.lane = frame.lane;
                            match downstream(&rank, &fwd, &f16_to_payload(&out)) {
                                Ok((dr, dp)) => {
                                    wire::write_packet(s, &dr, &dp, cipher.as_ref())?;
                                }
                                Err(e) => {
                                    let mut r = Frame::new("error");
                                    r.error = Some(e);
                                    wire::write_packet(s, &r, &[], cipher.as_ref())?;
                                }
                            }
                        } else if is_v && rank.shard.owns_head {
                            let mut r = Frame::new("result");
                            match sample(
                                l,
                                &rank.man,
                                frame.temp.unwrap_or(rank.temp),
                                frame.top_p.unwrap_or(rank.top_p),
                            ) {
                                Some(t) => {
                                    r.token = Some(t);
                                    r.done = Some(t == rank.eos);
                                    let i32p = (t as i32).to_le_bytes();
                                    wire::write_packet(s, &r, &i32p, cipher.as_ref())?;
                                }
                                None => {
                                    r.error = Some("no head on this rank".into());
                                    wire::write_packet(s, &r, &[], cipher.as_ref())?;
                                }
                            }
                        } else {
                            let mut r = Frame::new("result");
                            r.token = Some(l.pos as i64);
                            r.witness = Some(vec![(
                                "rank-step".into(),
                                format!("rank{}:lane{}:pos{}", rank.shard.rank, lane_id, l.pos),
                            )]);
                            wire::write_packet(s, &r, &f16_to_payload(&out), cipher.as_ref())?;
                        }
                    }
                    Err(e) => {
                        let mut r = Frame::new("error");
                        r.error = Some(e);
                        wire::write_packet(s, &r, &[], cipher.as_ref())?;
                    }
                }
            }
            "generate" => {
                if !rank.shard.owns_embed {
                    let mut r = Frame::new("error");
                    r.error = Some("generate needs the embedding rank (rank 0)".into());
                    wire::write_packet(s, &r, &[], cipher.as_ref())?;
                    continue;
                }
                let ids: Vec<i64> = if let Some(p) = frame.prompt.as_deref() {
                    // text mode — encode through the model's tokenizer.
                    match rank.tok.as_ref().and_then(|t| t.encode(p, false).ok()) {
                        Some(enc) => enc.get_ids().iter().map(|&i| i as i64).collect(),
                        None => {
                            let mut r = Frame::new("error");
                            r.error = Some(
                                "prompt text needs a tokenizer (shard dir or --tokenizer)".into(),
                            );
                            wire::write_packet(s, &r, &[], cipher.as_ref())?;
                            continue;
                        }
                    }
                } else {
                    payload
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                        .collect()
                };
                let max = frame.max_tokens.unwrap_or(32);
                let mut out_ids = Vec::new();
                let mut final_pos = 0usize;
                let mut pos0_out = 0usize;
                let mut shared_out = 0usize;
                let res = (|| -> Result<(), String> {
                    if ids.is_empty() {
                        return Err("generate: empty id payload".into());
                    }
                    let Some(l) = lane_at(lanes, lane_id) else {
                        return Err(format!("lane {lane_id} out of range"));
                    };
                    // every frame = one position on every rank. Prompt
                    // positions go as `step`; the LAST prompt position and
                    // each new token go as `vstep` (step+argmax in one
                    // frame) so no position is processed twice downstream.
                    let mut step_only = |l: &mut Lane, h: Vec<u16>| -> Result<(), String> {
                        let out = lane_forward(l, &rank.man, &h)?;
                        if !rank.shard.next_host.is_empty() {
                            let mut fwd = Frame::new("step");
                            fwd.lane = frame.lane;
                            downstream(&rank, &fwd, &f16_to_payload(&out))?;
                        }
                        Ok(())
                    };
                    let mut tok_of = |l: &mut Lane, h: Vec<u16>| -> Result<i64, String> {
                        let out = lane_forward(l, &rank.man, &h)?;
                        if !rank.shard.next_host.is_empty() {
                            let mut fwd = Frame::new("vstep");
                            fwd.lane = frame.lane;
                            fwd.temp = frame.temp;
                            fwd.top_p = frame.top_p;
                            let (dr, dp) = downstream(&rank, &fwd, &f16_to_payload(&out))?;
                            dp.chunks_exact(4)
                                .next()
                                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                                .or(dr.token)
                                .ok_or("vstep returned no token".to_string())
                        } else {
                            let (t, p) = (
                                frame.temp.unwrap_or(rank.temp),
                                frame.top_p.unwrap_or(rank.top_p),
                            );
                            sample(l, &rank.man, t, p).ok_or("no head on this rank".to_string())
                        }
                    };
                    // prefix cache: positions this lane already holds don't
                    // re-run — rewind to the shared prefix and only step the
                    // suffix. Ring semantics cap this: while pos < seq_len
                    // the KV is still a linear tape and rewind is exact;
                    // once wrapped, early slots were overwritten — the only
                    // honest move is reset + replay.
                    let pos0 = l.pos;
                    pos0_out = pos0;
                    let mut shared = ids
                        .iter()
                        .zip(l.last_ids.iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    if l.pos >= rank.man.seq_len || shared >= rank.man.seq_len {
                        shared = 0;
                    }
                    shared_out = shared;
                    if shared > 0 && shared < l.last_ids.len() {
                        l.pos = shared;
                        l.last_ids.truncate(shared);
                    } else if shared == 0 && !l.last_ids.is_empty() {
                        l.pipe
                            .reset_state()
                            .map_err(|e| format!("prefix reset lane {lane_id}: {e}"))?;
                        l.pos = 0;
                        l.last_ids.clear();
                    }
                    // downstream ranks hold the same sequence — align their
                    // KV to the shared prefix before the suffix flows.
                    if !rank.shard.next_host.is_empty() {
                        let mut sync = Frame::new(if shared > 0 { "rewind" } else { "reset" });
                        sync.lane = frame.lane;
                        if shared > 0 {
                            sync.token = Some(shared as i64);
                        }
                        downstream(&rank, &sync, &[])?;
                    }
                    let mut new_ids: Vec<i64> = ids[shared..].to_vec();
                    if new_ids.is_empty() {
                        new_ids.push(*ids.last().unwrap());
                    }
                    for &id in &new_ids[..new_ids.len() - 1] {
                        step_only(l, embed_row(&rank, id))?;
                        l.last_ids.push(id);
                    }
                    let mut emit = |tok: i64| -> Result<(), String> {
                        let mut p = Frame::new("result");
                        p.token = Some(tok);
                        p.done = Some(false);
                        p.lane = frame.lane;
                        if let Some(t) = rank.tok.as_ref() {
                            p.text = t.decode(&[tok.max(0) as u32], true).ok();
                        }
                        wire::write_packet(s, &p, &tok.to_le_bytes(), cipher.as_ref())
                    };
                    let mut tok = tok_of(l, embed_row(&rank, *new_ids.last().unwrap()))?;
                    l.last_ids.push(*new_ids.last().unwrap());
                    out_ids.push(tok);
                    emit(tok)?;
                    for _ in 1..max {
                        if tok == rank.eos {
                            break;
                        }
                        tok = tok_of(l, embed_row(&rank, tok))?;
                        l.last_ids.push(tok);
                        out_ids.push(tok);
                        emit(tok)?;
                    }
                    final_pos = l.pos;
                    Ok(())
                })();
                match res {
                    Ok(()) => {
                        // terminal frame: done + full text + all ids.
                        let mut r = Frame::new("result");
                        r.done = Some(true);
                        r.witness = Some(vec![(
                            "rank-generate".into(),
                            format!(
                                "rank{}:lane{}:pos{}->{}:shared{}:tok{}",
                                rank.shard.rank,
                                lane_id,
                                pos0_out,
                                final_pos,
                                shared_out,
                                out_ids.len()
                            ),
                        )]);
                        if let Some(t) = rank.tok.as_ref() {
                            let ids32: Vec<u32> =
                                out_ids.iter().map(|&i| i.max(0) as u32).collect();
                            r.text = t.decode(&ids32, true).ok();
                        }
                        let mut pl = Vec::with_capacity(out_ids.len() * 4);
                        for t in &out_ids {
                            pl.extend_from_slice(&(*t as i32).to_le_bytes());
                        }
                        wire::write_packet(s, &r, &pl, cipher.as_ref())?;
                    }
                    Err(e) => {
                        let mut r = Frame::new("error");
                        r.error = Some(e);
                        wire::write_packet(s, &r, &[], cipher.as_ref())?;
                    }
                }
            }
            "bye" => return Ok(()),
            other => {
                let mut r = Frame::new("error");
                r.error = Some(format!("unknown op {other}"));
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
        }
    }
}

// ------------------------------------------------------------------
// `ranknode --cy4b <st.safetensors>` — the fused int8 CY4B seat.
//
// `cy4b_decode serve` (mil_spec) owns the resident ANE programs and
// speaks a stdin line protocol; this bridge speaks the mesh wire on
// the front so the game's `offline-kernel brain` client connects
// unchanged. Single decode process, single connection at a time —
// the substrate proposes, the character's council disposes.
// ------------------------------------------------------------------

fn cy4b_decode_bin() -> PathBuf {
    std::env::var_os("CY4B_DECODE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
            home.join("mil_spec/target/release/examples/cy4b_decode")
        })
}

/// One resident `cy4b_decode serve` child: stdin commands, stdout
/// line protocol. Jobs run to completion before the next command is
/// read, so a `gen` blocks until exactly `n` `tok` lines (or eos/err).
struct Cy4bDecode {
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    child: std::process::Child,
}

impl Cy4bDecode {
    fn spawn(st: &Path) -> Result<Self, String> {
        let bin = cy4b_decode_bin();
        let mut child = std::process::Command::new(&bin)
            .arg(st)
            .arg("serve")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", bin.display()))?;
        let stdin = child.stdin.take().ok_or("decode stdin")?;
        let stdout_raw = child.stdout.take().ok_or("decode stdout")?;
        let mut stdout = BufReader::new(stdout_raw);
        // The resident load takes real seconds — wait for "ready".
        let mut line = String::new();
        loop {
            line.clear();
            let n = stdout
                .read_line(&mut line)
                .map_err(|e| format!("decode ready: {e}"))?;
            if n == 0 {
                return Err("decode exited before ready".into());
            }
            if line.trim() == "ready" {
                break;
            }
        }
        Ok(Cy4bDecode {
            stdin,
            stdout,
            child,
        })
    }

    fn cmd(&mut self, line: &str) -> Result<(), String> {
        use std::io::Write;
        writeln!(self.stdin, "{line}").map_err(|e| format!("decode cmd: {e}"))?;
        self.stdin.flush().map_err(|e| format!("decode flush: {e}"))
    }

    /// One `gen` job to completion — returns the generated token ids
    /// in emission order. Prompt-position "pre" lines are ignored.
    fn generate(&mut self, max: usize, ids: &[i64]) -> Result<Vec<i64>, String> {
        let mut cmd = format!("gen {max}");
        for id in ids {
            cmd.push_str(&format!(" {id}"));
        }
        self.cmd(&cmd)?;
        let mut out = Vec::new();
        let mut line = String::new();
        loop {
            line.clear();
            let n = self
                .stdout
                .read_line(&mut line)
                .map_err(|e| format!("decode read: {e}"))?;
            if n == 0 {
                return Err("decode exited mid-job".into());
            }
            let t = line.trim();
            if let Some(rest) = t.strip_prefix("tok ") {
                if let Some(id) = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<i64>().ok())
                {
                    out.push(id);
                    if out.len() >= max {
                        return Ok(out);
                    }
                    continue;
                }
            }
            if t == "eos" || t.starts_with("err") || t == "stop" {
                return Ok(out);
            }
        }
    }
}

impl Drop for Cy4bDecode {
    fn drop(&mut self) {
        let _ = self.cmd("quit");
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn ranknode_cy4b(args: &[String]) -> Result<(), String> {
    let get = |flag: &str| -> Option<String> {
        args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
    };
    let st = get("--cy4b").ok_or("ranknode --cy4b <st.safetensors>")?;
    let listen = get("--listen").unwrap_or_else(|| "127.0.0.1:9710".into());
    let tok = get("--tokenizer").and_then(|p| {
        Tokenizer::from_file(&p)
            .map_err(|e| eprintln!("tokenizer {p}: {e}"))
            .ok()
    });
    if tok.is_some() {
        eprintln!("  tokenizer: {}", get("--tokenizer").unwrap_or_default());
    }
    let mut dec = Cy4bDecode::spawn(Path::new(&st))?;

    // The world's action grammar — constrain generated tokens to
    // these verbs (bare + leading-space BPE variants), exactly the
    // trie cy4b_decode's `verbs` command builds.
    if let Some(t) = &tok {
        let seqs = cy4b_verb_seqs(t);
        if !seqs.is_empty() {
            let groups: Vec<String> = seqs
                .iter()
                .map(|s| {
                    s.iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .collect();
            let _ = dec.cmd(&format!("verbs {}", groups.join(";")));
        }
    }

    let secret = wire::resolve_secret()?;
    let enc = std::env::var("COYOTE_MESH_ENC").ok().as_deref() != Some("0");
    let listener = TcpListener::bind(&listen).map_err(|e| format!("bind {listen}: {e}"))?;
    println!("cy4b ranknode — {st} resident on {listen}");
    for conn in listener.incoming() {
        let mut s = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept: {e}");
                continue;
            }
        };
        let _ = s.set_nodelay(true);
        let mut g = |max: usize, ids: &[i64]| dec.generate(max, ids);
        if let Err(e) = serve_decode_conn(&mut s, &mut g, tok.as_ref(), &secret, enc) {
            let mut f = Frame::new("error");
            f.error = Some(e);
            let _ = wire::write_frame(&mut s, &f);
        }
    }
    Ok(())
}

fn serve_decode_conn(
    s: &mut TcpStream,
    gen_fn: &mut dyn FnMut(usize, &[i64]) -> Result<Vec<i64>, String>,
    tok: Option<&Tokenizer>,
    secret: &[u8],
    enc: bool,
) -> Result<(), String> {
    wire::server_handshake(s, secret)?;
    let cipher = enc.then(|| wire::SessionCipher::new(secret));
    loop {
        let (frame, payload) = match wire::read_packet(s, cipher.as_ref()) {
            Ok(x) => x,
            Err(_) => return Ok(()), // client closed
        };
        match frame.op.as_str() {
            "generate" => {
                let ids: Vec<i64> = if let Some(p) = frame.prompt.as_deref() {
                    match tok.and_then(|t| t.encode(p, false).ok()) {
                        Some(e) => e.get_ids().iter().map(|&i| i as i64).collect(),
                        None => {
                            let mut r = Frame::new("error");
                            r.error = Some("prompt text needs a tokenizer (--tokenizer)".into());
                            wire::write_packet(s, &r, &[], cipher.as_ref())?;
                            continue;
                        }
                    }
                } else {
                    payload
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                        .collect()
                };
                if ids.is_empty() {
                    let mut r = Frame::new("error");
                    r.error = Some("generate: empty prompt".into());
                    wire::write_packet(s, &r, &[], cipher.as_ref())?;
                    continue;
                }
                let max = frame.max_tokens.unwrap_or(32);
                let gen = match gen_fn(max, &ids) {
                    Ok(g) => g,
                    Err(e) => {
                        let mut r = Frame::new("error");
                        r.error = Some(e);
                        wire::write_packet(s, &r, &[], cipher.as_ref())?;
                        continue;
                    }
                };
                let text = tok
                    .and_then(|t| {
                        t.decode(&gen.iter().map(|&i| i as u32).collect::<Vec<_>>(), true)
                            .ok()
                    })
                    .unwrap_or_default();
                for &id in &gen {
                    let mut r = Frame::new("result");
                    r.token = Some(id);
                    r.lane = frame.lane;
                    let p = (id as i32).to_le_bytes();
                    wire::write_packet(s, &r, &p, cipher.as_ref())?;
                }
                let mut r = Frame::new("result");
                r.token = gen.last().copied();
                r.done = Some(true);
                r.text = Some(text);
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            "ping" => {
                let mut r = Frame::new("result");
                r.done = Some(true);
                r.text = Some("pong".into());
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            other => {
                let mut r = Frame::new("error");
                r.error = Some(format!("unsupported op {other}"));
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
        }
    }
}

// ------------------------------------------------------------------
// `ranknode --cy <model.cy>` — the sovereign CY4B seat, native coyote.
//
// gen_cy4b's decode organism is compiled by coyote's own frontend into
// resident ANE shards (`shardize`), cached beside the model and rebuilt
// when the .cy or any safetensors it binds changes. Decode runs through
// mil_infer::ane_decode::Decoder — one execute_sync per token, KV state
// on-chip. No cy4b_decode subprocess in the loop: the seat is
// coyote-apple end to end.
// ------------------------------------------------------------------

fn cy_shard_dir(cy: &Path) -> PathBuf {
    let mut p = cy.as_os_str().to_os_string();
    p.push(".shards");
    PathBuf::from(p)
}

/// Every safetensors the organism binds (params + table emb), deduped.
fn cy_bound_models(cy: &Path) -> Vec<PathBuf> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    if let Ok(text) = std::fs::read_to_string(cy) {
        for tok in text.split('"') {
            let path = tok.split("::").next().unwrap_or(tok);
            if path.ends_with(".safetensors") {
                let pb = PathBuf::from(path);
                if seen.insert(pb.clone()) {
                    out.push(pb);
                }
            }
        }
    }
    out
}

fn shards_fresh(cy: &Path, dir: &Path) -> bool {
    let Ok(man) = std::fs::metadata(dir.join("conversion_manifest.json")) else {
        return false;
    };
    let Ok(mtime) = man.modified() else {
        return false;
    };
    let newer = |p: &Path| {
        std::fs::metadata(p)
            .and_then(|m| m.modified())
            .map(|t| t > mtime)
            .unwrap_or(false)
    };
    !newer(cy) && cy_bound_models(cy).iter().all(|m| !newer(m))
}

fn ensure_cy_shards(cy: &Path, force: bool) -> Result<PathBuf, String> {
    let dir = cy_shard_dir(cy);
    if !force && shards_fresh(cy, &dir) {
        eprintln!("  shards: fresh {}", dir.display());
        return Ok(dir);
    }
    eprintln!("  shards: shardize {} -> {}", cy.display(), dir.display());
    crate::shardize::run(&[
        cy.to_string_lossy().into(),
        "-o".into(),
        dir.to_string_lossy().into(),
    ])
    .map_err(|e| format!("shardize: exit {e}"))?;
    Ok(dir)
}

/// CY4B_VERBS env → token-id sequences: each verb encoded bare and
/// with a leading space, exactly the forms cy4b_decode's `verbs`
/// command builds. No env or empty list → unconstrained decode.
fn cy4b_verb_seqs(tok: &Tokenizer) -> Vec<Vec<u32>> {
    let Ok(verbs) = std::env::var("CY4B_VERBS") else {
        return Vec::new();
    };
    let mut seqs = Vec::new();
    for v in verbs.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        for form in [v.to_string(), format!(" {v}")] {
            if let Ok(enc) = tok.encode(form.as_str(), false) {
                let ids: Vec<u32> = enc.get_ids().to_vec();
                if !ids.is_empty() {
                    seqs.push(ids);
                }
            }
        }
    }
    seqs
}

fn logits_argmax(logits: &[u16]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            half::f16::from_bits(**a)
                .to_f32()
                .total_cmp(&half::f16::from_bits(**b).to_f32())
        })
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// `Decoder::generate` with the cy4b_decode `verbs` semantics: while
/// the generated stream sits inside a known action prefix, argmax is
/// masked to trie continuations — the substrate picks WHICH verb by
/// its own probabilities, grammar does the rest. Releases once every
/// live path completes a verb. Empty `vseqs` → plain greedy decode.
fn cy_generate(
    dec: &mut Decoder,
    prompt: &[u32],
    max_new: usize,
    vseqs: &[Vec<u32>],
) -> Result<Vec<u32>, mil_infer::e5rt::E5Error> {
    if prompt.is_empty() {
        return Err(mil_infer::e5rt::E5Error::Call("decoder: empty prompt", -1));
    }
    for &t in &prompt[..prompt.len() - 1] {
        dec.push(t, false)?;
    }
    // The last prompt token steps the pipeline; the head logits then
    // predict the first generated token.
    dec.push(prompt[prompt.len() - 1], false)?;

    let mut vactive: Vec<(usize, usize)> = (0..vseqs.len()).map(|i| (i, 0)).collect();
    let mut vdone = vseqs.is_empty();
    let mut out = Vec::new();
    for _ in 0..max_new {
        let logits = dec.logits();
        let mut next = u32::MAX;
        if !vdone {
            let mut bv = f32::NEG_INFINITY;
            for &(si, p) in &vactive {
                if p < vseqs[si].len() {
                    let id = vseqs[si][p] as usize;
                    if id < logits.len() {
                        let v = half::f16::from_bits(logits[id]).to_f32();
                        if v > bv {
                            bv = v;
                            next = id as u32;
                        }
                    }
                }
            }
            if next != u32::MAX {
                let chosen = next;
                vactive.retain_mut(|(si, p)| {
                    if *p < vseqs[*si].len() && vseqs[*si][*p] == chosen {
                        *p += 1;
                        true
                    } else {
                        false
                    }
                });
                if vactive.iter().all(|&(si, p)| p >= vseqs[si].len()) {
                    vdone = true;
                }
            } else {
                vdone = true;
            }
        }
        if next == u32::MAX {
            next = logits_argmax(&logits);
        }
        out.push(next);
        dec.push(next, false)?;
    }
    Ok(out)
}

fn ranknode_cy(args: &[String]) -> Result<(), String> {
    let get = |flag: &str| -> Option<String> {
        args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
    };
    let cy = get("--cy").ok_or("ranknode --cy <model.cy>")?;
    let listen = get("--listen").unwrap_or_else(|| "127.0.0.1:19710".into());
    let tok = get("--tokenizer").and_then(|p| {
        Tokenizer::from_file(&p)
            .map_err(|e| eprintln!("tokenizer {p}: {e}"))
            .ok()
    });
    if tok.is_some() {
        eprintln!("  tokenizer: {}", get("--tokenizer").unwrap_or_default());
    }
    let vseqs = tok.as_ref().map(cy4b_verb_seqs).unwrap_or_default();
    if !vseqs.is_empty() {
        eprintln!("  verbs: {} sequences constrain decode", vseqs.len());
    }
    let force = args.iter().any(|a| a == "--reshardize");
    let shards = ensure_cy_shards(Path::new(&cy), force)?;
    eprintln!("  decoder: opening {}", shards.display());
    let mut dec =
        Decoder::open_dir(&shards, DeviceMask::Ane).map_err(|e| format!("decoder open: {e}"))?;

    let secret = wire::resolve_secret()?;
    let enc = std::env::var("COYOTE_MESH_ENC").ok().as_deref() != Some("0");
    let listener = TcpListener::bind(&listen).map_err(|e| format!("bind {listen}: {e}"))?;
    println!("cy ranknode — {cy} resident on {listen}");
    for conn in listener.incoming() {
        let mut s = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept: {e}");
                continue;
            }
        };
        let _ = s.set_nodelay(true);
        let mut g = |max: usize, ids: &[i64]| -> Result<Vec<i64>, String> {
            dec.reset().map_err(|e| format!("decoder reset: {e}"))?;
            let prompt: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
            let out = cy_generate(&mut dec, &prompt, max, &vseqs)
                .map_err(|e| format!("decoder generate: {e}"))?;
            Ok(out.into_iter().map(|t| t as i64).collect())
        };
        if let Err(e) = serve_decode_conn(&mut s, &mut g, tok.as_ref(), &secret, enc) {
            let mut f = Frame::new("error");
            f.error = Some(e);
            let _ = wire::write_frame(&mut s, &f);
        }
    }
    Ok(())
}
