//! replay.rs — witness entries as re-executable objects.
//!
//! Every `serve-request`/`serve-resume` receipt on the journal is a
//! self-contained execution object: it carries the exact prompt (or id
//! stream) and params the fleet consumed. `replay` re-verifies the
//! journal's hash chain, rebuilds the frame, re-executes it against a
//! live tier, and diffs the recomputed output against the recorded
//! `serve-result` content hash — deterministic reproduction, or an
//! explicit divergence report. Sampled requests (temp > 0) are flagged:
//! byte-equality is a det property, stoch replays compare shape.

use crate::wire::{self, Frame};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn sha_hex(s: &str) -> String {
    let h = Sha256::digest(s.as_bytes());
    h.iter().map(|b| format!("{b:02x}")).collect()
}

struct Entry {
    seq: u64,
    tag: String,
    val: String,
}

/// Read + chain-verify a journal. Any break is a hard error — a replayed
/// receipt is only as trustworthy as the chain it lives on.
fn load_journal(path: &str) -> Result<Vec<Entry>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("journal {path}: {e}"))?;
    let mut entries = Vec::new();
    let mut tip = "genesis".to_string();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value =
            serde_json::from_str(line).map_err(|e| format!("journal line {}: {e}", i + 1))?;
        let tag = v["tag"].as_str().unwrap_or("").to_string();
        let val = v["val"].as_str().unwrap_or("").to_string();
        let hash = v["hash"].as_str().unwrap_or("");
        let want = sha_hex(&format!("{tip}:{tag}:{val}"));
        if hash != want {
            return Err(format!(
                "journal chain broken at seq {} (line {}): hash {} != recomputed {} — refusing to replay from a tampered journal",
                v["seq"], i + 1, hash, want
            ));
        }
        tip = hash.to_string();
        entries.push(Entry {
            seq: v["seq"].as_u64().unwrap_or(0),
            tag,
            val,
        });
    }
    Ok(entries)
}

fn get(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

pub fn run(args: &[String]) -> Result<(), String> {
    let journal = get(args, "--journal")
        .or_else(|| get(args, "-j"))
        .unwrap_or_else(|| "/tmp/realworld/serve_witness.jsonl".into());
    let serve = get(args, "--serve").unwrap_or_else(|| "127.0.0.1:9744".into());
    let entries = load_journal(&journal)?;
    eprintln!("  chain verified — {} entries, tip sound", entries.len());

    // select: --seq N, else last entry matching --tag, else last serve-request
    let pick: Option<&Entry> = if let Some(s) = get(args, "--seq") {
        let n: u64 = s.parse().map_err(|_| "--seq needs a number")?;
        entries.iter().find(|e| e.seq == n)
    } else if let Some(tag) = get(args, "--tag") {
        entries.iter().rev().find(|e| e.tag == tag)
    } else {
        entries.iter().rev().find(|e| e.tag == "serve-request")
    };
    let e = pick.ok_or("no replayable entry found (try --seq N or --tag serve-request)")?;
    let val: Value = serde_json::from_str(&e.val)
        .map_err(|x| format!("seq {}: entry payload isn't replayable JSON ({x})", e.seq))?;

    let secret = wire::resolve_secret()?;
    let enc = std::env::var("COYOTE_MESH_ENC").ok().as_deref() != Some("0");

    match e.tag.as_str() {
        "serve-request" => {
            let lane = val["lane"].as_u64().unwrap_or(0);
            let max = val["max"].as_u64().unwrap_or(8);
            let temp = val["temp"].as_f64().unwrap_or(0.0);
            let prompt = val["prompt"].as_str().unwrap_or("").to_string();
            let ids: Option<Vec<u8>> = val["ids"].as_array().map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64())
                    .flat_map(|i| (i as u32).to_le_bytes())
                    .collect()
            });
            let mut f = Frame::new("generate");
            f.prompt = if prompt.is_empty() {
                None
            } else {
                Some(prompt)
            };
            f.max_tokens = Some(max as usize);
            f.temp = Some(temp);
            f.top_p = val["top_p"].as_f64();
            f.lane = Some(lane as usize);
            let sampled = temp > 1e-6;
            // the receipt records what the lane held: `shared` ids were
            // reused (rewind) or 0 (the request reset). Rebuild that
            // pre-state before re-executing — replay is state-faithful.
            let rank_ev = entries
                .iter()
                .find(|x| x.seq > e.seq && x.tag == "rank-generate")
                .map(|x| x.val.clone())
                .unwrap_or_default();
            let shared: usize = rank_ev
                .split("shared")
                .nth(1)
                .and_then(|s| s.split(':').next())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            eprintln!(
                "  replay seq {}: lane{} max{} temp{} shared{}{}",
                e.seq,
                lane,
                max,
                temp,
                shared,
                if sampled {
                    " — SAMPLED: byte-equality is not the contract"
                } else {
                    " — det replay (argmax)"
                }
            );
            let (mut s, c) = wire::connect(&serve, &secret, enc, 600)?;
            {
                // pre-state: reset (shared==0) or rebuild the prefix the
                // original rewound to (snapshot the lane's first `shared`
                // ids — exact while the lane is undisturbed; the content
                // hash diff will show it if it drifted).
                let pre_ids: Vec<u8> = if shared > 0 {
                    let mut snap = Frame::new("snapshot");
                    snap.lane = Some(lane as usize);
                    wire::write_packet(&mut s, &snap, &[], c.as_ref())?;
                    let (_, cur) = wire::read_packet(&mut s, c.as_ref())?;
                    cur.chunks_exact(4)
                        .take(shared)
                        .flatten()
                        .copied()
                        .collect()
                } else {
                    Vec::new()
                };
                let mut rs = Frame::new("resume");
                rs.lane = Some(lane as usize);
                wire::write_packet(&mut s, &rs, &pre_ids, c.as_ref())?;
                let (rr, _) = wire::read_packet(&mut s, c.as_ref())?;
                if let Some(err) = rr.error {
                    return Err(format!("replay pre-state: {err}"));
                }
            }
            wire::write_packet(&mut s, &f, ids.as_deref().unwrap_or(&[]), c.as_ref())?;
            let mut text = String::new();
            let mut toks = 0usize;
            loop {
                let (rf, rp) = wire::read_packet(&mut s, c.as_ref())?;
                if let Some(err) = rf.error {
                    return Err(format!("replay: fleet error: {err}"));
                }
                toks = rp.len();
                if rf.done == Some(true) {
                    text = rf.text.unwrap_or_default();
                    break;
                }
            }
            // find the serve-result recorded just after this request
            let recorded = entries
                .iter()
                .find(|x| x.seq > e.seq && x.tag == "serve-result")
                .and_then(|x| serde_json::from_str::<Value>(&x.val).ok());
            let live_hash = sha_hex(&text);
            match recorded.and_then(|r| r["thash"].as_str().map(String::from)) {
                Some(th) if th == live_hash => println!(
                    "✓ replay seq {} identical — {} toks, thash {}… matches receipt",
                    e.seq,
                    toks,
                    &live_hash[..12]
                ),
                Some(th) => println!(
                    "✗ replay seq {} divergent — recorded {}… vs live {}…{}",
                    e.seq,
                    &th[..12.min(th.len())],
                    &live_hash[..12],
                    if sampled {
                        " (sampled — divergence may be legitimate)"
                    } else {
                        " — det violation"
                    }
                ),
                None => println!(
                    "· replay seq {} executed — {} toks, thash {}… (no recorded thash to diff)",
                    e.seq,
                    toks,
                    &live_hash[..12]
                ),
            }
            Ok(())
        }
        "serve-resume" => {
            let lane = val["lane"].as_u64().unwrap_or(0);
            let ids: Vec<u8> = val["ids"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64())
                        .flat_map(|i| (i as u32).to_le_bytes())
                        .collect()
                })
                .ok_or("seq resume entry carries no id checkpoint")?;
            let mut f = Frame::new("resume");
            f.lane = Some(lane as usize);
            let (mut s, c) = wire::connect(&serve, &secret, enc, 600)?;
            wire::write_packet(&mut s, &f, &ids, c.as_ref())?;
            let (r, _) = wire::read_packet(&mut s, c.as_ref())?;
            if let Some(err) = r.error {
                return Err(format!("replay resume: {err}"));
            }
            println!(
                "✓ replay seq {} resume — {} ids rebuilt on lane{} (pos {:?})",
                e.seq,
                ids.len() / 4,
                lane,
                r.token
            );
            Ok(())
        }
        other => Err(format!(
            "seq {} is `{other}` — an audit fact, not a re-executable request",
            e.seq
        )),
    }
}
