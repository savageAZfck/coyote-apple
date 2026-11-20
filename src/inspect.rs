//! inspect.rs — `coyote-apple inspect` — the org-grade inspection front-end.
//!
//! Wraps `coyote::inspect` (the language's dissection primitives) as a CLI:
//! single artifacts, bounded directory audits, fleet/world audits, and
//! live URL targets. Every report is evidence JSON — sha256 identity,
//! detected format, real-parser findings, declared limitations. Read-only
//! by construction; URL fetches are GET-only with a hard size cap.

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

const MAX_AUDIT_FILES: usize = 5000;
const MAX_AUDIT_FILE: u64 = 256 * 1024 * 1024;
const MAX_URL_BODY: u64 = 32 * 1024 * 1024;
const MAX_URL_TIMEOUT: Duration = Duration::from_secs(20);
const WALK_DEPTH: usize = 6;

fn get(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

// ───────────────────────────── directory audit ─────────────────────────────

fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>, skipped: &mut Vec<Value>) {
    if depth > WALK_DEPTH || out.len() >= MAX_AUDIT_FILES {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut ents: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    ents.sort();
    for p in ents {
        if out.len() >= MAX_AUDIT_FILES {
            break;
        }
        let Ok(meta) = p.metadata() else { continue };
        if meta.is_dir() {
            walk(&p, depth + 1, out, skipped);
        } else if meta.is_file() {
            if meta.len() > MAX_AUDIT_FILE {
                skipped.push(json!({
                    "path": p.display().to_string(),
                    "size": meta.len(),
                    "note": "exceeds audit file cap — skipped",
                }));
            } else {
                out.push(p);
            }
        }
    }
}

/// Per-file evidence summary pulled out of the full evidence payload.
fn summarize(path: &str) -> Value {
    match coyote::inspect::inspect_path(path) {
        Ok(out) => {
            let v: Value = serde_json::from_str(&out.payload).unwrap_or(Value::Null);
            json!({
                "path": path,
                "sha256": v["sha256"].clone(),
                "size": v["size"].clone(),
                "format": v["format"].clone(),
                "confidence": v["confidence"].clone(),
            })
        }
        Err(e) => json!({"path": path, "error": e}),
    }
}

fn audit_dir(dir: &Path) -> Result<Value, String> {
    let mut files = Vec::new();
    let mut skipped = Vec::new();
    walk(dir, 0, &mut files, &mut skipped);
    let mut histogram: BTreeMap<String, u64> = BTreeMap::new();
    let mut attention = Vec::new();
    let mut entries = Vec::new();
    for p in &files {
        let s = p.to_string_lossy().into_owned();
        let sum = summarize(&s);
        if let Some(f) = sum["format"].as_str() {
            *histogram.entry(f.to_string()).or_default() += 1;
        }
        // attention: parse failures, unknown formats, low-confidence hits
        if sum.get("error").is_some()
            || sum["format"].as_str() == Some("unknown")
            || sum["confidence"].as_str() == Some("low")
        {
            attention.push(sum.clone());
        }
        entries.push(sum);
    }
    Ok(json!({
        "tool": "coyote-apple inspect",
        "kind": "dir-audit",
        "dir": dir.display().to_string(),
        "files_scanned": entries.len(),
        "files_skipped": skipped.len(),
        "format_histogram": histogram,
        "attention": attention.iter().take(100).collect::<Vec<_>>(),
        "files": entries.iter().take(2000).collect::<Vec<_>>(),
        "truncated": entries.len() > 2000,
        "skipped": skipped.iter().take(100).collect::<Vec<_>>(),
    }))
}

// ───────────────────────────── fleet / world audit ─────────────────────────────

fn world_path(arg: &str) -> Result<(PathBuf, PathBuf), String> {
    let p = PathBuf::from(arg);
    if p.is_dir() {
        let j = p.join("world.json");
        if j.is_file() {
            return Ok((j, p));
        }
        return Err(format!("{arg}: no world.json inside"));
    }
    let parent = p.parent().unwrap_or(Path::new(".")).to_path_buf();
    Ok((p, parent))
}

fn audit_world(arg: &str) -> Result<Value, String> {
    let (jpath, wdir) = world_path(arg)?;
    let text =
        std::fs::read_to_string(&jpath).map_err(|e| format!("world {}: {e}", jpath.display()))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("world json: {e}"))?;
    let mut ranks = Vec::new();
    for r in v["ranks"].as_array().cloned().unwrap_or_default() {
        let rank = r["rank"].as_u64().unwrap_or(0);
        let host = r["host"].as_str().unwrap_or("").to_string();
        let dir = r["dir"].as_str().unwrap_or("").to_string();
        // rank dirs are relative to the world dir
        let shard = if Path::new(&dir).is_absolute() {
            PathBuf::from(&dir)
        } else {
            wdir.join(&dir)
        };
        let mut entry = json!({"rank": rank, "host": host, "dir": dir});
        if shard.is_dir() {
            match coyote::inspect::inspect_path(shard.to_str().unwrap_or("")) {
                Ok(out) => {
                    let ev: Value = serde_json::from_str(&out.payload).unwrap_or(Value::Null);
                    entry["bundle_sha256"] = ev["sha256"].clone();
                    entry["entries"] = ev["findings"]["entries"].clone();
                    entry["entry_count"] = ev["findings"]["entry_count"].clone();
                }
                Err(e) => entry["error"] = json!(e),
            }
        } else {
            entry["error"] = json!(format!("shard dir missing: {}", shard.display()));
        }
        ranks.push(entry);
    }
    // the world plan itself is an artifact worth fingerprinting
    let plan_sha = coyote::lower::sha256_hex(text.as_bytes());
    Ok(json!({
        "tool": "coyote-apple inspect",
        "kind": "fleet-audit",
        "world": v["world"].clone(),
        "world_json": jpath.display().to_string(),
        "plan_sha256": plan_sha,
        "ranks_expected": v["ranks"].as_array().map(|a| a.len()).unwrap_or(0),
        "ranks": ranks,
    }))
}

// ───────────────────────────── live URL target ─────────────────────────────

fn inspect_url(url: &str) -> Result<Value, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(MAX_URL_TIMEOUT))
        .build()
        .new_agent();
    let mut resp = agent
        .get(url)
        .call()
        .map_err(|e| format!("fetch {url}: {e}"))?;
    let status = resp.status().as_u16();
    let mut headers = serde_json::Map::new();
    for (k, v) in resp.headers() {
        headers.insert(k.to_string(), json!(v.to_str().unwrap_or("")));
    }
    let mut body = Vec::new();
    let read = resp
        .body_mut()
        .as_reader()
        .take(MAX_URL_BODY + 1)
        .read_to_end(&mut body)
        .map_err(|e| format!("read body: {e}"))?;
    let _ = read;
    let truncated = body.len() as u64 > MAX_URL_BODY;
    body.truncate(MAX_URL_BODY as usize);

    // the body is itself an artifact — run the same engine on it
    let body_ev = if body.is_empty() {
        Value::Null
    } else {
        match coyote::inspect::inspect_data(url, &body) {
            Ok(out) => serde_json::from_str(&out.payload).unwrap_or(Value::Null),
            Err(e) => json!({"error": e}),
        }
    };
    Ok(json!({
        "tool": "coyote-apple inspect",
        "kind": "url",
        "url": url,
        "status": status,
        "headers": headers,
        "body_bytes": body.len(),
        "body_truncated": truncated,
        "body_sha256": coyote::lower::sha256_hex(&body),
        "body_evidence": body_ev,
    }))
}

// ───────────────────────────── entry ─────────────────────────────

pub fn run(args: &[String]) -> Result<(), i32> {
    let r = (|| -> Result<Value, String> {
        if let Some(dir) = get(args, "--dir") {
            return audit_dir(Path::new(&dir));
        }
        if let Some(w) = get(args, "--world") {
            return audit_world(&w);
        }
        let target = args.iter().find(|a| !a.starts_with('-')).ok_or(
            "usage: coyote-apple inspect <path|url> [--dir <dir>] [--world <world.json|dir>]",
        )?;
        if target.starts_with("http://") || target.starts_with("https://") {
            return inspect_url(target);
        }
        let p = PathBuf::from(target);
        if !p.exists() {
            return Err(format!("{target}: no such file or directory"));
        }
        if p.is_dir() {
            return audit_dir(&p);
        }
        let out = coyote::inspect::inspect_path(target).map_err(|e| format!("inspect: {e}"))?;
        let v: Value = serde_json::from_str(&out.payload).unwrap_or(Value::Null);
        Ok(json!({"tool": "coyote-apple inspect", "evidence": v}))
    })();
    match r {
        Ok(v) => {
            println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
            Ok(())
        }
        Err(e) => {
            eprintln!("✗ {e}");
            Err(1)
        }
    }
}
