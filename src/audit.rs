//! `coyote-apple audit <dir> [--verify]` — provenance for a build dir.
//!
//! Every artifact in the directory (net.mil, weight blobs, the plan,
//! the witness log) is hashed; entries are chained
//! `H = sha256(H_prev ‖ name ‖ sha256(file))` and written to
//! `audit.json`. `--verify` recomputes the chain and diffs it — a
//! tampered blob or a reordered artifact breaks the head hash.

use coyote::lower::sha256_hex;
use std::path::{Path, PathBuf};

const AUDIT_FILE: &str = "audit.json";

fn artifacts(dir: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let mut files: Vec<(String, PathBuf)> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| {
            let p = e.ok()?.path();
            if !p.is_file() {
                return None;
            }
            let n = p.file_name()?.to_str()?.to_string();
            (n != AUDIT_FILE && !n.ends_with(".cypack")).then_some((n, p))
        })
        .collect();
    files.sort();
    Ok(files)
}

fn hash_set(dir: &Path) -> Result<(Vec<(String, usize, String)>, String), String> {
    let mut chain = "0".repeat(64);
    let mut entries = Vec::new();
    for (name, path) in artifacts(dir)? {
        let bytes = std::fs::read(&path).map_err(|e| format!("{name}: {e}"))?;
        let sha = sha256_hex(&bytes);
        chain = sha256_hex(format!("{chain}{name}{sha}").as_bytes());
        entries.push((name, bytes.len(), sha));
    }
    Ok((entries, chain))
}

fn emit(dir: &Path) -> Result<(), String> {
    let (entries, chain) = hash_set(dir)?;
    let mut j = String::from("{\n  \"tool\": \"coyote-apple 0.1\",\n");
    j.push_str(&format!("  \"dir\": {:?},\n", dir.display().to_string()));
    j.push_str("  \"files\": [\n");
    for (i, (name, bytes, sha)) in entries.iter().enumerate() {
        let comma = if i + 1 == entries.len() { "" } else { "," };
        j.push_str(&format!(
            "    {{\"name\": \"{name}\", \"bytes\": {bytes}, \"sha256\": \"{sha}\"}}{comma}\n"
        ));
    }
    j.push_str(&format!("  ],\n  \"chain\": \"{chain}\"\n}}\n"));
    std::fs::write(dir.join(AUDIT_FILE), &j).map_err(|e| e.to_string())?;
    println!("✓ audit — {} artifacts chained", entries.len());
    for (name, bytes, sha) in &entries {
        println!("  {name:<24} {bytes:>8} B  {sha}");
    }
    println!("  chain head: {chain}");
    println!("  → {}", dir.join(AUDIT_FILE).display());
    Ok(())
}

/// pull `"sha256": "…"` off the audit line for `name`, and the head.
fn recorded(text: &str, name: &str) -> Option<String> {
    for line in text.lines() {
        if line.contains(&format!("\"name\": \"{name}\"")) {
            return line
                .split("\"sha256\": \"")
                .nth(1)
                .and_then(|s| s.split('"').next())
                .map(|s| s.to_string());
        }
    }
    None
}

fn verify(dir: &Path) -> Result<(), String> {
    let text = std::fs::read_to_string(dir.join(AUDIT_FILE))
        .map_err(|e| format!("{}: no audit chain — {e}", dir.join(AUDIT_FILE).display()))?;
    let recorded_chain = text
        .split("\"chain\": \"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .ok_or("audit.json: missing chain head")?;
    let (entries, chain) = hash_set(dir)?;
    let mut bad = 0;
    for (name, _, sha) in &entries {
        match recorded(&text, name) {
            Some(r) if r == *sha => println!("  ✓ {name}"),
            Some(r) => {
                println!("  ✗ {name} — recorded {r}");
                bad += 1;
            }
            None => {
                println!("  ✗ {name} — not in audit");
                bad += 1;
            }
        }
    }
    let head_ok = chain == recorded_chain;
    println!(
        "  chain head: {} ({})",
        if head_ok { "✓ intact" } else { "✗ BROKEN" },
        chain
    );
    if bad > 0 || !head_ok {
        return Err(format!("audit failed — {bad} mismatched files"));
    }
    println!("✓ verified — {} artifacts, chain intact", entries.len());
    Ok(())
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(dir) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: coyote-apple audit <build_dir> [--verify]");
        return Err(2);
    };
    let dir = PathBuf::from(dir);
    let r = if args.iter().any(|a| a == "--verify") {
        verify(&dir)
    } else {
        emit(&dir)
    };
    r.map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}
