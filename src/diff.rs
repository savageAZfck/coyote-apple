//! `coyote-apple diff <dirA> <dirB>` — what changed between two builds.
//!
//! Recursive per-file delta: same-name files compared by sha256, files
//! only present on one side listed as added/removed. Build dirs nest
//! per-program subdirs (`affine/net.mil`, `affine/w.bin`), so paths
//! are reported relative to the build root. The receipt for "what did
//! this edit actually change" — pairs with `audit`, which proves the
//! state of a single build.

use coyote::lower::sha256_hex;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn file_set(dir: &Path) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).map_err(|e| format!("{}: {e}", d.display()))? {
            let p = e.map_err(|e| e.to_string())?.path();
            if p.is_dir() {
                stack.push(p);
            } else if let Some(rel) = p.strip_prefix(dir).ok().and_then(|r| r.to_str()) {
                let bytes = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
                out.insert(rel.replace('\\', "/"), sha256_hex(&bytes));
            }
        }
    }
    Ok(out)
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let dirs: Vec<PathBuf> = args
        .iter()
        .filter(|a| !a.starts_with('-'))
        .map(PathBuf::from)
        .collect();
    let [a, b] = dirs.as_slice() else {
        eprintln!("usage: coyote-apple diff <dirA> <dirB>");
        return Err(2);
    };
    let r = (|| -> Result<(), String> {
        let sa = file_set(a)?;
        let sb = file_set(b)?;
        let mut added = 0usize;
        let mut removed = 0usize;
        let mut changed = 0usize;
        let mut same = 0usize;
        for (rel, ha) in &sa {
            match sb.get(rel) {
                Some(hb) if hb == ha => same += 1,
                Some(hb) => {
                    changed += 1;
                    println!("  ~ {rel}  {ha:.8} → {hb:.8}");
                }
                None => {
                    removed += 1;
                    println!("  - {rel}  (only in {})", a.display());
                }
            }
        }
        for rel in sb.keys() {
            if !sa.contains_key(rel) {
                added += 1;
                println!("  + {rel}  (only in {})", b.display());
            }
        }
        if added + removed + changed == 0 {
            println!("✓ identical — {same} files match");
        } else {
            println!("Δ {same} same, {changed} changed, +{added} added, -{removed} removed");
        }
        Ok(())
    })();
    r.map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}
