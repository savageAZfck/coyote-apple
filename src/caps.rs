//! `coyote-apple caps <file.cy> [--deny a,b] [--allow a,b] [--pure] [--json]`
//!
//! Capability enforcement — the `cap` decls are already in the grammar;
//! this verb makes them a *gate*. Reports the program's capability
//! surface by stratum — program-level `cap` decls, each tensor
//! program's declared caps, host-fn signature caps, extern-fn caps —
//! then applies a policy:
//!
//!   --deny  a,b     fail if any declared cap is in the deny list
//!   --allow a,b     fail if any declared cap is NOT in the allow list
//!   --pure          fail if ANY capability is declared (provably pure
//!                   compute — the organ can't reach anything)
//!
//! A `.cy` organ's attack surface becomes a checkable fact, not a
//! promise: `caps organ.cy --pure` proves it touches nothing;
//! `caps organ.cy --deny net` proves it can't exfiltrate.

use coyote::parser;
use coyote::typeck;
use serde_json::json;
use std::collections::BTreeSet;
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(file) = args
        .iter()
        .find(|a| !a.starts_with('-') && !a.contains('='))
    else {
        eprintln!(
            "usage: coyote-apple caps <file.cy> [--deny a,b] [--allow a,b] [--pure] [--json]"
        );
        return Err(2);
    };
    let list = |flag: &str| -> Vec<String> {
        args.iter()
            .find(|a| a.starts_with(&format!("{flag}=")))
            .map(|a| {
                a[flag.len() + 1..]
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect()
            })
            .unwrap_or_default()
    };
    let deny = list("--deny");
    let allow = list("--allow");
    let pure = args.iter().any(|a| a == "--pure");
    let json_out = args.iter().any(|a| a == "--json");

    let r = (|| -> Result<Vec<String>, String> {
        let path = PathBuf::from(file);
        let (_s, ast) = parser::parse_file(&path).map_err(|d| d.to_string())?;
        let c = typeck::check_all(&ast).map_err(|d| d.to_string())?;

        let mut all: BTreeSet<String> = BTreeSet::new();
        // program-level `cap` decls
        for cap in &c.caps {
            all.insert(cap.clone());
        }
        // per-stratum tensor programs
        let mut progs = Vec::new();
        for (name, p) in &c.ane {
            for cap in &p.caps {
                all.insert(cap.clone());
            }
            progs.push(json!({
                "name": name, "target": p.target.name(), "caps": p.caps,
            }));
        }
        // host fns: signature caps
        let mut hosts = Vec::new();
        if let Some(h) = &c.host {
            for f in &h.fns {
                for cap in &f.caps {
                    all.insert(cap.clone());
                }
                hosts.push(json!({"name": f.name, "caps": f.caps}));
            }
            // externs: declared capability requirements
            for e in &h.externs {
                for cap in &e.caps {
                    all.insert(cap.clone());
                }
                hosts.push(json!({"extern": e.name, "caps": e.caps}));
            }
        }
        let caps_vec: Vec<String> = all.iter().cloned().collect();

        if json_out {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "file": path.display().to_string(),
                    "caps": caps_vec,
                    "programs": progs,
                    "host": hosts,
                }))
                .unwrap()
            );
        } else {
            println!("coyote-apple caps — {}", path.display());
            for (name, p) in &c.ane {
                println!(
                    "  {} stratum {:<10} caps: {}",
                    p.target.name(),
                    name,
                    if p.caps.is_empty() {
                        "<none>".into()
                    } else {
                        p.caps.join(", ")
                    }
                );
            }
            for cap in &c.caps {
                println!("  program cap {cap}");
            }
            println!(
                "  total: {} distinct capabilit{}",
                caps_vec.len(),
                if caps_vec.len() == 1 { "y" } else { "ies" }
            );
        }

        // policy gate
        let mut violations = Vec::new();
        for cap in &caps_vec {
            if deny.contains(cap) {
                violations.push(format!("'{cap}' is denied"));
            }
            if !allow.is_empty() && !allow.contains(cap) {
                violations.push(format!("'{cap}' not in allow list"));
            }
        }
        if pure && !caps_vec.is_empty() {
            violations.push(format!("--pure: {} caps declared", caps_vec.len()));
        }
        Ok(violations)
    })();

    match r {
        Ok(v) if v.is_empty() => Ok(()),
        Ok(v) => {
            eprintln!("✗ cap violations:");
            for x in v {
                eprintln!("  {x}");
            }
            Err(1)
        }
        Err(e) => {
            eprintln!("✗ {e}");
            Err(1)
        }
    }
}
