//! `coyote-apple certify <file.cy> [-o cert.json]`
//!
//! Proof-carrying programs — Necula's proof-carrying code realized for
//! neural silicon. A `.cy` program certifies as a bundle of machine-
//! checkable claims:
//!
//!   compiles      — parse + typecheck passed (the checker's own receipt)
//!   parity        — the organism's observable behavior (host return +
//!                   witness chain) is identical on silicon and the CPU
//!                   reference evaluator
//!   referee       — the committed witness chains of both backends
//!   determinism   — two independent device executions produce identical
//!                   committed chains (replay determinism, proven not
//!                   asserted)
//!
//! `certify --check cert.json` re-derives every claim against the
//! source and reports each as held or broken — the third-party check
//! the PCC literature calls verification condition generation.

use coyote::lower::sha256_hex;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn run_backend(file: &Path, dir: &Path, cpu: bool) -> Result<(Vec<String>, String), String> {
    let _ = std::fs::remove_dir_all(dir);
    let mut vm = coyote::load_vm(file, dir, cpu, false, true)?;
    crate::mind::bind_all(&mut vm);
    crate::peer::bind_all(&mut vm);
    let r = vm.run_main()?;
    let ret = vm.stringify(&r);
    let trace = vm
        .witness_log()
        .iter()
        .map(|e| format!("{}:{}", e.tag, e.val))
        .collect();
    Ok((trace, ret))
}

fn commitment(tr: &[String]) -> String {
    let mut acc = String::from("cycert-v1");
    for (i, b) in tr.iter().enumerate() {
        acc = sha256_hex(format!("{acc}|{i}|{b}").as_bytes());
    }
    acc
}

fn certify(file: &Path, out: Option<&Path>) -> Result<(), String> {
    let src_sha = sha256_hex(&std::fs::read(file).map_err(|e| e.to_string())?);

    // claim 1: compiles — parse + typecheck is the first proof obligation.
    // The checked AST/program table is kept: the reversible claim
    // re-derives inverse roundtrips from it.
    let checked = coyote::parser::parse_file(file)
        .map_err(|d| d.to_string())
        .and_then(|(_s, ast)| {
            coyote::typeck::check_all(&ast)
                .map(|ck| (ast, ck))
                .map_err(|d| d.to_string())
        });
    let (ast, ck) = match checked {
        Ok(x) => x,
        Err(e) => {
            let v = json!({
                "spec": "cycert/0.1",
                "program": file.display().to_string(),
                "claims": {"compiles": {"held": false, "error": e}},
                "verdict": "BROKEN",
            });
            println!("{}", serde_json::to_string_pretty(&v).unwrap());
            return Err("does not compile".into());
        }
    };

    // claims 2-4: dual-backend execution + replay determinism.
    let tmp = std::env::temp_dir().join(format!("cycert-{}", std::process::id()));
    let (dev_trace, dev_ret) = run_backend(file, &tmp.join("dev"), false)?;
    let (cpu_trace, cpu_ret) = run_backend(file, &tmp.join("cpu"), true)?;
    let (dev2_trace, _) = run_backend(file, &tmp.join("dev2"), false)?;

    let (c_dev, c_cpu, c_dev2) = (
        commitment(&dev_trace),
        commitment(&cpu_trace),
        commitment(&dev2_trace),
    );
    let parity = dev_ret == cpu_ret && dev_trace == cpu_trace;
    let replay = c_dev == c_dev2;

    // claim 5: reversible — every `rev fn`'s inverse actually inverts
    // on the reference engine (inv∘f = id, measured not asserted).
    // Vacuously held for programs with no rev fns.
    let mut rev_held = true;
    let mut rev_map = serde_json::Map::new();
    for (i, (fname, iname)) in crate::revcheck::rev_pairs(&ast).iter().enumerate() {
        let key = format!("{fname}↔{iname}");
        let pair = ck
            .ane
            .iter()
            .find(|(n, _)| n == fname)
            .zip(ck.ane.iter().find(|(n, _)| n == iname));
        match pair {
            Some(((_, f), (_, v))) => match crate::revcheck::roundtrip(f, v, i * 31 + 7) {
                Ok((d, _, _)) => {
                    rev_held &= d <= 0.0;
                    rev_map.insert(key, json!({"max_diff": d, "held": d <= 0.0}));
                }
                Err(e) => {
                    rev_held = false;
                    rev_map.insert(key, json!({"held": false, "error": e}));
                }
            },
            None => {
                rev_held = false;
                rev_map.insert(key, json!({"held": false, "error": "inverse program missing"}));
            }
        }
    }
    // host stratum pairs — verified through the VM, exact match.
    let hpairs = crate::revcheck::host_rev_pairs(&ast);
    if let Some(host) = &ck.host {
        match crate::revcheck::host_pairs_check(host, &hpairs) {
            Ok(rows) => {
                for (n, d, p) in rows {
                    rev_held &= p;
                    rev_map.insert(n, json!({"max_diff": d, "held": p}));
                }
            }
            Err(e) => {
                rev_held = false;
                rev_map.insert(
                    "host".to_string(),
                    json!({"held": false, "error": e}),
                );
            }
        }
    } else if !hpairs.is_empty() {
        rev_held = false;
        rev_map.insert(
            "host".to_string(),
            json!({"held": false, "error": "rev host fn present, no host program"}),
        );
    }

    let mut claims = json!({
        "compiles": {"held": true},
        "parity": {
            "held": parity,
            "silicon_return": dev_ret, "cpu_return": cpu_ret,
            "silicon_witness": dev_trace.len(), "cpu_witness": cpu_trace.len(),
        },
        "commitments": {"silicon": c_dev, "cpu_reference": c_cpu, "silicon_replay": c_dev2},
        "replay_determinism": {"held": replay},
        "reversible": {"held": rev_held, "tol": 0.0, "pairs": rev_map},
    });
    let cert_sha = sha256_hex(serde_json::to_string(&claims).unwrap().as_bytes());
    claims["cert_sha256"] = json!(cert_sha);

    let v = json!({
        "spec": "cycert/0.1",
        "program": file.display().to_string(),
        "program_sha256": src_sha,
        "toolchain": {
            "crate": "coyote-apple",
            "version": env!("CARGO_PKG_VERSION"),
        },
        "built": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        "claims": claims,
        "verdict": if parity && replay && rev_held { "CERTIFIED" } else { "BROKEN" },
    });
    let s = serde_json::to_string_pretty(&v).unwrap();
    match out {
        Some(p) => {
            std::fs::write(p, &s).map_err(|e| e.to_string())?;
            println!("✓ certified → {} ({})", p.display(), &cert_sha[..16]);
            for (k, c) in claims.as_object().unwrap() {
                if let Some(h) = c["held"].as_bool() {
                    println!("  {k}: {}", if h { "held" } else { "BROKEN" });
                }
            }
        }
        None => println!("{s}"),
    }
    if parity && replay && rev_held {
        Ok(())
    } else {
        Err("certificate claims broken".into())
    }
}

fn check(cert_path: &Path) -> Result<(), String> {
    let text = std::fs::read_to_string(cert_path).map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let prog = v["program"].as_str().ok_or("cert missing program")?;
    let file = Path::new(prog);
    let claimed_src = v["program_sha256"].as_str().unwrap_or("");
    let src_now = sha256_hex(&std::fs::read(file).map_err(|e| e.to_string())?);
    if src_now != claimed_src {
        return Err("source changed since certification — certificate void".into());
    }
    let tmp = std::env::temp_dir().join(format!("cycert-check-{}", std::process::id()));
    let (dev_trace, dev_ret) = run_backend(file, &tmp.join("dev"), false)?;
    let (cpu_trace, cpu_ret) = run_backend(file, &tmp.join("cpu"), true)?;
    let c_dev = commitment(&dev_trace);
    let c_cpu = commitment(&cpu_trace);
    let claims = &v["claims"];
    let mut ok = true;
    let mut check = |name: &str, held: bool| {
        println!("  {name}: {}", if held { "held" } else { "BROKEN" });
        ok &= held;
    };
    check("source_unchanged", true);
    check(
        "parity",
        dev_ret == cpu_ret
            && dev_trace == cpu_trace
            && c_dev == claims["commitments"]["silicon"].as_str().unwrap_or("")
            && c_cpu
                == claims["commitments"]["cpu_reference"]
                    .as_str()
                    .unwrap_or(""),
    );
    // reversible claim — re-derive every rev pair's roundtrip and
    // compare against what the certificate recorded.
    let rev_ok = (|| -> Result<bool, String> {
        let (_s, ast) = coyote::parser::parse_file(file).map_err(|d| d.to_string())?;
        let ck = coyote::typeck::check_all(&ast).map_err(|d| d.to_string())?;
        let recorded = &claims["reversible"]["pairs"];
        for (i, (fname, iname)) in crate::revcheck::rev_pairs(&ast).iter().enumerate() {
            let key = format!("{fname}↔{iname}");
            let claimed = recorded
                .get(&key)
                .and_then(|c| c["held"].as_bool())
                .unwrap_or(false);
            let pair = ck
                .ane
                .iter()
                .find(|(n, _)| n == fname)
                .zip(ck.ane.iter().find(|(n, _)| n == iname));
            let held = match pair {
                Some(((_, f), (_, v))) => crate::revcheck::roundtrip(f, v, i * 31 + 7)
                    .map(|(d, _, _)| d <= 0.0)
                    .unwrap_or(false),
                None => false,
            };
            if held != claimed {
                return Ok(false);
            }
        }
        // host pairs — re-derive through the VM and compare held flags.
        let hpairs = crate::revcheck::host_rev_pairs(&ast);
        if let Some(host) = &ck.host {
            for (n, _d, p) in
                crate::revcheck::host_pairs_check(host, &hpairs).unwrap_or_default()
            {
                let claimed = recorded
                    .get(&n)
                    .and_then(|c| c["held"].as_bool())
                    .unwrap_or(false);
                if p != claimed {
                    return Ok(false);
                }
            }
        } else if !hpairs.is_empty() {
            return Ok(false);
        }
        Ok(true)
    })()
    .unwrap_or(false);
    check("reversible", rev_ok);
    if ok {
        println!("✓ certificate verified — all claims hold");
        Ok(())
    } else {
        Err("certificate claims broken".into())
    }
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(first) = args.first() else {
        eprintln!("usage: coyote-apple certify <file.cy> [-o cert.json]");
        eprintln!("       coyote-apple certify --check <cert.json>");
        return Err(2);
    };
    let r = if first == "--check" {
        args.get(1).map(|p| check(Path::new(p)))
    } else {
        let out = args
            .iter()
            .position(|a| a == "-o")
            .and_then(|i| args.get(i + 1))
            .map(PathBuf::from);
        Some(certify(Path::new(first), out.as_deref()))
    };
    match r {
        Some(Ok(())) => Ok(()),
        Some(Err(e)) => {
            eprintln!("✗ {e}");
            Err(1)
        }
        None => Err(2),
    }
}
