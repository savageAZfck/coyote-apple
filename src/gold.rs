//! `coyote-apple gold record <file.cy> [-o f.gold]` — snapshot outputs
//! `coyote-apple gold check   <file.cy> [-o f.gold]` — diff vs snapshot
//!
//! Golden regression for tensor programs: run each program through
//! the CPU reference evaluator (the executable spec) on the same
//! deterministic inputs `coyote run` uses, hash the outputs into a
//! `<file>.gold` record. `check` recomputes — a source or param change
//! that alters outputs breaks the golden and exits 1. `oracle` proves
//! silicon matches spec; `gold` proves spec matches history.

use coyote::eval::Evaluator;
use coyote::lower::{init_weight_pub, sha256_hex};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

struct Golden {
    prog: String,
    elems: usize,
    sha: String,
    head: Vec<f32>,
}

fn evaluate(file: &Path) -> Result<Vec<Golden>, String> {
    let (_src, ast) = coyote::parser::parse_file(file).map_err(|d| d.to_string())?;
    let c = coyote::typeck::check_all(&ast).map_err(|d| d.to_string())?;
    if c.ane.is_empty() {
        return Err(format!("{}: no tensor programs to gold", file.display()));
    }
    let mut out = Vec::new();
    for (name, tp) in &c.ane {
        let mut inputs: HashMap<String, Vec<f32>> = HashMap::new();
        for (port, dir, ty) in &tp.ios {
            if *dir != coyote::ast::IoDir::In {
                continue;
            }
            let n = ty.numel() as usize;
            inputs.insert(port.clone(), init_weight_pub(&format!("io::{port}"), n));
        }
        let mut ev = Evaluator::new(tp);
        let vals = ev.run(&inputs).map_err(|d| format!("eval {name}: {d}"))?;
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        out.push(Golden {
            prog: name.clone(),
            elems: vals.len(),
            sha: sha256_hex(&bytes),
            head: vals.iter().take(8).copied().collect(),
        });
    }
    Ok(out)
}

fn gold_path(file: &Path, args: &[String]) -> PathBuf {
    args.windows(2)
        .find(|w| w[0] == "-o")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(|| file.with_extension("gold"))
}

fn render(file: &Path, gs: &[Golden], src_sha: &str) -> String {
    let mut s = String::from("{\n");
    s.push_str(&format!("  \"file\": {:?},\n", file.display().to_string()));
    s.push_str(&format!("  \"src_sha256\": \"{src_sha}\",\n"));
    s.push_str("  \"programs\": [\n");
    for (i, g) in gs.iter().enumerate() {
        let comma = if i + 1 == gs.len() { "" } else { "," };
        let head = g
            .head
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(", ");
        s.push_str(&format!(
            "    {{\"name\": \"{}\", \"elems\": {}, \"sha256\": \"{}\", \"head\": [{head}]}}{comma}\n",
            g.prog, g.elems, g.sha
        ));
    }
    s.push_str("  ]\n}\n");
    s
}

/// pull `"name"` → `sha256` pairs back out of a golden file
fn parse(text: &str) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for line in text.lines() {
        let Some(n) = line.split("\"name\": \"").nth(1) else {
            continue;
        };
        let name = n.split('"').next().unwrap_or("");
        if let Some(s) = line.split("\"sha256\": \"").nth(1) {
            m.insert(
                name.to_string(),
                s.split('"').next().unwrap_or("").to_string(),
            );
        }
    }
    m
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(sub) = args.first().map(|s| s.as_str()) else {
        eprintln!("usage: coyote-apple gold <record|check> <file.cy> [-o f.gold]");
        return Err(2);
    };
    let rest = &args[1..];
    let Some(file) = rest.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: coyote-apple gold <record|check> <file.cy> [-o f.gold]");
        return Err(2);
    };
    let file = PathBuf::from(file);
    let gp = gold_path(&file, rest);

    let r = (|| -> Result<(), String> {
        let src = std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
        let src_sha = sha256_hex(src.as_bytes());
        let gs = evaluate(&file)?;
        match sub {
            "record" => {
                std::fs::write(&gp, render(&file, &gs, &src_sha))
                    .map_err(|e| format!("{}: {e}", gp.display()))?;
                println!("✓ gold recorded → {}", gp.display());
                for g in &gs {
                    println!("  {:<12} {} elems  {}", g.prog, g.elems, g.sha);
                }
                Ok(())
            }
            "check" => {
                let text = std::fs::read_to_string(&gp).map_err(|e| {
                    format!(
                        "{}: no golden — {e} (run `gold record` first)",
                        gp.display()
                    )
                })?;
                let recorded = parse(&text);
                let mut bad = 0;
                for g in &gs {
                    match recorded.get(&g.prog) {
                        Some(r) if *r == g.sha => {
                            println!("  ✓ {:<12} {} elems", g.prog, g.elems)
                        }
                        Some(r) => {
                            println!(
                                "  ✗ {:<12} golden {} → now {}",
                                g.prog,
                                &r[..16],
                                &g.sha[..16]
                            );
                            bad += 1;
                        }
                        None => {
                            println!("  ✗ {:<12} not in golden", g.prog);
                            bad += 1;
                        }
                    }
                }
                if bad > 0 {
                    return Err(format!("gold broken — {bad} program(s) differ"));
                }
                println!("✓ gold intact — {} program(s) match", gs.len());
                Ok(())
            }
            _ => {
                eprintln!("usage: coyote-apple gold <record|check> <file.cy> [-o f.gold]");
                Err("usage".into())
            }
        }
    })();
    r.map_err(|e| {
        if e != "usage" {
            eprintln!("✗ {e}");
        }
        1
    })
}
