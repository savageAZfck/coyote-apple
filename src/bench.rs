//! `coyote-apple bench <file.cy>` — receipts, timed.
//!
//! Two measurements, both real:
//!   * pipeline — `coyote run` end to end (emit cache → compile →
//!     execute → parity diff), averaged over `--runs` iterations.
//!   * exec — for organism files, `host fn main` wall time inside a
//!     resident VM on both backends (ane compile amortized at load).
//!
//! Everything is labelled as what it is: pipeline numbers include
//! compile; exec numbers are the hot path alone.

use coyote::parser;
use coyote::typeck;
use std::path::PathBuf;
use std::time::Instant;

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: coyote-apple bench <file.cy> [-o dir] [--steps N] [--runs N] [--cpu]");
        return Err(2);
    };
    let path = PathBuf::from(file);
    let dir = args
        .windows(2)
        .find(|w| w[0] == "-o")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(|| PathBuf::from("coyote_bench"));
    let steps: Option<usize> = args
        .windows(2)
        .find(|w| w[0] == "--steps")
        .and_then(|w| w[1].parse().ok());
    let runs: usize = args
        .windows(2)
        .find(|w| w[0] == "--runs")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(1);
    let use_cpu = args.iter().any(|a| a == "--cpu");
    let json = args.iter().any(|a| a == "--json");

    let r = (|| -> Result<(), String> {
        if !json {
            println!("coyote-apple bench — {}", path.display());
        }

        // ---- pipeline: `coyote run` end to end ----
        let mut times = Vec::new();
        for _ in 0..runs {
            let mut argv: Vec<String> = vec![
                "coyote".into(),
                "run".into(),
                path.to_str().unwrap_or(file).into(),
                "-o".into(),
                dir.to_str().unwrap_or("coyote_bench").into(),
            ];
            if let Some(s) = steps {
                argv.push("--steps".into());
                argv.push(s.to_string());
            }
            if use_cpu {
                argv.push("--cpu".into());
            }
            let t0 = Instant::now();
            coyote::cli::run(&argv).map_err(|c| format!("run failed (exit {c})"))?;
            times.push(t0.elapsed());
        }
        let avg = times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / times.len() as f64;
        let min = times
            .iter()
            .map(|t| t.as_secs_f64())
            .fold(f64::INFINITY, f64::min);
        if !json {
            println!(
                "  pipeline    {} · avg {:.2}s · best {:.2}s  ({runs} run{}, emit+compile+exec)",
                if use_cpu { "cpu" } else { "ane" },
                avg,
                min,
                if runs == 1 { "" } else { "s" }
            );
        }

        // ---- exec: host fn main inside a resident vm ----
        let (_src, ast) = parser::parse_file(&path).map_err(|d| d.to_string())?;
        let checked = typeck::check_all(&ast).map_err(|d| d.to_string())?;
        let mut exec = None;
        if checked.organism {
            let mut vm = coyote::load_vm(&path, &dir, use_cpu, false, true)?;
            crate::mind::bind_all(&mut vm);
            let mut exec_times = Vec::new();
            for _ in 0..runs.max(3) {
                let t0 = Instant::now();
                vm.run_main()?;
                exec_times.push(t0.elapsed());
            }
            let eavg =
                exec_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / exec_times.len() as f64;
            let emin = exec_times
                .iter()
                .map(|t| t.as_secs_f64())
                .fold(f64::INFINITY, f64::min);
            let w = vm.witness_log().len();
            if !json {
                println!(
                    "  exec (main) {} · avg {:.2}ms · best {:.2}ms  ({} calls, hot path)",
                    if vm.is_ane() { "ane" } else { "cpu" },
                    eavg * 1e3,
                    emin * 1e3,
                    exec_times.len()
                );
                if w > 0 {
                    println!("  witness     {w} entries appended during bench");
                }
            }
            exec = Some((eavg, emin, exec_times.len(), w));
        } else if !json {
            println!("  exec (main) — (no host fn main; pipeline timing only)");
        }
        if json {
            // machine-readable report — always the last stdout line
            let exec_j = exec
                .map(|(a, m, n, w)| {
                    format!(
                        "{{\"avg_ms\":{:.4},\"best_ms\":{:.4},\"calls\":{n},\"witness\":{w}}}",
                        a * 1e3,
                        m * 1e3
                    )
                })
                .unwrap_or_else(|| "null".into());
            println!(
                "{{\"file\":{:?},\"backend\":\"{}\",\"pipeline\":{{\"avg_s\":{:.4},\"best_s\":{:.4},\"runs\":{runs}}},\"exec\":{exec_j}}}",
                path.display().to_string(),
                if use_cpu { "cpu" } else { "ane" },
                avg,
                min
            );
        }
        Ok(())
    })();
    r.map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}
