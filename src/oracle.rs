//! `coyote-apple oracle <file.cy> [--steps N] [--tol X] [--json]`
//!
//! The semantic gate: does silicon agree with the spec? `coyote run`
//! already diffs every output against the CPU reference evaluator —
//! this verb runs it twice (ANE backend and CPU backend), harvests the
//! printed parity numbers, and turns them into a CI verdict:
//!
//!   tensor program   → worst |ane−cpu| across all steps, vs --tol
//!   organism         → `host fn main` return value + witness count,
//!                      identical across both backends or fail
//!
//! Exit 0 = silicon agreed. Exit 1 = divergence or error. `--json`
//! emits the machine-readable report on stdout.

use std::path::PathBuf;
use std::process::Command;

fn coyote_bin() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    // 1. same target dir (monorepo workspace build)
    let sib = exe.with_file_name("coyote");
    if sib.is_file() {
        return Ok(sib);
    }
    // 2. sibling repo: coyote-apple/target/... → coyote/target/{debug,release}
    if let Some(repo) = exe.ancestors().nth(3) {
        for prof in ["release", "debug"] {
            let cand = repo.join("../coyote/target").join(prof).join("coyote");
            if cand.is_file() {
                return Ok(cand);
            }
        }
    }
    // 3. PATH
    Ok(PathBuf::from("coyote"))
}

struct RunOut {
    stdout: String,
    status: i32,
}

fn coyote_run(file: &str, dir: &PathBuf, steps: usize, cpu: bool) -> Result<RunOut, String> {
    let mut a = vec![
        "run".to_string(),
        file.to_string(),
        "-o".to_string(),
        dir.to_string_lossy().to_string(),
        "--steps".to_string(),
        steps.to_string(),
    ];
    if cpu {
        a.push("--cpu".into());
    }
    let out = Command::new(coyote_bin()?)
        .args(&a)
        .output()
        .map_err(|e| format!("spawn coyote: {e}"))?;
    Ok(RunOut {
        stdout: String::from_utf8_lossy(&out.stdout).to_string()
            + &String::from_utf8_lossy(&out.stderr),
        status: out.status.code().unwrap_or(-1),
    })
}

/// worst `|ane−cpu| pred` / `w` numbers from `coyote run` output, plus
/// the single-shot `max |ane - cpu| =` line if present.
fn diffs(text: &str) -> (f64, f64) {
    let (mut pred, mut w) = (0f64, 0f64);
    for l in text.lines() {
        if l.contains("ane−cpu| pred") || l.contains("ane-cpu| pred") {
            if let Some(v) = l
                .split("pred ")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<f64>().ok())
            {
                pred = pred.max(v);
            }
            for key in [" w ", " kv "] {
                if let Some(v) = l
                    .split(key)
                    .nth(1)
                    .and_then(|s| s.split_whitespace().next())
                    .and_then(|s| s.parse::<f64>().ok())
                {
                    w = w.max(v);
                }
            }
        } else if l.contains("max |ane - cpu| =") || l.contains("max |ane − cpu| =") {
            if let Some(v) = l.rsplit('=').next() {
                if let Ok(v) = v.trim().parse::<f64>() {
                    pred = pred.max(v);
                }
            }
        }
    }
    (pred, w)
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: coyote-apple oracle <file.cy> [--steps N] [--tol X] [--json]");
        return Err(2);
    };
    let json = args.iter().any(|a| a == "--json");
    // default 2 steps: recompiled nodes can schedule differently
    // run-to-run, and weight feedback amplifies the drift — a short
    // gate window keeps the verdict on deterministic ground. Pass
    // --steps/--tol explicitly for deeper or looser gates.
    let steps = args
        .windows(2)
        .find(|w| w[0] == "--steps")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(2);
    let tol: f64 = args
        .windows(2)
        .find(|w| w[0] == "--tol")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(0.05);
    let tmp = std::env::temp_dir().join(format!("cyoracle-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let dir_ane = tmp.join("ane");
    let dir_cpu = tmp.join("cpu");

    let r = (|| -> Result<(), String> {
        let organism = {
            let (_s, ast) = coyote::parser::parse_file(std::path::Path::new(file))
                .map_err(|d| d.to_string())?;
            coyote::typeck::check_all(&ast)
                .map_err(|d| d.to_string())?
                .organism
        };
        if organism {
            // dual resident VMs — ane backend vs cpu backend, same host
            // stratum, compare observable behavior: return + witness
            let mut va = coyote::load_vm(std::path::Path::new(file), &dir_ane, false, false, true)?;
            let mut vc = coyote::load_vm(std::path::Path::new(file), &dir_cpu, true, false, true)?;
            let ra = va.run_main()?;
            let rc = vc.run_main()?;
            let ar = va.stringify(&ra);
            let cr = vc.stringify(&rc);
            let (aw, cw) = (va.witness_log().len(), vc.witness_log().len());
            let pass = ar == cr && aw == cw;
            if json {
                println!(
                    "{{\"file\":{file:?},\"mode\":\"organism\",\"ane_return\":{ar:?},\"cpu_return\":{cr:?},\"ane_witness\":{aw},\"cpu_witness\":{cw},\"verdict\":\"{}\"}}",
                    if pass { "pass" } else { "fail" }
                );
            } else {
                println!("oracle — organism parity ({file})");
                println!("  ane: main → {ar}   witness {aw} entries");
                println!("  cpu: main → {cr}   witness {cw} entries");
                println!(
                    "{}",
                    if pass {
                        "✓ pass — same observable behavior on both backends"
                    } else {
                        "✗ fail — organism diverges across backends"
                    }
                );
            }
            if !pass {
                return Err("organism parity failed".into());
            }
        } else {
            let ane = coyote_run(file, &dir_ane, steps, false)?;
            if ane.status != 0 {
                return Err(format!(
                    "coyote run (ane) exited {}:\n{}",
                    ane.status, ane.stdout
                ));
            }
            // tensor mode — harvest the per-step parity diffs
            let (pred, w) = diffs(&ane.stdout);
            let pass = pred <= tol;
            if json {
                println!(
                    "{{\"file\":{file:?},\"mode\":\"tensor-parity\",\"steps\":{steps},\"max_pred_diff\":{pred:.6},\"max_state_diff\":{w:.6},\"tol\":{tol},\"verdict\":\"{}\"}}",
                    if pass { "pass" } else { "fail" }
                );
            } else {
                println!("oracle — silicon vs spec ({file}, {steps} steps)");
                println!("  max |ane−cpu| pred {pred:.6}   state/w {w:.6}   tol {tol}");
                println!(
                    "{}",
                    if pass {
                        "✓ pass — silicon agrees with spec within tolerance"
                    } else {
                        "✗ fail — divergence exceeds tolerance"
                    }
                );
            }
            if !pass {
                return Err(format!("parity {pred:.6} exceeds tol {tol}"));
            }
        }
        let _ = std::fs::remove_dir_all(&tmp);
        Ok(())
    })();
    r.map_err(|e| {
        let _ = std::fs::remove_dir_all(&tmp);
        eprintln!("✗ {e}");
        1
    })
}
