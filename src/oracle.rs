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

/// The gate invokes THIS binary's `run` passthrough — the public
/// `coyote` binary deliberately cannot dispatch every stratum (gpu and
/// any future first-party targets only have engines registered here).
fn self_bin() -> Result<PathBuf, String> {
    std::env::current_exe().map_err(|e| e.to_string())
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
    let out = Command::new(self_bin()?)
        .args(&a)
        .output()
        .map_err(|e| format!("spawn coyote: {e}"))?;
    Ok(RunOut {
        stdout: String::from_utf8_lossy(&out.stdout).to_string()
            + &String::from_utf8_lossy(&out.stderr),
        status: out.status.code().unwrap_or(-1),
    })
}

/// worst `|<backend>−cpu| pred` / `w` numbers from `run` output, plus
/// the single-shot `max |<backend> - cpu| =` line if present. Returns
/// the backend label actually seen (ane, gpu, …) for reporting.
fn diffs(text: &str) -> (f64, f64, &'static str) {
    let (mut pred, mut w) = (0f64, 0f64);
    let mut label = "sil";
    for l in text.lines() {
        if l.contains("−cpu| pred") || l.contains("-cpu| pred") {
            for b in ["gpu", "ane"] {
                if l.contains(&format!("{b}−cpu")) || l.contains(&format!("{b}-cpu")) {
                    label = b;
                }
            }
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
        } else if l.contains("max |") && l.contains("cpu| =") {
            if let Some(v) = l.rsplit('=').next() {
                if let Ok(v) = v.trim().parse::<f64>() {
                    pred = pred.max(v);
                }
            }
        }
    }
    (pred, w, label)
}

/// `diffprobe` sibling binary — the resident-pipeline differential.
/// Lives next to this exe in cargo builds.
fn probe_bin() -> Result<PathBuf, String> {
    Ok(self_bin()?
        .parent()
        .ok_or("oracle: no exe dir")?
        .join("diffprobe"))
}

/// first index of a `top5 [(I, v), ...]` line — the argmax token.
fn top1(line: &str) -> Option<u32> {
    line.split("[(")
        .nth(1)?
        .split(',')
        .next()?
        .trim()
        .parse()
        .ok()
}

/// worst `max|d|=` per stat category diffprobe prints —
/// (hidden, kv-state, logits) — plus whether the argmax agreed at
/// every step. Absolute error scales with tensor magnitude: logits
/// stay ~17, residual hidden ~400 at depth, KV ~30 — the gate reads
/// them separately.
fn probe_diffs(text: &str) -> (f64, f64, f64, bool) {
    let (mut hid, mut kv, mut lg) = (0f64, 0f64, 0f64);
    let mut agree = true;
    let mut cpu_top: Option<u32> = None;
    for l in text.lines() {
        if l.contains("cpu top5") {
            cpu_top = top1(l);
        } else if l.contains("ane top5") {
            if cpu_top.is_some() && top1(l) != cpu_top {
                agree = false;
            }
            cpu_top = None;
        }
        let Some(v) = l
            .split("max|d|=")
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse::<f64>().ok())
        else {
            continue;
        };
        if l.contains("hidden") {
            hid = hid.max(v);
        } else if l.contains("logits") {
            lg = lg.max(v);
        } else if l.contains("kv_") {
            kv = kv.max(v);
        }
    }
    (hid, kv, lg, agree)
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: coyote-apple oracle <file.cy> [--steps N] [--tol X] [--json]");
        eprintln!("       coyote-apple oracle <file.cy> --resident <shard_dir> [--steps N] [--tol X] [--json]");
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
    let resident = args
        .windows(2)
        .find(|w| w[0] == "--resident")
        .map(|w| PathBuf::from(&w[1]));
    // logits tol: resident mode defaults to 0.75 — fp16-vs-f32 drift
    // across a 28-layer stack runs ~0.2-0.5 and keeps growing with
    // steps; the hard gate is argmax agreement anyway. Per-op 0.05.
    let tol: f64 = args
        .windows(2)
        .find(|w| w[0] == "--tol")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(if resident.is_some() { 0.75 } else { 0.05 });
    let tmp = std::env::temp_dir().join(format!("cyoracle-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let dir_ane = tmp.join("ane");
    let dir_cpu = tmp.join("cpu");

    let r = (|| -> Result<(), String> {
        if let Some(shard) = &resident {
            // resident-pipeline parity: CPU evaluator vs the ANE
            // resident decode, tensor-level. Three categories gate on
            // different scales — logits drive argmax (strict, --tol),
            // hidden/state ride residual magnitudes ~30× logits at
            // depth (fp16 accumulation, so tol·32).
            let probe = probe_bin()?;
            let out = Command::new(&probe)
                .arg(file)
                .arg(shard)
                .arg(steps.to_string())
                .output()
                .map_err(|e| format!("spawn diffprobe: {e}"))?;
            let text = String::from_utf8_lossy(&out.stdout).to_string()
                + &String::from_utf8_lossy(&out.stderr);
            if !out.status.success() {
                return Err(format!("diffprobe exited {:?}:\n{text}", out.status.code()));
            }
            let (hid, kv, lg, agree) = probe_diffs(&text);
            let pass = agree && lg <= tol && hid <= tol * 32.0 && kv <= tol * 32.0;
            if json {
                println!(
                    "{{\"file\":{file:?},\"mode\":\"resident-parity\",\"shards\":{:?},\"steps\":{steps},\"argmax_agree\":{agree},\"logits_diff\":{lg:.6},\"hidden_diff\":{hid:.6},\"state_diff\":{kv:.6},\"tol\":{tol},\"verdict\":\"{}\"}}",
                    shard.display().to_string(),
                    if pass { "pass" } else { "fail" }
                );
            } else {
                println!("oracle — resident pipeline parity ({file}, {steps} steps)");
                println!(
                    "  argmax {}   logits {lg:.6} (tol {tol})   hidden {hid:.6}   state {kv:.6} (tol {})",
                    if agree { "agrees" } else { "DIVERGES" },
                    tol * 32.0
                );
                println!(
                    "{}",
                    if pass {
                        "✓ pass — resident silicon agrees with spec"
                    } else {
                        "✗ fail — resident divergence exceeds tolerance"
                    }
                );
            }
            if !pass {
                return Err(format!(
                    "resident parity: argmax_agree={agree} logits {lg:.6} hidden {hid:.6} state {kv:.6}"
                ));
            }
            return Ok(());
        }
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
            crate::mind::bind_all(&mut va);
            let mut vc = coyote::load_vm(std::path::Path::new(file), &dir_cpu, true, false, true)?;
            crate::mind::bind_all(&mut vc);
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
            let (pred, w, label) = diffs(&ane.stdout);
            let pass = pred <= tol;
            if json {
                println!(
                    "{{\"file\":{file:?},\"mode\":\"tensor-parity\",\"backend\":{label:?},\"steps\":{steps},\"max_pred_diff\":{pred:.6},\"max_state_diff\":{w:.6},\"tol\":{tol},\"verdict\":\"{}\"}}",
                    if pass { "pass" } else { "fail" }
                );
            } else {
                println!("oracle — silicon vs spec ({file}, {steps} steps)");
                println!("  max |{label}−cpu| pred {pred:.6}   state/w {w:.6}   tol {tol}");
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
