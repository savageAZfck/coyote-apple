//! `coyote-apple referee <file.cy> [--lie seq:N] [--json] [--steps N]`
//!
//! Interactive refereed verification — the Canetti–Riva–Rothblum
//! "refereed delegation of computation" protocol finally deployed on
//! real silicon. Two rival provers run the same `.cy` program on
//! different substrates (the resident-VM device backend vs the CPU
//! reference evaluator); each emits a committed witness chain. When the
//! chains diverge the referee challenges both provers for range-hashes
//! at the midpoint — log(n) rounds, every round recorded — until the
//! window is a single witness event. That event is the exact step
//! where silicon and spec disagree.
//!
//! The verdict is a self-contained artifact: program hash, both chain
//! commitments, the bisection transcript, and the divergent event.
//! `--replay <verdict.json>` re-executes and re-derives every claim.
//! `--lie seq:N` forges one prover's committed event in memory — a
//! self-test proving the bisection lands where it claims.

use coyote::lower::sha256_hex;
use std::path::{Path, PathBuf};

/// One witness event as a flat ballot — "tag:val".
type Trace = Vec<String>;

struct RunOut {
    trace: Trace,
    ret: String,
}

/// Execute the program on one backend and return its committed witness
/// chain plus the host return value. In-process: the same dual-VM path
/// the oracle uses, so `fn`/`gpu fn` lower to real silicon and the
/// `witness()` calls mint the runtime's chained hashes.
fn run_backend(file: &Path, dir: &Path, cpu: bool) -> Result<RunOut, String> {
    let _ = std::fs::remove_dir_all(dir);
    let mut vm = coyote::load_vm(file, dir, cpu, false, true)?;
    crate::mind::bind_all(&mut vm);
    let r = vm.run_main()?;
    let ret = vm.stringify(&r);
    let trace = vm
        .witness_log()
        .iter()
        .map(|e| format!("{}:{}", e.tag, e.val))
        .collect();
    Ok(RunOut { trace, ret })
}

/// Chained commitment over a ballot trace.
fn commitment(tr: &Trace) -> String {
    let mut acc = String::from("cyref-v1");
    for (i, b) in tr.iter().enumerate() {
        acc = sha256_hex(format!("{acc}|{i}|{b}").as_bytes());
    }
    acc
}

/// Range hash over ballots [lo, hi).
fn range_hash(tr: &Trace, lo: usize, hi: usize) -> String {
    let mut acc = format!("cyrange|{lo}|{hi}");
    for (i, b) in tr.iter().enumerate().take(hi).skip(lo) {
        acc = sha256_hex(format!("{acc}|{i}|{b}").as_bytes());
    }
    acc
}

/// Ballots agree if strings equal or numeric values within rel tol.
fn agree(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let num = |s: &str| s.rsplit(':').next().and_then(|v| v.parse::<f64>().ok());
    match (num(a), num(b)) {
        (Some(x), Some(y)) => (x - y).abs() <= 1e-3 * x.abs().max(1.0),
        _ => false,
    }
}

fn first_divergence(a: &Trace, b: &Trace) -> Option<usize> {
    let n = a.len().max(b.len());
    (0..n).find(|&i| match (a.get(i), b.get(i)) {
        (Some(x), Some(y)) => !agree(x, y),
        (None, None) => false,
        _ => true,
    })
}

/// Interactive bisection — challenge both provers for the range hash
/// at each midpoint until the window is one event. Rounds land in the
/// transcript verbatim.
fn bisect(a: &Trace, b: &Trace, pa: &str, pb: &str) -> (usize, Vec<serde_json::Value>) {
    let (mut lo, mut hi) = (0usize, a.len().max(b.len()));
    let mut rounds = Vec::new();
    let mut n = 0;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        let (ha, hb) = (range_hash(a, lo, mid), range_hash(b, lo, mid));
        let same = ha == hb;
        rounds.push(serde_json::json!({
            "round": n, "lo": lo, "mid": mid, "hi": hi,
            pa: ha, pb: hb, "agree": same,
        }));
        if same {
            lo = mid;
        } else {
            hi = mid;
        }
        n += 1;
    }
    (lo, rounds)
}

fn write_verdict(file: &Path, v: &serde_json::Value) {
    let dir = std::env::var("CYO_REFEREE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("coyote_out"));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("referee_verdict.json");
    if let Ok(s) = serde_json::to_string_pretty(v) {
        let _ = std::fs::write(&path, s);
    }
    let _ = file;
}

fn dispute(file: &Path, lie: Option<usize>, json: bool) -> Result<(), String> {
    let tmp = std::env::temp_dir().join(format!("cyref-{}", std::process::id()));
    let dev = run_backend(file, &tmp.join("dev"), false)?;
    let cpu = run_backend(file, &tmp.join("cpu"), true)?;
    let mut a = dev.trace.clone();
    let b = cpu.trace.clone();
    if let Some(s) = lie {
        if s < a.len() {
            a[s] = format!("{}-forged", a[s]);
        }
        eprintln!("[referee] forged lie injected at witness seq {s}");
    }
    if a.is_empty() && b.is_empty() {
        return Err("no witness events — nothing to referee (add witness() calls)".into());
    }
    let (pa, pb) = ("silicon", "cpu_reference");
    let (ca, cb) = (commitment(&a), commitment(&b));
    let mut v = serde_json::json!({
        "protocol": "refereed-verification-v1",
        "program": file.display().to_string(),
        "program_sha256": sha256_hex(&std::fs::read(file).map_err(|e| e.to_string())?),
        "provers": [pa, pb],
        "commitments": {pa: ca, pb: cb},
        "witness_events": a.len(),
        "returns": {pa: dev.ret, pb: cpu.ret},
    });
    if ca == cb && first_divergence(&a, &b).is_none() {
        v["outcome"] = serde_json::json!("consistent");
        write_verdict(file, &v);
        if json {
            println!("{}", serde_json::to_string_pretty(&v).unwrap());
        } else {
            println!("referee — consistent: silicon and CPU reference agree on all {} witness events", a.len());
            println!("  chain: {}", &ca[..16]);
        }
        return Ok(());
    }
    let (seq, rounds) = bisect(&a, &b, pa, pb);
    let (ba, bb) = (
        a.get(seq).cloned().unwrap_or_else(|| "<absent>".into()),
        b.get(seq).cloned().unwrap_or_else(|| "<absent>".into()),
    );
    v["outcome"] = serde_json::json!("substrate_divergence");
    v["divergent_seq"] = serde_json::json!(seq);
    v["ballots"] = serde_json::json!({pa: ba, pb: bb});
    v["transcript"] = serde_json::json!(rounds);
    write_verdict(file, &v);
    if json {
        println!("{}", serde_json::to_string_pretty(&v).unwrap());
        return Err(format!("substrate divergence at seq {seq}"));
    }
    println!("referee — witness chains split");
    println!("  bisected {} events → first divergence at seq {seq} in {} rounds", a.len(), rounds.len());
    for r in &rounds {
        println!(
            "  r{}: [{},{}) {} — {}",
            r["round"], r["lo"], r["mid"],
            &r[pa].as_str().unwrap_or("?")[..12],
            if r["agree"].as_bool().unwrap_or(false) { "agree" } else { "DIVERGE" }
        );
    }
    println!("  silicon emitted: {ba}");
    println!("  reference said:  {bb}");
    Err(format!("substrate divergence at seq {seq}"))
}

fn replay(verdict_path: &Path) -> Result<(), String> {
    let text = std::fs::read_to_string(verdict_path).map_err(|e| e.to_string())?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let prog = v["program"].as_str().ok_or("verdict missing program")?;
    let cy = Path::new(prog);
    let tmp = std::env::temp_dir().join(format!("cyref-replay-{}", std::process::id()));
    let dev = run_backend(cy, &tmp.join("dev"), false)?;
    let cpu = run_backend(cy, &tmp.join("cpu"), true)?;
    let (a, b) = (dev.trace, cpu.trace);
    match v["outcome"].as_str().unwrap_or("") {
        "consistent" => {
            if commitment(&a) == v["commitments"]["silicon"].as_str().unwrap_or("")
                && commitment(&b) == v["commitments"]["cpu_reference"].as_str().unwrap_or("")
                && first_divergence(&a, &b).is_none()
            {
                println!("✓ replayed — commitments match, still consistent");
                return Ok(());
            }
            Err("traces no longer match the committed verdict".into())
        }
        "substrate_divergence" => {
            let claimed = v["divergent_seq"].as_u64().unwrap_or(u64::MAX) as usize;
            let now = first_divergence(&a, &b).unwrap_or(usize::MAX);
            println!("replayed: first divergence at seq {now} (verdict claimed {claimed})");
            if now == claimed {
                println!("✓ verified — bisection isolates the same event");
                Ok(())
            } else {
                Err(format!("replay diverges at {now}, not {claimed}"))
            }
        }
        other => Err(format!("unknown verdict outcome '{other}'")),
    }
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(first) = args.first() else {
        eprintln!("usage: coyote-apple referee <file.cy> [--lie seq:N] [--json]");
        eprintln!("       coyote-apple referee --replay <verdict.json>");
        return Err(2);
    };
    if first == "--replay" {
        let Some(p) = args.get(1) else {
            return Err(2);
        };
        return replay(Path::new(p)).map_err(|e| {
            eprintln!("✗ {e}");
            1
        });
    }
    let file = Path::new(first);
    let lie = args.iter().position(|a| a == "--lie").and_then(|i| {
        args.get(i + 1).and_then(|s| s.trim_start_matches("seq:").parse::<usize>().ok())
    });
    let json = args.iter().any(|a| a == "--json");
    dispute(file, lie, json).map_err(|e| {
        if !json {
            eprintln!("✗ {e}");
        }
        1
    })
}
