//! `coyote-apple watch <file.cy> [--run] [-o dir] [--steps N]`
//!
//! The dev loop: poll the file (and its `import`ed modules) every
//! 400ms, and on any change re-run `coyote check` — or `coyote run`
//! with `--run`. Failures print and the loop keeps going; the next
//! save is the next build. Ctrl-C to stop.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

fn deps(file: &Path) -> Vec<PathBuf> {
    let mut out = vec![file.to_path_buf()];
    let Ok(src) = std::fs::read_to_string(file) else {
        return out;
    };
    let base = file.parent().unwrap_or_else(|| Path::new("."));
    for line in src.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("import") {
            if let Some(q) = rest
                .trim()
                .strip_prefix('"')
                .and_then(|r| r.strip_suffix('"'))
            {
                let mut p = base.join(q);
                if !p.is_file() {
                    p = base.join(format!("{q}.cy"));
                }
                if let Ok(pp) = std::env::var("CYO_PATH") {
                    if !p.is_file() {
                        for d in pp.split(':').filter(|s| !s.is_empty()) {
                            let cand = Path::new(d).join(q).with_extension("cy");
                            if cand.is_file() {
                                p = cand;
                                break;
                            }
                        }
                    }
                }
                out.push(p);
            }
        }
    }
    out
}

fn newest(files: &[PathBuf]) -> Option<SystemTime> {
    files
        .iter()
        .filter_map(|f| std::fs::metadata(f).ok()?.modified().ok())
        .max()
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: coyote-apple watch <file.cy> [--run] [-o dir] [--steps N]");
        return Err(2);
    };
    let do_run = args.iter().any(|a| a == "--run");
    let dir = args
        .windows(2)
        .find(|w| w[0] == "-o")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| {
            format!(
                "out/{}",
                Path::new(file)
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap_or("watch")
            )
        });
    let steps = args
        .windows(2)
        .find(|w| w[0] == "--steps")
        .map(|w| w[1].clone());

    println!("watch — {file}  ({})", if do_run { "run" } else { "check" });
    let mut last = None;
    loop {
        let ds = deps(Path::new(file));
        let m = newest(&ds);
        if m != last {
            last = m;
            let mut argv = vec![
                "coyote".into(),
                if do_run { "run".into() } else { "check".into() },
                file.to_string(),
                "-o".into(),
                dir.clone(),
            ];
            if let Some(s) = &steps {
                argv.push("--steps".into());
                argv.push(s.clone());
            }
            let t = std::time::Instant::now();
            // `check` builds no VM — straight through. `run` needs this
            // distribution's pipeline so mind externs are bound before
            // `main` executes — the public run can't bind them.
            let r = if do_run {
                let mut a = vec![file.to_string(), "-o".into(), dir.clone()];
                if let Some(s) = &steps {
                    a.push("--steps".into());
                    a.push(s.clone());
                }
                crate::run::run(&a)
            } else {
                coyote::cli::run(&argv)
            };
            match r {
                Ok(()) => println!(
                    "  ✓ {} ({:.1}s, {} deps)\n",
                    if do_run { "run" } else { "check" },
                    t.elapsed().as_secs_f64(),
                    ds.len()
                ),
                Err(c) => println!("  ✗ failed (exit {c}) — waiting for next save\n"),
            }
        }
        std::thread::sleep(Duration::from_millis(400));
    }
}
