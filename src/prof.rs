//! `coyote-apple prof <file.cy> [--runs N] [--json]`
//!
//! The nsight-lite: per-dispatch profiling for tensor programs. Emits
//! and loads the program through its real backend, runs N dispatches,
//! and reports host wall time plus *device-side* kernel time where the
//! backend can measure it (gpu: Metal command-buffer GPU timestamps;
//! ane: host wall only — the silicon's counters aren't public).
//! A parity diff against the CPU evaluator rides along — timing
//! without correctness is noise.

use coyote::lower::init_weight_pub;
use coyote::typeck::TProgram;
use coyote::{eval::Evaluator, vm};
use std::path::PathBuf;
use std::time::Instant;

struct Stats {
    emit_ms: f64,
    host_min: f64,
    host_avg: f64,
    dev_min: Option<f64>,
    dev_avg: Option<f64>,
    parity: f32,
    outs: usize,
}

fn summarize(times: &mut Vec<f64>) -> (f64, f64) {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times.first().copied().unwrap_or(0.0);
    let avg = times.iter().sum::<f64>() / times.len().max(1) as f64;
    (min, avg)
}

fn eval_parity(prog: &TProgram, inputs: &[(String, Vec<f32>)], got: &[f32]) -> f32 {
    let mut ev = Evaluator::new(prog);
    let map: std::collections::HashMap<String, Vec<f32>> = inputs.iter().cloned().collect();
    match ev.run(&map) {
        Ok(cpu) => got
            .iter()
            .zip(&cpu)
            .map(|(a, c)| (a - c).abs())
            .fold(0f32, f32::max),
        Err(_) => f32::NAN,
    }
}

/// Registered-backend program (gpu today): host wall + device kernel
/// time + parity.
fn prof_ext(src: &str, prog: &TProgram, dir: &PathBuf, runs: usize) -> Result<Stats, String> {
    let be = coyote::backend::backend_for(prog.target)?;
    let t0 = Instant::now();
    let out = be.emit(src, "main", prog, dir)?;
    let emit_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut rt = be
        .load(prog, "main", &out, dir)
        .map_err(|e| format!("ext load: {e}"))?;
    let mut writes: Vec<(String, Vec<f32>)> = Vec::new();
    let mut inputs = Vec::new();
    for (name, ty, _c) in &out.inputs {
        let data = init_weight_pub(&format!("io::{name}"), ty.numel() as usize);
        writes.push((name.clone(), data.clone()));
        inputs.push((name.clone(), data));
    }
    for (_k, port, ty) in &out.states {
        writes.push((port.clone(), vec![0.0; ty.numel() as usize]));
    }
    let mut host = Vec::new();
    let mut dev = Vec::new();
    // parity compares the FIRST run's output — update programs
    // legitimately evolve weights every dispatch, so only step 0 is
    // spec-equivalent to a single eval pass.
    let mut first = Vec::new();
    for i in 0..=runs {
        let t = Instant::now();
        let r = rt.run(&writes)?;
        if i == 0 && !r.is_empty() {
            first = r;
        }
        host.push(t.elapsed().as_secs_f64() * 1e3);
        if let Some(d) = rt.device_ms() {
            dev.push(d);
        }
    }
    host.remove(0); // warmup
    if !dev.is_empty() {
        dev.remove(0);
    }
    let (host_min, host_avg) = summarize(&mut host);
    let (dev_min, dev_avg) = if dev.is_empty() {
        (None, None)
    } else {
        let (mn, av) = summarize(&mut dev);
        (Some(mn), Some(av))
    };
    Ok(Stats {
        emit_ms,
        host_min,
        host_avg,
        dev_min,
        dev_avg,
        parity: eval_parity(prog, &inputs, &first),
        outs: out.output.1.numel() as usize,
    })
}

/// ANE program: host wall per dispatch (no public device counters) +
/// parity vs the evaluator.
fn prof_ane(src: &str, prog: &TProgram, dir: &PathBuf, runs: usize) -> Result<Stats, String> {
    let t0 = Instant::now();
    let out = coyote::emit_program(src, prog, dir, false)?;
    let emit_ms = t0.elapsed().as_secs_f64() * 1e3;
    let plan = mil_infer::graph::compile_graph(
        dir.join("net.mil").to_str().unwrap(),
        dir.to_str().unwrap(),
    )
    .map_err(|e| format!("compile_graph: {e}"))?;
    let mut rt =
        mil_infer::graph::GraphRuntime::load(plan).map_err(|e| format!("runtime load: {e}"))?;
    let mut inputs = Vec::new();
    for (name, ty, _c) in &out.inputs {
        let data = init_weight_pub(&format!("io::{name}"), ty.numel() as usize);
        vm::write_surface(&mut rt, name, &vm::dims_usize(ty), &data);
        inputs.push((name.clone(), data));
    }
    for (pname, port, cty, src) in &out.trainable {
        let (n, k) = (cty.dims[1] as usize, cty.dims[0] as usize);
        let w = coyote::weights::resolve(pname, Some(&[n, k]), src.as_deref(), init_weight_pub)
            .map_err(|e| e.to_string())?;
        vm::write_surface(&mut rt, port, &[n, k], &w);
    }
    for (_k, port, ty) in &out.states {
        vm::write_surface(
            &mut rt,
            port,
            &vm::dims_usize(ty),
            &vec![0.0; ty.numel() as usize],
        );
    }
    let (oname, oty) = &out.output;
    let mut host = Vec::new();
    let mut got = Vec::new();
    for _ in 0..=runs {
        let t = Instant::now();
        rt.run().map_err(|e| format!("ane run: {e}"))?;
        host.push(t.elapsed().as_secs_f64() * 1e3);
        if got.is_empty() {
            let idx = rt
                .out_node(oname)
                .ok_or_else(|| format!("output node {oname} missing"))?;
            let desc = rt.out_desc(idx);
            let raw = rt.output(idx).to_vec();
            got = vm::read_surface_dims(&raw, &desc, &vm::dims_usize(oty));
        }
    }
    host.remove(0);
    let (host_min, host_avg) = summarize(&mut host);
    Ok(Stats {
        emit_ms,
        host_min,
        host_avg,
        dev_min: None,
        dev_avg: None,
        parity: eval_parity(prog, &inputs, &got),
        outs: oty.numel() as usize,
    })
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: coyote-apple prof <file.cy> [--runs N] [--json]");
        return Err(2);
    };
    let json = args.iter().any(|a| a == "--json");
    let runs: usize = args
        .windows(2)
        .find(|w| w[0] == "--runs")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(20);
    let dir = std::env::temp_dir().join(format!("cyprof-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).ok();

    let r = (|| -> Result<(), String> {
        let (src, ast) =
            coyote::parser::parse_file(std::path::Path::new(file)).map_err(|d| d.to_string())?;
        let checked = coyote::typeck::check_all(&ast).map_err(|d| d.to_string())?;
        if checked.organism {
            return Err("prof profiles tensor programs — use `bench` for organisms".into());
        }
        let (name, prog) = checked
            .ane
            .first()
            .ok_or("no tensor program in file")?
            .clone();
        let s = match prog.target {
            coyote::ast::Target::Ane => prof_ane(&src, &prog, &dir, runs),
            _ => prof_ext(&src, &prog, &dir, runs),
        }?;
        if json {
            println!(
                "{{\"file\":{file:?},\"program\":{name:?},\"backend\":{:?},\"emit_ms\":{:.3},\"host_min_ms\":{:.3},\"host_avg_ms\":{:.3},\"device_min_ms\":{},\"device_avg_ms\":{},\"parity\":{:.6},\"outputs\":{}}}",
                prog.target.name(),
                s.emit_ms,
                s.host_min,
                s.host_avg,
                s.dev_min.map(|v| format!("{v:.3}")).unwrap_or("null".into()),
                s.dev_avg.map(|v| format!("{v:.3}")).unwrap_or("null".into()),
                s.parity,
                s.outs
            );
        } else {
            println!("prof — {name}  [{} backend]", prog.target.name());
            println!("  emit      {emit:>8.3} ms", emit = s.emit_ms);
            println!(
                "  dispatch  {min:>8.3} ms min   {avg:.3} avg   ({runs} runs)",
                min = s.host_min,
                avg = s.host_avg,
                runs = runs
            );
            match (s.dev_min, s.dev_avg) {
                (Some(mn), Some(av)) => {
                    println!("  device    {mn:>8.3} ms min   {av:.3} avg   (GPU timestamps)")
                }
                _ => println!("  device    n/a — backend exposes no kernel counters"),
            }
            println!("  parity    {:.6}   outputs {}", s.parity, s.outs);
        }
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&dir);
    r.map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}
