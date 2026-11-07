//! `coyote-apple prof <file.cy> [--runs N] [--json]`
//!
//! The nsight-lite: per-dispatch profiling for tensor programs. Emits
//! and loads the program through its real backend, runs N dispatches,
//! and reports host wall time plus *device-side* kernel time where the
//! backend can measure it (gpu: Metal command-buffer GPU timestamps;
//! ane: host wall only — the silicon's counters aren't public).
//! A parity diff against the CPU evaluator rides along — timing
//! without correctness is noise.

use coyote::lower::{init_weight_dims, init_weight_pub};
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
    /// kernel name → best device ms (CYO_PKT per-kernel timestamps).
    kernels: Vec<(String, f64)>,
    /// same program through the reference CPU evaluator — the third
    /// backend in the same-source comparison.
    cpu_min: f64,
    cpu_avg: f64,
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

/// CPU column: the same typed program through the reference evaluator,
/// state persistent across dispatches like the silicon runtimes.
fn prof_cpu(prog: &TProgram, inputs: &[(String, Vec<f32>)], runs: usize) -> (f64, f64) {
    let map: std::collections::HashMap<String, Vec<f32>> = inputs.iter().cloned().collect();
    let mut ev = Evaluator::new(prog);
    let mut times = Vec::new();
    for i in 0..=runs {
        let t = Instant::now();
        if ev.run(&map).is_err() {
            break;
        }
        if i > 0 {
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    summarize(&mut times)
}

/// Registered-backend program (gpu today): host wall + device kernel
/// time + parity.
fn prof_ext(
    src: &str,
    prog: &TProgram,
    dir: &PathBuf,
    runs: usize,
    dump: bool,
    no_cpu: bool,
) -> Result<Stats, String> {
    let be = coyote::backend::backend_for(prog.target)?;
    let t0 = Instant::now();
    let out = be.emit(src, "main", prog, dir)?;
    let emit_ms = t0.elapsed().as_secs_f64() * 1e3;
    // gpu loads concretely so the run loop keeps buffer access
    // (`read_buf` for --dump); other backends go through the trait.
    enum Rt {
        Gpu(crate::gpu_rt::GpuRuntime),
        Other(Box<dyn coyote::backend::BackendExec>),
    }
    impl Rt {
        fn run(&mut self, w: &[(String, Vec<f32>)]) -> Result<Vec<f32>, String> {
            match self {
                Rt::Gpu(r) => r.run(w),
                Rt::Other(r) => r.run(w),
            }
        }
        fn device_ms(&self) -> Option<f64> {
            match self {
                Rt::Gpu(r) => coyote::backend::BackendExec::device_ms(r),
                Rt::Other(r) => r.device_ms(),
            }
        }
        fn kernel_ms(&self) -> &[(String, f64)] {
            match self {
                Rt::Gpu(r) => r.kernel_ms(),
                Rt::Other(_) => &[],
            }
        }
    }
    let mut rt = if let Some(g) = out.opaque.downcast_ref::<crate::gpu_emit::GpuOut>() {
        Rt::Gpu(crate::gpu_rt::GpuRuntime::load(prog, "main", g, None)?)
    } else {
        Rt::Other(
            be.load(prog, "main", &out, dir)
                .map_err(|e| format!("ext load: {e}"))?,
        )
    };
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
    // per-kernel mins, kept in emission order (kernel_ms returns
    // pipe order).
    let mut kmins: Vec<(String, f64)> = Vec::new();
    for i in 0..=runs {
        let t = Instant::now();
        let r = rt.run(&writes)?;
        if i == 0 && !r.is_empty() {
            first = r;
        }
        host.push(t.elapsed().as_secs_f64() * 1e3);
        if i == 0 {
            continue; // warmup — don't time emit-side stalls
        }
        if let Some(d) = rt.device_ms() {
            dev.push(d);
        }
        for (j, (kn, kms)) in rt.kernel_ms().iter().enumerate() {
            match kmins.get_mut(j) {
                Some((_, m)) => *m = m.min(*kms),
                None => kmins.push((kn.clone(), *kms)),
            }
        }
    }
    host.remove(0); // warmup
    let (host_min, host_avg) = summarize(&mut host);
    let (dev_min, dev_avg) = if dev.is_empty() {
        (None, None)
    } else {
        let (mn, av) = summarize(&mut dev);
        (Some(mn), Some(av))
    };
    if dump {
        if let Rt::Gpu(g) = &rt {
            for (pname, _port, ty, src) in &out.trainable {
                let expect: Vec<usize> = if ty.dims.len() == 2 {
                    vec![ty.dims[1] as usize, ty.dims[0] as usize]
                } else {
                    vec![ty.numel() as usize]
                };
                let init =
                    coyote::weights::resolve(pname, Some(&expect), src.as_deref(), init_weight_dims)
                        .unwrap_or_default();
                if let Some(got) = g.read_buf(pname) {
                    let drift = got
                        .iter()
                        .zip(&init)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    println!(
                        "  param {pname:<8} max|Δ vs init| {drift:.6}   w[0] {:.6} (init {:.6})",
                        got.first().copied().unwrap_or(0.0),
                        init.first().copied().unwrap_or(0.0),
                    );
                }
            }
        } else {
            println!("  dump    n/a — backend exposes no buffer readback");
        }
    }
    // `--no-cpu` skips the reference-evaluator parity pass — at
    // organism scale the scalar eval materializes the whole tape and
    // gets jetsam-killed before the device numbers can print.
    let (cpu_min, cpu_avg, parity) = if no_cpu {
        (0.0f64, 0.0f64, f32::NAN)
    } else {
        let (cmin, cavg) = prof_cpu(prog, &inputs, runs);
        (cmin, cavg, eval_parity(prog, &inputs, &first))
    };
    Ok(Stats {
        emit_ms,
        host_min,
        host_avg,
        dev_min,
        dev_avg,
        kernels: kmins,
        cpu_min,
        cpu_avg,
        parity,
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
        let w = coyote::weights::resolve(pname, Some(&[n, k]), src.as_deref(), init_weight_dims)
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
    let (cpu_min, cpu_avg) = prof_cpu(prog, &inputs, runs);
    Ok(Stats {
        emit_ms,
        host_min,
        host_avg,
        dev_min: None,
        dev_avg: None,
        kernels: Vec::new(),
        cpu_min,
        cpu_avg,
        parity: eval_parity(prog, &inputs, &got),
        outs: oty.numel() as usize,
    })
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: coyote-apple prof <file.cy> [--runs N] [--json] [--no-cpu]");
        return Err(2);
    };
    let json = args.iter().any(|a| a == "--json");
    let dump = args.iter().any(|a| a == "--dump");
    let no_cpu = args.iter().any(|a| a == "--no-cpu");
    // `--only <name>` profiles a single tensor program — big files
    // (own4b: forward + 9 blocks + head) OOM/jetsam when every program
    // emits and loads in one process.
    let only: Option<String> = args
        .windows(2)
        .find(|w| w[0] == "--only")
        .map(|w| w[1].clone());
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
        // organisms are welcome: their tensor programs are still
        // first-class emit units, and a file carrying `fn` + `gpu fn`
        // for the same math is the same-source cross-backend story.
        if checked.ane.is_empty() {
            return Err("no tensor program in file".into());
        }
        // every tensor program in the file gets its own row — a file
        // that carries `fn` + `gpu fn` for the same math prints the
        // same-source comparison across ane, gpu and the cpu evaluator.
        for (name, prog) in checked.ane.iter() {
            if let Some(o) = &only {
                if name != o {
                    continue;
                }
            }
            let pdir = dir.join(name);
            std::fs::create_dir_all(&pdir).ok();
            let s = match prog.target {
                coyote::ast::Target::Ane => prof_ane(&src, prog, &pdir, runs),
                _ => prof_ext(&src, prog, &pdir, runs, dump, no_cpu),
            }?;
            if json {
                println!(
                    "{{\"file\":{file:?},\"program\":{name:?},\"backend\":{:?},\"emit_ms\":{:.3},\"host_min_ms\":{:.3},\"host_avg_ms\":{:.3},\"device_min_ms\":{},\"device_avg_ms\":{},\"cpu_min_ms\":{:.3},\"cpu_avg_ms\":{:.3},\"parity\":{:.6},\"outputs\":{}}}",
                    prog.target.name(),
                    s.emit_ms,
                    s.host_min,
                    s.host_avg,
                    s.dev_min.map(|v| format!("{v:.3}")).unwrap_or("null".into()),
                    s.dev_avg.map(|v| format!("{v:.3}")).unwrap_or("null".into()),
                    s.cpu_min,
                    s.cpu_avg,
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
                for (kn, kms) in &s.kernels {
                    println!("    {kn:<32} {kms:>8.3} ms");
                }
                println!(
                    "  cpu eval  {min:>8.3} ms min   {avg:.3} avg   (same program, reference evaluator)",
                    min = s.cpu_min,
                    avg = s.cpu_avg
                );
                println!("  parity    {:.6}   outputs {}", s.parity, s.outs);
            }
        }
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&dir);
    r.map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}
