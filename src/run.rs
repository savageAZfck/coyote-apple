//! run.rs — `coyote-apple run <file.cy>`: the same organism pipeline as
//! `coyote run` (parse → typecheck → tensor programs → resident VM →
//! `host fn main`), plus this distribution's extern layer bound onto
//! the VM before `main` executes.

use crate::mind;
use std::path::PathBuf;

fn out_dir(args: &[String]) -> PathBuf {
    args.windows(2)
        .find(|w| w[0] == "-o")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(|| PathBuf::from("coyote_out"))
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(file) = args
        .first()
        .filter(|a| !a.starts_with('-'))
        .map(PathBuf::from)
    else {
        eprintln!("coyote-apple run <file.cy> [-o dir] [--cpu] [--rebuild] [--resume snap]");
        return Err(2);
    };
    let dir = out_dir(&args[1..]);
    let use_cpu = args.iter().any(|a| a == "--cpu");
    let rebuild = args.iter().any(|a| a == "--rebuild");
    let resume = args
        .windows(2)
        .find(|w| w[0] == "--resume")
        .map(|w| PathBuf::from(&w[1]));
    let export = args
        .windows(2)
        .find(|w| w[0] == "--export")
        .map(|w| PathBuf::from(&w[1]));

    let r = (|| -> Result<(), String> {
        let mut vm = coyote::load_vm(&file, &dir, use_cpu, rebuild, false)?;
        mind::bind_all(&mut vm);
        crate::peer::bind_all(&mut vm);
        let ret = if let Some(snap) = resume {
            vm.resume(snap.to_str().unwrap_or(""))?;
            vm.run_resumed()?
        } else {
            vm.run_main()?
        };
        // the witness log is part of the artifact — hash-chained,
        // written next to the binaries.
        let mut wl = String::new();
        for w in vm.witness_log() {
            wl.push_str(&format!(
                "{{\"seq\":{},\"tag\":\"{}\",\"val\":\"{}\",\"hash\":\"{}\"}}\n",
                w.seq, w.tag, w.val, w.hash
            ));
        }
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("witness.jsonl"), &wl).map_err(|e| e.to_string())?;
        println!("✓ organism returned {}", vm.stringify(&ret));
        if !vm.witness_log().is_empty() {
            println!(
                "  witness: {} entries → {}",
                vm.witness_log().len(),
                dir.join("witness.jsonl").display()
            );
        }
        if let Some(ep) = &export {
            // streaming checkpoint export — the plan carries only
            // metadata (name/dims/dtype/source) so the header's
            // data_offsets are computable up front; tensors then
            // stream one param at a time straight into the file.
            // Peak host memory is one param (~30MB for lmh), not
            // 3× the whole model as f32 — `params_snapshot`'s
            // accumulate-then-write path needed ~45GB and wedged.
            let plan = vm.param_export_plan()?;
            let mut hdr = String::from("{");
            let mut off = 0u64;
            for (i, it) in plan.iter().enumerate() {
                if i > 0 {
                    hdr.push(',');
                }
                let esz = if it.f16 { 2u64 } else { 4u64 };
                let nb = it.dims.iter().product::<usize>() as u64 * esz;
                let shape_s = it
                    .dims
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                hdr.push_str(&format!(
                    "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{}, {}]}}",
                    it.name,
                    if it.f16 { "F16" } else { "F32" },
                    shape_s,
                    off,
                    off + nb
                ));
                off += nb;
            }
            hdr.push('}');
            let pad = (8 - (hdr.len() % 8)) % 8;
            hdr.push_str(&" ".repeat(pad));
            let mut file = std::io::BufWriter::new(
                std::fs::File::create(ep).map_err(|e| e.to_string())?,
            );
            use std::io::Write;
            file.write_all(&(hdr.len() as u64).to_le_bytes())
                .map_err(|e| e.to_string())?;
            file.write_all(hdr.as_bytes()).map_err(|e| e.to_string())?;
            let n_tensors = plan.len();
            vm.param_export_emit(&plan, |_, _, data, f16| {
                if f16 {
                    for v in data {
                        file.write_all(&half::f16::from_f32(*v).to_le_bytes())
                            .map_err(|e| e.to_string())?;
                    }
                } else {
                    for v in data {
                        file.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
                    }
                }
                Ok(())
            })?;
            file.flush().map_err(|e| e.to_string())?;
            println!("  exported {} tensors → {}", n_tensors, ep.display());
        }
        Ok(())
    })();
    if let Err(e) = r {
        eprintln!("✗ {e}");
        return Err(1);
    }
    Ok(())
}
