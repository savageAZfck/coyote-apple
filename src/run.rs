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
        Ok(())
    })();
    if let Err(e) = r {
        eprintln!("✗ {e}");
        return Err(1);
    }
    Ok(())
}
