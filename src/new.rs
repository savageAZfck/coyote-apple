//! `coyote-apple new <name> [--program]` — scaffold a coyote project.
//!
//! Creates `<name>/main.cy` plus a `.gitignore` for build output. The
//! default template is a full organism: invariant, provenance, a
//! tensor program, a `host fn main` that drives it, and a `test_*`
//! fn so `coyote-apple test` has something to run on first contact.
//! `--program` scaffolds a bare tensor program instead (no host
//! stratum — for graphs that are the deliverable, not the driver).

use std::path::PathBuf;

const ORGANISM: &str = r#"cap print, witness
invariant never_sells_self

provenance {
    author: "owner"
    purpose: "organism scaffold — coyote-apple new"
}

param w: f16[64, 64]
param b: f16[64]

// a top-level tensor fn is a program — compiled to silicon, callable
// by name or passed as a first-class `ane(...) -> ...` value.
fn affine(x: f16[64]) -> f16[64] {
    relu(x @ w) + b
}

host fn main() -> i64 {
    let me = self_token();
    witness("born", me);
    let x = zeros(64);
    let y = affine(x);
    print("hello from {name} — argmax", argmax(y));
    witness("done", me);
    return 0;
}

host fn test_affine() -> i64 {
    let x = zeros(64);
    let y = affine(x);
    if len(y) != 64 {
        return 1;
    }
    return 0;
}
"#;

const PROGRAM: &str = r#"invariant never_sells_self

provenance {
    author: "owner"
    purpose: "tensor program scaffold — coyote-apple new"
}

param w: f16[64, 64]
param b: f16[64]

io x: in f16[64]
io y: out f16[64]

fn affine(x: f16[64], w: f16[64, 64], b: f16[64]) -> f16[64] {
    (x @ w) + b
}

main(x: f16[64]) -> f16[64] {
    affine(x, w, b)
}
"#;

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(name) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: coyote-apple new <name> [--program]");
        return Err(2);
    };
    let program_only = args.iter().any(|a| a == "--program");
    let dir = PathBuf::from(name);
    let r = (|| -> Result<(), String> {
        if dir.exists() {
            return Err(format!("{}: already exists", dir.display()));
        }
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let src = if program_only { PROGRAM } else { ORGANISM };
        std::fs::write(dir.join("main.cy"), src.replace("{name}", name))
            .map_err(|e| e.to_string())?;
        std::fs::write(dir.join(".gitignore"), "/out\n/target\n").map_err(|e| e.to_string())?;
        println!(
            "✓ {name}/ scaffolded — {}",
            if program_only {
                "tensor program"
            } else {
                "organism"
            }
        );
        println!(
            "  main.cy     {}",
            if program_only {
                "param/io + affine main"
            } else {
                "param/io + organism main + test_affine"
            }
        );
        if !program_only {
            println!("  next:  coyote-apple test {name}/main.cy --cpu");
        } else {
            println!("  next:  coyote run {name}/main.cy -o {name}/out");
        }
        Ok(())
    })();
    r.map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}
