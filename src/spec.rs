//! `coyote-apple spec <file.cy>` — the capability matrix.
//!
//! Parses and shape-checks the program (the real typechecker — this is
//! what the program *is*, not a guess) and reports structure, declared
//! constitution (invariants/caps/provenance), and every tensor program's
//! surface: io, params, states, return shape.

use coyote::ast::{IoDir, Qual};
use coyote::parser;
use coyote::typeck::{self, CTy, TProgram};
use std::path::PathBuf;

fn dims(t: &CTy) -> String {
    format!("{:?}{:?}", t.dt, t.dims).to_lowercase()
}

fn qual(q: &Qual) -> &'static str {
    match q {
        Qual::Trainable => "trainable",
        Qual::Frozen => "frozen",
        Qual::Sealed => "sealed",
        Qual::Plain => "plain",
    }
}

fn dir(d: &IoDir) -> &'static str {
    match d {
        IoDir::In => "in ",
        IoDir::Out => "out",
    }
}

fn show_prog(name: &str, p: &TProgram) {
    println!("  program {name}");
    for (n, d, t) in &p.ios {
        println!("    io      {:<8} {:<4} {}", n, dir(d), dims(t));
    }
    for pr in &p.params {
        let src = pr.src.as_deref().unwrap_or("-");
        println!(
            "    param   {:<8} {:<9} {}  @ {}",
            pr.name,
            qual(&pr.qual),
            dims(&pr.ty),
            src
        );
    }
    for (k, port, t) in &p.states {
        println!("    state   {:<8} {:<8} {}", k, port, dims(t));
    }
    println!("    body    {} stmts → {}", p.body.len(), dims(&p.ret.ty));
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: coyote-apple spec <file.cy>");
        return Err(2);
    };
    let path = PathBuf::from(file);
    let r = (|| -> Result<(), String> {
        let (_src, ast) = parser::parse_file(&path).map_err(|d| d.to_string())?;
        let c = typeck::check_all(&ast).map_err(|d| d.to_string())?;

        println!("coyote-apple spec — {}", path.display());
        println!("  toolchain  coyote 0.1 (apple profile) · hwx emitter library");
        println!(
            "  programs   {} tensor · {} host fns",
            c.ane.len(),
            c.host.as_ref().map(|h| h.fns.len()).unwrap_or(0)
        );
        if !c.invariants.is_empty() {
            println!("  invariants {}", c.invariants.join(", "));
        }
        if !c.caps.is_empty() {
            println!("  caps       {}", c.caps.join(", "));
        }
        for (name, p) in &c.ane {
            show_prog(name, p);
            if !p.provenance.is_empty() {
                for (k, v) in &p.provenance {
                    println!("    provenance {k}: {v}");
                }
            }
        }
        if let Some(h) = &c.host {
            let nref = h.fns.iter().filter(|f| f.is_reflex).count();
            println!(
                "  host       {} fns ({} reflex), {} globals, {} structs, {} enums",
                h.fns.len(),
                nref,
                h.globals.len(),
                h.structs.len(),
                h.enums.len()
            );
            for f in &h.fns {
                let args = f
                    .args
                    .iter()
                    .map(|(n, t)| format!("{n}: {}", t.show(&h.structs, &h.enums)))
                    .collect::<Vec<_>>()
                    .join(", ");
                let tag = if f.is_reflex { " [reflex]" } else { "" };
                let ret = f.ret.show(&h.structs, &h.enums);
                println!("    {:<16} ({args}) -> {ret}{tag}", f.name);
            }
        }
        Ok(())
    })();
    r.map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}
