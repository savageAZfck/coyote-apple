//! `coyote-apple weights inspect <file.safetensors> [--grep pat]`
//! `coyote-apple weights slice <file> <tensor>[a:b,c:d] [-o out.st] [--name n]`
//!
//! Checkpoint tooling for `param ... @ "file::tensor"` sources.
//! `inspect` inventories a safetensors file — every tensor's dtype,
//! shape, element count, and payload size — so param declarations get
//! written against facts, not guesses. `slice` cuts a tensor (or a
//! strided box out of it, same range syntax the param loader accepts)
//! into a new F32 safetensors a `.cy` file can bind directly.

use crate::stfile::safetensors_f32;
use coyote::weights::SafeTensors;
use std::path::PathBuf;

fn inspect(args: &[String]) -> Result<(), String> {
    let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: coyote-apple weights inspect <file.safetensors> [--grep pat]");
        return Err("usage".into());
    };
    let grep = args
        .windows(2)
        .find(|w| w[0] == "--grep")
        .map(|w| w[1].clone());
    let st = SafeTensors::load(file)?;
    let names = st.tensor_names();
    let (mut shown, mut elems, mut bytes) = (0usize, 0usize, 0usize);
    println!("{} — {} tensors", file, names.len());
    for n in names {
        if let Some(g) = &grep {
            if !n.contains(g.as_str()) {
                continue;
            }
        }
        let shape = st.shape(n).unwrap_or(&[]);
        let ne: usize = shape.iter().product();
        let nb = st.bytes(n).unwrap_or(0);
        elems += ne;
        bytes += nb;
        shown += 1;
        let shp = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x");
        println!(
            "  {n:<60} {:>4} [{shp:<14}] {ne:>10} elems  {:>8.1} KiB",
            st.dtype(n).unwrap_or("?"),
            nb as f64 / 1024.0
        );
    }
    println!(
        "  — {shown} shown · {elems} elements · {:.1} MiB payload",
        bytes as f64 / 1048576.0
    );
    Ok(())
}

fn slice(args: &[String]) -> Result<(), String> {
    // positionals — anything not starting with '-' and not consumed as
    // a value by a flag that takes one
    let takes_val = ["-o", "--name"];
    let mut pos = Vec::new();
    for (i, a) in args.iter().enumerate() {
        if a.starts_with('-') {
            continue;
        }
        if i > 0 && takes_val.contains(&args[i - 1].as_str()) {
            continue;
        }
        pos.push(a);
    }
    let [file, spec] = pos.as_slice() else {
        eprintln!(
            "usage: coyote-apple weights slice <file> <tensor>[a:b,c:d] [-o out.safetensors] [--name n]"
        );
        return Err("usage".into());
    };
    let out = args
        .windows(2)
        .find(|w| w[0] == "-o")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(|| PathBuf::from("slice.safetensors"));
    let (tensor, ranges) = match spec.split_once('[') {
        Some((t, rest)) => {
            let inner = rest
                .strip_suffix(']')
                .ok_or("unterminated range — want tensor[a:b,c:d]")?;
            let mut rs = Vec::new();
            for part in inner.split(',') {
                let (a, b) = part
                    .split_once(':')
                    .ok_or_else(|| format!("bad range {part:?}"))?;
                rs.push((
                    a.parse::<usize>()
                        .map_err(|_| format!("bad range {part:?}"))?,
                    b.parse::<usize>()
                        .map_err(|_| format!("bad range {part:?}"))?,
                ));
            }
            (t.to_string(), Some(rs))
        }
        None => (spec.to_string(), None),
    };
    let name = args
        .windows(2)
        .find(|w| w[0] == "--name")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| tensor.clone());

    let st = SafeTensors::load(file)?;
    let shape = st.shape(&tensor).ok_or(format!("missing {tensor}"))?;
    let (vals, oshape) = match &ranges {
        None => (st.f32s(&tensor)?, shape.to_vec()),
        Some(rs) => (
            st.f32s_box(&tensor, rs)?,
            rs.iter().map(|(a, b)| b - a).collect(),
        ),
    };
    std::fs::write(&out, &safetensors_f32(&name, &oshape, &vals))
        .map_err(|e| format!("{}: {e}", out.display()))?;
    println!(
        "✓ {tensor}{} → {}::{name} — {oshape:?}, {} elems (f32)",
        ranges
            .as_ref()
            .map(|rs| {
                format!(
                    "[{}]",
                    rs.iter()
                        .map(|(a, b)| format!("{a}:{b}"))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .unwrap_or_default(),
        out.display(),
        vals.len()
    );
    Ok(())
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(sub) = args.first().map(|s| s.as_str()) else {
        eprintln!("usage: coyote-apple weights <inspect|slice> ...");
        return Err(2);
    };
    let r = match sub {
        "inspect" | "ls" => inspect(&args[1..]),
        "slice" | "cut" => slice(&args[1..]),
        _ => {
            eprintln!("usage: coyote-apple weights <inspect|slice> ...");
            return Err(2);
        }
    };
    r.map_err(|e| {
        if e != "usage" {
            eprintln!("✗ {e}");
        }
        1
    })
}
