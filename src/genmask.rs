//! `coyote-apple genmask <name>:<rank>:<dim> -o mask.safetensors`
//! `coyote-apple genmask <name>:<rank>:<rows>:<cols> -o mask.safetensors`
//!
//! Emit a rank-selector mask as a safetensors file. In the language the
//! mask sits on the bottleneck VECTOR, which is what makes a full-shape
//! trainable adapter behave like a rank-`r` one:
//!
//!   `((x @ la) * sel) @ lb` — sel[i<r]=1 zeroes the dead coordinates,
//!   so la's masked columns and lb's dead rows receive exactly zero
//!   gradient. Forward stays on the skeleton table; the rank lives in
//!   the data, not the shape.
//!
//!   sel:8:64        →  "sel" [64],     sel[i] = 1 iff i < 8
//!   am:8:64:64      →  "am"  [64,64],  am[i,j] = 1 iff j < 8 (column mask)
//!   am:8:64:64 rows →  `--rows` makes it a row mask instead

use std::path::PathBuf;

fn safetensors_f32(name: &str, shape: &[usize], data: &[f32]) -> Vec<u8> {
    let nbytes = data.len() * 4;
    let shape_s = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let header = format!(
        "{{\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape_s}],\"data_offsets\":[0, {nbytes}]}}}}"
    );
    let pad = (8 - (header.len() % 8)) % 8;
    let header = format!("{header}{:pad$}", "", pad = pad);
    let mut out = Vec::with_capacity(8 + header.len() + nbytes);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let usage = || {
        eprintln!("usage: coyote-apple genmask <name>:<rank>:<dim> [-o mask.safetensors]");
        eprintln!("       coyote-apple genmask <name>:<rank>:<rows>:<cols> [--rows] [-o file]");
    };
    let Some(spec) = args.iter().find(|a| !a.starts_with('-')) else {
        usage();
        return Err(2);
    };
    let row_mask = args.iter().any(|a| a == "--rows");
    let out = args
        .windows(2)
        .find(|w| w[0] == "-o")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(|| PathBuf::from("mask.safetensors"));
    let f: Vec<&str> = spec.split(':').collect();

    let r = (|| -> Result<(), String> {
        let (name, _rank, shape, data) = match f.as_slice() {
            // 1-D vector selector — the common LoRA bottleneck mask
            [n, r, d] => {
                let (r, d) = parse_pair(r, d)?;
                check_rank(r, d)?;
                let data: Vec<f32> = (0..d).map(|i| (i < r) as u8 as f32).collect();
                (*n, r, vec![d], data)
            }
            // 2-D column (or row) selector
            [n, r, rows, cols] => {
                let (rows, cols) = parse_pair(rows, cols)?;
                let rank: usize = r
                    .parse()
                    .map_err(|_| "rank must be an integer".to_string())?;
                check_rank(rank, if row_mask { rows } else { cols })?;
                let mut data = vec![0f32; rows * cols];
                for i in 0..rows {
                    for j in 0..cols {
                        let live = if row_mask { i < rank } else { j < rank };
                        if live {
                            data[i * cols + j] = 1.0;
                        }
                    }
                }
                (*n, rank, vec![rows, cols], data)
            }
            _ => {
                usage();
                return Err("bad mask spec".into());
            }
        };
        std::fs::write(&out, &safetensors_f32(name, &shape, &data))
            .map_err(|e| format!("{}: {e}", out.display()))?;
        let live = data.iter().filter(|v| **v != 0.0).count();
        println!(
            "✓ {spec} → {} — {} mask {shape:?}, {live} live of {}",
            out.display(),
            if shape.len() == 1 {
                "selector"
            } else if row_mask {
                "row"
            } else {
                "column"
            },
            data.len()
        );
        Ok(())
    })();
    r.map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}

fn parse_pair(a: &str, b: &str) -> Result<(usize, usize), String> {
    Ok((
        a.parse().map_err(|_| format!("{a}: expected integer"))?,
        b.parse().map_err(|_| format!("{b}: expected integer"))?,
    ))
}

fn check_rank(rank: usize, extent: usize) -> Result<(), String> {
    if rank == 0 || rank >= extent {
        return Err(format!("rank must be in 1..{extent}"));
    }
    Ok(())
}
