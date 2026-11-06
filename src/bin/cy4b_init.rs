//! `cy4b_init` — stream a CY4B init checkpoint straight from the
//! deterministic param fallback, no VM load, no Metal compile.
//!
//! ```text
//! cy4b_init <own4b.cy> <out.safetensors>
//! ```
//!
//! `param_fallback(name, numel)` is name-seeded — the same bytes the
//! CPU evaluator, emitted blobs, and GPU runtime buffers all produce
//! (verified bit-exact against a real `--export` snapshot: payload is
//! flat declared-dims row-major, F16). own4b's 36-layer forward MSL is
//! ~13MB and takes ~15min+ of Metal compile; an init checkpoint needs
//! none of that — params are pure function of (name, shape).
//!
//! Output layout matches `params_snapshot`: declared dims, F16,
//! tensors in declaration order — directly consumable by `own4b_hf`.

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [src, out] = args.as_slice() else {
        eprintln!("usage: cy4b_init <own4b.cy> <out.safetensors>");
        std::process::exit(2);
    };
    let text = std::fs::read_to_string(src).unwrap_or_else(|e| {
        eprintln!("cy4b_init: {src}: {e}");
        std::process::exit(1);
    });

    // param decls: `param NAME: trainable|frozen f16[d1, d2, ...]`
    let mut decls: Vec<(String, Vec<usize>)> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("param ") else {
            continue;
        };
        let Some((name, ty)) = rest.split_once(':') else {
            continue;
        };
        let ty = ty.trim();
        if !ty.starts_with("trainable f16[") && !ty.starts_with("frozen f16[") {
            eprintln!("cy4b_init: skipping non-f16 param {name}: {ty}");
            continue;
        }
        let dims = ty
            .trim_start_matches("trainable f16[")
            .trim_start_matches("frozen f16[")
            .trim_end_matches(']')
            .split(',')
            .filter_map(|d| d.trim().parse::<usize>().ok())
            .collect::<Vec<_>>();
        if dims.is_empty() {
            eprintln!("cy4b_init: bad dims for {name}: {ty}");
            std::process::exit(1);
        }
        decls.push((name.trim().to_string(), dims));
    }
    if decls.is_empty() {
        eprintln!("cy4b_init: no param decls in {src}");
        std::process::exit(1);
    }
    eprintln!("cy4b_init: {} param decls", decls.len());

    let mut w = File::create(out).unwrap_or_else(|e| {
        eprintln!("cy4b_init: {out}: {e}");
        std::process::exit(1);
    });
    // header region — ~6194 tensors × ~150B JSON ≈ 1MB; 4MB is safe.
    const HREG: u64 = 1 << 22;
    w.seek(SeekFrom::Start(8 + HREG)).unwrap();

    let mut meta: Vec<(String, Vec<usize>, u64, u64)> = Vec::new();
    let mut cursor = 8 + HREG;
    let mut total = 0u64;
    for (name, dims) in &decls {
        let numel: usize = dims.iter().product();
        // param_fallback already f16-rounds — F16 store is exact.
        let vals = coyote::weights::param_fallback(name, numel);
        debug_assert_eq!(vals.len(), numel);
        let nbytes = (numel * 2) as u64;
        meta.push((name.clone(), dims.clone(), cursor - (8 + HREG), nbytes));
        let mut buf = Vec::with_capacity(numel * 2);
        for v in &vals {
            buf.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes());
        }
        w.write_all(&buf).unwrap();
        cursor += nbytes;
        total += nbytes;
        if meta.len() % 500 == 0 {
            eprintln!("  {}/{} params ({:.1} GiB)", meta.len(), decls.len(), total as f64 / (1u64 << 30) as f64);
        }
    }

    let mut header = String::from("{");
    for (i, (name, dims, off, nb)) in meta.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let shp = dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"F16\",\"shape\":[{shp}],\"data_offsets\":[{off}, {}]}}",
            off + nb
        ));
    }
    header.push('}');
    if header.len() as u64 > HREG {
        eprintln!("cy4b_init: header {} > reserved {HREG}", header.len());
        std::process::exit(1);
    }
    while (header.len() as u64) < HREG {
        header.push(' ');
    }
    w.seek(SeekFrom::Start(0)).unwrap();
    w.write_all(&(HREG as u64).to_le_bytes()).unwrap();
    w.write_all(header.as_bytes()).unwrap();
    w.flush().unwrap();
    println!(
        "cy4b_init: {} tensors, {:.2} GiB → {}",
        meta.len(),
        total as f64 / (1u64 << 30) as f64,
        out
    );
}
