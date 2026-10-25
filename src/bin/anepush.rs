//! anepush — resident-ANE decode smoke for a shardized organism.
//! `coyote-apple-anepush <shard_dir> <tok,tok,...> [max_new]` prints the
//! generated token ids. No tokenizer needed — ids in, ids out.

use mil_infer::ane_decode::Decoder;
use mil_infer::e5rt::DeviceMask;
use std::path::Path;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: anepush <dir> <ids> [n]");
    let ids: Vec<u32> = std::env::args()
        .nth(2)
        .expect("token ids")
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let n: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let t0 = std::time::Instant::now();
    let mut dec = Decoder::open_dir(Path::new(&dir), DeviceMask::Ane)
        .map_err(|e| format!("{e}"))
        .unwrap();
    eprintln!("open: {:?}", t0.elapsed());
    let t0 = std::time::Instant::now();
    let out = dec.generate(&ids, n).map_err(|e| format!("{e}")).unwrap();
    let el = t0.elapsed();
    eprintln!(
        "generate: {:?} ({:.1} ms/tok, {:.1} tok/s)",
        el,
        el.as_secs_f64() * 1000.0 / (out.len() + 1).max(1) as f64,
        (out.len() + 1) as f64 / el.as_secs_f64()
    );
    println!("{}", out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
}
