//! diffprobe — run one `.cy` decode step through BOTH the CPU evaluator
//! and the ANE resident Pipeline, feeding identical inputs, and dump
//! per-layer + logits divergence. `coyote-apple-diffprobe file.cy dir`

use std::collections::HashMap;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use coyote::eval::Evaluator;
use coyote::weights;
use mil_infer::ane_decode::Manifest;
use mil_infer::ane_pipeline::Pipeline;
use mil_infer::e5rt::DeviceMask;

const MASKED: f32 = -10_000.0;

fn f16v(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect()
}

fn stats(name: &str, a: &[f32], b: &[f32]) {
    if a.len() != b.len() {
        println!("  {name}: len mismatch {} vs {}", a.len(), b.len());
        return;
    }
    let mut sum = 0f64;
    let mut worst: Vec<(usize, f32)> = Vec::new();
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        sum += d as f64;
        worst.push((i, d));
    }
    worst.sort_by(|x, y| y.1.total_cmp(&x.1));
    let max_d = worst.first().map(|w| w.1).unwrap_or(0.0);
    let max_i = worst.first().map(|w| w.0).unwrap_or(0);
    println!(
        "  {name}: max|d|={max_d:.6} @{max_i}  mean|d|={:.6}  n={}",
        sum / a.len() as f64,
        a.len()
    );
    println!(
        "    cpu[..6]={:?}",
        &a[..6.min(a.len())]
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect::<Vec<_>>()
    );
    println!(
        "    ane[..6]={:?}",
        &b[..6.min(b.len())]
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect::<Vec<_>>()
    );
    for (i, d) in worst.iter().take(4) {
        println!("    worst @{i}: cpu={:.5} ane={:.5} d={d:.5}", a[*i], b[*i]);
    }
}

fn top5(v: &[f32]) -> Vec<(usize, f32)> {
    let mut ix: Vec<usize> = (0..v.len()).collect();
    ix.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
    ix.iter().take(5).map(|&i| (i, v[i])).collect()
}

fn main() -> Result<(), String> {
    let file = PathBuf::from(std::env::args().nth(1).expect("file.cy"));
    let dir = PathBuf::from(std::env::args().nth(2).expect("shard dir"));
    let steps: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    // ---------- CPU side: parse, emit plans, make evaluators ----------
    let (src, ast) = coyote::parser::parse_file(&file).map_err(|d| d.to_string())?;
    let checked = coyote::typeck::check_all(&ast).map_err(|d| d.to_string())?;
    let manifest = Manifest::load(&dir).map_err(|e| e.to_string())?;
    let n_layers = manifest.layers.len();
    let mut order: Vec<String> = (0..n_layers).map(|i| format!("l{i}")).collect();
    order.push("head".into());

    let tmp = std::env::temp_dir().join("diffprobe_emit");
    let mut evals = Vec::new();
    let mut state_sizes: Vec<HashMap<String, usize>> = Vec::new();
    for name in &order {
        let tp = &checked
            .ane
            .iter()
            .find(|(n, _)| n == name)
            .ok_or_else(|| format!("no fn {name}"))?
            .1;
        let out = coyote::emit_program(&src, tp, &tmp.join(name), false)?;
        let mut ev = Evaluator::new(tp);
        let mut params = HashMap::new();
        for (pname, _port, ty, psrc) in &out.trainable {
            let decl: Vec<usize> = ty.dims.iter().map(|&d| d as usize).collect();
            let expect = if ty.dims.len() == 2 {
                vec![decl[1], decl[0]]
            } else {
                decl
            };
            let mut data = weights::resolve(
                pname,
                Some(&expect),
                psrc.as_deref(),
                weights::param_fallback,
            )?;
            for v in data.iter_mut() {
                *v = half::f16::from_f32(*v).to_f32();
            }
            params.insert(pname.clone(), data);
        }
        ev.set_params(params);
        let mut sizes = HashMap::new();
        for (_key, port, ty) in &out.states {
            sizes.insert(
                port.clone(),
                ty.dims
                    .iter()
                    .map(|&d| d as usize)
                    .product::<usize>()
                    .max(1),
            );
        }
        println!(
            "cpu: {name} ins={:?} states={:?}",
            out.inputs
                .iter()
                .map(|(p, _, _)| p.clone())
                .collect::<Vec<_>>(),
            sizes.keys().collect::<Vec<_>>()
        );
        evals.push((ev, out));
        state_sizes.push(sizes);
    }

    // ---------- ANE side: pipeline over the shard dir ----------
    // memory-tight hosts: only the first head shard is resident — the
    // logits comparison runs on its vocab window.
    let heads: Vec<PathBuf> = manifest.heads.iter().take(1).cloned().collect();
    let pipe =
        Pipeline::open(&manifest.layers, &heads, DeviceMask::Ane).map_err(|e| e.to_string())?;
    let emb = std::fs::File::open(&manifest.embedding).map_err(|e| e.to_string())?;
    let half = manifest.rope_dim / 2;
    let inv_freq: Vec<f64> = (0..half)
        .map(|i| manifest.rope_freq_base.powf(-(i as f64) / half as f64))
        .collect();
    let seq = manifest.seq_len;
    let attn_rows = pipe.shared_elems("attn_mask").unwrap_or(seq) / seq;
    let mut attn = vec![half::f16::from_f32(MASKED).to_bits(); attn_rows * seq];
    let mut kv_mask = vec![0u16; seq];
    pipe.reset_state().map_err(|e| e.to_string())?;
    pipe.set_shared("attn_mask", &attn)
        .map_err(|e| e.to_string())?;
    pipe.set_shared("kv_write_mask", &kv_mask)
        .map_err(|e| e.to_string())?;

    let mut cpu_states: Vec<HashMap<String, Vec<f32>>> = state_sizes
        .iter()
        .map(|m| m.iter().map(|(k, &n)| (k.clone(), vec![0f32; n])).collect())
        .collect();

    // embedding row width is independent of the probed fn's output shape
    let emb_row = (std::fs::metadata(&manifest.embedding)
        .map_err(|e| e.to_string())?
        .len()
        / manifest.vocab_size as u64) as usize;

    let mut tok = 151643u32;
    for step in 0..steps {
        // ---------- shared inputs ----------
        let mut row = vec![0u8; emb_row];
        emb.read_exact_at(&mut row, tok as u64 * emb_row as u64)
            .map_err(|e| e.to_string())?;
        let x_f32: Vec<f32> = row
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let pos_f: Vec<f32> = (0..seq)
            .map(|i| if i == step { 1.0 } else { 0.0 })
            .collect();
        let mask_f: Vec<f32> = (0..seq)
            .map(|j| if j <= step { 0.0 } else { MASKED })
            .collect();
        let cos_f: Vec<f32> = inv_freq
            .iter()
            .map(|f| (step as f64 * f).cos() as f32)
            .collect();
        let sin_f: Vec<f32> = inv_freq
            .iter()
            .map(|f| (step as f64 * f).sin() as f32)
            .collect();

        // ---------- CPU chain ----------
        let mut h = x_f32.clone();
        let mut hids = Vec::new();
        for (li, (ev, out)) in evals.iter_mut().enumerate() {
            let mut ins: HashMap<String, Vec<f32>> = HashMap::new();
            for (port, _ty, _) in &out.inputs {
                let v = match port.as_str() {
                    "x" | "h" => h.clone(),
                    "pos" | "kv_write_mask" => pos_f.clone(),
                    "mask" | "attn_mask" => mask_f.clone(),
                    "cos" | "rope_cos" => cos_f.clone(),
                    "sin" | "rope_sin" => sin_f.clone(),
                    other => return Err(format!("unbound input {other} on {li}")),
                };
                ins.insert(port.clone(), v);
            }
            ev.set_states(cpu_states[li].clone());
            let res = ev.run(&ins).map_err(|d| format!("cpu {li}: {d}"))?;
            for (port, _) in cpu_states[li].clone() {
                if let Some((_, vals)) = ev.param_updates().iter().find(|(n, _)| n == &port) {
                    cpu_states[li].insert(port, vals.clone());
                }
            }
            if li < n_layers {
                hids.push(res.clone());
            }
            h = res;
        }
        let cpu_logits = h.clone();

        // ---------- ANE step ----------
        let (slot, _n) = pipe.embedding_slot().map_err(|e| e.to_string())?;
        // the slot buffer is sized by the op's output; `x` reads the
        // first emb_row bytes — write the row and leave the rest as-is.
        unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), slot as *mut u8, row.len()) };
        let cos16: Vec<u16> = cos_f
            .iter()
            .map(|v| half::f16::from_f32(*v).to_bits())
            .collect();
        let sin16: Vec<u16> = sin_f
            .iter()
            .map(|v| half::f16::from_f32(*v).to_bits())
            .collect();
        pipe.set_shared("rope_cos", &cos16)
            .map_err(|e| e.to_string())?;
        pipe.set_shared("rope_sin", &sin16)
            .map_err(|e| e.to_string())?;
        let z = 0u16;
        for r in 0..attn_rows {
            attn[r * seq + step] = z;
        }
        if step > 0 {
            kv_mask[step - 1] = 0;
        }
        kv_mask[step] = half::f16::from_f32(1.0).to_bits();
        pipe.set_shared("attn_mask", &attn)
            .map_err(|e| e.to_string())?;
        pipe.set_shared("kv_write_mask", &kv_mask)
            .map_err(|e| e.to_string())?;
        pipe.step().map_err(|e| e.to_string())?;

        println!("== step {step} tok {tok}");
        // `last_hidden` reads the two ping-pong buffers post-step —
        // only the LAST TWO layers' captures are valid; earlier slots
        // were overwritten by later layers in the same step.
        for li in 0..n_layers {
            if li + 2 >= n_layers {
                if let Some(b) = pipe.last_hidden(li + 1) {
                    stats(&format!("layer_{li:02} hidden"), &hids[li], &f16v(&b));
                }
            }
            for st in ["kv_k0lo", "kv_v0lo", "kv_k0hi"] {
                if let (Some(abits), Some(cst)) = (pipe.state_f16(li, st), cpu_states[li].get(st)) {
                    stats(&format!("layer_{li:02} {st}"), cst, &f16v(&abits));
                }
            }
        }
        let ane_logits = f16v(&pipe.logits());
        stats(
            "logits[0..37984]",
            &cpu_logits[..ane_logits.len().min(cpu_logits.len())],
            &ane_logits,
        );
        println!(
            "    cpu top5 {:?}",
            top5(&cpu_logits[..ane_logits.len().min(cpu_logits.len())])
        );
        println!("    ane top5 {:?}", top5(&ane_logits));
        tok = cpu_logits[..ane_logits.len().min(cpu_logits.len())]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    Ok(())
}
