//! `own4b_hf` — repack a coyote-apple own-4B param export into the
//! HuggingFace safetensors layout `milc convert` consumes, plus the
//! `config.json` for the CY4B geometry.
//!
//! ```text
//! own4b_hf <params.safetensors> <corpus.safetensors> <out_dir>
//! ```
//!
//! Streaming in both directions — the param export is ~7.5GB and the
//! HF checkpoint ~7.6GB; nothing gets slurped. own-4B stores matmul
//! weights `[in, out]` row-major with per-head `lo`/`hi` RoPE halves
//! and two summed FFN blocks; HF wants `[out, in]` with concatenated
//! heads/blocks. Every transform below is a transpose+concat that
//! preserves the exact function own4b.cy computes, with TWO known
//! deltas:
//!
//! 1. `qk-norm` — own-4B rmsnorms each 64-dim RoPE half separately
//!    (`qnl`/`qnh`), HF qwen3 applies one norm over the full 128-dim
//!    head. We emit `cat(qnl, qnh)` — same per-dim gains, joint
//!    denominator. Data-dependent, no exact weight-space fix.
//! 2. attention scale — own4b multiplies the 128-dim q·k dot by
//!    `1/sqrt(64) = 0.125` (gen_brain `hd=64`), HF qwen3 uses
//!    `1/sqrt(128) ≈ 0.0884`. Both q and k pass through scale-invariant
//!    rmsnorm so projection weights can't carry the fix — but the
//!    POST-norm gain vector can: we emit `q_norm.weight =
//!    cat(qnl, qnh) * sqrt(2)`, giving HF scores exactly `0.125 * dot`.
//!    Verified: rmsnorm is scale-invariant, RoPE is linear, and
//!    `rope_cos(pos,64)` computes `theta^(-i/64)` = HF head_dim=128
//!    inv_freq exactly (theta=1e6 both sides).

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const D: usize = 2560;
const L: usize = 36;
const QH: usize = 32; // query heads
const KVH: usize = 8; // kv heads
const HD: usize = 64; // per-half head dim → head_dim 128
const V: usize = 3103;
const FB: usize = 2; // ffn blocks
const FD: usize = 5120; // per-block intermediate → 10240 total
const FHD: usize = 128; // HF head_dim

struct TensorLoc {
    off: u64,
    len: u64,
    shape: Vec<usize>,
    dtype: String,
}

/// streaming safetensors reader — header parsed once, tensor payloads
/// fetched by seek+read. Never holds the file in memory.
struct St {
    file: File,
    base: u64,
    index: HashMap<String, TensorLoc>,
}

fn json_str(obj: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":\"");
    let i = obj.find(&pat)? + pat.len();
    let j = obj[i..].find('"')? + i;
    Some(obj[i..j].to_string())
}

fn json_nums(obj: &str, key: &str) -> Vec<i64> {
    let pat = format!("\"{key}\":[");
    let Some(i) = obj.find(&pat).map(|p| p + pat.len()) else {
        return Vec::new();
    };
    let Some(j) = obj[i..].find(']') else {
        return Vec::new();
    };
    obj[i..i + j]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

impl St {
    fn load(path: &str) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mut lb = [0u8; 8];
        file.read_exact(&mut lb).map_err(|e| e.to_string())?;
        let hlen = u64::from_le_bytes(lb) as usize;
        let mut head = vec![0u8; hlen];
        file.read_exact(&mut head).map_err(|e| e.to_string())?;
        let head = String::from_utf8(head).map_err(|e| e.to_string())?;
        let base = 8 + hlen as u64;
        let mut index = HashMap::new();
        let bytes = head.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'"' {
                let Some(end) = head[i + 1..].find('"').map(|e| i + 1 + e) else {
                    break;
                };
                let name = head[i + 1..end].to_string();
                i = end + 1;
                while i < bytes.len() && (bytes[i] == b':' || bytes[i] == b' ') {
                    i += 1;
                }
                if i >= bytes.len() || bytes[i] != b'{' {
                    continue;
                }
                let start = i;
                let mut depth = 0;
                while i < bytes.len() {
                    match bytes[i] {
                        b'{' => depth += 1,
                        b'}' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
                let obj = &head[start..=i];
                i += 1;
                if name == "__metadata__" {
                    continue;
                }
                let dt = json_str(obj, "dtype").unwrap_or_default();
                let shape: Vec<usize> = json_nums(obj, "shape")
                    .iter()
                    .map(|&v| v as usize)
                    .collect();
                let offs = json_nums(obj, "data_offsets");
                if offs.len() == 2 {
                    index.insert(
                        name,
                        TensorLoc {
                            off: base + offs[0] as u64,
                            len: (offs[1] - offs[0]) as u64,
                            shape,
                            dtype: dt,
                        },
                    );
                }
            } else {
                i += 1;
            }
        }
        Ok(St { file, base, index })
    }

    fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// whole tensor → f32, decoded from F16 or F32 payload.
    fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let loc = self
            .index
            .get(name)
            .ok_or_else(|| format!("safetensors: missing {name}"))?;
        let mut buf = vec![0u8; loc.len as usize];
        let mut f = self.file.try_clone().map_err(|e| e.to_string())?;
        f.seek(SeekFrom::Start(loc.off)).map_err(|e| e.to_string())?;
        f.read_exact(&mut buf).map_err(|e| e.to_string())?;
        let n: usize = loc.shape.iter().product();
        match loc.dtype.as_str() {
            "F16" => Ok(buf
                .chunks_exact(2)
                .take(n)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()),
            "F32" => Ok(buf
                .chunks_exact(4)
                .take(n)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            other => Err(format!("{name}: unsupported dtype {other}")),
        }
    }
}

/// own-4B `[in, out]` row-major param → HF `[out, in]` transpose.
fn transpose(src: &[f32], inn: usize, out: usize) -> Vec<f32> {
    let mut dst = vec![0f32; src.len()];
    for i in 0..inn {
        for j in 0..out {
            dst[j * inn + i] = src[i * out + j];
        }
    }
    dst
}

/// own-4B `[in, out]` → placed transposed into `dst` columns
/// `[col0, col0+out)` — for o_proj/down_proj column-block assembly.
fn transpose_into(src: &[f32], inn: usize, out: usize, dst: &mut [f32], col0: usize, rown: usize) {
    for i in 0..inn {
        for j in 0..out {
            dst[j * rown + col0 + i] = src[i * out + j];
        }
    }
}

/// assemble one layer's HF tensors. Returns (name, shape, f32 vals).
fn layer_tensors(st: &St, l: usize) -> Result<Vec<(String, Vec<usize>, Vec<f32>)>, String> {
    let p = |s: &str| format!("l{l}_{s}");
    let mut out = Vec::new();
    let pre = format!("model.layers.{l}");

    // norms — 1-D copies
    for (dst, src) in [
        ("input_layernorm.weight", "ln1"),
        ("post_attention_layernorm.weight", "ln2"),
    ] {
        let v = st.tensor_f32(&p(src))?;
        out.push((format!("{pre}.{dst}"), vec![D], v));
    }
    // qk norms: cat the two 64-dim half-norms → [128]. q_norm gains an
    // extra sqrt(2) — own4b scores the 128-dim dot at 1/sqrt(64)=0.125
    // while HF applies 1/sqrt(128); the post-norm gain is the only
    // weight-space lever that survives scale-invariant rmsnorm.
    for (dst, a, b, qscale) in [
        ("self_attn.q_norm.weight", "qnl", "qnh", std::f32::consts::SQRT_2),
        ("self_attn.k_norm.weight", "knl", "knh", 1.0f32),
    ] {
        let mut v = st.tensor_f32(&p(a))?;
        v.extend(st.tensor_f32(&p(b))?);
        if qscale != 1.0 {
            for x in &mut v {
                *x *= qscale;
            }
        }
        out.push((format!("{pre}.{dst}"), vec![FHD], v));
    }
    // q/k/v projections — per-head lo/hi halves concatenated.
    // own4b param [d,64] per half → HF [out,in] rows h*128+half*64+i.
    for (proj, heads, pr) in [
        ("q_proj", QH, "q"),
        ("k_proj", KVH, "k"),
        ("v_proj", KVH, "v"),
    ] {
        let rows = heads * FHD;
        let mut w = vec![0f32; rows * D];
        for h in 0..heads {
            for (hi, hf) in ["lo", "hi"].iter().enumerate() {
                let src = st.tensor_f32(&format!("{}{}{}{}", p(pr), h, hf, ""))?;
                // src is [D,64] → transpose → [64,D] into rows
                let base = (h * FHD + hi * HD) * D;
                for i in 0..HD {
                    for j in 0..D {
                        w[base + i * D + j] = src[j * HD + i];
                    }
                }
            }
        }
        out.push((
            format!("{pre}.self_attn.{proj}.weight"),
            vec![rows, D],
            w,
        ));
    }
    // o_proj [D, QH*FHD] — column blocks per (head, half)
    {
        let mut w = vec![0f32; D * QH * FHD];
        for h in 0..QH {
            for (hi, hf) in ["lo", "hi"].iter().enumerate() {
                let src = st.tensor_f32(&format!("{}o{}{}", p(""), h, hf))?;
                // src [64,D] = [in,out] → dst col block, transposed
                transpose_into(&src, HD, D, &mut w, h * FHD + hi * HD, QH * FHD);
            }
        }
        out.push((
            format!("{pre}.self_attn.o_proj.weight"),
            vec![D, QH * FHD],
            w,
        ));
    }
    // FFN — two summed blocks ≡ one wide SwiGLU: gate/up rows are
    // per-block transposes; down columns are per-block transposes.
    {
        let mut gate = vec![0f32; FB * FD * D];
        let mut up = vec![0f32; FB * FD * D];
        let mut down = vec![0f32; D * FB * FD];
        for b in 0..FB {
            let wg = st.tensor_f32(&format!("{}wg{}", p(""), b))?;
            let wu = st.tensor_f32(&format!("{}wu{}", p(""), b))?;
            let wd = st.tensor_f32(&format!("{}wd{}", p(""), b))?;
            // wg/wu [D,FD] → transpose → [FD,D] rows b*FD..
            for i in 0..FD {
                for j in 0..D {
                    gate[(b * FD + i) * D + j] = wg[j * FD + i];
                    up[(b * FD + i) * D + j] = wu[j * FD + i];
                }
            }
            // wd [FD,D] = [in,out] → columns b*FD.. of down [D, FB*FD]
            transpose_into(&wd, FD, D, &mut down, b * FD, FB * FD);
        }
        out.push((
            format!("{pre}.mlp.gate_proj.weight"),
            vec![FB * FD, D],
            gate,
        ));
        out.push((
            format!("{pre}.mlp.up_proj.weight"),
            vec![FB * FD, D],
            up,
        ));
        out.push((
            format!("{pre}.mlp.down_proj.weight"),
            vec![D, FB * FD],
            down,
        ));
    }
    Ok(out)
}

fn write_f16(w: &mut File, vals: &[f32]) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(vals.len() * 2);
    for v in vals {
        buf.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes());
    }
    w.write_all(&buf)
}

fn write_f32(w: &mut File, vals: &[f32]) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    w.write_all(&buf)
}

fn run(args: &[String]) -> Result<(), String> {
    let [params, corpus, out_dir] = args else {
        eprintln!("usage: own4b_hf <params.safetensors> <corpus.safetensors> <out_dir>");
        return Err("usage".into());
    };
    let st = St::load(params)?;
    let cst = St::load(corpus)?;
    let out_dir = Path::new(out_dir);
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let out_path = out_dir.join("model.safetensors");
    let mut w = File::create(&out_path).map_err(|e| e.to_string())?;

    // header region reserved up front — JSON + space padding must fit.
    const HREG: u64 = 1 << 16;
    w.seek(SeekFrom::Start(8 + HREG)).map_err(|e| e.to_string())?;

    // (name, shape, dtype, data_offset, nbytes) recorded as we stream.
    let mut meta: Vec<(String, Vec<usize>, String, u64, u64)> = Vec::new();
    let mut cursor = 8 + HREG;
    let mut emit = |w: &mut File,
                    meta: &mut Vec<(String, Vec<usize>, String, u64, u64)>,
                    cursor: &mut u64,
                    name: String,
                    shape: Vec<usize>,
                    dtype: &str,
                    vals: &[f32]|
     -> Result<(), String> {
        let nbytes = vals.len() * if dtype == "F16" { 2 } else { 4 };
        meta.push((
            name,
            shape,
            dtype.to_string(),
            *cursor - (8 + HREG),
            nbytes as u64,
        ));
        if dtype == "F16" {
            write_f16(w, vals).map_err(|e| e.to_string())?;
        } else {
            write_f32(w, vals).map_err(|e| e.to_string())?;
        }
        *cursor += nbytes as u64;
        Ok(())
    };

    // embed_tokens — F32, corpus cemb is already [V,D] HF layout.
    let emb = cst.tensor_f32("cemb")?;
    if emb.len() != V * D {
        return Err(format!("cemb: {} elems, want {}", emb.len(), V * D));
    }
    emit(
        &mut w,
        &mut meta,
        &mut cursor,
        "model.embed_tokens.weight".into(),
        vec![V, D],
        "F32",
        &emb,
    )?;

    for l in 0..L {
        for (name, shape, vals) in layer_tensors(&st, l)? {
            emit(&mut w, &mut meta, &mut cursor, name, shape, "F16", &vals)?;
        }
        if l % 6 == 5 {
            eprintln!("  layer {l} done");
        }
    }

    // final norm + lm_head
    let lnf = st.tensor_f32("lnf")?;
    emit(
        &mut w,
        &mut meta,
        &mut cursor,
        "model.norm.weight".into(),
        vec![D],
        "F16",
        &lnf,
    )?;
    let lmh = st.tensor_f32("lmh")?; // [D,V] own4b → transpose → [V,D]
    let lmh_t = transpose(&lmh, D, V);
    emit(
        &mut w,
        &mut meta,
        &mut cursor,
        "lm_head.weight".into(),
        vec![V, D],
        "F16",
        &lmh_t,
    )?;

    // header — JSON padded with spaces to exactly HREG.
    let mut header = String::from("{");
    for (i, (name, shape, dtype, off, nb)) in meta.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let shp = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shp}],\"data_offsets\":[{off}, {}]}}",
            off + nb
        ));
    }
    header.push('}');
    if header.len() as u64 + 1 > HREG {
        return Err(format!("header {} > reserved {HREG}", header.len()));
    }
    while (header.len() as u64) < HREG {
        header.push(' ');
    }
    w.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    w.write_all(&(HREG as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;
    w.write_all(header.as_bytes()).map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())?;

    // config.json — CY4B geometry in the qwen3 layout milc consumes.
    let cfg = format!(
        r#"{{
  "model_type": "qwen3",
  "architectures": ["Cy4bForCausalLM"],
  "hidden_size": {D},
  "num_hidden_layers": {L},
  "num_attention_heads": {QH},
  "num_key_value_heads": {KVH},
  "head_dim": {FHD},
  "intermediate_size": {},
  "vocab_size": {V},
  "max_position_embeddings": 512,
  "rms_norm_eps": 1e-5,
  "rope_theta": 1000000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16"
}}"#,
        FB * FD
    );
    std::fs::write(out_dir.join("config.json"), cfg).map_err(|e| e.to_string())?;
    println!(
        "cy4b hf export → {} ({} tensors, {:.1} GiB)",
        out_path.display(),
        meta.len(),
        cursor as f64 / (1u64 << 30) as f64
    );
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("own4b_hf: {e}");
        std::process::exit(1);
    }
}
