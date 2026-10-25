//! shardize — emit `fn` (ANE) programs as resident-pipeline shards.
//!
//! `coyote-apple shardize <organism.cy> -o <dir>` writes a shard
//! directory the `mil_infer::ane_pipeline`/`ane_decode` path consumes
//! directly: every `fn` becomes `layer_NN.mlmodelc`/`head_NN.mlmodelc`
//! whose `model.mil` is coyote's emission canonicalized to strict MIL —
//! packed HWX-convention shapes redeclared as batch-(T,D) forms, state
//! in/out pairs rewritten to `state<>` ports driven by `read_state`/
//! `write_state`, `rms_norm` expanded to reduce_mean/rsqrt, live inputs
//! renamed to the pipeline's shared-port names, and weight blobs packed
//! under `weights/`. A `table emb` declaration is spilled to
//! `embedding.f16`; `conversion_manifest.json` ties it together. The
//! result decodes with one `execute_sync` per token — the resident
//! path, no per-op dispatch tax.

use coyote::ast::{Decl, Target};
use coyote::lower::MilOut;
use coyote::typeck::TProgram;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

// ---------- strict-shape canonicalization ----------
//
// coyote emits mil_infer's packed conventions: vec[D] → [1,D,1,1],
// mat[T,D] → [1,T,1,D], matvec outs → [1,1,1,N]. The e5 compiler
// wants standard MIL: last-two matmul dims, numpy broadcast. Logical
// coyote shapes → strict:
//   (D,)   → [1,1,1,D]      row
//   (T,)   → [1,1,1,T]      row — except the scatter one-hot, a column
//   (T,D)  → [1,1,T,D]
//   (N,K)  → [N,K]          param storage stays 2-D
//   scalar → fp16 / bool scalar const

#[derive(Clone, Debug)]
struct Val {
    /// logical (coyote) dims
    dims: Vec<i64>,
    /// strict emitted shape text, e.g. "[1, 1, 1, 64]"
    strict: String,
}

fn strict4(dims: &[i64]) -> String {
    match dims.len() {
        1 => format!("[1, 1, 1, {}]", dims[0]),
        2 => format!("[1, 1, {}, {}]", dims[0], dims[1]),
        _ => format!(
            "[{}]",
            dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
        ),
    }
}

fn strict_col(dims: &[i64]) -> String {
    // a (T,) one-hot in column form so matmul(Tx1, 1xD) is a clean
    // outer product with no transpose flags.
    format!("[1, 1, {}, 1]", dims[0])
}

/// unpack a coyote-emitted mil shape back to logical dims:
/// [1,A,1,B]→(A,B); [1,A,1,1]→(A,); [A,B]→(A,B); [A]→(A,).
fn unshape(shape: &str) -> Vec<i64> {
    let v: Vec<i64> = shape
        .trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    match v.as_slice() {
        [1, a, 1, b] => {
            if *b == 1 {
                vec![*a]
            } else {
                vec![*a, *b]
            }
        }
        other => other.to_vec(),
    }
}

struct Op {
    out_ty: String,     // "tensor<fp16, [1, 64, 1, 1]>" or "bool" etc
    out: String,
    name: String,       // matmul, add, const, ...
    args: Vec<(String, String)>,
    line: String,
}

fn parse_line(line: &str) -> Option<Op> {
    let t = line.trim();
    if t.is_empty() || t == "}" {
        return None;
    }
    // `write_state(...)` and other statement forms have no `name =`.
    let (decl, rhs) = match t.split_once(" = ") {
        Some((d, r)) => (d, r),
        None => {
            return Some(Op {
                out_ty: String::new(),
                out: String::new(),
                name: String::new(),
                args: Vec::new(),
                line: t.to_string(),
            })
        }
    };
    let out = decl.rsplit(' ').next()?.to_string();
    let out_ty = decl[..decl.len() - out.len()].trim().to_string();
    let op_end = rhs.find('(')?;
    let opname = rhs[..op_end].trim().to_string();
    // strip the trailing `[name = string("...")];` before taking the
    // arg list — its `)` would beat `rfind` otherwise.
    let core = rhs
        .trim_end()
        .trim_end_matches(';')
        .split("[name = string(")
        .next()
        .unwrap_or(rhs);
    let argstr = &core[op_end + 1..core.rfind(')')?];
    let mut args = Vec::new();
    for a in split_args(argstr) {
        if let Some((k, v)) = a.split_once(" = ") {
            args.push((k.trim().to_string(), v.trim().to_string()));
        } else if let Some((k, v)) = a.trim().split_once('=') {
            args.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    Some(Op {
        out_ty,
        out,
        name: opname,
        args,
        line: t.to_string(),
    })
}

/// f16 hexfloat literal like Apple's (`fp16(0x1p-3)`); decimal form
/// also parses but hex matches the house style.
fn f16_lit(bits: u16) -> String {
    let f = half::f16::from_bits(bits).to_f64();
    format!("fp16({f})")
}

fn shape_of(ty: &str) -> Option<String> {
    let l = ty.find('[')?;
    let r = ty.rfind(']')?;
    Some(ty[l..=r].to_string())
}

/// `weight.bin` v2 container — 64-byte header (`count`,`2`,zeros),
/// then 64-aligned records: a 64-byte metadata block
/// (`0xdeadbeef`, dtype code, byte len, data offset, zeros) followed
/// by the tensor bytes. `BLOBFILE offset` points at the metadata
/// block, data follows at `offset + 64`.
struct WeightBin {
    data: Vec<u8>,
    cursor: u64,
    count: u32,
}

impl WeightBin {
    fn new() -> Self {
        let mut data = Vec::with_capacity(64);
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&2u32.to_le_bytes());
        data.resize(64, 0);
        WeightBin {
            data,
            cursor: 64,
            count: 0,
        }
    }

    fn put(&mut self, dtype_code: u32, bytes: &[u8]) -> u64 {
        let pad = (64 - self.cursor % 64) % 64;
        self.data.resize(self.data.len() + pad as usize, 0);
        self.cursor += pad;
        let meta_off = self.cursor;
        let data_off = meta_off + 64;
        let mut meta = Vec::with_capacity(64);
        meta.extend_from_slice(&0xDEADBEEFu32.to_le_bytes());
        meta.extend_from_slice(&dtype_code.to_le_bytes());
        meta.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        meta.extend_from_slice(&data_off.to_le_bytes());
        meta.extend_from_slice(&0u64.to_le_bytes());
        meta.resize(64, 0);
        self.data.extend_from_slice(&meta);
        self.data.extend_from_slice(bytes);
        self.cursor = self.data.len() as u64;
        self.count += 1;
        meta_off
    }

    fn finish(mut self) -> Vec<u8> {
        self.data[0..4].copy_from_slice(&self.count.to_le_bytes());
        self.data
    }
}

/// MIL element type → `MILBlob::BlobDataType` for the record header.
fn blob_dtype(ty: &str) -> u32 {
    if ty.starts_with("tensor<fp16") {
        1
    } else if ty.starts_with("tensor<fp32") {
        2
    } else if ty.starts_with("tensor<int8") {
        4
    } else if ty.starts_with("tensor<int4") {
        8
    } else if ty.starts_with("tensor<uint4") {
        11
    } else if ty.starts_with("tensor<int32") {
        14
    } else {
        1
    }
}

/// Canonicalize one emitted program to strict shard MIL. Returns the
/// MIL text plus the packed `weights/weight.bin` container — every
/// `BLOBFILE` ref is rewritten to point at its deadbeef record inside
/// the shared blob file.
fn shard_mil(
    out: &MilOut,
    mil: &str,
    blobs: &[(String, Vec<u8>)],
) -> Result<(String, Vec<u8>), String> {
    // ---- locate sig / body / ret ----
    let fpos = mil.find("func main<ios").ok_or("emit: no func main")?;
    let apos = mil[fpos..].find('(').map(|i| fpos + i).ok_or("emit: no args")?;
    let mut depth = 0i32;
    let mut aend = 0usize;
    for (i, ch) in mil[apos..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    aend = apos + i;
                    break;
                }
            }
            _ => {}
        }
    }
    let args = &mil[apos + 1..aend];
    let body_beg = mil[aend..].find('{').map(|i| aend + i).ok_or("emit: no body")?;
    let ret_pos = mil.rfind("-> (").ok_or("emit: no ret")?;
    let body_raw = &mil[body_beg + 1..ret_pos];
    let close = body_raw.rfind('}').ok_or("emit: no body close")?;
    let body = &body_raw[..close];

    // ---- logical dims per sig arg (from the emit plan) ----
    let mut arg_logical: HashMap<String, Vec<i64>> = HashMap::new();
    for (port, ty, _) in &out.inputs {
        arg_logical.insert(port.clone(), ty.dims.clone());
    }
    for (_, port, ty) in &out.states {
        arg_logical.insert(port.clone(), ty.dims.clone());
    }

    // state port names for the transform
    let state_ports: std::collections::HashSet<String> =
        out.states.iter().map(|(_, p, _)| p.clone()).collect();

    // ---- role scan: which (T,) args are used as scatter columns ----
    // kv_write_mask / pos → column form; every other 1-D arg → row.
    let col_args: std::collections::HashSet<String> =
        ["pos", "kv_write_mask"].iter().map(|s| s.to_string()).collect();

    // operand renames: live inputs take pipeline port names, state
    // ports are read through aliases — `__s_` for the pre-update read,
    // `__u_` for the re-read after write_state (attention must see
    // the row it just wrote; the e5 frontend also rejects a
    // write_state whose port is never read again).
    let mut ren: HashMap<String, String> = {
        let mut m = HashMap::new();
        for port in arg_logical.keys() {
            if state_ports.contains(port) {
                m.insert(port.clone(), format!("__s_{port}"));
            } else if let Some(to) = port_rename(port) {
                m.insert(port.clone(), to.to_string());
            }
        }
        m
    };

    fn renof(ren: &HashMap<String, String>, v: &str) -> String {
        ren.get(v).cloned().unwrap_or_else(|| v.to_string())
    }
    // update-value name → (state port, strict state shape): after the
    // producer op, emit write_state + a fresh read_state alias.
    let mut upd_port: HashMap<String, (String, String)> = HashMap::new();
    for (port, upd, ty) in &out.state_outs {
        upd_port.insert(upd.clone(), (port.clone(), strict4(&ty.dims)));
    }

    // ---- parse body ops ----
    let mut vals: HashMap<String, Val> = HashMap::new();
    let mut ops: Vec<Op> = Vec::new();
    for line in body.lines() {
        if let Some(op) = parse_line(line) {
            if !op.out.is_empty() && !op.out_ty.is_empty() {
                if let Some(sh) = shape_of(&op.out_ty) {
                    let dims = unshape(&sh);
                    let strict = strict4(&dims);
                    vals.insert(op.out.clone(), Val { dims, strict });
                }
            }
            ops.push(op);
        }
    }

    // ---- collapse the fp16-ill-conditioned sigmoid poly ----
    // lower.rs lowers sigmoid (and therefore silu) on >1024-element
    // tensors to a clamped degree-13 Horner poly. Its last two
    // coefficients are ~1e-8/1e-10 — fp16 quantization rounds them to
    // -5.96e-8/0, breaking the term cancellation the minimax relies
    // on at the [-6,6] clamp edge: σ(6) comes out -8.83 instead of
    // ~0.994, so silu(9.94) = -87.8 on the engine while CPU gives
    // ~9.94. The e5 compiler takes a real `sigmoid` op at these
    // widths (verified: max|Δ| 0.017 on a 3072-wide layer), so
    // collapse each chain to it here. Chain shape in ops order:
    //   min = minimum(ORIG, 6); max = maximum(min, -6); u = max*max;
    //   ... 13-op Horner on u ... ; sh = mul(max, h); σ = add(sh, 0.5)
    // Consumers of σ (the silu `x·σ` mul, or the program output for a
    // bare sigmoid) are untouched — σ's name is re-emitted as the op.
    let mut sig_sub: HashMap<String, String> = HashMap::new(); // σ out → ORIG
    let mut poly_dead: HashSet<String> = HashSet::new();
    {
        let blob_of: HashMap<&str, &[u8]> =
            blobs.iter().map(|(n, b)| (n.as_str(), b.as_slice())).collect();
        let mut cval: HashMap<String, f64> = HashMap::new();
        for op in &ops {
            if op.name != "const" {
                continue;
            }
            // scalar `val = fp16(N)` form
            if let Some(v) = op
                .line
                .split("val = fp16(")
                .nth(1)
                .and_then(|s| s.split(')').next())
                .and_then(|s| s.parse::<f64>().ok())
            {
                cval.insert(op.out.clone(), v);
                continue;
            }
            // broadcast `val = tensor<...>(BLOBFILE(...))` form — a
            // uniform fp16 fill resolves to its first element.
            if let Some(fname) = op
                .line
                .split("BLOBFILE(path = string(\"")
                .nth(1)
                .and_then(|s| s.split('"').next())
            {
                if let Some(b) = blob_of.get(fname) {
                    if b.len() >= 2 && b.chunks_exact(2).all(|c| c == &b[0..2]) {
                        let v = half::f16::from_le_bytes([b[0], b[1]]).to_f64();
                        cval.insert(op.out.clone(), v);
                    }
                }
            }
        }
        let is_c = |n: &str, want: f64| {
            cval.get(n).map(|v| (*v - want).abs() < 1e-3).unwrap_or(false)
        };
        for i in 0..ops.len() {
            let op = &ops[i];
            if op.name != "maximum" || !is_c(&arg_of(op, "y"), -6.0) {
                continue;
            }
            let min_name = arg_of(op, "x");
            let Some(min_i) = ops
                .iter()
                .position(|o| o.name == "minimum" && o.out == min_name && is_c(&arg_of(o, "y"), 6.0))
            else {
                continue;
            };
            // scan forward for σ = add(mul(max, h), 0.5)
            for j in i + 1..ops.len() {
                let a = &ops[j];
                if a.name != "add" {
                    continue;
                }
                let (ax, ay) = (arg_of(a, "x"), arg_of(a, "y"));
                let other = if is_c(&ay, 0.5) {
                    ax
                } else if is_c(&ax, 0.5) {
                    ay
                } else {
                    continue;
                };
                if ops.iter().any(|m| {
                    m.name == "mul"
                        && m.out == other
                        && (arg_of(m, "x") == op.out || arg_of(m, "y") == op.out)
                }) {
                    for k in min_i..j {
                        poly_dead.insert(ops[k].out.clone());
                    }
                    sig_sub.insert(a.out.clone(), arg_of(&ops[min_i], "x"));
                    break;
                }
            }
        }
    }

    // ---- rewrite ----
    let blob_of: HashMap<&str, &[u8]> =
        blobs.iter().map(|(n, b)| (n.as_str(), b.as_slice())).collect();
    let mut wb = WeightBin::new();
    let mut sig_out = Vec::new();
    for arg in split_args(args) {
        let arg = arg.trim().to_string();
        if arg.is_empty() {
            continue;
        }
        let name = arg.rsplit(' ').next().unwrap_or("").to_string();
        let logical = arg_logical.get(&name).cloned().unwrap_or_default();
        if state_ports.contains(&name) {
            sig_out.push(format!("state<tensor<fp16, {}>> {name}", strict4(&logical)));
        } else {
            let rename = port_rename(&name).unwrap_or(&name).to_string();
            let strict = if col_args.contains(&name) {
                strict_col(&logical)
            } else {
                strict4(&logical)
            };
            sig_out.push(format!("tensor<fp16, {strict}> {rename}"));
        }
    }

    let mut rms_eps: HashMap<String, String> = HashMap::new(); // literal → const name
    // pre-collect eps literals so their consts declare up front.
    for op in &ops {
        if op.name == "rms_norm" {
            let eps = arg_of(op, "eps");
            if !rms_eps.contains_key(&eps) {
                rms_eps.insert(eps, format!("__eps{}", rms_eps.len()));
            }
        }
    }

    let mut out_lines: Vec<String> = Vec::new();
    out_lines.push("        bool __bt = const()[val = bool(true)];".into());
    out_lines.push("        bool __bf = const()[val = bool(false)];".into());
    out_lines.push("        int32 __axm1 = const()[val = int32(-1)];".into());
    out_lines.push("        fp16 __zro = const()[val = fp16(0x0p+0)];".into());
    out_lines.push(
        "        tensor<int32, [1]> __axv = const()[val = tensor<int32, [1]>([-1])];".into(),
    );
    for (lit, name) in &rms_eps {
        let v = lit
            .trim_start_matches("fp16(")
            .trim_end_matches(')')
            .to_string();
        out_lines.push(format!("        fp16 {name} = const()[val = fp16({v})];"));
    }
    for (_, port, ty) in &out.states {
        out_lines.push(format!(
            "        tensor<fp16, {}> __s_{port} = read_state(input = {port})[name = string(\"__s_{port}\")];",
            strict4(&ty.dims)
        ));
    }

    let logical_of = |n: &str| -> Vec<i64> {
        arg_logical
            .get(n)
            .cloned()
            .or_else(|| vals.get(n).map(|v| v.dims.clone()))
            .unwrap_or_default()
    };
    let strict_of = |n: &str| -> String {
        if let Some(d) = arg_logical.get(n) {
            if col_args.contains(n) {
                strict_col(d)
            } else {
                strict4(d)
            }
        } else {
            vals.get(n).map(|v| v.strict.clone()).unwrap_or_default()
        }
    };

    for op in &ops {
        if poly_dead.contains(&op.out) {
            continue;
        }
        if let Some(orig) = sig_sub.get(&op.out) {
            let x = renof(&ren, orig);
            let os = strict_of(&op.out);
            out_lines.push(format!(
                "        tensor<fp16, {os}> {} = sigmoid(x = {x})[name = string(\"{}\")];",
                op.out, op.out
            ));
            continue;
        }
        match op.name.as_str() {
            "const" => {
                // val forms: bool(true), tensor<...>(BLOBFILE...),
                // tensor<...>([literals]), fp16(...)
                let line = &op.line;
                if line.contains("BLOBFILE") {
                    // broadcast consts ([1,T,1,1]-style emitted shape
                    // whose blob is a uniform scalar fill) → scalar.
                    let lshape = shape_of(&op.out_ty).unwrap_or_default();
                    let fname = line
                        .split("BLOBFILE(path = string(\"")
                        .nth(1)
                        .and_then(|s| s.split('"').next())
                        .unwrap_or("");
                    let uni = blob_of.get(fname).and_then(|b| {
                        if b.len() >= 2
                            && b.chunks_exact(2).all(|c| c == &b[0..2])
                        {
                            Some(u16::from_le_bytes([b[0], b[1]]))
                        } else {
                            None
                        }
                    });
                    if let (true, Some(u)) = (is_broadcast_shape(&lshape), uni) {
                        out_lines.push(format!(
                            "        fp16 {} = const()[val = {}];",
                            op.out,
                            f16_lit(u)
                        ));
                    } else {
                        // param / weight const — keep 2-D (or strict vec),
                        // packed into the shared deadbeef weight.bin.
                        let vdims = unshape(&lshape);
                        let vsh = if vdims.len() == 1 {
                            strict4(&vdims)
                        } else {
                            lshape.clone()
                        };
                        let bytes = blob_of.get(fname).copied().unwrap_or(&[]);
                        let meta_off = wb.put(blob_dtype(&op.out_ty), bytes);
                        let shaped = line.replace(
                            &format!("tensor<fp16, {lshape}>"),
                            &format!("tensor<fp16, {vsh}>"),
                        );
                        // BLOBFILE(...) is one arg span — rebuild it outright.
                        let bpos = shaped.find("BLOBFILE").unwrap();
                        let bend = shaped[bpos..]
                            .find("))")
                            .map(|i| bpos + i + 2)
                            .unwrap_or(shaped.len());
                        let nl = format!(
                            "{}BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({meta_off})){}",
                            &shaped[..bpos],
                            &shaped[bend..]
                        );
                        out_lines.push(format!("        {nl}"));
                    }
                } else if op.out_ty == "tensor<bool, []>" {
                    // coyote's transpose flag — handled via __bt/__bf.
                } else {
                    out_lines.push(format!("        {}", op.line));
                }
            }
            "matmul" => {
                let xn = arg_of(op, "x");
                let yn = arg_of(op, "y");
                let x = renof(&ren, &xn);
                let y = renof(&ren, &yn);
                let tx = op.args.iter().any(|(k, _)| k == "transpose_x");
                let ty = op.args.iter().any(|(k, _)| k == "transpose_y");
                let xl = logical_of(&xn);
                let yl = logical_of(&yn);
                let out_v = vals.get(&op.out).ok_or("matmul out untyped")?;
                let mut args = String::new();
                // route by logical operand dims:
                //   (T,)·(D,) outer     → col@row, no flags
                //   (T,)·(T,D) s@v      → row@mat, no flags
                //   (D,)·(T,D) scores   → row@matᵀ
                //   (D,)·(N,K) matvec   → row@wᵀ
                let (fx, fy) = match (xl.len(), yl.len(), tx, ty) {
                    (1, 1, true, _) => ("__bf", "__bf"), // outer: col@row
                    (1, 2, true, _) => ("__bf", "__bf"), // s@v: row@mat
                    (1, 2, false, true) => ("__bf", "__bt"), // scores / matvec
                    _ => {
                        if ty {
                            ("__bf", "__bt")
                        } else {
                            ("__bf", "__bf")
                        }
                    }
                };
                args.push_str(&format!(
                    "transpose_x = {fx}, transpose_y = {fy}, x = {x}, y = {y}"
                ));
                out_lines.push(format!(
                    "        tensor<fp16, {}> {} = matmul({})[name = string(\"{}\")];",
                    out_v.strict, op.out, args, op.out
                ));
            }
            "rms_norm" => {
                // expand: mean(x²) → +eps → rsqrt → x·r → ·w
                let xn = arg_of(op, "x");
                let wn = arg_of(op, "w");
                let x = renof(&ren, &xn);
                let w = renof(&ren, &wn);
                let xs = strict_of(&xn);
                let ws = strict_of(&wn);
                let eps = arg_of(op, "eps");
                let en = match rms_eps.get(&eps) {
                    Some(n) => n.clone(),
                    None => {
                        let n = format!("__eps{}", rms_eps.len());
                        rms_eps.insert(eps.clone(), n.clone());
                        n
                    }
                };
                let o = &op.out;
                let os = out_lines.len();
                let _ = os;
                out_lines.push(format!(
                    "        tensor<fp16, {xs}> {o}__sq = mul(x = {x}, y = {x})[name = string(\"{o}__sq\")];"
                ));
                out_lines.push(format!(
                    "        tensor<fp16, [1, 1, 1, 1]> {o}__mn = reduce_mean(axes = __axv, keep_dims = __bt, x = {o}__sq)[name = string(\"{o}__mn\")];"
                ));
                out_lines.push(format!(
                    "        tensor<fp16, [1, 1, 1, 1]> {o}__ve = add(x = {o}__mn, y = {en})[name = string(\"{o}__ve\")];"
                ));
                out_lines.push(format!(
                    "        tensor<fp16, [1, 1, 1, 1]> {o}__rs = rsqrt(epsilon = __zro, x = {o}__ve)[name = string(\"{o}__rs\")];"
                ));
                out_lines.push(format!(
                    "        tensor<fp16, {xs}> {o}__nr = mul(x = {x}, y = {o}__rs)[name = string(\"{o}__nr\")];"
                ));
                out_lines.push(format!(
                    "        tensor<fp16, {xs}> {o} = mul(x = {o}__nr, y = {w})[name = string(\"{o}\")];",
                ));
                let _ = ws;
            }
            "softmax" => {
                let x = renof(&ren, &arg_of(op, "x"));
                let out_v = vals.get(&op.out).ok_or("softmax out untyped")?;
                out_lines.push(format!(
                    "        tensor<fp16, {}> {} = softmax(axis = __axm1, x = {x})[name = string(\"{}\")];",
                    out_v.strict, op.out, op.out
                ));
            }
            "" => {
                // bare statements (none emitted today)
                out_lines.push(format!("        {}", op.line));
            }
            _ => {
                // elementwise family — re-emit with the strict out shape.
                if let Some(out_v) = vals.get(&op.out) {
                    let argstr = op
                        .args
                        .iter()
                        .map(|(k, v)| format!("{k} = {}", renof(&ren, v)))
                        .collect::<Vec<_>>()
                        .join(", ");
                    out_lines.push(format!(
                        "        tensor<fp16, {}> {} = {}({})[name = string(\"{}\")];",
                        out_v.strict, op.out, op.name, argstr, op.out
                    ));
                } else {
                    out_lines.push(format!("        {}", op.line));
                }
            }
        }

        // a produced value that updates a state port → write_state +
        // fresh read_state; later uses rebind to the `__u_` alias.
        if let Some((port, sshape)) = upd_port.get(&op.out) {
            out_lines.push(format!(
                "        write_state(data = {}, input = {port})[name = string(\"{port}_ws\")];",
                op.out
            ));
            out_lines.push(format!(
                "        tensor<fp16, {sshape}> __u_{port} = read_state(input = {port})[name = string(\"__u_{port}\")];"
            ));
            ren.insert(port.clone(), format!("__u_{port}"));
        }
    }


    // ret: the program's return value only — state updates live in
    // write_state now. The pipeline binds outputs positionally so the
    // port name doesn't matter.
    let ret_name = out.output.0.clone();
    let _ = ret_pos;

    let mut full = String::from("program(1.3)\n[buildInfo = dict<string, string>({{\"coyote\", \"apple\"}})]\n{\n    func main<ios19>(");
    full.push_str(&sig_out.join(", "));
    full.push_str(") {\n");
    for l in &out_lines {
        full.push_str(l);
        full.push('\n');
    }
    full.push_str(&format!("    }} -> ({ret_name});\n}}\n"));
    Ok((full, wb.finish()))
}

fn arg_of(op: &Op, k: &str) -> String {
    op.args
        .iter()
        .find(|(n, _)| n == k)
        .map(|(_, v)| v.clone())
        .unwrap_or_default()
}

/// emitted broadcast-const shapes look like [1,T,1,1] / [1,T,1,D]-ish
/// packs with a leading 1 — param weights are plain [N,K].
fn is_broadcast_shape(lshape: &str) -> bool {
    let v: Vec<i64> = lshape
        .trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    v.len() == 4 && v[0] == 1 && v[2] == 1
}

/// coyote arg name → pipeline shared-port name.
fn port_rename(name: &str) -> Option<&'static str> {
    match name {
        "x" => None,
        "pos" | "kv_write_mask" => Some("kv_write_mask"),
        "mask" | "attn_mask" => Some("attn_mask"),
        "cos" | "rope_cos" => Some("rope_cos"),
        "sin" | "rope_sin" => Some("rope_sin"),
        _ => None,
    }
}

/// split a `func main<ios19>(...)` arg list on top-level commas.
fn split_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let (mut depth, mut cur) = (0i32, String::new());
    for ch in s.chars() {
        match ch {
            '<' | '[' | '(' => depth += 1,
            '>' | ']' | ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(std::mem::take(&mut cur));
                continue;
            }
            _ => {}
        }
        cur.push(ch);
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

/// Emit one `head` program as N vocab-range shards — Apple's own
/// converted models split lm_head the same way (`head_v*-.mlmodelc`
/// with `vocab_start` ranges). The emitted head's largest BLOBFILE
/// record is the head weight, stored `[vocab, hidden]` row-major
/// fp16, so a vocab range is a literal byte slice of that record.
/// Every shard shares the emitted graph text with two patches: the
/// head-weight const becomes `[w, hidden]` and the logits out shape
/// `[1, 1, 1, vocab]` becomes `[1, 1, 1, w]`.
fn write_head_shards(
    dir: &Path,
    mil_out: &MilOut,
    vocab: usize,
    tmpl: &Path,
) -> Result<Vec<String>, String> {
    const SHARD_W: usize = 37984; // Apple's head-shard width
    let n_shards = vocab.div_ceil(SHARD_W).max(1);
    let (mil, wb) = shard_mil(mil_out, &mil_out.mil, &mil_out.blobs)?;

    // Walk `offset = uint64(N)` refs in encounter order — the records
    // were packed in the same order by shard_mil.
    let mut recs: Vec<(u64, u32, Vec<u8>)> = Vec::new();
    let mut scan = mil.as_str();
    while let Some(p) = scan.find("offset = uint64(") {
        let rest = &scan[p + 16..];
        let end = rest.find(')').ok_or("head shard: bad offset")?;
        let off: u64 = rest[..end]
            .trim()
            .parse()
            .map_err(|_| "head shard: bad offset")?;
        let meta = wb
            .get(off as usize..off as usize + 64)
            .ok_or("head shard: offset out of weight.bin")?;
        let dtype = u32::from_le_bytes(meta[4..8].try_into().unwrap());
        let len = u64::from_le_bytes(meta[8..16].try_into().unwrap()) as usize;
        let doff = u64::from_le_bytes(meta[16..24].try_into().unwrap()) as usize;
        let bytes = wb
            .get(doff..doff + len)
            .ok_or("head shard: blob out of weight.bin")?
            .to_vec();
        recs.push((off, dtype, bytes));
        scan = &scan[p + 16..];
    }
    if recs.is_empty() {
        return Err("head shard: no weight records".into());
    }
    let big = recs
        .iter()
        .enumerate()
        .max_by_key(|(_, (_, _, b))| b.len())
        .map(|(i, _)| i)
        .unwrap();
    let hidden = mil_out.inputs.first().map(|(_, t, _)| t.dims[0] as usize).unwrap_or(0);
    let v0 = if hidden > 0 {
        recs[big].2.len() / 2 / hidden
    } else {
        0
    };
    if v0 == 0 || recs[big].2.len() != v0 * hidden * 2 {
        return Err("head shard: can't derive head-weight shape".into());
    }

    let old_wshape = format!("[{v0}, {hidden}]");
    let old_oshape = format!("[1, 1, 1, {v0}]");
    if !mil.contains(&old_wshape) || !mil.contains(&old_oshape) {
        return Err(format!(
            "head shard: expected shapes {old_wshape}/{old_oshape} in emitted mil"
        ));
    }

    let mut entries = Vec::new();
    for s in 0..n_shards {
        let start = s * SHARD_W;
        let end = (start + SHARD_W).min(vocab);
        let w = end - start;
        let mut text = mil
            .replace(&format!("tensor<fp16, {old_wshape}>"), &format!("tensor<fp16, [{w}, {hidden}]>"))
            .replace(&format!("tensor<fp16, {old_oshape}>"), &format!("tensor<fp16, [1, 1, 1, {w}]>"));

        let mut wb2 = WeightBin::new();
        for (idx, (old_off, dtype, bytes)) in recs.iter().enumerate() {
            let slice: &[u8] = if idx == big {
                &bytes[start * hidden * 2..end * hidden * 2]
            } else {
                bytes
            };
            let new_off = wb2.put(*dtype, slice);
            text = text.replace(
                &format!("offset = uint64({old_off})"),
                &format!("offset = uint64({new_off})"),
            );
        }

        let rel = format!("head_{s:02}.mlmodelc");
        let sd = dir.join(&rel);
        let wdir = sd.join("weights");
        std::fs::create_dir_all(&wdir).map_err(|e| e.to_string())?;
        std::fs::write(sd.join("model.mil"), &text).map_err(|e| e.to_string())?;
        std::fs::write(wdir.join("weight.bin"), wb2.finish()).map_err(|e| e.to_string())?;
        for extra in ["metadata.json", "coremldata.bin"] {
            let src = tmpl.join(extra);
            if src.exists() {
                std::fs::copy(&src, sd.join(extra)).map_err(|e| e.to_string())?;
            }
        }
        entries.push(format!("{{\"vocab_start\":{start},\"compiled_path\":\"{rel}\"}}"));
        eprintln!("  shard {rel} ← head vocab {start}..{end}");
    }
    Ok(entries)
}

/// `model.mil` + weight blobs → `<dir>/<name>.mlmodelc/` with the
/// bundle files the e5 compiler expects.
fn write_shard(
    dir: &Path,
    name: &str,
    mil_out: &MilOut,
    tmpl: &Path,
) -> Result<PathBuf, String> {
    let sd = dir.join(format!("{name}.mlmodelc"));
    let wdir = sd.join("weights");
    std::fs::create_dir_all(&wdir).map_err(|e| e.to_string())?;
    let (mil_text, weight_bin) = shard_mil(mil_out, &mil_out.mil, &mil_out.blobs)?;
    if !weight_bin.is_empty() {
        std::fs::write(wdir.join("weight.bin"), weight_bin)
            .map_err(|e| format!("weight.bin: {e}"))?;
    }
    std::fs::write(sd.join("model.mil"), mil_text).map_err(|e| e.to_string())?;
    for extra in ["metadata.json", "coremldata.bin"] {
        let src = tmpl.join(extra);
        if src.exists() {
            std::fs::copy(&src, sd.join(extra)).map_err(|e| e.to_string())?;
        }
    }
    Ok(sd)
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let r = (|| -> Result<(), String> {
        let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
            return Err(
                "usage: coyote-apple shardize <file.cy> -o <dir> [--rope-base N] [--tmpl <shard.mlmodelc>]"
                    .into(),
            );
        };
        let out_dir = args
            .windows(2)
            .find(|w| w[0] == "-o")
            .map(|w| PathBuf::from(&w[1]))
            .ok_or("shardize needs -o <dir>")?;
        let rope_base: f64 = args
            .windows(2)
            .find(|w| w[0] == "--rope-base")
            .and_then(|w| w[1].parse().ok())
            .unwrap_or(1e6);
        let tmpl = args
            .windows(2)
            .find(|w| w[0] == "--tmpl")
            .map(|w| PathBuf::from(&w[1]))
            .unwrap_or_else(|| {
                PathBuf::from(
                    "/Users/savag3/bad_apple/ane_artifacts/qwen3b_v2/layer_s00-01.mlmodelc",
                )
            });
        let only: Option<String> = args
            .windows(2)
            .find(|w| w[0] == "--only")
            .map(|w| w[1].clone());

        let path = PathBuf::from(file);
        let (src, ast) = coyote::parser::parse_file(&path).map_err(|d| d.to_string())?;
        let checked = coyote::typeck::check_all(&ast).map_err(|d| d.to_string())?;
        let by_name: HashMap<&str, &TProgram> = checked
            .ane
            .iter()
            .map(|(n, p)| (n.as_str(), p))
            .collect();

        let mut layers: Vec<&str> = Vec::new();
        let mut heads: Vec<&str> = Vec::new();
        let mut emb_src: Option<String> = None;
        for d in &ast.decls {
            match d {
                Decl::Fn(f) if f.v.target == Target::Ane && by_name.contains_key(f.v.name.as_str()) => {
                    if f.v.name == "head" {
                        heads.push(f.v.name.as_str());
                    } else {
                        layers.push(f.v.name.as_str());
                    }
                }
                Decl::Table { name, src, .. } if name.v == "emb" => {
                    emb_src = src.as_ref().map(|s| s.v.clone());
                }
                _ => {}
            }
        }
        if let Some(o) = &only {
            layers.retain(|n| *n == o.as_str());
            heads.retain(|n| *n == o.as_str());
        }
        if layers.is_empty() && heads.is_empty() {
            return Err(format!("no `fn` programs matched ({:?})", only));
        }
        if only.is_none() && heads.is_empty() {
            return Err("no `head` program — the pipeline needs a logits shard".into());
        }

        std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
        let stage = out_dir.join(".emit");
        let mut shard_entries = Vec::new();
        for (i, name) in layers.iter().enumerate() {
            let prog = by_name[*name];
            let edir = stage.join(name);
            let out = coyote::emit_program(&src, prog, &edir, true)?;
            let rel = format!("layer_{i:02}.mlmodelc");
            write_shard(&out_dir, rel.trim_end_matches(".mlmodelc"), &out, &tmpl)?;
            shard_entries.push(format!(
                "{{\"name\":\"layer_{i:02}\",\"layer_start\":{i},\"layer_end\":{},\"status\":\"compiled\",\"compiled_path\":\"{rel}\"}}",
                i + 1
            ));
            eprintln!("  shard {rel} ← fn {name}");
        }
        let vocab = match &emb_src {
            Some(s) => s
                .split_once("::")
                .map(|(_, tn)| tn)
                .and_then(|tn| {
                    let f = s.split_once("::").map(|(f, _)| f)?;
                    let st = coyote::weights::SafeTensors::load(
                        &path
                            .parent()
                            .unwrap_or(Path::new("."))
                            .join(f)
                            .to_string_lossy(),
                    )
                    .ok()?;
                    st.shape(tn).map(|d| d[0])
                })
                .unwrap_or(151936),
            None => 151936,
        } as usize;

        let mut head_entries = Vec::new();
        for name in heads.iter() {
            let prog = by_name[*name];
            let edir = stage.join(name);
            let out = coyote::emit_program(&src, prog, &edir, true)?;
            head_entries.extend(write_head_shards(&out_dir, &out, vocab, &tmpl)?);
        }

        let l0 = by_name[layers[0]];
        let hidden = l0.ret.ty.numel() as usize;
        let seq = l0
            .states
            .first()
            .map(|(_, _, t)| t.dims[0] as usize)
            .unwrap_or(64);
        let rope_dim = l0
            .ios
            .iter()
            .find(|(n, _, _)| n.contains("cos"))
            .map(|(_, _, t)| t.numel() as usize * 2)
            .unwrap_or(128);

        if only.is_none() {
            let Some(esrc) = &emb_src else {
                return Err("no `table emb` — decoder needs embedding.f16".into());
            };
            let (file_part, tensor_name) = esrc
                .split_once("::")
                .ok_or("table emb src needs file::tensor")?;
            let spath = path.parent().unwrap_or(Path::new(".")).join(file_part);
            let spath = if spath.exists() {
                spath
            } else {
                Path::new(file_part).to_path_buf()
            };
            let st = coyote::weights::SafeTensors::load(&spath.to_string_lossy())
                .map_err(|e| format!("safetensors: {e}"))?;
            let vals = st.f32s(tensor_name)?;
            let mut bytes = Vec::with_capacity(vals.len() * 2);
            for v in &vals {
                bytes.extend(half::f16::from_f32(*v).to_bits().to_le_bytes());
            }
            std::fs::write(out_dir.join("embedding.f16"), &bytes).map_err(|e| e.to_string())?;
            eprintln!("  embedding.f16 — {vocab}×{hidden} fp16");
        }

        let manifest = format!(
            "{{\n \"model\": {{\"architecture\":\"coyote\",\"hidden_size\":{hidden},\"vocab_size\":{vocab},\"seq_len\":{seq},\"rope_dim\":{rope_dim},\"rope_freq_base\":{rope_base},\"stateful\":true,\"native\":true}},\n \"shards\": [{}],\n \"shared\": {{\"embedding\": {{\"path\":\"embedding.f16\"}}, \"lm_head_shards\": [{}]}}\n}}\n",
            shard_entries.join(","),
            head_entries.join(",")
        );
        std::fs::write(out_dir.join("conversion_manifest.json"), &manifest)
            .map_err(|e| e.to_string())?;
        let _ = std::fs::remove_dir_all(&stage);
        eprintln!(
            "✓ {} layer shards + {} head shard(s) → {}",
            layers.len(),
            heads.len(),
            out_dir.display()
        );
        Ok(())
    })();
    r.map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}
