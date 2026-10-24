//! gpu_emit — the third stratum's emitter: lower a checked `TProgram`
//! (target `gpu fn`) to Metal Shading Language source. One `kernel`
//! per program, one thread per output element; elementwise chains
//! inline, reductions and matmuls emit per-thread loops.
//!
//! Storage conventions match the rest of the system: io and state
//! buffers are declared-order row-major f32; 2-D params are stored
//! `[N, K]` (the same transposed layout ANE surfaces and the CPU
//! evaluator see), so `x @ w` reads coalesced rows of `w`.
//!
//! GPU v1 computes in `float` — dtype annotations still typecheck,
//! device storage is f32. `grad`/`adam`/state writes are not lowered
//! on this backend yet; they emit honest errors, not silent wrongness.

use coyote::ast::{BinOp, IoDir, Qual, UnOp};
use coyote::lower::sha256_hex;
use coyote::typeck::{CTy, TExpr, TIndex, TProgram, TStmt, Typed};
use std::collections::HashMap;

/// What the runtime needs to bind buffers and run the kernel.
pub struct GpuOut {
    pub msl: String,
    pub msl_sha: String,
    pub manifest: String,
    /// io in ports: (port, ty, contig-flag unused on gpu)
    pub inputs: Vec<(String, CTy, bool)>,
    /// trainable params: (name, port = name, ty, src) — same shape as MilOut
    pub trainable: Vec<(String, String, CTy, Option<String>)>,
    /// state fields: (key `s.f`, port `s_f`, ty)
    pub states: Vec<(String, String, CTy)>,
    /// kernel arg order after `out`: In ports, all params, state ports.
    pub args: Vec<Arg>,
    pub output: (String, CTy),
}

#[derive(Debug, Clone)]
pub enum Arg {
    In(String),
    Param(String),
    State(String),
}

impl Arg {
    pub fn port(&self) -> &str {
        match self {
            Arg::In(p) | Arg::Param(p) | Arg::State(p) => p,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Kind {
    /// every buffer reads row-major over its declared dims — params
    /// too: their [N, K] storage IS the value, matching the evaluator.
    Row,
    /// a param used as a matmul rhs — the one position where the
    /// [N, K] storage convention applies: logical (k, n) → n*K + k.
    /// `p.T` as rhs is the inverse: row-major over p.T's own dims.
    ParamT,
}

struct Ge {
    /// name → (arg, ty, kind) for every visible buffer
    bufs: HashMap<String, (String, CTy, Kind)>,
    /// let-bound names → their typed tree, *in binding order* (inlined
    /// lazily, recomputed per element — correctness first, fusion later)
    scopes: Vec<Vec<(String, Typed)>>,
    /// while resolving a let-bound expr at position i of scope s, only
    /// earlier bindings *of s* are visible — the evaluator binds
    /// eagerly in order; the emitter mirrors that lazily. This is what
    /// keeps inliner alias pairs (`__arg0 := w`, `w := __arg0`) and
    /// shadow-lets (`let x = x*2`) from recursing forever: the inner
    /// `w`/`x` refers to whatever was bound *before*, i.e. the buffer.
    /// Entries are (scope index, position) — a bound expr can open a
    /// nested Block, so the limit travels with its scope.
    resolve_limit: Vec<(usize, usize)>,
    lines: Vec<String>,
    tmp: usize,
}

impl Ge {
    fn tmp(&mut self) -> String {
        self.tmp += 1;
        format!("t{}", self.tmp)
    }

    /// find `name` among let bindings, innermost scope first, honoring
    /// `resolve_limit` per scope. Returns (scope, position, expr).
    fn pos_of(&self, name: &str) -> Option<(usize, usize, &Typed)> {
        for (si, s) in self.scopes.iter().enumerate().rev() {
            let lim = self
                .resolve_limit
                .iter()
                .rev()
                .find(|(s2, _)| *s2 == si)
                .map(|(_, p)| *p)
                .unwrap_or(s.len());
            if let Some(p) = s[..lim.min(s.len())].iter().rposition(|(n, _)| n == name) {
                return Some((si, p, &s[p].1));
            }
        }
        None
    }

    /// flat buffer index for node dims `dims` at logical coords `c`,
    /// under storage `kind`.
    fn flat(dims: &[i64], c: &[String], kind: &Kind) -> String {
        if dims.is_empty() || dims.iter().product::<i64>() == 1 {
            return "0".into();
        }
        if dims.len() == 2 && matches!(kind, Kind::ParamT) {
            // stored [N, K]: (k, n) → n*K + k
            return format!("({c1})*{k}+({c0})", c0 = c[0], c1 = c[1], k = dims[0]);
        }
        // row-major
        let mut terms = Vec::new();
        for (i, ci) in c.iter().enumerate() {
            let stride: i64 = dims[i + 1..].iter().product();
            terms.push(format!("({ci})*{stride}"));
        }
        terms.join("+")
    }

    /// element access for `e` in an elementwise context at `coords`
    /// (len = consumer's rank): scalar broadcast, same-rank, or
    /// trailing-rank broadcast.
    fn elem(&mut self, e: &Typed, coords: &[String]) -> Result<String, String> {
        if e.ty.dims.len() == coords.len() {
            return self.gen(e, coords);
        }
        if e.ty.numel() == 1 {
            return self.gen(e, &[]);
        }
        if e.ty.dims.len() < coords.len() {
            let k = coords.len() - e.ty.dims.len();
            return self.gen(e, &coords[k..]);
        }
        Err(format!(
            "gpu emit: can't broadcast {} into rank-{} context",
            e.ty.show(),
            coords.len()
        ))
    }

    /// decompose a flat loop index `v` into per-dim coord exprs.
    fn coords_of(v: &str, dims: &[i64]) -> Vec<String> {
        dims.iter()
            .enumerate()
            .map(|(i, d)| {
                let stride: i64 = dims[i + 1..].iter().product();
                if stride == 1 {
                    format!("({v}%{d})")
                } else {
                    format!("(({v}/{stride})%{d})")
                }
            })
            .collect()
    }

    fn gen(&mut self, e: &Typed, coords: &[String]) -> Result<String, String> {
        match &e.e {
            TExpr::Lit(v) => Ok(format!("float({v:e})")),
            TExpr::Int(i) => Ok(format!("float({i})")),
            TExpr::Ident(s) => {
                if let Some((si, pos, b)) = self.pos_of(s) {
                    let b = b.clone();
                    self.resolve_limit.push((si, pos));
                    let r = self.gen(&b, coords);
                    self.resolve_limit.pop();
                    return r;
                }
                let Some((arg, ty, _kind)) = self.bufs.get(s) else {
                    return Err(format!("gpu emit: unbound tensor {s}"));
                };
                // params too read row-major over declared dims — their
                // [N,K] storage IS the value (matches the evaluator);
                // the transposed read only applies in matmul rhs.
                let flat = Self::flat(&ty.dims, coords, &Kind::Row);
                Ok(format!("{arg}[{flat}]"))
            }
            TExpr::Field(base, f) => {
                // `kv.k` — state field access.
                if let TExpr::Ident(s) = &base.e {
                    let key = format!("{s}.{f}");
                    if let Some((arg, ty, kind)) = self.bufs.get(&key) {
                        let flat = Self::flat(&ty.dims, coords, kind);
                        return Ok(format!("{arg}[{flat}]"));
                    }
                }
                Err("gpu emit: field access outside state decls".into())
            }
            TExpr::Unary(op, a) => {
                let v = self.elem(a, coords)?;
                Ok(match op {
                    UnOp::Neg => format!("(-({v}))"),
                    UnOp::Not => format!("(({v}) == 0.0f ? 1.0f : 0.0f)"),
                })
            }
            TExpr::Bin(BinOp::MatMul, a, b) => {
                let (ad, bd) = (&a.ty.dims, &b.ty.dims);
                let k = *ad.last().ok_or("gpu emit: matmul of scalars")?;
                let acc = self.tmp();
                let ki = format!("_k{}", self.tmp);
                self.lines.push(format!("float {acc} = 0.0f;"));
                self.lines
                    .push(format!("for (int {ki} = 0; {ki} < {k}; ++{ki}) {{"));
                let (lc, rc) = match (ad.len(), bd.len()) {
                    // [K] @ [K,N] → [N]
                    (1, 2) => (vec![ki.clone()], vec![ki.clone(), coords[0].clone()]),
                    // [M,K] @ [K,N] → [M,N]
                    (2, 2) => (
                        vec![coords[0].clone(), ki.clone()],
                        vec![ki.clone(), coords[1].clone()],
                    ),
                    // [M,K] @ [K] → [M]
                    (2, 1) => (vec![coords[0].clone(), ki.clone()], vec![ki.clone()]),
                    // [K] @ [K] → scalar
                    (1, 1) => (vec![ki.clone()], vec![ki.clone()]),
                    _ => return Err(format!("gpu emit: matmul {:?} @ {:?} not lowered", ad, bd)),
                };
                let lv = self.gen(a, &lc)?;
                // matmul rhs — eval reads EVERY named rhs's storage
                // flat as `w[n*K + k]` (params are physically [N,K];
                // io/state/let buffers get the same linear reinterpret).
                // `p.T` non-square is the one real storage swap; a
                // square-param `.T` is a no-op. State fields keep their
                // own logical convention (`kv.v` column contract,
                // `kv.k.T` swapped). A materialized rhs resolves to the
                // element at flat index n*K+k of its declared [K,N] view.
                let kf = if rc.len() > 1 {
                    format!("({n})*{k}+({kk})", n = rc[1], k = k, kk = rc[0])
                } else {
                    rc[0].clone()
                };
                let rv = match &b.e {
                    TExpr::Field(..) => self.gen(b, &rc)?,
                    TExpr::Transpose(i) if matches!(&i.e, TExpr::Field(..)) => self.gen(b, &rc)?,
                    TExpr::Transpose(i) if matches!(&i.e, TExpr::Ident(s) if self.bufs.contains_key(s)) =>
                    {
                        let (arg, ty, kind) = match &i.e {
                            TExpr::Ident(s) => self.bufs.get(s).unwrap().clone(),
                            _ => unreachable!(),
                        };
                        if matches!(kind, Kind::ParamT) && ty.dims[0] != ty.dims[1] {
                            format!("{arg}[{}]", Self::flat(&b.ty.dims, &rc, &Kind::Row))
                        } else {
                            format!("{arg}[{kf}]")
                        }
                    }
                    TExpr::Transpose(i) => {
                        // `.T` on a materialized name — element at flat
                        // n*K+k of the INNER tensor's declared layout.
                        let cs = Self::coords_of(&kf, &i.ty.dims);
                        self.gen(i, &cs)?
                    }
                    TExpr::Ident(s) if self.bufs.contains_key(s) => {
                        let (arg, _, _) = self.bufs.get(s).unwrap().clone();
                        format!("{arg}[{kf}]")
                    }
                    _ => {
                        let cs = Self::coords_of(&kf, &b.ty.dims);
                        self.gen(b, &cs)?
                    }
                };
                self.lines.push(format!("{acc} += ({lv}) * ({rv});"));
                self.lines.push("}".into());
                Ok(acc)
            }
            TExpr::Bin(op, a, b) => {
                let ae = self.elem(a, coords)?;
                let be = self.elem(b, coords)?;
                Ok(match op {
                    BinOp::Add => format!("({ae} + {be})"),
                    BinOp::Sub => format!("({ae} - {be})"),
                    BinOp::Mul => format!("({ae} * {be})"),
                    BinOp::Div => format!("({ae} / {be})"),
                    BinOp::Mod => format!("fmod({ae}, {be})"),
                    BinOp::Eq => format!("float(({ae}) == ({be}))"),
                    BinOp::Ne => format!("float(({ae}) != ({be}))"),
                    BinOp::Lt => format!("float(({ae}) < ({be}))"),
                    BinOp::Gt => format!("float(({ae}) > ({be}))"),
                    BinOp::Le => format!("float(({ae}) <= ({be}))"),
                    BinOp::Ge => format!("float(({ae}) >= ({be}))"),
                    BinOp::And => format!("((({ae}) != 0.0f) && (({be}) != 0.0f) ? 1.0f : 0.0f)"),
                    BinOp::Or => format!("((({ae}) != 0.0f) || (({be}) != 0.0f) ? 1.0f : 0.0f)"),
                    BinOp::MatMul => unreachable!(),
                })
            }
            TExpr::Transpose(inner) => {
                // logical transpose — reads through swapped coords, no
                // data movement (param storage already IS w.T's layout).
                let mut c = coords.to_vec();
                if c.len() < 2 {
                    return Err("gpu emit: .T on rank < 2".into());
                }
                c.swap(0, 1);
                self.gen(inner, &c)
            }
            TExpr::Index(base, idxs) => {
                // index ops consume leading dims of base: Lit pins,
                // Range offsets. Remaining coords fill free dims.
                let mut c = Vec::new();
                let mut used = 0usize;
                for ix in idxs {
                    match ix {
                        TIndex::Lit(i) => c.push(format!("{i}")),
                        TIndex::Range(lo, _) => {
                            if used >= coords.len() {
                                return Err("gpu emit: index arity".into());
                            }
                            c.push(format!("({lo} + {ci})", ci = coords[used]));
                            used += 1;
                        }
                    }
                }
                c.extend_from_slice(&coords[used..]);
                if c.len() != base.ty.dims.len() {
                    return Err(format!(
                        "gpu emit: index on {} leaves {} coords, base wants {}",
                        base.ty.show(),
                        c.len(),
                        base.ty.dims.len()
                    ));
                }
                self.gen(base, &c)
            }
            TExpr::Block(stmts, last) => {
                self.scopes.push(Vec::new());
                for s in stmts {
                    self.stmt(s)?;
                }
                let v = self.gen(last, coords);
                self.scopes.pop();
                v
            }
            TExpr::Call(name, args, _) => self.call(name, args, coords),
            TExpr::Grad { .. } => {
                Err("gpu stratum: grad/adam not lowered — train on cpu or ane".into())
            }
        }
    }

    fn call(&mut self, name: &str, args: &[Typed], coords: &[String]) -> Result<String, String> {
        match name {
            "relu" | "sigmoid" | "tanh" | "silu" | "gelu" => {
                let v = self.elem(&args[0], coords)?;
                Ok(match name {
                    "relu" => format!("fmax({v}, 0.0f)"),
                    "sigmoid" => format!("(1.0f / (1.0f + exp(-({v}))))"),
                    "tanh" => format!("tanh({v})"),
                    "silu" => format!("(({v}) / (1.0f + exp(-({v}))))"),
                    _ => format!(
                        "(0.5f * ({v}) * (1.0f + tanh(0.7978845608f * (({v}) + 0.044715f * ({v}) * ({v}) * ({v})))))"
                    ),
                })
            }
            "softmax" => {
                // eval semantics: max-subtracted exp over the whole
                // flat arg, normalized.
                let a = &args[0];
                let n: i64 = a.ty.dims.iter().product();
                let (mx, sm, ix) = (self.tmp(), self.tmp(), self.tmp());
                self.lines.push(format!("float {mx} = -3.402823e38f;"));
                self.lines
                    .push(format!("for (int {ix} = 0; {ix} < {n}; ++{ix}) {{"));
                let cs = Self::coords_of(&ix, &a.ty.dims);
                let v = self.gen(a, &cs)?;
                self.lines.push(format!("{mx} = fmax({mx}, {v});"));
                self.lines.push("}".into());
                self.lines.push(format!("float {sm} = 0.0f;"));
                let cs = Self::coords_of(&ix, &a.ty.dims);
                self.lines
                    .push(format!("for (int {ix} = 0; {ix} < {n}; ++{ix}) {{"));
                let v = self.gen(a, &cs)?;
                self.lines.push(format!("{sm} += exp({v} - {mx});"));
                self.lines.push("}".into());
                let v = self.gen(a, coords)?;
                Ok(format!("(exp({v} - {mx}) / {sm})"))
            }
            "rmsnorm" => {
                // eval: x * (1/sqrt(mean(x²)+1e-5)) * w — reduce over x.
                let x = &args[0];
                let n: i64 = x.ty.dims.iter().product();
                let (ms, ix) = (self.tmp(), self.tmp());
                self.lines.push(format!("float {ms} = 0.0f;"));
                self.lines
                    .push(format!("for (int {ix} = 0; {ix} < {n}; ++{ix}) {{"));
                let cs = Self::coords_of(&ix, &x.ty.dims);
                let v = self.gen(x, &cs)?;
                self.lines.push(format!("{ms} += ({v}) * ({v});"));
                self.lines.push("}".into());
                self.lines
                    .push(format!("{ms} = rsqrt({ms} / float({n}) + 1e-5f);"));
                let xv = self.gen(x, coords)?;
                let wv = self.gen(&args[1], coords)?;
                Ok(format!("(({xv}) * {ms} * ({wv}))"))
            }
            "mse" => {
                // scalar loss — reduce over the full flat pair.
                let a = &args[0];
                let n: i64 = a.ty.dims.iter().product();
                let (acc, ix) = (self.tmp(), self.tmp());
                self.lines.push(format!("float {acc} = 0.0f;"));
                self.lines
                    .push(format!("for (int {ix} = 0; {ix} < {n}; ++{ix}) {{"));
                let cs = Self::coords_of(&ix, &a.ty.dims);
                let av = self.gen(a, &cs)?;
                let bv = self.gen(&args[1], &cs)?;
                self.lines
                    .push(format!("{acc} += ({av} - {bv}) * ({av} - {bv});"));
                self.lines.push("}".into());
                Ok(format!("({acc} / float({n}))"))
            }
            "gather" | "embed" => {
                // eval synthesizes `idx @ tab` — rhs convention applies:
                // element at flat n*K+k of tab's storage (params: the
                // [N,K] storage row n). one-hot → storage row select.
                let (tab, idx) = (&args[0], &args[1]);
                let v = tab.ty.dims[0];
                let (acc, iv) = (self.tmp(), self.tmp());
                let f = format!("({})*{v}+({iv})", coords[0]);
                let tv = match &tab.e {
                    TExpr::Ident(s) if self.bufs.contains_key(s) => {
                        let (arg, _, _) = self.bufs.get(s).unwrap().clone();
                        format!("{arg}[{f}]")
                    }
                    _ => {
                        let cs = Self::coords_of(&f, &tab.ty.dims);
                        self.gen(tab, &cs)?
                    }
                };
                self.lines.push(format!("float {acc} = 0.0f;"));
                self.lines
                    .push(format!("for (int {iv} = 0; {iv} < {v}; ++{iv}) {{"));
                let wv = self.gen(idx, &[iv.clone()])?;
                self.lines.push(format!("{acc} += ({tv}) * ({wv});"));
                self.lines.push("}".into());
                Ok(acc)
            }
            "concat" => {
                // rank-1 concat: out[i] = i < K1 ? a[i] : b[i - K1].
                let k1 = args[0].ty.dims[0];
                let av = self.gen(&args[0], coords)?;
                let bv = self.gen(&args[1], &[format!("({c} - {k1})", c = coords[0])])?;
                Ok(format!("(({c}) < {k1} ? {av} : {bv})", c = coords[0]))
            }
            "outer" => {
                // out[i, j] = a[i] * b[j]
                let av = self.gen(&args[0], &coords[..1])?;
                let bv = self.gen(&args[1], &coords[1..])?;
                Ok(format!("(({av}) * ({bv}))"))
            }
            "grad" | "adam" => {
                Err("gpu stratum: grad/adam not lowered — train on cpu or ane".into())
            }
            _ => Err(format!("gpu emit: builtin {name} not lowered")),
        }
    }

    fn stmt(&mut self, s: &TStmt) -> Result<(), String> {
        match s {
            TStmt::Let { name, expr } => {
                self.scopes
                    .last_mut()
                    .unwrap()
                    .push((name.clone(), expr.clone()));
                Ok(())
            }
            TStmt::Assign { .. }
            | TStmt::ParamUpdate { .. }
            | TStmt::AdamUpdate { .. }
            | TStmt::StateUpdate { .. } => {
                Err("gpu stratum: param/state updates not lowered — forward passes only".into())
            }
            TStmt::Expr(_) => Ok(()),
        }
    }
}

/// Lower a checked gpu-target `TProgram` to MSL source + the buffer
/// plan the runtime binds against.
pub fn emit(src: &str, name: &str, p: &TProgram) -> Result<GpuOut, String> {
    let (oname, oty) = ("out".to_string(), p.ret.ty.clone());
    let mut ge = Ge {
        bufs: HashMap::new(),
        scopes: vec![Vec::new()],
        resolve_limit: Vec::new(), // (scope, pos) pairs
        lines: Vec::new(),
        tmp: 0,
    };
    let mut sig = vec![format!(
        "device float* _out [[buffer({})]]",
        0 // out is always buffer 0
    )];
    let mut args = Vec::new();
    let mut bi = 1usize;
    for (port, dir, ty) in &p.ios {
        if *dir != IoDir::In {
            continue;
        }
        let arg = format!("b_{port}");
        sig.push(format!("device const float* {arg} [[buffer({bi})]]"));
        ge.bufs.insert(port.clone(), (arg, ty.clone(), Kind::Row));
        args.push(Arg::In(port.clone()));
        bi += 1;
    }
    for t in &p.params {
        let arg = format!("b_{}", t.name);
        sig.push(format!("device const float* {arg} [[buffer({bi})]]"));
        ge.bufs
            .insert(t.name.clone(), (arg, t.ty.clone(), Kind::ParamT));
        args.push(Arg::Param(t.name.clone()));
        bi += 1;
    }
    let mut states = Vec::new();
    for (key, port, ty) in &p.states {
        let arg = format!("b_{port}");
        sig.push(format!("device float* {arg} [[buffer({bi})]]"));
        ge.bufs.insert(key.clone(), (arg, ty.clone(), Kind::Row));
        states.push((key.clone(), port.clone(), ty.clone()));
        args.push(Arg::State(port.clone()));
        bi += 1;
    }

    // gid → out coords (row-major over the output's dims)
    let mut coords = Vec::new();
    let r = oty.dims.len();
    for i in 0..r {
        let cname = format!("c{i}");
        let stride: i64 = oty.dims[i + 1..].iter().product();
        let d = oty.dims[i];
        if r == 1 {
            ge.lines.push(format!("uint {cname} = gid;"));
        } else if stride == 1 {
            ge.lines.push(format!("uint {cname} = gid % {d};"));
        } else {
            ge.lines
                .push(format!("uint {cname} = (gid / {stride}) % {d};"));
        }
        coords.push(cname);
    }

    // body: lets inline, everything else rejected up-front
    for s in &p.body {
        ge.stmt(s)?;
    }
    let v = ge.gen(&p.ret, &coords)?;
    ge.lines.push(format!("_out[gid] = ({v});"));

    let msl = format!(
        "// coyote gpu emit — {name}\n#include <metal_stdlib>\nusing namespace metal;\n\nkernel void coyote_{name}(\n    {},\n    uint gid [[thread_position_in_grid]])\n{{\n{}\n}}\n",
        sig.join(",\n    "),
        ge.lines
            .iter()
            .map(|l| format!("    {l}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
    let msl_sha = sha256_hex(msl.as_bytes());
    let manifest = format!(
        "{{\"program\":\"{name}\",\"backend\":\"metal\",\"src_sha\":\"{}\",\"msl_sha\":\"{msl_sha}\",\"out\":\"{oname}\",\"args\":{}}}",
        sha256_hex(src.as_bytes()),
        args.iter()
            .map(|a| format!("\"{}\"", a.port()))
            .collect::<Vec<_>>()
            .join(","),
    );
    let inputs = p
        .ios
        .iter()
        .filter(|(_, d, _)| *d == IoDir::In)
        .map(|(n, _, t)| (n.clone(), t.clone(), true))
        .collect();
    let trainable = p
        .params
        .iter()
        .filter(|t| t.qual == Qual::Trainable)
        .map(|t| (t.name.clone(), t.name.clone(), t.ty.clone(), t.src.clone()))
        .collect();
    Ok(GpuOut {
        msl,
        msl_sha,
        manifest,
        inputs,
        trainable,
        states,
        args,
        output: (oname.clone(), oty.clone()),
    })
}
