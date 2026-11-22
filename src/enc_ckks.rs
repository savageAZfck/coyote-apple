//! enc_ckks — the `enc` stratum's engine: encrypted execution.
//!
//! `enc fn` declares a program that runs on ciphertext. Inputs arrive
//! as plaintext ports and are encrypted under an LWE scheme at exec
//! time; every op in the typed body evaluates homomorphically against
//! plaintext params (the CHET deployment model — server weights stay
//! clear, client data stays encrypted); the output decrypts on the
//! way back. Linear ops only — `+`, `-`, `*`, `@`, `* const` — that
//! is what leveled schemes do without a bootstrap. Nonlinear builtins
//! are rejected at emit, the same constraint every CKKS compiler has.
//!
//! Noise is a first-class budget: ciphertexts track error growth and
//! a multiply that would blow the decode margin fails at run time
//! with the depth it died at — never a wrong answer.

use coyote::ast::{BinOp, IoDir};
use coyote::backend::{Backend, BackendEmit, BackendExec};
use coyote::typeck::{CTy, TExpr, TProgram, TStmt, Typed};
use std::collections::HashMap;
use std::path::Path;

const Q: u64 = 1u64 << 62;
const DELTA: u64 = 1u64 << 20;
/// deepest fixed-point level this engine carries — one plaintext-multiply
/// level (scale Δ); deeper needs a bootstrap we don't have.
const MAX_SCALE: u64 = DELTA;

/// (a, b) with b = a·s + e + m·Δ·scale mod q. `scale` is the accumulated
/// fixed-point multiplier from plaintext scalings — every ct_scale bumps
/// it by Δ, decrypt divides by Δ·scale.
/// `pub(crate)`: `pcc fn` + `enc fn` stacked (enc×pcc) ships these lanes
/// over the wire — 4 u64 LE words per lane, the "Ct quad".
#[derive(Clone)]
pub(crate) struct Ct {
    pub(crate) a: u64,
    pub(crate) b: u64,
    /// worst-case |e| bound (raw integer space) — multiplies scale it,
    /// adds accumulate it.
    pub(crate) noise: u64,
    /// fixed-point level: powers of DELTA. Fresh ciphertexts encode at 1.
    pub(crate) scale: u64,
}

/// Evaluated cell: lanes + declared shape (from the typed AST — never
/// guessed).
#[derive(Clone)]
pub(crate) enum V {
    Ct(Vec<Ct>, Vec<i64>),
    Pt(Vec<f64>, Vec<i64>),
}

impl V {
    fn shape(&self) -> &[i64] {
        match self {
            V::Ct(_, s) | V::Pt(_, s) => s,
        }
    }
    fn dom(&self) -> &'static str {
        match self {
            V::Ct(..) => "ciphertext",
            V::Pt(..) => "plaintext",
        }
    }
}

pub struct EncBackend;

// --- tiny rng (deterministic seeds, not a crypto PRNG — noise source) --------

pub(crate) fn rng(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

// --- LWE core -----------------------------------------------------------------

pub(crate) fn lwe_enc(m: f64, s: u64, rng_s: &mut u64) -> Ct {
    let a = rng(rng_s) % Q;
    let e = (rng(rng_s) % 5) as i64 - 2; // ±2 rounding noise
    let m_scaled = (m * DELTA as f64).round() as i64;
    let b = (a.wrapping_mul(s) as i128 + e as i128 + m_scaled as i128).rem_euclid(Q as i128) as u64;
    Ct {
        a,
        b,
        noise: e.unsigned_abs(),
        scale: 1,
    }
}

pub(crate) fn lwe_dec(c: &Ct, s: u64) -> f64 {
    let v = (c.b as i128 - c.a.wrapping_mul(s) as i128).rem_euclid(Q as i128);
    let v = if v > (Q as i128) / 2 {
        v - Q as i128
    } else {
        v
    };
    v as f64 / (DELTA as f64 * c.scale as f64)
}

/// lift a ciphertext to a deeper fixed-point level — multiplies payload
/// and noise by the scale gap (levels only ever differ by powers of Δ).
fn rescale(x: &Ct, target: u64) -> Ct {
    if x.scale >= target {
        return x.clone();
    }
    let f = (target / x.scale) as i128;
    Ct {
        a: ((x.a as i128 * f).rem_euclid(Q as i128)) as u64,
        b: ((x.b as i128 * f).rem_euclid(Q as i128)) as u64,
        noise: x.noise.saturating_mul(f as u64),
        scale: target,
    }
}

fn ct_add(x: &Ct, y: &Ct) -> Ct {
    let s = x.scale.max(y.scale);
    let x = rescale(x, s);
    let y = rescale(y, s);
    Ct {
        a: (x.a + y.a) % Q,
        b: (x.b + y.b) % Q,
        noise: x.noise + y.noise,
        scale: s,
    }
}
fn ct_neg(x: &Ct) -> Ct {
    Ct {
        a: (Q - x.a) % Q,
        b: (Q - x.b) % Q,
        noise: x.noise,
        scale: x.scale,
    }
}
fn ct_scale(x: &Ct, k: f64) -> Result<Ct, String> {
    let scale = x.scale.checked_mul(DELTA).unwrap_or(u64::MAX);
    if scale > MAX_SCALE {
        return Err(format!(
            "enc: multiply depth exceeded — this level carries no bootstrap \
             (scale {} > {MAX_SCALE})",
            scale
        ));
    }
    // fixed-point: multiply by round(k·Δ) so decrypt(Δ·scale) recovers m·k
    let ki = (k * DELTA as f64).round() as i128;
    Ok(Ct {
        a: ((x.a as i128 * ki).rem_euclid(Q as i128)) as u64,
        b: ((x.b as i128 * ki).rem_euclid(Q as i128)) as u64,
        noise: x.noise.saturating_mul(ki.unsigned_abs() as u64),
        scale,
    })
}
fn ct_plus_plain(x: &Ct, m: f64) -> Ct {
    let d = (m * DELTA as f64 * x.scale as f64).round() as i128;
    Ct {
        a: x.a,
        b: ((x.b as i128 + d).rem_euclid(Q as i128)) as u64,
        noise: x.noise,
        scale: x.scale,
    }
}

// --- evaluator: typed body over both domains ----------------------------------

struct Env {
    vals: HashMap<String, V>,
}

impl Env {
    fn check(&self, v: &V, where_: &str) -> Result<(), String> {
        if let V::Ct(cs, _) = v {
            for c in cs {
                // decode margin: the payload (noise + message·Δ·scale) must
                // round back — noise past a quarter of the scale gap can't.
                let margin = (DELTA as u128 * c.scale as u128 / 4) as u128;
                if c.noise as u128 >= margin {
                    return Err(format!(
                        "enc: noise budget exceeded at {where_} ({} > {margin}) — \
                         deeper circuit needs a bootstrap this level doesn't carry",
                        c.noise
                    ));
                }
            }
        }
        Ok(())
    }
}

/// broadcast a pt lane vector to n elements (scalar or same-size).
fn pt_bc(p: &[f64], n: usize) -> Result<Vec<f64>, String> {
    if p.len() == n {
        Ok(p.to_vec())
    } else if p.len() == 1 {
        Ok(vec![p[0]; n])
    } else {
        Err(format!("enc: can't broadcast {} lanes to {n}", p.len()))
    }
}

fn pt_bin(
    op: BinOp,
    a: &[f64],
    ash: &[i64],
    b: &[f64],
    bsh: &[i64],
) -> Result<(Vec<f64>, Vec<i64>), String> {
    let f = |x: f64, y: f64| match op {
        BinOp::Add => x + y,
        BinOp::Sub => x - y,
        BinOp::Mul => x * y,
        BinOp::Div => x / y,
        _ => f64::NAN,
    };
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
            let n = a.len().max(b.len());
            let av = pt_bc(a, n)?;
            let bv = pt_bc(b, n)?;
            let sh = if a.len() >= b.len() { ash } else { bsh };
            Ok(((0..n).map(|i| f(av[i], bv[i])).collect(), sh.to_vec()))
        }
        BinOp::MatMul => {
            // [m,k] @ [k,n] — 1-D lhs promotes to row vector
            let (m, k) = match ash {
                [m, k] => (*m as usize, *k as usize),
                [k] => (1usize, *k as usize),
                _ => return Err("enc: matmul lhs rank".into()),
            };
            let (k2, n) = match bsh {
                [k, n] => (*k as usize, *n as usize),
                _ => return Err("enc: matmul rhs must be a matrix".into()),
            };
            if k != k2 {
                return Err(format!("enc: matmul inner dims {k} vs {k2}"));
            }
            let mut o = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    for t in 0..k {
                        o[i * n + j] += a[i * k + t] * b[t * n + j];
                    }
                }
            }
            let sh = if ash.len() == 1 {
                vec![n as i64]
            } else {
                vec![m as i64, n as i64]
            };
            Ok((o, sh))
        }
        other => Err(format!("enc: pt op {other:?} unsupported")),
    }
}

fn v_bin(op: BinOp, x: &V, y: &V, env: &Env) -> Result<V, String> {
    let out = match (x, y) {
        (V::Ct(a, ash), V::Ct(b, bsh)) if a.len() == b.len() => {
            let r: Vec<Ct> = match op {
                BinOp::Add => a.iter().zip(b).map(|(u, v)| ct_add(u, v)).collect(),
                BinOp::Sub => a
                    .iter()
                    .zip(b)
                    .map(|(u, v)| ct_add(u, &ct_neg(v)))
                    .collect(),
                other => {
                    return Err(format!(
                        "enc: ct×ct '{other:?}' needs relinearization — not at this level"
                    ))
                }
            };
            V::Ct(r, ash.clone())
        }
        (V::Ct(a, ash), V::Pt(p, _)) | (V::Pt(p, _), V::Ct(a, ash))
            if matches!(op, BinOp::Add | BinOp::Sub) =>
        {
            let pv = pt_bc(p, a.len())?;
            let sign = if matches!(op, BinOp::Sub) && matches!(x, V::Pt(..)) {
                -1.0
            } else if matches!(op, BinOp::Sub) && matches!(y, V::Pt(..)) {
                -1.0
            } else {
                1.0
            };
            // Sub: x-y — if the pt is on the right it subtracts; pt-ct
            // isn't representable without negating the ct.
            let vals: Vec<Ct> = a
                .iter()
                .enumerate()
                .map(|(i, u)| ct_plus_plain(u, sign * pv[i]))
                .collect();
            let vals = if matches!(op, BinOp::Sub) && matches!(x, V::Pt(..)) {
                // pt - ct = negate ct then add pt
                vals.iter().map(ct_neg).collect()
            } else {
                vals
            };
            V::Ct(vals, ash.clone())
        }
        (V::Ct(a, ash), V::Pt(p, _)) | (V::Pt(p, _), V::Ct(a, ash)) if matches!(op, BinOp::Mul) => {
            let pv = pt_bc(p, a.len())?;
            V::Ct(
                a.iter()
                    .enumerate()
                    .map(|(i, u)| ct_scale(u, pv[i]))
                    .collect::<Result<_, _>>()?,
                ash.clone(),
            )
        }
        (V::Ct(x, xsh), V::Pt(w, wsh)) if matches!(op, BinOp::MatMul) => {
            let (m, k) = match xsh.as_slice() {
                [m, k] => (*m as usize, *k as usize),
                [k] => (1usize, *k as usize),
                _ => return Err("enc: matmul lhs rank".into()),
            };
            let (k2, n) = match wsh.as_slice() {
                [k, n] => (*k as usize, *n as usize),
                _ => return Err("enc: matmul rhs must be a matrix".into()),
            };
            if k != k2 {
                return Err(format!("enc: matmul inner dims {k} vs {k2}"));
            }
            let mut out = Vec::with_capacity(m * n);
            for i in 0..m {
                for j in 0..n {
                    let mut acc = Ct {
                        a: 0,
                        b: 0,
                        noise: 0,
                        scale: 1,
                    };
                    for t in 0..k {
                        acc = ct_add(&acc, &ct_scale(&x[i * k + t], w[t * n + j])?);
                    }
                    out.push(acc);
                }
            }
            let sh = if xsh.len() == 1 {
                vec![n as i64]
            } else {
                vec![m as i64, n as i64]
            };
            V::Ct(out, sh)
        }
        (V::Pt(a, ash), V::Pt(b, bsh)) => {
            let (o, sh) = pt_bin(op, a, ash, b, bsh)?;
            V::Pt(o, sh)
        }
        _ => {
            return Err(format!(
                "enc: unsupported {op:?} over ({} {:?}, {} {:?})",
                x.dom(),
                x.shape(),
                y.dom(),
                y.shape()
            ))
        }
    };
    env.check(&out, "binary op")?;
    Ok(out)
}

fn eval(e: &Typed, env: &mut Env) -> Result<V, String> {
    match &e.e {
        TExpr::Lit(f) => Ok(V::Pt(vec![*f], vec![])),
        TExpr::Int(i) => Ok(V::Pt(vec![*i as f64], vec![])),
        TExpr::Ident(n) => env
            .vals
            .get(n)
            .cloned()
            .ok_or_else(|| format!("enc: unbound {n}")),
        TExpr::Bin(op, a, b) => {
            let x = eval(a, env)?;
            let y = eval(b, env)?;
            v_bin(*op, &x, &y, env)
        }
        TExpr::Unary(u, a) => {
            let x = eval(a, env)?;
            match format!("{u:?}").to_lowercase().as_str() {
                "neg" => Ok(match x {
                    V::Ct(c, s) => V::Ct(c.iter().map(ct_neg).collect(), s),
                    V::Pt(p, s) => V::Pt(p.iter().map(|v| -v).collect(), s),
                }),
                _ => Err(format!("enc: unsupported unary {u:?}")),
            }
        }
        TExpr::Transpose(a) => {
            let x = eval(a, env)?;
            let sh = x.shape().to_vec();
            if sh.len() != 2 {
                return Err("enc: transpose needs a matrix".into());
            }
            let (d, k) = (sh[0] as usize, sh[1] as usize);
            match x {
                V::Ct(c, _) => {
                    let mut o = Vec::with_capacity(c.len());
                    for j in 0..k {
                        for i in 0..d {
                            o.push(c[i * k + j].clone());
                        }
                    }
                    Ok(V::Ct(o, vec![k as i64, d as i64]))
                }
                V::Pt(p, _) => {
                    let mut o = vec![0.0; p.len()];
                    for i in 0..d {
                        for j in 0..k {
                            o[j * d + i] = p[i * k + j];
                        }
                    }
                    Ok(V::Pt(o, vec![k as i64, d as i64]))
                }
            }
        }
        TExpr::Call(name, _, _) => Err(format!(
            "enc: builtin '{name}' is nonlinear — leveled enc circuits \
             support +,-,*,@ only"
        )),
        TExpr::Index(base, _) => {
            let x = eval(base, env)?;
            Ok(match x {
                V::Ct(mut c, mut s) => {
                    let v = vec![c.remove(0)];
                    if !s.is_empty() {
                        s.remove(0);
                    }
                    V::Ct(v, s)
                }
                V::Pt(p, mut s) => {
                    if !s.is_empty() {
                        s.remove(0);
                    }
                    V::Pt(vec![p.first().copied().unwrap_or(0.0)], s)
                }
            })
        }
        TExpr::Block(stmts, tail) => {
            for s in stmts {
                exec_stmt(s, env)?;
            }
            eval(tail, env)
        }
        other => Err(format!("enc: unsupported expr {other:?}")),
    }
}

fn exec_stmt(s: &TStmt, env: &mut Env) -> Result<(), String> {
    match s {
        TStmt::Let { name, expr } | TStmt::Assign { name, expr, .. } => {
            let v = eval(expr, env)?;
            env.vals.insert(name.clone(), v);
            Ok(())
        }
        TStmt::Swap { left, right } => {
            let a = env.vals.get(left).cloned();
            let b = env.vals.get(right).cloned();
            if let Some(v) = b {
                env.vals.insert(left.clone(), v);
            }
            if let Some(v) = a {
                env.vals.insert(right.clone(), v);
            }
            Ok(())
        }
        TStmt::Expr(e) => {
            eval(e, env)?;
            Ok(())
        }
        TStmt::StateUpdate { .. } | TStmt::ParamUpdate { .. } | TStmt::AdamUpdate { .. } => {
            Err("enc: state/param updates are host-side — keep them out of \
                 the encrypted stratum"
                .into())
        }
    }
}

// --- emit-time validity: reject nonlinear ops before the circuit exists ------

fn check_expr(t: &Typed) -> Result<(), String> {
    match &t.e {
        TExpr::Call(name, _, _) => Err(format!(
            "enc: '{name}' is nonlinear — enc fn supports linear ops only"
        )),
        TExpr::Bin(_, a, b) => {
            check_expr(a)?;
            check_expr(b)
        }
        TExpr::Unary(_, a) | TExpr::Transpose(a) | TExpr::Index(a, _) => check_expr(a),
        TExpr::Block(ss, tail) => {
            for s in ss {
                match s {
                    TStmt::Let { expr, .. } | TStmt::Assign { expr, .. } | TStmt::Expr(expr) => {
                        check_expr(expr)?
                    }
                    _ => {}
                }
            }
            check_expr(tail)
        }
        _ => Ok(()),
    }
}

// --- Backend plumbing ----------------------------------------------------------

/// Evaluate `tp` with caller-supplied ciphertexts — the enc×pcc seam.
/// `ct_ins` carries client-encrypted lanes per port (declared shape
/// comes from `tp.ios`); `pt_ins` carries the node's plaintext params.
/// Returns the still-encrypted output lanes — the caller decrypts, the
/// node never sees plaintext.
pub(crate) fn run_ciphertext(
    tp: &TProgram,
    ct_ins: &[(String, Vec<Ct>)],
    pt_ins: &[(String, Vec<f64>)],
) -> Result<Vec<Ct>, String> {
    let mut env = Env {
        vals: HashMap::new(),
    };
    let dims_of = |name: &str| -> Vec<i64> {
        tp.ios
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, _, t)| t.dims.clone())
            .or_else(|| {
                tp.params
                    .iter()
                    .find(|p| p.name == name)
                    .map(|p| p.ty.dims.clone())
            })
            .unwrap_or_default()
    };
    for (port, lanes) in ct_ins {
        env.vals
            .insert(port.clone(), V::Ct(lanes.clone(), dims_of(port)));
    }
    for (port, vals) in pt_ins {
        env.vals
            .insert(port.clone(), V::Pt(vals.clone(), dims_of(port)));
    }
    for (arg, port) in &tp.arg_ports {
        if let Some(v) = env.vals.get(port).cloned() {
            env.vals.insert(arg.clone(), v);
        }
    }
    for p in &tp.params {
        env.vals
            .entry(p.name.clone())
            .or_insert_with(|| V::Pt(vec![0.0; p.ty.numel() as usize], p.ty.dims.clone()));
    }
    for s in &tp.body {
        exec_stmt(s, &mut env)?;
    }
    match eval(&tp.ret.clone(), &mut env)? {
        V::Ct(cs, _) => Ok(cs),
        V::Pt(..) => Err("enc: program ended in plaintext domain".into()),
    }
}

struct EncExec {
    tp: TProgram,
    secret: u64,
    rng: u64,
    /// io ports that get encrypted on entry.
    enc_ports: Vec<String>,
}

impl BackendExec for EncExec {
    fn run(&mut self, writes: &[(String, Vec<f32>)]) -> Result<Vec<f32>, String> {
        let mut env = Env {
            vals: HashMap::new(),
        };
        // declared types per port for shapes
        let ty_of = |port: &str| -> Vec<i64> {
            for (n, _, t) in &self.tp.ios {
                if n == port {
                    return t.dims.clone();
                }
            }
            for p in &self.tp.params {
                if p.name == port {
                    return p.ty.dims.clone();
                }
            }
            Vec::new()
        };
        for (port, data) in writes {
            let sh = ty_of(port);
            if self.enc_ports.contains(port) {
                let cts = data
                    .iter()
                    .map(|m| lwe_enc(*m as f64, self.secret, &mut self.rng))
                    .collect();
                env.vals.insert(port.clone(), V::Ct(cts, sh));
            } else {
                // backend ABI stages params [N,K] — the declared type is
                // [K,N]; re-layout so the evaluator reads declared order.
                // states and io ride in declared order already.
                let is_param = self.tp.params.iter().any(|p| p.name == *port);
                let vals: Vec<f64> = if is_param && sh.len() == 2 {
                    let (n, k) = (sh[1] as usize, sh[0] as usize);
                    let mut host = vec![0.0; data.len()];
                    for a in 0..n {
                        for b in 0..k {
                            host[b * n + a] = data[a * k + b] as f64;
                        }
                    }
                    host
                } else {
                    data.iter().map(|v| *v as f64).collect()
                };
                env.vals.insert(port.clone(), V::Pt(vals, sh));
            }
        }
        for (arg, port) in &self.tp.arg_ports {
            if let Some(v) = env.vals.get(port).cloned() {
                env.vals.insert(arg.clone(), v);
            }
        }
        for p in &self.tp.params {
            env.vals
                .entry(p.name.clone())
                .or_insert_with(|| V::Pt(vec![0.0; p.ty.numel() as usize], p.ty.dims.clone()));
        }
        for s in &self.tp.body {
            exec_stmt(s, &mut env)?;
        }
        let out = eval(&self.tp.ret.clone(), &mut env)?;
        match out {
            V::Ct(cs, _) => Ok(cs.iter().map(|c| lwe_dec(c, self.secret) as f32).collect()),
            V::Pt(p, _) => Ok(p.iter().map(|v| *v as f32).collect()),
        }
    }
}

impl Backend for EncBackend {
    fn emit(
        &self,
        _src: &str,
        name: &str,
        tp: &TProgram,
        dir: &Path,
    ) -> Result<BackendEmit, String> {
        // emit-time rejection of nonlinear ops — a bad enc program
        // never reaches an engine.
        for s in &tp.body {
            match s {
                TStmt::Let { expr, .. } | TStmt::Assign { expr, .. } | TStmt::Expr(expr) => {
                    check_expr(expr)?
                }
                _ => {}
            }
        }
        check_expr(&tp.ret)?;
        let inputs: Vec<(String, CTy, bool)> = tp
            .ios
            .iter()
            .filter(|(_, d, _)| matches!(d, IoDir::In))
            .map(|(n, _, t)| (n.clone(), t.clone(), true))
            .collect();
        let trainable: Vec<(String, String, CTy, Option<String>)> = tp
            .params
            .iter()
            .map(|p| (p.name.clone(), p.name.clone(), p.ty.clone(), p.src.clone()))
            .collect();
        let states = tp.states.clone();
        let output = ("out".to_string(), tp.ret.ty.clone());
        let param_outs = trainable
            .iter()
            .map(|(n, p, t, _)| (n.clone(), p.clone(), t.clone()))
            .collect();
        let state_outs = states
            .iter()
            .map(|(_k, p, t)| (p.clone(), p.clone(), t.clone()))
            .collect();
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        std::fs::write(
            dir.join("enc_circuit.json"),
            serde_json::json!({
                "name": name,
                "scheme": "lwe/ct×pt level-1",
                "q_bits": 62, "scale_bits": 20,
                "io_encrypted": inputs.iter().map(|(n,_,_)| n.clone()).collect::<Vec<_>>(),
                "ops": tp.body.len(),
            })
            .to_string(),
        )
        .map_err(|e| e.to_string())?;
        Ok(BackendEmit {
            inputs,
            trainable,
            states,
            param_outs,
            state_outs,
            output,
            note: "lwe ct×pt level-1".into(),
            opaque: Box::new(()),
        })
    }

    fn load(
        &self,
        tp: &TProgram,
        _name: &str,
        _emit: &BackendEmit,
        _dir: &Path,
    ) -> Result<Box<dyn BackendExec>, String> {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x5eed);
        Ok(Box::new(EncExec {
            tp: tp.clone(),
            secret: (seed | 1) % 1000 + 1,
            rng: seed | 0x9e37_79b9,
            enc_ports: tp.arg_ports.iter().map(|(_, p)| p.clone()).collect(),
        }))
    }
}
