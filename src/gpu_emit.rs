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
//! device storage is f32 unless `CYO_HALF` is set: then `f16`-declared
//! params stage as real `half` buffers (bandwidth halves on
//! param-heavy kernels; eval already f16-quantizes param init, so
//! values are bit-identical). Reads wrap `float()`, writes round
//! through `half` — matching the f16 surface drivers write back to;
//! SGD updates gain ~1e-3 rounding drift vs the f32 eval env chain
//! (adam is exact — eval rounds adam writeback too). `grad(loss, w)`
//! mirroring `Evaluator::back` rule-for-rule, ending in `mse` like
//! eval's v1 scope — and `w op= s*grad` folds to per-param update
//! kernels dispatched before the main kernel (eval's stmt→ret
//! order). `adam(w, g, lr)` keeps eval's within-call semantics —
//! the param reads unchanged until `main` returns — by running the
//! raw-grad kernel stmt-position and the on-device moment update
//! (m/v/step in persistent zero-init scratch, fp16-rounded weight
//! writeback like `adam_step`) after the out kernel. State writes
//! (`s.f op= e`, one-hot scatters `s.f[p] op= e`) emit stmt-position
//! kernels against the resident state buffer — the kv-cache story —
//! and `Assign` lowers both ways: param assigns write the buffer at
//! stmt position; let-assigns shadow-bind the scope (eval's
//! env.insert, with `resolve_limit` keeping `a = a*2` honest).

use coyote::ast::{AssignOp, BinOp, DType, IoDir, Qual, UnOp};
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
    /// f32 params share one pooled device buffer (`__params_pool`);
    /// the runtime allocates it once and stages each param at its
    /// `param_off` element offset. `CYO_HALF` params stay per-arg.
    pub param_pool: bool,
    /// at least one kernel exceeded the per-signature buffer budget
    /// and was lowered through the `__CoyotePtrs` pointer table —
    /// the runtime must build the table buffer and bind it.
    pub ptrs: bool,
    pub output: (String, CTy),
    /// kernels to dispatch in order — the main program kernel is
    /// always LAST (eval runs stmts incl. updates before ret); each
    /// entry is (msl function name, dispatch shape).
    pub kernels: Vec<(String, Dispatch)>,
    /// per-kernel buffer binding — global buffer indices (0 = out,
    /// then `args` order) in the order that kernel's signature
    /// declares them. Each kernel only declares the buffers its
    /// body references, so programs with more than 31 total buffers
    /// fit inside Metal's per-kernel buffer limit.
    pub kernel_uses: Vec<Vec<u32>>,
    /// per-kernel descriptor tables for deduped kernels — parallel to
    /// `kernels`. When `kernel_uses[i]` contains the desc sentinel the
    /// runtime allocates one `device uint` buffer holding these
    /// values; the shared kernel function indexes `__ptrs->p` and
    /// `__params_pool` offsets through `__desc[slot]` instead of
    /// baked constants, so structurally identical kernels (every
    /// layer/head of a transformer) collapse to one MSL function.
    /// Empty for kernels on the direct-binding path.
    pub desc_templates: Vec<Vec<u32>>,
}

/// how a kernel maps threads: flat elementwise (`Linear`) or a
/// threadgroup-cooperative tiled matmul over a `t×t` tile
/// (`Tiled` — grid is n×m threads in t×t threadgroups).
#[derive(Debug, Clone, Copy)]
pub enum Dispatch {
    Linear(usize),
    /// one threadgroup of N threads — kernels that need threadgroup
    /// barriers to cover the whole problem (fused softmax/rmsnorm).
    Group(usize),
    /// N threadgroups of 64 — one group per output element (the
    /// cooperative-load matvec path).
    Gemv(usize),
    /// grid = ceil(n/(t·mt))×t × ceil(m/(t·mt))×t threads, t×t groups —
    /// each thread owns an mt×mt micro-tile of the output.
    Tiled {
        m: usize,
        n: usize,
        t: usize,
        mt: usize,
    },
    /// 64×64 blocks, 128 threads (4 simdgroups) each — grid is
    /// ceil(n/64) block-columns × ceil(m/64) block-rows.
    Simdgemm {
        m: usize,
        n: usize,
    },
}

#[derive(Debug, Clone)]
pub enum Arg {
    In(String),
    Param(String),
    State(String),
    /// materialized-subexpression buffer — allocated zeroed by the
    /// runtime, written by a prelude kernel, read like an input.
    Scratch(String, usize),
    /// an `f16`-declared param staged as a half buffer (`CYO_HALF`) —
    /// kernels load `float(b[i])` for fp32 compute; update kernels
    /// write `half(v)`, matching the surface round-trip eval drivers
    /// already perform.
    ParamHalf(String),
}

impl Arg {
    pub fn port(&self) -> &str {
        match self {
            Arg::In(p) | Arg::Param(p) | Arg::State(p) | Arg::Scratch(p, _) | Arg::ParamHalf(p) => {
                p
            }
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
    /// parallel to `scopes`, same push/pop/binding order — but entries
    /// keep the ORIGINAL let rhs even when `scopes` rebinds the name
    /// to a materialized scratch buffer. `gen`/`rhs_read` resolve
    /// through `scopes` (buffer reads = correct primal values);
    /// `back`/`can_grad` resolve through this (the expr DAG = correct
    /// gradient structure). Without it every reduction-valued let
    /// (matmul/softmax/norm outputs — i.e. the whole network) reads
    /// as an opaque buffer and `grad` reports "loss does not depend".
    orig_scopes: Vec<Vec<(String, Typed)>>,
    /// while resolving a let-bound expr at position i of scope s, only
    /// earlier bindings *of s* are visible — the evaluator binds
    /// eagerly in order; the emitter mirrors that lazily. This is what
    /// keeps inliner alias pairs (`__arg0 := w`, `w := __arg0`) and
    /// shadow-lets (`let x = x*2`) from recursing forever: the inner
    /// `w`/`x` refers to whatever was bound *before*, i.e. the buffer.
    /// Entries are (scope index, position) — a bound expr can open a
    /// nested Block, so the limit travels with its scope.
    /// Two stacks, one per concern: `resolve_limit` constrains the
    /// backward walk's structural recursion (`pos_of_orig`); adjoint
    /// closures regenerate primal values through `gen`, whose own
    /// inline-expansion recursion is guarded by `gen_limit` instead.
    /// Sharing one stack let deep backward limits hide later lets from
    /// closure-time `gen` calls (`unbound tensor f0` on real brains).
    resolve_limit: Vec<(usize, usize)>,
    gen_limit: Vec<(usize, usize)>,
    /// kernels queued in dispatch order — scratch kernels (materialized
    /// reductions) and update kernels (`w op= s*grad(loss, w)`) land
    /// here as they're discovered; the main out-kernel is appended
    /// last by `emit`, matching eval's stmt→ret order.
    emit_q: Vec<EK>,
    /// kernels dispatched AFTER the out kernel — `adam(w, g, lr)`:
    /// eval leaves the param untouched during the call (the driver
    /// applies moments between steps), so the on-device moment update
    /// must not precede the output read. Adam grad kernels still go
    /// in `emit_q` (grad is stmt-position, pre-ret).
    post_q: Vec<EK>,
    /// completed scratch kernels awaiting queue placement — inner
    /// scratches finish before their outer (nested bodies resolve
    /// while the outer body is still generating).
    scratch: Vec<EK>,
    /// expr pointer → scratch buffer name, dedup for non-named
    /// subexprs read repeatedly (let binds get buffer idents instead).
    mat_ptr: HashMap<usize, String>,
    /// scratch buffer creation order — sig/arg indices follow it.
    scratch_args: Vec<(String, usize)>,
    /// param name → numel, for update-kernel dispatch sizes.
    param_numel: HashMap<String, usize>,
    /// param name → element offset inside the shared `__params_pool`
    /// buffer. Metal caps a kernel signature at 31 `[[buffer(n)]]`
    /// slots; per-param args blew past it on real training graphs
    /// (one adjoint kernel referenced ~45). Pooling every f32 param
    /// into one device buffer collapses those references to a single
    /// slot — the per-param offset is baked into each index
    /// expression instead. `CYO_HALF` params keep per-param args
    /// (half storage can't share the f32 pool).
    param_off: HashMap<String, i64>,
    /// arg names (`b_w`) backed by `half` storage under `CYO_HALF` —
    /// every read site wraps `float(...)`, every write wraps `half(...)`.
    half_args: std::collections::HashSet<String>,
    lines: Vec<String>,
    tmp: usize,
}

/// a queued kernel: msl function name, dispatch shape, body lines,
/// and the param it may write (update kernels only — everyone else
/// is read-only over params).
struct EK {
    name: String,
    dispatch: Dispatch,
    body: Vec<String>,
    writable: Option<String>,
}

/// cooperative-matmul geometry: a TT×TT threadgroup covers a
/// TILE×TILE output block, each thread computing TM×TM outputs —
/// the register-blocking structure cuBLAS-style kernels use.
const TILE: usize = 32;
const TT: usize = 8;
const TM: usize = 4;
const BK: usize = 16;

/// tile geometry for autotuning sweeps — `CYO_TILE`/`CYO_TM` let a
/// run try other blockings without recompiling; defaults are the
/// measured best on dev silicon.
fn tile_geom() -> (usize, usize, usize) {
    let p = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    (p("CYO_TILE", TILE), p("CYO_TM", TM), bk_depth())
}

/// k-slice depth for the tiled matmul — `CYO_BK` overrides for
/// autotuning sweeps; 16 is the measured sweet spot on dev silicon.
fn bk_depth() -> usize {
    std::env::var("CYO_BK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(BK)
}
/// simdgroup path: SGTHREADS threads (4 simdgroups) per SGBLK×SGBLK
/// block — each simdgroup owns a 32×32 quadrant of 8×8 fragments.
const SGBLK: usize = 64;
const SGTHREADS: usize = 128;

/// shared device buffer every pooled (non-`CYO_HALF`) param reads
/// through — one `[[buffer(n)]]` slot covers all of them; per-param
/// element offsets live in `Ge::param_off`. The leading underscores
/// keep it collision-free against `b_{port}` arg names.
const POOL_ARG: &str = "__params_pool";

impl Ge {
    fn tmp(&mut self) -> String {
        self.tmp += 1;
        format!("t{}", self.tmp)
    }

    /// find `name` among let bindings, innermost scope first, honoring
    /// `resolve_limit` per scope. Returns (scope, position, expr).
    fn pos_in(
        &self,
        name: &str,
        limits: &[(usize, usize)],
    ) -> Option<(usize, usize, &Typed)> {
        for (si, s) in self.scopes.iter().enumerate().rev() {
            let lim = limits
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

    /// value-generation binding lookup — `gen`/`rhs_read`/`resolve_buf`
    /// resolve let names to their materialized form under `gen_limit`.
    fn pos_of(&self, name: &str) -> Option<(usize, usize, &Typed)> {
        self.pos_in(name, &self.gen_limit)
    }

    /// structural binding lookup for the backward walk — same
    /// (scope, pos) discipline as `pos_of`, but resolves under
    /// `resolve_limit` and returns the pre-scratch-rebind expr so
    /// `back`/`can_grad` see the DAG. Entries align 1:1 with
    /// `scopes`, so the (si, pos) found indexes `orig_scopes`.
    fn pos_of_orig(&self, name: &str) -> Option<(usize, usize, &Typed)> {
        let (si, p, _) = self.pos_in(name, &self.resolve_limit)?;
        self.orig_scopes
            .get(si)?
            .get(p)
            .map(|(_, t)| (si, p, t))
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

    /// buffer element read honoring half storage — `float(b[i])`
    /// under `CYO_HALF`, `b[i]` otherwise. `idx` is the raw index
    /// expression; callers compute it with the usual conventions.
    fn ld(&self, arg: &str, idx: &str) -> String {
        if self.half_args.contains(arg) {
            format!("float({arg}[{idx}])")
        } else {
            format!("{arg}[{idx}]")
        }
    }

    /// float4 vector load at `idx` (which must be 4-aligned) —
    /// half buffers load `half4` and widen.
    fn ld4(&self, arg: &str, idx: &str) -> String {
        if self.half_args.contains(arg) {
            format!("float4(*(device const half4*)({arg} + ({idx})))")
        } else {
            format!("*(device const float4*)({arg} + ({idx}))")
        }
    }

    /// pooled-param element offset for a flat index expression —
    /// `name` is the buffer's key in `bufs` (param name for params).
    /// Pooled params read `__params_pool[off + idx]`; everything else
    /// (ios, states, scratch, half params, let-bound names) passes
    /// through untouched.
    fn poff(&self, name: &str, flat: &str) -> String {
        match self.param_off.get(name) {
            Some(off) => format!("({off} + ({flat}))"),
            None => flat.to_string(),
        }
    }

    /// writable-param store honoring storage dtype — `op_s`
    /// is the MSL assignment operator (`=`,`+=`,`-=`,`*=`,`/=`); half
    /// buffers do an explicit float read-modify-write. Pooled params
    /// store at `__params_pool[param_off + gid]`.
    fn param_st(&self, param: &str, op_s: &str, v: &str) -> String {
        let (arg, idx) = match self.param_off.get(param) {
            Some(off) => (POOL_ARG.to_string(), format!("{off} + gid")),
            None => (format!("b_{param}"), "gid".to_string()),
        };
        if self.half_args.contains(&arg) {
            let cur = format!("float({arg}[{idx}])");
            let expr = match op_s {
                "=" => v.to_string(),
                "+=" => format!("({cur} + ({v}))"),
                "-=" => format!("({cur} - ({v}))"),
                "*=" => format!("({cur} * ({v}))"),
                _ => format!("({cur} / ({v}))"),
            };
            format!("{arg}[{idx}] = half({expr});")
        } else {
            format!("{arg}[{idx}] {op_s} ({v});")
        }
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
    /// `v` may be a compound expression — always parenthesize it or
    /// `a*b+c/d` precedence bites.
    fn coords_of(v: &str, dims: &[i64]) -> Vec<String> {
        dims.iter()
            .enumerate()
            .map(|(i, d)| {
                let stride: i64 = dims[i + 1..].iter().product();
                if stride == 1 {
                    format!("(({v})%{d})")
                } else {
                    format!("((({v})/{stride})%{d})")
                }
            })
            .collect()
    }

    /// true when `e`'s subtree contains a reduction — matmul or a
    /// row-loop builtin — so per-element inlining would recompute
    /// superlinear work per read. Such tensors materialize to a
    /// scratch buffer via a prelude kernel instead (what a CUDA
    /// programmer does by hand; here it's automatic).
    fn needs_scratch(e: &Typed) -> bool {
        match &e.e {
            TExpr::Lit(_) | TExpr::Int(_) | TExpr::Ident(_) | TExpr::Field(..) => false,
            TExpr::Unary(_, a) | TExpr::Transpose(a) => Self::needs_scratch(a),
            TExpr::Index(a, _) => Self::needs_scratch(a),
            TExpr::Bin(op, a, b) => {
                *op == BinOp::MatMul || Self::needs_scratch(a) || Self::needs_scratch(b)
            }
            TExpr::Call(n, args, _) => {
                matches!(
                    n.as_str(),
                    "softmax" | "rsoftmax" | "rmsnorm" | "gather" | "embed" | "outer"
                ) || args.iter().any(Self::needs_scratch)
            }
            TExpr::Block(stmts, ret) => {
                stmts.iter().any(|s| match s {
                    TStmt::Let { expr, .. } => Self::needs_scratch(expr),
                    _ => false,
                }) || Self::needs_scratch(ret)
            }
            TExpr::Grad { .. } => true,
        }
    }

    /// element-read entry: non-scalar subexprs containing reductions
    /// materialize to a scratch buffer (one prelude kernel, dispatched
    /// in discovery order) instead of recomputing per read. Kernel
    /// roots call `gen_raw` — they ARE the computation.
    fn gen(&mut self, e: &Typed, coords: &[String]) -> Result<String, String> {
        if e.ty.numel() > 1 && Self::needs_scratch(e) {
            let buf = self.scratch_of(e)?;
            let f = Self::flat(&e.ty.dims, coords, &Kind::Row);
            return Ok(format!("{buf}[{f}]"));
        }
        self.gen_raw(e, coords)
    }

    /// materialize `e` to `b__m{n}`: emit a prelude kernel writing the
    /// whole tensor row-major, return the buffer name. Nested
    /// scratchable subexprs register their own kernels first — the
    /// `scratch` queue's push order IS dependency order.
    fn scratch_of(&mut self, e: &Typed) -> Result<String, String> {
        let key = e as *const Typed as usize;
        if let Some(b) = self.mat_ptr.get(&key) {
            return Ok(b.clone());
        }
        let buf = format!("b__m{}", self.scratch_args.len());
        let dims = e.ty.dims.clone();
        let numel = e.ty.numel() as usize;
        self.scratch_args.push((buf.clone(), numel));
        self.mat_ptr.insert(key, buf.clone());
        // registered as a row-major buffer so rebound let-names
        // (Ident(b__mN)) and direct reads both hit storage.
        self.bufs
            .insert(buf.clone(), (buf.clone(), e.ty.clone(), Kind::Row));

        let saved = std::mem::take(&mut self.lines);
        let (body, dispatch) = match self.matmul_body(e, &buf) {
            Some((b, d)) => (b, d),
            None => match self.reduce_pair(e, &buf)? {
                Some(r) => r,
                None => {
                    let mut coords = Vec::new();
                    for (i, &d) in dims.iter().enumerate() {
                        let cname = format!("s_c{i}");
                        let stride: i64 = dims[i + 1..].iter().product();
                        if dims.len() == 1 {
                            self.lines.push(format!("uint {cname} = gid;"));
                        } else if stride == 1 {
                            self.lines.push(format!("uint {cname} = gid % {d};"));
                        } else {
                            self.lines
                                .push(format!("uint {cname} = (gid / {stride}) % {d};"));
                        }
                        coords.push(cname);
                    }
                    let v = self.gen_raw(e, &coords)?;
                    self.lines.push(format!("{buf}[gid] = ({v});"));
                    (std::mem::take(&mut self.lines), Dispatch::Linear(numel))
                }
            },
        };
        self.lines = saved;
        self.scratch.push(EK {
            name: format!("mat_{buf}"),
            dispatch,
            body,
            writable: None,
        });
        Ok(buf)
    }

    /// chase `let`-bound names to their underlying buffer — eagerly
    /// materialized binds rebind to `Ident(b__mN)`, so one hop usually
    /// suffices; loop guards deeper alias chains.
    /// returns (bufs key, arg, ty, kind) — the key is the buffer's
    /// name in `bufs` (param name for params), needed for pooled-param
    /// offset lookup via `poff`.
    fn resolve_buf(&self, e: &Typed) -> Option<(String, String, CTy, Kind)> {
        let mut cur = e;
        for _ in 0..8 {
            let TExpr::Ident(s) = &cur.e else { return None };
            if let Some((a, t, k)) = self.bufs.get(s) {
                return Some((s.clone(), a.clone(), t.clone(), *k));
            }
            match self.pos_of(s) {
                Some((_, _, b)) => cur = b,
                None => return None,
            }
        }
        None
    }

    /// If `e` is a plain `A @ B` over buffers, produce the operand
    /// index expressions both matmul templates share: `a_e(m, tk)`,
    /// `b_e(bn, tk)`, dims (m, k, n). Index formulas reuse eval's
    /// conventions: lhs row-major `a[m*K+k]`; named rhs flat
    /// `b[n*K+k]`; the one strided case (non-square param `.T`) reads
    /// `b[k*N+n]`.
    fn matmul_operands(&self, e: &Typed) -> Option<(String, String, usize, usize, usize)> {
        let TExpr::Bin(BinOp::MatMul, a, b) = &e.e else {
            return None;
        };
        if a.ty.dims.len() != 2 || b.ty.dims.len() != 2 {
            return None;
        }
        let (m_dim, k_dim, n_dim) = (
            a.ty.dims[0] as usize,
            a.ty.dims[1] as usize,
            b.ty.dims[1] as usize,
        );
        let (a_key, a_arg, _, _) = self.resolve_buf(a)?;
        let a_e = self.ld(&a_arg, &self.poff(&a_key, &format!("(m)*{k_dim}+(tk)")));
        let b_e = match &b.e {
            TExpr::Ident(_) => {
                let (b_key, arg, _, _) = self.resolve_buf(b)?;
                self.ld(&arg, &self.poff(&b_key, &format!("(bn)*{k_dim}+(tk)")))
            }
            TExpr::Transpose(inner) => {
                let (i_key, arg, ity, kind) = self.resolve_buf(inner)?;
                if matches!(kind, Kind::ParamT) && ity.dims[0] != ity.dims[1] {
                    self.ld(&arg, &self.poff(&i_key, &format!("(tk)*{n_dim}+(bn)")))
                } else {
                    self.ld(&arg, &self.poff(&i_key, &format!("(bn)*{k_dim}+(tk)")))
                }
            }
            _ => return None,
        };
        Some((a_e, b_e, m_dim, k_dim, n_dim))
    }

    /// simdgroup_matrix path — the tensor-core analog: 4 simdgroups
    /// per threadgroup, each owning a 32×32 quadrant of 8×8
    /// `simdgroup_float8x8` fragments inside a 64×64 block. As loads
    /// `[m][k]`; Bs is stored TRANSPOSED `[k][n]` so fragments feed
    /// `simdgroup_multiply_accumulate` directly. Edge blocks store
    /// frags to a Cs scratch and scalar-copy the in-bounds region —
    /// any M,N,K ≥ 8 works. k-order is preserved at 8-granularity;
    /// intra-frag reduction order is hardware-defined, so parity may
    /// drift to ~1e-4 (still honest, still gated).
    fn simd_body(&self, e: &Typed, out_buf: &str) -> Option<(Vec<String>, Dispatch)> {
        let (a_e, b_e, m_dim, k_dim, n_dim) = self.matmul_operands(e)?;
        if m_dim < 8 || n_dim < 8 || k_dim < 8 {
            return None;
        }
        let (blk, sgt, bk) = (SGBLK, SGTHREADS, BK);
        let a_e = a_e.replace("(m)", "(m0 + r)").replace("(tk)", "(t + c)");
        let b_e = b_e.replace("(bn)", "(n0 + r)").replace("(tk)", "(t + c)");
        let each = blk * bk / sgt;
        let mut l = Vec::new();
        l.push(format!("threadgroup float As[{blk}][{}];", bk + 1));
        l.push(format!("threadgroup float Bs[{bk}][{}];", blk + 1));
        l.push(format!("threadgroup float Cs[{blk}][{}];", blk + 1));
        l.push("uint lid = lid2.x;".into());
        l.push("uint sg = lid / 32;".into());
        l.push("uint sgr = sg / 2, sgc = sg % 2;".into());
        l.push(format!("uint m0 = tgid2.y*{blk};"));
        l.push(format!("uint n0 = tgid2.x*{blk};"));
        l.push("simdgroup_float8x8 acc[4][4];".into());
        l.push(format!("for (uint t = 0; t < {k_dim}; t += {bk}) {{"));
        l.push(format!("    for (uint i = 0; i < {each}; ++i) {{ uint idx = lid + i*{sgt}; uint r = idx/{bk}; uint c = idx%{bk};"));
        l.push(format!(
            "        As[r][c] = (m0 + r < {m_dim} && t + c < {k_dim}) ? ({a_e}) : 0.0f;"
        ));
        l.push(format!(
            "        Bs[c][r] = (n0 + r < {n_dim} && t + c < {k_dim}) ? ({b_e}) : 0.0f;"
        ));
        l.push("    }".into());
        l.push("    threadgroup_barrier(mem_flags::mem_threadgroup);".into());
        l.push("    simdgroup_float8x8 af, bf;".into());
        l.push(format!("    for (uint kk = 0; kk < {bk}; kk += 8) {{"));
        l.push("        for (uint i = 0; i < 4; ++i) {".into());
        l.push(format!(
            "            simdgroup_load(af, &As[sgr*32 + i*8][kk], {});",
            bk + 1
        ));
        l.push("            for (uint j = 0; j < 4; ++j) {".into());
        l.push(format!(
            "                simdgroup_load(bf, &Bs[kk][sgc*32 + j*8], {});",
            blk + 1
        ));
        l.push(
            "                if (t == 0 && kk == 0) simdgroup_multiply(acc[i][j], af, bf);".into(),
        );
        l.push(
            "                else simdgroup_multiply_accumulate(acc[i][j], af, bf, acc[i][j]);"
                .into(),
        );
        l.push("            }".into());
        l.push("        }".into());
        l.push("    }".into());
        l.push("    threadgroup_barrier(mem_flags::mem_threadgroup);".into());
        l.push("}".into());
        l.push("for (uint i = 0; i < 4; ++i) for (uint j = 0; j < 4; ++j)".into());
        l.push(format!(
            "    simdgroup_store(acc[i][j], &Cs[sgr*32 + i*8][sgc*32 + j*8], {});",
            blk + 1
        ));
        l.push("threadgroup_barrier(mem_flags::mem_threadgroup);".into());
        l.push(format!(
            "for (uint idx = lid; idx < {}; idx += {sgt}) {{ uint r = idx/{blk}, c = idx%{blk}; uint m = m0 + r, n = n0 + c; if (m < {m_dim} && n < {n_dim}) {out_buf}[(m)*{n_dim}+(n)] = Cs[r][c]; }}",
            blk * blk
        ));
        Some((l, Dispatch::Simdgemm { m: m_dim, n: n_dim }))
    }

    /// matmul template selection: cooperative-tile is the measured
    /// winner on this GPU (f32 simdgroup_matrix is ALU-emulated
    /// pre-M4); `CYO_SIMD=1` forces the simdgroup path for
    /// benchmarking on matrix-hardware silicon.
    fn matmul_body(&mut self, e: &Typed, out_buf: &str) -> Option<(Vec<String>, Dispatch)> {
        if std::env::var_os("CYO_SIMD").is_some() {
            if let Some(r) = self.simd_body(e, out_buf) {
                return Some(r);
            }
        }
        if std::env::var_os("CYO_NOGEMV").is_none() {
            if let Some(r) = self.gemv_body(e, out_buf) {
                return Some(r);
            }
        }
        self.tiled_body(e, out_buf)
    }

    /// Vector-lhs matmul (`x @ w`, `q @ kv.k.T`, `s @ kv.v`) — the
    /// decode-step shape. One thread per output walks a 4KB row:
    /// fully uncoalesced, ~13 GB/s effective on the real-layer probe.
    /// Two kernel shapes instead, both keeping eval's exact serial
    /// accumulation so parity stays bit-exact:
    ///
    /// rows — one threadgroup per output element: the whole group
    ///   coalesces the contraction row into threadgroup memory, then
    ///   a single thread serial-dots it (reads hit SRAM, not global).
    ///   Covers `x @ w` and `q @ kv.k.T` (contiguous-row contraction).
    ///
    /// cols — one threadgroup for ALL outputs: each thread owns a
    ///   strided slice of the output and accumulates serially over
    ///   the contraction dim; every read across the group is a
    ///   contiguous block. Covers `s @ kv.v` (contract over dim0) and
    ///   the strided non-square-param `.T` case.
    fn gemv_body(&mut self, e: &Typed, out_buf: &str) -> Option<(Vec<String>, Dispatch)> {
        let TExpr::Bin(BinOp::MatMul, a, b) = &e.e else {
            return None;
        };
        if a.ty.dims.len() != 1 || b.ty.dims.len() != 2 {
            return None;
        }
        let k = a.ty.dims[0];
        // threadgroup staging caps the row at 24KB of f32s.
        if k > 6144 {
            return None;
        }

        // classify the rhs: rows-style contiguous contraction vs
        // cols-style strided contraction (and which buffer/index it
        // reads).
        enum Shape {
            /// contiguous row per output: rhs[oidx*K + k]
            Rows {
                buf: String,
                stride: i64,
                /// pooled-param element offset, when `buf` is
                /// `__params_pool` — baked into every index expr.
                off: Option<i64>,
            },
            /// contract over rows, output over cols: rhs[t*D + oidx]
            Cols {
                buf: String,
                dim_t: i64,
                dim_d: i64,
                off: Option<i64>,
            },
        }
        let field_buf = |e: &Typed| -> Option<(String, CTy)> {
            if let TExpr::Field(bx, f) = &e.e {
                if let TExpr::Ident(s) = &bx.e {
                    let key = format!("{s}.{f}");
                    return self.bufs.get(&key).map(|(a, t, _)| (a.clone(), t.clone()));
                }
            }
            None
        };
        let shape = match &b.e {
            // `q @ kv.k.T` — out[t] = Σ_d x[d]·v[t*D+d]; v rows are
            // contiguous over the contraction dim d.
            TExpr::Transpose(i) if field_buf(i).is_some() => {
                let (buf, ty) = field_buf(i)?;
                Shape::Rows {
                    buf,
                    stride: ty.dims[1],
                    off: None,
                }
            }
            // `s @ kv.v` — out[d] = Σ_t x[t]·v[t*D+d] — contract over
            // the row index: strided output reads.
            TExpr::Field(..) => {
                let (buf, ty) = field_buf(b)?;
                Shape::Cols {
                    buf,
                    dim_t: ty.dims[0],
                    dim_d: ty.dims[1],
                    off: None,
                }
            }
            // non-square param .T keeps declared-order [K,N] storage:
            // out[j] = Σ_k x[k]·w[k*N+j] — strided output reads.
            TExpr::Transpose(i) if matches!(&i.e, TExpr::Ident(s) if matches!(self.bufs.get(s), Some((_, t, Kind::ParamT)) if t.dims[0] != t.dims[1])) =>
            {
                let (pk, buf, ty, _) = self.resolve_buf(i)?;
                Shape::Cols {
                    buf,
                    dim_t: ty.dims[0],
                    dim_d: ty.dims[1],
                    off: self.param_off.get(&pk).copied(),
                }
            }
            // `x @ w` (param/io/scratch) and square `.T` — the named
            // rhs reads flat [N,K]: out[j] = Σ_k x[k]·w[j*K+k].
            _ => {
                let (pk, buf, _, _) = self.resolve_buf(b)?;
                Shape::Rows {
                    buf,
                    stride: k,
                    off: self.param_off.get(&pk).copied(),
                }
            }
        };

        let mut l = Vec::new();
        match shape {
            Shape::Rows {
                buf,
                stride,
                off,
            } => {
                // one thread per output, serial-k accumulator —
                // eval's exact order — but the row is consumed as
                // float4s so each thread issues a quarter of the
                // load instructions. (The fully-tiled variant with
                // cooperative threadgroup staging measured 2.4x
                // slower on this silicon: per-thread sequential
                // rows prefetch well and the barriers expose
                // latency at every slice.)
                let n = b.ty.dims[1] as usize;
                if stride % 4 != 0 || k % 4 != 0 {
                    return None;
                }
                let x4: Option<Vec<String>> = (0..4)
                    .map(|i| self.gen(a, &[format!("t*4+{i}")]).ok())
                    .collect();
                let x4 = x4?;
                l.push("uint j = tgid*64 + lid;".into());
                l.push("float acc = 0.0f;".into());
                l.push(format!("for (uint t = 0; t < {}; ++t) {{", k / 4));
                l.push(format!("float4 xv = float4({});", x4.join(", ")));
                l.push(format!(
                    "float4 wv = {};",
                    self.ld4(
                        &buf,
                        &match off {
                            Some(o) => format!("({o} + (j*{stride}+t*4))"),
                            None => format!("j*{stride}+t*4"),
                        }
                    )
                ));
                l.push(
                    "acc += xv.x*wv.x; acc += xv.y*wv.y; acc += xv.z*wv.z; acc += xv.w*wv.w;"
                        .into(),
                );
                l.push("}".into());
                l.push(format!("if (j < {n}) {out_buf}[j] = acc;"));
                Some((l, Dispatch::Gemv(n)))
            }
            Shape::Cols {
                buf,
                dim_t,
                dim_d,
                off,
            } => {
                // contraction over the leading dim: out[d] =
                // Σ_t x[t]·v[t*D+d]. one thread per output — for a
                // fixed t the group's reads are already contiguous —
                // serial t keeps eval's accumulation order.
                l.push("uint d = tgid*64 + lid;".into());
                l.push("float acc = 0.0f;".into());
                l.push(format!("for (uint t = 0; t < {dim_t}; ++t) {{"));
                let xv = self.gen(a, &["t".to_string()]).ok()?;
                let bv = self.ld(
                    &buf,
                    &match off {
                        Some(o) => format!("({o} + (t*{dim_d}+d))"),
                        None => format!("t*{dim_d}+d"),
                    },
                );
                l.push(format!("acc += ({xv}) * ({bv});"));
                l.push("}".into());
                l.push(format!("if (d < {dim_d}) {out_buf}[d] = acc;"));
                Some((l, Dispatch::Gemv(dim_d as usize)))
            }
        }
    }

    /// softmax / rmsnorm — one threadgroup does the whole op. The
    /// order-free part (max) tree-reduces in parallel; the
    /// order-sensitive part (the sum) stays serial on thread 0 so the
    /// accumulation sequence is eval's — bit-exact parity preserved.
    /// exp goes parallel into a scratch buffer between the two, so
    /// the serial tail is plain adds. Without this fusion every
    /// element's thread re-ran the whole reduction — the O(n²)
    /// inlining trap that cost a 64×64 softmax ~0.18ms.
    fn reduce_pair(
        &mut self,
        e: &Typed,
        out_buf: &str,
    ) -> Result<Option<(Vec<String>, Dispatch)>, String> {
        const T: usize = 256;
        let TExpr::Call(name, args, _) = &e.e else {
            return Ok(None);
        };
        // rsoftmax is a different fusion shape: one thread per row,
        // serial inside the row in eval's exact accumulation order —
        // parallel across rows instead of one threadgroup for the
        // whole tensor, so attention scales with L not L².
        if name == "rsoftmax" {
            let x = &args[0];
            let (rows, cols) = (x.ty.dims[0], x.ty.dims[1]);
            let cs: Vec<String> = vec!["gid".into(), "j".into()];
            let saved = std::mem::take(&mut self.lines);
            self.lines.push("float m = -3.402823e38f;".into());
            self.lines
                .push(format!("for (uint j = 0; j < {cols}; ++j) {{"));
            let v = self.gen(x, &cs)?;
            self.lines.push(format!("m = fmax(m, {v});"));
            self.lines.push("}".into());
            self.lines.push("float s = 0.0f;".into());
            self.lines
                .push(format!("for (uint j = 0; j < {cols}; ++j) {{"));
            let v = self.gen(x, &cs)?;
            self.lines.push(format!(
                "float ex = exp({v} - m); {out_buf}[gid*{cols}+j] = ex; s += ex;"
            ));
            self.lines.push("}".into());
            self.lines.push(format!(
                "for (uint j = 0; j < {cols}; ++j) {out_buf}[gid*{cols}+j] /= s;"
            ));
            let body = std::mem::replace(&mut self.lines, saved);
            return Ok(Some((body, Dispatch::Linear(rows as usize))));
        }
        if !matches!(name.as_str(), "softmax" | "rmsnorm") {
            return Ok(None);
        }
        let x = &args[0];
        let dims = x.ty.dims.clone();
        let n: i64 = dims.iter().product();
        let cs = Self::coords_of(&"i".to_string(), &dims);

        let saved = std::mem::take(&mut self.lines);
        match name.as_str() {
            "softmax" => {
                // scratch holding exp(x - m) between the parallel exp
                // and the serial sum — same values eval sums, in the
                // same order.
                let eb = format!("b__e{}", self.scratch_args.len());
                self.scratch_args.push((eb.clone(), n as usize));
                self.lines.push(format!("threadgroup float tr[{T}];"));
                self.lines.push("threadgroup float ts;".into());
                // max is order-free — tree-reduce it.
                self.lines.push("float part = -3.402823e38f;".into());
                self.lines
                    .push(format!("for (uint i = lid; i < {n}; i += {T}) {{"));
                let v = self.gen(x, &cs)?;
                self.lines.push(format!("part = fmax(part, {v});"));
                self.lines.push("}".into());
                self.lines.push("tr[lid] = part;".into());
                self.lines
                    .push("threadgroup_barrier(mem_flags::mem_threadgroup);".into());
                self.lines
                    .push(format!("for (uint s = {}; s > 0; s >>= 1) {{", T / 2));
                self.lines
                    .push("if (lid < s) tr[lid] = fmax(tr[lid], tr[lid + s]);".into());
                self.lines
                    .push("threadgroup_barrier(mem_flags::mem_threadgroup);".into());
                self.lines.push("}".into());
                // exp is elementwise — parallel, then the
                // device-memory barrier makes every thread's writes
                // visible to the serial sum.
                self.lines
                    .push(format!("for (uint i = lid; i < {n}; i += {T}) {{"));
                let v = self.gen(x, &cs)?;
                self.lines.push(format!("{eb}[i] = exp({v} - tr[0]);"));
                self.lines.push("}".into());
                self.lines
                    .push("threadgroup_barrier(mem_flags::mem_device);".into());
                // the sum is order-sensitive — serial, eval order.
                self.lines.push(format!(
                    "if (lid == 0) {{ float s = 0.0f; for (uint i = 0; i < {n}; ++i) s += {eb}[i]; ts = s; }}"
                ));
                self.lines
                    .push("threadgroup_barrier(mem_flags::mem_threadgroup);".into());
                self.lines.push(format!(
                    "for (uint i = lid; i < {n}; i += {T}) {out_buf}[i] = {eb}[i] / ts;"
                ));
            }
            _ => {
                self.lines.push("threadgroup float tv;".into());
                self.lines.push("if (lid == 0) {".into());
                self.lines.push("float acc = 0.0f;".into());
                self.lines
                    .push(format!("for (uint i = 0; i < {n}; ++i) {{"));
                let v = self.gen(x, &cs)?;
                self.lines.push(format!("acc += ({v}) * ({v});"));
                self.lines.push("}".into());
                self.lines
                    .push(format!("tv = rsqrt(acc / float({n}) + 1e-5f);"));
                self.lines.push("}".into());
                self.lines
                    .push("threadgroup_barrier(mem_flags::mem_threadgroup);".into());
                let wv = self.gen(&args[1], &cs)?;
                self.lines
                    .push(format!("for (uint i = lid; i < {n}; i += {T}) {{"));
                let v = self.gen(x, &cs)?;
                self.lines
                    .push(format!("{out_buf}[i] = ({v}) * tv * ({wv});"));
                self.lines.push("}".into());
            }
        }
        let body = std::mem::replace(&mut self.lines, saved);
        Ok(Some((body, Dispatch::Group(T))))
    }

    /// If `e` is a plain `A @ B` over buffers, emit the cooperative-
    /// tile kernel: a TILE×TILE threadgroup loads A/B slices into
    /// threadgroup memory once and reuses them — the difference
    /// between a memory-bound ~150 GFLOP/s naive pass and a
    /// compute-bound kernel. Index formulas reuse eval's conventions:
    /// lhs row-major `a[m*K+k]`; named rhs flat `b[n*K+k]`; the one
    /// strided case (non-square param `.T`) reads `b[k*N+n]`. Bounds
    /// are guarded so any M,N,K shape works; accumulation order is
    /// still sequential-k, so parity vs eval is unchanged.
    fn tiled_body(&self, e: &Typed, out_buf: &str) -> Option<(Vec<String>, Dispatch)> {
        let (a_e, b_e, m_dim, k_dim, n_dim) = self.matmul_operands(e)?;
        let (tt, tm, t, bk) = {
            let (t2, m2, b2) = tile_geom();
            (TT, m2, t2, b2)
        };
        let mut l = Vec::new();
        // Register-blocked tile: TT×TT threads cover a TILE×TILE block,
        // each thread owns a TM×TM micro-tile. Cooperative loads fill
        // As/Bs once per BK-deep k-slice; the inner loop then does
        // TM+TM shared reads per TM² FMAs — the compute-per-byte ratio
        // that separates a tuned GEMM from a naive one. +1 pad on the
        // tile stride kills bank conflicts.
        l.push(format!("threadgroup float As[{t}][{}];", bk + 1));
        l.push(format!("threadgroup float Bs[{t}][{}];", bk + 1));
        l.push(format!("uint lf = lid2.y*{tt} + lid2.x;"));
        // block origin = threadgroup origin × micro-tile: each thread
        // owns TM outputs, so the tgid offset scales by TM.
        l.push(format!("uint m0 = (gid2.y - lid2.y)*{tm};"));
        l.push(format!("uint n0 = (gid2.x - lid2.x)*{tm};"));
        l.push(format!("float acc[{tm}][{tm}];"));
        l.push(format!(
            "for (uint i = 0; i < {tm}; ++i) for (uint j = 0; j < {tm}; ++j) acc[i][j] = 0.0f;"
        ));
        l.push(format!("for (uint t = 0; t < {k_dim}; t += {bk}) {{"));
        // cooperative load: TILE×BK elems per operand over TT² threads
        let each = t * bk / (tt * tt);
        l.push(format!(
            "    for (uint i = 0; i < {each}; ++i) {{ uint idx = lf + i*{ttsq}; uint r = idx/{bk}; uint c = idx%{bk};",
            ttsq = tt * tt
        ));
        l.push(format!(
            "        As[r][c] = (m0 + r < {m_dim} && t + c < {k_dim}) ? ({}) : 0.0f;",
            a_e.replace("(m)", "(m0 + r)").replace("(tk)", "(t + c)")
        ));
        l.push(format!(
            "        Bs[r][c] = (n0 + r < {n_dim} && t + c < {k_dim}) ? ({}) : 0.0f;",
            b_e.replace("(bn)", "(n0 + r)").replace("(tk)", "(t + c)")
        ));
        l.push("    }".into());
        l.push("    threadgroup_barrier(mem_flags::mem_threadgroup);".into());
        l.push(format!("    for (uint kk = 0; kk < {bk}; ++kk) {{"));
        l.push(format!(
            "        float av[{tm}], bv[{tm}]; for (uint i = 0; i < {tm}; ++i) {{ av[i] = As[lid2.y*{tm}+i][kk]; bv[i] = Bs[lid2.x*{tm}+i][kk]; }}"
        ));
        l.push(format!(
            "        for (uint i = 0; i < {tm}; ++i) for (uint j = 0; j < {tm}; ++j) acc[i][j] += av[i]*bv[j];"
        ));
        l.push("    }".into());
        l.push("    threadgroup_barrier(mem_flags::mem_threadgroup);".into());
        l.push("}".into());
        l.push(format!(
            "for (uint i = 0; i < {tm}; ++i) for (uint j = 0; j < {tm}; ++j) {{ uint m = m0 + lid2.y*{tm} + i; uint n = n0 + lid2.x*{tm} + j; if (m < {m_dim} && n < {n_dim}) {out_buf}[(m)*{n_dim}+(n)] = acc[i][j]; }}"
        ));
        Some((
            l,
            Dispatch::Tiled {
                m: m_dim,
                n: n_dim,
                t: tt,
                mt: tm,
            },
        ))
    }

    fn gen_raw(&mut self, e: &Typed, coords: &[String]) -> Result<String, String> {
        match &e.e {
            TExpr::Lit(v) => Ok(format!("float({v:e})")),
            TExpr::Int(i) => Ok(format!("float({i})")),
            TExpr::Ident(s) => {
                if let Some((si, pos, b)) = self.pos_of(s) {
                    let b = b.clone();
                    self.gen_limit.push((si, pos));
                    let r = self.gen(&b, coords);
                    self.gen_limit.pop();
                    return r;
                }
                let Some((arg, ty, _kind)) = self.bufs.get(s) else {
                    if std::env::var("COYOTE_GPU_DEBUG").is_ok() {
                        eprintln!("backtrace:\n{}", std::backtrace::Backtrace::force_capture());
                    }
                    return Err(format!("gpu emit: unbound tensor {s}"));
                };
                // params too read row-major over declared dims — their
                // [N,K] storage IS the value (matches the evaluator);
                // the transposed read only applies in matmul rhs.
                let flat = Self::flat(&ty.dims, coords, &Kind::Row);
                Ok(self.ld(&arg, &self.poff(s, &flat)))
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
                let rv = self.rhs_read(b, &rc, k)?;
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
                // eval's 2-D `base[r]` is a one-hot gather under the
                // rhs storage convention — `out[j] = base[j*D0 + r]` —
                // a column read over flat storage, NOT a row slice.
                if base.ty.dims.len() == 2 && idxs.len() == 1 {
                    if let TIndex::Lit(r) = idxs[0] {
                        let Some(j) = coords.first() else {
                            return Err("gpu emit: index result needs a free coord".into());
                        };
                        let d0 = base.ty.dims[0];
                        let (pk, arg) = match self.resolve_buf(base) {
                            Some((k, a, _, _)) => (Some(k), a),
                            None => (None, self.scratch_of(base)?),
                        };
                        let idx = format!("({j})*{d0} + {r}");
                        let idx = match &pk {
                            Some(k) => self.poff(k, &idx),
                            None => idx,
                        };
                        return Ok(self.ld(&arg, &idx));
                    }
                }
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
                if std::env::var("COYOTE_GPU_DEBUG").is_ok() {
                    eprintln!("gen_raw Block: {} stmts", stmts.len());
                }
                self.scopes.push(Vec::new());
                self.orig_scopes.push(Vec::new());
                for s in stmts {
                    self.stmt(s)?;
                }
                let v = self.gen(last, coords);
                self.scopes.pop();
                self.orig_scopes.pop();
                v
            }
            TExpr::Call(name, args, _) => self.call(name, args, coords),
            TExpr::Grad { loss, wrt } => {
                if !self.is_param(wrt) {
                    return Err(format!("gpu emit: grad target {wrt} is not a param"));
                }
                // out flat position → grads[wrt] element. The leaf rule
                // owns the flat-order convention ([N,K] for matmul rhs,
                // declared order elsewhere) — same as eval's acc.
                let f = Self::flat(&e.ty.dims, coords, &Kind::Row);
                match self.grad_elem(loss, wrt, &f)? {
                    Some(v) => Ok(v),
                    None => Err(format!("gpu emit: loss does not depend on {wrt}")),
                }
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
                        self.ld(&arg, &self.poff(s, &f))
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

    /// element of a matmul rhs `b` at logical coords `rc` ([contract,
    /// out]). eval reads EVERY named rhs's storage flat as
    /// `w[n*K + k]` — params are physically [N,K], io/state/let
    /// buffers get the same linear reinterpret. `p.T` non-square is
    /// the one real storage swap; square-param `.T` is a no-op.
    /// State fields keep their own logical convention. A materialized
    /// rhs resolves to the element at flat index n*K+k of its
    /// declared [K,N] view.
    fn rhs_read(&mut self, b: &Typed, rc: &[String], k: i64) -> Result<String, String> {
        let kf = if rc.len() > 1 {
            format!("({n})*{k}+({kk})", n = rc[1], k = k, kk = rc[0])
        } else {
            rc[0].clone()
        };
        match &b.e {
            TExpr::Field(..) => self.gen(b, rc),
            TExpr::Transpose(i) if matches!(&i.e, TExpr::Field(..)) => self.gen(b, rc),
            TExpr::Transpose(i) if matches!(&i.e, TExpr::Ident(s) if self.bufs.contains_key(s)) => {
                let (arg, ty, kind, pk) = match &i.e {
                    TExpr::Ident(s) => {
                        let (a, t, k) = self.bufs.get(s).unwrap().clone();
                        (a, t, k, s.clone())
                    }
                    _ => unreachable!(),
                };
                if matches!(kind, Kind::ParamT) && ty.dims[0] != ty.dims[1] {
                    Ok(self.ld(
                        &arg,
                        &self.poff(&pk, &Self::flat(&b.ty.dims, rc, &Kind::Row)),
                    ))
                } else {
                    Ok(self.ld(&arg, &self.poff(&pk, &kf)))
                }
            }
            TExpr::Transpose(i) => {
                let cs = Self::coords_of(&kf, &i.ty.dims);
                self.gen(i, &cs)
            }
            TExpr::Ident(s) if self.bufs.contains_key(s) => {
                let (arg, _, _) = self.bufs.get(s).unwrap().clone();
                Ok(self.ld(&arg, &self.poff(s, &kf)))
            }
            _ => {
                let cs = Self::coords_of(&kf, &b.ty.dims);
                self.gen(b, &cs)
            }
        }
    }

    fn is_param(&self, s: &str) -> bool {
        matches!(self.bufs.get(s), Some((_, _, Kind::ParamT)))
    }

    /// does `wrt` appear anywhere under `e` (through lets)? Mirrors
    /// the evaluator's `needs_grad` — prunes dead branches so the
    /// backward pass never emits code for them.
    fn can_grad(&self, e: &Typed, wrt: &str) -> bool {
        self.can_grad_d(e, wrt, 0)
    }

    fn can_grad_d(&self, e: &Typed, wrt: &str, depth: usize) -> bool {
        if depth > 32 {
            return false;
        }
        match &e.e {
            TExpr::Ident(s) => {
                (s == wrt && self.is_param(s))
                    || self
                        .pos_of_orig(s)
                        .map(|(_, _, b)| self.can_grad_d(b, wrt, depth + 1))
                        .unwrap_or(false)
            }
            TExpr::Unary(_, a) | TExpr::Transpose(a) => self.can_grad_d(a, wrt, depth + 1),
            TExpr::Index(a, _) | TExpr::Field(a, _) => self.can_grad_d(a, wrt, depth + 1),
            TExpr::Bin(_, a, b) => {
                self.can_grad_d(a, wrt, depth + 1) || self.can_grad_d(b, wrt, depth + 1)
            }
            TExpr::Call(_, args, named) => {
                args.iter().any(|a| self.can_grad_d(a, wrt, depth + 1))
                    || named
                        .iter()
                        .any(|(_, a)| self.can_grad_d(a, wrt, depth + 1))
            }
            TExpr::Block(stmts, ret) => {
                stmts.iter().any(|s| match s {
                    TStmt::Let { expr, .. }
                    | TStmt::Assign { expr, .. }
                    | TStmt::StateUpdate { expr, .. }
                    | TStmt::Expr(expr) => self.can_grad_d(expr, wrt, depth + 1),
                    TStmt::ParamUpdate { grad, .. } | TStmt::AdamUpdate { grad, .. } => {
                        self.can_grad_d(grad, wrt, depth + 1)
                    }
                }) || self.can_grad_d(ret, wrt, depth + 1)
            }
            _ => false,
        }
    }

    /// `grad(loss, wrt)` — the seed: loss must terminate in `mse`
    /// (same v1 scope as eval). Returns the MSL expression for the
    /// grads flat position `f`, or None when `wrt` is unreachable.
    fn grad_elem(&mut self, loss: &Typed, wrt: &str, f: &str) -> Result<Option<String>, String> {
        self.scopes.push(Vec::new());
        self.orig_scopes.push(Vec::new());
        let r = (|| {
            let terminal = match &loss.e {
                TExpr::Block(stmts, ret) => {
                    for s in stmts {
                        match s {
                            TStmt::Let { .. } => self.stmt(s)?,
                            // same rule as `back`'s block walk — state
                            // writes/expr statements are primal effects,
                            // not adjoint paths.
                            TStmt::StateUpdate { .. } | TStmt::Expr(_) => {}
                            _ => {
                                return Err(
                                    "gpu emit: non-let statement inside grad loss".into()
                                )
                            }
                        }
                    }
                    (**ret).clone()
                }
                _ => loss.clone(),
            };
            match &terminal.e {
                TExpr::Call(name, args, _) if name == "mse" => {
                    let n = args[0].ty.numel().max(1);
                    let (parg, targ) = (args[0].clone(), args[1].clone());
                    let mut seed = move |em: &mut Ge, c: &[String]| -> Result<String, String> {
                        let p = em.gen(&parg, c)?;
                        let t = em.gen(&targ, c)?;
                        Ok(format!("(2.0f * (({p}) - ({t})) / float({n}))"))
                    };
                    self.back(&args[0], wrt, f, &mut seed)
                }
                _ => Err("gpu emit: grad v1 losses must end in mse".into()),
            }
        })();
        self.scopes.pop();
        self.orig_scopes.pop();
        r
    }

    /// reverse-mode adjoint — mirrors `Evaluator::back` rule-for-rule.
    /// `adj` produces the downstream adjoint element at logical coords
    /// of `e`'s value; `f` is the flat position inside grads[wrt]
    /// (the leaf rule owns the storage-layout convention). Returns
    /// Some(expr) for the contribution when this subtree reaches wrt.
    fn back(
        &mut self,
        e: &Typed,
        wrt: &str,
        f: &str,
        adj: &mut dyn FnMut(&mut Ge, &[String]) -> Result<String, String>,
    ) -> Result<Option<String>, String> {
        fn add(a: Option<String>, b: Option<String>) -> Option<String> {
            match (a, b) {
                (Some(x), Some(y)) => Some(format!("(({x}) + ({y}))")),
                (x, None) => x,
                (None, y) => y,
            }
        }
        match &e.e {
            TExpr::Ident(s) => {
                if s == wrt && self.is_param(s) {
                    // elementwise param leaf — grads flat order is the
                    // declared-dims order here.
                    let cs = Self::coords_of(f, &e.ty.dims);
                    return adj(self, &cs).map(Some);
                }
                match self.pos_of_orig(s) {
                    Some((si, p, b)) => {
                        let b = b.clone();
                        self.resolve_limit.push((si, p));
                        let r = self.back(&b, wrt, f, adj);
                        self.resolve_limit.pop();
                        r
                    }
                    None => Ok(None),
                }
            }
            TExpr::Lit(_) | TExpr::Int(_) => Ok(None),
            TExpr::Unary(UnOp::Neg, a) => self.back(a, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                Ok(format!("(-({}))", adj(em, c)?))
            }),
            TExpr::Unary(UnOp::Not, _) => Err("gpu grad: `!` is host-stratum".into()),
            TExpr::Bin(BinOp::Add, a, b) => {
                let ra = self.back(a, wrt, f, &mut *adj)?;
                let rb = self.back(b, wrt, f, adj)?;
                Ok(add(ra, rb))
            }
            TExpr::Bin(BinOp::Sub, a, b) => {
                let ra = self.back(a, wrt, f, &mut *adj)?;
                let rb = self.back(b, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                    Ok(format!("(-({}))", adj(em, c)?))
                })?;
                Ok(add(ra, rb))
            }
            TExpr::Bin(BinOp::Mul, a, b) => {
                let ra = if self.can_grad(a, wrt) {
                    self.back(a, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                        let d = adj(em, c)?;
                        let bv = em.gen(b, c)?;
                        Ok(format!("(({d}) * ({bv}))"))
                    })?
                } else {
                    None
                };
                let rb = if self.can_grad(b, wrt) {
                    self.back(b, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                        let d = adj(em, c)?;
                        let av = em.gen(a, c)?;
                        Ok(format!("(({d}) * ({av}))"))
                    })?
                } else {
                    None
                };
                Ok(add(ra, rb))
            }
            TExpr::Bin(BinOp::Div, a, b) => {
                if let TExpr::Lit(s) = b.e {
                    self.back(a, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                        Ok(format!("(({}) / {}f)", adj(em, c)?, s))
                    })
                } else {
                    let ra = if self.can_grad(a, wrt) {
                        self.back(a, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                            let d = adj(em, c)?;
                            let bv = em.gen(b, c)?;
                            Ok(format!("(({d}) / ({bv}))"))
                        })?
                    } else {
                        None
                    };
                    let rb = if self.can_grad(b, wrt) {
                        self.back(b, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                            let d = adj(em, c)?;
                            let av = em.gen(a, c)?;
                            let bv = em.gen(b, c)?;
                            Ok(format!("(-(({d}) * ({av})) / (({bv}) * ({bv})))"))
                        })?
                    } else {
                        None
                    };
                    Ok(add(ra, rb))
                }
            }
            TExpr::Bin(BinOp::MatMul, a, b) => {
                // state-field rhs (`q @ kv.k.T`, `s @ kv.v`) is a
                // constant for this step — eval reads its primal
                // value and routes the adjoint into the lhs only.
                // `rhs_read` already emits those reads, and the
                // param-weight branch can't fire (fields aren't
                // params), so the generic path below IS the M3b
                // semantics: dx_a = adj · bᵀ, no db.
                let kd = *a.ty.dims.last().ok_or("gpu grad: matmul of scalars")? as usize;
                let nn = b.ty.dims.get(1).copied().unwrap_or(1) as usize;
                let pname = match &b.e {
                    TExpr::Ident(s) => Some(s.clone()),
                    TExpr::Transpose(i) => match &i.e {
                        TExpr::Ident(s) => Some(s.clone()),
                        _ => None,
                    },
                    _ => None,
                };
                let mut out = None;
                let m_batch = if a.ty.dims.len() == 2 {
                    a.ty.dims[0] as usize
                } else {
                    1
                };
                if let Some(p) = &pname {
                    if p == wrt && self.is_param(p) {
                        // dW[j,i] = Σ_br adj[br,j] · x[br,i] — [N,K]
                        // storage; batch rows accumulate into the
                        // shared weight.
                        let sc = Self::coords_of(f, &[nn as i64, kd as i64]);
                        if m_batch == 1 {
                            let av = adj(self, &sc[..1])?;
                            let xv = self.gen(a, &sc[1..])?;
                            out = Some(format!("(({av}) * ({xv}))"));
                        } else {
                            let acc = self.tmp();
                            let br = self.tmp();
                            self.lines.push(format!("float {acc} = 0.0f;"));
                            self.lines
                                .push(format!("for (int {br} = 0; {br} < {m_batch}; ++{br}) {{"));
                            let av = adj(self, &[br.clone(), sc[0].clone()])?;
                            let xv = self.gen(a, &[br.clone(), sc[1].clone()])?;
                            self.lines.push(format!("{acc} += ({av}) * ({xv});"));
                            self.lines.push("}".into());
                            out = Some(acc);
                        }
                    }
                }
                if self.can_grad(a, wrt) {
                    let b2 = (**b).clone();
                    let batched = a.ty.dims.len() == 2;
                    let ra = self.back(a, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                        // dx[br,i] = Σ_j adj[br,j] · w[j,i] — vector
                        // lhs is the br=0 case with 1-D coords.
                        let acc = em.tmp();
                        let j = em.tmp();
                        em.lines.push(format!("float {acc} = 0.0f;"));
                        em.lines
                            .push(format!("for (int {j} = 0; {j} < {nn}; ++{j}) {{"));
                        let av = if batched {
                            adj(em, &[c[0].clone(), j.clone()])?
                        } else {
                            adj(em, &[j.clone()])?
                        };
                        let kc = if batched { c[1].clone() } else { c[0].clone() };
                        let wv = em.rhs_read(&b2, &[kc, j.clone()], kd as i64)?;
                        em.lines.push(format!("{acc} += ({av}) * ({wv});"));
                        em.lines.push("}".into());
                        Ok(acc)
                    })?;
                    out = add(out, ra);
                }
                Ok(out)
            }
            TExpr::Call(name, args, _) => {
                let x = args[0].clone();
                let e2 = e.clone();
                let n = e.ty.numel().max(1) as usize;
                match name.as_str() {
                    "sigmoid" => self.back(&x, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                        let d = adj(em, c)?;
                        let h = em.gen(&e2, c)?;
                        Ok(format!("({d}) * ({h}) * (1.0f - ({h}))"))
                    }),
                    "tanh" => self.back(&x, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                        let d = adj(em, c)?;
                        let t = em.gen(&e2, c)?;
                        Ok(format!("({d}) * (1.0f - ({t}) * ({t}))"))
                    }),
                    "relu" => self.back(&x, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                        let d = adj(em, c)?;
                        let xv = em.gen(&x, c)?;
                        Ok(format!("(({xv}) > 0.0f ? ({d}) : 0.0f)"))
                    }),
                    "silu" => self.back(&x, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                        let d = adj(em, c)?;
                        let xv = em.gen(&x, c)?;
                        Ok(format!(
                            "({d}) * (1.0f / (1.0f + exp(-({xv})))) * (1.0f + ({xv}) * (1.0f - (1.0f / (1.0f + exp(-({xv}))))))"
                        ))
                    }),
                    "gelu" => self.back(&x, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                        let d = adj(em, c)?;
                        let xv = em.gen(&x, c)?;
                        // u = a*(v + b*v³); t = tanh(u)
                        // dg = 0.5(1+t) + 0.5*v*(1-t²)*a*(1+3bv²)
                        Ok(format!(
                            "({d}) * (0.5f * (1.0f + tanh(0.7978845608028654f * (({xv}) + 0.044715f * ({xv}) * ({xv}) * ({xv})))) + 0.5f * ({xv}) * (1.0f - pow(tanh(0.7978845608028654f * (({xv}) + 0.044715f * ({xv}) * ({xv}) * ({xv}))), 2.0f)) * 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * ({xv}) * ({xv})))"
                        ))
                    }),
                    "softmax" => {
                        // dx_i = s_i · (adj_i − Σ_j adj_j·s_j)
                        let es = e2.clone();
                        self.back(&x, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                            let dot = em.tmp();
                            let j = em.tmp();
                            em.lines.push(format!("float {dot} = 0.0f;"));
                            em.lines
                                .push(format!("for (int {j} = 0; {j} < {n}; ++{j}) {{"));
                            let av = adj(em, &[j.clone()])?;
                            let sv = em.gen(&es, &[j.clone()])?;
                            em.lines.push(format!("{dot} += ({av}) * ({sv});"));
                            em.lines.push("}".into());
                            let d = adj(em, c)?;
                            let s = em.gen(&es, c)?;
                            Ok(format!("({s}) * (({d}) - {dot})"))
                        })
                    }
                    "rsoftmax" => {
                        // per-row: dx[r,j] = s[r,j]·(adj[r,j] − Σ_k adj[r,k]·s[r,k])
                        let es = e2.clone();
                        let cols = x.ty.dims[1];
                        self.back(&x, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                            let dot = em.tmp();
                            let k = em.tmp();
                            em.lines.push(format!("float {dot} = 0.0f;"));
                            em.lines
                                .push(format!("for (int {k} = 0; {k} < {cols}; ++{k}) {{"));
                            let av = adj(em, &[c[0].clone(), k.clone()])?;
                            let sv = em.gen(&es, &[c[0].clone(), k.clone()])?;
                            em.lines.push(format!("{dot} += ({av}) * ({sv});"));
                            em.lines.push("}".into());
                            let d = adj(em, c)?;
                            let s = em.gen(&es, c)?;
                            Ok(format!("({s}) * (({d}) - {dot})"))
                        })
                    }
                    "rmsnorm" => {
                        // eval: dw = adj·xhat; dx = r·(adj·w − xhat·mean(g·xhat))
                        let w2 = args[1].clone();
                        let ms = self.tmp();
                        let ix = self.tmp();
                        self.lines.push(format!("float {ms} = 0.0f;"));
                        self.lines
                            .push(format!("for (int {ix} = 0; {ix} < {n}; ++{ix}) {{"));
                        let xv = self.gen(&x, &[ix.clone()])?;
                        self.lines.push(format!("{ms} += ({xv}) * ({xv});"));
                        self.lines.push("}".into());
                        let rvar = self.tmp();
                        self.lines.push(format!(
                            "float {rvar} = rsqrt({ms} / float({n}) + 1e-5f);"
                        ));
                        let mut out = None;
                        if let TExpr::Ident(wp) = &args[1].e {
                            if wp == wrt && self.is_param(wp) {
                                let cs = Self::coords_of(f, &e.ty.dims);
                                let d = adj(self, &cs)?;
                                let xv = self.gen(&x, &cs)?;
                                out = Some(format!("(({d}) * ({xv}) * {rvar})"));
                            }
                        }
                        if self.can_grad(&x, wrt) {
                            let (w2c, xc) = (w2.clone(), x.clone());
                            let rv = rvar.clone();
                            let rb = self.back(&x, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                                // mean_gx = Σ (adj·w·xhat) / n
                                let mg = em.tmp();
                                let j = em.tmp();
                                em.lines.push(format!("float {mg} = 0.0f;"));
                                em.lines.push(format!(
                                    "for (int {j} = 0; {j} < {n}; ++{j}) {{"
                                ));
                                let av = adj(em, &[j.clone()])?;
                                let wv = em.gen(&w2c, &[j.clone()])?;
                                let xv = em.gen(&xc, &[j.clone()])?;
                                em.lines.push(format!(
                                    "{mg} += ({av}) * ({wv}) * ({xv}) * {rv};"
                                ));
                                em.lines.push("}".into());
                                let d = adj(em, c)?;
                                let wv = em.gen(&w2c, c)?;
                                let xv = em.gen(&xc, c)?;
                                Ok(format!(
                                    "{rv} * (({d}) * ({wv}) - ({xv}) * {rv} * ({mg} / float({n})))"
                                ))
                            })?;
                            out = add(out, rb);
                        }
                        Ok(out)
                    }
                    "concat" => {
                        let k1 = args[0].ty.dims[0];
                        let ra = if self.can_grad(&args[0], wrt) {
                            self.back(&args[0], wrt, f, &mut *adj)?
                        } else {
                            None
                        };
                        let rb = if self.can_grad(&args[1], wrt) {
                            self.back(&args[1], wrt, f, &mut |em: &mut Ge, c: &[String]| {
                                adj(em, &[format!("({}) + {k1}", c[0])])
                            })?
                        } else {
                            None
                        };
                        Ok(add(ra, rb))
                    }
                    "gather" | "embed" => {
                        // eval synthesizes `idx @ tab` and backs that.
                        let synth = Typed {
                            e: TExpr::Bin(
                                BinOp::MatMul,
                                Box::new(args[1].clone()),
                                Box::new(args[0].clone()),
                            ),
                            ty: e.ty.clone(),
                        };
                        self.back(&synth, wrt, f, adj)
                    }
                    "mse" => {
                        let (p2, t2) = (args[0].clone(), args[1].clone());
                        self.back(&args[0], wrt, f, &mut |em: &mut Ge, c: &[String]| {
                            let d = adj(em, c)?;
                            let p = em.gen(&p2, c)?;
                            let t = em.gen(&t2, c)?;
                            Ok(format!("(({d}) * 2.0f * (({p}) - ({t})) / float({n}))"))
                        })
                    }
                    _ => Err(format!("gpu emit: no grad rule for {name}")),
                }
            }
            TExpr::Transpose(inner) => {
                if !self.can_grad(inner, wrt) {
                    return Ok(None);
                }
                // adjoint of a transpose is the transpose of the adjoint
                self.back(inner, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                    let mut sw = c.to_vec();
                    sw.swap(0, 1);
                    adj(em, &sw)
                })
            }
            TExpr::Index(base, idxs) => {
                if !self.can_grad(base, wrt) {
                    return Ok(None);
                }
                if base.ty.dims.len() == 1 && idxs.len() == 1 {
                    if let TIndex::Range(lo, hi) = idxs[0] {
                        let (lo2, hi2) = (lo, hi);
                        return self.back(base, wrt, f, &mut |em: &mut Ge, c: &[String]| {
                            let i = c[0].clone();
                            let d = adj(em, &[format!("({i}) - {lo2}")])?;
                            Ok(format!(
                                "((({i}) >= {lo2} && ({i}) < {hi2}) ? ({d}) : 0.0f)"
                            ))
                        });
                    }
                }
                Err("gpu emit: grad through this index form not lowered".into())
            }
            TExpr::Block(stmts, ret) => {
                self.scopes.push(Vec::new());
                self.orig_scopes.push(Vec::new());
                let r = (|| {
                    for s in stmts {
                        match s {
                            TStmt::Let { .. } => self.stmt(s)?,
                            // eval's backward ignores block statements
                            // wholesale (`Block(_stmts, ret) → back(ret)`)
                            // — state writes are primal side effects the
                            // forward pass already emitted, and state
                            // reads are non-differentiable on both
                            // substrates, so they carry no adjoint.
                            TStmt::StateUpdate { .. } | TStmt::Expr(_) => {}
                            other => {
                                let k = match other {
                                    TStmt::Assign { .. } => "Assign",
                                    TStmt::ParamUpdate { .. } => "ParamUpdate",
                                    TStmt::AdamUpdate { .. } => "AdamUpdate",
                                    TStmt::Let { .. }
                                    | TStmt::StateUpdate { .. }
                                    | TStmt::Expr(_) => unreachable!(),
                                };
                                return Err(format!(
                                    "gpu emit: non-let statement ({k}) inside grad path"
                                ));
                            }
                        }
                    }
                    self.back(ret, wrt, f, adj)
                })();
                self.scopes.pop();
                self.orig_scopes.pop();
                r
            }
            TExpr::Field(..) | TExpr::Grad { .. } => Err("gpu grad: unsupported node".into()),
            _ => Ok(None),
        }
    }

    fn stmt(&mut self, s: &TStmt) -> Result<(), String> {
        match s {
            TStmt::Let { name, expr } => {
                // eval binds eagerly — a reduction-valued let becomes a
                // scratch buffer NOW (at stmt position, so later param
                // updates can't retroactively change its value), and
                // the name rebinds to a buffer ident that every reader
                // (rhs_read, backward helpers) resolves as storage.
                // orig_scopes keeps the real rhs either way — `back`
                // and `can_grad` need the expr DAG, not the buffer.
                self.orig_scopes
                    .last_mut()
                    .unwrap()
                    .push((name.clone(), expr.clone()));
                if expr.ty.numel() > 1 && Self::needs_scratch(expr) {
                    let buf = self.scratch_of(expr)?;
                    self.emit_q.extend(self.scratch.drain(..));
                    let e = Typed {
                        e: TExpr::Ident(buf),
                        ty: expr.ty.clone(),
                    };
                    self.scopes.last_mut().unwrap().push((name.clone(), e));
                } else {
                    self.scopes
                        .last_mut()
                        .unwrap()
                        .push((name.clone(), expr.clone()));
                }
                Ok(())
            }
            TStmt::ParamUpdate { param, scale, grad } => match &grad.e {
                TExpr::Grad { loss, wrt } if wrt == param && self.is_param(param) => {
                    // body generated at stmt position — lets bound so
                    // far are exactly eval's visible set.
                    let numel = self
                        .param_numel
                        .get(param)
                        .copied()
                        .ok_or_else(|| format!("gpu emit: {param} not a param"))?;
                    let mark = self.scratch.len();
                    let saved = std::mem::take(&mut self.lines);
                    let g = self
                        .grad_elem(loss, param, "gid")
                        .map_err(|e| format!("grad({param}): {e}"))?
                        .ok_or_else(|| format!("gpu emit: loss does not depend on {param}"))?;
                    self.lines
                        .push(self.param_st(param, "+=", &format!("float({scale}) * ({g})")));
                    let body = std::mem::replace(&mut self.lines, saved);
                    self.emit_q.extend(self.scratch.drain(mark..));
                    self.emit_q.push(EK {
                        name: format!("upd_{param}_{}", self.emit_q.len()),
                        dispatch: Dispatch::Linear(numel),
                        body,
                        writable: Some(param.clone()),
                    });
                    Ok(())
                }
                _ => Err("gpu emit: param updates expect `grad(loss, param)` on a param".into()),
            },
            TStmt::AdamUpdate { param, lr, grad } => match &grad.e {
                TExpr::Grad { loss, wrt } if wrt == param && self.is_param(param) => {
                    // eval's AdamUpdate records the raw gradient; the
                    // driver applies m/v moments between calls. On the
                    // GPU the whole step rides along: per-param state
                    // (ag/am/av/at) lives in zero-init scratch that
                    // persists across dispatches, the grad kernel runs
                    // stmt-position like any update, and the moment
                    // apply + step bump run AFTER the out kernel —
                    // eval leaves `param` untouched during `main`.
                    let numel = self
                        .param_numel
                        .get(param)
                        .copied()
                        .ok_or_else(|| format!("gpu emit: {param} not a param"))?;
                    for (tag, n) in [("ag", numel), ("am", numel), ("av", numel), ("at", 1)] {
                        let b = format!("{tag}_{param}");
                        if !self.scratch_args.iter().any(|(x, _)| x == &b) {
                            self.scratch_args.push((b, n));
                        }
                    }
                    let mark = self.scratch.len();
                    let saved = std::mem::take(&mut self.lines);
                    let g = self
                        .grad_elem(loss, param, "gid")
                        .map_err(|e| format!("grad({param}): {e}"))?
                        .ok_or_else(|| format!("gpu emit: loss does not depend on {param}"))?;
                    self.lines.push(format!("ag_{param}[gid] = ({g});"));
                    let body = std::mem::replace(&mut self.lines, saved);
                    self.emit_q.extend(self.scratch.drain(mark..));
                    self.emit_q.push(EK {
                        name: format!("grd_{param}_{}", self.emit_q.len()),
                        dispatch: Dispatch::Linear(numel),
                        body,
                        writable: None, // writes ag_ scratch — always device
                    });
                    // adam_step parity: b1=0.9 b2=0.999 eps=1e-2
                    // (fp16-safe), w rounded to half after the update.
                    // Pooled params read-modify-write inside
                    // `__params_pool` at their element offset.
                    let i = self.post_q.len();
                    let (warg, widx) = match self.param_off.get(param.as_str()) {
                        Some(off) => (POOL_ARG.to_string(), format!("{off} + gid")),
                        None => (format!("b_{param}"), "gid".to_string()),
                    };
                    let wline = if self.half_args.contains(&format!("b_{param}")) {
                        format!(
                            "{warg}[{widx}] = half(float({warg}[{widx}]) - {lr} * (m / bc1) / (sqrt(v / bc2) + 1e-2f));"
                        )
                    } else {
                        format!(
                            "{warg}[{widx}] = float(half({warg}[{widx}] - {lr} * (m / bc1) / (sqrt(v / bc2) + 1e-2f)));"
                        )
                    };
                    self.post_q.push(EK {
                        name: format!("adm_{param}_{i}"),
                        dispatch: Dispatch::Linear(numel),
                        body: vec![
                            format!("float t = at_{param}[0] + 1.0f;"),
                            "float bc1 = 1.0f - pow(0.9f, t);".into(),
                            "float bc2 = 1.0f - pow(0.999f, t);".into(),
                            format!("float g = ag_{param}[gid];"),
                            format!("float m = 0.9f * am_{param}[gid] + 0.1f * g;"),
                            format!("float v = 0.999f * av_{param}[gid] + 0.001f * g * g;"),
                            format!("am_{param}[gid] = m;"),
                            format!("av_{param}[gid] = v;"),
                            wline,
                        ],
                        writable: Some(param.clone()),
                    });
                    self.post_q.push(EK {
                        name: format!("bump_{param}_{i}"),
                        dispatch: Dispatch::Linear(1),
                        body: vec![format!("at_{param}[0] += 1.0f;")],
                        writable: None,
                    });
                    Ok(())
                }
                _ => Err("gpu emit: adam expects `adam(param, grad(loss, param) [, lr])`".into()),
            },
            TStmt::StateUpdate { key, port, expr } => {
                // typeck already desugared `s.f op= e` / `s.f[p] op= e`
                // into the full new field value — emit a stmt-position
                // kernel writing it into the resident state buffer.
                // eval's env-insert semantics fall out for free: later
                // statements read the same device buffer.
                let numel = self
                    .bufs
                    .get(key)
                    .map(|(_, t, _)| t.numel() as usize)
                    .ok_or_else(|| format!("gpu emit: {key} not a state field"))?;
                let mark = self.scratch.len();
                let saved = std::mem::take(&mut self.lines);
                let r = expr.ty.dims.len();
                let mut coords = Vec::new();
                for i in 0..r {
                    let cname = format!("c{i}");
                    let stride: i64 = expr.ty.dims[i + 1..].iter().product();
                    let d = expr.ty.dims[i];
                    if r == 1 {
                        self.lines.push(format!("uint {cname} = gid;"));
                    } else if stride == 1 {
                        self.lines.push(format!("uint {cname} = gid % {d};"));
                    } else {
                        self.lines
                            .push(format!("uint {cname} = (gid / {stride}) % {d};"));
                    }
                    coords.push(cname);
                }
                let v = self.gen(expr, &coords)?;
                self.lines.push(format!("b_{port}[gid] = ({v});"));
                let body = std::mem::replace(&mut self.lines, saved);
                self.emit_q.extend(self.scratch.drain(mark..));
                self.emit_q.push(EK {
                    name: format!("stw_{port}_{}", self.emit_q.len()),
                    dispatch: Dispatch::Linear(numel),
                    body,
                    writable: None, // state args are device float* in every sig
                });
                Ok(())
            }
            TStmt::Assign { name, op, expr } => {
                if self.is_param(name) {
                    // eval computes `nv = cur op v` elementwise and
                    // inserts it — a stmt-position kernel writing the
                    // param buffer mirrors that, and later readers see
                    // the device-side update.
                    let numel = self
                        .param_numel
                        .get(name)
                        .copied()
                        .ok_or_else(|| format!("gpu emit: {name} not a param"))?;
                    let mark = self.scratch.len();
                    let saved = std::mem::take(&mut self.lines);
                    let r = expr.ty.dims.len();
                    let mut coords = Vec::new();
                    for i in 0..r {
                        let cname = format!("c{i}");
                        let stride: i64 = expr.ty.dims[i + 1..].iter().product();
                        let d = expr.ty.dims[i];
                        if r == 1 {
                            self.lines.push(format!("uint {cname} = gid;"));
                        } else if stride == 1 {
                            self.lines.push(format!("uint {cname} = gid % {d};"));
                        } else {
                            self.lines
                                .push(format!("uint {cname} = (gid / {stride}) % {d};"));
                        }
                        coords.push(cname);
                    }
                    if coords.is_empty() {
                        coords.push("gid".to_string());
                    }
                    let v = self.gen(expr, &coords)?;
                    let op_s = match op {
                        AssignOp::Set => "=",
                        AssignOp::Add => "+=",
                        AssignOp::Sub => "-=",
                        AssignOp::Mul => "*=",
                        AssignOp::Div => "/=",
                    };
                    self.lines.push(self.param_st(name, op_s, &v));
                    let body = std::mem::replace(&mut self.lines, saved);
                    self.emit_q.extend(self.scratch.drain(mark..));
                    self.emit_q.push(EK {
                        name: format!("asg_{name}_{}", self.emit_q.len()),
                        dispatch: Dispatch::Linear(numel),
                        body,
                        writable: Some(name.clone()),
                    });
                    return Ok(());
                }
                // let-name assign — eval's env.insert: shadow-bind in
                // the current scope. `rposition` finds the newest `a`;
                // `resolve_limit` keeps an `a` INSIDE the new expr
                // (`a = a * 2`) resolving to the previous binding.
                let op_t = match op {
                    AssignOp::Set => None,
                    AssignOp::Add => Some(BinOp::Add),
                    AssignOp::Sub => Some(BinOp::Sub),
                    AssignOp::Mul => Some(BinOp::Mul),
                    AssignOp::Div => Some(BinOp::Div),
                };
                if self.pos_of(name).is_none() && self.bufs.get(name).is_none() {
                    return Err(format!("gpu emit: assign to unbound {name}"));
                }
                let new_e = match op_t {
                    None => expr.clone(),
                    Some(bop) => Typed {
                        ty: expr.ty.clone(),
                        e: TExpr::Bin(
                            bop,
                            Box::new(Typed {
                                ty: expr.ty.clone(),
                                e: TExpr::Ident(name.clone()),
                            }),
                            Box::new(expr.clone()),
                        ),
                    },
                };
                self.scopes.last_mut().unwrap().push((name.clone(), new_e.clone()));
                self.orig_scopes
                    .last_mut()
                    .unwrap()
                    .push((name.clone(), new_e));
                Ok(())
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
        orig_scopes: vec![Vec::new()],
        resolve_limit: Vec::new(), // (scope, pos) pairs — backward walk
        gen_limit: Vec::new(),     // (scope, pos) pairs — value gen
        emit_q: Vec::new(),
        post_q: Vec::new(),
        scratch: Vec::new(),
        mat_ptr: HashMap::new(),
        scratch_args: Vec::new(),
        half_args: std::collections::HashSet::new(),
        param_numel: p
            .params
            .iter()
            .map(|t| (t.name.clone(), t.ty.numel() as usize))
            .collect(),
        param_off: HashMap::new(),
        lines: Vec::new(),
        tmp: 0,
    };
    // arg names — buffer indices are assigned once scratch buffers
    // are counted (they append after the declared args).
    let mut base_args: Vec<(String, String, bool)> = Vec::new(); // (arg, port, state?)
    let mut args = Vec::new();
    for (port, dir, ty) in &p.ios {
        if *dir != IoDir::In {
            continue;
        }
        let arg = format!("b_{port}");
        ge.bufs
            .insert(port.clone(), (arg.clone(), ty.clone(), Kind::Row));
        base_args.push((arg, port.clone(), false));
        args.push(Arg::In(port.clone()));
    }
    // CYO_HALF: f16-declared params stage as real half buffers —
    // bandwidth halves on param-heavy kernels while compute stays
    // fp32. Update writes round through `half`, matching the f16
    // surface the eval drivers already write back to. Everything else
    // pools into `__params_pool`: one signature slot for all f32
    // params, per-param element offsets in `param_off`.
    let half_mode = std::env::var_os("CYO_HALF").is_some();
    let mut pool_next = 0i64;
    for t in &p.params {
        let pooled = !(half_mode && t.ty.dt == DType::F16);
        let arg = if pooled {
            ge.param_off.insert(t.name.clone(), pool_next);
            pool_next += (t.ty.numel() as i64 + 15) & !15;
            POOL_ARG.to_string()
        } else {
            format!("b_{}", t.name)
        };
        ge.bufs
            .insert(t.name.clone(), (arg.clone(), t.ty.clone(), Kind::ParamT));
        if pooled {
            args.push(Arg::Param(t.name.clone()));
        } else {
            ge.half_args.insert(arg.clone());
            args.push(Arg::ParamHalf(t.name.clone()));
        }
        base_args.push((arg, t.name.clone(), false));
    }
    let mut states = Vec::new();
    for (key, port, ty) in &p.states {
        let arg = format!("b_{port}");
        ge.bufs
            .insert(key.clone(), (arg.clone(), ty.clone(), Kind::Row));
        states.push((key.clone(), port.clone(), ty.clone()));
        base_args.push((arg, port.clone(), true));
        args.push(Arg::State(port.clone()));
    }

    // body walk emits kernels in discovery order: let-bound
    // reductions materialize eagerly (eval binds eagerly), updates
    // generate at their stmt position.
    for s in &p.body {
        ge.stmt(s)?;
    }

    // main out-kernel — last in dispatch order. A bare `A @ B` ret
    // takes the cooperative-tile path instead of per-element loops.
    let mark = ge.scratch.len();
    let (body, dispatch) = match ge.matmul_body(&p.ret, "_out") {
        Some((b, d)) => (b, d),
        None => match ge.reduce_pair(&p.ret, "_out")? {
            Some(r) => r,
            None => {
                ge.lines.clear();
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
                let v = ge.gen_raw(&p.ret, &coords)?;
                ge.lines.push(format!("_out[gid] = ({v});"));
                (
                    std::mem::take(&mut ge.lines),
                    Dispatch::Linear(oty.numel() as usize),
                )
            }
        },
    };
    ge.emit_q.extend(ge.scratch.drain(mark..));
    ge.emit_q.push(EK {
        name: "main".into(),
        dispatch,
        body,
        writable: None,
    });
    // adam applies land after main — eval leaves the param untouched
    // within the call.
    ge.emit_q.extend(ge.post_q.drain(..));

    // signature entries WITHOUT buffer indices: each kernel declares
    // only the buffers its body references (local indices), so total
    // buffer count is unbounded — Metal's 31-slot limit applies per
    // kernel, and real layers have hundreds of scratch intermediates.
    let mut entries: Vec<(String, String)> = vec![("_out".into(), "device float* _out".into())];
    for (arg, _port, writable) in &base_args {
        let elem_ty = if ge.half_args.contains(arg) {
            "half"
        } else {
            "float"
        };
        let qual = if *writable {
            format!("device {elem_ty}*")
        } else {
            format!("device const {elem_ty}*")
        };
        entries.push((arg.clone(), format!("{qual} {arg}")));
    }
    for (buf, n) in &ge.scratch_args {
        entries.push((buf.clone(), format!("device float* {buf}")));
        args.push(Arg::Scratch(buf.clone(), *n));
    }

    // identifiers referenced by a kernel body — buffer usage is the
    // intersection with the entry names (tokenized, so `b_k` never
    // matches `b_k0lo`).
    fn body_idents(body: &[String]) -> std::collections::HashSet<String> {
        let mut set = std::collections::HashSet::new();
        for line in body {
            let mut cur = String::new();
            for ch in line.chars() {
                if ch.is_alphanumeric() || ch == '_' {
                    cur.push(ch);
                } else if !cur.is_empty() {
                    set.insert(std::mem::take(&mut cur));
                }
            }
            if !cur.is_empty() {
                set.insert(std::mem::take(&mut cur));
            }
        }
        set
    }

    /// pointer-table lowering for oversized kernels: every buffer
    /// ident `name` becomes `(({cast})__ptrs->p[{K}])` — K is the
    /// entry's global buffer index (first occurrence wins, matching
    /// the `uses` dedup so pooled params always resolve to the real
    /// pool slot, never a dummy). The cast matches storage dtype so
    /// subscripts, pointer arithmetic, and half-widening reads all
    /// keep their original element semantics.
    fn ptrs_body(
        body: &[String],
        entries: &[(String, String)],
        half_args: &std::collections::HashSet<String>,
    ) -> Vec<String> {
        let mut map: HashMap<&str, (usize, &str)> = HashMap::new();
        for (i, (n, _)) in entries.iter().enumerate() {
            let cast = if half_args.contains(n) {
                "device half*"
            } else {
                "device float*"
            };
            map.entry(n.as_str()).or_insert((i, cast));
        }
        body.iter()
            .map(|line| {
                let mut out = String::with_capacity(line.len() + 24);
                let mut cur = String::new();
                for ch in line.chars().chain(std::iter::once(' ')) {
                    if ch.is_alphanumeric() || ch == '_' {
                        cur.push(ch);
                    } else {
                        match map.get(cur.as_str()) {
                            Some(&(k, cast)) => {
                                out.push_str(&format!("(({cast})__ptrs->p[{k}])"))
                            }
                            None => out.push_str(&cur),
                        }
                        cur.clear();
                        out.push(ch);
                    }
                }
                out
            })
            .collect()
    }

    /// descriptor lowering: every buffer ident becomes
    /// `(({cast})__ptrs->p[__desc[{slot}]])` where slot is a
    /// per-kernel-local index whose value the runtime supplies via a
    /// tiny `device uint` descriptor buffer. Baked constant offsets
    /// (`[<digits> +` — pooled-param element offsets and friends)
    /// become `__desc[{slot}]` too, so two kernels that differ ONLY
    /// in which tensors/params they touch normalize to identical
    /// source and share one compiled function. Returns the normalized
    /// lines plus the slot→value template (entry indices for buffer
    /// slots, literal offsets for offset slots).
    fn desc_body(
        body: &[String],
        name_to_k: &HashMap<&str, u32>,
        half_args: &std::collections::HashSet<String>,
    ) -> (Vec<String>, Vec<u32>) {
        let mut tpl: Vec<u32> = Vec::new();
        // slot per unique entry index — K is Copy, and duplicate entry
        // names (the shared `__params_pool`) resolve to the same K, so
        // keying by K also dedups them for free.
        let mut local: HashMap<u32, u32> = HashMap::new();
        let mut lines = Vec::with_capacity(body.len());
        for line in body {
            let mut out = String::with_capacity(line.len() + 32);
            let mut cur = String::new();
            for ch in line.chars().chain(std::iter::once(' ')) {
                if ch.is_alphanumeric() || ch == '_' {
                    cur.push(ch);
                } else {
                    if let Some(&k) = name_to_k.get(cur.as_str()) {
                        let slot = *local.entry(k).or_insert_with(|| {
                            tpl.push(k);
                            (tpl.len() - 1) as u32
                        });
                        let cast = if half_args.contains(cur.as_str()) {
                            "device half*"
                        } else {
                            "device float*"
                        };
                        out.push_str(&format!("(({cast})__ptrs->p[__desc[{slot}]])"));
                    } else {
                        out.push_str(&cur);
                    }
                    cur.clear();
                    out.push(ch);
                }
            }
            lines.push(out);
        }
        // second pass: baked `[<digits> +` / `[(<digits> +` index
        // offsets become descriptor slots carrying the literal value.
        let mut out_lines = Vec::with_capacity(lines.len());
        for line in &lines {
            let b = line.as_bytes();
            let mut out = String::with_capacity(line.len() + 16);
            let mut i = 0usize;
            while i < b.len() {
                if b[i] == b'[' {
                    let mut j = i + 1;
                    let mut paren = false;
                    if j < b.len() && b[j] == b'(' {
                        paren = true;
                        j += 1;
                    }
                    let ds = j;
                    while j < b.len() && b[j].is_ascii_digit() {
                        j += 1;
                    }
                    if j > ds
                        && j + 1 < b.len()
                        && b[j] == b' '
                        && b[j + 1] == b'+'
                        && matches!(b.get(j + 2), Some(b' ' | b'('))
                    {
                        let val: u64 = line[ds..j].parse().unwrap_or(0);
                        let slot = tpl.len() as u32;
                        tpl.push(val.min(u32::MAX as u64) as u32);
                        out.push('[');
                        if paren {
                            out.push('(');
                        }
                        out.push_str(&format!("__desc[{slot}]"));
                        i = j;
                        continue;
                    }
                }
                // ASCII-safe: MSL bodies are ASCII; push the byte's
                // char losslessly for our generated subset.
                out.push(b[i] as char);
                i += 1;
            }
            out_lines.push(out);
        }
        (out_lines, tpl)
    }

    // format every queued kernel with its own minimal signature;
    // update kernels re-declare their param arg writable. Kernels
    // whose live-buffer count would exceed Metal's 31-slot cap are
    // lowered through the pointer table instead: one `__ptrs` arg,
    // every buffer access becomes `__ptrs->p[K]` — K is the entry's
    // global buffer index, so the runtime binds one table of device
    // addresses and every tensor stays resident exactly once.
    let ptrs_max = std::env::var("CYO_PTRS_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(28);
    const PTRS_SLOTS: usize = 65536;
    // kernel dedup: structurally identical kernels (every layer and
    // head of a transformer emits the same op shapes) share one
    // compiled function — per-dispatch constants move into a tiny
    // `device uint` descriptor buffer. Without this a 36-layer
    // forward emits ~23k kernels and blows the AGX compiled-
    // variants footprint at pipeline-state creation. `CYO_DEDUP=0`
    // forces the per-kernel path; default auto-enables past 2048
    // queued kernels.
    let dedup = match std::env::var("CYO_DEDUP").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => ge.emit_q.len() > 2048,
    };
    let mut ptrs_used = false;
    let mut kernel_srcs = Vec::new();
    let mut kernels = Vec::new();
    let mut kernel_uses = Vec::new();
    let mut desc_templates: Vec<Vec<u32>> = Vec::new();
    if dedup {
        if entries.len() > PTRS_SLOTS {
            return Err(format!(
                "gpu emit ({name}): {} buffers exceeds pointer-table capacity {PTRS_SLOTS}",
                entries.len()
            ));
        }
        ptrs_used = true;
        // entry name → first global index (duplicate names like the
        // shared `__params_pool` keep the first slot, matching the
        // runtime's table construction).
        let mut name_to_k: HashMap<&str, u32> = HashMap::new();
        for (i, (n, _)) in entries.iter().enumerate() {
            name_to_k.entry(n.as_str()).or_insert(i as u32);
        }
        let mut groups: HashMap<(String, String), usize> = HashMap::new();
        let mut group_names: Vec<String> = Vec::new();
        for ek in &ge.emit_q {
            let (lines, tpl) = desc_body(&ek.body, &name_to_k, &ge.half_args);
            let body_txt = lines
                .iter()
                .map(|l| format!("    {l}"))
                .collect::<Vec<_>>()
                .join("\n");
            let key = (format!("{:?}", ek.dispatch), body_txt.clone());
            let gi = match groups.get(&key) {
                Some(&gi) => gi,
                None => {
                    let gi = group_names.len();
                    let gname = format!("coyote_{name}_g{gi}");
                    let thread_decl = match ek.dispatch {
                        Dispatch::Linear(_) => "uint gid [[thread_position_in_grid]]",
                        Dispatch::Group(_) => "uint lid [[thread_position_in_threadgroup]]",
                        Dispatch::Gemv(_) => {
                            "uint tgid [[threadgroup_position_in_grid]],\n    uint lid [[thread_position_in_threadgroup]]"
                        }
                        Dispatch::Tiled { .. } => {
                            "uint2 gid2 [[thread_position_in_grid]],\n    uint2 lid2 [[thread_position_in_threadgroup]]"
                        }
                        Dispatch::Simdgemm { .. } => {
                            "uint2 tgid2 [[threadgroup_position_in_grid]],\n    uint2 lid2 [[thread_position_in_threadgroup]]"
                        }
                    };
                    group_names.push(gname.clone());
                    kernel_srcs.push(format!(
                        "kernel void {gname}(\n    device __CoyotePtrs* __ptrs [[buffer(0)]],\n    device uint* __desc [[buffer(1)]],\n    {thread_decl})\n{{\n{body_txt}\n}}\n"
                    ));
                    groups.insert(key, gi);
                    gi
                }
            };
            kernels.push((group_names[gi].clone(), ek.dispatch));
            // buffer(0) = pointer table, buffer(1) = this dispatch's
            // descriptor — MAX-1 is the desc sentinel the runtime
            // resolves to the per-kernel desc buffer index.
            kernel_uses.push(vec![u32::MAX, u32::MAX - 1]);
            desc_templates.push(tpl);
        }
    } else {
    for ek in &ge.emit_q {
        let kname = format!("coyote_{name}_{}", ek.name);
        let idents = body_idents(&ek.body);
        // pooled params all share the `__params_pool` entry name —
        // dedup keeps the FIRST index so each kernel binds the pool
        // exactly once regardless of how many params it touches.
        let mut seen = std::collections::HashSet::new();
        let uses: Vec<u32> = entries
            .iter()
            .enumerate()
            .filter(|(_, (n, _))| idents.contains(n) && seen.insert(n.clone()))
            .map(|(i, _)| i as u32)
            .collect();
        if uses.len() > ptrs_max {
            if entries.len() > PTRS_SLOTS {
                return Err(format!(
                    "gpu emit ({name}/{}): {} buffers exceeds pointer-table capacity {PTRS_SLOTS}",
                    ek.name,
                    entries.len()
                ));
            }
            ptrs_used = true;
            kernel_srcs.push(format!(
                "kernel void {kname}(\n    device __CoyotePtrs* __ptrs [[buffer(0)]],\n    {}\n)\n{{\n{}\n}}\n",
                match ek.dispatch {
                    Dispatch::Linear(_) => "uint gid [[thread_position_in_grid]]",
                    Dispatch::Group(_) => "uint lid [[thread_position_in_threadgroup]]",
                    Dispatch::Gemv(_) => {
                        "uint tgid [[threadgroup_position_in_grid]],\n    uint lid [[thread_position_in_threadgroup]]"
                    }
                    Dispatch::Tiled { .. } => {
                        "uint2 gid2 [[thread_position_in_grid]],\n    uint2 lid2 [[thread_position_in_threadgroup]]"
                    }
                    Dispatch::Simdgemm { .. } => {
                        "uint2 tgid2 [[threadgroup_position_in_grid]],\n    uint2 lid2 [[thread_position_in_threadgroup]]"
                    }
                },
                ptrs_body(&ek.body, &entries, &ge.half_args)
                    .iter()
                    .map(|l| format!("    {l}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            ));
            kernels.push((kname, ek.dispatch));
            kernel_uses.push(vec![u32::MAX]); // sentinel → ptrs table
            continue;
        }
        let ksig: Vec<String> = uses
            .iter()
            .enumerate()
            .map(|(li, &gi)| {
                let s = format!("{} [[buffer({li})]]", entries[gi as usize].1);
                match &ek.writable {
                    Some(param) => {
                        let s = s
                            .replace(
                                &format!("device const float* b_{param}"),
                                &format!("device float* b_{param}"),
                            )
                            .replace(
                                &format!("device const half* b_{param}"),
                                &format!("device half* b_{param}"),
                            );
                        // a pooled param's update kernel writes through
                        // the shared pool — make its decl writable.
                        if ge.param_off.contains_key(param.as_str()) {
                            s.replace(
                                "device const float* __params_pool",
                                "device float* __params_pool",
                            )
                        } else {
                            s
                        }
                    }
                    None => s,
                }
            })
            .collect();
        let thread_decl = match ek.dispatch {
            Dispatch::Linear(_) => "uint gid [[thread_position_in_grid]]",
            Dispatch::Group(_) => "uint lid [[thread_position_in_threadgroup]]",
            Dispatch::Gemv(_) => {
                "uint tgid [[threadgroup_position_in_grid]],\n    uint lid [[thread_position_in_threadgroup]]"
            }
            Dispatch::Tiled { .. } => {
                "uint2 gid2 [[thread_position_in_grid]],\n    uint2 lid2 [[thread_position_in_threadgroup]]"
            }
            Dispatch::Simdgemm { .. } => {
                "uint2 tgid2 [[threadgroup_position_in_grid]],\n    uint2 lid2 [[thread_position_in_threadgroup]]"
            }
        };
        kernel_srcs.push(format!(
            "kernel void {kname}(\n    {},\n    {thread_decl})\n{{\n{}\n}}\n",
            ksig.join(",\n    "),
            ek.body
                .iter()
                .map(|l| format!("    {l}"))
                .collect::<Vec<_>>()
                .join("\n"),
        ));
        kernels.push((kname, ek.dispatch));
        kernel_uses.push(uses);
    }
    }
    desc_templates.resize(kernels.len(), Vec::new());

    let msl = format!(
        "// coyote gpu emit — {name}\n#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n\nstruct __CoyotePtrs {{ device char* p[{PTRS_SLOTS}]; }};\n\n{}",
        kernel_srcs.join("\n")
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
        param_pool: !ge.param_off.is_empty(),
        ptrs: ptrs_used,
        output: (oname.clone(), oty.clone()),
        kernels,
        kernel_uses,
        desc_templates,
    })
}
