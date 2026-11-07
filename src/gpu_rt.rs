//! gpu_rt — Metal execution backend for `gpu fn` programs. Compiles
//! the emitted MSL at load, keeps every buffer resident across calls
//! (weights never re-upload), and dispatches one thread per output
//! element over shared-memory buffers — unified memory means the host
//! staging write is the same bytes the GPU reads.
//!
//! Convention: buffer 0 is the output; the rest follow
//! `GpuOut.args` — io ins, all params ([N, K] storage for 2-D), then
//! state ports.

use crate::gpu_emit::Arg;
use coyote::typeck::TProgram;
use metal::{CompileOptions, MTLResourceOptions, MTLSize};
use objc::{msg_send, sel, sel_impl};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

/// Shared device-resident parameter storage across every `GpuRuntime`
/// loaded through one `MetalBackend`. Programs whose declared param
/// set is identical (same fingerprint) share a single f32 pool buffer
/// and one f16 buffer per param — an optimizer update landing in
/// `train_l0`'s buffers is the same bytes `forward` and `train_l1`
/// read, with no host round-trip and no per-runtime weight copies.
/// Unified memory makes this an alias, not a copy: `StorageModeShared`
/// buffers are system DRAM both the CPU staging path and every GPU
/// dispatch already address directly.
///
/// Programs with different param universes get separate arena states
/// — name collisions across unrelated models can never alias.
pub struct GpuArena {
    inner: Mutex<HashMap<u64, ArenaState>>,
    /// compiled `MTLLibrary`s keyed by emit sha — evicted-then-reloaded
    /// programs reuse the compiled source instead of paying the Metal
    /// compile again.
    libs: Mutex<HashMap<String, metal::Library>>,
}

struct ArenaState {
    /// shared `__params_pool` buffer + capacity in f32 elements.
    pool: Option<(metal::Buffer, usize)>,
    /// param names already staged into the pool — reloads after
    /// eviction must not memcpy init values over trained weights.
    pool_filled: std::collections::HashSet<String>,
    /// param name → shared f16 buffer (+ element count).
    half: HashMap<String, (metal::Buffer, usize)>,
}

impl GpuArena {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            libs: Mutex::new(HashMap::new()),
        }
    }

    /// the compiled library for `msl`, compiling once per sha.
    pub fn library(
        &self,
        sha: &str,
        msl: &str,
        dev: &metal::Device,
    ) -> Result<metal::Library, String> {
        let mut g = self.libs.lock().map_err(|_| "gpu arena poisoned")?;
        if let Some(l) = g.get(sha) {
            return Ok(l.clone());
        }
        let l = dev
            .new_library_with_source(msl, &metal::CompileOptions::new())
            .map_err(|e| format!("msl compile: {}", String::from_utf8_lossy(e.as_bytes())))?;
        g.insert(sha.to_string(), l.clone());
        Ok(l)
    }

    /// Ensure the shared pool exists for this param universe and stage
    /// `data` at `off` — but only the first runtime writes; later
    /// loads (post-eviction reloads) keep the resident trained bytes.
    /// Returns the pool buffer.
    fn pool_stage(
        &self,
        f: u64,
        cap_elems: usize,
        name: &str,
        off: usize,
        data: &[f32],
        dev: &metal::Device,
    ) -> Result<metal::Buffer, String> {
        let mut g = self.inner.lock().map_err(|_| "gpu arena poisoned")?;
        let st = g.entry(f).or_insert_with(|| ArenaState {
            pool: None,
            pool_filled: std::collections::HashSet::new(),
            half: HashMap::new(),
        });
        let b = match &st.pool {
            Some((b, cap)) => {
                if *cap < cap_elems {
                    return Err(format!(
                        "gpu arena pool: {} elems shared, program wants {}",
                        cap, cap_elems
                    ));
                }
                b.clone()
            }
            None => {
                let nb = dev.new_buffer(
                    (cap_elems * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                st.pool = Some((nb.clone(), cap_elems));
                nb
            }
        };
        if st.pool_filled.insert(name.to_string()) {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    (b.contents() as *mut f32).add(off),
                    data.len(),
                );
            }
        }
        Ok(b)
    }

    /// identity of a program's whole param universe — every gpu fn of
    /// one model declares the same top-level params in the same order,
    /// so all of them hash identically and share one arena state.
    fn fingerprint(p: &TProgram) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for t in &p.params {
            t.name.hash(&mut h);
            t.ty.dims.hash(&mut h);
            format!("{:?}", t.ty.dt).hash(&mut h);
            t.src.hash(&mut h);
        }
        h.finish()
    }

    /// the shared f16 buffer for `name`, allocating + filling on first
    /// use. A later program with the same param universe reuses the
    /// resident bytes — including values a trainer already updated.
    fn half_buf(
        &self,
        f: u64,
        name: &str,
        data: &[f32],
        dev: &metal::Device,
    ) -> Result<metal::Buffer, String> {
        let mut g = self.inner.lock().map_err(|_| "gpu arena poisoned")?;
        let st = g.entry(f).or_insert_with(|| ArenaState {
            pool: None,
            pool_filled: std::collections::HashSet::new(),
            half: HashMap::new(),
        });
        if let Some((b, n)) = st.half.get(name) {
            if *n != data.len() {
                return Err(format!(
                    "gpu arena {name}: {} elems shared, program wants {}",
                    n,
                    data.len()
                ));
            }
            return Ok(b.clone());
        }
        let b = buf_f16(dev, data);
        st.half.insert(name.to_string(), (b.clone(), data.len()));
        Ok(b)
    }
}

pub struct GpuRuntime {
    _dev: metal::Device,
    queue: metal::CommandQueue,
    /// (pipeline, dispatch, name, buffer binds, indirect-residency set)
    /// per kernel — update kernels first, the main out kernel last,
    /// matching eval's stmt→ret order. Each kernel declares only the
    /// buffers it references; `uses` maps signature slots → `bufs`
    /// indices. `indirect` lists the `bufs` indices the kernel reaches
    /// only through `__ptrs->p[...]` — Metal requires an explicit
    /// `useResources` declaration for those per encoder.
    pipes: Vec<(
        metal::ComputePipelineState,
        crate::gpu_emit::Dispatch,
        String,
        Vec<u32>,
        Vec<u32>,
    )>,
    out_n: usize,
    /// buffer objects in arg order — index 0 is out. Pooled params
    /// share one buffer: every `Arg::Param` position after the first
    /// holds a small dummy (never bound — `uses` dedups to the first
    /// `__params_pool` entry), keeping `bufs` index-aligned with the
    /// emitter's entry order.
    bufs: Vec<metal::Buffer>,
    /// port/param/state name → (buffer index, element offset). The
    /// offset is nonzero only for pooled params — their weights live
    /// inside the shared pool at `param_off`-assigned slots.
    slot: HashMap<String, (usize, usize)>,
    /// staged element count per port name — per-param inside the pool.
    sizes: HashMap<String, usize>,
    /// buffer indices backed by `half` storage (`CYO_HALF` params) —
    /// `read_buf` decodes f16 → f32 for the host.
    half_slots: std::collections::HashSet<usize>,
    /// GPU-side duration of the last dispatch (command buffer
    /// GPUStartTime→GPUEndTime), milliseconds.
    last_ms: f64,
    /// host wall time of commit→completed across all chunks — the
    /// fallback clock when GPU timestamps report ~0 on huge CBs.
    last_wall_ms: f64,
    /// CYO_PKT: one command buffer per kernel, so each pipeline gets
    /// its own GPU timestamp — per-kernel breakdown in `prof`.
    per_kernel: bool,
    /// kernel name → ms for the last run (per_kernel mode only).
    last_kernels: Vec<(String, f64)>,
    /// `bufs` index of the pointer-table buffer (`__CoyotePtrs`), or
    /// `usize::MAX` when no kernel needed it. Kernels whose `uses`
    /// contain the `u32::MAX` sentinel bind this one buffer; their
    /// bodies reach every tensor through `__ptrs->p[K]`.
    ptrs_i: usize,
}

fn buf_f32(dev: &metal::Device, data: &[f32]) -> metal::Buffer {
    dev.new_buffer_with_data(
        data.as_ptr() as *const c_void,
        (data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn buf_f16(dev: &metal::Device, data: &[f32]) -> metal::Buffer {
    let h: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
    dev.new_buffer_with_data(
        h.as_ptr() as *const c_void,
        (h.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

impl GpuRuntime {
    /// compile the emitted MSL and allocate resident buffers. Params
    /// resolve exactly like the CPU evaluator's — `[N, K]` storage for
    /// 2-D, `param_fallback` deterministic init when no `@ src`.
    /// `arena` shares param storage across runtimes of the same model
    /// — pass `None` for a private, self-contained runtime.
    pub fn load(
        p: &TProgram,
        name: &str,
        out: &crate::gpu_emit::GpuOut,
        arena: Option<&GpuArena>,
    ) -> Result<Self, String> {
        let dev = metal::Device::system_default().ok_or("gpu: no Metal device on this host")?;
        // arena-backed runtimes share compiled libraries — a program
        // evicted under residency pressure reloads without paying the
        // Metal compile again.
        eprintln!("gpu load {name}: msl={}B kernels={}", out.msl.len(), out.kernels.len());
        let lib = match arena {
            Some(a) => a.library(&out.msl_sha, &out.msl, &dev)?,
            None => dev
                .new_library_with_source(&out.msl, &CompileOptions::new())
                .map_err(|e| {
                    format!(
                        "msl compile ({name}): {}",
                        String::from_utf8_lossy(e.as_bytes())
                    )
                })?,
        };
        eprintln!("gpu load {name}: lib done");
        let mut pipes = Vec::new();
        for (k_i, (kname, disp)) in out.kernels.iter().enumerate() {
            let f = lib
                .get_function(kname, None)
                .map_err(|e| format!("msl kernel ({name}:{kname}): {e:?}"))?;
            let pipe = dev
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("msl pipeline ({name}:{kname}): {e:?}"))?;
            let uses = out
                .kernel_uses
                .get(k_i)
                .cloned()
                .unwrap_or_else(|| (0..out.args.len() as u32 + 1).collect());
            let indirect = out.kernel_indirect.get(k_i).cloned().unwrap_or_else(|| {
                // emit always populates this; if a hand-built GpuOut
                // lacks it, a ptrs kernel gets the safe fallback —
                // u32::MAX marks "every data buffer" at dispatch.
                if uses.contains(&u32::MAX) {
                    vec![u32::MAX]
                } else {
                    Vec::new()
                }
            });
            pipes.push((pipe, *disp, kname.clone(), uses, indirect));
        }
        eprintln!("gpu load {name}: pipes done={}", pipes.len());
        let queue = dev.new_command_queue();

        let mut bufs = Vec::new();
        let mut slot = HashMap::new();
        let mut sizes = HashMap::new();
        let mut half_slots = std::collections::HashSet::new();
        let out_n = out.output.1.numel() as usize;
        bufs.push(buf_f32(&dev, &vec![0.0f32; out_n]));
        slot.insert("out".into(), (0, 0));
        sizes.insert("out".into(), out_n);
        // pooled f32 params share one device buffer sized up front —
        // 16-element-aligned slots, the same rule gpu_emit's
        // `param_off` uses, so host offsets match kernel indices.
        // With an arena the pool is shared across every runtime of the
        // same param universe — offsets are canonical because emit
        // assigns them by declaration order over the same `p.params`.
        let pool_mode = out.param_pool;
        let fprint = arena.map(|_| GpuArena::fingerprint(p));
        let mut pool_total = 0usize;
        if pool_mode {
            for a in &out.args {
                if let Arg::Param(pn) = a {
                    let t = p
                        .params
                        .iter()
                        .find(|t| &t.name == pn)
                        .ok_or_else(|| format!("gpu arg {pn}: not a param"))?;
                    pool_total += (t.ty.numel() as usize + 15) & !15;
                }
            }
        }
        // arena-backed pools stage each param once per universe —
        // eviction reloads keep trained weights instead of memcpy'ing
        // init values over them.
        let shared_arena = match (arena, fprint, pool_mode) {
            (Some(a), Some(f), true) => Some((f, a)),
            _ => None,
        };
        let mut pool_idx = usize::MAX;
        let mut pool_cursor = 0usize;
        // allocation forecast — printed before any buffer is created so
        // oversized programs report their footprint even when the load
        // is killed mid-allocation.
        {
            let (mut p16, mut s16, mut s32, mut io, mut st) = (0usize, 0usize, 0usize, 0usize, 0usize);
            for a in &out.args {
                match a {
                    Arg::ParamHalf(pn) => {
                        if let Some(t) = p.params.iter().find(|t| &t.name == pn) {
                            p16 += t.ty.numel() as usize;
                        }
                    }
                    Arg::Param(_) => {}
                    Arg::ScratchHalf(_, n) => s16 += *n,
                    Arg::Scratch(_, n) => s32 += *n,
                    Arg::In(pt) => {
                        if let Some((_, ty, _)) = out.inputs.iter().find(|(n, _, _)| n == pt) {
                            io += ty.numel() as usize;
                        }
                    }
                    Arg::State(pt) => {
                        if let Some((_, _, ty)) = out.states.iter().find(|(_, p, _)| p == pt) {
                            st += ty.numel() as usize;
                        }
                    }
                }
            }
            eprintln!(
                "gpu load {name}: args={} out={} pool_f32={} p16={:.1}MB s16={:.1}MB s32={:.1}MB io={:.1}MB st={:.1}MB",
                out.args.len(),
                out_n,
                pool_total,
                p16 as f64 * 2.0 / 1e6,
                s16 as f64 * 2.0 / 1e6,
                s32 as f64 * 4.0 / 1e6,
                io as f64 * 4.0 / 1e6,
                st as f64 * 4.0 / 1e6,
            );
        }
        for (a_i, a) in out.args.iter().enumerate() {
            let i = a_i + 1;
            slot.insert(a.port().to_string(), (i, 0));
            match a {
                Arg::In(port) => {
                    let ty = out
                        .inputs
                        .iter()
                        .find(|(n, _, _)| n == port)
                        .map(|(_, t, _)| t.clone())
                        .ok_or_else(|| format!("gpu arg {port}: no io type"))?;
                    let n = ty.numel() as usize;
                    bufs.push(buf_f32(&dev, &vec![0.0; n]));
                    sizes.insert(port.clone(), n);
                }
                Arg::State(port) => {
                    let ty = out
                        .states
                        .iter()
                        .find(|(_, p, _)| p == port)
                        .map(|(_, _, t)| t.clone())
                        .ok_or_else(|| format!("gpu arg {port}: no state type"))?;
                    let n = ty.numel() as usize;
                    bufs.push(buf_f32(&dev, &vec![0.0; n]));
                    sizes.insert(port.clone(), n);
                }
                Arg::Param(pname) => {
                    let t = p
                        .params
                        .iter()
                        .find(|t| &t.name == pname)
                        .ok_or_else(|| format!("gpu arg {pname}: not a param"))?;
                    let expect: Vec<usize> = if t.ty.dims.len() == 2 {
                        vec![t.ty.dims[1] as usize, t.ty.dims[0] as usize]
                    } else {
                        vec![t.ty.numel() as usize]
                    };
                    let mut data = coyote::weights::resolve(
                        pname,
                        Some(&expect),
                        t.src.as_deref(),
                        coyote::weights::param_fallback,
                    )?;
                    // eval quantizes q4 params at init — same values
                    // must land in the buffer or parity diverges.
                    if t.ty.dt == coyote::ast::DType::Q4 {
                        data = coyote::lower::q4_quantize(&data);
                    }
                    if pool_mode {
                        let off = pool_cursor;
                        pool_cursor += (t.ty.numel() as usize + 15) & !15;
                        match shared_arena {
                            Some((f, a)) => {
                                let b = a.pool_stage(f, pool_total, pname, off, &data, &dev)?;
                                if pool_idx == usize::MAX {
                                    pool_idx = bufs.len();
                                    bufs.push(b);
                                } else {
                                    // alignment placeholder — `uses`
                                    // never binds this position.
                                    bufs.push(buf_f32(&dev, &[0.0; 4]));
                                }
                            }
                            None => {
                                if pool_idx == usize::MAX {
                                    pool_idx = bufs.len();
                                    bufs.push(buf_f32(&dev, &vec![0.0; pool_total]));
                                } else {
                                    bufs.push(buf_f32(&dev, &[0.0; 4]));
                                }
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        data.as_ptr(),
                                        (bufs[pool_idx].contents() as *mut f32).add(off),
                                        data.len(),
                                    );
                                }
                            }
                        }
                        slot.insert(pname.clone(), (pool_idx, off));
                        sizes.insert(pname.clone(), data.len());
                    } else {
                        bufs.push(buf_f32(&dev, &data));
                        sizes.insert(pname.clone(), data.len());
                    }
                }
                Arg::ParamHalf(pname) => {
                    let t = p
                        .params
                        .iter()
                        .find(|t| &t.name == pname)
                        .ok_or_else(|| format!("gpu arg {pname}: not a param"))?;
                    let expect: Vec<usize> = if t.ty.dims.len() == 2 {
                        vec![t.ty.dims[1] as usize, t.ty.dims[0] as usize]
                    } else {
                        vec![t.ty.numel() as usize]
                    };
                    let mut data = coyote::weights::resolve(
                        pname,
                        Some(&expect),
                        t.src.as_deref(),
                        coyote::weights::param_fallback,
                    )?;
                    if t.ty.dt == coyote::ast::DType::Q4 {
                        data = coyote::lower::q4_quantize(&data);
                    }
                    // shared f16 params alias the arena buffer — the
                    // first runtime of this param universe filled it;
                    // every later one (forward, other trainers) reads
                    // the same resident bytes, updates included.
                    let hb = match (arena, fprint) {
                        (Some(a), Some(f)) => Some(a.half_buf(f, pname, &data, &dev)?),
                        _ => None,
                    };
                    match hb {
                        Some(b) => bufs.push(b),
                        None => bufs.push(buf_f16(&dev, &data)),
                    }
                    half_slots.insert(i);
                    sizes.insert(pname.clone(), data.len());
                }
                Arg::Scratch(name, numel) => {
                    bufs.push(buf_f32(&dev, &vec![0.0; *numel]));
                    sizes.insert(name.clone(), *numel);
                }
                Arg::ScratchHalf(name, numel) => {
                    // f16 scratch (`CYO_F16_SCRATCH`) — activations and
                    // tape adjoints; half the footprint of f32. Zeroed
                    // like every scratch: contrib kernels accumulate
                    // with `+=` and the emit side queues explicit
                    // zero kernels ahead of first use each dispatch.
                    bufs.push(buf_f16(&dev, &vec![0.0; *numel]));
                    half_slots.insert(i);
                    sizes.insert(name.clone(), *numel);
                }
            }
        }
        // pointer table for oversized kernels — one device buffer
        // holding every resident buffer's address; `__ptrs->p[K]`
        // in the kernel indexes it. Unified memory means
        // `contents()` IS the device-visible address.
        let ptrs_i = if out.ptrs {
            const PTRS_SLOTS: usize = 131072;
            let n = bufs.len();
            let tbl = dev.new_buffer(
                (PTRS_SLOTS * 8) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let base = tbl.contents() as *mut u64;
            for (i, b) in bufs.iter().enumerate() {
                unsafe {
                    let ga: u64 = msg_send![b.as_ref(), gpuAddress];
                    *base.add(i) = ga;
                }
            }
            bufs.push(tbl);
            n
        } else {
            usize::MAX
        };
        // descriptor buffers for deduped kernels — each dispatch
        // binds its own tiny `device uint` table (emit-time slot
        // values); the shared kernel function reads `__desc[i]` for
        // pointer-table indices and pool offsets. The u32::MAX-1
        // sentinel in `uses` is rewritten to this buffer's index.
        const DESC_SENTINEL: u32 = u32::MAX - 1;
        for (i, pipe) in pipes.iter_mut().enumerate() {
            if !pipe.3.contains(&DESC_SENTINEL) {
                continue;
            }
            let tpl = out
                .desc_templates
                .get(i)
                .cloned()
                .unwrap_or_default();
            let one = [0u32; 1];
            let (ptr, len) = if tpl.is_empty() {
                (one.as_ptr() as *const c_void, 4u64)
            } else {
                (tpl.as_ptr() as *const c_void, (tpl.len() * 4) as u64)
            };
            let dbuf = dev.new_buffer_with_data(
                ptr,
                len,
                MTLResourceOptions::StorageModeShared,
            );
            bufs.push(dbuf);
            let idx = (bufs.len() - 1) as u32;
            for u in pipe.3.iter_mut() {
                if *u == DESC_SENTINEL {
                    *u = idx;
                }
            }
        }
        eprintln!("gpu load {name}: args done bufs={}", bufs.len());
        Ok(Self {
            _dev: dev,
            queue,
            pipes,
            out_n,
            bufs,
            slot,
            sizes,
            half_slots,
            ptrs_i,
            last_ms: 0.0,
            last_wall_ms: 0.0,
            per_kernel: std::env::var_os("CYO_PKT").is_some(),
            last_kernels: Vec::new(),
        })
    }

    /// per-kernel device times from the last run — populated when
    /// `CYO_PKT` puts each kernel in its own command buffer.
    pub fn kernel_ms(&self) -> &[(String, f64)] {
        &self.last_kernels
    }

    /// stage `writes` (port/param/state name → f32 data) into resident
    /// buffers, dispatch, and return the output row-major.
    pub fn run(&mut self, writes: &[(String, Vec<f32>)]) -> Result<Vec<f32>, String> {
        for (port, data) in writes {
            let &(bi, off) = self
                .slot
                .get(port)
                .ok_or_else(|| format!("gpu run: no buffer for {port}"))?;
            let want = *self
                .sizes
                .get(port)
                .ok_or_else(|| format!("gpu run: no size for {port}"))?;
            if data.len() != want {
                return Err(format!(
                    "gpu run {port}: {} elems staged, buffer wants {}",
                    data.len(),
                    want
                ));
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    (self.bufs[bi].contents() as *mut f32).add(off),
                    data.len(),
                );
            }
        }
        // dispatch boundaries: Linear is flat threads; Tiled's grid is
        // rounded UP to whole threadgroups — tile loads are
        // lid-indexed and read by every member, so partial groups
        // would leave stale threadgroup memory; kernel guards keep
        // out-of-range threads clean.
        fn dispatch(
            enc: &metal::ComputeCommandEncoderRef,
            pipe: &metal::ComputePipelineState,
            disp: &crate::gpu_emit::Dispatch,
        ) {
            match disp {
                crate::gpu_emit::Dispatch::Linear(numel) => {
                    let n = *numel as u64;
                    let tg = (pipe.max_total_threads_per_threadgroup() as u64)
                        .min(n)
                        .max(1);
                    enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(tg, 1, 1));
                }
                crate::gpu_emit::Dispatch::Group(threads) => {
                    // exactly one threadgroup — barriers in the kernel
                    // only synchronize within a group.
                    let t = *threads as u64;
                    enc.dispatch_threads(MTLSize::new(t, 1, 1), MTLSize::new(t, 1, 1));
                }
                crate::gpu_emit::Dispatch::Gemv(n) => {
                    // each 64-thread group owns 64 consecutive
                    // outputs — one output per thread, serial-k
                    // accumulation per thread, cooperative tile loads.
                    let tg = 64u64;
                    let groups = (*n as u64).div_ceil(tg);
                    enc.dispatch_threads(MTLSize::new(groups * tg, 1, 1), MTLSize::new(tg, 1, 1));
                }
                crate::gpu_emit::Dispatch::Tiled { m, n, t, mt } => {
                    let (tt, mt) = (*t as u64, *mt as u64);
                    let gx = (*n as u64).div_ceil(tt * mt) * tt;
                    let gy = (*m as u64).div_ceil(tt * mt) * tt;
                    enc.dispatch_threads(MTLSize::new(gx, gy, 1), MTLSize::new(tt, tt, 1));
                }
                crate::gpu_emit::Dispatch::Simdgemm { m, n } => {
                    // one 128-thread group per 64×64 block: threadgroup
                    // position IS the block origin (tgid2·64).
                    let gx = (*n as u64).div_ceil(64) * 128;
                    let gy = (*m as u64).div_ceil(64);
                    enc.dispatch_threads(MTLSize::new(gx, gy, 1), MTLSize::new(128, 1, 1));
                }
            }
        }
        // Buffers reached only through `__ptrs->p[K]` (gpuAddress)
        // are indirect accesses — Metal does not make them resident
        // from the pointer table alone; each compute encoder must
        // declare them via `useResources`. `set_buffer`-bound buffers
        // are resident automatically, so only the per-kernel indirect
        // set is marked — a handful of resources per encoder instead
        // of the whole ~6k-buffer table that exhausted
        // IOGPUDeviceShmem when marked indiscriminately.
        fn mark_indirect(
            enc: &metal::ComputeCommandEncoderRef,
            bufs: &[metal::Buffer],
            ptrs_i: usize,
            indirect: &[u32],
        ) {
            if indirect.is_empty() {
                return;
            }
            let all = indirect.contains(&u32::MAX);
            let refs: Vec<&metal::ResourceRef> = if all {
                (0..ptrs_i.min(bufs.len()))
                    .map(|i| {
                        let b = &bufs[i];
                        &***b
                    })
                    .collect()
            } else {
                indirect
                    .iter()
                    .map(|&i| i as usize)
                    .filter(|&i| i < bufs.len() && i != ptrs_i)
                    .map(|i| {
                        let b = &bufs[i];
                        &***b
                    })
                    .collect()
            };
            if !refs.is_empty() {
                enc.use_resources(
                    &refs,
                    metal::MTLResourceUsage::Read | metal::MTLResourceUsage::Write,
                );
            }
        }
        let gpu_ms = |cb: &metal::CommandBufferRef| unsafe {
            let s: f64 = msg_send![cb, GPUStartTime];
            let e: f64 = msg_send![cb, GPUEndTime];
            (e - s) * 1000.0
        };
        if self.per_kernel {
            // CYO_PKT — one command buffer per kernel so each pipeline
            // carries its own GPU timestamps. Costs a commit+wait per
            // kernel; the breakdown is the point, not the total.
            self.last_kernels.clear();
            self.last_ms = 0.0;
            for (pipe, disp, name, uses, indirect) in &self.pipes {
                let cb = self.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pipe);
                for (li, &gi) in uses.iter().enumerate() {
                    let bi = if gi == u32::MAX { self.ptrs_i } else { gi as usize };
                    enc.set_buffer(li as u64, Some(&self.bufs[bi]), 0);
                }
                mark_indirect(enc, &self.bufs, self.ptrs_i, indirect);
                dispatch(enc, pipe, disp);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                let ms = gpu_ms(cb);
                self.last_ms += ms;
                self.last_kernels.push((name.clone(), ms));
            }
        } else {
            // chunk dispatches across command buffers (512 encoders
            // per cb) so argument-buffer residency (`use_resources`)
            // never exhausts IOGPUDeviceShmem on 6,700-kernel programs.
            self.last_ms = 0.0;
            self.last_wall_ms = 0.0;
            self.last_kernels.clear();
            for (ci, chunk) in self.pipes.chunks(512).enumerate() {
                let cb = self.queue.new_command_buffer();
                for (pipe, disp, _, uses, indirect) in chunk {
                    let enc = cb.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(pipe);
                    for (li, &gi) in uses.iter().enumerate() {
                        let bi = if gi == u32::MAX { self.ptrs_i } else { gi as usize };
                        enc.set_buffer(li as u64, Some(&self.bufs[bi]), 0);
                    }
                    mark_indirect(enc, &self.bufs, self.ptrs_i, indirect);
                    dispatch(enc, pipe, disp);
                    enc.end_encoding();
                }
                let t0 = std::time::Instant::now();
                cb.commit();
                cb.wait_until_completed();
                let wall = t0.elapsed().as_secs_f64() * 1000.0;
                self.last_wall_ms += wall;
                let ms = gpu_ms(cb);
                self.last_ms += ms;
                // coarse per-chunk breakdown — 512-kernel granularity
                // is enough to locate hot regions in 6k-kernel programs
                // without CYO_PKT's per-kernel commit+wait cost.
                self.last_kernels
                    .push((format!("chunk_{ci}"), wall.max(ms)));
                // diagnostics: a faulted chunk reports ~0 GPU time while
                // the host still paid the encode cost — surface the
                // Metal error instead of silently trusting the clock.
                let status: usize = unsafe { msg_send![&*cb, status] };
                if status != 4 {
                    // MTLCommandBufferStatusCompleted == 4
                    let err: *mut objc::runtime::Object = unsafe { msg_send![&*cb, error] };
                    eprintln!(
                        "gpu_rt: chunk {ci} cb status {status} err {:?}",
                        if err.is_null() {
                            None
                        } else {
                            Some(unsafe {
                                let d: *const std::ffi::c_char =
                                    msg_send![err, UTF8String];
                                std::ffi::CStr::from_ptr(d).to_string_lossy().into_owned()
                            })
                        }
                    );
                }
            }
        }
        let out = unsafe {
            std::slice::from_raw_parts(self.bufs[0].contents() as *const f32, self.out_n).to_vec()
        };
        Ok(out)
    }

    /// read a resident buffer (param `p`, io port, or scratch name)
    /// back to the host — the trained-weight export path: kernels
    /// mutate `b_param` in place across dispatches, so the owner's
    /// checkpoint lives here, not in `param_outs`.
    pub fn read_buf(&self, port: &str) -> Option<Vec<f32>> {
        let &(bi, off) = self.slot.get(port)?;
        let n = *self.sizes.get(port)?;
        if self.half_slots.contains(&bi) {
            // half params are never pooled — `off` is always 0 here.
            Some(unsafe {
                std::slice::from_raw_parts(self.bufs[bi].contents() as *const u16, n)
                    .iter()
                    .map(|&b| half::f16::from_bits(b).to_f32())
                    .collect()
            })
        } else {
            Some(unsafe {
                std::slice::from_raw_parts(
                    (self.bufs[bi].contents() as *const f32).add(off),
                    n,
                )
                .to_vec()
            })
        }
    }
}

impl coyote::backend::BackendExec for GpuRuntime {
    fn run(&mut self, writes: &[(String, Vec<f32>)]) -> Result<Vec<f32>, String> {
        GpuRuntime::run(self, writes)
    }
    fn device_ms(&self) -> Option<f64> {
        // prefer the commit→completed wall clock: GPUStart/End report
        // ~0 on multi-chunk command buffers even while kernels run.
        if self.last_wall_ms > 0.0 {
            Some(self.last_wall_ms)
        } else {
            Some(self.last_ms)
        }
    }
    fn read(&self, port: &str) -> Option<Vec<f32>> {
        self.read_buf(port)
    }
}
