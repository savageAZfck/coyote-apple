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

pub struct GpuRuntime {
    _dev: metal::Device,
    queue: metal::CommandQueue,
    /// (pipeline, dispatch, name, buffer binds) per kernel — update
    /// kernels first, the main out kernel last, matching eval's
    /// stmt→ret order. Each kernel declares only the buffers it
    /// references; `uses` maps signature slots → `bufs` indices.
    pipes: Vec<(
        metal::ComputePipelineState,
        crate::gpu_emit::Dispatch,
        String,
        Vec<u32>,
    )>,
    out_n: usize,
    /// buffer objects in arg order — index 0 is out.
    bufs: Vec<metal::Buffer>,
    /// port/param/state name → buffer index in `bufs`.
    slot: HashMap<String, usize>,
    /// numel per buffer, for staging writes.
    elems: Vec<usize>,
    /// buffer indices backed by `half` storage (`CYO_HALF` params) —
    /// `read_buf` decodes f16 → f32 for the host.
    half_slots: std::collections::HashSet<usize>,
    /// GPU-side duration of the last dispatch (command buffer
    /// GPUStartTime→GPUEndTime), milliseconds.
    last_ms: f64,
    /// CYO_PKT: one command buffer per kernel, so each pipeline gets
    /// its own GPU timestamp — per-kernel breakdown in `prof`.
    per_kernel: bool,
    /// kernel name → ms for the last run (per_kernel mode only).
    last_kernels: Vec<(String, f64)>,
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
    pub fn load(p: &TProgram, name: &str, out: &crate::gpu_emit::GpuOut) -> Result<Self, String> {
        let dev = metal::Device::system_default().ok_or("gpu: no Metal device on this host")?;
        let lib = dev
            .new_library_with_source(&out.msl, &CompileOptions::new())
            .map_err(|e| {
                format!(
                    "msl compile ({name}): {}",
                    String::from_utf8_lossy(e.as_bytes())
                )
            })?;
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
            pipes.push((pipe, *disp, kname.clone(), uses));
        }
        let queue = dev.new_command_queue();

        let mut bufs = Vec::new();
        let mut slot = HashMap::new();
        let mut elems = Vec::new();
        let mut half_slots = std::collections::HashSet::new();
        let out_n = out.output.1.numel() as usize;
        bufs.push(buf_f32(&dev, &vec![0.0f32; out_n]));
        slot.insert("out".into(), 0);
        elems.push(out_n);
        for (a_i, a) in out.args.iter().enumerate() {
            let i = a_i + 1;
            slot.insert(a.port().to_string(), i);
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
                    elems.push(n);
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
                    elems.push(n);
                }
                Arg::Param(pname) | Arg::ParamHalf(pname) => {
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
                    if matches!(a, Arg::ParamHalf(_)) {
                        bufs.push(buf_f16(&dev, &data));
                        half_slots.insert(i);
                    } else {
                        bufs.push(buf_f32(&dev, &data));
                    }
                    elems.push(data.len());
                }
                Arg::Scratch(_name, numel) => {
                    bufs.push(buf_f32(&dev, &vec![0.0; *numel]));
                    elems.push(*numel);
                }
            }
        }
        Ok(Self {
            _dev: dev,
            queue,
            pipes,
            out_n,
            bufs,
            slot,
            elems,
            half_slots,
            last_ms: 0.0,
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
            let &i = self
                .slot
                .get(port)
                .ok_or_else(|| format!("gpu run: no buffer for {port}"))?;
            if data.len() != self.elems[i] {
                return Err(format!(
                    "gpu run {port}: {} elems staged, buffer wants {}",
                    data.len(),
                    self.elems[i]
                ));
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    self.bufs[i].contents() as *mut f32,
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
            for (pipe, disp, name, uses) in &self.pipes {
                let cb = self.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pipe);
                for (li, &gi) in uses.iter().enumerate() {
                    enc.set_buffer(li as u64, Some(&self.bufs[gi as usize]), 0);
                }
                dispatch(enc, pipe, disp);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                let ms = gpu_ms(cb);
                self.last_ms += ms;
                self.last_kernels.push((name.clone(), ms));
            }
        } else {
            let cb = self.queue.new_command_buffer();
            // one encoder per dispatch — encoders serialize, so update
            // kernels fully complete before the out kernel reads params.
            for (pipe, disp, _, uses) in &self.pipes {
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pipe);
                for (li, &gi) in uses.iter().enumerate() {
                    enc.set_buffer(li as u64, Some(&self.bufs[gi as usize]), 0);
                }
                dispatch(enc, pipe, disp);
                enc.end_encoding();
            }
            cb.commit();
            cb.wait_until_completed();
            // device-side kernel time: MTLCommandBuffer's own GPU
            // timestamps — the profiler's ground truth, not host wall.
            self.last_ms = gpu_ms(cb);
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
        let &i = self.slot.get(port)?;
        let n = self.elems[i];
        if self.half_slots.contains(&i) {
            Some(unsafe {
                std::slice::from_raw_parts(self.bufs[i].contents() as *const u16, n)
                    .iter()
                    .map(|&b| half::f16::from_bits(b).to_f32())
                    .collect()
            })
        } else {
            Some(unsafe {
                std::slice::from_raw_parts(self.bufs[i].contents() as *const f32, n).to_vec()
            })
        }
    }
}

impl coyote::backend::BackendExec for GpuRuntime {
    fn run(&mut self, writes: &[(String, Vec<f32>)]) -> Result<Vec<f32>, String> {
        GpuRuntime::run(self, writes)
    }
    fn device_ms(&self) -> Option<f64> {
        Some(self.last_ms)
    }
    fn read(&self, port: &str) -> Option<Vec<f32>> {
        self.read_buf(port)
    }
}
