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
    /// (pipeline, dispatch size) per kernel — update kernels first,
    /// the main out kernel last, matching eval's stmt→ret order.
    pipes: Vec<(metal::ComputePipelineState, crate::gpu_emit::Dispatch)>,
    out_n: usize,
    /// buffer objects in arg order — index 0 is out.
    bufs: Vec<metal::Buffer>,
    /// port/param/state name → buffer index in `bufs`.
    slot: HashMap<String, usize>,
    /// numel per buffer, for staging writes.
    elems: Vec<usize>,
    /// GPU-side duration of the last dispatch (command buffer
    /// GPUStartTime→GPUEndTime), milliseconds.
    last_ms: f64,
}

fn buf_f32(dev: &metal::Device, data: &[f32]) -> metal::Buffer {
    dev.new_buffer_with_data(
        data.as_ptr() as *const c_void,
        (data.len() * 4) as u64,
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
        for (kname, disp) in &out.kernels {
            let f = lib
                .get_function(kname, None)
                .map_err(|e| format!("msl kernel ({name}:{kname}): {e:?}"))?;
            let pipe = dev
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("msl pipeline ({name}:{kname}): {e:?}"))?;
            pipes.push((pipe, *disp));
        }
        let queue = dev.new_command_queue();

        let mut bufs = Vec::new();
        let mut slot = HashMap::new();
        let mut elems = Vec::new();
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
                    let data = coyote::weights::resolve(
                        pname,
                        Some(&expect),
                        t.src.as_deref(),
                        coyote::weights::param_fallback,
                    )?;
                    bufs.push(buf_f32(&dev, &data));
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
            last_ms: 0.0,
        })
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
        let cb = self.queue.new_command_buffer();
        // one encoder per dispatch — encoders serialize, so update
        // kernels fully complete before the out kernel reads params.
        for (pipe, disp) in &self.pipes {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipe);
            for (i, b) in self.bufs.iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            match disp {
                crate::gpu_emit::Dispatch::Linear(numel) => {
                    let n = *numel as u64;
                    let tg = (pipe.max_total_threads_per_threadgroup() as u64)
                        .min(n)
                        .max(1);
                    enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(tg, 1, 1));
                }
                crate::gpu_emit::Dispatch::Tiled { m, n, t, mt } => {
                    // each thread covers an mt×mt micro-tile → the grid
                    // is n/(t·mt) blocks × t threads. Grid is rounded
                    // UP to whole threadgroups — tile loads are
                    // lid-indexed and read by every member, so partial
                    // groups would leave stale threadgroup memory;
                    // kernel guards keep out-of-range threads clean.
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
            enc.end_encoding();
        }
        cb.commit();
        cb.wait_until_completed();
        // device-side kernel time: MTLCommandBuffer's own GPU
        // timestamps — the profiler's ground truth, not host wall.
        unsafe {
            let s: f64 = msg_send![&*cb, GPUStartTime];
            let e: f64 = msg_send![&*cb, GPUEndTime];
            self.last_ms = (e - s) * 1000.0;
        }
        let out = unsafe {
            std::slice::from_raw_parts(self.bufs[0].contents() as *const f32, self.out_n).to_vec()
        };
        Ok(out)
    }
}

impl coyote::backend::BackendExec for GpuRuntime {
    fn run(&mut self, writes: &[(String, Vec<f32>)]) -> Result<Vec<f32>, String> {
        GpuRuntime::run(self, writes)
    }
    fn device_ms(&self) -> Option<f64> {
        Some(self.last_ms)
    }
}
