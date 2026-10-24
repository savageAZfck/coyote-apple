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
use std::collections::HashMap;
use std::ffi::c_void;

pub struct GpuRuntime {
    _dev: metal::Device,
    queue: metal::CommandQueue,
    pipe: metal::ComputePipelineState,
    out_n: usize,
    /// buffer objects in arg order — index 0 is out.
    bufs: Vec<metal::Buffer>,
    /// port/param/state name → buffer index in `bufs`.
    slot: HashMap<String, usize>,
    /// numel per buffer, for staging writes.
    elems: Vec<usize>,
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
        let f = lib
            .get_function(&format!("coyote_{name}"), None)
            .map_err(|e| format!("msl kernel ({name}): {e:?}"))?;
        let pipe = dev
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| format!("msl pipeline ({name}): {e:?}"))?;
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
            }
        }
        Ok(Self {
            _dev: dev,
            queue,
            pipe,
            out_n,
            bufs,
            slot,
            elems,
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
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipe);
        for (i, b) in self.bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        let n = self.out_n as u64;
        let tg = (self.pipe.max_total_threads_per_threadgroup() as u64)
            .min(n)
            .max(1);
        enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
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
}
