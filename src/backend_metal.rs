//! backend_metal — the `gpu` stratum's engine, first-party edition.
//! Lowering + runtime live here in the distribution; the public crate
//! only knows a `gpu` decl exists and dispatches through this backend.

use crate::{gpu_emit, gpu_rt};
use coyote::backend::{Backend, BackendEmit, BackendExec};
use coyote::typeck::{CTy, TProgram};
use std::path::Path;

pub struct MetalBackend {
    /// param storage shared by every runtime this backend loads —
    /// one model's gpu fns see each other's weight updates on-device.
    arena: std::sync::Arc<gpu_rt::GpuArena>,
}

impl MetalBackend {
    pub fn new() -> Self {
        Self {
            arena: std::sync::Arc::new(gpu_rt::GpuArena::new()),
        }
    }
}

impl Backend for MetalBackend {
    /// Emit `net.msl` + `coyote_build.json` — same artifact layout as
    /// the ANE path, so `audit`/`package`/`diff` cover gpu builds.
    fn emit(
        &self,
        src: &str,
        name: &str,
        tp: &TProgram,
        dir: &Path,
    ) -> Result<BackendEmit, String> {
        let out = gpu_emit::emit(src, name, tp)?;
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        std::fs::write(dir.join("net.msl"), &out.msl).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("coyote_build.json"), &out.manifest).map_err(|e| e.to_string())?;
        // params and state live resident on-device — updates land in
        // the same buffers the host store reads back through `read`,
        // so param_outs/state_outs simply mirror the resident ports.
        let param_outs: Vec<(String, String, CTy)> = out
            .trainable
            .iter()
            .map(|(n, port, ty, _)| (n.clone(), port.clone(), ty.clone()))
            .collect();
        let state_outs: Vec<(String, String, CTy)> = out
            .states
            .iter()
            .map(|(_k, port, ty)| (port.clone(), port.clone(), ty.clone()))
            .collect();
        Ok(BackendEmit {
            inputs: out.inputs.clone(),
            trainable: out.trainable.clone(),
            states: out.states.clone(),
            param_outs,
            state_outs,
            output: out.output.clone(),
            note: format!("msl sha {}", &out.msl_sha[..16]),
            opaque: Box::new(out),
        })
    }

    fn load(
        &self,
        tp: &TProgram,
        name: &str,
        emit: &BackendEmit,
        _dir: &Path,
    ) -> Result<Box<dyn BackendExec>, String> {
        let out = emit
            .opaque
            .downcast_ref::<gpu_emit::GpuOut>()
            .ok_or("metal backend: wrong emit payload")?;
        Ok(Box::new(gpu_rt::GpuRuntime::load(
            tp,
            name,
            out,
            Some(&self.arena),
        )?))
    }
}
