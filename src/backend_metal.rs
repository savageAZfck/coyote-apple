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

impl MetalBackend {
    /// Cache key for an emitted gpu program: the checked program's
    /// full debug fingerprint plus every CYO_*/COYOTE_* env flag —
    /// the emit read switches (CYO_HALF, CYO_DEDUP, tile geometry…)
    /// live in env, so the cache key must too.
    fn emit_key(tp: &TProgram) -> String {
        let mut envs: Vec<String> = std::env::vars_os()
            .filter(|(k, _)| {
                k.to_str()
                    .map(|s| s.starts_with("CYO_") || s.starts_with("COYOTE_"))
                    .unwrap_or(false)
            })
            .map(|(k, v)| {
                format!(
                    "{}={}",
                    k.to_string_lossy(),
                    v.to_string_lossy()
                )
            })
            .collect();
        envs.sort();
        coyote::lower::sha256_hex(
            format!("{tp:?}\n{}", envs.join("\n")).as_bytes(),
        )
    }

    fn emit_to_backend(out: gpu_emit::GpuOut) -> BackendEmit {
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
        BackendEmit {
            inputs: out.inputs.clone(),
            trainable: out.trainable.clone(),
            states: out.states.clone(),
            param_outs,
            state_outs,
            output: out.output.clone(),
            note: format!("msl sha {}", &out.msl_sha[..16]),
            opaque: Box::new(out),
        }
    }
}

impl Backend for MetalBackend {
    /// Emit `net.msl` + `coyote_build.json` — same artifact layout as
    /// the ANE path, so `audit`/`package`/`diff` cover gpu builds.
    /// `gpuout.cache`+`gpuout.sha` persist the emitted payload keyed by
    /// (program, env) — a same-source restart skips the ~minute-long
    /// emit entirely instead of re-deriving it.
    fn emit(
        &self,
        src: &str,
        name: &str,
        tp: &TProgram,
        dir: &Path,
    ) -> Result<BackendEmit, String> {
        let key = Self::emit_key(tp);
        let cache_bin = dir.join("gpuout.cache");
        let cache_key = dir.join("gpuout.sha");
        let out = if std::fs::read_to_string(&cache_key)
            .map(|s| s.trim() == key)
            .unwrap_or(false)
        {
            match std::fs::read(&cache_bin)
                .ok()
                .and_then(|b| serde_json::from_slice::<gpu_emit::GpuOut>(&b).ok())
            {
                Some(out) => out,
                None => gpu_emit::emit(src, name, tp)?,
            }
        } else {
            gpu_emit::emit(src, name, tp)?
        };
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        std::fs::write(dir.join("net.msl"), &out.msl).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("coyote_build.json"), &out.manifest).map_err(|e| e.to_string())?;
        // persist for the next run — write key+payload atomically-ish
        // (tmp then rename so a torn write never reads as a hit).
        if let Ok(bin) = serde_json::to_vec(&out) {
            let tmp = dir.join("gpuout.cache.tmp");
            if std::fs::write(&tmp, &bin).is_ok()
                && std::fs::rename(&tmp, &cache_bin).is_ok()
            {
                let _ = std::fs::write(&cache_key, &key);
            }
        }
        // params and state live resident on-device — updates land in
        // the same buffers the host store reads back through `read`,
        // so param_outs/state_outs simply mirror the resident ports.
        Ok(Self::emit_to_backend(out))
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
