//! rev_backend — the reversible stratum's engine.
//!
//! `rev fn` programs are bijective by construction (`revck` proves
//! it) and carry a compiler-derived inverse (`rev_lower`) compiled as
//! an ordinary program — `uncall`/direct calls resolve through the
//! normal program table. The reference semantics is the CPU
//! evaluator, the same one `--cpu` selects: reversible correctness is
//! defined here first, and every other substrate must agree with it.
//!
//! `emit` produces the same io plan every stratum hands the VM, so
//! port staging is identical regardless of target. `load` returns an
//! `Evaluator`-backed executor — the reversible floor, not a silicon
//! claim.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use coyote::ast::IoDir;
use coyote::backend::{Backend, BackendEmit, BackendExec};
use coyote::eval::Evaluator;
use coyote::typeck::TProgram;

pub struct RevBackend;

impl Backend for RevBackend {
    fn emit(
        &self,
        _src: &str,
        name: &str,
        tp: &TProgram,
        dir: &Path,
    ) -> Result<BackendEmit, String> {
        let inputs: Vec<(String, coyote::typeck::CTy, bool)> = tp
            .ios
            .iter()
            .filter(|(_, d, _)| matches!(d, IoDir::In))
            .map(|(n, _, t)| (n.clone(), t.clone(), true))
            .collect();
        let trainable: Vec<(String, String, coyote::typeck::CTy, Option<String>)> = tp
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
            dir.join("rev_manifest.json"),
            serde_json::json!({
                "name": name,
                "stratum": "rev",
                "engine": "eval (reference reversible semantics)",
                "ops": tp.body.len(),
                "inverse": format!("{name}_inv"),
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
            note: "rev/eval".into(),
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
        Ok(Box::new(EvalExec::new(tp)))
    }
}

/// `Evaluator`-backed executor. The evaluator borrows its `TProgram`,
/// so one is constructed per dispatch — param/state updates surface
/// through the host mirror between calls, same contract as every ext
/// backend.
struct EvalExec {
    tp: TProgram,
    in_ports: HashSet<String>,
    param_ports: HashSet<String>,
    state_ports: HashSet<String>,
}

impl EvalExec {
    fn new(tp: &TProgram) -> Self {
        Self {
            tp: tp.clone(),
            in_ports: tp
                .ios
                .iter()
                .filter(|(_, d, _)| matches!(d, IoDir::In))
                .map(|(n, _, _)| n.clone())
                .collect(),
            param_ports: tp.params.iter().map(|p| p.name.clone()).collect(),
            state_ports: tp.states.iter().map(|(_, p, _)| p.clone()).collect(),
        }
    }
}

impl BackendExec for EvalExec {
    fn run(&mut self, writes: &[(String, Vec<f32>)]) -> Result<Vec<f32>, String> {
        let mut inputs: HashMap<String, Vec<f32>> = HashMap::new();
        let mut params: HashMap<String, Vec<f32>> = HashMap::new();
        let mut states: HashMap<String, Vec<f32>> = HashMap::new();
        for (port, data) in writes {
            if self.in_ports.contains(port) {
                inputs.insert(port.clone(), data.clone());
            } else if self.param_ports.contains(port) {
                params.insert(port.clone(), data.clone());
            } else if self.state_ports.contains(port) {
                states.insert(port.clone(), data.clone());
            }
        }
        let mut ev = Evaluator::new(&self.tp);
        ev.set_params(params);
        ev.set_states(states);
        ev.run(&inputs)
            .map_err(|d| format!("rev eval: {d}"))
    }
}
