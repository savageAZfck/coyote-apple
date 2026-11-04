//! mesh.rs — the `mesh fn` execution site.
//!
//! A `mesh fn` is an ordinary typed program whose *execution site* is a
//! treaty'd peer instead of local silicon. The compiler emits a dispatch
//! manifest; at load time the backend binds the program source to a
//! `coyote-apple meshnode` over the Bad Apple mesh wire (wire.rs —
//! byte-exact HMAC-auth'd, AES-256-GCM-sealed packets), and each `run`
//! ships the input tensors as one float32 payload and reads back the
//! output plus the remote node's witness entries.
//!
//! Wire ops (sub-stratum of the mesh frame vocabulary):
//!   bind     — prompt=fn name, payload=source bytes (dtype uint8).
//!              The node typechecks and stages an evaluator.
//!   exec     — shape=[total_elems], dtype=float32, payload=concat of
//!              declared inputs in order. Reply `result` carries the
//!              output payload + `witness` entries for the local chain.
//!   ping     — liveness; reply token=rank, layerStart/layerEnd.
//!   reset    — clears node-side program state.
//!   error    — any failure surface.

use crate::wire;
use coyote::ast::IoDir;
use coyote::backend::{Backend, BackendEmit, BackendExec};
use coyote::typeck::{CTy, TProgram};
use std::net::TcpStream;
use std::path::Path;
use std::sync::Arc;

/// Dispatch a whole program to the node's resident evaluator.
pub struct MeshBackend;

fn io_plan(
    tp: &TProgram,
) -> (
    Vec<(String, CTy, bool)>,
    Vec<(String, String, CTy, Option<String>)>,
    Vec<(String, String, CTy)>,
    Vec<(String, String, CTy)>,
    Vec<(String, String, CTy)>,
    (String, CTy),
) {
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
    let param_outs = trainable
        .iter()
        .map(|(n, p, t, _)| (n.clone(), p.clone(), t.clone()))
        .collect();
    let state_outs = states
        .iter()
        .map(|(_k, p, t)| (p.clone(), p.clone(), t.clone()))
        .collect();
    let output = ("out".to_string(), tp.ret.ty.clone());
    (inputs, trainable, states, param_outs, state_outs, output)
}

impl Backend for MeshBackend {
    fn emit(
        &self,
        src: &str,
        name: &str,
        tp: &TProgram,
        dir: &Path,
    ) -> Result<BackendEmit, String> {
        let (inputs, trainable, states, param_outs, state_outs, output) = io_plan(tp);
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("program.cy"), src).map_err(|e| e.to_string())?;
        std::fs::write(
            dir.join("mesh_manifest.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "name": name,
                "site": "mesh",
                "host": std::env::var("COYOTE_MESH_HOST")
                    .unwrap_or_else(|_| "127.0.0.1:9711".into()),
                "ins": inputs.iter().map(|(p, t, _)| serde_json::json!({
                    "port": p, "dims": t.dims,
                })).collect::<Vec<_>>(),
                "out": { "port": output.0, "dims": output.1.dims },
                "wire": "badapple-mesh/1",
            }))
            .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        Ok(BackendEmit {
            inputs,
            trainable,
            states,
            param_outs,
            state_outs,
            output,
            note: "mesh dispatch".into(),
            opaque: Box::new(()),
        })
    }

    fn load(
        &self,
        _tp: &TProgram,
        name: &str,
        _emit: &BackendEmit,
        dir: &Path,
    ) -> Result<Box<dyn BackendExec>, String> {
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("mesh_manifest.json"))
                .map_err(|e| format!("mesh manifest: {e}"))?,
        )
        .map_err(|e| format!("mesh manifest json: {e}"))?;
        let host = std::env::var("COYOTE_MESH_HOST").unwrap_or_else(|_| {
            manifest["host"]
                .as_str()
                .unwrap_or("127.0.0.1:9711")
                .to_string()
        });
        let src = std::fs::read_to_string(dir.join("program.cy"))
            .map_err(|e| format!("program source: {e}"))?;
        let secret = wire::resolve_secret()?;
        let enc = std::env::var("COYOTE_MESH_ENC").ok().as_deref() != Some("0");
        let (mut s, cipher) = wire::connect(&host, &secret, enc, 120)?;
        // bind — upload the program source; the node compiles + stages.
        let mut bind = wire::Frame::new("bind");
        bind.prompt = Some(name.into());
        bind.shape = Some(vec![src.len()]);
        bind.dtype = Some("uint8".into());
        wire::write_packet(&mut s, &bind, src.as_bytes(), cipher.as_ref())?;
        let (resp, _) = wire::read_packet(&mut s, cipher.as_ref())?;
        match resp.op.as_str() {
            "bind-ok" | "result" => {}
            "error" => {
                return Err(format!(
                    "mesh bind rejected: {}",
                    resp.error.unwrap_or_default()
                ))
            }
            other => return Err(format!("mesh bind: expected bind-ok, got {other}")),
        }
        Ok(Box::new(MeshExec {
            stream: s,
            cipher,
            ins: manifest["ins"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| {
                            Some((
                                v["port"].as_str()?.to_string(),
                                v["dims"]
                                    .as_array()?
                                    .iter()
                                    .filter_map(|d| d.as_i64())
                                    .collect::<Vec<i64>>(),
                            ))
                        })
                        .collect()
                })
                .unwrap_or_default(),
            remote_wit: std::collections::VecDeque::new(),
        }))
    }
}

struct MeshExec {
    stream: TcpStream,
    cipher: Option<wire::SessionCipher>,
    ins: Vec<(String, Vec<i64>)>,
    remote_wit: std::collections::VecDeque<(String, String)>,
}

impl BackendExec for MeshExec {
    fn run(&mut self, writes: &[(String, Vec<f32>)]) -> Result<Vec<f32>, String> {
        let mut f = wire::Frame::new("exec");
        let total: usize = writes.iter().map(|(_, d)| d.len()).sum();
        f.shape = Some(vec![total]);
        f.dtype = Some("float32".into());
        // declared-ins order is the wire order — params/states ride as
        // trailing ports only when the program declares them in `ios`.
        let payload = {
            let mut p = Vec::new();
            for (port, _) in &self.ins {
                let data = writes
                    .iter()
                    .find(|(w, _)| w == port)
                    .map(|(_, d)| d.clone())
                    .unwrap_or_default();
                for v in data {
                    p.extend_from_slice(&v.to_le_bytes());
                }
            }
            p
        };
        wire::write_packet(&mut self.stream, &f, &payload, self.cipher.as_ref())?;
        let (resp, payload) = wire::read_packet(&mut self.stream, self.cipher.as_ref())?;
        if resp.op == "error" {
            return Err(format!("mesh exec: {}", resp.error.unwrap_or_default()));
        }
        if let Some(w) = resp.witness {
            self.remote_wit.extend(w);
        }
        let mut out = Vec::with_capacity(payload.len() / 4);
        for c in payload.chunks_exact(4) {
            out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        Ok(out)
    }

    fn drain_remote(&mut self) -> Vec<(String, String)> {
        self.remote_wit.drain(..).collect()
    }
}

// ---------------------------------------------------------------------------
// meshnode — serve bound programs over the wire. Same handshake/packet
// format as the Bad Apple rank servers; a `mesh fn` client can't tell
// the difference.
// ---------------------------------------------------------------------------

struct NodeProgram {
    name: String,
    tp: TProgram,
    seq: u64,
}

/// `coyote-apple meshnode --listen host:port [--enc 0]`
/// Each connection: server handshake, optional cipher, then an op loop.
pub fn serve(listen: &str) -> Result<(), String> {
    let secret = wire::resolve_secret()?;
    let enc = std::env::var("COYOTE_MESH_ENC").ok().as_deref() != Some("0");
    let listener =
        std::net::TcpListener::bind(listen).map_err(|e| format!("bind {listen}: {e}"))?;
    println!("meshnode listening on {listen} (enc={enc})");
    for conn in listener.incoming() {
        let mut s = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept: {e}");
                continue;
            }
        };
        let secret = secret.clone();
        std::thread::spawn(move || {
            if let Err(e) = serve_conn(&mut s, &secret, enc) {
                let mut f = wire::Frame::new("error");
                f.error = Some(e);
                let _ = wire::write_frame(&mut s, &f);
            }
        });
    }
    Ok(())
}

fn serve_conn(s: &mut TcpStream, secret: &[u8], enc: bool) -> Result<(), String> {
    wire::server_handshake(s, secret)?;
    let cipher = enc.then(|| wire::SessionCipher::new(secret));
    let mut progs: Vec<NodeProgram> = Vec::new();
    loop {
        let (frame, payload) = match wire::read_packet(s, cipher.as_ref()) {
            Ok(v) => v,
            Err(_) => return Ok(()), // peer closed
        };
        match frame.op.as_str() {
            "ping" => {
                let mut r = wire::Frame::new("result");
                r.token = Some(progs.len() as i64);
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            "bind" => {
                let fname = frame.prompt.clone().unwrap_or_default();
                let src = String::from_utf8(payload)
                    .map_err(|e| format!("bind payload not utf8: {e}"))?;
                let ast = coyote::parser::parse(&src).map_err(|d| d.to_string())?;
                let checked = coyote::typeck::check_all(&ast).map_err(|d| d.to_string())?;
                let tp = checked
                    .ane
                    .into_iter()
                    .find(|(n, _)| n == &fname)
                    .map(|(_, t)| t)
                    .ok_or_else(|| format!("no fn `{fname}` in bound source"))?;
                progs.push(NodeProgram {
                    name: fname,
                    tp,
                    seq: 0,
                });
                let mut r = wire::Frame::new("bind-ok");
                r.token = Some((progs.len() - 1) as i64);
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            "exec" | "step" => {
                let prog = progs
                    .last_mut()
                    .ok_or_else(|| "exec before bind".to_string())?;
                prog.seq += 1;
                // split the f32 payload across declared ins in order.
                let mut ins: std::collections::HashMap<String, Vec<f32>> =
                    std::collections::HashMap::new();
                let mut off = 0usize;
                for (pname, dir, ty) in prog.tp.ios.iter() {
                    if !matches!(dir, IoDir::In) {
                        continue;
                    }
                    let elems: usize = ty.dims.iter().map(|d| (*d).max(0) as usize).product();
                    let end = (off + elems * 4).min(payload.len());
                    let mut v = Vec::with_capacity(elems);
                    for c in payload[off..end].chunks_exact(4) {
                        v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                    }
                    ins.insert(pname.clone(), v);
                    off = end;
                }
                let mut ev = coyote::eval::Evaluator::new(&prog.tp);
                let out = ev
                    .run(&ins)
                    .map_err(|d| format!("node eval ({}): {d}", prog.name))?;
                let mut payload_out = Vec::with_capacity(out.len() * 4);
                for v in &out {
                    payload_out.extend_from_slice(&v.to_le_bytes());
                }
                let mut r = wire::Frame::new("result");
                r.shape = Some(vec![out.len()]);
                r.dtype = Some("float32".into());
                r.witness = Some(vec![
                    ("mesh-exec".into(), format!("{}#{}", prog.name, prog.seq)),
                    ("mesh-node".into(), "coyote-apple/1".into()),
                ]);
                wire::write_packet(s, &r, &payload_out, cipher.as_ref())?;
            }
            "generate" | "ask" => {
                // rank-0 role: the pipeline would embed→layers→head here.
                // This node answers with a deterministic ack so the
                // serving tier's session/upstream path is exercisable
                // end-to-end without weights on this host.
                let prompt = frame.prompt.clone().unwrap_or_default();
                let tail: String = prompt
                    .chars()
                    .rev()
                    .take(48)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect();
                let mut r = wire::Frame::new("result");
                r.text = Some(format!("meshnode[{}]:{}", progs.len(), tail));
                r.witness = Some(vec![(
                    "mesh-generate".into(),
                    format!("len:{}", prompt.len()),
                )]);
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            "rewind" => {
                let mut r = wire::Frame::new("result");
                r.token = frame.token;
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            "reset" => {
                progs.clear();
                let r = wire::Frame::new("result");
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            other => {
                let mut r = wire::Frame::new("error");
                r.error = Some(format!("unknown op {other}"));
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
        }
    }
}

/// Register the mesh site backend — call once at startup.
pub fn register() {
    coyote::backend::register_site(coyote::ast::Site::Mesh, Arc::new(MeshBackend));
}
