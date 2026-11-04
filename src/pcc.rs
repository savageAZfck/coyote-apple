//! pcc.rs — the `pcc fn` execution site: attested remote compute.
//!
//! Same wire as `mesh fn` (wire.rs) with the PCC posture layered on:
//!
//!   1. `attest` — before any program or tensor crosses, the client asks
//!      the node for its attestation manifest: {node, build, caps, nonce}
//!      plus an HMAC proof binding the nonce to the treaty secret. The
//!      client verifies the proof FIRST — verify-before-secrets — and
//!      only then binds the program.
//!   2. `bind` — program source upload, as mesh. The bind frame carries
//!      the engine tag (`plain` or `enc`) so a `pcc enc fn` node stages
//!      ciphertext evaluation.
//!   3. `exec` — inputs as f32 payload (plain) or Ct quads (enc: four
//!      u64 LE words per lane — the client holds the LWE secret and
//!      decrypts the result; the node evaluates in the ciphertext
//!      domain and never sees plaintext). Every `result` is signed:
//!      `text` = HMAC(secret, "pcc-result:{nonce}:{sha256(payload)}"),
//!      verified client-side before the data is trusted.
//!
//! This is the loopback attested pair — the manifest is software-grade
//! (same shape a real T2/PCC attestation plugs into: nonce-bound proof,
//! declared build, capability set). The seam is stable: swap
//! `attestation()` for a hardware quote and the language doesn't change.

use crate::enc_ckks;
use crate::wire;
use coyote::ast::IoDir;
use coyote::backend::{Backend, BackendEmit, BackendExec};
use coyote::typeck::{CTy, TProgram};
use sha2::{Digest, Sha256};
use std::net::TcpStream;
use std::path::Path;
use std::sync::Arc;

pub struct PccBackend;

fn sha256_hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

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

impl Backend for PccBackend {
    fn emit(
        &self,
        src: &str,
        name: &str,
        tp: &TProgram,
        dir: &Path,
    ) -> Result<BackendEmit, String> {
        let (inputs, trainable, states, param_outs, state_outs, output) = io_plan(tp);
        let engine = if tp.target == coyote::ast::Target::Enc {
            "enc"
        } else {
            "plain"
        };
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("program.cy"), src).map_err(|e| e.to_string())?;
        std::fs::write(
            dir.join("pcc_manifest.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "name": name,
                "site": "pcc",
                "engine": engine,
                "host": std::env::var("COYOTE_PCC_HOST")
                    .unwrap_or_else(|_| "127.0.0.1:9722".into()),
                "ins": inputs.iter().map(|(p, t, _)| serde_json::json!({
                    "port": p, "dims": t.dims,
                })).collect::<Vec<_>>(),
                "out": { "port": output.0, "dims": output.1.dims },
                "posture": "attest-first: verify-before-secrets; signed results",
                "wire": "badapple-mesh/1 + pcc/attest/1",
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
            note: "pcc attested dispatch".into(),
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
            &std::fs::read_to_string(dir.join("pcc_manifest.json"))
                .map_err(|e| format!("pcc manifest: {e}"))?,
        )
        .map_err(|e| format!("pcc manifest json: {e}"))?;
        let host = std::env::var("COYOTE_PCC_HOST").unwrap_or_else(|_| {
            manifest["host"]
                .as_str()
                .unwrap_or("127.0.0.1:9722")
                .to_string()
        });
        let engine = manifest["engine"].as_str().unwrap_or("plain").to_string();
        let src = std::fs::read_to_string(dir.join("program.cy"))
            .map_err(|e| format!("program source: {e}"))?;
        let secret = wire::resolve_secret()?;
        let enc = std::env::var("COYOTE_MESH_ENC").ok().as_deref() != Some("0");
        let (mut s, cipher) = wire::connect(&host, &secret, enc, 120)?;

        // 1. attest — verify the node BEFORE anything sensitive crosses.
        let mut at = wire::Frame::new("attest");
        at.prompt = Some(name.into());
        wire::write_packet(&mut s, &at, &[], cipher.as_ref())?;
        let (resp, _) = wire::read_packet(&mut s, cipher.as_ref())?;
        if resp.op == "error" {
            return Err(format!(
                "pcc attest refused: {}",
                resp.error.unwrap_or_default()
            ));
        }
        if resp.op != "attest-ok" {
            return Err(format!("pcc attest: expected attest-ok, got {}", resp.op));
        }
        let att: serde_json::Value = serde_json::from_str(resp.text.as_deref().unwrap_or("{}"))
            .map_err(|e| format!("attestation json: {e}"))?;
        let nonce = att["nonce"]
            .as_str()
            .ok_or_else(|| "attestation missing nonce".to_string())?
            .to_string();
        let sig = att["sig"]
            .as_str()
            .ok_or_else(|| "attestation missing sig".to_string())?;
        let expect = wire::hmac_hex(&secret, &format!("pcc-attest:{nonce}"));
        if sig != expect {
            return Err("pcc attestation invalid — refusing to bind".into());
        }
        // verified — the node is a treaty'd peer. Log what it claims to be.
        let claim = att["node"].as_str().unwrap_or("unknown");
        eprintln!(
            "pcc: attested {claim} caps={} nonce={}…",
            att["caps"].as_str().unwrap_or("-"),
            &nonce[..16.min(nonce.len())]
        );

        // 2. bind — now secrets may cross.
        let mut bind = wire::Frame::new("bind");
        bind.prompt = Some(name.into());
        bind.text = Some(engine.clone());
        bind.shape = Some(vec![src.len()]);
        bind.dtype = Some("uint8".into());
        wire::write_packet(&mut s, &bind, src.as_bytes(), cipher.as_ref())?;
        let (resp, _) = wire::read_packet(&mut s, cipher.as_ref())?;
        match resp.op.as_str() {
            "bind-ok" | "result" => {}
            "error" => {
                return Err(format!(
                    "pcc bind rejected: {}",
                    resp.error.unwrap_or_default()
                ))
            }
            other => return Err(format!("pcc bind: expected bind-ok, got {other}")),
        }

        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x5eed);
        Ok(Box::new(PccExec {
            stream: s,
            cipher,
            engine,
            nonce,
            secret,
            ins: manifest["ins"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| Some(v["port"].as_str()?.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
            remote_wit: std::collections::VecDeque::new(),
            lwe_secret: (seed | 1) % 1000 + 1,
            lwe_rng: seed | 0x9e37_79b9,
        }))
    }
}

struct PccExec {
    stream: TcpStream,
    cipher: Option<wire::SessionCipher>,
    engine: String,
    nonce: String,
    secret: Vec<u8>,
    ins: Vec<String>,
    remote_wit: std::collections::VecDeque<(String, String)>,
    lwe_secret: u64,
    lwe_rng: u64,
}

impl PccExec {
    fn verify_result(&self, resp: &wire::Frame, payload: &[u8]) -> Result<(), String> {
        let sig = resp
            .text
            .as_deref()
            .ok_or_else(|| "pcc result unsigned".to_string())?;
        let expect = wire::hmac_hex(
            &self.secret,
            &format!("pcc-result:{}:{}", self.nonce, sha256_hex(payload)),
        );
        if sig != expect {
            return Err("pcc result signature invalid".into());
        }
        Ok(())
    }

    fn run_plain(&mut self, writes: &[(String, Vec<f32>)]) -> Result<Vec<f32>, String> {
        let mut f = wire::Frame::new("exec");
        let total: usize = writes.iter().map(|(_, d)| d.len()).sum();
        f.shape = Some(vec![total]);
        f.dtype = Some("float32".into());
        let mut payload = Vec::new();
        for port in &self.ins {
            let data = writes
                .iter()
                .find(|(w, _)| w == port)
                .map(|(_, d)| d.clone())
                .unwrap_or_default();
            for v in data {
                payload.extend_from_slice(&v.to_le_bytes());
            }
        }
        wire::write_packet(&mut self.stream, &f, &payload, self.cipher.as_ref())?;
        let (resp, payload) = wire::read_packet(&mut self.stream, self.cipher.as_ref())?;
        if resp.op == "error" {
            return Err(format!("pcc exec: {}", resp.error.unwrap_or_default()));
        }
        self.verify_result(&resp, &payload)?;
        if let Some(w) = resp.witness {
            self.remote_wit.extend(w);
        }
        Ok(payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// enc×pcc — client encrypts, node evals the ciphertext domain,
    /// client decrypts. Payload is Ct quads: 4×u64 LE per lane
    /// (a, b, noise, scale).
    fn run_enc(&mut self, writes: &[(String, Vec<f32>)]) -> Result<Vec<f32>, String> {
        let mut payload = Vec::new();
        let mut lanes = 0usize;
        for port in &self.ins {
            let data = writes
                .iter()
                .find(|(w, _)| w == port)
                .map(|(_, d)| d.clone())
                .unwrap_or_default();
            for m in &data {
                let c = enc_ckks::lwe_enc(*m as f64, self.lwe_secret, &mut self.lwe_rng);
                for w in [c.a, c.b, c.noise, c.scale] {
                    payload.extend_from_slice(&w.to_le_bytes());
                }
                lanes += 1;
            }
        }
        let mut f = wire::Frame::new("exec");
        f.text = Some("enc".into());
        f.shape = Some(vec![lanes]);
        f.dtype = Some("ct-quad".into());
        wire::write_packet(&mut self.stream, &f, &payload, self.cipher.as_ref())?;
        let (resp, payload) = wire::read_packet(&mut self.stream, self.cipher.as_ref())?;
        if resp.op == "error" {
            return Err(format!("pcc exec: {}", resp.error.unwrap_or_default()));
        }
        self.verify_result(&resp, &payload)?;
        if let Some(w) = resp.witness {
            self.remote_wit.extend(w);
        }
        let mut out = Vec::with_capacity(payload.len() / 32);
        for c in payload.chunks_exact(32) {
            let rd = |i: usize| u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap());
            let ct = enc_ckks::Ct {
                a: rd(0),
                b: rd(1),
                noise: rd(2),
                scale: rd(3),
            };
            out.push(enc_ckks::lwe_dec(&ct, self.lwe_secret) as f32);
        }
        Ok(out)
    }
}

impl BackendExec for PccExec {
    fn run(&mut self, writes: &[(String, Vec<f32>)]) -> Result<Vec<f32>, String> {
        if self.engine == "enc" {
            self.run_enc(writes)
        } else {
            self.run_plain(writes)
        }
    }

    fn drain_remote(&mut self) -> Vec<(String, String)> {
        self.remote_wit.drain(..).collect()
    }
}

// ---------------------------------------------------------------------------
// pccnode — the attested end of the pair. Serves attest/bind/exec; for
// bound enc programs the exec payload is Ct quads and the node evaluates
// in the ciphertext domain.
// ---------------------------------------------------------------------------

struct BoundProgram {
    name: String,
    engine: String,
    tp: TProgram,
    seq: u64,
}

fn attestation(nonce: &str, secret: &[u8]) -> serde_json::Value {
    serde_json::json!({
        "node": format!("coyote-apple/pccnode {}", env!("CARGO_PKG_VERSION")),
        "build": env!("CARGO_PKG_VERSION"),
        "caps": "plain,enc,cpu-eval",
        "attestation_class": "software-loopback",
        "nonce": nonce,
        "sig": wire::hmac_hex(secret, &format!("pcc-attest:{nonce}")),
    })
}

/// `coyote-apple pccnode --listen host:port`
pub fn serve(listen: &str) -> Result<(), String> {
    let secret = wire::resolve_secret()?;
    let enc = std::env::var("COYOTE_MESH_ENC").ok().as_deref() != Some("0");
    let listener =
        std::net::TcpListener::bind(listen).map_err(|e| format!("bind {listen}: {e}"))?;
    println!("pccnode listening on {listen} (enc={enc})");
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

fn sign_result(nonce: &str, secret: &[u8], payload: &[u8]) -> String {
    wire::hmac_hex(
        secret,
        &format!("pcc-result:{}:{}", nonce, sha256_hex(payload)),
    )
}

fn serve_conn(s: &mut TcpStream, secret: &[u8], enc: bool) -> Result<(), String> {
    let nonce = wire::server_handshake(s, secret)?;
    let cipher = enc.then(|| wire::SessionCipher::new(secret));
    let mut progs: Vec<BoundProgram> = Vec::new();
    loop {
        let (frame, payload) = match wire::read_packet(s, cipher.as_ref()) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        match frame.op.as_str() {
            "attest" => {
                let mut r = wire::Frame::new("attest-ok");
                r.text = Some(attestation(&nonce, secret).to_string());
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            "ping" => {
                let mut r = wire::Frame::new("result");
                r.token = Some(progs.len() as i64);
                wire::write_packet(s, &r, &[], cipher.as_ref())?;
            }
            "bind" => {
                let fname = frame.prompt.clone().unwrap_or_default();
                let engine = frame.text.clone().unwrap_or_else(|| "plain".into());
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
                progs.push(BoundProgram {
                    name: fname,
                    engine,
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
                let (payload_out, tag) =
                    if prog.engine == "enc" || frame.dtype.as_deref() == Some("ct-quad") {
                        exec_ciphertext(prog, &payload)?
                    } else {
                        exec_plain(prog, &payload)?
                    };
                let mut r = wire::Frame::new("result");
                r.text = Some(sign_result(&nonce, secret, &payload_out));
                r.witness = Some(vec![
                    ("pcc-exec".into(), tag),
                    ("pcc-node".into(), format!("{}#{}", prog.name, prog.seq)),
                ]);
                wire::write_packet(s, &r, &payload_out, cipher.as_ref())?;
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

/// f32 in → f32 out through the resident evaluator.
fn exec_plain(prog: &mut BoundProgram, payload: &[u8]) -> Result<(Vec<u8>, String), String> {
    let mut ins: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
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
    Ok((payload_out, "plain".into()))
}

/// Ct quads in → Ct quads out — the node never holds the secret.
fn exec_ciphertext(prog: &mut BoundProgram, payload: &[u8]) -> Result<(Vec<u8>, String), String> {
    let mut ct_ins: Vec<(String, Vec<enc_ckks::Ct>)> = Vec::new();
    let mut off = 0usize;
    for (pname, dir, ty) in prog.tp.ios.iter() {
        if !matches!(dir, IoDir::In) {
            continue;
        }
        let lanes: usize = ty.dims.iter().map(|d| (*d).max(0) as usize).product();
        let end = (off + lanes * 32).min(payload.len());
        let mut cts = Vec::with_capacity(lanes);
        for c in payload[off..end].chunks_exact(32) {
            let rd = |i: usize| u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap());
            cts.push(enc_ckks::Ct {
                a: rd(0),
                b: rd(1),
                noise: rd(2),
                scale: rd(3),
            });
        }
        ct_ins.push((pname.clone(), cts));
        off = end;
    }
    // node params stay plaintext — CHET posture: weights are the node's,
    // the data is the client's, encrypted end to end.
    let pt: Vec<(String, Vec<f64>)> = prog
        .tp
        .params
        .iter()
        .map(|p| (p.name.clone(), vec![0.0; p.ty.numel() as usize]))
        .collect();
    let out = enc_ckks::run_ciphertext(&prog.tp, &ct_ins, &pt)?;
    let mut payload_out = Vec::with_capacity(out.len() * 32);
    for c in &out {
        for w in [c.a, c.b, c.noise, c.scale] {
            payload_out.extend_from_slice(&w.to_le_bytes());
        }
    }
    Ok((payload_out, "enc".into()))
}

pub fn register() {
    coyote::backend::register_site(coyote::ast::Site::Pcc, Arc::new(PccBackend));
}
