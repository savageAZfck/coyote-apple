//! wire.rs — the Bad Apple mesh-brain wire protocol, byte-exact.
//!
//! One format, two packings:
//!   plaintext:  [u32 BE jsonLen][json][payload]
//!   encrypted:  [u32 BE ctLen][nonce(12) || AES-256-GCM(u32BE jsonLen
//!               || json || payload) || tag(16)]
//!
//! The JSON header is `Frame` — field-for-field the `MeshBrainFrame`
//! shared with BadAppleShard.swift. `payload` rides behind the header
//! and is sized by `shape` × `dtype_size(dtype)` on the plaintext path
//! (the receiver knows how many bytes to pull).
//!
//! Mutual auth is the mesh handshake: server challenges with a hex
//! nonce, client answers HMAC-SHA256(key, "mb-c:{nonce}") as hex, server
//! counter-proves HMAC-SHA256(key, "mb-s:{nonce}"). Session cipher is
//! AES-256-GCM keyed by SHA-256(secret) — the same derivation Bad
//! Apple's p2p engram crypto uses.

use aes_gcm::aead::{Aead, AeadCore, OsRng};
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

type HmacSha256 = Hmac<Sha256>;
const NONCE_LEN: usize = 12;
/// plaintext header cap — same bound as the mesh server.
const MAX_HEADER: usize = 64 * 1024;
/// encrypted packet cap — same bound as the mesh server.
const MAX_PACKET: usize = 512 * 1024 * 1024;

/// Frame header shared with BadAppleShard.swift:
/// [u32 BE jsonLen][json][payload].
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Frame {
    pub op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxTokens")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "topP")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "layerStart")]
    pub layer_start: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "layerEnd")]
    pub layer_end: Option<usize>,
    /// Witness entries the remote node produced while executing this
    /// frame — `(tag, value)` pairs the caller lands on its local chain.
    #[serde(skip_serializing_if = "Option::is_none", rename = "witness")]
    pub witness: Option<Vec<(String, String)>>,
    /// KV lane id — which per-session state lane services this frame.
    /// coyote-apple extension: unknown fields are ignored by serde, so
    /// bad_apple peers drop it silently and single-stream semantics hold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lane: Option<usize>,
}

impl Frame {
    pub fn new(op: &str) -> Self {
        Frame {
            op: op.into(),
            ..Default::default()
        }
    }
}

/// element size of a wire dtype — matches mesh_brain::dtype_size.
pub fn dtype_size(name: &str) -> usize {
    match name {
        "float64" | "int64" | "uint64" => 8,
        "float32" | "int32" | "uint32" => 4,
        "bool" | "uint8" | "int8" => 1,
        _ => 2, // f16/bf16/u16/i16 + anything else small
    }
}

/// AES-256-GCM session cipher keyed by SHA-256(secret) — the same
/// derivation Bad Apple's P2PCipher uses; ciphertexts interop.
pub struct SessionCipher {
    aead: Aes256Gcm,
}

impl SessionCipher {
    pub fn new(secret: &[u8]) -> Self {
        let key = Sha256::digest(secret);
        SessionCipher {
            aead: Aes256Gcm::new_from_slice(&key).expect("sha256 is 32 bytes"),
        }
    }

    /// `[nonce || ciphertext || tag]` — same layout as P2PCipher::encrypt.
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>, String> {
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ct = self
            .aead
            .encrypt(&nonce, plaintext)
            .map_err(|e| format!("aead encrypt: {e}"))?;
        let mut out = Vec::with_capacity(NONCE_LEN + ct.len());
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&ct);
        Ok(out)
    }

    pub fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>, String> {
        if ciphertext.len() < NONCE_LEN {
            return Err("ciphertext too short for nonce".into());
        }
        let (nonce_bytes, body) = ciphertext.split_at(NONCE_LEN);
        let nonce = Nonce::from_slice(nonce_bytes);
        self.aead
            .decrypt(nonce, body)
            .map_err(|e| format!("aead decrypt: {e}"))
    }
}

pub fn hmac_hex(key: &[u8], msg: &str) -> String {
    let mut mac = <HmacSha256 as Mac>::new_from_slice(key).expect("hmac accepts any key");
    mac.update(msg.as_bytes());
    mac.finalize()
        .into_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `[u32 BE jsonLen][json]` — no payload framing; the caller appends
/// payload bytes directly.
pub fn write_frame(s: &mut TcpStream, frame: &Frame) -> Result<(), String> {
    let json = serde_json::to_vec(frame).map_err(|e| e.to_string())?;
    s.write_all(&(json.len() as u32).to_be_bytes())
        .map_err(|e| e.to_string())?;
    s.write_all(&json).map_err(|e| e.to_string())
}

pub fn read_frame(s: &mut TcpStream) -> Result<Frame, String> {
    let mut len_buf = [0u8; 4];
    s.read_exact(&mut len_buf).map_err(|e| e.to_string())?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_HEADER {
        return Err(format!("frame header too large: {len}"));
    }
    let mut buf = vec![0u8; len];
    s.read_exact(&mut buf).map_err(|e| e.to_string())?;
    serde_json::from_slice(&buf).map_err(|e| format!("bad frame json: {e}"))
}

/// Write one packet — plaintext `[jsonLen][json][payload]`, or the whole
/// inner buffer sealed when a session cipher is installed.
pub fn write_packet(
    s: &mut TcpStream,
    frame: &Frame,
    payload: &[u8],
    cipher: Option<&SessionCipher>,
) -> Result<(), String> {
    let json = serde_json::to_vec(frame).map_err(|e| e.to_string())?;
    match cipher {
        Some(c) => {
            let mut inner = Vec::with_capacity(4 + json.len() + payload.len());
            inner.extend_from_slice(&(json.len() as u32).to_be_bytes());
            inner.extend_from_slice(&json);
            inner.extend_from_slice(payload);
            let ct = c.encrypt(&inner)?;
            s.write_all(&(ct.len() as u32).to_be_bytes())
                .map_err(|e| e.to_string())?;
            s.write_all(&ct).map_err(|e| e.to_string())
        }
        None => {
            s.write_all(&(json.len() as u32).to_be_bytes())
                .map_err(|e| e.to_string())?;
            s.write_all(&json).map_err(|e| e.to_string())?;
            s.write_all(payload).map_err(|e| e.to_string())
        }
    }
}

/// Read one packet. On the plaintext path the payload size comes from
/// `shape` × `dtype`; on the sealed path the inner buffer is split at
/// the embedded `jsonLen`.
pub fn read_packet(
    s: &mut TcpStream,
    cipher: Option<&SessionCipher>,
) -> Result<(Frame, Vec<u8>), String> {
    match cipher {
        Some(c) => {
            let mut len_buf = [0u8; 4];
            s.read_exact(&mut len_buf).map_err(|e| e.to_string())?;
            let len = u32::from_be_bytes(len_buf) as usize;
            if len > MAX_PACKET {
                return Err(format!("encrypted frame too large: {len}"));
            }
            let mut ct = vec![0u8; len];
            s.read_exact(&mut ct).map_err(|e| e.to_string())?;
            let inner = c.decrypt(&ct)?;
            if inner.len() < 4 {
                return Err("decrypted frame too short".into());
            }
            let json_len = u32::from_be_bytes(inner[..4].try_into().unwrap()) as usize;
            if inner.len() < 4 + json_len {
                return Err("decrypted frame truncated".into());
            }
            let frame: Frame = serde_json::from_slice(&inner[4..4 + json_len])
                .map_err(|e| format!("bad frame json: {e}"))?;
            Ok((frame, inner[4 + json_len..].to_vec()))
        }
        None => {
            let frame = read_frame(s)?;
            let payload = if let Some(shape) = &frame.shape {
                let elems: usize = shape.iter().product();
                let size = elems * dtype_size(frame.dtype.as_deref().unwrap_or("float16"));
                let mut p = vec![0u8; size];
                s.read_exact(&mut p).map_err(|e| e.to_string())?;
                p
            } else {
                Vec::new()
            };
            Ok((frame, payload))
        }
    }
}

/// Client side of the mutual-auth handshake: answer the server's nonce,
/// then verify the counter-proof. Detects an auth-disabled server by its
/// challenge absence — callers that need auth should fail closed first.
pub fn client_handshake(s: &mut TcpStream, key: &[u8]) -> Result<(), String> {
    let challenge = read_frame(s)?;
    if challenge.op != "auth" {
        return Err(format!("expected auth challenge, got {}", challenge.op));
    }
    let nonce = challenge
        .text
        .ok_or_else(|| "auth challenge missing nonce".to_string())?;
    let proof = hmac_hex(key, &format!("mb-c:{nonce}"));
    let mut resp_f = Frame::new("auth");
    resp_f.text = Some(proof);
    write_frame(s, &resp_f)?;
    let resp = read_frame(s)?;
    match resp.op.as_str() {
        "auth-ok" => {
            let expect = hmac_hex(key, &format!("mb-s:{nonce}"));
            if resp.text.as_deref() != Some(expect.as_str()) {
                return Err("server counter-proof invalid — wrong mesh secret?".into());
            }
            Ok(())
        }
        "error" => Err(format!("auth rejected: {}", resp.error.unwrap_or_default())),
        other => Err(format!("expected auth-ok, got {other}")),
    }
}

/// Server side of the handshake — challenge, verify `mb-c:`, counter-
/// prove `mb-s:`. Returns the nonce for callers that bind session state
/// to it.
pub fn server_handshake(s: &mut TcpStream, key: &[u8]) -> Result<String, String> {
    let mut raw = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut raw))
        .map_err(|e| e.to_string())?;
    let nonce: String = raw.iter().map(|b| format!("{b:02x}")).collect();
    let mut ch = Frame::new("auth");
    ch.text = Some(nonce.clone());
    write_frame(s, &ch)?;
    let resp = read_frame(s)?;
    let expected = hmac_hex(key, &format!("mb-c:{nonce}"));
    if resp.op != "auth" || resp.text.as_deref() != Some(expected.as_str()) {
        return Err("peer auth failed".into());
    }
    let mut ok = Frame::new("auth-ok");
    ok.text = Some(hmac_hex(key, &format!("mb-s:{nonce}")));
    write_frame(s, &ok)?;
    Ok(nonce)
}

/// Shared-secret resolution — same precedence as Bad Apple:
/// env `COYOTE_MESH_KEY`/`BADAPPLE_MESH_KEY`/`BADAPPLE_P2P_SECRET`, then
/// the SLICKS key file. Minimum 32 bytes.
pub fn resolve_secret() -> Result<Vec<u8>, String> {
    for var in [
        "COYOTE_MESH_KEY",
        "BADAPPLE_MESH_KEY",
        "BADAPPLE_P2P_SECRET",
    ] {
        if let Ok(s) = std::env::var(var) {
            if s.is_empty() {
                return Err(format!("{var} is empty"));
            }
            if s.len() < 32 {
                return Err("mesh secret must be at least 32 bytes".into());
            }
            return Ok(s.into_bytes());
        }
    }
    let path = std::env::var("BADAPPLE_SLICKS_KEY_PATH")
        .unwrap_or_else(|_| "/var/lib/bad_apple/slicks.key".into());
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("no mesh secret in env and cannot read {path}: {e}"))?
        .trim()
        .to_string();
    if raw.len() < 32 {
        return Err("mesh secret must be at least 32 bytes".into());
    }
    Ok(raw.into_bytes())
}

/// Authenticated connection: TCP + handshake + optional session cipher.
/// `enc: true` seals every packet after auth; `false` authenticates
/// only (matches BADAPPLE_MESH_ENC=0 semantics).
pub fn connect(
    host: &str,
    secret: &[u8],
    enc: bool,
    timeout_secs: u64,
) -> Result<(TcpStream, Option<SessionCipher>), String> {
    let mut s = TcpStream::connect(host).map_err(|e| format!("connect {host}: {e}"))?;
    s.set_read_timeout(Some(Duration::from_secs(timeout_secs)))
        .ok();
    s.set_write_timeout(Some(Duration::from_secs(30))).ok();
    s.set_nodelay(true).ok();
    client_handshake(&mut s, secret)?;
    Ok((s, enc.then(|| SessionCipher::new(secret))))
}
