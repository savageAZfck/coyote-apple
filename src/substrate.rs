//! substrate.rs — coyote-apple's path to the resident mind.
//!
//! Speaks SLICKS over the daemon's Unix socket:
//! Hello → Challenge → Execute → Accepted → Token* → Done | Response | Error.
//! Authentication mirrors the reference client: v2 (identity-agent
//! signature, Secure Enclave) when the agent socket is up, v1
//! (shared-key HMAC) otherwise. Prompts may be plain text or the
//! `__BADAPPLE_AGENT__` JSON-RPC envelope that reaches the daemon's
//! agent-method surface (`inference`, `invoke_tool`, `run_agent_task`,
//! `audit_event`, `list_models`, …).

use hmac::{Hmac, Mac};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const SLICKS_V1: u8 = 1;
const SLICKS_V2: u8 = 2;
const MAX_FRAME_BYTES: usize = 1024 * 1024;
const MAX_PROMPT_BYTES: usize = 64 * 1024;
const MAX_NEW_TOKENS: usize = 4096;
const HANDSHAKE_MAX_SKEW_MS: u64 = 30_000;

type HmacSha256 = Hmac<Sha256>;

fn socket_path() -> PathBuf {
    std::env::var_os("BADAPPLE_SOCKET_PATH")
        .map_or_else(|| PathBuf::from("/var/run/badapple/substrate.sock"), PathBuf::from)
}

fn identity_path() -> PathBuf {
    std::env::var_os("BADAPPLE_IDENTITY_AGENT_SOCKET")
        .map_or_else(|| PathBuf::from("/var/run/badapple/identity.sock"), PathBuf::from)
}

fn key_path() -> PathBuf {
    std::env::var_os("BADAPPLE_SLICKS_KEY_PATH")
        .map_or_else(|| PathBuf::from("/var/lib/bad_apple/slicks.key"), PathBuf::from)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn decode_hex(v: &str) -> Option<Vec<u8>> {
    if v.is_empty() || v.len() % 2 != 0 {
        return None;
    }
    v.as_bytes()
        .chunks_exact(2)
        .map(|p| std::str::from_utf8(p).ok().and_then(|s| u8::from_str_radix(s, 16).ok()))
        .collect()
}

fn random_nonce() -> String {
    let mut b = [0u8; 32];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        let _ = f.read_exact(&mut b);
    }
    hex_encode(&b)
}

fn nonce_ok(n: &str) -> bool {
    n.len() == 64 && n.bytes().all(|b| b.is_ascii_hexdigit())
}

fn hmac_hex(secret: &[u8], material: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(secret).expect("hmac accepts any key length");
    mac.update(material.as_bytes());
    hex_encode(&mac.finalize().into_bytes())
}

fn hmac_verify(secret: &[u8], material: &str, proof: &str) -> bool {
    let Some(proof) = decode_hex(proof) else { return false };
    let Ok(mut mac) = HmacSha256::new_from_slice(secret) else { return false };
    mac.update(material.as_bytes());
    mac.verify_slice(&proof).is_ok()
}

fn sha256_hex(b: &[u8]) -> String {
    hex_encode(&Sha256::digest(b))
}

// ── identity agent (v2: Secure Enclave signing) ─────────────────────────

fn identity_call(req: &Value) -> Result<Value, String> {
    let path = identity_path();
    let stream = UnixStream::connect(&path)
        .map_err(|e| format!("identity agent connect {}: {e}", path.display()))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(10)))
        .map_err(|e| e.to_string())?;
    let mut writer = stream.try_clone().map_err(|e| e.to_string())?;
    let mut frame = serde_json::to_vec(req).map_err(|e| e.to_string())?;
    frame.push(b'\n');
    writer.write_all(&frame).map_err(|e| e.to_string())?;
    writer.flush().map_err(|e| e.to_string())?;
    let mut line = String::new();
    let n = BufReader::new(stream)
        .read_line(&mut line)
        .map_err(|e| e.to_string())?;
    if n == 0 {
        return Err("identity agent closed the connection".into());
    }
    serde_json::from_str(&line).map_err(|e| format!("identity agent frame: {e}"))
}

fn identity_public_key() -> Option<String> {
    let resp = identity_call(&json!({"command": "public_key"})).ok()?;
    if resp.get("ok").and_then(Value::as_bool).unwrap_or(false) {
        resp.get("public_key")
            .and_then(Value::as_str)
            .map(String::from)
    } else {
        None
    }
}

fn identity_sign(material: &str) -> Result<String, String> {
    let b64 = base64_encode(material.as_bytes());
    let resp = identity_call(&json!({"command": "sign", "message_b64": b64}))?;
    if !resp.get("ok").and_then(Value::as_bool).unwrap_or(false) {
        return Err(format!(
            "identity sign failed: {}",
            resp.get("error").and_then(Value::as_str).unwrap_or("unknown")
        ));
    }
    resp.get("signature")
        .and_then(Value::as_str)
        .map(String::from)
        .ok_or_else(|| "identity agent returned no signature".into())
}

fn identity_verify(material: &str, signature: &str, pubkey: &str) -> Result<bool, String> {
    let b64 = base64_encode(material.as_bytes());
    let resp = identity_call(&json!({
        "command": "verify",
        "message_b64": b64,
        "signature": signature,
        "public_key": pubkey,
    }))?;
    if !resp.get("ok").and_then(Value::as_bool).unwrap_or(false) {
        return Err(format!(
            "identity verify failed: {}",
            resp.get("error").and_then(Value::as_str).unwrap_or("unknown")
        ));
    }
    Ok(resp.get("valid").and_then(Value::as_bool).unwrap_or(false))
}

// base64 (standard alphabet, padded) — one tiny helper instead of a dep.
fn base64_encode(b: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(b.len().div_ceil(3) * 4);
    for chunk in b.chunks(3) {
        let n = chunk.len();
        let buf = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let v = ((buf[0] as u32) << 16) | ((buf[1] as u32) << 8) | buf[2] as u32;
        out.push(A[(v >> 18) as usize & 63] as char);
        out.push(A[(v >> 12) as usize & 63] as char);
        out.push(if n > 1 { A[(v >> 6) as usize & 63] as char } else { '=' });
        out.push(if n > 2 { A[v as usize & 63] as char } else { '=' });
    }
    out
}

// ── auth mode resolution (v2 when the identity agent is up, else v1) ────

enum Mode {
    V1(Vec<u8>),
    V2 { client_pubkey: String },
}

fn load_secret() -> Result<Vec<u8>, String> {
    let raw = match std::env::var_os("BADAPPLE_SLICKS_SECRET") {
        Some(s) => s.to_string_lossy().into_owned(),
        None => std::fs::read_to_string(key_path())
            .map_err(|e| format!("unable to read the SLICKS key at {}: {e}", key_path().display()))?,
    };
    let trimmed = raw.trim();
    let secret = decode_hex(trimmed).unwrap_or_else(|| trimmed.as_bytes().to_vec());
    if secret.len() < 16 {
        return Err("SLICKS key must contain at least 128 bits".into());
    }
    Ok(secret)
}

fn resolve_mode() -> Result<Mode, String> {
    match std::env::var("BADAPPLE_SLICKS2").as_deref() {
        Ok("0" | "false" | "no" | "off" | "disable" | "disabled") => {
            Ok(Mode::V1(load_secret()?))
        }
        Ok("1" | "true" | "yes" | "on" | "enable" | "enabled") => {
            let pk = identity_public_key()
                .ok_or("BADAPPLE_SLICKS2=1 requires a running identity agent")?;
            Ok(Mode::V2 { client_pubkey: pk })
        }
        Ok(other) => Err(format!("BADAPPLE_SLICKS2 unrecognized value '{other}'")),
        Err(_) => match identity_public_key() {
            Some(pk) => Ok(Mode::V2 { client_pubkey: pk }),
            None => Ok(Mode::V1(load_secret()?)),
        },
    }
}

// ── frames ──────────────────────────────────────────────────────────────

fn write_frame(stream: &mut UnixStream, v: &Value) -> Result<(), String> {
    let mut frame = serde_json::to_vec(v).map_err(|e| e.to_string())?;
    if frame.len() > MAX_FRAME_BYTES {
        return Err("substrate frame exceeds the size limit".into());
    }
    frame.push(b'\n');
    stream.write_all(&frame).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())
}

fn read_frame(reader: &mut BufReader<UnixStream>) -> Result<Value, String> {
    let mut line = String::new();
    let n = Read::take(reader, (MAX_FRAME_BYTES + 1) as u64)
        .read_line(&mut line)
        .map_err(|e| e.to_string())?;
    if n == 0 {
        return Err("substrate closed the connection".into());
    }
    if n > MAX_FRAME_BYTES || !line.ends_with('\n') {
        return Err("substrate frame exceeds the size limit or is unterminated".into());
    }
    serde_json::from_str(&line).map_err(|e| format!("invalid substrate frame: {e}"))
}

// ── the query ────────────────────────────────────────────────────────────

/// One SLICKS round-trip. Returns (final_text, metrics) — `metrics`
/// arrives on Done frames; agent-method calls come back as Response
/// with `text` holding the serialized result.
pub fn query(prompt: &str, max_new_tokens: usize) -> Result<(String, Option<Value>), String> {
    if prompt.trim().is_empty() {
        return Err("prompt must not be empty".into());
    }
    if prompt.len() > MAX_PROMPT_BYTES {
        return Err(format!("prompt exceeds the {MAX_PROMPT_BYTES}-byte limit"));
    }
    if !(1..=MAX_NEW_TOKENS).contains(&max_new_tokens) {
        return Err(format!("max_new_tokens must be between 1 and {MAX_NEW_TOKENS}"));
    }
    let mode = resolve_mode()?;
    let ts = now_ms();
    let cnonce = random_nonce();
    let path = socket_path();
    let mut stream = UnixStream::connect(&path).map_err(|e| {
        format!(
            "the substrate is unreachable at {} — is Bad Apple running? ({e})",
            path.display()
        )
    })?;
    stream
        .set_read_timeout(Some(Duration::from_secs(45 * 60)))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(30)))
        .map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);

    // ── hello ──
    let (version, client_pubkey) = match &mode {
        Mode::V1(_) => (SLICKS_V1, None),
        Mode::V2 { client_pubkey } => (SLICKS_V2, Some(client_pubkey.clone())),
    };
    let mut hello = json!({
        "type": "hello",
        "version": version,
        "timestamp_ms": ts,
        "client_nonce": cnonce,
    });
    if let Some(pk) = &client_pubkey {
        hello["client_pubkey"] = json!(pk);
    }
    write_frame(&mut stream, &hello)?;

    // ── challenge ──
    let ch = read_frame(&mut reader)?;
    if ch.get("type").and_then(Value::as_str) == Some("error") {
        return Err(format!(
            "substrate rejected the handshake: {}",
            ch.get("message").and_then(Value::as_str).unwrap_or("")
        ));
    }
    if ch.get("type").and_then(Value::as_str) != Some("challenge") {
        return Err("substrate returned an invalid challenge".into());
    }
    let snonce = ch
        .get("server_nonce")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if !nonce_ok(&snonce) {
        return Err("substrate challenge nonce is invalid".into());
    }
    let proof = ch.get("proof").and_then(Value::as_str).unwrap_or("");
    match &mode {
        Mode::V1(secret) => {
            let material =
                format!("BADAPPLE-SLICKS/{SLICKS_V1}|server|{ts}|{cnonce}|{snonce}");
            if !hmac_verify(secret, &material, proof) {
                return Err("substrate failed server authentication".into());
            }
        }
        Mode::V2 { .. } => {
            let server_pubkey = ch
                .get("server_pubkey")
                .and_then(Value::as_str)
                .ok_or("v2 challenge missing server public key")?;
            let material =
                format!("BADAPPLE-SLICKS/{SLICKS_V2}|server|{ts}|{cnonce}|{snonce}");
            if !identity_verify(&material, proof, server_pubkey)? {
                return Err("substrate failed v2 server authentication".into());
            }
        }
    }

    // ── execute ──
    let proof = match &mode {
        Mode::V1(secret) => hmac_hex(
            secret,
            &format!(
                "BADAPPLE-SLICKS/{SLICKS_V1}|client|{ts}|{cnonce}|{snonce}|{max_new_tokens}|{}",
                sha256_hex(prompt.as_bytes())
            ),
        ),
        Mode::V2 { .. } => identity_sign(&format!(
            "BADAPPLE-SLICKS/{SLICKS_V2}|client|{ts}|{cnonce}|{snonce}|{max_new_tokens}|{}",
            sha256_hex(prompt.as_bytes())
        ))?,
    };
    // the server enforces timestamp freshness — the v2 path can stall
    // in the identity agent (Secure Enclave signing may wait on user
    // interaction); if the handshake has already run past the skew
    // bound, fail fast instead of sending a proof the gatekeeper
    // will reject as stale.
    if now_ms().saturating_sub(ts) > HANDSHAKE_MAX_SKEW_MS {
        return Err("SLICKS handshake exceeded the 30s skew bound — retry".into());
    }
    let mut exec = json!({
        "type": "execute",
        "version": version,
        "timestamp_ms": ts,
        "client_nonce": cnonce,
        "server_nonce": snonce,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "proof": proof,
    });
    if let Some(pk) = &client_pubkey {
        exec["client_pubkey"] = json!(pk);
    }
    write_frame(&mut stream, &exec)?;

    // ── stream ──
    let mut accepted = false;
    let mut text = String::new();
    loop {
        let f = read_frame(&mut reader)?;
        match f.get("type").and_then(Value::as_str) {
            Some("accepted") => accepted = true,
            Some("token") if accepted => {
                if let Some(t) = f.get("text").and_then(Value::as_str) {
                    text.push_str(t);
                }
            }
            Some("done") if accepted => {
                let final_text = f
                    .get("text")
                    .and_then(Value::as_str)
                    .map(String::from)
                    .unwrap_or(text);
                return Ok((final_text, f.get("metrics").cloned()));
            }
            Some("response") if accepted => {
                let result = f.get("result").cloned().unwrap_or(Value::Null);
                return Ok((serde_json::to_string(&result).unwrap_or_default(), None));
            }
            Some("error") => {
                return Err(format!(
                    "substrate request failed: {}",
                    f.get("message").and_then(Value::as_str).unwrap_or("")
                ));
            }
            _ => return Err("substrate returned an out-of-order frame".into()),
        }
    }
}

/// JSON-RPC through the agent envelope — reaches `inference`,
/// `invoke_tool`, `run_agent_task`, `audit_event`, `list_models`, …
pub fn call_agent(method: &str, params: Option<Value>, max_new_tokens: usize) -> Result<Value, String> {
    let mut req = serde_json::Map::new();
    req.insert("id".into(), Value::String("coyote-1".into()));
    req.insert("method".into(), Value::String(method.into()));
    if let Some(p) = params {
        req.insert("params".into(), p);
    }
    let prompt = format!(
        "__BADAPPLE_AGENT__ {}",
        serde_json::to_string(&Value::Object(req)).map_err(|e| e.to_string())?
    );
    let (text, _) = query(&prompt, max_new_tokens)?;
    serde_json::from_str(&text).or(Ok(Value::String(text)))
}

/// `inference` agent method — raw generation on the resident brain.
/// `system` empty ⇒ the daemon's normal generate path; `brain` selects
/// the chamber ("auto" default).
pub fn infer(
    prompt: &str,
    system: &str,
    max_tokens: usize,
    temperature: f64,
    brain: &str,
) -> Result<Value, String> {
    call_agent(
        "inference",
        Some(json!({
            "prompt": prompt,
            "system_prompt": system,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "brain": brain,
        })),
        max_tokens.max(256),
    )
}

/// Is the substrate socket reachable right now?
pub fn available() -> bool {
    socket_path().exists() && UnixStream::connect(socket_path()).is_ok()
}

/// `cortex_ask` — the live-mind query behind `mind_cortex`/`ask`.
/// Returns "" on failure so callers can fall back deterministically.
pub fn cortex_ask(prompt: &str) -> String {
    if !available() {
        return String::new();
    }
    match infer(prompt, "", 512, 0.0, "auto") {
        Ok(v) => v
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        Err(_) => String::new(),
    }
}
