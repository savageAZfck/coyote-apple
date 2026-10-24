//! Minimal LSP server — `coyote lsp` speaks Language Server Protocol
//! over stdio (Content-Length framed JSON-RPC).
//!
//! v1 surface: full-document sync, diagnostics (parse + typecheck),
//! and document symbols. Diagnostics run through `parse_src_at` and
//! `check_all` — the same pipeline `coyote check` uses — so what the
//! editor flags is what the compiler would reject.

use coyote::ast::Decl;
use coyote::parser::parse_src_at;
use coyote::typeck::check_all;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

struct Doc {
    path: PathBuf,
    text: String,
}

/// LSP positions are 0-based; coyote diags are 1-based (or 0 for
/// span-less errors — clamped).
fn pos(line: usize, col: usize) -> Value {
    json!({
        "line": line.saturating_sub(1),
        "character": col.saturating_sub(1),
    })
}

fn diag_json(msg: &str, line: usize, col: usize) -> Value {
    json!({
        "range": {
            "start": pos(line, col),
            "end": pos(line, col + 1),
        },
        "severity": 1,
        "source": "coyote",
        "message": msg,
    })
}

/// parse + typecheck a document. Parse stops at the first error;
/// typecheck runs when parse succeeds.
fn diagnostics(doc: &Doc) -> Vec<Value> {
    match parse_src_at(&doc.path, &doc.text) {
        Err(d) => vec![diag_json(&d.msg, d.line, d.col)],
        Ok((_, ast)) => match check_all(&ast) {
            Ok(_) => vec![],
            Err(d) => vec![diag_json(&d.msg, d.line, d.col)],
        },
    }
}

/// (name, line, col, SymbolKind) per top-level declaration.
fn symbols(doc: &Doc) -> Vec<Value> {
    let Ok((_, ast)) = parse_src_at(&doc.path, &doc.text) else {
        return vec![];
    };
    let mut out = Vec::new();
    for d in &ast.decls {
        // LSP SymbolKind: 2 Module, 5 Class, 7 Property, 10 Enum,
        // 12 Function, 13 Variable, 23 Struct
        let (name, line, col, kind) = match d {
            Decl::Import(s) => (s.v.clone(), s.line, s.col, 2),
            Decl::Invariant(s) => (s.v.clone(), s.line, s.col, 7),
            Decl::Cap(v) => match v.first() {
                Some(s) => (s.v.clone(), s.line, s.col, 7),
                None => continue,
            },
            Decl::Provenance(v) => match v.first() {
                Some((s, _)) => (format!("provenance:{}", s.v), s.line, s.col, 7),
                None => continue,
            },
            Decl::Param { name, .. } => (name.v.clone(), name.line, name.col, 13),
            Decl::Io { name, .. } => (name.v.clone(), name.line, name.col, 13),
            Decl::State { name, .. } => (name.v.clone(), name.line, name.col, 5),
            Decl::Table { name, .. } => (name.v.clone(), name.line, name.col, 13),
            Decl::Struct { name, .. } => (name.v.clone(), name.line, name.col, 23),
            Decl::Enum { name, .. } => (name.v.clone(), name.line, name.col, 10),
            Decl::ExternFn(f) => (f.v.name.clone(), f.line, f.col, 12),
            Decl::Fn(f) => (f.v.name.clone(), f.line, f.col, 12),
            Decl::HostFn(f) => (f.v.name.clone(), f.line, f.col, 12),
        };
        out.push(json!({
            "name": name,
            "kind": kind,
            "range": {"start": pos(line, col), "end": pos(line, col + 1)},
            "selectionRange": {"start": pos(line, col), "end": pos(line, col + 1)},
        }));
    }
    out
}

fn read_message(r: &mut impl BufRead) -> Option<Vec<u8>> {
    let mut len = 0usize;
    loop {
        let mut line = String::new();
        let n = r.read_line(&mut line).ok()?;
        if n == 0 {
            return None; // EOF
        }
        let t = line.trim();
        if t.is_empty() {
            break; // end of headers
        }
        if let Some(v) = t.strip_prefix("Content-Length:") {
            len = v.trim().parse().unwrap_or(0);
        }
    }
    if len == 0 {
        return None;
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body).ok()?;
    Some(body)
}

fn send(w: &mut impl Write, v: &Value) {
    let body = v.to_string();
    write!(w, "Content-Length: {}\r\n\r\n{}", body.len(), body).ok();
    w.flush().ok();
}

fn uri_to_path(uri: &str) -> PathBuf {
    let mut s = uri.strip_prefix("file://").unwrap_or(uri).to_string();
    // minimal percent-decode for spaces
    s = s.replace("%20", " ");
    #[cfg(windows)]
    if s.starts_with('/') && s.len() > 2 && s.as_bytes()[2] == b':' {
        s.remove(0);
    }
    PathBuf::from(s)
}

pub fn run() -> Result<(), String> {
    let stdin = std::io::stdin();
    let mut r = BufReader::new(stdin.lock());
    let mut out = std::io::stdout();
    let mut docs: HashMap<String, Doc> = HashMap::new();
    let mut shutdown = false;

    while let Some(body) = read_message(&mut r) {
        let Ok(msg) = serde_json::from_slice::<Value>(&body) else {
            continue;
        };
        let method = msg["method"].as_str().unwrap_or("");
        let id = msg.get("id").cloned();
        let params = &msg["params"];

        match method {
            "initialize" => {
                if let Some(id) = id {
                    send(
                        &mut out,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "capabilities": {
                                    "textDocumentSync": {"openClose": true, "change": 1},
                                    "documentSymbolProvider": true,
                                },
                                "serverInfo": {"name": "coyote-lsp", "version": env!("CARGO_PKG_VERSION")},
                            }
                        }),
                    );
                }
            }
            "initialized" => {}
            "shutdown" => {
                shutdown = true;
                if let Some(id) = id {
                    send(
                        &mut out,
                        &json!({"jsonrpc": "2.0", "id": id, "result": null}),
                    );
                }
            }
            "exit" => std::process::exit(if shutdown { 0 } else { 1 }),
            "textDocument/didOpen" | "textDocument/didChange" => {
                let td = &params["textDocument"];
                let uri = td["uri"].as_str().unwrap_or("").to_string();
                let text = if method.ends_with("didOpen") {
                    td["text"].as_str().unwrap_or("").to_string()
                } else {
                    params["contentChanges"][0]["text"]
                        .as_str()
                        .unwrap_or("")
                        .to_string()
                };
                let doc = Doc {
                    path: uri_to_path(&uri),
                    text,
                };
                let diags = diagnostics(&doc);
                docs.insert(uri.clone(), doc);
                send(
                    &mut out,
                    &json!({
                        "jsonrpc": "2.0",
                        "method": "textDocument/publishDiagnostics",
                        "params": {"uri": uri, "diagnostics": diags},
                    }),
                );
            }
            "textDocument/didClose" => {
                let uri = params["textDocument"]["uri"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                docs.remove(&uri);
                send(
                    &mut out,
                    &json!({
                        "jsonrpc": "2.0",
                        "method": "textDocument/publishDiagnostics",
                        "params": {"uri": uri, "diagnostics": []},
                    }),
                );
            }
            "textDocument/documentSymbol" => {
                let uri = params["textDocument"]["uri"].as_str().unwrap_or("");
                if !docs.contains_key(uri) {
                    // fall back to reading the file if never opened
                    let p = uri_to_path(uri);
                    if let Ok(text) = std::fs::read_to_string(&p) {
                        docs.insert(uri.to_string(), Doc { path: p, text });
                    }
                }
                let syms = docs.get(uri).map(symbols).unwrap_or_default();
                if let Some(id) = id {
                    send(
                        &mut out,
                        &json!({"jsonrpc": "2.0", "id": id, "result": syms}),
                    );
                }
            }
            _ => {
                // unknown request → respond null so the client isn't stuck;
                // notifications are ignored.
                if let Some(id) = id {
                    send(
                        &mut out,
                        &json!({"jsonrpc": "2.0", "id": id, "result": null}),
                    );
                }
            }
        }
    }
    Ok(())
}
