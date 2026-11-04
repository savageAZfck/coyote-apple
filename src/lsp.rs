//! LSP server — `coyote-apple lsp` speaks Language Server Protocol
//! over stdio (Content-Length framed JSON-RPC).
//!
//! Surface: full-document sync, diagnostics (parse + typecheck through
//! the real pipeline), document symbols, completion (keywords,
//! builtins, snippets, doc symbols), hover (declaration signatures),
//! go-to-definition, and semantic tokens.

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

fn pos(line: usize, col: usize) -> Value {
    json!({"line": line.saturating_sub(1), "character": col.saturating_sub(1)})
}

fn diag_json(msg: &str, line: usize, col: usize) -> Value {
    json!({
        "range": {"start": pos(line, col), "end": pos(line, col + 1)},
        "severity": 1, "source": "coyote", "message": msg,
    })
}

fn diagnostics(doc: &Doc) -> Vec<Value> {
    match parse_src_at(&doc.path, &doc.text) {
        Err(d) => vec![diag_json(&d.msg, d.line, d.col)],
        Ok((_, ast)) => match check_all(&ast) {
            Ok(_) => vec![],
            Err(d) => vec![diag_json(&d.msg, d.line, d.col)],
        },
    }
}

// --- symbol table ------------------------------------------------------------

struct Sym {
    name: String,
    line: usize,
    col: usize,
    kind: u32,
    /// hover text — a signature or decl summary.
    detail: String,
}

/// `f16[64, 64]`-style render for tensor `Ty`; host `HTy` via debug.
fn fmt_ty(t: &coyote::ast::Ty) -> String {
    let dt = format!("{:?}", t.dt).to_lowercase();
    if t.dims.is_empty() {
        return dt;
    }
    let dims: Vec<String> = t
        .dims
        .iter()
        .map(|d| match d {
            coyote::ast::Dim::Lit(n) => n.to_string(),
            coyote::ast::Dim::Name(s) => s.clone(),
        })
        .collect();
    format!("{dt}[{}]", dims.join(", "))
}

fn symbols_of(text: &str, path: &PathBuf) -> Vec<Sym> {
    let Ok((_, ast)) = parse_src_at(path, text) else {
        return vec![];
    };
    let mut out = Vec::new();
    for d in &ast.decls {
        let (name, line, col, kind, detail) = match d {
            Decl::Import { path: s, alias } => (
                s.v.clone(),
                s.line,
                s.col,
                2,
                format!(
                    "import \"{}\"{}",
                    s.v,
                    alias
                        .as_ref()
                        .map(|a| format!(" as {}", a.v))
                        .unwrap_or_default()
                ),
            ),
            Decl::Invariant(s) => (s.v.clone(), s.line, s.col, 7, format!("invariant {}", s.v)),
            Decl::Cap(v) => match v.first() {
                Some(s) => (
                    s.v.clone(),
                    s.line,
                    s.col,
                    7,
                    format!(
                        "cap {}",
                        v.iter()
                            .map(|c| c.v.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                ),
                None => continue,
            },
            Decl::Provenance(v) => match v.first() {
                Some((s, t)) => (
                    format!("{}", s.v),
                    s.line,
                    s.col,
                    7,
                    format!("provenance {} = {}", s.v, t.v),
                ),
                None => continue,
            },
            Decl::Param { name, ty, qual, .. } => (
                name.v.clone(),
                name.line,
                name.col,
                13,
                format!("param {}: {} ({:?})", name.v, fmt_ty(&ty.v), qual),
            ),
            Decl::Io { name, ty, dir, .. } => (
                name.v.clone(),
                name.line,
                name.col,
                13,
                format!("io {}: {} ({:?})", name.v, fmt_ty(&ty.v), dir),
            ),
            Decl::State { name, fields } => (
                name.v.clone(),
                name.line,
                name.col,
                5,
                format!("state {} ({} fields)", name.v, fields.len()),
            ),
            Decl::Table { name, .. } => (
                name.v.clone(),
                name.line,
                name.col,
                13,
                format!("table {}", name.v),
            ),
            Decl::Struct { name, .. } => (
                name.v.clone(),
                name.line,
                name.col,
                23,
                format!("struct {}", name.v),
            ),
            Decl::Enum { name, .. } => (
                name.v.clone(),
                name.line,
                name.col,
                10,
                format!("enum {}", name.v),
            ),
            Decl::ExternFn(f) => {
                let args: Vec<String> =
                    f.v.args
                        .iter()
                        .map(|(n, t)| format!("{}: {:?}", n.v, t.v))
                        .collect();
                let caps = if f.v.caps.is_empty() {
                    String::new()
                } else {
                    format!(
                        " cap {}",
                        f.v.caps
                            .iter()
                            .map(|c| c.v.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                };
                (
                    f.v.name.clone(),
                    f.line,
                    f.col,
                    12,
                    format!("extern fn {}({}){}", f.v.name, args.join(", "), caps),
                )
            }
            Decl::Fn(f) => {
                let t = f.v.target.name();
                let args: Vec<String> =
                    f.v.args
                        .iter()
                        .map(|(n, ty)| format!("{}: {}", n.v, fmt_ty(&ty.v)))
                        .collect();
                let ret =
                    f.v.ret
                        .as_ref()
                        .map(|r| format!(" -> {}", fmt_ty(&r.v)))
                        .unwrap_or_default();
                (
                    f.v.name.clone(),
                    f.line,
                    f.col,
                    12,
                    format!(
                        "{}fn {}({}){}",
                        if t == "ane" { "" } else { "gpu/enc " },
                        f.v.name,
                        args.join(", "),
                        ret
                    ),
                )
            }
            Decl::HostFn(f) => (
                f.v.name.clone(),
                f.line,
                f.col,
                12,
                format!("host fn {}", f.v.name),
            ),
            Decl::Treaty { name, allow, deny } => (
                name.v.clone(),
                name.line,
                name.col,
                7,
                format!(
                    "treaty {} (allow {}, deny {})",
                    name.v,
                    allow.len(),
                    deny.len()
                ),
            ),
        };
        out.push(Sym {
            name,
            line,
            col,
            kind,
            detail,
        });
    }
    out
}

fn symbols(doc: &Doc) -> Vec<Value> {
    symbols_of(&doc.text, &doc.path)
        .into_iter()
        .map(|s| {
            json!({
                "name": s.name, "kind": s.kind,
                "range": {"start": pos(s.line, s.col), "end": pos(s.line, s.col + 1)},
                "selectionRange": {"start": pos(s.line, s.col), "end": pos(s.line, s.col + 1)},
            })
        })
        .collect()
}

// --- cursor helpers ----------------------------------------------------------

fn word_at(doc: &Doc, line0: usize, col0: usize) -> String {
    let Some(line) = doc.text.lines().nth(line0) else {
        return String::new();
    };
    let chars: Vec<char> = line.chars().collect();
    let is_id = |c: char| c.is_alphanumeric() || c == '_' || c == '.';
    let mut lo = col0.min(chars.len());
    let mut hi = lo;
    while lo > 0 && is_id(chars[lo - 1]) {
        lo -= 1;
    }
    while hi < chars.len() && is_id(chars[hi]) {
        hi += 1;
    }
    chars[lo..hi].iter().collect()
}

// --- completion ----------------------------------------------------------------

const KEYWORDS: &[&str] = &[
    "fn",
    "gpu fn",
    "enc fn",
    "mesh fn",
    "pcc fn",
    "host fn",
    "det",
    "treaty",
    "peer",
    "param",
    "state",
    "io",
    "cap",
    "invariant",
    "provenance",
    "reflex",
    "struct",
    "enum",
    "match",
    "let",
    "var",
    "if",
    "else",
    "while",
    "for",
    "return",
    "import",
    "extern",
    "witness",
    "table",
    "break",
    "continue",
    "in",
    "true",
    "false",
];

const BUILTINS: &[&str] = &[
    "print",
    "witness",
    "argmax",
    "softmax",
    "zeros",
    "onehot",
    "randn",
    "len",
    "sum",
    "exp",
    "log",
    "tanh",
    "relu",
    "silu",
    "sigmoid",
    "matmul",
    "transpose",
    "slice",
    "concat",
    "max",
    "min",
    "abs",
];

fn completions(doc: &Doc) -> Vec<Value> {
    let mut out = Vec::new();
    for k in KEYWORDS {
        out.push(json!({"label": k, "kind": 14, "detail": "keyword"}));
    }
    for b in BUILTINS {
        out.push(json!({"label": b, "kind": 3, "detail": "builtin"}));
    }
    for s in symbols_of(&doc.text, &doc.path) {
        out.push(json!({"label": s.name, "kind": s.kind, "detail": s.detail}));
    }
    // snippets
    for (label, snip) in [
        (
            "fn (tensor program)",
            "fn ${1:name}(${2:x: f16[64]}) -> f16[64] {\n    $0\n}",
        ),
        (
            "gpu fn",
            "gpu fn ${1:name}(${2:x: f16[64]}) -> f16[64] {\n    $0\n}",
        ),
        (
            "enc fn",
            "enc fn ${1:name}(${2:x: f16[64]}) -> f16[64] {\n    $0\n}",
        ),
        (
            "host fn main",
            "host fn main() -> i64 {\n    $0\n    return 0;\n}",
        ),
        ("param", "param ${1:w}: f16[${2:64}, ${3:64}]"),
        (
            "state",
            "state ${1:kv} {\n    ${2:k}: f16[${3:256}, ${4:64}]\n}",
        ),
        ("witness", "witness(\"${1:tag}\", ${2:val});"),
    ] {
        out.push(json!({
            "label": label, "kind": 15, "detail": "snippet",
            "insertText": snip, "insertTextFormat": 2,
        }));
    }
    out
}

// --- semantic tokens ---------------------------------------------------------

/// token type legend index → LSP token type
const SEM_TYPES: &[&str] = &[
    "keyword",
    "type",
    "function",
    "variable",
    "parameter",
    "string",
    "number",
    "comment",
    "operator",
    "property",
];

/// Scan source into (line0, col0, len, type_idx) — self-contained
/// scanner (idents, numbers, strings, comments, punctuation). Keyword
/// set drives the keyword class; everything else is structural.
fn sem_tokens(text: &str) -> Vec<Value> {
    let kw: std::collections::HashSet<&str> = [
        "fn",
        "gpu",
        "enc",
        "host",
        "det",
        "param",
        "state",
        "io",
        "cap",
        "invariant",
        "provenance",
        "reflex",
        "struct",
        "enum",
        "match",
        "let",
        "var",
        "if",
        "else",
        "while",
        "for",
        "return",
        "import",
        "extern",
        "witness",
        "table",
        "break",
        "continue",
        "in",
        "true",
        "false",
        "f16",
        "f32",
        "i32",
        "i64",
        "u8",
        "bool",
        "str",
    ]
    .into_iter()
    .collect();
    let ty_kw: std::collections::HashSet<&str> = ["f16", "f32", "i32", "i64", "u8", "bool", "str"]
        .into_iter()
        .collect();
    let mut raw: Vec<(usize, usize, usize, usize)> = Vec::new();
    for (li, line) in text.lines().enumerate() {
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0usize;
        while i < chars.len() {
            let c = chars[i];
            if c == '/' && chars.get(i + 1) == Some(&'/') {
                raw.push((li, i, chars.len() - i, 7));
                break;
            }
            if c.is_whitespace() {
                i += 1;
                continue;
            }
            if c == '"' {
                let s = i;
                i += 1;
                while i < chars.len() && chars[i] != '"' {
                    i += if chars[i] == '\\' { 2 } else { 1 };
                }
                i = (i + 1).min(chars.len());
                raw.push((li, s, i - s, 5));
                continue;
            }
            if c.is_ascii_alphabetic() || c == '_' {
                let s = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let w: String = chars[s..i].iter().collect();
                let t = if ty_kw.contains(w.as_str()) {
                    1
                } else if kw.contains(w.as_str()) {
                    0
                } else {
                    // capitalized → type-ish, otherwise variable
                    if w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        1
                    } else {
                        3
                    }
                };
                raw.push((li, s, i - s, t));
                continue;
            }
            if c.is_ascii_digit() {
                let s = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '.') {
                    i += 1;
                }
                raw.push((li, s, i - s, 6));
                continue;
            }
            raw.push((li, i, 1, 8));
            i += 1;
        }
    }
    // delta encoding
    let mut out = Vec::new();
    let (mut pl, mut pc) = (0usize, 0usize);
    for (l, c, len, t) in raw {
        let dl = l - pl;
        let dc = if dl == 0 { c - pc } else { c };
        out.push(json!([dl, dc, len, t, 0]));
        pl = l;
        pc = c;
    }
    out
}

// --- framing + dispatch --------------------------------------------------------

fn read_message(r: &mut impl BufRead) -> Option<Vec<u8>> {
    let mut len = 0usize;
    loop {
        let mut line = String::new();
        let n = r.read_line(&mut line).ok()?;
        if n == 0 {
            return None;
        }
        let t = line.trim();
        if t.is_empty() {
            break;
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
    s = s.replace("%20", " ");
    PathBuf::from(s)
}

fn get_doc<'a>(docs: &'a mut HashMap<String, Doc>, uri: &str) -> Option<&'a Doc> {
    if !docs.contains_key(uri) {
        let p = uri_to_path(uri);
        if let Ok(text) = std::fs::read_to_string(&p) {
            docs.insert(uri.to_string(), Doc { path: p, text });
        }
    }
    docs.get(uri)
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
                            "jsonrpc": "2.0", "id": id,
                            "result": {
                                "capabilities": {
                                    "textDocumentSync": {"openClose": true, "change": 1},
                                    "documentSymbolProvider": true,
                                    "completionProvider": {"triggerCharacters": [" ", ":", "<"]},
                                    "hoverProvider": true,
                                    "definitionProvider": true,
                                    "semanticTokensProvider": {
                                        "legend": {
                                            "tokenTypes": SEM_TYPES,
                                            "tokenModifiers": [],
                                        },
                                        "full": true,
                                    },
                                },
                                "serverInfo": {"name": "coyote-apple-lsp", "version": env!("CARGO_PKG_VERSION")},
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
                let syms = get_doc(&mut docs, uri).map(symbols).unwrap_or_default();
                if let Some(id) = id {
                    send(
                        &mut out,
                        &json!({"jsonrpc": "2.0", "id": id, "result": syms}),
                    );
                }
            }
            "textDocument/completion" => {
                let uri = params["textDocument"]["uri"].as_str().unwrap_or("");
                let items = get_doc(&mut docs, uri).map(completions).unwrap_or_default();
                if let Some(id) = id {
                    send(
                        &mut out,
                        &json!({"jsonrpc": "2.0", "id": id, "result": items}),
                    );
                }
            }
            "textDocument/hover" => {
                let uri = params["textDocument"]["uri"].as_str().unwrap_or("");
                let line = params["position"]["line"].as_u64().unwrap_or(0) as usize;
                let ch = params["position"]["character"].as_u64().unwrap_or(0) as usize;
                let result = get_doc(&mut docs, uri).and_then(|d| {
                    let w = word_at(d, line, ch);
                    if w.is_empty() {
                        return None;
                    }
                    symbols_of(&d.text, &d.path)
                        .into_iter()
                        .find(|s| s.name == w || s.name.ends_with(&format!("{w}")))
                        .map(|s| {
                            json!({
                                "contents": {"kind": "markdown", "value": format!("```cy\n{}\n```", s.detail)},
                                "range": {
                                    "start": pos(line + 1, ch + 1),
                                    "end": pos(line + 1, ch + 1 + w.len()),
                                },
                            })
                        })
                });
                if let Some(id) = id {
                    send(
                        &mut out,
                        &json!({"jsonrpc": "2.0", "id": id, "result": result}),
                    );
                }
            }
            "textDocument/definition" => {
                let uri = params["textDocument"]["uri"].as_str().unwrap_or("");
                let line = params["position"]["line"].as_u64().unwrap_or(0) as usize;
                let ch = params["position"]["character"].as_u64().unwrap_or(0) as usize;
                let result = get_doc(&mut docs, uri).and_then(|d| {
                    let w = word_at(d, line, ch);
                    symbols_of(&d.text, &d.path)
                        .into_iter()
                        .find(|s| s.name == w)
                        .map(|s| {
                            json!({
                                "uri": uri,
                                "range": {
                                    "start": pos(s.line, s.col),
                                    "end": pos(s.line, s.col + 1),
                                },
                            })
                        })
                });
                if let Some(id) = id {
                    send(
                        &mut out,
                        &json!({"jsonrpc": "2.0", "id": id, "result": result}),
                    );
                }
            }
            "textDocument/semanticTokens/full" => {
                let uri = params["textDocument"]["uri"].as_str().unwrap_or("");
                let data = get_doc(&mut docs, uri)
                    .map(|d| sem_tokens(&d.text))
                    .unwrap_or_default();
                if let Some(id) = id {
                    send(
                        &mut out,
                        &json!({"jsonrpc": "2.0", "id": id, "result": {"data": data}}),
                    );
                }
            }
            _ => {
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
