//! coyote-apple repl — the distribution's interactive shell.
//!
//! Same two-machine model as the language shell:
//!
//!   * the **live** VM — the loaded organism. `:call`, `:main`,
//!     `:progs`, `:params`, `:w` all operate on it; witness entries
//!     and mutated state are real.
//!   * the **scratch** VM — a host-stratum-only session built from
//!     the file's source plus whatever declarations you add at the
//!     prompt.
//!
//! This distribution's copy exists for one reason: every VM built
//! here — live, bare, or per-line scratch — gets the `mind` extern
//! layer bound before it executes, so `mind::think` works at the
//! prompt exactly like it does under `run`.

use crate::mind;
use coyote::lower::sha256_hex;
use coyote::vm::{Val, Vm};
use coyote::{load_vm, parser, typeck};
use std::cell::RefCell;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::rc::Rc;

const BANNER: &str = "coyote-apple repl — source to silicon, interactively
  :help            this text
  :fns             host fns + signatures
  :progs           tensor programs + backend + io
  :params          persistent tensors (params/tables/state) + dims
  name(a, b, \"s\")    call a host fn on the live VM  (:call works too)
  :main            run host fn main
  :step tok [tok…] feed decoder tokens one step each (:decode N)
  :w [n]           last n witness entries
  :reset           reset program state surfaces
  :load file.cy    reload organism + reset session
  :src             show session source size / accumulated decls
  :q               quit
  anything else    eval as a host-fn body, or a decl to keep
";

/// `coyote-apple repl [file.cy] [-o dir] [--cpu] [--rebuild]`
pub fn run(args: &[String]) -> Result<(), i32> {
    let file = args
        .first()
        .filter(|a| !a.starts_with('-'))
        .map(PathBuf::from);
    let dir = args
        .iter()
        .position(|a| a == "-o")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("coyote_build"));
    let cpu = args.iter().any(|a| a == "--cpu");
    let rebuild = args.iter().any(|a| a == "--rebuild");
    shell(file.as_deref(), &dir, cpu, rebuild).map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}

/// depth-aware split on top-level commas (strings, brackets, braces).
fn split_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let (mut depth, mut in_str, mut start) = (0i32, false, 0usize);
    for (i, ch) in s.char_indices() {
        match ch {
            '"' => in_str = !in_str,
            '(' | '[' | '{' if !in_str => depth += 1,
            ')' | ']' | '}' if !in_str => depth -= 1,
            ',' if !in_str && depth == 0 => {
                out.push(s[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    let tail = s[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    out
}

/// scalar/vector literal → Val. i64, f64, bool, "str", [..].
fn parse_val(s: &str) -> Result<Val, String> {
    let s = s.trim();
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        return Ok(Val::S(Rc::from(&s[1..s.len() - 1])));
    }
    match s {
        "true" => return Ok(Val::B(true)),
        "false" => return Ok(Val::B(false)),
        _ => {}
    }
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        let items: Vec<Val> = if inner.trim().is_empty() {
            Vec::new()
        } else {
            split_args(inner)
                .iter()
                .map(|p| parse_val(p))
                .collect::<Result<_, _>>()?
        };
        return Ok(Val::V(Rc::new(RefCell::new(items))));
    }
    if let Ok(i) = s.parse::<i64>() {
        return Ok(Val::I(i));
    }
    if let Ok(f) = s.parse::<f64>() {
        return Ok(Val::F(f));
    }
    Err(format!(
        "can't parse `{s}` — try i64, f64, \"str\", bool, or [..]"
    ))
}

/// split a statement line at the last top-level `;` — `(head, tail)` when
/// the tail looks like a value position (`let x = 1; x + 1` → `x + 1`).
fn split_last_stmt(line: &str) -> Option<(String, String)> {
    let (mut depth, mut in_str, mut last) = (0i32, false, None);
    for (i, ch) in line.char_indices() {
        match ch {
            '"' => in_str = !in_str,
            '(' | '[' | '{' if !in_str => depth += 1,
            ')' | ']' | '}' if !in_str => depth -= 1,
            ';' if !in_str && depth == 0 => last = Some(i),
            _ => {}
        }
    }
    let i = last?;
    let tail = line[i + 1..].trim();
    if tail.is_empty() || tail.starts_with('}') {
        return None;
    }
    Some((line[..=i].to_string(), tail.to_string()))
}

/// `name(a, b, …)` → (name, args). Requires the full line be one call.
fn parse_call(line: &str) -> Option<(String, Vec<Val>)> {
    let line = line.trim();
    let open = line.find('(')?;
    if !line.ends_with(')') {
        return None;
    }
    let name = line[..open].trim();
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
    {
        return None;
    }
    let inner = &line[open + 1..line.len() - 1];
    let args = if inner.trim().is_empty() {
        Vec::new()
    } else {
        split_args(inner)
            .iter()
            .map(|p| parse_val(p))
            .collect::<Result<Vec<_>, _>>()
            .ok()?
    };
    Some((name.to_string(), args))
}

/// lines that extend the session source verbatim rather than evaluating.
fn looks_like_decl(line: &str) -> bool {
    let t = line.trim_start();
    [
        "host",
        "reflex",
        "struct",
        "enum",
        "param",
        "table",
        "state",
        "io",
        "extern",
        "invariant",
        "import",
        "fn",
    ]
    .iter()
    .any(|p| t == *p || t.starts_with(&format!("{p} ")))
}

struct Session {
    /// the live organism VM (what :call/:main/witness drive)
    vm: Vm,
    /// file source + accumulated repl decls — scratch compiles from this
    src: String,
    /// path the session source is anchored at — imports in decls and
    /// eval lines resolve relative to it (and CYO_PATH)
    anchor: PathBuf,
    /// decls added at the prompt, for :src listing
    decls: Vec<String>,
    /// autoregressive position for :step
    pos: usize,
}

impl Session {
    fn load(file: &Path, out: &Path, cpu: bool, rebuild: bool) -> Result<Self, String> {
        let mut vm = load_vm(file, out, cpu, rebuild, false)?;
        mind::bind_all(&mut vm);
        let src = std::fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
        Ok(Self {
            vm,
            src,
            anchor: file.to_path_buf(),
            decls: Vec::new(),
            pos: 0,
        })
    }

    fn bare(out: &Path) -> Result<Self, String> {
        // no file — session starts empty; scratch eval only. the ping fn
        // guarantees a host stratum so the VM can always be built.
        let src = "// coyote-apple repl session\n\nhost fn __repl_ping() -> i64 {\n    return 0;\n}\n"
            .to_string();
        let anchor = out.join("__repl_session.cy");
        let h = check_src(&anchor, &src)?;
        let host = h.host.ok_or("internal: no host stratum")?;
        let mut vm = Vm::new(&host, Vec::new(), &sha256_hex(src.as_bytes()))?;
        mind::bind_all(&mut vm);
        Ok(Self {
            vm,
            src,
            anchor,
            decls: Vec::new(),
            pos: 0,
        })
    }

    /// extend session source; returns error and rolls back on failure.
    fn add_decl(&mut self, decl: &str) -> Result<String, String> {
        let start = appended_line(&self.src);
        let candidate = format!("{}\n\n{decl}\n", self.src);
        check_src(&self.anchor, &candidate).map_err(|e| rebase(&e, start))?; // validate before committing
        self.src = candidate;
        let label = decl.lines().next().unwrap_or(decl).trim().to_string();
        self.decls.push(label.clone());
        Ok(label)
    }
}

/// Typecheck `src` with imports spliced — anchored at `path` so
/// `import "mind.cy"` inside session code resolves relative to the
/// session file's directory and CYO_PATH, same as `build` would.
fn check_src(path: &Path, src: &str) -> Result<typeck::Checked, String> {
    let (_combined, ast) = parser::parse_src_at(path, src).map_err(|d| d.to_string())?;
    typeck::check_all(&ast).map_err(|d| d.to_string())
}

/// count of '\n' — absolute line a freshly-appended decl/body lands on
/// (`{src}\n\n` + text → line nl(src) + 3).
fn appended_line(src: &str) -> usize {
    src.matches('\n').count() + 3
}

/// `line:col: msg` → rebased against the typed input. Errors landing
/// above the input point back at accumulated session code.
fn rebase(err: &str, input_start: usize) -> String {
    let Some((l, rest)) = err.split_once(':') else {
        return err.to_string();
    };
    let Some((c, m)) = rest.split_once(": ") else {
        return err.to_string();
    };
    let (Ok(l), Ok(c)) = (l.parse::<usize>(), c.parse::<usize>()) else {
        return err.to_string();
    };
    if l < input_start {
        format!("session line {l}: {m}")
    } else {
        let rel = (l - input_start + 1).max(1);
        format!("{rel}:{c}: {m}")
    }
}

fn show_call(vm: &mut Vm, line: &str) -> Result<(), String> {
    let Some((name, args)) = parse_call(line.strip_prefix(":call").unwrap_or(line)) else {
        return Err("usage: name(arg, …) — scalars: i64, f64, \"str\", bool, [..]".into());
    };
    let v = vm.call_fn(&name, args)?;
    println!("{}", vm.stringify(&v));
    Ok(())
}

fn show_progs(vm: &Vm) {
    let names = vm.prog_names();
    if names.is_empty() {
        println!("  (no tensor programs)");
        return;
    }
    for n in names {
        let backend = if vm.is_ane() { "ane" } else { "cpu" };
        match vm.prog_ios(n) {
            Some(ios) => {
                let io = ios
                    .iter()
                    .map(|(n, d)| format!("{n}{d:?}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                println!("  {n:<16} {backend}  io: {io}");
            }
            None => println!("  {n:<16} {backend}"),
        }
    }
}

fn show_params(vm: &Vm) {
    let mut names = vm.param_names();
    names.sort();
    if names.is_empty() {
        println!("  (no persistent tensors)");
        return;
    }
    for n in names {
        match vm.param_slice(&n) {
            Some(s) => {
                let head: Vec<String> = s.iter().take(4).map(|v| format!("{v:.4}")).collect();
                println!("  {n:<24} [{} elems]  {head:?}…", s.len());
            }
            None => println!("  {n:<24} (unresolved)"),
        }
    }
}

fn show_witness(vm: &Vm, tail: Option<usize>) {
    let w = vm.witness_log();
    if w.is_empty() {
        println!("  (witness log empty)");
        return;
    }
    let start = tail.map(|n| w.len().saturating_sub(n)).unwrap_or(0);
    for e in &w[start..] {
        println!(
            "  #{:<3} {:<20} {:<40} {}",
            e.seq,
            e.tag,
            e.val,
            &e.hash[..16]
        );
    }
    if start > 0 {
        println!("  … {} earlier entries", start);
    }
}

fn shell(file: Option<&Path>, out: &Path, cpu: bool, rebuild: bool) -> Result<(), String> {
    let mut s = match file {
        Some(f) => Session::load(f, out, cpu, rebuild)?,
        None => Session::bare(out)?,
    };
    let where_str = file
        .map(|f| format!("{}", f.display()))
        .unwrap_or_else(|| "empty session".to_string());
    println!("coyote-apple repl — {where_str}  (:help, :q)");

    let stdin = std::io::stdin();
    loop {
        print!("cy> ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) | Err(_) => break, // eof / read error
            _ => {}
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let r: Result<(), String> = (|| {
            if let Some(rest) = line.strip_prefix(':') {
                let mut it = rest.split_whitespace();
                match it.next().unwrap_or("") {
                    "q" | "quit" | "exit" => return Err(String::new()), // sentinel: clean quit
                    "h" | "help" | "?" => print!("{BANNER}"),
                    "fns" => {
                        let fns = s.vm.host_fns();
                        if fns.is_empty() {
                            println!("  (no host fns)");
                        }
                        for (_, sig, reflex) in fns {
                            let tag = if reflex { "  [reflex]" } else { "" };
                            println!("  {sig}{tag}");
                        }
                    }
                    "progs" => show_progs(&s.vm),
                    "params" => show_params(&s.vm),
                    "w" | "witness" => {
                        let n = it.next().and_then(|x| x.parse::<usize>().ok());
                        show_witness(&s.vm, n);
                    }
                    "main" => {
                        let v = s.vm.run_main()?;
                        println!("main() → {}", s.vm.stringify(&v));
                    }
                    "reset" => {
                        s.vm.reset_states();
                        s.pos = 0;
                        println!("✓ state surfaces zeroed");
                    }
                    "step" => {
                        let toks: Vec<u32> = it.filter_map(|t| t.parse::<u32>().ok()).collect();
                        if toks.is_empty() {
                            return Err("usage: :step <token_id> [token_id…]".into());
                        }
                        for (i, t) in toks.iter().enumerate() {
                            let project = i + 1 == toks.len();
                            match s.vm.step_decoder_token(*t, s.pos, project) {
                                Ok(Some(tok)) => println!("  pos {} tok {t} → {tok}", s.pos),
                                Ok(None) => println!("  pos {} tok {t} → (no projection)", s.pos),
                                Err(e) => return Err(e),
                            }
                            s.pos += 1;
                        }
                    }
                    "load" => {
                        let f = it.next().ok_or("usage: :load <file.cy>")?;
                        s = Session::load(Path::new(f), out, cpu, rebuild)?;
                        println!("✓ loaded {f}");
                    }
                    "src" => {
                        println!("  {} bytes session source", s.src.len());
                        for d in &s.decls {
                            println!("    + {d}");
                        }
                    }
                    "call" => show_call(&mut s.vm, line)?,
                    other => return Err(format!("unknown :{other} — :help lists commands")),
                }
                return Ok(());
            }
            // bare `name(...)` → live call when the organism has the fn;
            // session-declared fns fall through to scratch eval below.
            if let Some((name, _)) = parse_call(line) {
                if !line.contains(" let ") && s.vm.host_fns().iter().any(|(n, _, _)| n == &name) {
                    return show_call(&mut s.vm, line);
                }
            }
            // decl or eval through the scratch VM
            if looks_like_decl(line) {
                match s.add_decl(line) {
                    Ok(label) => println!("✓ kept `{label}`"),
                    Err(e) => return Err(format!("decl rejected: {e}")),
                }
                return Ok(());
            }
            // bare expression — wrap in a synthetic host fn, eval in scratch.
            // `-> str` declared so `return str(expr)` handles any type;
            // `let a = 1; a + 1` → `{ let a = 1; return str(a + 1); }`.
            let n = s.decls.len();
            let (body, ret) = match split_last_stmt(line) {
                Some((head, tail)) => (format!("{head} return str({tail});"), " -> str"),
                None if line.trim_end().ends_with(';') || line.trim_end().ends_with('}') => {
                    (line.to_string(), "")
                }
                None => (format!("return str({line});"), " -> str"),
            };
            let eval_src = format!(
                "{}\n\nhost fn __repl_eval_{n}(){ret} {{\n    {body}\n}}\n",
                s.src
            );
            // if the return-wrap fails to check, try the raw line as
            // statements before giving up.
            let eval_src = if check_src(&s.anchor, &eval_src).is_err() {
                format!(
                    "{}\n\nhost fn __repl_eval_{n}() {{\n    {line}\n}}\n",
                    s.src
                )
            } else {
                eval_src
            };
            match check_src(&s.anchor, &eval_src) {
                Ok(h) => {
                    let host = h.host.ok_or("internal: no host stratum")?;
                    let mut vm = Vm::new(&host, Vec::new(), &sha256_hex(eval_src.as_bytes()))?;
                    mind::bind_all(&mut vm);
                    let v = vm.call_fn(&format!("__repl_eval_{n}"), Vec::new())?;
                    println!("{}", vm.stringify(&v));
                }
                Err(e) => {
                    if line.contains('(') && parse_call(line).is_some() {
                        return show_call(&mut s.vm, line);
                    }
                    // errors point into the typed line, not the
                    // accumulated session source
                    return Err(rebase(&e, appended_line(&s.src) + 1));
                }
            }
            Ok(())
        })();
        match r {
            Ok(()) => {}
            Err(e) if e.is_empty() => break, // :q
            Err(e) => println!("✗ {e}"),
        }
    }
    println!("✓ session closed");
    Ok(())
}
