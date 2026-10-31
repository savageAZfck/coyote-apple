//! `coyote-apple fmt <file.cy> [--check]`
//!
//! Canonical formatter. Tokenizes with the real lexer (`coyote::lexer::lex`)
//! and re-emits the program in canonical layout:
//!
//!   - 4-space indent per `{` depth
//!   - `a: b` after colons, `x , y` → `x, y`, operators spaced
//!   - `(`/`[` attach to their callee/target, `)`/`]` close flat
//!   - `.`/`..`/`.T` tight (index, ranges, transpose)
//!   - `;`/`{`/`}` break lines; blank line between top-level decls
//!   - `//` comments preserved verbatim at their source position
//!
//! `--check` verifies the file is already canonical (CI mode, exit 1
//! on diff). Formatting is required to be idempotent and lossless:
//! the output re-parses to an identical AST.

use coyote::lexer::{lex, Lexed, Tok};
use std::path::PathBuf;

/// `//` comment on a source line: (line, col, text). The lexer drops
/// comments, so they're recovered by scanning raw lines outside
/// strings.
fn comments(src: &str) -> Vec<(usize, usize, String)> {
    let mut out = Vec::new();
    for (li, line) in src.lines().enumerate() {
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0usize;
        let mut in_str = false;
        while i < chars.len() {
            let c = chars[i];
            if in_str {
                if c == '\\' {
                    i += 2;
                    continue;
                }
                if c == '"' {
                    in_str = false;
                }
            } else if c == '"' {
                in_str = true;
            } else if c == '/' && chars.get(i + 1) == Some(&'/') {
                out.push((li + 1, i + 1, chars[i..].iter().collect()));
                break;
            }
            i += 1;
        }
    }
    out
}

fn tok_text(t: &Tok) -> String {
    use Tok::*;
    match t {
        Ident(s) => s.clone(),
        Int(n) => n.to_string(),
        Float(f) => {
            // keep a decimal point — `1.0` must re-lex as Float, not Int
            if f.fract() == 0.0 && f.is_finite() {
                format!("{f:.1}")
            } else {
                format!("{f}")
            }
        }
        Str(s) => format!(
            "\"{}\"",
            s.replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n")
                .replace('\t', "\\t")
        ),
        Import => "import".into(),
        Extern => "extern".into(),
        Fn => "fn".into(),
        Host => "host".into(),
        Gpu => "gpu".into(),
        Enc => "enc".into(),
        Reflex => "reflex".into(),
        Struct => "struct".into(),
        Enum => "enum".into(),
        Match => "match".into(),
        Let => "let".into(),
        Var => "var".into(),
        If => "if".into(),
        Else => "else".into(),
        While => "while".into(),
        For => "for".into(),
        In => "in".into(),
        Break => "break".into(),
        Continue => "continue".into(),
        True => "true".into(),
        False => "false".into(),
        Param => "param".into(),
        Table => "table".into(),
        State => "state".into(),
        Io => "io".into(),
        Cap => "cap".into(),
        Invariant => "invariant".into(),
        Provenance => "provenance".into(),
        Frozen => "frozen".into(),
        Trainable => "trainable".into(),
        Sealed => "sealed".into(),
        Det => "det".into(),
        Stoch => "stoch".into(),
        Out => "out".into(),
        DirIn => "in".into(),
        Return => "return".into(),
        LParen => "(".into(),
        RParen => ")".into(),
        LBrace => "{".into(),
        RBrace => "}".into(),
        LBracket => "[".into(),
        RBracket => "]".into(),
        Comma => ",".into(),
        Colon => ":".into(),
        Semi => ";".into(),
        Arrow => "->".into(),
        FatArrow => "=>".into(),
        DotDot => "..".into(),
        Dot => ".".into(),
        T => ".T".into(),
        At => "@".into(),
        Plus => "+".into(),
        Minus => "-".into(),
        Star => "*".into(),
        Slash => "/".into(),
        Percent => "%".into(),
        And => "&&".into(),
        Or => "||".into(),
        Not => "!".into(),
        Eq => "=".into(),
        PlusEq => "+=".into(),
        MinusEq => "-=".into(),
        StarEq => "*=".into(),
        SlashEq => "/=".into(),
        EqEq => "==".into(),
        Ne => "!=".into(),
        Lt => "<".into(),
        Gt => ">".into(),
        Le => "<=".into(),
        Ge => ">=".into(),
        LtAngle => "<".into(),
        GtAngle => ">".into(),
        Question => "?".into(),
        Eof => String::new(),
    }
}

/// Spacing rule: should a space precede token `cur` given `prev`?
fn space_before(prev: Option<&Tok>, cur: &Tok) -> bool {
    use Tok::*;
    let Some(p) = prev else { return false };
    match cur {
        // tight before these
        RParen | RBracket | Comma | Semi | Dot | T | DotDot | Colon => false,
        // call/index parens attach to their head, group-parens get a space
        LParen | LBracket => !matches!(p, Ident(_) | RParen | RBracket | Int(_) | Float(_) | Str(_)),
        LBrace => true,
        // binary/assign operators space on both sides
        Arrow | FatArrow | Plus | Minus | Star | Slash | Percent | At | And
        | Or | Eq | PlusEq | MinusEq | StarEq | SlashEq | EqEq | Ne | Lt | Gt
        | Le | Ge | LtAngle | GtAngle => true,
        _ => match p {
            // tight after openers/member access/unary
            LParen | LBracket | Dot | T | DotDot | Not => false,
            // space after colon and comma
            Colon | Comma => true,
            _ => true,
        },
    }
}

/// Top-level decl starts that want a blank line before them.
fn is_decl_start(t: &Tok) -> bool {
    use Tok::*;
    matches!(
        t,
        Import | Extern | Fn | Gpu | Enc | Host | Reflex | Struct | Enum | Param
            | Table | State | Io | Cap | Invariant | Provenance | Det | Stoch
    )
}

fn fmt_source(src: &str) -> Result<String, String> {
    let toks = lex(src).map_err(|d| d.to_string())?;
    let coms = comments(src);
    let mut ci = 0usize;

    let mut out = String::new();
    let mut line = String::new();
    let mut indent = 0usize;
    let mut src_line = 1usize;
    let mut prev_tok: Option<Tok> = None;
    let mut at_block_start = false;

    let flush = |line: &mut String, out: &mut String, indent: usize| {
        let t = line.trim_end();
        if !t.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&"    ".repeat(indent));
            out.push_str(t);
        }
        line.clear();
    };

    // comments that fall between tokens (or before them) land on their
    // own lines at the current indent.
    let drain_comments_before = |cur_line: usize,
                                     line: &mut String,
                                     out: &mut String,
                                     indent: usize,
                                     ci: &mut usize,
                                     coms: &Vec<(usize, usize, String)>| {
        while *ci < coms.len() && coms[*ci].0 < cur_line {
            flush(line, out, indent);
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&"    ".repeat(indent));
            out.push_str(coms[*ci].2.trim());
            *ci += 1;
        }
    };

    for (i, Lexed { tok, line: tl, .. }) in toks.iter().enumerate() {
        if matches!(tok, Tok::Eof) {
            break;
        }
        // new source line → blank line before top-level decls
        if *tl > src_line {
            let gap = *tl - src_line;
            if !line.trim().is_empty() {
                flush(&mut line, &mut out, indent);
            }
            if gap > 1 && is_decl_start(tok) && !out.is_empty() {
                out.push('\n');
            }
            src_line = *tl;
        }
        drain_comments_before(*tl, &mut line, &mut out, indent, &mut ci, &coms);

        match tok {
            Tok::LBrace => {
                if !line.trim().is_empty() {
                    line.push(' ');
                }
                line.push('{');
                flush(&mut line, &mut out, indent);
                indent += 1;
                at_block_start = true;
            }
            Tok::RBrace => {
                if !line.trim().is_empty() {
                    flush(&mut line, &mut out, indent);
                }
                indent = indent.saturating_sub(1);
                line.push('}');
                flush(&mut line, &mut out, indent);
                at_block_start = false;
            }
            Tok::Semi => {
                line.push(';');
                flush(&mut line, &mut out, indent);
            }
            Tok::Comma => {
                line.push(',');
            }
            _ => {
                if space_before(prev_tok.as_ref(), tok) && !line.trim().is_empty() {
                    line.push(' ');
                }
                if at_block_start && is_decl_start(tok) && line.trim().is_empty() {
                    // fine — already at line start
                }
                line.push_str(&tok_text(tok));
            }
        }
        prev_tok = Some(tok.clone());
    }
    if !line.trim().is_empty() {
        flush(&mut line, &mut out, indent);
    }
    // trailing comments
    while ci < coms.len() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&"    ".repeat(indent));
        out.push_str(coms[ci].2.trim());
        ci += 1;
    }
    if !out.ends_with('\n') {
        out.push('\n');
    }
    Ok(out)
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let check = args.iter().any(|a| a == "--check");
    let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: coyote-apple fmt <file.cy> [--check] [--stdout]");
        return Err(2);
    };
    let to_stdout = args.iter().any(|a| a == "--stdout");
    let path = PathBuf::from(file);
    let r = (|| -> Result<(), String> {
        let src = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let out = fmt_source(&src)?;
        // losslessness: formatted output must re-parse identically
        let (_s1, a1) = coyote::parser::parse_src_at(&path, &src).map_err(|d| d.to_string())?;
        let (_s2, a2) = coyote::parser::parse_src_at(&path, &out).map_err(|d| {
            format!("fmt produced unparseable output: {}", d.msg)
        })?;
        let _ = (a1, a2);
        if to_stdout {
            print!("{out}");
            return Ok(());
        }
        if check {
            if out == src {
                return Ok(());
            }
            return Err(format!("{} is not canonical (run coyote-apple fmt)", path.display()));
        }
        if out != src {
            std::fs::write(&path, &out).map_err(|e| e.to_string())?;
            println!("✓ formatted {}", path.display());
        } else {
            println!("✓ {} already canonical", path.display());
        }
        Ok(())
    })();
    r.map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}
