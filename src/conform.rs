//! `coyote-apple conform <file.cy> <protocol.cyproto>`
//!
//! Session conformance — the multiparty-session-types discipline
//! applied to the runtime's own committed trace. A `.cyproto` file
//! declares the expected witness-event protocol; `conform` executes
//! the program on silicon and replays its committed witness chain
//! against the protocol automaton. A program that drifts off-protocol
//! fails the session.
//!
//! Protocol grammar (one term per line):
//!   tag        — exactly one witness event with this tag
//!   tag?       — zero or one
//!   tag*       — zero or more
//!   tag+       — one or more
//!   a|b        — either tag a or tag b (repeatable: a|b*)
//!   # comment
//!
//! The automaton is a sequential matcher over the emitted tag
//! sequence — deterministic, no backtracking needed since tags are
//! observed, not chosen. The verdict lands in coyote_out/conform.json.

use serde_json::json;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
enum Term {
    One(Vec<String>),
    Opt(Vec<String>),
    Star(Vec<String>),
    Plus(Vec<String>),
}

fn parse_protocol(text: &str) -> Result<Vec<Term>, String> {
    let mut terms = Vec::new();
    for (ln, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (body, rep) = match line.chars().last() {
            Some('?') => (&line[..line.len() - 1], 'q'),
            Some('*') => (&line[..line.len() - 1], 's'),
            Some('+') => (&line[..line.len() - 1], 'p'),
            _ => (line, 'o'),
        };
        let alts: Vec<String> = body.split('|').map(|s| s.trim().to_string()).collect();
        if alts.iter().any(|a| a.is_empty()) {
            return Err(format!("line {}: empty alternative", ln + 1));
        }
        terms.push(match rep {
            'q' => Term::Opt(alts),
            's' => Term::Star(alts),
            'p' => Term::Plus(alts),
            _ => Term::One(alts),
        });
    }
    Ok(terms)
}

/// Greedy sequential match — the emitted trace is fixed, so greedy is
/// complete for this grammar (each term consumes a maximal run).
fn match_protocol(terms: &[Term], tags: &[String]) -> Result<usize, String> {
    let mut ti = 0usize;
    for term in terms {
        let alts = match term {
            Term::One(a) | Term::Opt(a) | Term::Star(a) | Term::Plus(a) => a,
        };
        let mut consumed = 0usize;
        while ti < tags.len() && alts.contains(&tags[ti]) {
            ti += 1;
            consumed += 1;
        }
        match term {
            Term::One(_) if consumed != 1 => {
                return Err(format!(
                    "expected {:?}, saw {}",
                    alts,
                    tags.get(ti).map(String::as_str).unwrap_or("<end>")
                ))
            }
            Term::Plus(_) if consumed == 0 => {
                return Err(format!("expected one or more of {alts:?}"))
            }
            _ => {}
        }
    }
    Ok(ti)
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(file) = args.first() else {
        eprintln!("usage: coyote-apple conform <file.cy> <protocol.cyproto>");
        return Err(2);
    };
    let Some(proto_path) = args.get(1) else {
        eprintln!("usage: coyote-apple conform <file.cy> <protocol.cyproto>");
        return Err(2);
    };
    let r = (|| -> Result<(), String> {
        let cy = Path::new(file);
        let terms =
            parse_protocol(&std::fs::read_to_string(proto_path).map_err(|e| e.to_string())?)?;
        // execute on silicon — the committed witness chain is the session trace
        let dir = std::env::temp_dir().join(format!("cyconform-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut vm = coyote::load_vm(cy, &dir, false, false, true)?;
        crate::mind::bind_all(&mut vm);
        crate::peer::bind_all(&mut vm);
        let ret = vm
            .run_main()
            .map(|r| vm.stringify(&r))
            .unwrap_or_else(|e| e);
        let tags: Vec<String> = vm.witness_log().iter().map(|e| e.tag.clone()).collect();
        let verdict = match match_protocol(&terms, &tags) {
            Ok(consumed) if consumed == tags.len() => Ok("conformant"),
            Ok(consumed) => Err(format!(
                "trailing events past protocol: {} extra (tags {:?})",
                tags.len() - consumed,
                &tags[consumed..]
            )),
            Err(e) => Err(e),
        };
        let doc = json!({
            "protocol": "session-conformance-v1",
            "program": cy.display().to_string(),
            "spec": proto_path,
            "return": ret,
            "witness_events": tags.len(),
            "verdict": match &verdict { Ok(s) => json!(s), Err(e) => json!(format!("nonconformant: {e}")) },
        });
        let out_dir = std::env::var("CYO_CONFORM_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("coyote_out"));
        let _ = std::fs::create_dir_all(&out_dir);
        let _ = std::fs::write(
            out_dir.join("conform.json"),
            serde_json::to_string_pretty(&doc).unwrap(),
        );
        match verdict {
            Ok(s) => {
                println!("✓ {s} — {} witness events matched the protocol", tags.len());
                Ok(())
            }
            Err(e) => Err(format!("nonconformant — {e}")),
        }
    })();
    r.map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}
