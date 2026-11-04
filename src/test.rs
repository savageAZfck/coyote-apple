//! `coyote-apple test <file.cy> [--cpu] [--rebuild]`
//!
//! The language's test runner. Any zero-arg host fn named `test_*` is
//! a test: `return 0;` (or unit/true) passes, a nonzero return or a
//! raised error fails. Tests run inside a resident VM on the selected
//! backend, so a test file can exercise host logic AND `ane(...) ->
//! ...` program calls through the same surface the organism uses.
//!
//!   host fn test_add() -> i64 {
//!       if fib(5) != 5 { return 1; }
//!       return 0;
//!   }

use std::path::Path;

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: coyote-apple test <file.cy> [--cpu] [--rebuild]");
        return Err(2);
    };
    let use_cpu = args.iter().any(|a| a == "--cpu");
    let rebuild = args.iter().any(|a| a == "--rebuild");
    let path = Path::new(file);
    let dir = std::env::temp_dir().join(format!("cytest-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let r = (|| -> Result<(), String> {
        // discovery needs the checked AST — zero-arg test_* fns only
        let (_s, ast) = coyote::parser::parse_file(path).map_err(|d| d.to_string())?;
        let checked = coyote::typeck::check_all(&ast).map_err(|d| d.to_string())?;
        let tests: Vec<String> = checked
            .host
            .as_ref()
            .map(|h| {
                h.fns
                    .iter()
                    .filter(|f| f.name.starts_with("test_") && f.args.is_empty())
                    .map(|f| f.name.clone())
                    .collect()
            })
            .unwrap_or_default();
        if tests.is_empty() {
            return Err(format!(
                "{file}: no zero-arg `host fn test_*` found — a test fn returns 0 on pass"
            ));
        }
        let mut vm = coyote::load_vm(path, &dir, use_cpu, rebuild, true)?;
        crate::mind::bind_all(&mut vm);
        crate::peer::bind_all(&mut vm);
        println!(
            "test — {file}  ({} backend)",
            if use_cpu { "cpu" } else { "ane" }
        );
        let mut failed = 0usize;
        for t in &tests {
            match vm.call_fn(t, vec![]) {
                Ok(v) => {
                    let s = vm.stringify(&v);
                    let ok = s == "()" || s == "0" || s == "true";
                    if ok {
                        println!("  ✓ {t}");
                    } else {
                        println!("  ✗ {t} — returned {s}");
                        failed += 1;
                    }
                }
                Err(e) => {
                    println!("  ✗ {t} — {e}");
                    failed += 1;
                }
            }
        }
        let n = tests.len();
        if failed == 0 {
            println!("✓ {n} test{} passed", if n == 1 { "" } else { "s" });
            Ok(())
        } else {
            Err(format!("{failed} of {n} tests failed"))
        }
    })();
    let _ = std::fs::remove_dir_all(&dir);
    r.map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}
