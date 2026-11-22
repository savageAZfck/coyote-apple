//! `coyote-apple revcheck <file.cy> [--tol X] [--json]`
//!
//! The reversibility gate: for every `rev fn` — derived inverse or
//! `inv`-declared — synthesize deterministic inputs, run the forward
//! program through the reference evaluator, feed the output plus ctx
//! args into the inverse, and demand the original state comes back.
//! `inv∘f = id`, executed and measured — the first rung of the
//! rev-verified ladder (oracle → certify → referee).
//!
//! Inputs are small integers: exact in f32, safe for both xor chains
//! (bitwise bijection) and +=/-= chains (no rounding below 2²⁴).

use std::collections::HashMap;
use std::path::Path;

use coyote::ast::{Decl, IoDir, Target};
use coyote::eval::Evaluator;
use coyote::hostck::{HProgram, RTy};
use coyote::typeck::{CTy, TProgram};
use coyote::vm::{Val, Vm};

/// deterministic synthetic tensor — small ints, exact in f32, varied
/// per port so ctx args don't alias the state thread.
fn synth(ty: &CTy, salt: usize) -> Vec<f32> {
    let n = ty.numel().max(1) as usize;
    (0..n)
        .map(|i| (((i * 7 + salt * 13 + 3) % 17) as f32) - 8.0)
        .collect()
}

fn in_ports(tp: &TProgram) -> Vec<(&String, &CTy)> {
    tp.ios
        .iter()
        .filter(|(_, d, _)| matches!(d, IoDir::In))
        .map(|(n, _, t)| (n, t))
        .collect()
}

/// every concrete `rev fn` paired with its inverse program name —
/// `inv` clause target or derived `{name}_inv`. Shared with certify.
pub(crate) fn rev_pairs(ast: &coyote::ast::Program) -> Vec<(String, String)> {
    ast.decls
        .iter()
        .filter_map(|d| match d {
            Decl::Fn(f)
                if f.v.target == Target::Rev && f.v.dims.is_empty() =>
            {
                let inv = f
                    .v
                    .inv
                    .as_ref()
                    .map(|i| i.v.clone())
                    .unwrap_or_else(|| format!("{}_inv", f.v.name));
                Some((f.v.name.clone(), inv))
            }
            _ => None,
        })
        .collect()
}

/// run one rev pair: forward on synth inputs, inverse on
/// (output, ctx...), compare recovered state vs original state input.
/// Returns (max_abs_diff, recovered_len, expected_len).
pub(crate) fn roundtrip(
    fwd: &TProgram,
    inv: &TProgram,
    salt: usize,
) -> Result<(f64, usize, usize), String> {
    let fins = in_ports(fwd);
    let iins = in_ports(inv);
    if iins.len() != fins.len() {
        return Err(format!(
            "inverse arity {} != forward {} — `inv` clause signature mismatch",
            iins.len(),
            fins.len()
        ));
    }
    // forward inputs, positional
    let mut inputs: HashMap<String, Vec<f32>> = HashMap::new();
    let mut fwd_vals: Vec<Vec<f32>> = Vec::with_capacity(fins.len());
    for (i, (port, ty)) in fins.iter().enumerate() {
        let v = synth(ty, salt + i);
        inputs.insert((*port).clone(), v.clone());
        fwd_vals.push(v);
    }
    let mut fev = Evaluator::new(fwd);
    let out = fev
        .run(&inputs)
        .map_err(|d| format!("forward eval: {d}"))?;

    // inverse inputs: port[0] ← forward output; ports[1..] ← forward
    // ctx values positionally (derived inverses keep arg names; a
    // hand-written `inv` may rename them — position is the contract).
    let mut iinputs: HashMap<String, Vec<f32>> = HashMap::new();
    for (i, (port, _)) in iins.iter().enumerate() {
        let v = if i == 0 {
            out.clone()
        } else {
            fwd_vals[i].clone()
        };
        iinputs.insert((*port).clone(), v);
    }
    let mut iev = Evaluator::new(inv);
    let back = iev
        .run(&iinputs)
        .map_err(|d| format!("inverse eval: {d}"))?;

    // compare recovered state (inverse output) vs forward state input
    let expect = &fwd_vals[0];
    let n = expect.len().min(back.len());
    let mut maxd = 0f64;
    for i in 0..n {
        maxd = maxd.max((back[i] - expect[i]).abs() as f64);
    }
    if back.len() != expect.len() {
        maxd = f64::INFINITY;
    }
    Ok((maxd, back.len(), expect.len()))
}

/// every `rev host fn` paired with its derived `{name}_inv`.
pub(crate) fn host_rev_pairs(ast: &coyote::ast::Program) -> Vec<(String, String)> {
    ast.decls
        .iter()
        .filter_map(|d| match d {
            Decl::HostFn(f) if f.v.rev => {
                Some((f.v.name.clone(), format!("{}_inv", f.v.name)))
            }
            _ => None,
        })
        .collect()
}

/// deterministic host value for revcheck — small ints, varied per
/// position so ctx args don't alias the state thread. Structs synth
/// field-wise so multi-var state threads get real coverage.
fn synth_val(
    ty: &RTy,
    salt: usize,
    structs: &[(String, Vec<(String, RTy)>)],
) -> Result<Val, String> {
    let n = (salt * 37 + 11) as i64;
    Ok(match ty {
        RTy::I64 => Val::I(n % 1000 - 500),
        RTy::F64 => Val::F((n % 200 - 100) as f64 / 4.0),
        RTy::Bool => Val::B(salt % 2 == 0),
        RTy::Str => Val::S(std::rc::Rc::from(format!("s{salt}"))),
        RTy::Struct(si) => {
            let Some((_, fields)) = structs.get(*si as usize) else {
                return Err(format!("revcheck: unknown struct index {si}"));
            };
            let vals: Vec<Val> = fields
                .iter()
                .enumerate()
                .map(|(i, (_, fty))| synth_val(fty, salt + i + 1, structs))
                .collect::<Result<_, _>>()?;
            Val::St(
                *si,
                std::rc::Rc::new(std::cell::RefCell::new(vals)),
            )
        }
        other => {
            return Err(format!(
                "revcheck: unsupported host type for synthesis ({other:?})"
            ))
        }
    })
}

fn val_diff(a: &Val, b: &Val) -> f64 {
    match (a, b) {
        (Val::I(x), Val::I(y)) => (*x - *y).unsigned_abs() as f64,
        (Val::F(x), Val::F(y)) => (x - y).abs(),
        (Val::B(x), Val::B(y)) => {
            if x == y {
                0.0
            } else {
                f64::INFINITY
            }
        }
        (Val::S(x), Val::S(y)) => {
            if x == y {
                0.0
            } else {
                f64::INFINITY
            }
        }
        (Val::St(_, xa), Val::St(_, xb)) => {
            let (xa, xb) = (xa.borrow(), xb.borrow());
            if xa.len() != xb.len() {
                return f64::INFINITY;
            }
            xa.iter()
                .zip(xb.iter())
                .map(|(u, v)| val_diff(u, v))
                .fold(0.0, f64::max)
        }
        _ => f64::INFINITY,
    }
}

/// run one `rev host fn` pair through the VM: forward on synthetic
/// args, inverse on (output, ctx...), demand the state comes back.
/// `fns` is the compiled HProgram's fn table (name → HFnT).
pub(crate) fn host_roundtrip(
    vm: &mut Vm,
    fns: &[(String, coyote::hostck::HFnT)],
    structs: &[(String, Vec<(String, RTy)>)],
    fwd_name: &str,
    inv_name: &str,
    salt: usize,
) -> Result<(f64, usize, usize), String> {
    let find = |n: &str| fns.iter().find(|(name, _)| name == n).map(|(_, f)| f);
    let (Some(fwd), Some(inv)) = (find(fwd_name), find(inv_name)) else {
        return Err(format!("host rev pair incomplete: `{fwd_name}` ↔ `{inv_name}`"));
    };
    if fwd.args.len() != inv.args.len() {
        return Err(format!(
            "host inverse arity {} != forward {}",
            inv.args.len(),
            fwd.args.len()
        ));
    }
    // forward args, positional — arg[0] is the state thread
    let mut fwd_args = Vec::with_capacity(fwd.args.len());
    for (i, (_, ty)) in fwd.args.iter().enumerate() {
        fwd_args.push(synth_val(ty, salt + i, structs)?);
    }
    let out = vm
        .call_fn(fwd_name, fwd_args.clone())
        .map_err(|d| format!("forward host call: {d}"))?;

    // inverse args: [0] ← forward output; [1..] ← forward ctx values
    let mut inv_args = Vec::with_capacity(inv.args.len());
    for (i, (_, ty)) in inv.args.iter().enumerate() {
        inv_args.push(if i == 0 {
            out.clone()
        } else {
            fwd_args[i].clone()
        });
        let _ = ty;
    }
    let back = vm
        .call_fn(inv_name, inv_args)
        .map_err(|d| format!("inverse host call: {d}"))?;

    let d = val_diff(&back, &fwd_args[0]);
    Ok((d, 1, 1))
}

/// verify every `rev host fn` pair in `host` — builds a throwaway
/// VM (no tensor programs; host rev fns are call-free by revck).
pub(crate) fn host_pairs_check(
    host: &HProgram,
    pairs: &[(String, String)],
) -> Result<Vec<(String, f64, bool)>, String> {
    if pairs.is_empty() {
        return Ok(Vec::new());
    }
    let fns: Vec<(String, coyote::hostck::HFnT)> = host
        .fns
        .iter()
        .map(|f| (f.name.clone(), f.clone()))
        .collect();
    let mut vm = Vm::new(host, Vec::new(), "revcheck").map_err(|e| e)?;
    let mut rows = Vec::new();
    for (i, (f, v)) in pairs.iter().enumerate() {
        match host_roundtrip(&mut vm, &fns, &host.structs, f, v, i * 31 + 7) {
            Ok((d, _, _)) => rows.push((format!("{f} ↔ {v}"), d, d == 0.0)),
            Err(e) => {
                rows.push((format!("{f} ↔ {v}"), f64::INFINITY, false));
                eprintln!("  ✗ {f} ↔ {v}: {e}");
            }
        }
    }
    Ok(rows)
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: coyote-apple revcheck <file.cy> [--tol X] [--json]");
        return Err(2);
    };
    let json = args.iter().any(|a| a == "--json");
    let tol: f64 = args
        .windows(2)
        .find(|w| w[0] == "--tol")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(0.0);

    let r = (|| -> Result<(), String> {
        let (_src, ast) = coyote::parser::parse_file(Path::new(file))
            .map_err(|d| d.to_string())?;
        let ck = coyote::typeck::check_all(&ast).map_err(|d| d.to_string())?;
        let progs: HashMap<&str, &TProgram> =
            ck.ane.iter().map(|(n, p)| (n.as_str(), p)).collect();

        let pairs = rev_pairs(&ast);
        let hpairs = host_rev_pairs(&ast);
        if pairs.is_empty() && hpairs.is_empty() {
            if json {
                println!("{{\"file\":{file:?},\"mode\":\"revcheck\",\"pairs\":0,\"verdict\":\"pass\"}}");
            } else {
                println!("revcheck — {file}: no `rev fn`s found");
            }
            return Ok(());
        }

        let mut rows: Vec<(String, f64, bool)> = Vec::new();
        let mut all_pass = true;
        for (i, (fwd_name, inv_name)) in pairs.iter().enumerate() {
            let (Some(fwd), Some(inv)) = (progs.get(fwd_name.as_str()), progs.get(inv_name.as_str()))
            else {
                return Err(format!(
                    "rev pair incomplete: `{fwd_name}` ↔ `{inv_name}` — inverse program missing"
                ));
            };
            match roundtrip(fwd, inv, i * 31 + 7) {
                Ok((maxd, bl, el)) => {
                    let pass = maxd <= tol;
                    all_pass &= pass;
                    rows.push((format!("{fwd_name} ↔ {inv_name}"), maxd, pass));
                    if bl != el {
                        all_pass = false;
                    }
                }
                Err(e) => {
                    all_pass = false;
                    rows.push((format!("{fwd_name} ↔ {inv_name}"), f64::INFINITY, false));
                    if json {
                        println!("{{\"pair\":{fwd_name:?},\"error\":{e:?}}}");
                    } else {
                        eprintln!("  ✗ {fwd_name} ↔ {inv_name}: {e}");
                    }
                }
            }
        }

        // host stratum: `rev host fn` pairs verify through the VM —
        // forward call, inverse call on (output, ctx...), exact match.
        let hrows = match &ck.host {
            Some(host) => host_pairs_check(host, &hpairs)
                .map_err(|e| format!("host revcheck: {e}"))?,
            None if !hpairs.is_empty() => {
                return Err("`rev host fn` present but no host program compiled".into());
            }
            None => Vec::new(),
        };
        for (n, d, p) in &hrows {
            all_pass &= p;
            rows.push((n.clone(), if *p { *d } else { f64::INFINITY }, *p));
        }

        if json {
            let inner: Vec<String> = rows
                .iter()
                .map(|(n, d, p)| {
                    format!(
                        "{{\"pair\":{n:?},\"max_diff\":{d:.6},\"pass\":{p}}}"
                    )
                })
                .collect();
            println!(
                "{{\"file\":{file:?},\"mode\":\"revcheck\",\"pairs\":[{}],\"tol\":{tol},\"verdict\":\"{}\"}}",
                inner.join(","),
                if all_pass { "pass" } else { "fail" }
            );
        } else {
            println!("revcheck — {file} ({} pairs, tol {tol})", rows.len());
            for (n, d, p) in &rows {
                println!(
                    "  {} {:<36} max|Δ| {:.6} {}",
                    if *p { "✓" } else { "✗" },
                    n,
                    if d.is_infinite() { f64::NAN } else { *d },
                    if *p { "" } else { "FAIL" }
                );
            }
            println!(
                "{}",
                if all_pass {
                    "✓ pass — inv∘f = id on every rev pair"
                } else {
                    "✗ fail — inverse diverges from identity"
                }
            );
        }
        if !all_pass {
            return Err("revcheck failed".into());
        }
        Ok(())
    })();
    r.map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}
