//! coyote-apple — the internal profile of the coyote toolchain.
//!
//! Every public verb passes straight through to `coyote::cli` — the
//! pipeline is identical: same parser, same typechecker, same emitters,
//! same resident VM, same witness chain. The apple verbs add what an
//! org needs on top:
//!
//!   coyote-apple spec     <file.cy>            capability + structure matrix
//!   coyote-apple audit    <build_dir>          hash-chained build provenance
//!   coyote-apple package  <file.cy> [-o pkg]   single-file .cypack distributable
//!   coyote-apple verify   <pkg.cypack>         integrity check a package
//!   coyote-apple unpack   <pkg.cypack> -o dir  extract a package
//!   coyote-apple bench    <file.cy> [--runs N] e2e + organism-exec timing
//!   coyote-apple oracle   <file.cy> [--steps N] [--json]
//!                                            semantic parity gate (ANE vs CPU)
//!   coyote-apple genmask  <name>:<rank>:<dim>  rank-selector mask → safetensors
//!   coyote-apple diff     <dirA> <dirB>        build-vs-build artifact delta
//!   coyote-apple test     <file.cy>            run host fn test_*() cases
//!   coyote-apple new      <name>               scaffold a coyote project
//!   coyote-apple weights  <inspect|slice>      safetensors checkpoint tooling
//!   coyote-apple doc      <file.cy>            generated API markdown
//!   coyote-apple gold     <record|check>       golden-output regression
//!   coyote-apple watch    <file.cy> [--run]    rebuild on save
//!
//! The distribution also ships `lib/std/` — the stdlib. `import "std/x"`
//! resolves through CYO_PATH, which this binary extends with its own
//! lib/ automatically (user paths still win).

mod audit;
mod backend_metal;
mod bench;
mod caps;
mod certify;
mod conform;
mod diff;
mod doc;
mod emit;
mod enc_ckks;
mod fleet;
mod fmt;
mod genmask;
mod gold;
mod gpu_emit;
mod gpu_rt;
mod inspect;
mod lsp;
mod mesh;
mod mind;
mod new;
mod oracle;
mod package;
mod pcc;
mod peer;
mod prof;
mod ranknode;
mod referee;
mod registry;
mod repl;
mod replay;
mod run;
mod serve;
mod shardize;
mod spec;
mod stfile;
mod substrate;
mod test;
mod watch;
mod weights;
mod wire;

fn usage() {
    eprintln!("coyote-apple — internal profile of the coyote toolchain");
    eprintln!("  spec     <file.cy>");
    eprintln!("  audit    <build_dir> [--verify]");
    eprintln!("  inspect  <path|url> [--dir d] [--world w.json]   evidence-grade dissection");
    eprintln!("  package  <file.cy> [-o pkg.cypack]");
    eprintln!("  verify   <pkg.cypack>");
    eprintln!("  unpack   <pkg.cypack> -o <dir>");
    eprintln!("  bench    <file.cy> [-o dir] [--steps N] [--runs N] [--cpu] [--json]");
    eprintln!("  prof     <file.cy> [--runs N] [--json]   per-dispatch + device timing");
    eprintln!("  oracle   <file.cy> [--steps N] [--json]   semantic parity gate");
    eprintln!("  genmask  <name>:<rank>:<dim> [--rows] -o mask.safetensors");
    eprintln!("  diff     <dirA> <dirB>   per-file delta between two build dirs");
    eprintln!("  test     <file.cy> [--cpu]       run host fn test_*() cases");
    eprintln!("  new      <name> [--program]      scaffold a coyote project");
    eprintln!("  weights  inspect|slice ...       safetensors checkpoint tooling");
    eprintln!("  doc      <file.cy> [-o doc.md]   generated API markdown");
    eprintln!("  gold     record|check <file.cy>  golden-output regression");
    eprintln!("  watch    <file.cy> [--run]       rebuild on save");
    eprintln!("  referee  <file.cy> [--lie seq:N] [--json]   refereed verification");
    eprintln!("  certify  <file.cy> [-o cert.json]         proof-carrying certificate");
    eprintln!("  conform  <file.cy> <spec.cyproto>         session conformance");
    eprintln!("  emit     <file.cy> --swift       generate the Swift bridge class");
    eprintln!("  + all coyote verbs (check/build/run/repl) pass through unchanged");
}

fn main() {
    // the gpu stratum's engine lives here — the public crate parses
    // `gpu fn` but only this distribution can lower and run it.
    coyote::backend::register_backend(
        coyote::ast::Target::Gpu,
        std::sync::Arc::new(backend_metal::MetalBackend::new()),
    );
    // the `enc` stratum's engine — encrypted circuits, private to
    // this distribution the same way Metal is.
    coyote::backend::register_backend(
        coyote::ast::Target::Enc,
        std::sync::Arc::new(enc_ckks::EncBackend),
    );
    // execution sites — `mesh fn` dispatches to treaty'd peers, `pcc fn`
    // to attested nodes. Both ride the Bad Apple mesh wire (wire.rs).
    mesh::register();
    pcc::register();
    // every Vm-building verb binds the mind externs on the Vm it
    // constructs — run.rs / bench.rs / test.rs / oracle.rs / prof.rs
    // / repl.rs each call mind::bind_all before executing.
    // the distribution's stdlib rides on CYO_PATH — user's own entries
    // come first, our lib/ last, so project modules always win.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(repo) = exe.ancestors().nth(3) {
            let lib = repo.join("lib");
            if lib.is_dir() {
                let cur = std::env::var("CYO_PATH").unwrap_or_default();
                let sep = if cur.is_empty() { "" } else { ":" };
                std::env::set_var("CYO_PATH", format!("{cur}{sep}{}", lib.display()));
            }
        }
    }
    // the package registry rides on CYO_PATH too — `import "name"`
    // resolves anything `pkg install` dropped in ~/.coyote/pkg/.
    {
        let reg = registry::registry_dir();
        if reg.is_dir() {
            let cur = std::env::var("CYO_PATH").unwrap_or_default();
            let sep = if cur.is_empty() { "" } else { ":" };
            std::env::set_var("CYO_PATH", format!("{cur}{sep}{}", reg.display()));
        }
    }
    let args: Vec<String> = std::env::args().collect();
    let Some(cmd) = args.get(1).map(|s| s.as_str()) else {
        usage();
        std::process::exit(2);
    };
    let r = match cmd {
        "spec" => spec::run(&args[2..]),
        "audit" => audit::run(&args[2..]),
        "inspect" => inspect::run(&args[2..]),
        "package" | "pack" => package::run(&args[2..]),
        "verify" => package::verify(&args[2..]),
        "unpack" => package::unpack(&args[2..]),
        "bench" => bench::run(&args[2..]),
        "prof" => prof::run(&args[2..]),
        "oracle" => oracle::run(&args[2..]),
        "genmask" => genmask::run(&args[2..]),
        "diff" => diff::run(&args[2..]),
        "test" => test::run(&args[2..]),
        "new" => new::run(&args[2..]),
        "weights" => weights::run(&args[2..]),
        "doc" => doc::run(&args[2..]),
        "gold" => gold::run(&args[2..]),
        // `run` is intercepted: the public pipeline builds the VM
        // internally, so organisms that declare `mind_*` externs need
        // this distribution to build (and bind) it instead.
        "replay" => replay::run(&args[2..]).map_err(|e| {
            eprintln!("replay: {e}");
            1
        }),
        "run" => run::run(&args[2..]),
        // `repl` is intercepted for the same reason — the public shell
        // builds its VMs internally; this distribution's copy binds
        // the mind externs on live, bare, and scratch VMs alike.
        "repl" => repl::run(&args[2..]),
        "watch" => watch::run(&args[2..]),
        "emit" => emit::run(&args[2..]),
        "referee" => referee::run(&args[2..]),
        "certify" => certify::run(&args[2..]),
        "conform" => conform::run(&args[2..]),
        "caps" => caps::run(&args[2..]),
        "fmt" => fmt::run(&args[2..]),
        "pkg" => registry::run(&args[2..]),
        "shardize" => shardize::run(&args[2..]),
        "meshnode" => {
            let listen = args
                .iter()
                .position(|a| a == "--listen")
                .and_then(|i| args.get(i + 1))
                .cloned()
                .unwrap_or_else(|| "127.0.0.1:9711".into());
            mesh::serve(&listen).map_err(|_| 1i32)
        }
        "pccnode" => {
            let listen = args
                .iter()
                .position(|a| a == "--listen")
                .and_then(|i| args.get(i + 1))
                .cloned()
                .unwrap_or_else(|| "127.0.0.1:9722".into());
            pcc::serve(&listen).map_err(|_| 1i32)
        }
        "peernode" => {
            let listen = args
                .iter()
                .position(|a| a == "--listen")
                .and_then(|i| args.get(i + 1))
                .cloned()
                .unwrap_or_else(|| "127.0.0.1:9733".into());
            let treaty = args
                .iter()
                .position(|a| a == "--treaty")
                .and_then(|i| args.get(i + 1))
                .cloned()
                .unwrap_or_else(|| "treaties.json".into());
            let prog_dir = args
                .iter()
                .position(|a| a == "--programs")
                .and_then(|i| args.get(i + 1))
                .cloned();
            peer::serve(&listen, &treaty, prog_dir.as_deref()).map_err(|_| 1i32)
        }
        "serve" => serve::serve(&args[2..]).map_err(|e| {
            eprintln!("serve: {e}");
            1i32
        }),
        "ask" => serve::ask(&args[2..]).map_err(|e| {
            eprintln!("ask: {e}");
            1i32
        }),
        "mind" => mind::mind(&args[2..]).map_err(|e| {
            eprintln!("mind: {e}");
            1i32
        }),
        "fleet" => fleet::fleet(&args[2..]).map_err(|e| {
            eprintln!("fleet: {e}");
            1i32
        }),
        "fleetcheck" => fleet::fleetcheck(&args[2..]).map_err(|e| {
            eprintln!("fleetcheck: {e}");
            1i32
        }),
        "ranknode" => ranknode::ranknode(&args[2..]).map_err(|e| {
            eprintln!("ranknode: {e}");
            1i32
        }),
        "lsp" => lsp::run().map_err(|_| 1i32),
        _ => coyote::cli::run(&args),
    };
    if let Err(code) = r {
        if code == 2 {
            usage();
        }
        std::process::exit(code);
    }
}
