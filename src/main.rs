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
//!   coyote-apple oracle   <file.cy> [--steps N] [--json] [--cpu]
//!                                            semantic parity gate (ANE vs CPU)
//!   coyote-apple genmask  <name>:<rank>:<dim>  rank-selector mask → safetensors
//!   coyote-apple diff     <dirA> <dirB>        build-vs-build artifact delta
//!   coyote-apple test     <file.cy>            run host fn test_*() cases
//!   coyote-apple new      <name>               scaffold a coyote project
//!   coyote-apple weights  <inspect|slice>      safetensors checkpoint tooling
//!   coyote-apple doc      <file.cy>            generated API markdown

mod audit;
mod bench;
mod diff;
mod doc;
mod genmask;
mod new;
mod oracle;
mod package;
mod spec;
mod stfile;
mod test;
mod weights;

fn usage() {
    eprintln!("coyote-apple — internal profile of the coyote toolchain");
    eprintln!("  spec     <file.cy>");
    eprintln!("  audit    <build_dir> [--verify]");
    eprintln!("  package  <file.cy> [-o pkg.cypack]");
    eprintln!("  verify   <pkg.cypack>");
    eprintln!("  unpack   <pkg.cypack> -o <dir>");
    eprintln!("  bench    <file.cy> [-o dir] [--steps N] [--runs N] [--cpu] [--json]");
    eprintln!("  oracle   <file.cy> [--steps N] [--json] [--cpu]   semantic parity gate");
    eprintln!("  genmask  <name>:<rank>:<dim> [--rows] -o mask.safetensors");
    eprintln!("  diff     <dirA> <dirB>   per-file delta between two build dirs");
    eprintln!("  test     <file.cy> [--cpu]       run host fn test_*() cases");
    eprintln!("  new      <name> [--program]      scaffold a coyote project");
    eprintln!("  weights  inspect|slice ...       safetensors checkpoint tooling");
    eprintln!("  doc      <file.cy> [-o doc.md]   generated API markdown");
    eprintln!("  + all coyote verbs (check/build/run/repl) pass through unchanged");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(cmd) = args.get(1).map(|s| s.as_str()) else {
        usage();
        std::process::exit(2);
    };
    let r = match cmd {
        "spec" => spec::run(&args[2..]),
        "audit" => audit::run(&args[2..]),
        "package" | "pack" => package::run(&args[2..]),
        "verify" => package::verify(&args[2..]),
        "unpack" => package::unpack(&args[2..]),
        "bench" => bench::run(&args[2..]),
        "oracle" => oracle::run(&args[2..]),
        "genmask" => genmask::run(&args[2..]),
        "diff" => diff::run(&args[2..]),
        "test" => test::run(&args[2..]),
        "new" => new::run(&args[2..]),
        "weights" => weights::run(&args[2..]),
        "doc" => doc::run(&args[2..]),
        _ => coyote::cli::run(&args),
    };
    if let Err(code) = r {
        if code == 2 {
            usage();
        }
        std::process::exit(code);
    }
}
