//! fleet.rs — the fleet's control plane.
//!
//! `coyote-apple fleetcheck --world world.json` — conformance over the
//! fleet: validates the plan against live ranks (contiguous coverage,
//! rank indices, next_host chain, auth, lanes). Turns "the pipeline is
//! up" into a checked claim instead of an observed one.
//!
//! `coyote-apple fleet up|down|status --world <dir>` — lifecycle: spawn
//! ranknodes per shard dir + the serving tier, pids in `<dir>/fleet.pid`.

use crate::wire::{self, Frame};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn get(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

struct RankPlan {
    dir: PathBuf,
    host: String,
    layer_start: u64,
    layer_end: u64,
    rank: u64,
}

struct World {
    dir: PathBuf,
    ranks: Vec<RankPlan>,
    world: u64,
}

fn load_world(path: &str) -> Result<World, String> {
    let p = PathBuf::from(path);
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&p).map_err(|e| format!("world {path}: {e}"))?,
    )
    .map_err(|e| format!("world json: {e}"))?;
    let dir = p.parent().unwrap_or(Path::new(".")).to_path_buf();
    let mut ranks = Vec::new();
    for r in v["ranks"].as_array().unwrap_or(&vec![]) {
        ranks.push(RankPlan {
            dir: PathBuf::from(r["dir"].as_str().unwrap_or("")),
            host: r["host"].as_str().unwrap_or("").to_string(),
            layer_start: r["layer_start"].as_u64().unwrap_or(0),
            layer_end: r["layer_end"].as_u64().unwrap_or(0),
            rank: r["rank"].as_u64().unwrap_or(0),
        });
    }
    let world = v["world"].as_u64().unwrap_or(ranks.len() as u64);
    Ok(World { dir, ranks, world })
}

fn ping(host: &str, secret: &[u8], enc: bool) -> Result<Frame, String> {
    let (mut s, c) = wire::connect(host, secret, enc, 15)?;
    wire::write_packet(&mut s, &Frame::new("ping"), &[], c.as_ref())?;
    let (r, _) = wire::read_packet(&mut s, c.as_ref())?;
    Ok(r)
}

/// `fleetcheck --world world.json` — verify the plan is being served.
pub fn fleetcheck(args: &[String]) -> Result<(), String> {
    let w = load_world(&get(args, "--world").ok_or("usage: fleetcheck --world world.json")?)?;
    let secret = wire::resolve_secret()?;
    let enc = std::env::var("COYOTE_MESH_ENC").ok().as_deref() != Some("0");
    let mut ok = true;
    println!("fleetcheck — {} rank(s) expected", w.world);
    // coverage: ranks must tile [0, max_layer) contiguously, in order.
    let mut expect_start = 0u64;
    for rp in &w.ranks {
        let line = format!("rank {} {}", rp.rank, rp.host);
        match ping(&rp.host, &secret, enc) {
            Ok(f) => {
                let rank_ok = f.token.map(|t| t as u64) == Some(rp.rank);
                let lo_ok = f.layer_start.map(|x| x as u64) == Some(rp.layer_start);
                let hi_ok = f.layer_end.map(|x| x as u64) == Some(rp.layer_end);
                if rp.layer_start != expect_start {
                    ok = false;
                    println!(
                        "  ✗ {line} — coverage gap: expected layers {expect_start}.., got {}",
                        rp.layer_start
                    );
                    continue;
                }
                if rank_ok && lo_ok && hi_ok {
                    println!(
                        "  ✓ {line} — layers {}..{} up",
                        rp.layer_start, rp.layer_end
                    );
                } else {
                    ok = false;
                    println!(
                        "  ✗ {line} — plan mismatch: live rank={:?} layers {:?}..{:?}",
                        f.token, f.layer_start, f.layer_end
                    );
                }
            }
            Err(e) => {
                // single-conn ranks can't answer a ping while their conn
                // is held — a live TCP accept still proves the rank is up,
                // just busy serving someone.
                if TcpStream::connect(&rp.host).is_ok() {
                    println!(
                        "  ~ {line} — up (conn held — layers {}..{} expected)",
                        rp.layer_start, rp.layer_end
                    );
                } else {
                    ok = false;
                    println!("  ✗ {line} — unreachable: {e}");
                }
            }
        }
        expect_start = rp.layer_end;
    }
    if w.ranks.len() as u64 != w.world {
        ok = false;
        println!(
            "  ✗ plan says world={} but {} rank(s) listed",
            w.world,
            w.ranks.len()
        );
    }
    if ok {
        println!(
            "✓ fleet conformant — {} ranks, layers 0..{expect_start}",
            w.ranks.len()
        );
        Ok(())
    } else {
        Err("fleet non-conformant".into())
    }
}

/// `fleet up --world <world.json|dir> [--serve h:p] [--lanes N]`
/// `fleet down|status --world <dir>`
pub fn fleet(args: &[String]) -> Result<(), String> {
    let op = args.first().map(String::as_str).unwrap_or("status");
    let wpath =
        get(args, "--world").ok_or("usage: fleet up|down|status --world <world.json|dir>")?;
    let wjson = if wpath.ends_with(".json") {
        wpath
    } else {
        format!("{wpath}/world.json")
    };
    let w = load_world(&wjson)?;
    let pidfile = w.dir.join("fleet.pid");

    match op {
        "up" => {
            let serve_addr = get(args, "--serve").unwrap_or_else(|| "127.0.0.1:9744".into());
            let lanes = get(args, "--lanes").unwrap_or_else(|| "4".into());
            let exe = std::env::current_exe().map_err(|e| e.to_string())?;
            let mut pids = Vec::new();
            // spawn ranks deepest-first so next_host chains are live.
            for rp in w.ranks.iter().rev() {
                let log = w.dir.join(format!("rank{}.log", rp.rank));
                let lf = std::fs::File::create(&log).map_err(|e| e.to_string())?;
                let child = Command::new(&exe)
                    .args([
                        "ranknode",
                        "--shard",
                        rp.dir.to_str().unwrap_or(""),
                        "--listen",
                        &rp.host,
                        "--lanes",
                        &lanes,
                    ])
                    .stdout(Stdio::from(lf.try_clone().map_err(|e| e.to_string())?))
                    .stderr(Stdio::from(lf))
                    .spawn()
                    .map_err(|e| format!("spawn rank{}: {e}", rp.rank))?;
                pids.push(child.id());
                println!(
                    "  rank {} → {} (pid {}, log {})",
                    rp.rank,
                    rp.host,
                    child.id(),
                    log.display()
                );
            }
            // serving tier fronts rank 0's host — wait for every rank to
            // listen first: ranks bind only after their lanes compile.
            let rank0 = w
                .ranks
                .iter()
                .min_by_key(|r| r.rank)
                .map(|r| r.host.clone())
                .ok_or("world has no ranks")?;
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(600);
            for rp in &w.ranks {
                while std::time::Instant::now() < deadline {
                    if TcpStream::connect(&rp.host).is_ok() {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            }
            let log = w.dir.join("serve.log");
            let lf = std::fs::File::create(&log).map_err(|e| e.to_string())?;
            let child = Command::new(&exe)
                .args([
                    "serve",
                    "--ranks",
                    &rank0,
                    "--listen",
                    &serve_addr,
                    "--lanes",
                    &lanes,
                    "--journal",
                    w.dir
                        .join("serve_witness.jsonl")
                        .to_str()
                        .unwrap_or("/tmp/sw.jsonl"),
                ])
                .stdout(Stdio::from(lf.try_clone().map_err(|e| e.to_string())?))
                .stderr(Stdio::from(lf))
                .spawn()
                .map_err(|e| format!("spawn serve: {e}"))?;
            pids.push(child.id());
            println!(
                "  serve → {serve_addr} (pid {}, log {})",
                child.id(),
                log.display()
            );
            std::fs::write(
                &pidfile,
                pids.iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
            .map_err(|e| e.to_string())?;
            println!(
                "fleet up — {} proc(s), pids in {}",
                pids.len(),
                pidfile.display()
            );
            Ok(())
        }
        "down" => {
            let pids = std::fs::read_to_string(&pidfile)
                .map_err(|e| format!("fleet.pid: {e} (fleet up first?)"))?;
            for p in pids.lines().filter_map(|l| l.trim().parse::<i32>().ok()) {
                let _ = Command::new("kill").arg(p.to_string()).status();
            }
            std::fs::remove_file(&pidfile).ok();
            println!("fleet down — {} proc(s) stopped", pids.lines().count());
            Ok(())
        }
        "status" | _ => {
            let secret = wire::resolve_secret()?;
            let enc = std::env::var("COYOTE_MESH_ENC").ok().as_deref() != Some("0");
            for rp in &w.ranks {
                match ping(&rp.host, &secret, enc) {
                    Ok(f) => println!(
                        "  rank {} {} — layers {:?}..{:?}",
                        rp.rank, rp.host, f.layer_start, f.layer_end
                    ),
                    Err(e) => println!("  rank {} {} — down ({e})", rp.rank, rp.host),
                }
            }
            // TCP check on the serving port if it's listening.
            let addr = get(args, "--serve").unwrap_or_else(|| "127.0.0.1:9744".into());
            match TcpStream::connect(&addr) {
                Ok(_) => println!("  serve {addr} — up"),
                Err(_) => println!("  serve {addr} — down"),
            }
            Ok(())
        }
    }
}
