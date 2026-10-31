//! `coyote-apple pkg` — the package registry.
//!
//! A local package store at `~/.coyote/pkg/` — `pkg install` unpacks a
//! `.cypack` into it, and the toolchain extends `CYO_PATH` with the
//! registry root, so `import "name"` resolves installed packages with
//! zero extra wiring. `pkg list` shows what's installed, `pkg remove`
//! deletes, `pkg search <q>` filters the index.

use serde_json::json;
use std::path::PathBuf;

fn root() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".coyote").join("pkg")
}

/// Registry root for CYO_PATH extension — main.rs calls this at boot.
pub fn registry_dir() -> PathBuf {
    root()
}

fn index_path() -> PathBuf {
    root().join("index.json")
}

fn load_index() -> serde_json::Value {
    std::fs::read_to_string(index_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({"packages": []}))
}

fn save_index(v: &serde_json::Value) -> Result<(), String> {
    std::fs::create_dir_all(root()).map_err(|e| e.to_string())?;
    std::fs::write(index_path(), serde_json::to_string_pretty(v).unwrap())
        .map_err(|e| e.to_string())
}

fn install(pkg: &str) -> Result<(), String> {
    let path = PathBuf::from(pkg);
    // read the real .cypack — extracts into <name>/ under the registry
    let (name, members, data) = crate::package::read_pack(&path)?;
    let dest = root().join(&name);
    if dest.exists() {
        std::fs::remove_dir_all(&dest).map_err(|e| e.to_string())?;
    }
    std::fs::create_dir_all(&dest).map_err(|e| e.to_string())?;
    let mut nfiles = 0usize;
    for m in &members {
        let start = m.offset as usize;
        let end = (start + m.len as usize).min(data.len());
        let p = dest.join(&m.name);
        if let Some(par) = p.parent() {
            std::fs::create_dir_all(par).map_err(|e| e.to_string())?;
        }
        std::fs::write(&p, &data[start..end]).map_err(|e| e.to_string())?;
        nfiles += 1;
    }
    // record in the index
    let mut idx = load_index();
    let arr = idx["packages"].as_array_mut().ok_or("bad index")?;
    arr.retain(|p| p["name"].as_str() != Some(name.as_str()));
    arr.push(json!({
        "name": name,
        "path": dest.display().to_string(),
        "files": nfiles,
        "installed": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    }));
    save_index(&idx)?;
    println!("✓ installed {name} → {}", dest.display());
    println!("  import \"{name}\" now resolves through the registry");
    Ok(())
}

fn list(json_out: bool) -> Result<(), String> {
    let idx = load_index();
    let arr = idx["packages"].as_array().cloned().unwrap_or_default();
    if json_out {
        println!("{}", serde_json::to_string_pretty(&arr).unwrap());
    } else {
        println!("coyote-apple pkg — {} installed", arr.len());
        for p in arr {
            println!(
                "  {:<24} {} file{}  {}",
                p["name"].as_str().unwrap_or("?"),
                p["files"].as_u64().unwrap_or(0),
                if p["files"].as_u64() == Some(1) { "" } else { "s" },
                p["path"].as_str().unwrap_or(""),
            );
        }
    }
    Ok(())
}

fn remove(name: &str) -> Result<(), String> {
    let dest = root().join(name);
    if dest.exists() {
        std::fs::remove_dir_all(&dest).map_err(|e| e.to_string())?;
    }
    let mut idx = load_index();
    if let Some(arr) = idx["packages"].as_array_mut() {
        arr.retain(|p| p["name"].as_str() != Some(name));
    }
    save_index(&idx)?;
    println!("✓ removed {name}");
    Ok(())
}

fn search(q: &str) -> Result<(), String> {
    let idx = load_index();
    let hits: Vec<_> = idx["packages"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|p| {
            p["name"]
                .as_str()
                .map(|n| n.contains(q))
                .unwrap_or(false)
        })
        .collect();
    for p in &hits {
        println!("  {}", p["name"].as_str().unwrap_or("?"));
    }
    if hits.is_empty() {
        println!("  (no packages match '{q}')");
    }
    Ok(())
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let r = match args.first().map(|s| s.as_str()) {
        Some("install") => args.get(1).map(|s| install(s)).unwrap_or_else(|| {
            eprintln!("usage: coyote-apple pkg install <pkg.cypack>");
            Err("missing package".into())
        }),
        Some("list") => list(args.iter().any(|a| a == "--json")),
        Some("remove") => args.get(1).map(|s| remove(s)).unwrap_or_else(|| {
            eprintln!("usage: coyote-apple pkg remove <name>");
            Err("missing name".into())
        }),
        Some("search") => args.get(1).map(|s| search(s)).unwrap_or_else(|| {
            eprintln!("usage: coyote-apple pkg search <q>");
            Err("missing query".into())
        }),
        _ => {
            eprintln!("usage: coyote-apple pkg install|list|remove|search ...");
            Err("bad subcommand".into())
        }
    };
    r.map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}
