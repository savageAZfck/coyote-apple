//! `.cypack` — a coyote program as a single distributable file.
//!
//! `coyote-apple package model.cy -o model.cypack` builds the program
//! through the normal pipeline (emit → blobs → manifest) and packs the
//! artifacts: `CYPK1\n` + u32 header length + JSON header (per-file
//! sha256 + offsets) + concatenated payload bytes. `verify` re-hashes
//! every member against the header; `unpack` extracts it back into a
//! runnable build dir.

use coyote::lower::sha256_hex;
use std::path::{Path, PathBuf};

const MAGIC: &[u8] = b"CYPK1\n";

pub struct Member {
    pub name: String,
    pub sha: String,
    pub offset: u64,
    pub len: u64,
}

fn write_pack(out: &Path, name: &str, files: &[(String, Vec<u8>)]) -> Result<(), String> {
    let mut offset = 0u64;
    let mut members = Vec::new();
    let mut header = String::from("{\n");
    header.push_str(&format!("  \"name\": \"{name}\",\n"));
    header.push_str(&format!(
        "  \"built\": {},\n",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    ));
    header.push_str("  \"files\": [\n");
    for (i, (fname, bytes)) in files.iter().enumerate() {
        let sha = sha256_hex(bytes);
        let comma = if i + 1 == files.len() { "" } else { "," };
        header.push_str(&format!(
            "    {{\"name\": \"{fname}\", \"len\": {}, \"sha256\": \"{sha}\", \"offset\": {offset}}}{comma}\n",
            bytes.len()
        ));
        members.push(Member {
            name: fname.clone(),
            sha,
            offset,
            len: bytes.len() as u64,
        });
        offset += bytes.len() as u64;
    }
    header.push_str("  ]\n}\n");
    let mut pkg = Vec::with_capacity(MAGIC.len() + 4 + header.len() + offset as usize);
    pkg.extend_from_slice(MAGIC);
    pkg.extend_from_slice(&(header.len() as u32).to_le_bytes());
    pkg.extend_from_slice(header.as_bytes());
    for (_, bytes) in files {
        pkg.extend_from_slice(bytes);
    }
    std::fs::write(out, &pkg).map_err(|e| format!("{}: {e}", out.display()))?;
    let total: u64 = members.iter().map(|m| m.len).sum();
    println!(
        "✓ packaged {name} → {} ({} files, {total} payload bytes)",
        out.display(),
        members.len()
    );
    for m in &members {
        println!("  {:<24} {:>8} B  {}", m.name, m.len, m.sha);
    }
    Ok(())
}

pub fn read_pack(path: &Path) -> Result<(String, Vec<Member>, Vec<u8>), String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !raw.starts_with(MAGIC) {
        return Err(format!("{}: not a .cypack (bad magic)", path.display()));
    }
    let hlen = u32::from_le_bytes(raw[MAGIC.len()..MAGIC.len() + 4].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&raw[MAGIC.len() + 4..MAGIC.len() + 4 + hlen])
        .map_err(|e| format!("header: {e}"))?;
    let name = header
        .split("\"name\": \"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .unwrap_or("pkg")
        .to_string();
    // each member line: {"name": "f", "len": n, "sha256": "h", "offset": o}
    let mut members = Vec::new();
    for line in header.lines() {
        if !line.contains("\"sha256\"") {
            continue;
        }
        let grab = |key: &str| -> Option<String> {
            line.split(&format!("\"{key}\": ")).nth(1).map(|s| {
                s.trim_start_matches('"')
                    .split([',', '"', '}'].as_ref())
                    .next()
                    .unwrap_or("")
                    .to_string()
            })
        };
        members.push(Member {
            name: grab("name").unwrap_or_default(),
            sha: grab("sha256").unwrap_or_default(),
            offset: grab("offset").and_then(|s| s.parse().ok()).unwrap_or(0),
            len: grab("len").and_then(|s| s.parse().ok()).unwrap_or(0),
        });
    }
    Ok((name, members, raw[MAGIC.len() + 4 + hlen..].to_vec()))
}

pub fn run(args: &[String]) -> Result<(), i32> {
    let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: coyote-apple package <file.cy> [-o pkg.cypack]");
        return Err(2);
    };
    let path = PathBuf::from(file);
    let out = args
        .windows(2)
        .find(|w| w[0] == "-o")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(|| path.with_extension("cypack"));
    let build_dir = std::env::temp_dir().join(format!(
        "cypack_{}",
        path.file_stem().and_then(|s| s.to_str()).unwrap_or("pkg")
    ));

    let r = (|| -> Result<(), String> {
        // the normal pipeline — emit + compile exactly as `coyote build`
        let argv: Vec<String> = [
            "coyote",
            "build",
            path.to_str().unwrap_or(file),
            "-o",
            build_dir.to_str().unwrap_or("cypack_tmp"),
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        coyote::cli::run(&argv).map_err(|c| format!("build failed (exit {c})"))?;

        let mut files: Vec<(String, Vec<u8>)> = Vec::new();
        fn walk(dir: &Path, base: &Path, out: &mut Vec<(String, Vec<u8>)>) {
            if let Ok(rd) = std::fs::read_dir(dir) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        walk(&p, base, out);
                    } else if let Ok(rel) = p.strip_prefix(base) {
                        if let Some(name) = rel.to_str() {
                            if let Ok(bytes) = std::fs::read(&p) {
                                out.push((name.to_string(), bytes));
                            }
                        }
                    }
                }
            }
        }
        walk(&build_dir, &build_dir, &mut files);
        // ship the program's source too — `import "name"` resolves
        // `name/mod.cy`, so the pack carries everything needed to
        // rebuild or re-namespace it downstream.
        let (combined, _prog) =
            coyote::parser::parse_file(&path).map_err(|d| d.to_string())?;
        files.push(("mod.cy".to_string(), combined.into_bytes()));
        files.sort_by(|a, b| a.0.cmp(&b.0));
        if files.is_empty() {
            return Err("build dir empty — nothing to package".into());
        }
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("program")
            .to_string();
        write_pack(&out, &name, &files)
    })();
    r.map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}

pub fn verify(args: &[String]) -> Result<(), i32> {
    let Some(pkg) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: coyote-apple verify <pkg.cypack>");
        return Err(2);
    };
    let r = (|| -> Result<(), String> {
        let (name, members, data) = read_pack(Path::new(pkg))?;
        let mut bad = 0;
        for m in &members {
            let start = m.offset as usize;
            let end = start + m.len as usize;
            if end > data.len() {
                println!("  ✗ {} — truncated payload", m.name);
                bad += 1;
                continue;
            }
            let sha = sha256_hex(&data[start..end]);
            if sha == m.sha {
                println!("  ✓ {:<24} {}", m.name, sha);
            } else {
                println!("  ✗ {:<24} recorded {}", m.name, m.sha);
                bad += 1;
            }
        }
        if bad > 0 {
            return Err(format!("{name}: {bad} corrupted members"));
        }
        println!("✓ {name} — {} members intact", members.len());
        Ok(())
    })();
    r.map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}

pub fn unpack(args: &[String]) -> Result<(), i32> {
    let Some(pkg) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: coyote-apple unpack <pkg.cypack> -o <dir>");
        return Err(2);
    };
    let out = args
        .windows(2)
        .find(|w| w[0] == "-o")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(|| PathBuf::from("cypack_out"));
    let r = (|| -> Result<(), String> {
        let (name, members, data) = read_pack(Path::new(pkg))?;
        std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
        for m in &members {
            let start = m.offset as usize;
            let end = (start + m.len as usize).min(data.len());
            std::fs::write(out.join(&m.name), &data[start..end])
                .map_err(|e| format!("{}: {e}", m.name))?;
            println!("  ✓ {:<24} {} B", m.name, m.len);
        }
        println!("✓ unpacked {name} → {}", out.display());
        Ok(())
    })();
    r.map_err(|e| {
        eprintln!("✗ {e}");
        1
    })
}
