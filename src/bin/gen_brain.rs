// gen_brain — emit a parameterized Coyote-Apple decoder organism.
//
// Rust port of the brain generator — no Python in this stack.
// Co-designed for the verified ANE activation×activation skeleton
// table; weights are born on-chip via init_weight; kv projections stay
// frozen at the state-cache boundary (the autodiff tape ends at
// StateUpdate).
//
// Matmul convention (coyote eval): `x @ w` — w declared f16[in, out],
// storage [out, in] row-major.
//
//   coyote-apple-gen-brain --layers 2 --dmodel 64 --out brain2.cy

#[derive(Default)]
struct Spec {
    layers: usize,
    dmodel: usize,
    head_dim: usize,
    qheads: usize,
    kvheads: usize,
    ffn_branches: usize,
    ffn_dim: usize,
    vocab: usize,
    seq_len: usize,
    positions: usize,
    epochs: usize,
    rounds: usize,
    text: Option<String>,
    corpus_len: usize,
}

fn arg<T: std::str::FromStr>(args: &[String], flag: &str, cur: &mut T) {
    if let Some(i) = args.iter().position(|a| a == flag) {
        if let Some(v) = args.get(i + 1) {
            *cur = v.parse().unwrap_or_else(|_| panic!("bad value for {flag}"));
        }
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn uni_f32(st: &mut u64) -> f32 {
    *st = st.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *st;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    let u = (z >> 11) as f64 / (1u64 << 53) as f64;
    (((u * 2.0) - 1.0) * 0.1) as f32
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut s = Spec {
        layers: 1,
        dmodel: 64,
        head_dim: 0, // 0 = dmodel (back-compat); ANE route wants 64
        qheads: 4,
        kvheads: 2,
        ffn_branches: 2,
        ffn_dim: 64,
        vocab: 64,
        seq_len: 64,
        positions: 16,
        epochs: 60,
        rounds: 6,
        text: None,
        corpus_len: 8192,
    };
    arg(&args, "--layers", &mut s.layers);
    arg(&args, "--dmodel", &mut s.dmodel);
    arg(&args, "--head-dim", &mut s.head_dim);
    arg(&args, "--qheads", &mut s.qheads);
    arg(&args, "--kvheads", &mut s.kvheads);
    arg(&args, "--ffn-branches", &mut s.ffn_branches);
    arg(&args, "--ffn-dim", &mut s.ffn_dim);
    arg(&args, "--vocab", &mut s.vocab);
    arg(&args, "--seq-len", &mut s.seq_len);
    arg(&args, "--positions", &mut s.positions);
    arg(&args, "--epochs", &mut s.epochs);
    arg(&args, "--corpus-len", &mut s.corpus_len);
    arg(&args, "--rounds", &mut s.rounds);
    if let Some(i) = args.iter().position(|a| a == "--text") {
        s.text = args.get(i + 1).cloned();
    }
    let out = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "brain.cy".into());
    let corpus_path = format!(
        "{}.corpus.safetensors",
        out.strip_suffix(".cy").unwrap_or(&out)
    );
    let corpus_abs = std::path::absolute(&corpus_path)
        .unwrap_or_else(|_| std::path::PathBuf::from(&corpus_path))
        .display()
        .to_string();

    // text mode: byte-level curriculum — vocab is the byte range, the
    // corpus supplies seq/tgts/cemb tensors instead of the synthetic
    // position-ordinal drill.
    let toks: Option<Vec<u8>> = s.text.as_ref().map(|p| {
        let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"));
        assert!(bytes.len() >= 2, "text corpus needs >= 2 bytes");
        bytes
            .into_iter()
            .take(s.corpus_len + 1)
            .collect::<Vec<u8>>()
    });
    if toks.is_some() {
        s.vocab = 256;
    }
    let n_corpus = toks.as_ref().map_or(0usize, |t| t.len() - 1);
    // chunked block-coordinate rounds: each round gives every train
    // program a `chunk`-epoch residency streak (no per-step program
    // switches → ANE programs stay loaded), while rounds preserve
    // block co-adaptation. Overshoot past `epochs` is harmless —
    // corpus indices wrap.
    let rounds = s.rounds.max(1).min(s.epochs.max(1));
    let chunk = s.epochs.div_ceil(rounds);

    assert!(
        s.qheads % s.kvheads == 0,
        "qheads must be a multiple of kvheads"
    );
    if toks.is_none() {
        assert!(
            s.positions < s.vocab,
            "positions must be < vocab (i+1 target)"
        );
        assert!(
            s.vocab <= s.seq_len,
            "vocab must be <= seq_len (gen gather)"
        );
    }
    assert!(
        s.dmodel <= 1024,
        "dmodel > 1024 exceeds the qwen3-0.6b embed slice"
    );

    if s.head_dim == 0 {
        s.head_dim = s.dmodel;
    }
    let (l_n, d, hd, q, kv, fb, fd, v, sq, pos, ep) = (
        s.layers,
        s.dmodel,
        s.head_dim,
        s.qheads,
        s.kvheads,
        s.ffn_branches,
        s.ffn_dim,
        s.vocab,
        s.seq_len,
        s.positions,
        s.epochs,
    );
    let ratio = q / kv;
    let scale = 1.0 / (hd as f64).sqrt();

    let mut o = String::new();
    macro_rules! w {
        ($line:expr) => {{
            o.push_str(&$line.to_string());
            o.push('\n');
        }};
    }

    w!("// GENERATED by coyote-apple-gen-brain — do not hand-edit.");
    w!(&format!(
        "// spec: L={l_n} D={d} HD={hd} Q={q} KV={kv} ffn={fb}x{fd} V={v} S={sq} pos={pos} ep={ep}"
    ));
    w!("// all transformer params born on-chip + trained on-chip (grad+adam);");
    w!("// kv projections frozen at the state-cache boundary (tape ends at");
    w!("// StateUpdate — they are used every forward, never updated).");
    w!("");
    w!("invariant never_sells_self");
    w!("");
    w!("provenance {");
    w!("    author: \"savage\"");
    w!("    purpose: \"organism-grown mind — trainable decoder on ANE\"");
    w!("    license: \"BSL-1.1\"");
    w!("}");
    w!("");

    // params
    for l in 0..l_n {
        w!(&format!("// ---- layer {l} ----"));
        for nm in ["ln1", "ln2"] {
            w!(&format!("param l{l}_{nm}: trainable f16[{d}]"));
        }
        for nm in ["qnl", "qnh", "knl", "knh"] {
            w!(&format!("param l{l}_{nm}: trainable f16[{hd}]"));
        }
        for h in 0..kv {
            for pr in ["k", "v"] {
                for hf in ["lo", "hi"] {
                    w!(&format!("param l{l}_{pr}{h}{hf}: frozen f16[{d}, {hd}]"));
                }
            }
        }
        for h in 0..q {
            for hf in ["lo", "hi"] {
                w!(&format!("param l{l}_q{h}{hf}: trainable f16[{d}, {hd}]"));
                w!(&format!("param l{l}_o{h}{hf}: trainable f16[{hd}, {d}]"));
            }
        }
        for b in 0..fb {
            w!(&format!("param l{l}_wg{b}: trainable f16[{d}, {fd}]"));
            w!(&format!("param l{l}_wu{b}: trainable f16[{d}, {fd}]"));
            w!(&format!("param l{l}_wd{b}: trainable f16[{fd}, {d}]"));
        }
        w!("");
    }
    w!(&format!("param lnf: trainable f16[{d}]"));
    w!(&format!("param lmh: trainable f16[{d}, {v}]"));
    w!("");

    // state
    w!("state kv {");
    for l in 0..l_n {
        for h in 0..kv {
            for pr in ["k", "v"] {
                for hf in ["lo", "hi"] {
                    w!(&format!("    l{l}_{pr}{h}{hf}: f16[{sq}, {hd}]"));
                }
            }
        }
    }
    w!("}");
    w!("");

    // tables
    if toks.is_some() {
        let n = toks.as_ref().unwrap().len() - 1;
        w!(&format!(
            "table seq: f16[{n}, {d}] @ \"{corpus_abs}::seq\""
        ));
        w!("");
        w!(&format!(
            "table tgts: f16[{n}, {v}] @ \"{corpus_abs}::tgts\""
        ));
        w!("");
        w!(&format!(
            "table cemb: f16[{v}, {d}] @ \"{corpus_abs}::cemb\""
        ));
        w!("");
    } else {
        w!(&format!(
            "table emb: f16[151936, {d}] @ \"models/qwen3-0.6b/model.safetensors::model.embed_tokens.weight[0:151936,0:{d}]\""
        ));
        w!("");
        w!(&format!(
            "table seq: f16[{sq}, {d}] @ \"{corpus_abs}::seq\""
        ));
        w!("");
    }

    // forward signature
    w!("fn forward(");
    w!(&format!(
        "    x: f16[{d}], pos: f16[{sq}], mask: f16[{sq}],"
    ));
    w!(&format!("    cos: f16[{hd}], sin: f16[{hd}]"));
    w!(&format!(") -> f16[{v}] {{"));

    // layer bodies
    let mut xin = "x".to_string();
    for l in 0..l_n {
        let mut b: Vec<String> = Vec::new();
        let a = &mut b;
        a.push(format!("    let h{l} = rmsnorm({xin}, l{l}_ln1);"));
        for h in 0..q {
            a.push(format!(
                "    let q{l}_{h}_lo = rmsnorm(h{l} @ l{l}_q{h}lo, l{l}_qnl);"
            ));
            a.push(format!(
                "    let q{l}_{h}_hi = rmsnorm(h{l} @ l{l}_q{h}hi, l{l}_qnh);"
            ));
            a.push(format!(
                "    let rq{l}_{h}_lo = q{l}_{h}_lo * cos - q{l}_{h}_hi * sin;"
            ));
            a.push(format!(
                "    let rq{l}_{h}_hi = q{l}_{h}_hi * cos + q{l}_{h}_lo * sin;"
            ));
        }
        for h in 0..kv {
            a.push(format!(
                "    let k{l}_{h}_lo = rmsnorm(h{l} @ l{l}_k{h}lo, l{l}_knl);"
            ));
            a.push(format!(
                "    let k{l}_{h}_hi = rmsnorm(h{l} @ l{l}_k{h}hi, l{l}_knh);"
            ));
            a.push(format!("    let v{l}_{h}_lo = h{l} @ l{l}_v{h}lo;"));
            a.push(format!("    let v{l}_{h}_hi = h{l} @ l{l}_v{h}hi;"));
            a.push(format!("    kv.l{l}_k{h}lo[pos] = k{l}_{h}_lo;"));
            a.push(format!("    kv.l{l}_k{h}hi[pos] = k{l}_{h}_hi;"));
            a.push(format!("    kv.l{l}_v{h}lo[pos] = v{l}_{h}_lo;"));
            a.push(format!("    kv.l{l}_v{h}hi[pos] = v{l}_{h}_hi;"));
        }
        for h in 0..q {
            let kvh = h / ratio;
            a.push(format!(
                "    let s{l}_{h} = softmax((rq{l}_{h}_lo @ kv.l{l}_k{kvh}lo.T + rq{l}_{h}_hi @ kv.l{l}_k{kvh}hi.T) * {scale:.6} + mask);"
            ));
            a.push(format!(
                "    let a{l}_{h} = (s{l}_{h} @ kv.l{l}_v{kvh}lo) @ l{l}_o{h}lo + (s{l}_{h} @ kv.l{l}_v{kvh}hi) @ l{l}_o{h}hi;"
            ));
        }
        let terms: Vec<String> = (0..q).map(|h| format!("a{l}_{h}")).collect();
        a.push(format!("    let att{l} = {};", terms.join(" + ")));
        a.push(format!("    let r{l} = {xin} + att{l};"));
        a.push(format!("    let n{l} = rmsnorm(r{l}, l{l}_ln2);"));
        for bc in 0..fb {
            a.push(format!(
                "    let f{l}_{bc} = (silu(n{l} @ l{l}_wg{bc}) * (n{l} @ l{l}_wu{bc})) @ l{l}_wd{bc};"
            ));
        }
        let fterms: Vec<String> = (0..fb).map(|bc| format!("f{l}_{bc}")).collect();
        a.push(format!("    let f{l} = r{l} + {};", fterms.join(" + ")));
        for line in b {
            w!(&line);
        }
        xin = format!("f{l}");
    }
    o.push_str(&format!("    rmsnorm({xin}, lnf) @ lmh\n}}\n\n"));

    // Per-layer train fns + head fn. A joint train_step re-emits the
    // forward once per param (~150 grads × full graph) and hits the
    // compile-time wall past ~2 layers; per-layer programs keep each
    // graph at ~1 layer's params. Host dispatches round-robin —
    // block coordinate descent, a real technique.
    w!("// teacher-forced step: forward + adam on this layer's params.");
    let fw = "forward(x, pos, mask, cos, sin)";
    for l in 0..l_n {
        w!(&format!("fn train_l{l}("));
        w!(&format!(
            "    x: f16[{d}], pos: f16[{sq}], mask: f16[{sq}],"
        ));
        w!(&format!("    cos: f16[{hd}], sin: f16[{hd}], tgt: f16[{v}]"));
        w!(&format!(") -> f16[{v}] {{"));
        for nm in ["ln1", "ln2", "qnl", "qnh"] {
            w!(&format!(
                "    adam(l{l}_{nm}, grad(mse({fw}, tgt), l{l}_{nm}), 0.01);"
            ));
        }
        for h in 0..q {
            for hf in ["lo", "hi"] {
                w!(&format!(
                    "    adam(l{l}_q{h}{hf}, grad(mse({fw}, tgt), l{l}_q{h}{hf}), 0.05);"
                ));
                w!(&format!(
                    "    adam(l{l}_o{h}{hf}, grad(mse({fw}, tgt), l{l}_o{h}{hf}), 0.05);"
                ));
            }
        }
        for bc in 0..fb {
            for pr in ["wg", "wu", "wd"] {
                w!(&format!(
                    "    adam(l{l}_{pr}{bc}, grad(mse({fw}, tgt), l{l}_{pr}{bc}), 0.05);"
                ));
            }
        }
        w!(&format!("    {fw}"));
        w!("}");
        w!("");
    }
    w!("// head step: final norm + logit head.");
    w!("fn train_head(");
    w!(&format!(
        "    x: f16[{d}], pos: f16[{sq}], mask: f16[{sq}],"
    ));
    w!(&format!("    cos: f16[{hd}], sin: f16[{hd}], tgt: f16[{v}]"));
    w!(&format!(") -> f16[{v}] {{"));
    w!(&format!("    adam(lnf, grad(mse({fw}, tgt), lnf), 0.01);"));
    w!(&format!("    adam(lmh, grad(mse({fw}, tgt), lmh), 0.05);"));
    w!(&format!("    {fw}"));
    w!("}");
    w!("");

    // host
    w!("host fn main() -> i64 {");
    w!("    let me = self_token();");
    w!("    witness(\"born\", me);");
    w!("    print(\"brain organism\", me.id);");
    w!("");
    // chunked block coordinate descent: `rounds` alternations of
    // phase streaks. Each streak runs `chunk` epochs through ONE
    // train program, so the host never switches ANE programs inside
    // a streak and residency survives (release_all on switch made
    // every step a cold ~7k-program reload). Blocks still co-adapt
    // every `chunk` epochs.
    w!("    var round = 0;");
    w!(&format!("    while round < {rounds} {{"));
    w!("        var phase = 0;");
    w!(&format!("        while phase < {} {{", l_n + 1));
    w!(&format!("            var ep = round * {chunk};"));
    w!(&format!("            var emax = ep + {chunk};"));
    w!("            while ep < emax {");
    w!(&format!("                var i = 0;\n                while i < {pos} {{"));
    if toks.is_some() {
        w!(&format!(
            "                    let ci = (ep * {pos} + i) % {n_corpus};"
        ));
        w!("                    let x = seq[ci];");
        w!("                    let tgt = tgts[ci];");
    } else {
        w!("                    let x = seq[i];");
        w!(&format!("                    let tgt = onehot(i + 1, {v});"));
    }
    w!(&format!("                    let p = onehot(i, {sq});"));
    w!(&format!("                    let m = causal(i, {sq});"));
    w!(&format!("                    let c = rope_cos(i, {hd});"));
    w!(&format!("                    let s = rope_sin(i, {hd});"));
    // each phase branch is self-contained: exactly one trainer runs
    // per step, so the host never touches a second ANE program inside
    // a streak and residency survives the whole streak.
    for l in 0..l_n {
        w!(&format!("                    if phase == {l} {{"));
        w!(&format!(
            "                        let lg = train_l{l}(x, p, m, c, s, tgt);"
        ));
        w!("                        let pred = argmax(lg);");
        if toks.is_some() {
            w!("                        print(\"ep\", ep, \"pos\", i, \"pred\", pred, \"want\", argmax(tgt));");
        } else {
            w!("                        print(\"ep\", ep, \"pos\", i, \"pred\", pred, \"want\", i + 1);");
        }
        w!("                        witness(\"train_pred\", pred);");
        w!("                    }");
    }
    w!(&format!("                    if phase == {l_n} {{"));
    w!("                        let lg = train_head(x, p, m, c, s, tgt);");
    w!("                        let pred = argmax(lg);");
    if toks.is_some() {
        w!("                        print(\"ep\", ep, \"pos\", i, \"pred\", pred, \"want\", argmax(tgt));");
    } else {
        w!("                        print(\"ep\", ep, \"pos\", i, \"pred\", pred, \"want\", i + 1);");
    }
    w!("                        witness(\"train_pred\", pred);");
    w!("                    }");
    w!("                    i += 1;");
    w!("                }");
    w!("                ep += 1;");
    w!("            }");
    w!("            phase += 1;");
    w!("        }");
    w!("        round += 1;");
    w!("    }");
    w!("");
    w!("    var tok = 0;");
    if toks.is_some() {
        w!("    var x = cemb[tok];");
    } else {
        w!("    var x = seq[tok];");
    }
    w!("    var step = 0;");
    w!("    while step < 8 {");
    w!("        let lg = forward(");
    w!("            x,");
    w!(&format!("            onehot(step % {sq}, {sq}),"));
    w!(&format!("            causal(step % {sq}, {sq}),"));
    w!(&format!("            rope_cos(step % {sq}, {hd}),"));
    w!(&format!("            rope_sin(step % {sq}, {hd})"));
    w!("        );");
    w!("        tok = argmax(lg);");
    w!("        print(\"gen\", step, \"tok\", tok);");
    w!("        witness(\"tok\", tok);");
    if toks.is_some() {
        w!("        x = cemb[tok];");
    } else {
        w!(&format!("        x = seq[tok % {sq}];"));
    }
    w!("        step += 1;");
    w!("    }");
    w!("    witness(\"done\", step);");
    w!("    return tok;");
    w!("}");

    std::fs::write(&out, &o).expect("write output");

    // companion corpus safetensors. Synthetic mode: seq [sq, d] of
    // spec-seeded ~U(-0.1,0.1) vectors (position-ordinal drill). Text
    // mode (--text): byte-level curriculum —
    //   cemb [v, d]   frozen seeded embedding table
    //   seq  [n, d]   cemb[tok[j]] per corpus position j
    //   tgts [n, v]   one-hot of tok[j+1]
    // Deterministic per (spec, text) so a given invocation regrows the
    // identical organism.
    let mut tensors: Vec<(&str, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut st = fnv1a(format!("gen_brain seq L{l_n} D{d} Q{q} S{sq}").as_bytes());
    if let Some(t) = &toks {
        for &b in t {
            st ^= b as u64;
            st = st.wrapping_mul(0x100000001b3);
        }
    }
    if let Some(t) = &toks {
        // cemb [v, d]
        let v = s.vocab;
        let mut cemb = vec![0f32; v * d];
        for x in cemb.iter_mut() {
            *x = uni_f32(&mut st);
        }
        // seq [n, d] = cemb[tok[j]]
        let n = t.len() - 1;
        let mut seqb = vec![0f32; n * d];
        for j in 0..n {
            let row = &cemb[t[j] as usize * d..t[j] as usize * d + d];
            seqb[j * d..j * d + d].copy_from_slice(row);
        }
        // tgts [n, v] one-hot of tok[j+1]
        let mut tg = vec![0f32; n * v];
        for j in 0..n {
            tg[j * v + t[j + 1] as usize] = 1.0;
        }
        let f32b = |vals: &[f32]| -> Vec<u8> {
            vals.iter().flat_map(|x| x.to_le_bytes()).collect()
        };
        tensors.push(("cemb", vec![v, d], f32b(&cemb)));
        tensors.push(("seq", vec![n, d], f32b(&seqb)));
        tensors.push(("tgts", vec![n, v], f32b(&tg)));
    } else {
        let n = sq * d;
        let mut vals = vec![0f32; n];
        for x in vals.iter_mut() {
            *x = uni_f32(&mut st);
        }
        tensors.push((
            "seq",
            vec![sq, d],
            vals.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ));
    }
    let mut parts = Vec::with_capacity(tensors.len());
    let mut off = 0usize;
    for (name, shape, bytes) in &tensors {
        let shape_str = shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        parts.push(format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape_str}],\"data_offsets\":[{off}, {}]}}",
            off + bytes.len()
        ));
        off += bytes.len();
    }
    let header = format!("{{{}}}", parts.join(","));
    let mut hbytes = header.into_bytes();
    while (8 + hbytes.len()) % 8 != 0 {
        hbytes.push(b' ');
    }
    let mut buf = Vec::with_capacity(8 + hbytes.len() + off);
    buf.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(&hbytes);
    for (_, _, bytes) in &tensors {
        buf.extend_from_slice(bytes);
    }
    std::fs::write(&corpus_path, &buf).expect("write corpus");

    eprintln!("wrote {out} ({} lines) + {corpus_path}", o.lines().count());
}
