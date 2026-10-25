fn main() {
    let path = std::env::args().nth(1).unwrap();
    let name = std::env::args().nth(2).unwrap();
    let dir = std::env::args().nth(3).unwrap();
    let (src, ast) = coyote::parser::parse_file(std::path::Path::new(&path)).unwrap();
    let checked = coyote::typeck::check_all(&ast).unwrap();
    for (n, p) in &checked.ane {
        if n == &name {
            let out = coyote::emit_program(&src, p, std::path::Path::new(&dir), true).unwrap();
            std::fs::write(format!("{dir}/net.mil"), &out.mil).unwrap();
            for (f, b) in &out.blobs { std::fs::write(format!("{dir}/{f}"), b).unwrap(); }
            println!("emitted {name} -> {dir}");
        }
    }
}
