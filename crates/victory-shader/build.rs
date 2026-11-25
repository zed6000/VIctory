//! `VICTORY_SHADER_SOURCE_HASH`: what the translator is built from (this
//! crate's sources, manifest and build script, the workspace's lock file
//! and the compiler's version), for caches of translations
//! (`victory_shader::SOURCE_HASH`).

use std::hash::{DefaultHasher, Hasher};
use std::path::{Path, PathBuf};

fn main() {
    let dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("set by cargo"));
    let mut files = vec![
        dir.join("Cargo.toml"),
        dir.join("build.rs"),
        dir.join("../../Cargo.lock"),
    ];
    sources(&dir.join("src"), &mut files);
    files.sort();
    // SipHash with fixed keys: stable for one compiler, whose version is
    // hashed too.
    let mut h = DefaultHasher::new();
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let version = std::process::Command::new(rustc).arg("-vV").output();
    h.write(&version.map(|o| o.stdout).unwrap_or_default());
    for f in &files {
        let name = f
            .strip_prefix(&dir)
            .unwrap_or(f)
            .to_string_lossy()
            .replace('\\', "/");
        h.write(name.as_bytes());
        h.write(&std::fs::read(f).unwrap_or_default());
    }
    for watched in ["src", "Cargo.toml", "build.rs", "../../Cargo.lock"] {
        println!("cargo:rerun-if-changed={watched}");
    }
    println!(
        "cargo:rustc-env=VICTORY_SHADER_SOURCE_HASH={:016x}",
        h.finish()
    );
}

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            sources(&path, out);
        } else {
            out.push(path);
        }
    }
}
