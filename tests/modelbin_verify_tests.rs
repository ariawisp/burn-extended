use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn bin(name: &str) -> Option<PathBuf> {
    let key = format!("CARGO_BIN_EXE_{}", name);
    env::var_os(&key).map(PathBuf::from)
}

fn tmp_dir(name: &str) -> PathBuf {
    let mut p = env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("{}_{}", name, nanos));
    p
}

#[test]
fn exporter_and_verifier_succeed_on_fixture() {
    // Prepare fixture
    let out = tmp_dir("gptoss_fixture_verify");
    fs::create_dir_all(&out).unwrap();
    let rust_bin = out.join("model_rust.bin");

    // Generate fixture via bin
    let status = if let Some(fixture) = bin("gptoss_fixture") {
        Command::new(&fixture)
            .args(["-o", out.to_str().unwrap()])
            .status()
            .expect("spawn fixture")
    } else {
        // Fallback: run via cargo
        Command::new("cargo")
            .args([
                "run",
                "-q",
                "-p",
                "burn-extended",
                "--bin",
                "gptoss_fixture",
                "--",
                "-o",
                out.to_str().unwrap(),
            ])
            .status()
            .expect("cargo run fixture")
    };
    assert!(status.success(), "fixture generation failed");

    // Run Rust exporter
    let status = if let Some(exporter) = bin("gptoss_export") {
        Command::new(&exporter)
            .args(["-s", out.to_str().unwrap(), "-d", rust_bin.to_str().unwrap()])
            .status()
            .expect("spawn exporter")
    } else {
        Command::new("cargo")
            .args([
                "run",
                "-q",
                "-p",
                "burn-extended",
                "--bin",
                "gptoss_export",
                "--",
                "-s",
                out.to_str().unwrap(),
                "-d",
                rust_bin.to_str().unwrap(),
            ])
            .status()
            .expect("cargo run exporter")
    };
    assert!(status.success(), "rust exporter failed");

    // Run verifier with safetensors_dir
    let status = if let Some(verify) = bin("gptoss_verify") {
        Command::new(&verify)
            .args(["-s", out.to_str().unwrap(), rust_bin.to_str().unwrap()])
            .status()
            .expect("spawn verify")
    } else {
        Command::new("cargo")
            .args([
                "run",
                "-q",
                "-p",
                "burn-extended",
                "--bin",
                "gptoss_verify",
                "--",
                "-s",
                out.to_str().unwrap(),
                rust_bin.to_str().unwrap(),
            ])
            .status()
            .expect("cargo run verify")
    };
    assert!(status.success(), "verifier failed on exported model.bin");
}

