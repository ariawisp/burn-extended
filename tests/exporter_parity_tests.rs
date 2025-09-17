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
fn exporter_runs_on_fixture_and_optionally_matches_python() {
    // Prepare paths
    let out = tmp_dir("gptoss_fixture");
    fs::create_dir_all(&out).unwrap();
    let rust_bin = out.join("model_rust.bin");
    let py_bin = out.join("model_python.bin");

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
    assert!(out.join("config.json").exists());
    assert!(out.join("model.safetensors").exists());

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
    let rust_len = fs::metadata(&rust_bin).unwrap().len();
    assert!(rust_len > 0, "rust artifact empty");

    // If WITH_PY env is set, run Python exporter and compare bytes
    if env::var("WITH_PY").is_ok() {
        // Resolve script: ../gpt-oss/gpt_oss/metal/scripts/create-local-model.py from cwd
        let mut script = env::current_dir().unwrap();
        script.push("../gpt-oss/gpt_oss/metal/scripts/create-local-model.py");
        assert!(script.exists(), "python exporter script not found: {:?}", script);

        let status = Command::new("python")
            .args([
                script.to_str().unwrap(),
                "-s",
                out.to_str().unwrap(),
                "-d",
                py_bin.to_str().unwrap(),
            ])
            .status()
            .expect("spawn python exporter");
        assert!(status.success(), "python exporter failed");
        let py_len = fs::metadata(&py_bin).unwrap().len();
        assert!(py_len > 0, "python artifact empty");

        let a = fs::read(&rust_bin).unwrap();
        let b = fs::read(&py_bin).unwrap();
        assert_eq!(a, b, "rust vs python model.bin mismatch");
    }
}
