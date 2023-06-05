use std::process::Command;

fn main() {
    let status = Command::new("python")
        .arg("setup.py")
        .arg("build_ext")
        .arg("--inplace")
        .status()
        .expect("Failed to run setup.py");

    if !status.success() {
        panic!("setup.py failed with exit code {:?}", status.code());
    }
}
