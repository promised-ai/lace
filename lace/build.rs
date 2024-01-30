use core::panic;
use std::path::{Path, PathBuf};

const DATASET_NAMES: [&str; 2] = ["animals", "satellites"];

fn copy_resources(
    dataset_name: &str,
    examples_dir: &Path,
    resources_dir: &Path,
) {
    let dataset_dir = examples_dir.join(dataset_name);
    if let Ok(()) = std::fs::create_dir_all(&dataset_dir) {
        std::fs::copy(
            resources_dir.join(dataset_name).join("data.csv"),
            dataset_dir.join("data.csv"),
        )
        .map_err(|err| format!("Failed to copy {dataset_name} data.csv: {err}"))
        .unwrap();

        std::fs::copy(
            resources_dir.join(dataset_name).join("codebook.yaml"),
            dataset_dir.join("codebook.yaml"),
        )
        .map_err(|err| format!("Failed to copy {dataset_name} codebook: {err}"))
        .unwrap();
    } else {
        panic!("Failed to create {:?}", dataset_dir);
    }
}

fn main() {
    // DOCS_RS indicates that you are building for the website `https://docs.rs`
    // CARGO_FEATURE_EXAMPLES indicates that you are building with the `exampes` feature set
    if std::env::var("DOCS_RS").is_err()
        && std::env::var("CARGO_FEATURE_EXAMPLES").is_ok()
    {
        // Copy Examples
        let examples_dir: PathBuf = dirs::data_dir()
            .map(|dir| dir.join("lace").join("examples"))
            .expect("Could not find data dir.");

        let resources_dir = Path::new("resources").join("datasets");

        std::fs::create_dir_all(&examples_dir)
            .expect("Could not create examples dir.");

        for dataset_name in DATASET_NAMES {
            copy_resources(
                dataset_name,
                examples_dir.as_path(),
                resources_dir.as_path(),
            )
        }
    }
}
