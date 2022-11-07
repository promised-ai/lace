use std::path::{Path, PathBuf};

fn main() {
    // Copy Examples
    let examples_dir: PathBuf = dirs::data_dir()
        .map(|dir| dir.join("braid").join("examples"))
        .expect("Could not find data dir.");

    let resources_dir = Path::new("resources").join("datasets");

    std::fs::create_dir_all(&examples_dir)
        .expect("Could not create examples dir.");

    // ANIMALS
    {
        let animals_dir = examples_dir.join("animals");
        if let Ok(()) = std::fs::create_dir(&animals_dir) {
            std::fs::copy(
                resources_dir.join("animals").join("data.csv"),
                animals_dir.join("data.csv"),
            )
            .expect("Could not copy animals CSV.");

            std::fs::copy(
                resources_dir.join("animals").join("codebook.yaml"),
                animals_dir.join("codebook.yaml"),
            )
            .expect("Could not copy animals codebook.");
        }
    }

    // SATELLITES
    {
        let animals_dir = examples_dir.join("satellites");
        if let Ok(()) = std::fs::create_dir(&animals_dir) {
            std::fs::copy(
                resources_dir.join("satellites").join("data.csv"),
                animals_dir.join("data.csv"),
            )
            .expect("Could not copy satellites CSV.");

            std::fs::copy(
                resources_dir.join("satellites").join("codebook.yaml"),
                animals_dir.join("codebook.yaml"),
            )
            .expect("Could not copy satellites codebook.");
        }
    }
}
