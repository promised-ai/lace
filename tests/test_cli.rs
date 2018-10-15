extern crate braid;
extern crate serde_yaml;
extern crate tempfile;

use std::fs;
use std::path::Path;
use std::process::Command;

#[cfg(test)]
mod tests {
    use super::*;
    use Command;

    const ANIMALS_CSV: &str = "resources/datasets/animals/animals.csv";
    const BRAID_CMD: &str = "./target/debug/braid";

    mod run {
        use super::*;

        #[test]
        fn with_default_args() {
            let dir = tempfile::TempDir::new();
            let output = Command::new(BRAID_CMD)
                .arg("run")
                .arg("--csv")
                .arg(ANIMALS_CSV)
                .output()
                .expect("failed to execute process");

            assert!(!output.status.success());
        }

    }

    mod codebook {
        use super::*;
        use braid::cc::Codebook;
        use std::io::Read;

        fn load_codebook(filename: &str) -> Codebook {
            let path = Path::new(&filename);
            let mut file = fs::File::open(&path).unwrap();
            let mut ser = String::new();
            file.read_to_string(&mut ser).unwrap();
            serde_yaml::from_str(&ser.as_str()).unwrap()
        }

        #[test]
        fn with_invalid_csv() {
            let fileout = tempfile::NamedTempFile::new().unwrap();
            let output = Command::new(BRAID_CMD)
                .arg("codebook")
                .arg("tortoise-cannot-swim.csv") // this doesn't exist
                .arg(fileout.path().to_str().unwrap())
                .output()
                .expect("failed to execute process");

            assert!(!output.status.success());
            assert!(
                String::from_utf8_lossy(&output.stderr)
                    .contains("swim.csv not found")
            );
        }

        #[test]
        fn with_default_args() {
            let fileout = tempfile::NamedTempFile::new().unwrap();
            let output = Command::new(BRAID_CMD)
                .arg("codebook")
                .arg(ANIMALS_CSV)
                .arg(fileout.path().to_str().unwrap())
                .output()
                .expect("failed to execute process");

            assert!(output.status.success());
            assert!(
                String::from_utf8_lossy(&output.stdout).contains("Wrote file")
            );
        }

        #[test]
        fn with_good_alpha_params() {
            let fileout = tempfile::NamedTempFile::new().unwrap();
            let output = Command::new(BRAID_CMD)
                .arg("codebook")
                .arg(ANIMALS_CSV)
                .arg(fileout.path().to_str().unwrap())
                .arg("--alpha-params")
                .arg("(2.3, 1.1)")
                .output()
                .expect("failed to execute process");

            assert!(output.status.success());

            let codebook = load_codebook(fileout.path().to_str().unwrap());

            let gamma_state = codebook.state_alpha_prior.unwrap();
            assert_eq!(gamma_state.shape, 2.3);
            assert_eq!(gamma_state.rate, 1.1);

            let gamma_view = codebook.view_alpha_prior.unwrap();
            assert_eq!(gamma_view.shape, 2.3);
            assert_eq!(gamma_view.rate, 1.1);
        }

        #[test]
        fn with_bad_alpha_params() {
            let fileout = tempfile::NamedTempFile::new().unwrap();
            let output = Command::new(BRAID_CMD)
                .arg("codebook")
                .arg(ANIMALS_CSV)
                .arg(fileout.path().to_str().unwrap())
                .arg("--alpha-params")
                .arg("[2.3, .1]")
                .output()
                .expect("failed to execute process");

            assert!(!output.status.success());
        }
    }
}
