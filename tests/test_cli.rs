use std::fs;
use std::path::Path;
use std::process::Command;

#[cfg(test)]
mod tests {
    use super::*;

    use crate::Command;
    use std::io;
    use std::process::Output;

    const ANIMALS_CSV: &str = "resources/datasets/animals/animals.csv";
    const ANIMALS_CODEBOOK: &str =
        "resources/datasets/animals/animals.codebook.yaml";
    const BRAID_CMD: &str = "./target/debug/braid";

    mod bench {
        use super::*;

        #[test]
        fn short_animals_run() {
            let output = Command::new(BRAID_CMD)
                .arg("bench")
                .args(&["--n-runs", "2", "--n-iters", "5"])
                .arg(ANIMALS_CODEBOOK)
                .arg(ANIMALS_CSV)
                .output()
                .expect("Failed to execute becnhmark");

            assert!(output.status.success());

            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(stdout.contains("time_sec"));
            assert!(stdout.contains("score"));
        }

        #[test]
        fn short_animals_run_with_slice_row_alg() {
            let output = Command::new(BRAID_CMD)
                .arg("bench")
                .args(&["--n-runs", "2", "--n-iters", "5"])
                .arg("--row-alg")
                .arg("slice")
                .arg(ANIMALS_CODEBOOK)
                .arg(ANIMALS_CSV)
                .output()
                .expect("Failed to execute becnhmark");

            assert!(output.status.success());

            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(stdout.contains("time_sec"));
            assert!(stdout.contains("score"));
        }

        #[test]
        fn short_animals_run_with_finite_cpu_row_alg() {
            let output = Command::new(BRAID_CMD)
                .arg("bench")
                .args(&["--n-runs", "2", "--n-iters", "5"])
                .arg("--row-alg")
                .arg("finite_cpu")
                .arg(ANIMALS_CODEBOOK)
                .arg(ANIMALS_CSV)
                .output()
                .expect("Failed to execute becnhmark");

            assert!(output.status.success());

            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(stdout.contains("time_sec"));
            assert!(stdout.contains("score"));
        }

        #[test]
        fn short_animals_run_with_gibbs_row_alg() {
            let output = Command::new(BRAID_CMD)
                .arg("bench")
                .args(&["--n-runs", "2", "--n-iters", "5"])
                .arg("--row-alg")
                .arg("gibbs")
                .arg(ANIMALS_CODEBOOK)
                .arg(ANIMALS_CSV)
                .output()
                .expect("Failed to execute becnhmark");

            assert!(output.status.success());

            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(stdout.contains("time_sec"));
            assert!(stdout.contains("score"));
        }

        #[test]
        fn no_csv_file_exists() {
            let output = Command::new(BRAID_CMD)
                .arg("bench")
                .args(&["--n-runs", "2", "--n-iters", "5"])
                .arg("--row-alg")
                .arg("gibbs")
                .arg(ANIMALS_CODEBOOK)
                .arg("should-not-exist.csv")
                .output()
                .expect("Failed to execute becnhmark");

            assert!(!output.status.success());

            let stderr = String::from_utf8_lossy(&output.stderr);
            println!("{}", stderr);
            assert!(
                stderr.contains("Could not read csv \"should-not-exist.csv\"")
            );
            assert!(stderr.contains("No such file or directory"));
        }

    }

    mod run {
        use super::*;

        fn create_animals_braidfile(dirname: &str) -> io::Result<Output> {
            Command::new(BRAID_CMD)
                .arg("run")
                .args(&["--n-states", "4", "--n-iters", "3"])
                .arg("--csv")
                .arg(ANIMALS_CSV)
                .arg(dirname)
                .output()
        }

        #[test]
        fn from_csv_with_default_args() {
            let dir = tempfile::TempDir::new().unwrap();
            let output = Command::new(BRAID_CMD)
                .arg("run")
                .args(&["--n-states", "4", "--n-iters", "3"])
                .arg("--csv")
                .arg(ANIMALS_CSV)
                .arg(dir.path().to_str().unwrap())
                .output()
                .expect("failed to execute process");

            assert!(output.status.success());
        }

        #[test]
        fn from_engine_with_default_args() {
            let dir = tempfile::TempDir::new().unwrap();
            let dirname = dir.path().to_str().unwrap();

            // first, create bradifile from a CSV
            let cmd_output = create_animals_braidfile(&dirname).unwrap();
            assert!(cmd_output.status.success());

            let output = Command::new(BRAID_CMD)
                .arg("run")
                .arg("--engine")
                .arg(dirname)
                .arg(dirname)
                .output()
                .expect("failed to execute process");

            println!("{:?}", output);
            assert!(output.status.success());
        }

        #[test]
        fn with_invalid_row_alg() {
            let dir = tempfile::TempDir::new().unwrap();
            let output = Command::new(BRAID_CMD)
                .arg("run")
                .args(&["--n-states", "4", "--n-iters", "3"])
                .arg("--csv")
                .arg(ANIMALS_CSV)
                .arg("--row-alg")
                .arg("row_magic")
                .arg(dir.path().to_str().unwrap())
                .output()
                .expect("failed to execute process");

            assert!(!output.status.success());
            assert!(String::from_utf8_lossy(&output.stderr)
                .contains("'row_magic' isn't a valid value for '--row-alg"));
        }

        #[test]
        fn with_invalid_col_alg() {
            let dir = tempfile::TempDir::new().unwrap();
            let output = Command::new(BRAID_CMD)
                .arg("run")
                .args(&["--n-states", "4", "--n-iters", "3"])
                .arg("--csv")
                .arg(ANIMALS_CSV)
                .arg("--col-alg")
                .arg("shovel")
                .arg(dir.path().to_str().unwrap())
                .output()
                .expect("failed to execute process");

            assert!(!output.status.success());
            assert!(String::from_utf8_lossy(&output.stderr)
                .contains("'shovel' isn't a valid value for '--col-alg"));
        }

        #[test]
        fn csv_and_sqlite_args_conflict() {
            let dir = tempfile::TempDir::new().unwrap();
            let output = Command::new(BRAID_CMD)
                .arg("run")
                .args(&["--n-states", "4", "--n-iters", "3"])
                .arg("--csv")
                .arg(ANIMALS_CSV)
                .arg("--sqlite")
                .arg("should-no-use.sqlite")
                .arg(dir.path().to_str().unwrap())
                .output()
                .expect("failed to execute process");

            assert!(!output.status.success());

            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(stderr.contains("cannot be used with"));
            assert!(stderr.contains("'--csv <csv_src>'"));
            assert!(stderr.contains("'--sqlite <sqlite_src>"));
        }

        #[test]
        fn csv_and_engine_args_conflict() {
            let dir = tempfile::TempDir::new().unwrap();
            let output = Command::new(BRAID_CMD)
                .arg("run")
                .args(&["--n-states", "4", "--n-iters", "3"])
                .arg("--csv")
                .arg(ANIMALS_CSV)
                .arg("--engine")
                .arg("should-no-use.braid")
                .arg(dir.path().to_str().unwrap())
                .output()
                .expect("failed to execute process");

            assert!(!output.status.success());

            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(stderr.contains("cannot be used with"));
            assert!(stderr.contains("'--csv <csv_src>'"));
            assert!(stderr.contains("'--engine <engine>"));
        }

        #[test]
        fn sqlite_and_engine_args_conflict() {
            let dir = tempfile::TempDir::new().unwrap();
            let output = Command::new(BRAID_CMD)
                .arg("run")
                .args(&["--n-states", "4", "--n-iters", "3"])
                .arg("--sqlite")
                .arg("should-no-use.sqlite")
                .arg("--engine")
                .arg("should-no-use.braid")
                .arg(dir.path().to_str().unwrap())
                .output()
                .expect("failed to execute process");

            assert!(!output.status.success());

            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(stderr.contains("cannot be used with"));
            assert!(stderr.contains("'--sqlite <sqlite_src>'"));
            assert!(stderr.contains("'--engine <engine>"));
        }
    }

    mod codebook {
        use super::*;
        use braid_codebook::codebook::Codebook;
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
            assert!(String::from_utf8_lossy(&output.stderr)
                .contains("swim.csv\" not found"));
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
