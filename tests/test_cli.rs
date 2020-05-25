use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;

#[cfg(test)]
mod tests {
    use super::*;

    use crate::Command;
    use braid_codebook::ColType;
    use braid_stats::prior::CrpPrior;
    use std::{io, process::Output};

    const ANIMALS_CSV: &str = "resources/datasets/animals/data.csv";
    const ANIMALS_CODEBOOK: &str = "resources/datasets/animals/codebook.yaml";
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

            println!("{}", String::from_utf8(output.stderr).unwrap());
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
            assert!(
                stderr.contains("Could not read csv \"should-not-exist.csv\"")
            );
            assert!(stderr.contains("No such file or directory"));
        }
    }

    mod run {
        use super::*;
        use indoc::indoc;

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

            // first, create braidfile from a CSV
            let cmd_output = create_animals_braidfile(&dirname).unwrap();
            assert!(cmd_output.status.success());

            let output = Command::new(BRAID_CMD)
                .arg("run")
                .arg("--engine")
                .arg(dirname)
                .arg(dirname)
                .args(&["--n-iters", "4"])
                .output()
                .expect("failed to execute process");

            assert!(output.status.success());
        }

        fn run_config_file() -> tempfile::NamedTempFile {
            let config = indoc!(
                "
                n_iters: 4
                timeout: 60
                save_path: ~
                transitions:
                  - row_assignment: slice
                  - view_alphas
                  - column_assignment: finite_cpu
                  - state_alpha
                  - feature_priors
                "
            );
            let mut f = tempfile::NamedTempFile::new().unwrap();
            f.write(config.as_bytes()).unwrap();
            f
        }

        #[test]
        fn from_engine_with_file_config() {
            let dir = tempfile::TempDir::new().unwrap();
            let dirname = dir.path().to_str().unwrap();

            // first, create braidfile from a CSV
            let cmd_output = create_animals_braidfile(&dirname).unwrap();
            assert!(cmd_output.status.success());

            let config = run_config_file();

            let output = Command::new(BRAID_CMD)
                .arg("run")
                .arg("--engine")
                .arg(dirname)
                .arg("--run-config")
                .arg(config.path())
                .arg(dirname)
                .output()
                .expect("failed to execute process");

            assert!(output.status.success());
        }

        #[test]
        fn from_engine_with_file_config_conflicts_with_n_iters() {
            let dir = tempfile::TempDir::new().unwrap();
            let dirname = dir.path().to_str().unwrap();

            // first, create braidfile from a CSV
            let cmd_output = create_animals_braidfile(&dirname).unwrap();
            assert!(cmd_output.status.success());

            let config = run_config_file();

            let output = Command::new(BRAID_CMD)
                .arg("run")
                .arg("--engine")
                .arg(dirname)
                .arg("--run-config")
                .arg(config.path())
                .arg("--n-iters")
                .arg("31")
                .arg(dirname)
                .output()
                .expect("failed to execute process");

            assert!(!output.status.success());
            assert!(String::from_utf8(output.stderr)
                .unwrap()
                .contains("cannot be used with"));
        }

        #[test]
        fn from_engine_with_file_config_conflicts_with_row_alg() {
            let dir = tempfile::TempDir::new().unwrap();
            let dirname = dir.path().to_str().unwrap();

            // first, create braidfile from a CSV
            let cmd_output = create_animals_braidfile(&dirname).unwrap();
            assert!(cmd_output.status.success());

            let config = run_config_file();

            let output = Command::new(BRAID_CMD)
                .arg("run")
                .arg("--engine")
                .arg(dirname)
                .arg("--run-config")
                .arg(config.path())
                .arg("--row-alg")
                .arg("slice")
                .arg(dirname)
                .output()
                .expect("failed to execute process");

            assert!(!output.status.success());
            assert!(String::from_utf8(output.stderr)
                .unwrap()
                .contains("cannot be used with"));
        }

        #[test]
        fn from_engine_with_file_config_conflicts_with_col_alg() {
            let dir = tempfile::TempDir::new().unwrap();
            let dirname = dir.path().to_str().unwrap();

            // first, create braidfile from a CSV
            let cmd_output = create_animals_braidfile(&dirname).unwrap();
            assert!(cmd_output.status.success());

            let config = run_config_file();

            let output = Command::new(BRAID_CMD)
                .arg("run")
                .arg("--engine")
                .arg(dirname)
                .arg("--run-config")
                .arg(config.path())
                .arg("--col-alg")
                .arg("slice")
                .arg(dirname)
                .output()
                .expect("failed to execute process");

            assert!(!output.status.success());
            assert!(String::from_utf8(output.stderr)
                .unwrap()
                .contains("cannot be used with"));
        }

        #[test]
        fn from_engine_with_file_config_conflicts_with_transitions() {
            let dir = tempfile::TempDir::new().unwrap();
            let dirname = dir.path().to_str().unwrap();

            // first, create braidfile from a CSV
            let cmd_output = create_animals_braidfile(&dirname).unwrap();
            assert!(cmd_output.status.success());

            let config = run_config_file();

            let output = Command::new(BRAID_CMD)
                .arg("run")
                .arg("--engine")
                .arg(dirname)
                .arg("--run-config")
                .arg(config.path())
                .arg("--transitions")
                .arg("state_alpha,row_assignment")
                .arg(dirname)
                .output()
                .expect("failed to execute process");

            assert!(!output.status.success());
            assert!(String::from_utf8(output.stderr)
                .unwrap()
                .contains("cannot be used with"));
        }

        fn get_n_iters(summary: String) -> Vec<usize> {
            summary
                .split("\n")
                .skip(1)
                .take_while(|&row| row != "")
                .map(|row| {
                    println!("'{}'", row);
                    let n = row.split_whitespace().nth(1).unwrap();
                    n.parse::<usize>().unwrap()
                })
                .collect()
        }

        #[test]
        fn add_iterations_to_engine() {
            let dir = tempfile::TempDir::new().unwrap();
            let dirname = dir.path().to_str().unwrap();

            // Runs 4 states w/ 100 existing iterations for 3 more iterations
            let cmd_output = create_animals_braidfile(&dirname).unwrap();
            assert!(cmd_output.status.success());

            {
                let output = Command::new(BRAID_CMD)
                    .arg("run")
                    .arg("--engine")
                    .arg(dirname)
                    .arg(dirname)
                    .output()
                    .expect("failed to execute process");

                assert!(output.status.success());
            }

            {
                let output = Command::new(BRAID_CMD)
                    .arg("summarize")
                    .arg(dirname)
                    .output()
                    .expect("failed to execute process");

                assert!(output.status.success());
                let summary =
                    String::from_utf8_lossy(&output.stdout).to_string();
                get_n_iters(summary)
                    .iter()
                    .for_each(|&n| assert_eq!(n, 103))
            }
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
            assert!(stderr.contains("'--csv <csv-src>'"));
            assert!(stderr.contains("'--engine <engine>"));
        }
    }

    mod codebook {
        use super::*;
        use braid_codebook::Codebook;
        use std::io::Read;

        fn load_codebook(filename: &str) -> Codebook {
            let path = Path::new(&filename);
            let mut file = fs::File::open(&path).unwrap();
            let mut ser = String::new();
            file.read_to_string(&mut ser).unwrap();
            serde_yaml::from_str(&ser.as_str())
                .map_err(|err| {
                    eprintln!("Error with {:?}: {:?}", path, err);
                    eprintln!("{}", ser);
                })
                .unwrap()
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
                .arg("Gamma(2.3, 1.1)")
                .output()
                .expect("failed to execute process");

            assert!(output.status.success());

            let codebook = load_codebook(fileout.path().to_str().unwrap());

            if let Some(CrpPrior::Gamma(gamma)) = codebook.state_alpha_prior {
                assert_eq!(gamma.shape(), 2.3);
                assert_eq!(gamma.rate(), 1.1);
            } else {
                panic!("No state_alpha_prior");
            }

            if let Some(CrpPrior::Gamma(gamma)) = codebook.view_alpha_prior {
                assert_eq!(gamma.shape(), 2.3);
                assert_eq!(gamma.rate(), 1.1);
            } else {
                panic!("No view_alpha_prior");
            }
        }

        #[test]
        fn with_bad_alpha_params() {
            let fileout = tempfile::NamedTempFile::new().unwrap();
            let output = Command::new(BRAID_CMD)
                .arg("codebook")
                .arg(ANIMALS_CSV)
                .arg(fileout.path().to_str().unwrap())
                .arg("--alpha-params")
                .arg("(2.3, .1)")
                .output()
                .expect("failed to execute process");

            assert!(!output.status.success());
        }

        #[test]
        fn uint_data_with_category_cutoff_becomes_count() -> std::io::Result<()>
        {
            let fileout = tempfile::NamedTempFile::new().unwrap();
            let mut data_file = tempfile::NamedTempFile::new().unwrap();

            // Write CSV with 21 distinct integer values
            let f = data_file.as_file_mut();
            writeln!(f, "ID,data")?;
            for i in 0..100 {
                writeln!(f, "{},{}", i, i % 21)?;
            }

            fn get_col_type(
                file_out: &tempfile::NamedTempFile,
            ) -> Option<ColType> {
                let codebook = Codebook::from_yaml(file_out.path())
                    .expect("Failed to read output codebook");
                let (_, metadata) =
                    codebook.col_metadata.get(&String::from("data"))?;
                let coltype = metadata.coltype.clone();
                Some(coltype)
            }

            // Default categorical cutoff should be 20
            let output_default = Command::new(BRAID_CMD)
                .arg("codebook")
                .arg(data_file.path().to_str().unwrap())
                .arg(fileout.path().to_str().unwrap())
                .output()
                .expect("Failed to execute process");

            assert!(output_default.status.success());

            let col_type = get_col_type(&fileout);
            match col_type {
                Some(ColType::Count { .. }) => {}
                _ => {
                    panic!("Expected Count ColType, got {:?}", col_type);
                }
            }

            // Set the value to 25 and confirm it labed the column to Categorical
            let fileout = tempfile::NamedTempFile::new().unwrap();
            let output = Command::new(BRAID_CMD)
                .arg("codebook")
                .args(&["-c", "25"])
                .arg(data_file.path().to_str().unwrap())
                .arg(fileout.path().to_str().unwrap())
                .output()
                .expect("Failed to execute process");
            assert!(output.status.success());

            let col_type = get_col_type(&fileout);
            match col_type {
                Some(ColType::Categorical {
                    k: _,
                    hyper: _,
                    value_map: _,
                }) => {}
                _ => {
                    panic!("Expected Categorical ColType, got {:?}", col_type);
                }
            }

            // Explicitly set the categorical cutoff below given distinct value count
            let fileout = tempfile::NamedTempFile::new().unwrap();
            let output = Command::new(BRAID_CMD)
                .arg("codebook")
                .args(&["-c", "15"])
                .arg(data_file.path().to_str().unwrap())
                .arg(fileout.path().to_str().unwrap())
                .output()
                .expect("Failed to execute process");
            assert!(output.status.success());

            let col_type = get_col_type(&fileout);
            match col_type {
                Some(ColType::Count { .. }) => {}
                _ => {
                    panic!("Expected Continuous ColType, got {:?}", col_type);
                }
            }

            Ok(())
        }

        #[test]
        fn heurstic_warnings() -> std::io::Result<()> {
            let fileout = tempfile::NamedTempFile::new().unwrap();
            let mut data_file = tempfile::NamedTempFile::new().unwrap();

            // Write CSV with two data_columns, one with 15% missing values
            // and a second with only one value
            let f = data_file.as_file_mut();
            writeln!(f, "ID,data_a,data_b")?;
            for i in 0..15 {
                writeln!(f, "{},,SINGLE_VALUE", i)?;
            }
            for i in 15..=100 {
                writeln!(f, "{},{},SINGLE_VALUE", i, i)?;
            }
            let output = Command::new(BRAID_CMD)
                .arg("codebook")
                .arg(data_file.path().to_str().unwrap())
                .arg(fileout.path().to_str().unwrap())
                .output()
                .expect("Failed to execute process");
            assert!(output.status.success());
            let stderr = String::from_utf8(output.stderr).unwrap();
            assert!(stderr.contains("WARNING: Column \"data_a\" is missing"));
            assert!(stderr.contains(
                "WARNING: Column \"data_b\" only takes on one value"
            ));

            Ok(())
        }
    }
}
