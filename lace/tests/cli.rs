use approx::assert_relative_eq;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use lace::HasStates;
use lace_codebook::ColType;
use std::{io, process::Output};

fn animals_path() -> PathBuf {
    Path::new("resources").join("datasets").join("animals")
}

fn satellites_path() -> PathBuf {
    Path::new("resources").join("datasets").join("satellites")
}

macro_rules! path_fn {
    ($mod: ident, $ext: expr) => {
        mod $mod {
            use super::*;

            pub fn animals() -> String {
                animals_path()
                    .join(format!("data.{}", $ext))
                    .into_os_string()
                    .into_string()
                    .unwrap()
            }

            pub fn satellites() -> String {
                satellites_path()
                    .join(format!("data.{}", $ext))
                    .into_os_string()
                    .into_string()
                    .unwrap()
            }
        }
    };
}

fn animals_codebook_path() -> String {
    animals_path()
        .join("codebook.yaml")
        .into_os_string()
        .into_string()
        .unwrap()
}

path_fn!(csv, "csv");
path_fn!(csvgz, "csv.gz");
path_fn!(jsonl, "jsonl");
path_fn!(feather, "feather");
path_fn!(parquet, "parquet");

#[test]
fn test_paths() {
    assert_eq!(
        csv::animals(),
        String::from("resources/datasets/animals/data.csv")
    );
    assert_eq!(
        animals_codebook_path(),
        String::from("resources/datasets/animals/codebook.yaml")
    );
    assert_eq!(
        csvgz::animals(),
        String::from("resources/datasets/animals/data.csv.gz")
    );
    assert_eq!(
        jsonl::animals(),
        String::from("resources/datasets/animals/data.jsonl")
    );
    assert_eq!(
        feather::animals(),
        String::from("resources/datasets/animals/data.feather")
    );
    assert_eq!(
        parquet::animals(),
        String::from("resources/datasets/animals/data.parquet")
    );
}

const LACE_CMD: &str = "./target/debug/lace";

mod run {
    use super::*;
    use indoc::indoc;

    fn simple_csv() -> tempfile::NamedTempFile {
        let csv = indoc!(
            "
            id,x,y,z
            a,0.1,0.2,0.3
            b,0.3,0.1,0.2
            c,0.2,0.3,0.1
            d,0.5,1.2,0.5
        "
        );
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        f
    }

    fn simple_csv_codebook_good() -> tempfile::NamedTempFile {
        let codebook = indoc!(
            "
            ---
            table_name: my_data
            state_alpha_prior:
              !Gamma
                shape: 1.0
                rate: 1.0
            view_alpha_prior:
              !Gamma
                shape: 1.0
                rate: 1.0
            col_metadata:
              - name: x
                coltype:
                  !Continuous
                    hyper: ~
                    prior:
                      m: 0.0
                      k: 1.0
                      v: 1.0
                      s2: 1.0
                notes: ~
              - name: y
                coltype:
                  !Continuous
                    hyper: ~
                    prior:
                      m: 0.0
                      k: 1.0
                      v: 1.0
                      s2: 1.0
                notes: ~
              - name: z
                coltype:
                  !Continuous
                    hyper: ~
                    prior:
                      m: 0.0
                      k: 1.0
                      v: 1.0
                      s2: 1.0
                notes: ~
            comments: Auto-generated codebook
            row_names:
              - a
              - b
              - c
              - d
        "
        );
        let mut f =
            tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        f.write_all(codebook.as_bytes()).unwrap();
        f
    }

    fn simple_csv_codebook_cols_unordered() -> tempfile::NamedTempFile {
        let codebook = indoc!(
            "
            ---
            table_name: my_data
            state_alpha_prior:
              !Gamma
                shape: 1.0
                rate: 1.0
            view_alpha_prior:
              !Gamma
                shape: 1.0
                rate: 1.0
            col_metadata:
              - name: z
                coltype:
                  !Continuous
                    hyper: ~
                    prior:
                      m: 0.0
                      k: 1.0
                      v: 1.0
                      s2: 1.0
                notes: ~
              - name: x
                coltype:
                  !Continuous
                    hyper: ~
                    prior:
                      m: 0.0
                      k: 1.0
                      v: 1.0
                      s2: 1.0
                notes: ~
              - name: y
                coltype:
                  !Continuous
                    hyper: ~
                    prior:
                      m: 0.0
                      k: 1.0
                      v: 1.0
                      s2: 1.0
                notes: ~
            comments: Auto-generated codebook
            row_names:
              - a
              - b
              - c
              - d
        "
        );
        let mut f =
            tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        f.write_all(codebook.as_bytes()).unwrap();
        f
    }

    fn create_animals_lacefile_args(
        src_flag: &str,
        src: &str,
        dst: &str,
    ) -> io::Result<Output> {
        Command::new(LACE_CMD)
            .arg("run")
            .arg("-q")
            .arg(src_flag)
            .arg(src)
            .args(["--n-states", "4", "--n-iters", "3"])
            .arg("-f")
            .arg("bincode")
            .arg(dst)
            .output()
    }

    fn create_animals_lacefile(dst: &str) -> io::Result<Output> {
        create_animals_lacefile_args("--csv", csv::animals().as_str(), dst)
    }

    #[test]
    fn from_csv_smoke() {
        let outdir = tempfile::tempdir().unwrap();
        let output = create_animals_lacefile_args(
            "--csv",
            csv::animals().as_str(),
            outdir.path().to_str().unwrap(),
        )
        .unwrap();
        assert!(output.status.success());
    }

    #[test]
    fn from_csvgz_smoke() {
        let outdir = tempfile::tempdir().unwrap();
        let output = create_animals_lacefile_args(
            "--csv",
            csvgz::animals().as_str(),
            outdir.path().to_str().unwrap(),
        )
        .unwrap();
        println!("{}", String::from_utf8_lossy(output.stderr.as_slice()));
        assert!(output.status.success());
    }

    #[test]
    fn from_jsonl_smoke() {
        let outdir = tempfile::tempdir().unwrap();
        let output = create_animals_lacefile_args(
            "--json",
            jsonl::animals().as_str(),
            outdir.path().to_str().unwrap(),
        )
        .unwrap();
        println!("{}", String::from_utf8_lossy(output.stderr.as_slice()));
        assert!(output.status.success());
    }

    #[test]
    fn from_parquet_smoke() {
        let outdir = tempfile::tempdir().unwrap();
        let output = create_animals_lacefile_args(
            "--parquet",
            parquet::animals().as_str(),
            outdir.path().to_str().unwrap(),
        )
        .unwrap();
        println!("{}", String::from_utf8_lossy(output.stderr.as_slice()));
        assert!(output.status.success());
    }

    #[test]
    fn from_feather_smoke() {
        let outdir = tempfile::tempdir().unwrap();
        let output = create_animals_lacefile_args(
            "--ipc",
            feather::animals().as_str(),
            outdir.path().to_str().unwrap(),
        )
        .unwrap();
        println!("{}", String::from_utf8_lossy(output.stderr.as_slice()));
        assert!(output.status.success());
    }

    #[test]
    fn from_csv_with_good_codebook() {
        let csv = simple_csv();
        let good_codebook = simple_csv_codebook_good();
        let dir = tempfile::TempDir::new().unwrap();
        let output = Command::new(LACE_CMD)
            .arg("run")
            .arg("-q")
            .args(["--n-states", "4", "--n-iters", "3"])
            .arg("--codebook")
            .arg(good_codebook.path().to_str().unwrap())
            .arg("--csv")
            .arg(csv.path().to_str().unwrap())
            .arg(dir.path().to_str().unwrap())
            .output()
            .expect("failed to execute process");

        println!("{}", String::from_utf8_lossy(output.stderr.as_slice()));
        assert!(output.status.success());
    }

    #[test]
    fn from_csv_with_misordered_codebook() {
        let csv = simple_csv();
        let misordered_codebook = simple_csv_codebook_cols_unordered();
        let dir = tempfile::TempDir::new().unwrap();
        let output = Command::new(LACE_CMD)
            .arg("run")
            .arg("-q")
            .args(["--n-states", "4", "--n-iters", "3"])
            .arg("--codebook")
            .arg(misordered_codebook.path().to_str().unwrap())
            .arg("--csv")
            .arg(csv.path().to_str().unwrap())
            .arg(dir.path().to_str().unwrap())
            .output()
            .expect("failed to execute process");

        assert!(output.status.success());
    }

    #[test]
    fn from_csv_with_default_args() {
        let dir = tempfile::TempDir::new().unwrap();
        let output = Command::new(LACE_CMD)
            .arg("run")
            .arg("-q")
            .args(["--n-states", "4", "--n-iters", "3"])
            .arg("--csv")
            .arg(csv::animals())
            .arg(dir.path().to_str().unwrap())
            .output()
            .expect("failed to execute process");

        assert!(output.status.success());
    }

    #[test]
    fn from_gzip_csv_with_default_args() {
        let path = animals_path()
            .join("data.csv")
            .into_os_string()
            .into_string()
            .unwrap();

        let dir = tempfile::TempDir::new().unwrap();
        let output = Command::new(LACE_CMD)
            .arg("run")
            .arg("-q")
            .args(["--n-states", "4", "--n-iters", "3"])
            .arg("--csv")
            .arg(path)
            .arg(dir.path().to_str().unwrap())
            .output()
            .expect("failed to execute process");

        assert!(output.status.success());
    }

    #[test]
    fn from_engine_with_default_args() {
        let dir = tempfile::TempDir::new().unwrap();
        let dirname = dir.path().to_str().unwrap();

        // first, create lacefile from a CSV
        let cmd_output = create_animals_lacefile(dirname).unwrap();

        assert!(cmd_output.status.success());

        let output = Command::new(LACE_CMD)
            .arg("run")
            .arg("-q")
            .arg("--engine")
            .arg(dirname)
            .args(["--n-iters", "4"])
            .arg(dirname)
            .output()
            .expect("failed to execute process");

        assert!(output.status.success());
    }

    fn run_config_file() -> tempfile::NamedTempFile {
        let config = indoc!(
            "
            n_iters: 4
            timeout: 60
            save_config: ~
            transitions:
              - !row_assignment slice
              - !view_alphas
              - !column_assignment finite_cpu
              - !state_alpha
              - !feature_priors
            "
        );
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(config.as_bytes()).unwrap();
        f
    }

    #[test]
    fn from_engine_with_file_config() {
        let dir = tempfile::TempDir::new().unwrap();
        let dirname = dir.path().to_str().unwrap();

        // first, create lacefile from a CSV
        let cmd_output = create_animals_lacefile(dirname).unwrap();
        assert!(cmd_output.status.success());

        let config = run_config_file();

        let output = Command::new(LACE_CMD)
            .arg("run")
            .arg("-q")
            .arg("--engine")
            .arg(dirname)
            .arg("--run-config")
            .arg(config.path())
            .arg(dirname)
            .output()
            .expect("failed to execute process");

        println!("{}", String::from_utf8(output.stderr).unwrap());

        assert!(output.status.success());
    }

    #[test]
    fn from_engine_with_file_config_conflicts_with_n_iters() {
        let dir = tempfile::TempDir::new().unwrap();
        let dirname = dir.path().to_str().unwrap();

        // first, create lacefile from a CSV
        let cmd_output = create_animals_lacefile(dirname).unwrap();
        assert!(cmd_output.status.success());

        let config = run_config_file();

        let output = Command::new(LACE_CMD)
            .arg("run")
            .arg("-q")
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

        // first, create lacefile from a CSV
        let cmd_output = create_animals_lacefile(dirname).unwrap();
        assert!(cmd_output.status.success());

        let config = run_config_file();

        let output = Command::new(LACE_CMD)
            .arg("run")
            .arg("-q")
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

        // first, create lacefile from a CSV
        let cmd_output = create_animals_lacefile(dirname).unwrap();
        assert!(cmd_output.status.success());

        let config = run_config_file();

        let output = Command::new(LACE_CMD)
            .arg("run")
            .arg("-q")
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

        // first, create lacefile from a CSV
        let cmd_output = create_animals_lacefile(dirname).unwrap();
        assert!(cmd_output.status.success());

        let config = run_config_file();

        let output = Command::new(LACE_CMD)
            .arg("run")
            .arg("-q")
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
            .split('\n')
            .skip(2)
            .take_while(|&row| !row.is_empty())
            .map(|row| {
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
        let cmd_output = create_animals_lacefile(dirname).unwrap();
        assert!(cmd_output.status.success());

        {
            let output = Command::new(LACE_CMD)
                .arg("run")
                .arg("-q")
                .arg("--n-iters")
                .arg("100")
                .arg("--engine")
                .arg(dirname)
                .arg(dirname)
                .output()
                .expect("failed to execute process");

            assert!(output.status.success());
        }

        {
            let output = Command::new(LACE_CMD)
                .arg("summarize")
                .arg(dirname)
                .output()
                .expect("failed to execute process");

            assert!(output.status.success());
            let summary = String::from_utf8_lossy(&output.stdout).to_string();
            get_n_iters(summary)
                .iter()
                .for_each(|&n| assert_eq!(n, 103))
        }
    }

    #[test]
    fn with_invalid_row_alg() {
        let dir = tempfile::TempDir::new().unwrap();
        let output = Command::new(LACE_CMD)
            .arg("run")
            .arg("-q")
            .args(["--n-states", "4", "--n-iters", "3"])
            .arg("--csv")
            .arg(csv::animals())
            .arg("--row-alg")
            .arg("row_magic")
            .arg(dir.path().to_str().unwrap())
            .output()
            .expect("failed to execute process");

        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr)
            .contains("\"row_magic\" isn't a valid value for '--row-alg"));
    }

    #[test]
    fn with_invalid_col_alg() {
        let dir = tempfile::TempDir::new().unwrap();
        let output = Command::new(LACE_CMD)
            .arg("run")
            .arg("-q")
            .args(["--n-states", "4", "--n-iters", "3"])
            .arg("--csv")
            .arg(csv::animals())
            .arg("--col-alg")
            .arg("shovel")
            .arg(dir.path().to_str().unwrap())
            .output()
            .expect("failed to execute process");

        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("\"shovel\" isn't a valid value for '--col-alg")
        );
    }

    #[test]
    fn csv_and_engine_args_conflict() {
        let dir = tempfile::TempDir::new().unwrap();
        let output = Command::new(LACE_CMD)
            .arg("run")
            .arg("-q")
            .args(["--n-states", "4", "--n-iters", "3"])
            .arg("--csv")
            .arg(csv::animals())
            .arg("--engine")
            .arg("should-no-use.lace")
            .arg(dir.path().to_str().unwrap())
            .output()
            .expect("failed to execute process");

        assert!(!output.status.success());

        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("cannot be used with"));
        assert!(stderr.contains("'--csv <CSV_SRC>'"));
        assert!(stderr.contains("'--engine <ENGINE>"));
    }

    #[test]
    fn from_csv_with_id_offset_saves_offsets_corectly() {
        use std::collections::HashSet;

        let dir = tempfile::TempDir::new().unwrap();
        let output = Command::new(LACE_CMD)
            .arg("run")
            .arg("-q")
            .args(["--n-states", "4", "--n-iters", "3", "-o", "4"])
            .arg("--csv")
            .arg(csv::animals())
            .arg(dir.path().to_str().unwrap())
            .output()
            .expect("failed to execute process");

        assert!(output.status.success());

        let files: HashSet<PathBuf> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|d| {
                d.unwrap()
                    .path()
                    .strip_prefix(dir.path())
                    .unwrap()
                    .to_owned()
            })
            .collect();

        // 4 states, 4 diagnostics, data, config, rng, codebook
        assert_eq!(files.len(), 12);

        assert!(!files.contains(&PathBuf::from("0.state")));
        assert!(!files.contains(&PathBuf::from("1.state")));
        assert!(!files.contains(&PathBuf::from("2.state")));
        assert!(!files.contains(&PathBuf::from("3.state")));

        assert!(files.contains(&PathBuf::from("4.state")));
        assert!(files.contains(&PathBuf::from("5.state")));
        assert!(files.contains(&PathBuf::from("6.state")));
        assert!(files.contains(&PathBuf::from("7.state")));

        assert!(files.contains(&PathBuf::from("lace.codebook")));
        assert!(files.contains(&PathBuf::from("lace.data")));
        assert!(files.contains(&PathBuf::from("config.yaml")));
        assert!(files.contains(&PathBuf::from("rng.yaml")));
    }

    #[test]
    fn run_with_flat_columns_leaves_1_view() {
        use approx::assert_relative_eq;
        use lace::OracleT;

        let dir = tempfile::TempDir::new().unwrap();
        let output = Command::new(LACE_CMD)
            .arg("run")
            .arg("-q")
            .args(["--n-states", "4", "--n-iters", "10", "--flat-columns"])
            .arg("--transitions")
            .arg("state_alpha,view_alphas,component_params,row_assignment,feature_priors")
            .arg("--csv")
            .arg(csv::animals())
            .arg(dir.path().to_str().unwrap())
            .output()
            .expect("failed to execute process");

        assert!(output.status.success());

        println!(
            "STDOUT: {}",
            String::from_utf8_lossy(output.stdout.as_slice())
        );
        println!(
            "STDERR: {}",
            String::from_utf8_lossy(output.stderr.as_slice())
        );

        let engine = lace::Engine::load(dir.path()).unwrap();
        let n_cols = engine.n_cols();
        assert_eq!(n_cols, 85);
        assert_eq!(engine.n_rows(), 50);
        for col_a in 0..n_cols {
            for col_b in 0..n_cols {
                assert_relative_eq!(
                    engine.depprob(col_a, col_b).unwrap(),
                    1.0,
                    epsilon = 1E-10
                )
            }
        }
    }
}

macro_rules! test_codebook_under_fmt {
    ($mod: ident, $flag: expr) => {
        mod $mod {
            use super::*;
            use lace_codebook::Codebook;
            use std::io::Read;

            fn load_codebook(filename: &str) -> Codebook {
                let path = Path::new(&filename);
                let mut file = fs::File::open(path).unwrap();
                let mut ser = String::new();
                file.read_to_string(&mut ser).unwrap();
                serde_yaml::from_str(ser.as_str())
                    .map_err(|err| {
                        eprintln!("Error with {:?}: {:?}", path, err);
                        eprintln!("{}", ser);
                    })
                    .unwrap()
            }

            #[test]
            fn with_default_args() {
                let fileout = tempfile::Builder::new()
                    .suffix(".yaml")
                    .tempfile()
                    .unwrap();
                let output = Command::new(LACE_CMD)
                    .arg("codebook")
                    .arg($flag)
                    .arg($crate::$mod::animals())
                    .arg(fileout.path().to_str().unwrap())
                    .output()
                    .expect("failed to execute process");

                assert!(output.status.success());
                assert!(String::from_utf8_lossy(&output.stdout)
                    .contains("Wrote file"));
            }

            #[test]
            fn with_no_hyper_has_no_hyper() {
                let fileout = tempfile::Builder::new()
                    .suffix(".yaml")
                    .tempfile()
                    .unwrap();
                let output = Command::new(LACE_CMD)
                    .arg("codebook")
                    .arg($flag)
                    .arg($crate::$mod::satellites())
                    .arg(fileout.path().to_str().unwrap())
                    .arg("--no-hyper")
                    .output()
                    .expect("failed to execute process");

                println!(
                    "STDERR: {}",
                    String::from_utf8_lossy(output.stderr.as_slice())
                );
                assert!(output.status.success());

                let codebook = load_codebook(fileout.path().to_str().unwrap());
                let no_hypers =
                    codebook.col_metadata.iter().all(|md| match md.coltype {
                        ColType::Continuous {
                            hyper: None,
                            prior: Some(_),
                            ..
                        } => true,
                        ColType::Categorical {
                            hyper: None,
                            prior: Some(_),
                            ..
                        } => true,
                        _ => false,
                    });
                assert!(no_hypers);
            }

            #[test]
            fn with_bad_alpha_params() {
                let fileout = tempfile::Builder::new()
                    .suffix(".yaml")
                    .tempfile()
                    .unwrap();
                let output = Command::new(LACE_CMD)
                    .arg("codebook")
                    .arg($flag)
                    .arg($crate::$mod::animals())
                    .arg(fileout.path().to_str().unwrap())
                    .arg("--alpha-params")
                    .arg("(2.3, .1)")
                    .output()
                    .expect("failed to execute process");

                assert!(!output.status.success());
            }
        }
    };
}

mod codebook {
    use super::*;
    use lace_codebook::Codebook;

    test_codebook_under_fmt!(csv, "--csv");
    test_codebook_under_fmt!(csvgz, "--csv");
    test_codebook_under_fmt!(jsonl, "--json");
    test_codebook_under_fmt!(feather, "--ipc");
    test_codebook_under_fmt!(parquet, "--parquet");

    #[test]
    fn with_invalid_csv() {
        let fileout =
            tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        let output = Command::new(LACE_CMD)
            .arg("codebook")
            .arg("--csv")
            .arg("tortoise-cannot-swim.csv") // this doesn't exist
            .arg(fileout.path().to_str().unwrap())
            .output()
            .expect("failed to execute process");

        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr)
            .contains("swim.csv\" not found"));
    }

    #[test]
    fn uint_data_with_category_cutoff_becomes_count() -> std::io::Result<()> {
        let fileout =
            tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        let mut data_file = tempfile::NamedTempFile::new().unwrap();

        // Write CSV with 21 distinct integer values
        let f = data_file.as_file_mut();
        writeln!(f, "ID,data")?;
        for i in 0..100 {
            writeln!(f, "{},{}", i, i % 21)?;
        }

        fn get_col_type(file_out: &tempfile::NamedTempFile) -> Option<ColType> {
            let codebook = Codebook::from_yaml(file_out.path())
                .expect("Failed to read output codebook");
            let (_, metadata) =
                codebook.col_metadata.get(&String::from("data"))?;
            let coltype = metadata.coltype.clone();
            Some(coltype)
        }

        // Default categorical cutoff should be 20
        let output_default = Command::new(LACE_CMD)
            .arg("codebook")
            .arg("--csv")
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
        let fileout =
            tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        let output = Command::new(LACE_CMD)
            .arg("codebook")
            .args(["-c", "25"])
            .arg("--csv")
            .arg(data_file.path().to_str().unwrap())
            .arg(fileout.path().to_str().unwrap())
            .output()
            .expect("Failed to execute process");
        assert!(output.status.success());

        let col_type = get_col_type(&fileout);
        match col_type {
            Some(ColType::Categorical { .. }) => {}
            _ => {
                panic!("Expected Categorical ColType, got {:?}", col_type);
            }
        }

        // Explicitly set the categorical cutoff below given distinct value count
        let fileout =
            tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        let output = Command::new(LACE_CMD)
            .arg("codebook")
            .args(["-c", "15"])
            .arg("--csv")
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
    fn heuristic_warnings() -> std::io::Result<()> {
        let fileout =
            tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        let mut data_file = tempfile::NamedTempFile::new().unwrap();

        // Write CSV with two data_columns, one with 15% missing values
        // and a second with only one value
        let f = data_file.as_file_mut();
        writeln!(f, "ID,data_a,data_b")?;
        for i in 0..100 {
            writeln!(f, "{},,SINGLE_VALUE", i)?;
        }
        for i in 100..=150 {
            writeln!(f, "{},{},SINGLE_VALUE", i, i)?;
        }
        let output = Command::new(LACE_CMD)
            .arg("codebook")
            .arg("--csv")
            .arg(data_file.path().to_str().unwrap())
            .arg(fileout.path().to_str().unwrap())
            .output()
            .expect("Failed to execute process");
        assert!(!output.status.success());
        let stderr = String::from_utf8(output.stderr).unwrap();
        dbg!(&stderr);
        assert!(stderr
            .contains("Column `data_b` contains only a single unique value"));
        // assert!(stderr.contains("NOTE: Column \"data_a\" is missing"));

        Ok(())
    }
}
