use clap::Parser;
use indicatif::ProgressBar;
use lace::prelude::*;
use lace::stats::rv::dist::{Gaussian, NormalInvChiSquared};
use lace::stats::rv::traits::Rv;
use polars::prelude::*;
use rand::SeedableRng;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[clap(rename_all = "kebab")]
struct Opt {
    #[clap(long, default_value = "4")]
    pub n_states: usize,
    #[clap(short = 'c', long, default_value = "9")]
    pub n_latent_cols: usize,
    #[clap(long, default_value = "100")]
    pub n_sweeps: usize,
    #[clap(short, long)]
    pub subample: Option<usize>,
    #[clap(long)]
    pub src: Option<PathBuf>,
    pub dst: PathBuf,
}

fn state_transitions() -> Vec<StateTransition> {
    let conditions: HashSet<usize> = (0..28 * 28).collect();

    vec![
        StateTransition::StateAlpha,
        StateTransition::RowAssignment(RowAssignAlg::ConditionalSlice(
            conditions,
        )),
        StateTransition::ComponentParams,
        StateTransition::ViewAlphas,
        StateTransition::FeaturePriors,
    ]
}

struct MnistData {
    train_data: DataFrame,
    train_labels: Series,
    test_data: DataFrame,
    test_labels: Series,
}

fn to_float(srs: &Series) -> Series {
    srs.i64()
        .unwrap()
        .into_iter()
        .map(|x_opt: Option<i64>| x_opt.map(|x| x as f64))
        .collect::<Float64Chunked>()
        .into_series()
}

fn process_pixels(mut df: DataFrame) -> DataFrame {
    for column in df.get_column_names_owned() {
        let name = String::from(column);
        df.apply(name.as_str(), jitter).unwrap();
    }
    let n_rows = df.shape().0;
    let index = (0..n_rows)
        .map(|i| format!("img_{i}"))
        .collect::<Vec<String>>();
    let index_col = Series::new("Index", index);
    df.with_column(index_col).unwrap();
    df
}

fn jitter(srs: &Series) -> Series {
    let mut rng = rand::thread_rng();
    let gauss = Gaussian::new(0.0, 0.001).unwrap();
    srs.i64()
        .unwrap()
        .into_iter()
        .map(|x_opt: Option<i64>| {
            let u: f64 = gauss.draw(&mut rng);
            x_opt.map(|x| (x as f64 / 255.0) + u)
        })
        .collect::<Float64Chunked>()
        .into_series()
}

fn load_mnist<P: AsRef<Path>>(src: P, subsample: Option<usize>) -> MnistData {
    let train_path = src.as_ref().join("mnist_train.csv.gz");
    let test_path = src.as_ref().join("mnist_test.csv.gz");

    let (train_data, train_labels) = {
        let mut train_data = CsvReader::from_path(train_path)
            .unwrap()
            .has_header(true)
            .finish()
            .unwrap();

        if let Some(n) = subsample {
            train_data =
                train_data.sample_n_literal(n, false, false, None).unwrap();
        };

        let train_labels = train_data.drop_in_place("label").unwrap();
        let train_data = process_pixels(train_data);

        (train_data, train_labels)
    };

    let (test_data, test_labels) = {
        let mut test_data = CsvReader::from_path(test_path)
            .unwrap()
            .has_header(true)
            .finish()
            .unwrap();

        let test_labels = test_data.drop_in_place("label").unwrap();
        (test_data, test_labels)
    };

    MnistData {
        train_data,
        train_labels,
        test_data,
        test_labels,
    }
}

fn build_engine(df: DataFrame, n_states: usize) -> Engine {
    let codebook = {
        let mut codebook = Codebook::from_df(&df, None, None, false).unwrap();
        for col_ix in 0..codebook.n_cols() {
            if let ColType::Continuous {
                ref mut hyper,
                ref mut prior,
            } = codebook.col_metadata[0].coltype
            {
                *hyper = None;
                *prior = Some(
                    NormalInvChiSquared::new(0.0, 0.001, 1.0, 0.01).unwrap(),
                );
            } else {
                panic!("col_ix {col_ix} was supposed to be continuous");
            };
        }
        codebook
    };
    let rng = rand_xoshiro::Xoshiro256Plus::from_entropy();
    Engine::new(n_states, codebook, DataSource::Polars(df), 0, rng).unwrap()
}

fn main() {
    let Opt {
        n_states,
        n_latent_cols,
        n_sweeps,
        subample,
        src,
        dst,
    } = Opt::parse();

    let src = src
        .unwrap_or_else(|| PathBuf::from("..").join("datasets").join("mnist"));

    let (mut engine, z_cols, n_cols) = {
        let data = load_mnist(src, subample);
        eprintln!("Train data shape: {:?}", data.train_data.shape());

        let mut engine = build_engine(data.train_data, n_states);

        let n_cols = engine.shape().1;

        let mut z_cols = Vec::with_capacity(n_latent_cols);
        for i in 0..n_latent_cols {
            let name = format!("z_{i}");
            engine
                .append_latent_column(
                    name,
                    LatentColumnType::Continuous(
                        NormalInvChiSquared::new_unchecked(0.0, 0.1, 1.0, 0.1),
                    ),
                )
                .unwrap();
            z_cols.push(n_cols + i);
        }

        engine.flatten_cols();
        eprintln!("Engine shape: {:?}", engine.shape());

        (engine, z_cols, n_cols)
    };

    let mut rng = rand_xoshiro::Xoshiro256Plus::from_entropy();

    let update_config = EngineUpdateConfig {
        n_iters: 100,
        transitions: state_transitions(),
        checkpoint: None,
        save_config: None,
    };
    let scale_contraint = Gaussian::standard();

    let col_ixs: Vec<usize> = (0..n_cols).collect();

    let pbar = ProgressBar::new(n_sweeps as u64);

    for _ in 0..n_sweeps {
        let proposal = {
            let ln_constraint =
                |row_ix: usize, data: &HashMap<usize, Datum>| {
                    let values = (0..n_cols)
                        .map(|col_ix| {
                            // no missing values in data, so we shouldn't have to
                            // filter on Datum::Missing
                            engine.cell(row_ix, col_ix)
                        })
                        .collect::<Vec<Datum>>();

                    let mut ln_scale = 0.0;
                    let conditions = data
                        .iter()
                        .map(|(&col_ix, datum)| {
                            if let Datum::Continuous(z) = datum {
                                ln_scale += scale_contraint.ln_f(z);
                            } else {
                                panic!("All zs should be continuous");
                            }
                            (col_ix, datum.clone())
                        })
                        .collect::<Vec<(usize, Datum)>>();

                    engine
                        .logp(
                            &col_ixs,
                            &[values],
                            &Given::Conditions(conditions),
                            None,
                        )
                        .unwrap()[0]
                        + ln_scale
                };
            engine
                .propose_latent_values_with(
                    &z_cols,
                    &ln_constraint,
                    100,
                    &mut rng,
                )
                .unwrap()
        };

        engine.set_latent_values(proposal).unwrap();
        engine.update(&update_config, ()).unwrap();
        pbar.inc(1);
    }
    pbar.finish();

    engine.save(dst, SerializedType::Bincode).unwrap()
}
