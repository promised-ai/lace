use clap::Parser;
use indicatif::ProgressBar;
use lace::prelude::*;
use lace::stats::rv::dist::Gaussian;
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
    let mut conditions = HashSet::new();
    conditions.insert(0);
    conditions.insert(1);

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
    let gauss = Gaussian::new(0.0, 0.01).unwrap();
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
            train_data = train_data.sample_n(n, false, false, None).unwrap();
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

        let codebook =
            Codebook::from_df(&data.train_data, None, None, false).unwrap();

        let mut engine = Engine::new(
            n_states,
            codebook,
            DataSource::Polars(data.train_data),
            0,
            rand_xoshiro::Xoshiro256Plus::from_entropy(),
        )
        .unwrap();

        let n_cols = engine.shape().1;

        let mut z_cols = Vec::with_capacity(n_latent_cols);
        for i in 0..n_latent_cols {
            let name = format!("z_{i}");
            engine
                .append_latent_column(name, LatentColumnType::Continuous)
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

    eprintln!("{:?}", z_cols);

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

        engine.set_latent_values(proposal);
        engine.update(update_config.clone(), ()).unwrap();
        pbar.inc(1);
    }
    pbar.finish();

    let given_nothing: Given<usize> = Given::Nothing;
    let n_rows = engine.shape().0;
    let mut pixels = {
        let rows = (0..n_rows)
            .map(|_| {
                let values = engine
                    .simulate(&z_cols, &given_nothing, 1, None, &mut rng)
                    .unwrap();
                let conditions = values[0]
                    .iter()
                    .zip(z_cols.iter())
                    .map(|(z, &ix)| (ix, z.clone()))
                    .collect::<Vec<(usize, Datum)>>();
                engine
                    .simulate(
                        &col_ixs,
                        &Given::Conditions(conditions),
                        1,
                        None,
                        &mut rng,
                    )
                    .unwrap()
                    .pop()
                    .unwrap()
            })
            .collect::<Vec<Vec<Datum>>>();
        let mut pixels = DataFrame::default();
        for col_ix in 0..n_cols {
            let col: Vec<f64> = rows
                .iter()
                .map(|row| row[col_ix].to_f64_opt().unwrap())
                .collect();
            let srs = Series::new(format!("px_{col_ix}").as_str(), col);
            pixels.with_column(srs).unwrap();
        }
        pixels
    };

    let mut file = std::fs::File::create(dst).unwrap();
    CsvWriter::new(&mut file).finish(&mut pixels).unwrap();
}
