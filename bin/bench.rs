use braid::bencher::Bencher;
use braid_cc::alg::{ColAssignAlg, RowAssignAlg};
use braid_cc::state::StateBuilder;

use braid_codebook::ColType;
use itertools::iproduct;
use log::info;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct BenchmarkResult {
    /// The number of categories
    ncats: Vec<usize>,
    /// The number of views
    nviews: Vec<usize>,
    /// The number of rows
    nrows: Vec<usize>,
    /// The number of columns
    ncols: Vec<usize>,
    /// The row reassignment algorithm
    row_asgn_alg: Vec<RowAssignAlg>,
    /// The column reassignment algorithm
    col_asgn_alg: Vec<ColAssignAlg>,
    /// Which repetition  of the run this is
    rep: Vec<usize>,
    /// The time in seconds the run took
    time_sec: Vec<f64>,
}

impl BenchmarkResult {
    fn new() -> Self {
        BenchmarkResult {
            ncats: Vec::new(),
            nviews: Vec::new(),
            nrows: Vec::new(),
            ncols: Vec::new(),
            row_asgn_alg: Vec::new(),
            col_asgn_alg: Vec::new(),
            rep: Vec::new(),
            time_sec: Vec::new(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_bench<R: Rng>(
    n_runs: usize,
    ncats: usize,
    nviews: usize,
    nrows: usize,
    ncols: usize,
    row_asgn_alg: RowAssignAlg,
    col_asgn_alg: ColAssignAlg,
    mut rng: &mut R,
) -> Vec<f64> {
    info!(
        "Running k: {}, v: {}, r: {}, c: {}, row_alg: {}, col_alg: {}",
        ncats, nviews, nrows, ncols, row_asgn_alg, col_asgn_alg
    );
    let state_builder = StateBuilder::new()
        .with_cats(ncats)
        .with_views(nviews)
        .with_rows(nrows)
        .add_column_configs(
            ncols,
            ColType::Continuous {
                hyper: None,
                prior: None,
            },
        );

    let mut bencher = Bencher::from_builder(state_builder)
        .with_n_iters(1)
        .with_n_runs(n_runs)
        .with_col_assign_alg(col_asgn_alg)
        .with_row_assign_alg(row_asgn_alg);

    let res = bencher.run(&mut rng);

    res.iter().map(|r| r.time_sec[0]).collect()
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BenchmarkRegressionConfig {
    #[serde(rename = "ncats")]
    pub ncats_list: Vec<usize>,
    #[serde(rename = "nviews")]
    pub nviews_list: Vec<usize>,
    #[serde(rename = "nrows")]
    pub nrows_list: Vec<usize>,
    #[serde(rename = "ncols")]
    pub ncols_list: Vec<usize>,
    #[serde(rename = "row_algs")]
    pub row_algs_list: Vec<RowAssignAlg>,
    #[serde(rename = "col_algs")]
    pub col_algs_list: Vec<ColAssignAlg>,
    pub n_runs: usize,
}

pub fn run_benches<R: Rng>(
    config: &BenchmarkRegressionConfig,
    mut rng: &mut R,
) -> BenchmarkResult {
    let prod = iproduct!(
        config.ncats_list.iter(),
        config.nviews_list.iter(),
        config.nrows_list.iter(),
        config.ncols_list.iter(),
        config.row_algs_list.iter(),
        config.col_algs_list.iter()
    );

    let mut results = BenchmarkResult::new();
    for (cats, views, rows, cols, row_alg, col_alg) in prod {
        if cols >= views && rows >= cats {
            let res = run_bench(
                config.n_runs,
                *cats,
                *views,
                *rows,
                *cols,
                *row_alg,
                *col_alg,
                &mut rng,
            );

            for (rep, time_sec) in res.iter().enumerate() {
                results.ncats.push(*cats);
                results.nviews.push(*views);
                results.nrows.push(*rows);
                results.ncols.push(*cols);
                results.row_asgn_alg.push(*row_alg);
                results.col_asgn_alg.push(*col_alg);
                results.rep.push(rep);
                results.time_sec.push(*time_sec);
            }
        }
    }
    results
}
