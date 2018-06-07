extern crate braid;
extern crate rand;

use self::braid::cc::codebook::ColMetadata;
use self::braid::cc::state::ColAssignAlg;
use self::braid::cc::view::RowAssignAlg;
use self::braid::data::StateBuilder;
use self::braid::interface::bencher::BencherResult;
use self::braid::interface::Bencher;
use self::rand::Rng;

#[derive(Serialize)]
pub struct BenchmarkResult {
    ncats: usize,
    nviews: usize,
    nrows: usize,
    ncols: usize,
    row_assign_alg: RowAssignAlg,
    col_assign_alg: ColAssignAlg,
    result: Vec<BencherResult>,
}

fn run_bench<R: Rng>(
    n_runs: usize,
    ncats: usize,
    nviews: usize,
    nrows: usize,
    ncols: usize,
    row_assign_alg: RowAssignAlg,
    col_assign_alg: ColAssignAlg,
    mut rng: &mut R,
) -> BenchmarkResult {
    // println!("Running k: {}, v: {}, r: {}, c: {}", ncats, nviews, nrows, ncols);
    let state_builder = StateBuilder::new()
        .with_cats(ncats)
        .with_views(nviews)
        .with_rows(nrows)
        .add_columns(ncols, ColMetadata::Continuous { hyper: None });

    let bencher = Bencher::from_builder(state_builder)
        .with_n_iters(1)
        .with_n_runs(n_runs)
        .with_col_assign_alg(col_assign_alg)
        .with_row_assign_alg(row_assign_alg);

    BenchmarkResult {
        ncats: ncats,
        nviews: nviews,
        ncols: ncols,
        nrows: nrows,
        row_assign_alg: row_assign_alg,
        col_assign_alg: col_assign_alg,
        result: bencher.run(&mut rng),
    }
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
) -> Vec<BenchmarkResult> {
    let prod = iproduct!(
        config.ncats_list.iter(),
        config.nviews_list.iter(),
        config.nrows_list.iter(),
        config.ncols_list.iter(),
        config.row_algs_list.iter(),
        config.col_algs_list.iter()
    );

    let mut results: Vec<BenchmarkResult> = Vec::new();
    for (cats, views, rows, cols, row_alg, col_alg) in prod {
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
        results.push(res)
    }
    results
}
