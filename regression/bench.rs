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

pub fn run_benches<R: Rng>(mut rng: &mut R) -> Vec<BenchmarkResult> {
    let n_runs = 10;

    let ncats: [usize; 3] = [1, 5, 20];
    let nviews: [usize; 4] = [1, 2, 5, 10];
    let nrows: [usize; 3] = [100, 1_000, 10_000];
    let ncols: [usize; 3] = [10, 50, 100];
    let row_algs: [RowAssignAlg; 1] = [RowAssignAlg::FiniteCpu];
    let col_algs: [ColAssignAlg; 1] = [ColAssignAlg::FiniteCpu];

    let prod = iproduct!(
        ncats.iter(),
        nviews.iter(),
        nrows.iter(),
        ncols.iter(),
        row_algs.iter(),
        col_algs.iter()
    );

    let mut results: Vec<BenchmarkResult> = Vec::new();
    for (cats, views, rows, cols, row_alg, col_alg) in prod {
        let res = run_bench(
            n_runs, *cats, *views, *rows, *cols, *row_alg, *col_alg, &mut rng,
        );
        results.push(res)
    }
    results
}
