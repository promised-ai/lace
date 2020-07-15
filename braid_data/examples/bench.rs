use braid_data::*;
use rand::Rng;

const MU: f64 = 2.0;
const SIGMA: f64 = 2.5;

#[derive(Clone)]
struct Parts<T> {
    /// The data values
    data: Vec<T>,
    /// Whether each datum is present (true) or missing (false)
    present: Vec<bool>,
    /// The total number of present data
    n_present: usize,
    /// The number of contiguous slices occupied by the present data
    n_slices: usize,
}

fn gen_parts<R, T, F>(
    n: usize,
    sparisty: f64,
    n_slices: usize,
    gen_fn: F,
    mut rng: &mut R,
) -> Parts<T>
where
    R: rand::Rng,
    T: Copy + Default,
    F: Fn(&mut R) -> T,
{
    assert!(0.0 <= sparisty && sparisty < 1.0);

    let n_present = (((n as f64) * (1.0 - sparisty)).trunc() + 0.5) as usize;

    let slice_size = n / n_slices;

    let mut markers_placed: usize = 0;
    let mut slice_ix = 0;
    let mut rem = 0;

    let mut present = vec![false; n];
    while markers_placed < n_present {
        present[rem + slice_size * slice_ix] = true;

        slice_ix += 1;
        if slice_ix == n_slices {
            rem += 1;
            slice_ix = 0;
        }
        markers_placed += 1;
    }

    let data: Vec<T> = (0..n).map(|_| gen_fn(&mut rng)).collect();

    Parts {
        data,
        present,
        n_slices,
        n_present,
    }
}

fn quick_parts(n: usize, sparisty: f64, n_slices: usize) -> Parts<f64> {
    let mut rng = rand::thread_rng();
    gen_parts(n, sparisty, n_slices, |r| r.gen::<f64>(), &mut rng)
}

fn containers_from_parts(
    parts: Parts<f64>,
) -> (DenseContainer<f64>, SparseContainer<f64>) {
    let c_dense = {
        let parts_clone = parts.clone();
        DenseContainer::new(parts_clone.data, parts_clone.present)
    };

    let c_sparse = {
        let parts_clone = parts.clone();
        SparseContainer::with_missing(parts_clone.data, &parts_clone.present)
    };

    (c_dense, c_sparse)
}

fn bench_dense(
    container: &DenseContainer<f64>,
    mut target: &mut Vec<f64>,
) -> f64 {
    use std::time::Instant;
    let t_start = Instant::now();

    let ln_sigma = SIGMA.ln();
    let score_fn = |x: &f64| {
        let term = (x - MU) / SIGMA;
        -ln_sigma - 0.5 * term * term
    };

    container.accum_score(&mut target, &score_fn);
    t_start.elapsed().as_secs_f64()
}

fn bench_sparse(
    container: &SparseContainer<f64>,
    mut target: &mut Vec<f64>,
) -> f64 {
    use std::time::Instant;
    let t_start = Instant::now();

    let ln_sigma = SIGMA.ln();
    let score_fn = |x: &f64| {
        let term = (x - MU) / SIGMA;
        -ln_sigma - 0.5 * term * term
    };

    container.accum_score(&mut target, &score_fn);
    t_start.elapsed().as_secs_f64()
}

#[derive(Clone, Debug)]
struct BenchResult {
    pub t_dense: f64,
    pub t_vec: f64,
}

#[derive(Clone, Debug)]
struct BenchStats {
    pub min: f64,
    pub mean: f64,
    pub max: f64,
    pub std: f64,
}

impl BenchStats {
    pub fn new(times: Vec<f64>) -> BenchStats {
        let nf = times.len() as f64;

        let min = *times
            .iter()
            .min_by(|a, b| a.partial_cmp(&b).unwrap())
            .unwrap();
        let max = *times
            .iter()
            .max_by(|a, b| a.partial_cmp(&b).unwrap())
            .unwrap();
        let mean = times.iter().sum::<f64>() / nf;
        let var =
            times.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / (nf - 1.0);
        let std = var.sqrt();

        BenchStats {
            min,
            mean,
            max,
            std,
        }
    }
}

fn bench_all_accum(parts: Parts<f64>, nreps: usize) -> Vec<BenchResult> {
    let mut target = vec![0.0; parts.present.len()];

    let (c_dense, c_sparse) = containers_from_parts(parts);

    (0..nreps)
        .map(|_| BenchResult {
            t_vec: bench_sparse(&c_sparse, &mut target),
            t_dense: bench_dense(&c_dense, &mut target),
        })
        .collect()
}

macro_rules! bench_set {
    ($container: ident, $positions: ident) => {{
        use std::time::Instant;
        let t_start = Instant::now();
        $positions.iter().for_each(|&ix| {
            $container.insert_overwrite(ix, 1.5);
        });
        t_start.elapsed().as_secs_f64()
    }};
}

fn bench_all_set(parts: Parts<f64>, nreps: usize) -> Vec<BenchResult> {
    use rand::seq::SliceRandom;

    let mut rng = rand::thread_rng();
    let n = parts.present.len();

    (0..nreps)
        .map(|_| {
            let (mut c_dense, mut c_sparse) =
                containers_from_parts(parts.clone());

            let positions = {
                let mut ixs = (0..n).collect::<Vec<_>>();
                ixs.shuffle(&mut rng);
                ixs.split_off((n as f64 * 0.9) as usize)
            };

            BenchResult {
                t_vec: bench_set!(c_sparse, positions),
                t_dense: bench_set!(c_dense, positions),
            }
        })
        .collect()
}

macro_rules! bench_get {
    ($container: ident, $n: expr) => {{
        use std::time::Instant;
        let t_start = Instant::now();
        (0..$n).for_each(|ix| {
            $container.get(ix);
        });
        t_start.elapsed().as_secs_f64()
    }};
}

fn bench_all_get(parts: Parts<f64>, nreps: usize) -> Vec<BenchResult> {
    let n = parts.present.len();
    let (c_dense, c_sparse) = containers_from_parts(parts);

    (0..nreps)
        .map(|_| BenchResult {
            t_vec: bench_get!(c_sparse, n),
            t_dense: bench_get!(c_dense, n),
        })
        .collect()
}

fn main() {
    let sparsity = 0.5;
    let n_slices = 20;
    let nreps = 100;
    let ndiscard = 20;

    {
        let n: usize = 1_000_000;
        let parts = quick_parts(n, sparsity, n_slices);
        let results = bench_all_accum(parts.clone(), nreps).split_off(ndiscard);

        let stat_dense =
            BenchStats::new(results.iter().map(|r| r.t_dense).collect());
        let stat_vec =
            BenchStats::new(results.iter().map(|r| r.t_vec).collect());

        println!("accum");
        println!("-----");
        println!("dense: {:#?}", stat_dense);
        println!("sparse: {:#?}", stat_vec);
    }

    {
        let n: usize = 1_000_000;
        let parts = quick_parts(n, sparsity, n_slices);
        let results = bench_all_get(parts, nreps).split_off(ndiscard);

        let stat_dense =
            BenchStats::new(results.iter().map(|r| r.t_dense).collect());
        let stat_vec =
            BenchStats::new(results.iter().map(|r| r.t_vec).collect());

        println!("get");
        println!("---");
        println!("dense: {:#?}", stat_dense);
        println!("sparse: {:#?}", stat_vec);
    }

    {
        let n: usize = 100_000;
        let parts = quick_parts(n, sparsity, n_slices);
        let results = bench_all_set(parts, nreps).split_off(ndiscard);

        let stat_dense =
            BenchStats::new(results.iter().map(|r| r.t_dense).collect());
        let stat_vec =
            BenchStats::new(results.iter().map(|r| r.t_vec).collect());

        println!("set");
        println!("---");
        println!("dense: {:#?}", stat_dense);
        println!("sparse: {:#?}", stat_vec);
    }
}
