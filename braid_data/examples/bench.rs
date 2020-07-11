use braid_data::*;
use rand::Rng;

const MU: f64 = 2.0;
const SIGMA: f64 = 2.5;

fn dense_accum(container: &DenseContainer<f64>, target: &mut Vec<f64>) {
    let ln_sigma = SIGMA.ln();
    container
        .values
        .iter()
        .zip(container.present.iter())
        .zip(target.iter_mut())
        .for_each(|((&x, &pr), y)| {
            if pr {
                let term = (MU - x) / SIGMA;
                *y -= ln_sigma + 0.5 * term * term;
            }
        })
}

fn vec_accum(container: &VecContainer<f64>, target: &mut Vec<f64>) {
    let ln_sigma = SIGMA.ln();
    let slices = container.get_slices();
    slices.iter().for_each(|(ix, xs)| {
        xs.iter().enumerate().for_each(|(i, &x)| {
            let term = (MU - x) / SIGMA;
            target[ix + i] -= ln_sigma + 0.5 * term * term;
        })
    })
}

fn lookup_accum(container: &LookupContainer<f64>, target: &mut Vec<f64>) {
    let ln_sigma = SIGMA.ln();
    let slices = container.get_slices();
    slices.iter().for_each(|(ix, xs)| {
        xs.iter().enumerate().for_each(|(i, &x)| {
            let term = (MU - x) / SIGMA;
            target[ix + i] -= ln_sigma + 0.5 * term * term;
        })
    })
}

fn lookup_accum2(container: &LookupContainer<f64>, target: &mut Vec<f64>) {
    let ln_sigma = SIGMA.ln();
    let lookup = container.lookup();
    let xs = container.data();
    lookup.iter().for_each(|&(ix, iix, n)| {
        (0..n).for_each(|i| {
            let term = (MU - xs[iix + i]) / SIGMA;
            target[ix + i] -= ln_sigma + 0.5 * term * term;
        })
    })
}

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
) -> (DenseContainer<f64>, VecContainer<f64>, LookupContainer<f64>) {
    let c_dense = {
        let parts_clone = parts.clone();
        DenseContainer::new(parts_clone.data, parts_clone.present)
    };

    let c_vec = {
        let parts_clone = parts.clone();
        VecContainer::new(parts_clone.data, &parts_clone.present)
    };

    let c_lookup = {
        let parts_clone = parts.clone();
        LookupContainer::new(&parts_clone.data, &parts_clone.present)
    };

    (c_dense, c_vec, c_lookup)
}

fn bench_dense(
    container: &DenseContainer<f64>,
    mut target: &mut Vec<f64>,
) -> f64 {
    use std::time::Instant;
    let t_start = Instant::now();
    dense_accum(&container, &mut target);
    t_start.elapsed().as_secs_f64()
}

fn bench_vec(container: &VecContainer<f64>, mut target: &mut Vec<f64>) -> f64 {
    use std::time::Instant;
    let t_start = Instant::now();
    vec_accum(&container, &mut target);
    t_start.elapsed().as_secs_f64()
}

fn bench_lookup(
    container: &LookupContainer<f64>,
    mut target: &mut Vec<f64>,
) -> f64 {
    use std::time::Instant;
    let t_start = Instant::now();
    lookup_accum(&container, &mut target);
    t_start.elapsed().as_secs_f64()
}

fn bench_lookup2(
    container: &LookupContainer<f64>,
    mut target: &mut Vec<f64>,
) -> f64 {
    use std::time::Instant;
    let t_start = Instant::now();
    lookup_accum2(&container, &mut target);
    t_start.elapsed().as_secs_f64()
}

#[derive(Clone, Debug)]
struct BenchResult {
    pub t_dense: f64,
    pub t_vec: f64,
    pub t_lookup: f64,
    pub t_lookup2: f64,
}

fn bench_all(parts: Parts<f64>, nreps: usize) -> Vec<BenchResult> {
    let mut target = vec![0.0; parts.present.len()];

    let (c_dense, c_vec, c_lookup) = containers_from_parts(parts);

    (0..nreps)
        .map(|_| BenchResult {
            t_dense: bench_dense(&c_dense, &mut target),
            t_vec: bench_vec(&c_vec, &mut target),
            t_lookup: bench_lookup(&c_lookup, &mut target),
            t_lookup2: bench_lookup2(&c_lookup, &mut target),
        })
        .collect()
}

fn main() {
    let n: usize = 1_000_000;
    let parts = quick_parts(n, 0.0, 1);
    let results = bench_all(parts, 20);
    println!("{:#?}", results);
}
