#![feature(const_fn)]
use braid_data::*;

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

fn main() {
    let mut rng = rand::thread_rng();
    let parts = gen_parts(100, 0.2, 3, |&mut _r| 1.0, &mut rng);
    let bits = parts
        .present
        .iter()
        .map(|b| if *b { "█" } else { "▁" })
        .collect::<String>();

    println!("{}", bits);
}
