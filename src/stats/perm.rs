extern crate itertools;
extern crate rand;

use self::itertools::Itertools;
use self::rand::distributions::Uniform;
use self::rand::Rng;

pub fn perm_test<T, F>(
    xs: &Vec<T>,
    ys: &Vec<T>,
    func: F,
    n_perms: usize,
    rng: &mut impl Rng,
) -> f64
where
    F: Fn(&Vec<T>, &Vec<T>) -> f64,
    T: Clone + Copy,
{
    let u = Uniform::new(0.0, 1.0);
    let f0 = func(&xs, &ys);

    let mut xy = xs.clone();
    xy.append(&mut ys.clone());

    let incr = 1.0 / n_perms as f64;
    (0..n_perms).fold(0.0, |acc, _| {
        let (x, y) = xy.iter().partition(|_| rng.sample(u) < 0.5);
        if func(&x, &y) > f0 {
            acc + incr
        } else {
            acc
        }
    })
}

pub fn uv_gauss_kernel(xs: &Vec<f64>, ys: &Vec<f64>) -> f64 {
    let h = 1.0;
    // Gaussian kernl w/ bandwitdh `h`
    fn k(x: f64, y: f64, h: f64) -> f64 {
        (-(x - y).powi(2).sqrt() / h).exp()
    }

    let n = xs.len() as f64;
    let m = ys.len() as f64;

    let dx = xs
        .iter()
        .combinations(2)
        .fold(0.0, |acc, x| acc + k(*x[0], *x[1], h));
    let dy = ys
        .iter()
        .combinations(2)
        .fold(0.0, |acc, y| acc + k(*y[0], *y[1], h));
    let dxy = xs
        .iter()
        .cartesian_product(ys.iter())
        .fold(0.0, |acc, (&x, &y)| acc + k(x, y, h));

    (2.0 * dx + n) / n.powi(2) - 2.0 / (m * n) * dxy
        + (2.0 * dy + m) / m.powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-8;

    #[test]
    fn uv_gauss_perm_test_should_be_one_if_xs_is_ys() {
        let xs = vec![0.1, 1.2, 3.2, 1.8, 0.1, 2.0];
        let mut rng = rand::thread_rng();
        let f = perm_test(&xs, &xs, uv_gauss_kernel, 1000, &mut rng);
        // won't be exactly zero because the original permutation will show up
        // every now and again because the permutations are random
        assert!(f >= 0.97);
    }

    #[test]
    fn uv_gauss_perm_test_should_be_zero_if_xs_very_different_from_ys() {
        let xs = vec![0.0; 5];
        let ys = vec![1.0; 5];
        let mut rng = rand::thread_rng();
        let f = perm_test(&xs, &ys, uv_gauss_kernel, 1000, &mut rng);
        assert_relative_eq!(f, 0.0, epsilon = TOL);
    }
}
