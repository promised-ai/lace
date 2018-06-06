extern crate itertools;
extern crate rand;

use self::itertools::Itertools;
use self::rand::distributions::Uniform;
use self::rand::Rng;

fn repartition<T: Clone>(xs: &Vec<T>, rng: &mut impl Rng) -> (Vec<T>, Vec<T>) {
    let n = xs.len();
    let mut xs_t = Vec::with_capacity(n / 2);
    let mut xs_f = Vec::with_capacity(n / 2);

    let u = Uniform::new(0.0, 1.0);
    xs.iter().for_each(|x| {
        if rng.sample(u) < 0.5 {
            xs_t.push(x.clone())
        } else {
            xs_f.push(x.clone())
        }
    });
    (xs_t, xs_f)
}

pub fn perm_test<T, F>(
    xs: &Vec<T>,
    ys: &Vec<T>,
    func: F,
    n_perms: usize,
    mut rng: &mut impl Rng,
) -> f64
where
    F: Fn(&Vec<T>, &Vec<T>) -> f64,
    T: Clone,
{
    let f0 = func(&xs, &ys);

    let mut xy = xs.clone();
    xy.append(&mut ys.clone());

    let incr = 1.0 / n_perms as f64;
    (0..n_perms).fold(0.0, |acc, _| {
        let (x, y) = repartition(&xy, &mut rng);
        if func(&x, &y) > f0 {
            acc + incr
        } else {
            acc
        }
    })
}

pub trait L2Norm {
    fn l2_dist(&self, y: &Self) -> f64;
}

impl L2Norm for f64 {
    fn l2_dist(&self, y: &f64) -> f64 {
        (self - y).powi(2).sqrt()
    }
}

impl L2Norm for Vec<f64> {
    fn l2_dist(&self, y: &Vec<f64>) -> f64 {
        self.iter()
            .zip(y.iter())
            .fold(0.0, |acc, (xi, yi)| acc + (xi - yi).powi(2))
            .sqrt()
    }
}

pub fn gauss_kernel<T: L2Norm>(xs: &Vec<T>, ys: &Vec<T>) -> f64 {
    let h = 1.0;
    // Gaussian kernl w/ bandwitdh `h`
    fn k<T: L2Norm>(x: &T, y: &T, h: f64) -> f64 {
        (-x.l2_dist(&y) / h).exp()
    }

    let n = xs.len() as f64;
    let m = ys.len() as f64;

    // This is so slow.
    let dx = xs
        .iter()
        .combinations(2)
        .fold(0.0, |acc, x| acc + k(x[0], x[1], h));
    let dy = ys
        .iter()
        .combinations(2)
        .fold(0.0, |acc, y| acc + k(y[0], y[1], h));
    let dxy = xs
        .iter()
        .cartesian_product(ys.iter())
        .fold(0.0, |acc, (x, y)| acc + k(x, y, h));

    (2.0 * dx + n) / n.powi(2) - 2.0 / (m * n) * dxy
        + (2.0 * dy + m) / m.powi(2)
}

pub fn gauss_perm_test<T: L2Norm + Clone>(
    xs: &Vec<T>,
    ys: &Vec<T>,
    n_perms: usize,
    mut rng: &mut impl Rng,
) -> f64 {
    perm_test(&xs, &ys, gauss_kernel, n_perms, &mut rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-8;

    #[test]
    fn l2_norm_of_identical_f64_should_be_zero() {
        assert_relative_eq!(0.4_f64.l2_dist(&0.4), 0.0, epsilon = TOL);
        assert_relative_eq!(0.0_f64.l2_dist(&0.0), 0.0, epsilon = TOL);
        assert_relative_eq!((-1.0_f64).l2_dist(&-1.0), 0.0, epsilon = TOL);
    }

    #[test]
    fn l2_norm_f64_value_check() {
        let x: f64 = 1.2;
        let y: f64 = -2.4;
        assert_relative_eq!(x.l2_dist(&y), 3.5999999999999996, epsilon = TOL);
    }

    #[test]
    fn l2_norm_of_identical_vec_f64_should_be_zero() {
        let x = vec![0.0, 1.0, 1.2, 3.4];
        assert_relative_eq!(x.l2_dist(&x), 0.0, epsilon = TOL);

        let y = vec![0.0, 0.0, 0.0];
        assert_relative_eq!(y.l2_dist(&y), 0.0, epsilon = TOL);

        let z = vec![-1.0, -2.0, 3.0];
        assert_relative_eq!(z.l2_dist(&z), 0.0, epsilon = TOL);
    }

    #[test]
    fn l2_norm_vec_f64_value_check() {
        let x = vec![4.0, 5.0, -6.0];
        let y = vec![3.2, 5.1, -5.8];
        assert_relative_eq!(x.l2_dist(&y), 0.83066238629180722, epsilon = TOL);
    }

    #[test]
    fn uv_gauss_perm_test_should_be_one_if_xs_is_ys() {
        let xs = vec![0.1, 1.2, 3.2, 1.8, 0.1, 2.0];
        let mut rng = rand::thread_rng();
        let f = gauss_perm_test(&xs, &xs, 1000, &mut rng);
        // won't be exactly zero because the original permutation will show up
        // every now and again because the permutations are random
        assert!(f >= 0.97);
    }

    #[test]
    fn uv_gauss_perm_test_should_be_zero_if_xs_very_different_from_ys() {
        let xs = vec![0.0; 5];
        let ys = vec![1.0; 5];
        let mut rng = rand::thread_rng();
        let f = gauss_perm_test(&xs, &ys, 1000, &mut rng);
        assert_relative_eq!(f, 0.0, epsilon = TOL);
    }
}
