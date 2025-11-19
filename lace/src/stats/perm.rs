//! Permutation tests and utilities
use rand::Rng;

pub struct PermTestData<T> {
    // The x and y data stored together
    data: Vec<T>,
    // The index at which the ys start
    border: usize,
}

impl<T> PermTestData<T> {
    fn new(mut xs: Vec<T>, mut ys: Vec<T>) -> Self {
        let border = xs.len();
        xs.append(&mut ys);
        PermTestData { data: xs, border }
    }

    fn repartition<R: rand::Rng>(&mut self, rng: &mut R) {
        let mut gate = self.border;
        (self.border..self.data.len()).for_each(|i| {
            let u: f64 = rng.random();
            if u < 0.5 {
                self.data.swap(gate, i);
                gate += 1;
            }
        });

        (0..self.border).for_each(|i| {
            let u: f64 = rng.random();
            if u < 0.5 {
                gate -= 1;
                self.data.swap(gate, i);
            }
        });

        self.border = gate;
    }

    fn xs(&self) -> &[T] {
        // This is sound because border can never extend outside of `data`
        unsafe {
            let ptr = self.data.as_ptr();
            std::slice::from_raw_parts(ptr, self.border)
        }
    }

    fn ys(&self) -> &[T] {
        // This is sound because border is strictly less than len, so len -
        // border can never extend outside of data.
        unsafe {
            let ptr = self.data.as_ptr().add(self.border);
            std::slice::from_raw_parts(ptr, self.data.len() - self.border)
        }
    }
}

/// Two-sample permutation test on samples `xs` and `ys` given the statistic-
/// generating function, `func`.
pub fn perm_test<T, F, R>(
    xs: Vec<T>,
    ys: Vec<T>,
    func: F,
    n_perms: u32,
    mut rng: &mut R,
) -> f64
where
    F: Fn(&PermTestData<T>) -> f64 + Send + Sync,
    T: Clone + Send + Sync,
    R: Rng,
{
    let mut data = PermTestData::new(xs, ys);
    let f0 = func(&data);

    let acc = (0..n_perms)
        .map(|_| {
            data.repartition(&mut rng);
            if func(&data) > f0 {
                1.0
            } else {
                0.0
            }
        })
        .sum::<f64>();

    acc / f64::from(n_perms)
}

pub fn gauss_kernel<T: L2Norm>(data: &PermTestData<T>) -> f64 {
    let h = 1.0;

    fn k<T: L2Norm>(x: &T, y: &T, h: f64) -> f64 {
        (-x.l2_dist(y) / h).exp()
    }

    let xs = data.xs();
    let ys = data.ys();

    let n = xs.len() as f64;
    let m = ys.len() as f64;

    let dx = xs
        .iter()
        .enumerate()
        .map(|(i, x1)| {
            xs.iter().skip(i + 1).map(|x2| k(x1, x2, h)).sum::<f64>()
        })
        .sum::<f64>();

    let dy = ys
        .iter()
        .enumerate()
        .map(|(i, y1)| {
            ys.iter().skip(i + 1).map(|y2| k(y1, y2, h)).sum::<f64>()
        })
        .sum::<f64>();

    let dxy = xs
        .iter()
        .map(|x| ys.iter().map(|y| k(x, y, h)).sum::<f64>())
        .sum::<f64>();

    2_f64.mul_add(dx, n) / n.powi(2)
        - (2.0 / (m * n)).mul_add(dxy, -(2_f64.mul_add(dy, m) / m.powi(2)))
}

pub trait L2Norm {
    fn l2_dist(&self, y: &Self) -> f64;
}

impl L2Norm for f64 {
    #[inline(always)]
    fn l2_dist(&self, y: &f64) -> f64 {
        (self - y).abs()
    }
}

impl L2Norm for Vec<f64> {
    fn l2_dist(&self, y: &Vec<f64>) -> f64 {
        self.iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - yi).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl L2Norm for (f64, f64) {
    fn l2_dist(&self, y: &(f64, f64)) -> f64 {
        let d0 = self.0 - y.0;
        let d1 = self.1 - y.1;
        d0.hypot(d1)
    }
}

/// Two-sample permutation test using the (slow) Gaussian Kernel statistic
pub fn gauss_perm_test<T: L2Norm + Clone + Send + Sync>(
    xs: Vec<T>,
    ys: Vec<T>,
    n_perms: u32,
    mut rng: &mut impl Rng,
) -> f64 {
    perm_test(xs, ys, gauss_kernel, n_perms, &mut rng)
}

#[cfg(test)]
mod tests {
    use approx::*;

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
        assert_relative_eq!(
            x.l2_dist(&y),
            3.599_999_999_999_999_6,
            epsilon = TOL
        );
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
        assert_relative_eq!(
            x.l2_dist(&y),
            0.830_662_386_291_807_2,
            epsilon = TOL
        );
    }

    #[test]
    fn uv_gauss_perm_test_should_be_one_if_xs_is_ys() {
        let xs = vec![0.1, 1.2, 3.2, 1.8, 0.1, 2.0];
        let mut rng = rand::rng();
        let f = gauss_perm_test(xs.clone(), xs, 1000, &mut rng);
        // won't be exactly one because the original permutation will show up
        // every now and again because the permutations are random
        assert!(f >= 0.97);
    }

    #[test]
    fn uv_gauss_perm_test_should_be_zero_if_xs_very_different_from_ys() {
        let xs = vec![0.0; 5];
        let ys = vec![1.0; 5];
        let mut rng = rand::rng();
        let f = gauss_perm_test(xs, ys, 1000, &mut rng);
        assert_relative_eq!(f, 0.0, epsilon = TOL);
    }

    #[test]
    fn perm_data_repartition_smoke_test() {
        let xs = vec![0_u8; 10];
        let ys = vec![1_u8; 10];
        let mut perm_data = PermTestData::new(xs, ys);
        let mut rng = rand::rng();
        for _ in 0..1000 {
            perm_data.repartition(&mut rng);
            let _xs_i = perm_data.xs();
            let _ys_i = perm_data.ys();
        }
    }

    #[test]
    fn perm_data_xs_and_ys() {
        let perm_data = {
            let xs = vec![0, 1, 2, 3, 4];
            let ys = vec![5, 6, 7, 8, 9];
            PermTestData::new(xs, ys)
        };
        let xs = perm_data.xs();
        let ys = perm_data.ys();

        assert_eq!(xs.len(), 5);
        assert_eq!(ys.len(), 5);

        assert_eq!(xs[0], 0);
        assert_eq!(xs[4], 4);

        assert_eq!(ys[0], 5);
        assert_eq!(ys[4], 9);
    }
}
