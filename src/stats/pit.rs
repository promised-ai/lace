extern crate rv;

use self::rv::dist::Mixture;
use self::rv::traits::{Cdf, Rv};

#[inline]
fn pit_quad_lower_part(a: f64, b: f64, f: f64) -> f64 {
    let q =
        0.5 * f * (a.powi(2) - b.powi(2)) + 1.0 / 3.0 * (b.powi(3) - a.powi(3));
    assert!(q >= 0.0);
    q
}

#[inline]
fn pit_quad_upper_part(a: f64, b: f64, f: f64) -> f64 {
    let q =
        0.5 * f * (b.powi(2) - a.powi(2)) + 1.0 / 3.0 * (a.powi(3) - b.powi(3));
    assert!(q >= 0.0);
    q
}

fn pit_area_quad_prtl(a: f64, b: f64, f: f64) -> f64 {
    if f <= a {
        // pit_quad_lower_part_area(a, b, f)
        f * (a - b) + 0.5 * (b * b - a * a)
    } else if f >= b {
        // pit_quad_upper_part_area(a, b, f)
        f * (b - a) + 0.5 * (a * a - b * b)
    } else {
        // Can exploit the f line's intersecting the uniform CDF line to
        // simplify the math. It becomes the area of two triangles.
        0.5 * (f - a).powi(2) + 0.5 * (b - f).powi(2)
    }
}

fn pit_quad_prtl(a: f64, b: f64, f: f64) -> f64 {
    if f <= a {
        pit_quad_lower_part(a, b, f)
    } else if f >= b {
        pit_quad_upper_part(a, b, f)
    } else {
        pit_quad_upper_part(a, f, f) + pit_quad_lower_part(f, b, f)
    }
}

/// Combines a set of mixtures into on large mixture
pub fn combine_mixtures<Fx>(fxs: &Vec<Mixture<Fx>>) -> Mixture<Fx>
where
    Fx: Clone,
{
    let mut components: Vec<Fx> = vec![];
    let mut weights: Vec<f64> = vec![];
    for fx in fxs {
        components.append(&mut fx.components.clone());
        weights.append(&mut fx.weights.clone());
    }

    let weight_sum: f64 = weights.iter().fold(0.0, |acc, w| acc + w);
    weights = weights.iter().map(|w| w / weight_sum).collect();

    Mixture::new(weights, components).unwrap()
}

struct EmpiricalDist<X>
where
    X: Clone,
{
    xs: Vec<X>,
    fx: Vec<f64>,
}

// Assumes xs is in ascending order
fn unique_ord<X>(xs: &Vec<X>) -> Vec<X>
where
    X: std::cmp::PartialEq + Copy,
{
    let mut unique: Vec<X> = vec![];
    xs.iter().for_each(|&x| {
        if unique.is_empty() {
            unique.push(x);
        } else if unique[unique.len() - 1] != x {
            unique.push(x);
        }
    });
    unique
}

impl<X> EmpiricalDist<X>
where
    X: Clone + Copy + std::cmp::PartialOrd<X> + std::fmt::Debug,
{
    fn new(mut samples: Vec<X>) -> Self {
        samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let xs: Vec<X> = unique_ord(&samples);

        let n = samples.len() as f64;
        let fx: Vec<f64> = xs
            .iter()
            .map(|x| match samples.iter().position(|y| x < y) {
                Some(ix) => (ix as f64) / n,
                None => 1.0,
            })
            .collect();

        EmpiricalDist { xs, fx }
    }
}

/// Probability Inverse Transform (PIT)
///
/// Returns a tuple containing the PIT error and the error's centroid.
pub fn pit<X, Fx>(xs: &Vec<X>, fx: &Mixture<Fx>) -> (f64, f64)
where
    Fx: Rv<X> + Cdf<X>,
{
    let ps: Vec<f64> = xs.iter().map(|x| fx.cdf(x)).collect();
    let empirical = EmpiricalDist::new(ps);

    let mut a: f64 = 0.0;
    let area = empirical.xs.iter().zip(empirical.fx.iter()).fold(
        0.0,
        |acc, (&b, &f)| {
            let q = pit_area_quad_prtl(a, b, f);
            a = b;
            acc + q
        },
    );

    a = 0.0;
    let quad = empirical.xs.iter().zip(empirical.fx.iter()).fold(
        0.0,
        |acc, (&b, &f)| {
            let q = pit_quad_prtl(a, b, f);
            a = b;
            acc + q
        },
    );

    let centroid = quad / area;
    let error = area; // should be in [0, 0.5]

    (error, centroid)
}

#[cfg(test)]
mod tests {
    use self::rv::dist::{Categorical, Gaussian};
    use super::*;

    const N_TRIES: usize = 5;

    #[test]
    fn unique_ord_f64_all_unique() {
        let xs: Vec<f64> = vec![0.1, 0.2, 0.3, 1.0, 2.0];
        assert_eq!(unique_ord(&xs), xs);
    }

    #[test]
    fn unique_ord_f64_repeats() {
        let xs: Vec<f64> = vec![0.1, 0.1, 0.2, 0.2, 0.3, 1.0, 2.0, 2.0];
        assert_eq!(unique_ord(&xs), vec![0.1, 0.2, 0.3, 1.0, 2.0]);
    }

    #[test]
    fn empirical_f64_no_repeats() {
        let xs: Vec<f64> = vec![0.1, 0.2, 0.3, 1.0, 2.0];
        let empirical = EmpiricalDist::new(xs.clone());

        assert_eq!(empirical.xs, vec![0.1, 0.2, 0.3, 1.0, 2.0]);
        assert_eq!(empirical.fx, vec![0.2, 0.4, 0.6, 0.8, 1.0]);
    }

    #[test]
    fn empirical_u8_binary() {
        let xs: Vec<u8> = vec![0, 0, 1, 1, 1];
        let empirical = EmpiricalDist::new(xs.clone());

        assert_eq!(empirical.xs, vec![0, 1]);
        assert_eq!(empirical.fx, vec![0.4, 1.0]);
    }

    #[test]
    fn gauss_pit_for_samples_from_target_should_have_low_error() {
        let mut rng = rand::thread_rng();
        let g = Gaussian::standard();
        let mixture = Mixture::new(vec![1.0], vec![g.clone()]).unwrap();

        let passed = (0..N_TRIES).any(|_| {
            let xs: Vec<f64> = g.sample(1000, &mut rng);
            let (error, centroid) = pit(&xs, &mixture);

            error < 0.05 && (centroid - 0.5).abs() < 0.1
        });

        assert!(passed);
    }

    #[test]
    fn gauss_pit_for_samples_from_narrow_should_have_high_error() {
        let mut rng = rand::thread_rng();
        let g_gen = Gaussian::standard();
        let g_target = Gaussian::new(0.0, 0.1).unwrap();
        let mixture = Mixture::new(vec![1.0], vec![g_target]).unwrap();

        let passed = (0..N_TRIES).any(|_| {
            let xs: Vec<f64> = g_gen.sample(1000, &mut rng);
            let (error, centroid) = pit(&xs, &mixture);

            // The means are the same, so the error centroid should be around
            // 0.5
            error > 0.2 && (centroid - 0.5).abs() < 0.07
        });

        assert!(passed);
    }

    #[test]
    fn gauss_pit_for_samples_from_wide_should_have_high_error() {
        let mut rng = rand::thread_rng();
        let g_gen = Gaussian::standard();
        let g_target = Gaussian::new(1.5, 1.0).unwrap();
        let mixture = Mixture::new(vec![1.0], vec![g_target]).unwrap();

        let passed = (0..N_TRIES).any(|_| {
            let xs: Vec<f64> = g_gen.sample(1000, &mut rng);
            let (error, centroid) = pit(&xs, &mixture);

            // Since the target is shifted right, the error centroid should be
            // to the left.
            error > 0.25 && centroid < 0.45
        });
    }

    #[test]
    fn ctgrl_pit_manual_computation() {
        let c_gen = Categorical::new(&vec![0.25, 0.75]).unwrap();
        let mixture = Mixture::new(vec![1.0], vec![c_gen]).unwrap();

        // CDFs = [0.25, 1.0]
        // EmpiricalF = [0.4, 1.0]
        let xs: Vec<u8> = vec![0, 0, 1, 1, 1];
        let (error, centroid) = pit(&xs, &mixture);

        // Computed these manually
        assert_eq!(error, 0.35);
        assert!((centroid - 0.1479166666666667 / 0.35).abs() < 1E-12);
    }

    #[test]
    fn ctgrl_pit_for_samples_from_target_should_have_low_error() {
        let mut rng = rand::thread_rng();
        let c_gen = Categorical::new(&vec![0.25, 0.75]).unwrap();
        let mixture = Mixture::new(vec![1.0], vec![c_gen.clone()]).unwrap();

        let passed = (0..N_TRIES).any(|_| {
            let xs: Vec<u8> = c_gen.sample(100, &mut rng);
            let (error, centroid) = pit(&xs, &mixture);

            error < 0.05 && (centroid - 0.5).abs() < 0.1
        });
    }
}
