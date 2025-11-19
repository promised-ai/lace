use rv::dist::Bernoulli;
use rv::dist::Categorical;
use rv::dist::Gaussian;
use rv::dist::Mixture;
use rv::dist::Poisson;
use rv::traits::HasDensity;
use rv::traits::Mean;
use rv::traits::QuadBounds;

/// Compute the normed mean Total Variation Distance of a set of mixture
/// distributions with the mean of distributions.
///
/// # Notes
/// - The output will be in [0, 1.0).
/// - Normalization is used to account for the fact that the maximum TVD is
///   limited by the number of mixtures. For example, if there are two mixtures
///   in `mixtures` the max TVD in only 1/2; if there are three, the max TVD is
///   2/3; if there are four the max TVD is 3/4; and so on. We divide the final
///   output by `(n - 1) / n`, where `n` is the number of mixtures, so that the
///   output can be interpreted similarly regardless of the input.
pub fn mixture_normed_tvd<Fx>(mixtures: &[Mixture<Fx>]) -> f64
where
    Fx: Clone,
    Mixture<Fx>: TotalVariationDistance,
{
    let n = mixtures.len() as f64;
    let norm = (n - 1.0) / n;

    let combined = Mixture::combine(mixtures.to_owned());
    let tvd = mixtures.iter().map(|mm| combined.tvd(mm)).sum::<f64>()
        / mixtures.len() as f64;

    tvd / norm
}

pub trait TotalVariationDistance {
    fn tvd(&self, other: &Self) -> f64;
}

fn gaussian_quad_points(
    f1: &Mixture<Gaussian>,
    f2: &Mixture<Gaussian>,
) -> Vec<f64> {
    // Get the lower and upper bound for quadrature
    let (a, b) = {
        let (a_1, b_1) = f1.quad_bounds();
        let (a_2, b_2) = f2.quad_bounds();
        (a_1.min(a_2), b_1.max(b_2))
    };

    // Get a list of sorted means and their associated stddevs
    let params = {
        let mut params = f1
            .components()
            .iter()
            .chain(f2.components().iter())
            .map(|cpnt| (cpnt.mu(), cpnt.sigma()))
            .collect::<Vec<_>>();
        params.sort_unstable_by(|(a, _), (b, _)| a.total_cmp(b));
        params
    };

    let mut last_mean = params[0].0;
    let mut last_std = params[0].1;
    let mut points = vec![a, last_mean];

    for &(mean, std) in params.iter().skip(1) {
        let dist = mean - last_mean;
        let z_dist = dist / ((last_std + std) / 2.0);
        if z_dist > 1.0 {
            points.push(mean);
            last_std = std;
            last_mean = mean;
        }
    }

    points.push(b);
    points
}

impl TotalVariationDistance for Mixture<Gaussian> {
    fn tvd(&self, other: &Self) -> f64 {
        use rv::misc::gauss_legendre_quadrature_cached;
        use rv::misc::gauss_legendre_table;

        let func = |x| (self.f(&x) - other.f(&x)).abs();

        let quad_level = 16;
        let quad_points = gaussian_quad_points(self, other);
        let (weights, roots) = gauss_legendre_table(quad_level);

        let mut right = quad_points[0];
        quad_points
            .iter()
            .skip(1)
            .map(|&x| {
                let q = gauss_legendre_quadrature_cached(
                    func,
                    (right, x),
                    &weights,
                    &roots,
                );
                right = x;
                q
            })
            .sum::<f64>()
            / 2.0
    }
}

impl TotalVariationDistance for Mixture<Categorical> {
    fn tvd(&self, other: &Self) -> f64 {
        let k = self.components()[0].k();
        assert_eq!(k, other.components()[0].k());
        (0..k)
            .map(|x| (self.f(&x) - other.f(&x)).abs())
            .sum::<f64>()
            / 2.0
    }
}

impl TotalVariationDistance for Mixture<Bernoulli> {
    fn tvd(&self, other: &Self) -> f64 {
        let q =
            (self.f(&0) - other.f(&0)).abs() + (self.f(&1) - other.f(&1)).abs();
        q / 2.0
    }
}

impl TotalVariationDistance for Mixture<Poisson> {
    fn tvd(&self, other: &Self) -> f64 {
        let threshold = 1e-14;
        let m: u32 = self.mean().unwrap().min(other.mean().unwrap()) as u32;

        let mut x: u32 = 0;
        let mut q: f64 = 0.0;
        loop {
            let f1 = self.f(&x);
            let f2 = other.f(&x);

            let diff = (f1 - f2).abs();

            q += diff;
            x += 1;

            if x > m && (f1 < threshold && f2 < threshold) {
                break;
            }
        }
        q / 2.0
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn gauss_moving_means_away_increases_tvd() {
        let mut last_tvd = 0.0;
        (0..10).for_each(|i| {
            let dist = 0.5 * (i + 1) as f64;
            let g1 = Gaussian::new(-dist / 2.0, 1.0).unwrap();
            let g2 = Gaussian::new(dist / 2.0, 1.0).unwrap();

            let m1 = Mixture::uniform(vec![g1]).unwrap();
            let m2 = Mixture::uniform(vec![g2]).unwrap();

            let tvd = mixture_normed_tvd(&[m1, m2]);

            eprintln!("{i} - d: {dist}, tvd: {tvd}");

            assert!(last_tvd < tvd);
            assert!(tvd <= 1.0);

            last_tvd = tvd;
        });
    }

    #[test]
    fn count_moving_means_away_increases_tvd() {
        let mut last_tvd = 0.0;
        (0..10).for_each(|i| {
            let p1 = Poisson::new(5.0).unwrap();
            let p2 = Poisson::new(5.0 + (i + 1) as f64).unwrap();

            let m1 = Mixture::uniform(vec![p1]).unwrap();
            let m2 = Mixture::uniform(vec![p2]).unwrap();

            let tvd = mixture_normed_tvd(&[m1, m2]);

            eprintln!("{i} tvd: {tvd}");

            assert!(last_tvd < tvd);
            assert!(tvd <= 1.0);

            last_tvd = tvd;
        });
    }

    #[test]
    fn bernoulli_moving_means_away_increases_tvd() {
        let mut last_tvd = std::f64::NEG_INFINITY;
        (0..10).for_each(|i| {
            let p = 0.5 / (i + 1) as f64;
            let b1 = Bernoulli::new(p).unwrap();
            let b2 = Bernoulli::new(1.0 - p).unwrap();

            let m1 = Mixture::uniform(vec![b1]).unwrap();
            let m2 = Mixture::uniform(vec![b2]).unwrap();

            let tvd = mixture_normed_tvd(&[m1, m2]);

            eprintln!("{i} p: {p}, tvd: {tvd}");

            assert!(last_tvd < tvd);
            assert!(tvd <= 1.0);

            last_tvd = tvd;
        });
    }

    #[test]
    fn categorical_moving_means_away_increases_tvd() {
        let mut last_tvd = std::f64::NEG_INFINITY;
        (0..10).for_each(|i| {
            let p = 0.5 / (i + 1) as f64;
            let c1 = Categorical::new(&[p, 1.0 - p]).unwrap();
            let c2 = Categorical::new(&[1.0 - p, p]).unwrap();

            let m1 = Mixture::uniform(vec![c1]).unwrap();
            let m2 = Mixture::uniform(vec![c2]).unwrap();

            let tvd = mixture_normed_tvd(&[m1, m2]);

            eprintln!("{i} p: {p}, tvd: {tvd}");

            assert!(last_tvd < tvd);
            assert!(tvd <= 1.0);

            last_tvd = tvd;
        });
    }
}
