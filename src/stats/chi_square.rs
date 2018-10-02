//! Chi-squared tests
extern crate special;

use self::special::Gamma;

fn chi_square_cdf(x: f64, k: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        (x / 2.0).inc_gamma(k / 2.0)
    }
}

/// Chi-square goodness of fit test comparing the observed (sample) frequencies
/// in `freq_obs` with the expected (true) frequencies, `freq_exp`.
pub fn chi_square_test(freq_obs: &[f64], freq_exp: &[f64]) -> (f64, f64) {
    let stat: f64 =
        freq_obs
            .iter()
            .zip(freq_exp.iter())
            .fold(0.0, |acc, (o, e)| {
                let diff = o - e;
                acc + diff * diff / e
            });

    let k = freq_obs.len() - 1;
    let p = 1.0 - chi_square_cdf(stat, k as f64);

    (stat, p)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-8;

    #[test]
    fn chi_square_should_be_zero_if_freqs_identical() {
        let freq_obs: Vec<f64> = vec![2.0, 2.0, 2.0, 3.0];
        let freq_exp: Vec<f64> = vec![2.0, 2.0, 2.0, 3.0];

        let (x2, p) = chi_square_test(&freq_obs, &freq_exp);
        assert_relative_eq!(0.0, x2, epsilon = TOL);
        assert_relative_eq!(1.0, p, epsilon = TOL);
    }

    #[test]
    fn chi_square_simple_value_test_1() {
        let freq_obs: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let freq_exp: Vec<f64> = vec![2.0, 3.0, 4.0, 1.0];

        let (x2, p) = chi_square_test(&freq_obs, &freq_exp);
        assert_relative_eq!(10.083333333333334, x2, epsilon = TOL);
        assert_relative_eq!(0.017870892893625558, p, epsilon = TOL);
    }

    #[test]
    fn chi_square_simple_value_test_2() {
        let freq_obs: Vec<f64> = vec![24.0, 20.0, 27.0, 29.0];
        let freq_exp: Vec<f64> = vec![19.0, 25.0, 26.0, 30.0];

        let (x2, p) = chi_square_test(&freq_obs, &freq_exp);
        assert_relative_eq!(2.3875843454790822, x2, epsilon = TOL);
        assert_relative_eq!(0.49594997742093094, p, epsilon = TOL);
    }

}
