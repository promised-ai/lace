pub fn digamma(x: f64) -> f64 {
    if x < 5.0 {
        // TODO: make this closed form instead of recursive?
        digamma(x + 1.0) - 1.0 / x
    } else {
        x.ln() - 1.0 / (2.0 * x) - 1.0 / (12.0 * x * x)
            + 1.0 / (120.0 * x.powi(4)) - 1.0 / (252.0 * x.powi(6))
            + 1.0 / (240.0 * x.powi(8)) - 5.0 / (660.0 * x.powi(10))
            + 691.0 / (32760.0 * x.powi(12)) - 1.0 / (12.0 * x.powi(14))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 10E-7;

    #[test]
    fn digamma_1() {
        assert_relative_eq!(
            digamma(1.0),
            -0.57721566490153309,
            epsilon = TOL
        );
    }

    #[test]
    fn digamma_5() {
        assert_relative_eq!(digamma(5.0), 1.5061176684318003, epsilon = TOL);
    }

    #[test]
    fn digamma_20() {
        assert_relative_eq!(digamma(20.0), 2.9705239922421489, epsilon = TOL);
    }

    #[test]
    fn digamma_large() {
        assert_relative_eq!(
            digamma(123.4),
            4.8113737751162775,
            epsilon = TOL
        );
    }

    #[test]
    fn digamma_small() {
        assert_relative_eq!(digamma(0.2), -5.2890398965921879, epsilon = TOL);
    }
}
