use std::f64::INFINITY;
use special::{gamma, gammaln, gammaln_sign};

const ASYMP_FACTOR: f64 = 1.0E6;
const MAXGAM: f64 = 171.624376956302725;
const MAXLOG: f64 = 7.08396418532264106224E2;


pub fn beta(mut a: f64, mut b: f64) -> f64 {
    let mut y: f64;
    let mut sign = 1.0;

    let a_overflows = a > (i64::max_value() as f64);
    let a_is_int = a == a.trunc();
    if a <= 0.0 {
        if a_overflows {
            return INFINITY
        }
        if a_is_int {
            return beta_negint(a, b);
        }
    }

    let b_overflows = b > (i64::max_value() as f64);
    let b_is_int = b == b.trunc();
    if b <= 0.0 {
        if b_overflows {
            return sign * INFINITY
        }
        if b_is_int {
            return beta_negint(b, a);
        }
    }


    if a.abs() < b.abs() {
        y = a; a = b; b = y;
    }

    if a.abs() > ASYMP_FACTOR * b.abs() && a > ASYMP_FACTOR {
        /* Avoid loss of precision in lgam(a + b) - lgam(a) */
        y = lbeta_asymp(a, b);
        return sign * y.exp();
    }

    y = a + b;
    if y.abs() > MAXGAM || a.abs() > MAXGAM || b.abs() > MAXGAM {
        let (y0, sgngam) = gammaln_sign(y);
        sign *= sgngam;		/* keep track of the sign */
        let (mut y1, sgngam) = gammaln_sign(b);
        y1 -= y0;
        sign *= sgngam;
        let (mut y2, sgngam) = gammaln_sign(a);
        y2 += y1;
        sign *= sgngam;
        if y2 > MAXLOG {
            return sign * INFINITY;
        }
        return sign * y2.exp();
    }

    y = gamma(y);
    a = gamma(a);
    b = gamma(b);
    if y == 0.0 {return INFINITY}

    if (a.abs() - y.abs()).abs() > (b.abs() - y.abs()).abs() {
        y = b / y;
        y *= a;
    } else {
        y = a / y;
        y *= b;
    }

    y
}


pub fn betaln(x: f64, y: f64) -> f64 {
    gammaln(x) + gammaln(y) - gammaln(x + y)
}


fn lbeta_asymp(a: f64, b: f64) -> f64 {
    let (mut r, _) = gammaln_sign(b);
    r -= b * a.ln();

    r += b * (1.0 - b) / (2.0*a);
    r += b * (1.0 - b) * (1.0 - 2.0 * b) / (12.0*a*a);
    r += - b * b * (1.0-b) * (1.0 - b) / (12.0*a*a*a);

    r
}

fn beta_negint(a: f64, b: f64) -> f64 {
    let b_is_int = b == b.trunc();

    if b_is_int && 1.0 - a - b > 0.0 {
        let b_int = b as i64;
        let sgn = if b_int % 2 == 0 {1.0} else {-1.0};
        sgn * beta(1.0 - a - b, b)
    } else {
        INFINITY
    }
}


// fn lbeta_negint(a: f64, b: f64) -> f64 {
//     let b_is_int = b == b.trunc();
//     if b_is_int && 1.0 - a - b > 0.0 {
//         betaln(1.0 - a - b, b)
//     } else {
//         INFINITY
//     }
// }


#[cfg(test)]
mod tests {
    use super::*;
    // gamma function
    // --------------
    #[test]
    fn beta_1_1_should_be_1() {
        assert_relative_eq!(beta(1.0, 1.0), 1.0, epsilon = 1E-10);
    }

    #[test]
    fn beta_neg1_1_should_be_neg1() {
        assert_relative_eq!(beta(-1.0, 1.0), -1.0, epsilon = 1E-10);
    }

    #[test]
    fn beta_1_neg1_should_be_neg1() {
        assert_relative_eq!(beta(1.0, -1.0), -1.0, epsilon = 1E-10);
    }

    #[test]
    fn beta_neg_value_check_1() {
        assert_relative_eq!(beta(-1.5, 3.0), 5.3333333333333339, epsilon = 1E-10);
    }

    #[test]
    fn beta_large_a_small_b_check_1() {
        assert_relative_eq!(beta(180.0, 0.5), 0.13220268535239055, max_relative = 1E-6);
    }

    #[test]
    fn beta_small_a_large_b_check_1() {
        assert_relative_eq!(beta(0.5, 180.0), 0.13220268535239055, max_relative = 1E-6);
    }

    #[test]
    fn beta_ab_sum_large_check_1() {
        assert_relative_eq!(beta(90.0, 90.0), 2.4416737907555258e-55, max_relative = 1E-6);
    }

    #[test]
    fn beta_moderate_values_check_1() {
        assert_relative_eq!(beta(5.0, 5.0), 0.0015873015873015873, max_relative = 1E-6);
    }

    #[test]
    fn beta_moderate_values_check_2() {
        assert_relative_eq!(beta(5.0, 2.0), 1.0/30.0, max_relative = 1E-6);
    }

    #[test]
    fn beta_moderate_values_check_3() {
        assert_relative_eq!(beta(2.0, 5.0), 1.0/30.0, max_relative = 1E-6);
    }

}
