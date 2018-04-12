use misc::poly::{poly_eval, poly_eval_nsc};
use std::f64::INFINITY;
use std::f64::consts::PI;

// All functions translated into rust scipy
const LOGPI: f64 = 1.14472988584940017414;
const MAXGAM: f64 = 171.624376956302725;

const P: [f64; 7] = [
    1.60119522476751861407E-4,
    1.19135147006586384913E-3,
    1.04213797561761569935E-2,
    4.76367800457137231464E-2,
    2.07448227648435975150E-1,
    4.94214826801497100753E-1,
    9.99999999999999996796E-1,
];

const Q: [f64; 8] = [
    -2.31581873324120129819E-5,
    5.39605580493303397842E-4,
    -4.45641913851797240494E-3,
    1.18139785222060435552E-2,
    3.58236398605498653373E-2,
    -2.34591795718243348568E-1,
    7.14304917030273074085E-2,
    1.00000000000000000320E0,
];

pub fn gammaln(z: f64) -> f64 {
    gammaln_sign(z).0
}

pub fn gamma(mut x: f64) -> f64 {
    let mut sgngam: f64 = 1.0;
    let mut z: f64;

    if !x.is_normal() {
        return x;
    }

    let mut q = x.abs();

    if q > 33.0 {
        if x < 0.0 {
            let mut p = q.trunc();
            if p == q {
                return INFINITY;
            }
            let i = p as u64;
            if (i & 1) == 0 {
                sgngam = -1.0;
            }
            z = q - p;
            if z > 0.5 {
                p += 1.0;
                z = q - p;
            }
            z = q * (PI * z).sin();
            if z == 0.0 {
                return sgngam * INFINITY;
            }
            z = z.abs();
            z = PI / (z * stirf(q));
        } else {
            z = stirf(x);
        }
        return sgngam * z;
    }

    z = 1.0;
    while x >= 3.0 {
        x -= 1.0;
        z *= x;
    }

    while x < 0.0 {
        if x > -1.0E-9 {
            return gamma_small(x, z);
        }
        z /= x;
        x += 1.0;
    }

    while x < 2.0 {
        if x < 1.0E-9 {
            return gamma_small(x, z);
        }
        z /= x;
        x += 1.0;
    }

    if x == 2.0 {
        return z;
    }

    x -= 2.0;
    let p = poly_eval(x, &P);
    q = poly_eval(x, &Q);
    z * p / q
}

fn gamma_small(x: f64, z: f64) -> f64 {
    if x == 0.0 {
        INFINITY
    } else {
        z / ((1.0 + 0.5772156649015329 * x) * x)
    }
}

const A: [f64; 5] = [
    8.11614167470508450300E-4,
    -5.95061904284301438324E-4,
    7.93650340457716943945E-4,
    -2.77777777730099687205E-3,
    8.33333333333331927722E-2,
];

const B: [f64; 6] = [
    -1.37825152569120859100E3,
    -3.88016315134637840924E4,
    -3.31612992738871184744E5,
    -1.16237097492762307383E6,
    -1.72173700820839662146E6,
    -8.53555664245765465627E5,
];

const C: [f64; 6] = [
    -3.51815701436523470549E2,
    -1.70642106651881159223E4,
    -2.20528590553854454839E5,
    -1.13933444367982507207E6,
    -2.53252307177582951285E6,
    -2.01889141433532773231E6,
];

const LS2PI: f64 = 0.91893853320467274178;

const MAXLGM: f64 = 2.556348e305;

pub fn gammaln_sign(mut x: f64) -> (f64, f64) {
    let mut sign = 1.0;

    if !x.is_normal() {
        return (x, sign);
    }

    if x < -34.0 {
        let q = -x;
        let (w, mut sign) = gammaln_sign(q);
        let mut p = q.trunc();
        if p == q {
            return (INFINITY, sign);
        }

        let i = p as i64;

        if (i & 1) == 0 {
            sign = -1.0;
        } else {
            sign = 1.0;
        }

        let mut z = q - p;

        if z > 0.5 {
            p += 1.0;
            z = p - q;
        }

        z = q * (PI * z).sin();
        if z == 0.0 {
            return (INFINITY, sign);
        }

        z = LOGPI - z.ln() - w;
        return (z, sign);
    }

    if x < 13.0 {
        let mut z = 1.0;
        let mut p = 0.0;
        let mut u = x;
        while u >= 3.0 {
            p -= 1.0;
            u = x + p;
            z *= u;
        }
        while u < 2.0 {
            if u == 0.0 {
                return (INFINITY, sign);
            }
            z /= u;
            p += 1.0;
            u = x + p;
        }
        if z < 0.0 {
            sign = -1.0;
            z = -z;
        } else {
            sign = 1.0;
        }

        if u == 2.0 {
            return (z.ln(), sign);
        }
        p -= 2.0;
        x += p;
        p = x * poly_eval(x, &B) / poly_eval_nsc(x, &C);
        return (z.ln() + p, sign);
    }

    if x > MAXLGM {
        return (sign * INFINITY, sign);
    }

    let mut q = (x - 0.5) * x.ln() - x + LS2PI;
    if x > 1.0e8 {
        return (q, sign);
    }

    let p = 1.0 / (x * x);
    if x >= 1000.0 {
        q += ((7.9365079365079365079365e-4 * p - 2.7777777777777777777778e-3)
            * p + 0.0833333333333333333333) / x;
    } else {
        q += poly_eval(p, &A) / x;
    }
    (q, sign)
}

const STIR: [f64; 5] = [
    7.87311395793093628397E-4,
    -2.29549961613378126380E-4,
    -2.68132617805781232825E-3,
    3.47222221605458667310E-3,
    8.33333333333482257126E-2,
];

const MAXSTIR: f64 = 143.01608;

const SQRT_PI: f64 = 1.7724538509055159;

// Stirling's approximation for gamma function
fn stirf(x: f64) -> f64 {
    if x >= MAXGAM {
        return INFINITY;
    }

    let mut y = x.exp();
    let mut w = 1.0 / x;

    w = 1.0 + w * poly_eval(w, &STIR);

    if x > MAXSTIR {
        /* Avoid overflow in pow() */
        let v = x.powf(0.5 * x - 0.25);
        y = v * (v / y);
    } else {
        y = x.powf(x - 0.5) / y;
    }
    SQRT_PI * y * w
}

#[cfg(test)]
mod tests {
    use super::*;
    // gamma function
    // --------------
    #[test]
    fn gamma_1_should_be_1() {
        assert_relative_eq!(gamma(1.0), 1.0, epsilon = 1E-10);
    }

    #[test]
    fn gamma_2_should_be_1() {
        assert_relative_eq!(gamma(2.0), 1.0, epsilon = 1E-10);
    }

    #[test]
    fn gamma_small_value_test_1() {
        assert_relative_eq!(gamma(0.00099), 1009.52477271, epsilon = 1E-3);
    }

    #[test]
    fn gamma_small_value_test_2() {
        assert_relative_eq!(gamma(0.00100), 999.423772485, epsilon = 1E-4);
    }

    #[test]
    fn gamma_small_value_test_3() {
        assert_relative_eq!(gamma(0.00101), 989.522792258, epsilon = 1E-4);
    }

    #[test]
    fn gamma_medium_value_test_1() {
        assert_relative_eq!(gamma(6.1), 142.451944066, epsilon = 1E-6);
    }

    #[test]
    fn gamma_medium_value_test_2() {
        assert_relative_eq!(gamma(11.999), 39819417.4793, epsilon = 1E-3);
    }

    #[test]
    fn gamma_large_value_test_1() {
        assert_relative_eq!(gamma(12.0), 39916800.0, epsilon = 1E-3);
    }

    #[test]
    fn gamma_large_value_test_2() {
        assert_relative_eq!(gamma(12.001), 40014424.1571, epsilon = 1E-3);
    }

    #[test]
    fn gamma_large_value_test_3() {
        assert_relative_eq!(gamma(15.2), 149037380723.0, epsilon = 1.0);
    }

    // gammaln (log gamma)
    // -------------------
    #[test]
    fn gammaln_small_value_test_1() {
        assert_relative_eq!(
            gammaln(0.9999),
            5.77297915613e-05,
            epsilon = 1E-5
        );
    }

    #[test]
    fn gammaln_small_value_test_2() {
        assert_relative_eq!(
            gammaln(1.0001),
            -5.77133422205e-05,
            epsilon = 1E-5
        );
    }

    #[test]
    fn gammaln_medium_value_test_1() {
        assert_relative_eq!(gammaln(3.1), 0.787375083274, epsilon = 1E-5);
    }

    #[test]
    fn gammaln_medium_value_test_2() {
        assert_relative_eq!(gammaln(6.3), 5.30734288962, epsilon = 1E-5);
    }

    #[test]
    fn gammaln_medium_value_test_3() {
        assert_relative_eq!(gammaln(11.9999), 17.5020635801, epsilon = 1E-5);
    }

    #[test]
    fn gammaln_large_value_test_1() {
        assert_relative_eq!(gammaln(12.0), 17.5023078459, epsilon = 1E-5);
    }

    #[test]
    fn gammaln_large_value_test_2() {
        assert_relative_eq!(gammaln(12.0001), 17.5025521125, epsilon = 1E-5);
    }

    #[test]
    fn gammaln_large_value_test_3() {
        assert_relative_eq!(gammaln(27.4), 62.5755868211, epsilon = 1E-5);
    }
}
