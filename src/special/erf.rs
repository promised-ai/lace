use std::f64;

const SQRT_PI: f64 = 1.772453850905515881919427556567825376987457275391;


// Approximation from wikipedia:
// https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
pub fn erf(x: f64) -> f64 {
    let x_is_negative = x < 0.0;

    let a1 = 0.0705230784;
    let a2 = 0.0422820123;
    let a3 = 0.0092705272;
    let a4 = 0.0001520143;
    let a5 = 0.0002765672;
    let a6 = 0.0000430638;
    let x1 = x.abs();
    let x2 = x1*x1;
    let x3 = x2*x1;
    let x4 = x3*x1;
    let x5 = x4*x1;
    let x6 = x5*x1;

    let denom = (1.0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5 + a6*x6).powi(16);
    let abs_erf = 1.0 - 1.0 / denom;

    if x_is_negative {
        -abs_erf
    } else {
        abs_erf
    }
}


// Code translated from C:
// https://scistatcalc.blogspot.com/2013/09/numerical-estimate-of-inverse-error.html
pub fn erfinv(z: f64) -> f64 {
      let mut w: f64 = -((1.0 - z) * (1.0 + z)).ln();
      let mut p: f64;
    
    if w < 5.0 {
        w -= 2.5;
        p = 2.81022636e-08;
        p = 3.43273939e-07 + p * w;
        p = -3.5233877e-06 + p * w;
        p = -4.39150654e-06 + p * w;
        p = 0.00021858087 + p * w;
        p = -0.00125372503 + p * w;
        p = -0.00417768164 + p * w;
        p = 0.246640727 + p * w;
        p = 1.50140941 + p * w;
    } else {
        w = w.sqrt() - 3.0;
        p =  -0.000200214257;
        p = 0.000100950558 + p * w;
        p = 0.00134934322 + p * w;
        p = -0.00367342844 + p * w;
        p = 0.00573950773 + p * w;
        p = -0.0076224613 + p * w;
        p = 0.00943887047 + p * w;
        p = 1.00167406 + p * w;
        p = 2.83297682 + p * w;
    }
    
    let res_ra = p * z; // assign to rational estimate variable
    
    // Halley's method to refine estimate of inverse erf
    let fx = erf(res_ra) - z;
    let df = 2.0/SQRT_PI * (-(res_ra * res_ra)).exp();
    let d2f = -2.0 * res_ra * df;
    
    res_ra - (2.0 * fx * df) / ((2.0 * df * df) - (fx * d2f))
}

#[cfg(test)]
mod tests {
    use super::*;
    // erf (error function)
    // --------------------
    #[test]
    fn erf_of_very_large_value_should_be_1() {
        assert_approx_eq!(erf(10.0), 1.0, 1E-7);
    }

    #[test]
    fn erf_of_very_small_negative_value_should_be_negative_1() {
        assert_approx_eq!(erf(-10.0), -1.0, 1E-7);
    }

    #[test]
    fn erf_of_zero_should_be_zero() {
        assert_approx_eq!(erf(0.0), 0.0, 1E-7);
    }

    #[test]
    fn erf_negative_value_test_1() {
        assert_approx_eq!(erf(-0.25), -0.2763263901682369, 1E-6);
    }

    #[test]
    fn erf_negative_value_test_2() {
        assert_approx_eq!(erf(-1.0), -0.84270079294971478, 1E-6);
    }

    #[test]
    fn erf_positive_value_test_1() {
        assert_approx_eq!(erf(0.446), 0.47178896844758522, 1E-6);
    }

    #[test]
    fn erf_positive_value_test_2() {
        assert_approx_eq!(erf(0.001), 0.0011283787909692363, 1E-6);
    }


    // inverf (inverse error function)
    // -------------------------------
    #[test]
    fn erfinv_of_zero_should_be_zero() {
        assert_approx_eq!(erfinv(0.0), 0.0, 1E-7);
    }

    #[test]
    fn erfinv_positive_value_test_1() {
        assert_approx_eq!(erfinv(0.5), 0.47693627620446982, 1E-6);
    }

    #[test]
    fn erfinv_positive_value_test_2() {
        assert_approx_eq!(erfinv(0.121), 0.10764782605515244, 1E-6);
    }

    #[test]
    fn erfinv_negative_value_test_1() {
        assert_approx_eq!(erfinv(-0.99), -1.8213863677184492, 1E-5);
    }

    #[test]
    fn erfinv_negative_value_test_2() {
        assert_approx_eq!(erfinv(-0.999), -2.3267537655135242, 1E-4);
    }
}
