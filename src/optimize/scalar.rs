use braid_utils::{argmin, sign};
use num::Float;

/// The method by which to optimize
pub enum Method {
    Combo,
    Bounded,
    BruteForce,
}

#[derive(Debug, Clone)]
pub struct GradientDescentResult {
    /// The min point
    pub x: f64,
    /// The rrror
    pub err: f64,
    /// The number of iterations performed
    pub iters: usize,
}

#[derive(Debug, Clone)]
pub struct GradientDescentParams {
    pub learning_rate: f64,
    pub momentum: f64,
    pub error_tol: f64,
    pub max_iters: usize,
}

impl Default for GradientDescentParams {
    fn default() -> Self {
        GradientDescentParams {
            learning_rate: 0.01,
            momentum: 0.5,
            error_tol: 1E-6,
            max_iters: 100,
        }
    }
}

/// Newton's zero-finding method
///
/// # Arguments
/// - f_dprime: a function that returns a tuple containing the first and
///   second derivative at a point x.
/// - x0: The inital guess.
/// - errtol_opt: Minimum relative error between x movements before stopping.
/// - maxiters_opt: Maximum number of iterations to perform
pub fn newton<F>(
    f_dprime: F,
    x0: f64,
    errtol_opt: Option<f64>,
    maxiters_opt: Option<usize>,
) -> f64
where
    F: Fn(f64) -> (f64, f64),
{
    let mut x = x0;
    let mut iters = 1;
    let max_iters = maxiters_opt.unwrap_or(100);
    let err_tol = errtol_opt.unwrap_or(1E-8);

    loop {
        let (fpr, fdr) = f_dprime(x);
        let xt = x - fpr / fdr;

        if relative_err(xt, x) < err_tol || iters == max_iters {
            return xt;
        }

        x = xt;
        iters += 1;
    }
}

/// Gradient descent with momentum
///
/// # Arguments
/// - f_prime: Computes the derivative at point x.
/// - x0: The inital guess.
/// - params: Gradient descent parameters
pub fn gradient_descent<F>(
    f_prime: F,
    x0: f64,
    params: GradientDescentParams,
) -> GradientDescentResult
where
    F: Fn(f64) -> f64,
{
    let mut x = x0;

    let GradientDescentParams {
        learning_rate,
        momentum,
        error_tol,
        max_iters,
    } = params;

    let mut v: f64 = 0.0;
    let mut iters: usize = 0;

    loop {
        v = momentum * v - learning_rate * f_prime(momentum.mul_add(v, x));
        let xt = x + v;

        iters += 1;
        let err = (xt - x).abs();
        if err < error_tol || iters >= max_iters {
            let res = GradientDescentResult { err, iters, x: xt };
            return res;
        }

        x = xt;
    }
}

#[inline]
fn relative_err(x_true: f64, x_est: f64) -> f64 {
    (1.0 - x_est / x_true).abs()
}

/// Scalar optimization function ported from Scipy's `fmindbound`.
///
/// # Notes
///
/// Doesn't work when there are spread out modes; will bias to the center mode.
#[allow(clippy::many_single_char_names)]
pub fn fmin_bounded<F>(
    f: F,
    bounds: (f64, f64),
    xatol_opt: Option<f64>,
    maxiter_opt: Option<usize>,
) -> f64
where
    F: Fn(f64) -> f64,
{
    let xatol = xatol_opt.unwrap_or(1.0E-5);
    let maxiter = maxiter_opt.unwrap_or(500);
    let maxfun: usize = maxiter;
    // Test bounds are of correct form
    let (x1, x2) = bounds;
    if x1 >= x2 {
        panic!("Lower bound ({}) exceeds upper ({}).", bounds.0, bounds.1);
    }

    let golden_mean: f64 = 0.5 * (3.0 - 5.0.sqrt());
    let sqrt_eps = 2.2E-16.sqrt();
    let (mut a, mut b) = bounds;
    let mut fulc = golden_mean.mul_add(b - a, a);
    let (mut nfc, mut xf) = (fulc, fulc);
    let mut rat = 0.0;
    let mut e = 0.0;
    let mut x = xf;
    let mut fx = f(x);
    let mut num = 1;
    // let mut fmin_data = (1, xf, fx);

    let mut ffulc = fx;
    let mut fnfc = fx;
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps.mul_add(xf.abs(), xatol / 3.0);
    let mut tol2 = 2.0 * tol1;

    while (xf - xm).abs() > (tol2 - 0.5 * (b - a)) {
        let mut golden = true;
        // Check for parabolic fit
        if e.abs() > tol1 {
            golden = false;
            let mut r = (xf - nfc) * (fx - ffulc);
            let q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            let mut q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            r = e;
            e = rat;

            // Check for acceptability of parabola
            if (p.abs() < (0.5 * q * r).abs())
                && (p > q * (a - xf))
                && (p < q * (b - xf))
            {
                rat = p / q;
                x = xf + rat;

                if ((x - a) < tol2) || ((b - x) < tol2) {
                    let si = sign(xm - xf) + {
                        if (xm - xf) == 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    };
                    rat = tol1 * si;
                }
            } else {
                // do a golden section step
                golden = true;
            }
        }

        if golden {
            // Do a golden-section step
            if xf >= xm {
                e = a - xf
            } else {
                e = b - xf;
            }
            rat = golden_mean * e;
        }

        let si = sign(rat) + {
            if rat == 0.0 {
                1.0
            } else {
                0.0
            }
        };
        x = si.mul_add(rat.abs().max(tol1), xf);
        let fu = f(x);
        num += 1;
        // fmin_data = (num, x, fu);

        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }

            if (fu <= fnfc) || (nfc - xf).abs() < std::f64::EPSILON {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if (fu <= ffulc)
                || (fulc - xf).abs() < std::f64::EPSILON
                || (fulc - nfc).abs() < std::f64::EPSILON
            {
                fulc = x;
                ffulc = fu;
            }
        }

        xm = 0.5 * (a + b);
        tol1 = sqrt_eps.mul_add(xf.abs(), xatol / 3.0);
        tol2 = 2.0 * tol1;

        if num >= maxfun {
            break;
        }
    }

    x
}

/// Find the min point in an interval by grid search.
pub fn fmin_brute<F>(f: &F, bounds: (f64, f64), n_grid: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    if bounds.0 >= bounds.1 {
        panic!(
            "lower bound ({}) exceeds upper bound ({})",
            bounds.0, bounds.1
        )
    }
    let step_size = (bounds.1 - bounds.0) / (n_grid as f64);
    let fxs: Vec<f64> = (0..=n_grid)
        .map(|ix| {
            let x = step_size.mul_add(ix as f64, bounds.0);
            f(x)
        })
        .collect();

    let ix = argmin(&fxs) as f64;

    ix.mul_add(step_size, bounds.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    const TOL: f64 = 1E-8;

    // Brute force (grid search)
    // -------------------------
    #[test]
    fn brute_force_min_x_square() {
        let square = |x| x * x;
        let fmin = fmin_brute(&square, (-1.0, 1.0), 20);

        assert_relative_eq!(fmin, 0.0, epsilon = TOL);
    }

    #[test]
    fn brute_force_min_x_cubed() {
        let cube = |x| x * x * x;
        let fmin = fmin_brute(&cube, (-1.0, 1.0), 20);

        assert_relative_eq!(fmin, -1.0, epsilon = TOL);
    }

    #[test]
    fn brute_force_min_neg_x_cubed() {
        let neg_cube = |x: f64| -x * x * x;
        let fmin = fmin_brute(&neg_cube, (-1.0, 1.0), 20);

        assert_relative_eq!(fmin, 1.0, epsilon = TOL);
    }

    #[test]
    fn brute_force_min_neg_gaussian_loglike() {
        let log_pdf = |x: f64| (x - 1.3) * (x - 1.3) / 2.0;
        let fmin = fmin_brute(&log_pdf, (0.0, 2.0), 20);

        assert_relative_eq!(fmin, 1.3, epsilon = 0.1);
    }

    // Bounded
    // -------
    #[test]
    fn brounded_min_x_square() {
        let square = |x| x * x;
        let fmin = fmin_bounded(square, (-1.0, 1.0), None, None);

        assert_relative_eq!(fmin, 0.0, epsilon = 10E-5);
    }

    #[test]
    fn bounded_min_x_cubed() {
        let cube = |x| x * x * x;
        let fmin = fmin_bounded(cube, (-1.0, 1.0), None, None);

        assert_relative_eq!(fmin, -1.0, epsilon = 10E-5);
    }

    #[test]
    fn bounded_min_neg_x_cubed() {
        let neg_cube = |x: f64| -x * x * x;
        let fmin = fmin_bounded(neg_cube, (-1.0, 1.0), None, None);

        assert_relative_eq!(fmin, 1.0, epsilon = 10E-5);
    }

    #[test]
    fn bounded_min_neg_gaussian_loglike() {
        let log_pdf = |x: f64| (x - 1.3) * (x - 1.3) / 2.0;
        let fmin = fmin_bounded(log_pdf, (0.0, 2.0), None, None);

        assert_relative_eq!(fmin, 1.3, epsilon = 10E-5);
    }

    #[test]
    fn bounded_should_find_global_min() {
        // set up function with two mins
        fn f(x: f64) -> f64 {
            -0.4 * (-x * x / 2.0).exp()
                - 0.6 * (-(x - 3.0) * (x - 3.0) / 2.0).exp()
        }
        let xf = fmin_bounded(f, (0.0, 3.0), None, None);
        assert_relative_eq!(xf, 2.9763354969615476, epsilon = 10E-5);
    }

    // Gradient Descent
    #[test]
    fn gradient_descent_fn1() {
        let f_prime = |x: f64| 4.0 * x.powi(3) - 9.0 * x * x;

        let params = GradientDescentParams {
            learning_rate: 0.01,
            momentum: 0.1,
            max_iters: 100,
            ..Default::default()
        };
        let fmin = gradient_descent(f_prime, 0.5, params);
        assert_relative_eq!(fmin.x, 9.0 / 4.0, epsilon = 10E-5);
    }

    // Newton's Method
    #[test]
    fn newton_fn1() {
        let f_dprime = |x: f64| {
            let r = 4.0 * x.powi(3) - 9.0 * x * x;
            let rr = 12.0 * x.powi(2) - 18.0 * x;
            (r, rr)
        };

        let fmin = newton(f_dprime, 2.0, None, None);
        assert_relative_eq!(fmin, 9.0 / 4.0, epsilon = 10E-5);
    }

    #[test]
    fn newton_gaussian() {
        let mu = 0.233;
        let sigma = 1.5;
        let sqrt2pi = (2.0 * std::f64::consts::PI).sqrt();
        let f_dprime = |x: f64| {
            let k = (-0.5 * ((x - mu) / sigma).powi(2)).exp();
            let r = -(x - mu) / sigma.powi(3) / sqrt2pi * k;
            let rr =
                -(mu + sigma - x) * (sigma + x - mu) / sigma.powi(5) / sqrt2pi
                    * k;
            (r, rr)
        };

        let fmin = newton(f_dprime, 1.0, None, None);
        assert_relative_eq!(fmin, 0.233, epsilon = 10E-5);
    }
}
