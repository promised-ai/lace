extern crate num;
use self::num::Float;
use misc::{argmin, sign};

pub enum Method {
    Combo,
    Bounded,
    BruteForce,
}

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

    let golden_mean = 0.5 * (3.0 - 5.0.sqrt());
    let sqrt_eps = 2.2E-16.sqrt();
    let (mut a, mut b) = bounds;
    let mut fulc = a + golden_mean * (b - a);
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
    let mut tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
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
        x = xf + si * rat.abs().max(tol1);
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

            if (fu <= fnfc) || (nfc == xf) {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if (fu <= ffulc) || (fulc == xf) || (fulc == nfc) {
                fulc = x;
                ffulc = fu;
            }
        }

        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
        tol2 = 2.0 * tol1;

        if num >= maxfun {
            break;
        }
    }

    x
}

pub fn fmin_brute<F>(f: F, bounds: (f64, f64), n_grid: usize) -> f64
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
    let fxs: Vec<f64> = (0..n_grid + 1)
        .map(|ix| {
            let x = bounds.0 + step_size * (ix as f64);
            f(x)
        })
        .collect();

    let ix = argmin(&fxs) as f64;

    ix * step_size + bounds.0
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-8;

    // Brute force (grid search)
    // -------------------------
    #[test]
    fn brute_force_min_x_square() {
        let square = |x| x * x;
        let fmin = fmin_brute(square, (-1.0, 1.0), 20);

        assert_relative_eq!(fmin, 0.0, epsilon = TOL);
    }

    #[test]
    fn brute_force_min_x_cubed() {
        let cube = |x| x * x * x;
        let fmin = fmin_brute(cube, (-1.0, 1.0), 20);

        assert_relative_eq!(fmin, -1.0, epsilon = TOL);
    }

    #[test]
    fn brute_force_min_neg_x_cubed() {
        let neg_cube = |x: f64| -x * x * x;
        let fmin = fmin_brute(neg_cube, (-1.0, 1.0), 20);

        assert_relative_eq!(fmin, 1.0, epsilon = TOL);
    }

    #[test]
    fn brute_force_min_neg_gaussian_loglike() {
        let log_pdf = |x: f64| (x - 1.3) * (x - 1.3) / 2.0;
        let fmin = fmin_brute(log_pdf, (0.0, 2.0), 20);

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
}
