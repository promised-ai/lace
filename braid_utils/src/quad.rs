#[derive(Debug, Clone)]
pub struct QuadConfig<'a> {
    pub max_depth: u32,
    pub err_tol: f64,
    pub seed_points: Option<&'a Vec<f64>>,
}

impl<'a> Default for QuadConfig<'a> {
    fn default() -> Self {
        QuadConfig {
            max_depth: 12,
            err_tol: 1e-16,
            seed_points: None,
        }
    }
}

#[inline]
fn simpsons_rule<F>(
    func: &F,
    a: f64,
    fa: f64,
    b: f64,
    fb: f64,
) -> (f64, f64, f64)
where
    F: Fn(f64) -> f64,
{
    let c = (a + b) / 2.0;
    let h3 = (b - a).abs() / 6.0;
    let fc = func(c);
    (c, fc, h3 * (fa + 4.0 * fc + fb))
}

// Quad recursion step
//
// # Notes:
//
// Variable name conventions:
// - a: lower bound on interval
// - b: upper bound on interval
// - m: mid point of interval
// - fa, fm, fb: function values at interval
// - err: cumulative error
// - whole: cumulative integral
// - depth: how many recursions so far
// - max_depth: max depth before just returning what we have
#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
fn quad_recr<F>(
    func: &F,
    a: f64,
    fa: f64,
    m: f64,
    fm: f64,
    b: f64,
    fb: f64,
    err: f64,
    whole: f64,
    depth: u32,
    max_depth: u32,
) -> f64
where
    F: Fn(f64) -> f64,
{
    let (ml, fml, left) = simpsons_rule(&func, a, fa, m, fm);
    let (mr, fmr, right) = simpsons_rule(&func, m, fm, b, fb);
    let eps = left + right - whole;
    if eps.abs() <= 15.0 * err || depth == max_depth {
        left + right + eps / 15.0
    } else {
        let half_err = err / 2.0;
        let next_depth = depth + 1;
        quad_recr(
            func, a, fa, ml, fml, m, fm, half_err, left, next_depth, max_depth,
        ) + quad_recr(
            func, m, fm, mr, fmr, b, fb, half_err, right, next_depth, max_depth,
        )
    }
}

pub fn quadp<F>(f: &F, a: f64, b: f64, config: QuadConfig) -> f64
where
    F: Fn(f64) -> f64,
{
    let default_points = vec![a, (a + b) / 2.0, b];
    let points = match config.seed_points {
        Some(points) => points,
        None => &default_points,
    };

    let tol = config.err_tol / (points.len() + 1) as f64;
    let fa = f(a);

    let (c, fc, res) = points.iter().fold((a, fa, 0.0), |(a, fa, res), &b| {
        let fb = f(b);
        let (m, fm, q) = simpsons_rule(&f, a, fa, b, fb);
        (
            b,
            fb,
            res + quad_recr(
                &f,
                a,
                fa,
                m,
                fm,
                b,
                fb,
                tol,
                q,
                1,
                config.max_depth,
            ),
        )
    });

    let fb = f(b);
    let (m, fm, q) = simpsons_rule(&f, c, fc, b, fb);
    res + quad_recr(&f, c, fc, m, fm, b, fb, tol, q, 1, config.max_depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn quadp_of_x2() {
        let func = |x: f64| x.powi(2);
        let q = quadp(&func, 0.0, 1.0, QuadConfig::default());
        assert!((q - 1.0 / 3.0).abs() <= 1e-15);
    }

    #[test]
    fn quadp_of_sin() {
        let func = |x: f64| x.sin();
        let q = quadp(&func, 0.0, 5.0 * PI, QuadConfig::default());
        assert!((q - 2.0).abs() <= 1e-15);
    }
}
