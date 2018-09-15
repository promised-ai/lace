extern crate rv;

use self::rv::dist::Mixture;
use self::rv::traits::{Cdf, Rv};

#[inline]
fn pit_quad_lower_part(a: f64, b: f64, f: f64) -> f64 {
    let q =
        0.5 * f * (a.powi(2) - b.powi(2)) + 1.0 / 3.0 * (b.powi(3) - a.powi(3));
    assert!(q > 0.0);
    q
}

#[inline]
fn pit_quad_upper_part(a: f64, b: f64, f: f64) -> f64 {
    let q =
        0.5 * f * (b.powi(2) - a.powi(2)) + 1.0 / 3.0 * (a.powi(3) - b.powi(3));
    assert!(q > 0.0);
    q
}

#[inline]
fn pit_quad_lower_part_area(a: f64, b: f64, f: f64) -> f64 {
    let q = f * (a - b) + 0.5 * (b.powi(2) - a.powi(2));
    assert!(q > 0.0);
    q
}

#[inline]
fn pit_quad_upper_part_area(a: f64, b: f64, f: f64) -> f64 {
    let q = f * (b - a) + 0.5 * (a.powi(2) - b.powi(2));
    assert!(q > 0.0);
    q
}

fn pit_area_quad_prtl(a: f64, b: f64, f: f64) -> f64 {
    if f < a {
        pit_quad_lower_part_area(a, b, f)
    } else if f > b {
        pit_quad_upper_part_area(a, b, f)
    } else {
        pit_quad_lower_part_area(a, f, f) + pit_quad_upper_part_area(f, b, f)
    }
}

fn pit_quad_prtl(a: f64, b: f64, f: f64) -> f64 {
    if f < a {
        pit_quad_lower_part(a, b, f)
    } else if f > b {
        pit_quad_upper_part(a, b, f)
    } else {
        pit_quad_lower_part(a, f, f) + pit_quad_upper_part(f, b, f)
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

/// PIT
///
/// Returns a tuple containing the PIT error and the error's centroid.
pub fn pit<X, Fx>(xs: &Vec<X>, fx: &Mixture<Fx>) -> (f64, f64)
where
    Fx: Rv<X> + Cdf<X>,
{
    let mut ps: Vec<f64> = xs.iter().map(|x| fx.cdf(x)).collect();
    ps.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let nf = ps.len() as f64;

    let mut a: f64 = 0.0;
    let area = ps.iter().enumerate().fold(0.0, |acc, (ix, &b)| {
        let f = (ix as f64) / nf;
        let q = pit_area_quad_prtl(a, b, f);
        a = b;
        acc + q
    });

    let quad = ps.iter().enumerate().fold(0.0, |acc, (ix, &b)| {
        let f = (ix as f64) / nf;
        let q = pit_quad_prtl(a, b, f);
        a = b;
        acc + q
    });

    let centroid = quad / area;
    let error = 2.0 * quad; // should be in [0, 1]

    (error, centroid)
}
