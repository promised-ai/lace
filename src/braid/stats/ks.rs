pub fn ks_test<F: Fn(f64) -> f64>(xs: &Vec<f64>, cdf: F) -> f64 {
    let mut xs_r: Vec<f64> = xs.clone().to_vec();
    xs_r.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let n: f64 = xs_r.len() as f64;
    xs_r.iter().enumerate().fold(0.0, |acc, (i, &x)| {
        let diff = ((i as f64 + 1.0) / n - cdf(x)).abs();
        if diff > acc {
            diff
        } else {
            acc
        }
    })
}

fn empirical_cdf(xs: &[f64], all_vals: &[f64]) -> Vec<f64> {
    let n: f64 = xs.len() as f64;
    let mut cdf: Vec<f64> = Vec::with_capacity(all_vals.len());
    let mut ix: usize = 0;
    all_vals.iter().for_each(|y| {
        if *y > xs[ix] {
            ix += 1;
        }
        let p = (ix as f64 + 1.0) / n;
        cdf.push(p);
    });
    cdf
}

pub fn ks2sample(mut xs: Vec<f64>, mut ys: Vec<f64>) -> f64 {
    let mut all_vals = xs.clone();
    all_vals.extend(ys.clone());

    xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    ys.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    all_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let cdf_x = empirical_cdf(&xs, &all_vals);
    let cdf_y = empirical_cdf(&ys, &all_vals);

    cdf_x.iter().zip(cdf_y).fold(0.0, |acc, (px, py)| {
        let diff = (px - py).abs();
        if diff > acc {
            diff
        } else {
            acc
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-8;

    #[test]
    fn ks2sample_stat_should_be_zero_when_samples_are_identical() {
        let xs: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let ys: Vec<f64> = vec![2.0, 1.0, 4.0, 3.0];

        assert_relative_eq!(0.0, ks2sample(xs, ys), epsilon = TOL);
    }

    #[test]
    fn ks2sample_stat_simple_value_test_1() {
        let xs: Vec<f64> = vec![1.0, 1.0, 4.0, 4.0];
        let ys: Vec<f64> = vec![1.0, 1.0, 1.0, 4.0];

        assert_relative_eq!(0.25, ks2sample(xs, ys), epsilon = TOL);
    }

    #[test]
    fn ks2sample_stat_simple_value_test_2() {
        let xs: Vec<f64> = vec![0.42, 0.24, 0.86, 0.85, 0.82, 0.82, 0.25, 0.78, 0.13, 0.27];
        let ys: Vec<f64> = vec![0.24, 0.27, 0.87, 0.29, 0.57, 0.44, 0.5, 0.00, 0.56, 0.03];

        assert_relative_eq!(0.4, ks2sample(xs, ys), epsilon = TOL);
    }

}
