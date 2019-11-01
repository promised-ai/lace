/// The mean of a vector of f64
pub fn mean(xs: &[f64]) -> f64 {
    let n: f64 = xs.len() as f64;
    xs.iter().fold(0.0, |acc, x| x + acc) / n
}

/// The variance of a vector of f64
pub fn var(xs: &[f64]) -> f64 {
    let n: f64 = xs.len() as f64;
    let m = mean(xs);
    let v = xs.iter().fold(0.0, |acc, x| (x - m).mul_add(x - m, acc));
    // TODO: Add dof and return 0 if n == 1
    v / n
}

/// The standard deviation of a vector of f64
pub fn std(xs: &[f64]) -> f64 {
    let v: f64 = var(xs);
    v.sqrt()
}

#[cfg(test)]
mod tests {
    extern crate approx;
    use super::*;
    use approx::*;

    // mean
    // ----
    #[test]
    fn mean_1() {
        let xs: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(mean(&xs), 2.0, epsilon = 10E-10);
    }

    #[test]
    fn mean_2() {
        let xs: Vec<f64> = vec![1.0 / 3.0, 2.0 / 3.0, 5.0 / 8.0, 11.0 / 12.0];
        assert_relative_eq!(mean(&xs), 0.63541666666666663, epsilon = 10E-8);
    }

    // var
    // ---
    #[test]
    fn var_1() {
        let xs: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(var(&xs), 2.0, epsilon = 10E-10);
    }

    #[test]
    fn var_2() {
        let xs: Vec<f64> = vec![1.0 / 3.0, 2.0 / 3.0, 5.0 / 8.0, 11.0 / 12.0];
        assert_relative_eq!(var(&xs), 0.04286024305555555, epsilon = 10E-8);
    }

    // FIXME: std tests
}
