/// The mean of a vector of f64
pub fn mean(xs: &[f64]) -> f64 {
    let n: f64 = xs.len() as f64;
    xs.iter().sum::<f64>() / n
}

/// The variance of a vector of f64
pub fn var(xs: &[f64]) -> f64 {
    let n: f64 = xs.len() as f64;
    let m = mean(xs);
    let v = xs.iter().fold(0.0, |acc, x| (x - m).mul_add(x - m, acc));
    // TODO: Add dof and return 0 if n == 1
    v / n
}

/// Compute the mean and variance faster than by calling mean and var separately
pub fn mean_var(xs: &[f64]) -> (f64, f64) {
    let n: f64 = xs.len() as f64;
    let m = mean(xs);
    let v = xs.iter().fold(0.0, |acc, x| (x - m).mul_add(x - m, acc));
    // TODO: Add dof and return 0 if n == 1
    (m, v / n)
}

/// The standard deviation of a vector of f64
pub fn std(xs: &[f64]) -> f64 {
    let v: f64 = var(xs);
    v.sqrt()
}

#[cfg(test)]
mod tests {
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
        assert_relative_eq!(
            mean(&xs),
            0.635_416_666_666_666_6,
            epsilon = 10E-8
        );
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
        assert_relative_eq!(var(&xs), 0.042_860_243_055_555_6, epsilon = 10E-8);
    }

    #[test]
    fn std_1() {
        let xs: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(std(&xs), 2_f64.sqrt(), epsilon = 1E-10);
    }

    #[test]
    fn std_2() {
        let xs: Vec<f64> = vec![1.0 / 3.0, 2.0 / 3.0, 5.0 / 8.0, 11.0 / 12.0];
        assert_relative_eq!(std(&xs), var(&xs).sqrt(), epsilon = 1E-10);
    }
}
