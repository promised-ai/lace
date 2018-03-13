/// Evaluate the polynomal defined in `coeffs` at point `x`.
pub fn poly_eval(x: f64, coeffs: &[f64]) -> f64 {
    coeffs.iter().fold(0.0, |acc, c| acc * x + c)
}

/// Evaluate the polynomal defined in `coeffs` at point `x` without the
/// scalar (degree 0) term
pub fn poly_eval_nsc(x: f64, coeffs: &[f64]) -> f64 {
    let (first, rest) = coeffs.split_first().unwrap();
    rest.iter().fold(x + first, |acc, c| acc * x + c)
}
