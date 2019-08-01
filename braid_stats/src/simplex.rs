//! Utilities for creating and dealing with points on a Simplex
use serde::{Deserialize, Serialize};

/// A point on the N-Simplex
#[derive(Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct SimplexPoint(Vec<f64>);

/// Describes invalid simplex points
#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub enum SimplexPointError {
    /// An item in the coordinate vector was not finite and positive.
    InvalidCoordinateError,
    /// The items in the coordinate vector do not sum to 1.
    DoesNotSumToOneError,
}

impl SimplexPoint {
    /// Create a new SimplexPoint
    ///
    /// # Example
    ///
    /// ```
    /// # use braid_stats::simplex::SimplexPoint;
    /// # use braid_stats::simplex::SimplexPointError;
    /// assert!(SimplexPoint::new(vec![0.5, 0.5]).is_ok());
    /// assert_eq!(
    ///     SimplexPoint::new(vec![0.5, 0.1]).unwrap_err(),
    ///     SimplexPointError::DoesNotSumToOneError,
    /// );
    /// assert_eq!(
    ///     SimplexPoint::new(vec![0.5, 0.6, -0.1]).unwrap_err(),
    ///     SimplexPointError::InvalidCoordinateError,
    /// );
    /// assert_eq!(
    ///     SimplexPoint::new(vec![0.5, std::f64::INFINITY]).unwrap_err(),
    ///     SimplexPointError::InvalidCoordinateError,
    /// );
    /// ```
    pub fn new(point: Vec<f64>) -> Result<Self, SimplexPointError> {
        if !point.iter().all(|&coord| coord.is_finite() && coord >= 0.0) {
            Err(SimplexPointError::InvalidCoordinateError)
        } else if (point.iter().sum::<f64>() - 1.0).abs() > 1e-10 {
            Err(SimplexPointError::DoesNotSumToOneError)
        } else {
            Ok(SimplexPoint(point))
        }
    }

    /// Create a new SimplexPoint and to hell with validity
    pub fn new_unchecked(point: Vec<f64>) -> Self {
        SimplexPoint(point)
    }

    /// Get a reference to the coordinate vector
    pub fn point(&self) -> &Vec<f64> {
        &self.0
    }

    /// Get the number of dimensions
    ///
    /// # Example
    ///
    /// ```
    /// # use braid_stats::simplex::SimplexPoint;
    /// let point = SimplexPoint::new(vec![0.5, 0.1, 0.4]).unwrap();
    /// assert_eq!(point.ndims(), 3);
    /// ```
    pub fn ndims(&self) -> usize {
        self.0.len()
    }
}

/// Convert a N-length vector xs = {x<sub>1</sub>, ..., x<sub>n</sub> :
/// x<sub>i</sub> ~ U(0, 1)} to a point on the N-simplex
///
/// # Example
///
/// Generate 100 quasi-random points on the 3-simplex
///
/// ```
/// # use braid_stats::seq::SobolSeq;
/// # use braid_stats::simplex::uvec_to_simplex;
/// SobolSeq::new(3)
///     .take(100)
///     .map(|uvec| uvec_to_simplex(uvec))
///     .for_each(|pt| {
///         assert_eq!(pt.ndims(), 3);
///         assert!( (pt.point().iter().sum::<f64>() - 1.0).abs() < 1e-6 );
///     })
/// ```
pub fn uvec_to_simplex(mut uvec: Vec<f64>) -> SimplexPoint {
    let n = uvec.len();
    uvec[n - 1] = 1.0;
    uvec.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut um = uvec[0];

    for i in 1..n {
        let diff = uvec[i] - um;
        um = uvec[i];
        uvec[i] = diff;
    }
    SimplexPoint::new_unchecked(uvec)
}
