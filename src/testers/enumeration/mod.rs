//! Enumeration tests
pub mod state;
pub mod view;

/// Convert a partition with to an integer index by converting a k-length
/// partition into a k-length base-k integer from left to right.
pub fn partition_to_ix(z: &[usize]) -> u64 {
    let k = z.len() as u64;
    z.iter()
        .enumerate()
        .fold(0_u64, |acc, (i, &zi)| acc + (zi as u64) * k.pow(i as u32))
}

/// Adjust the assignment for label switching. The resulting assignment will
/// have partition indices that start at zero; new partition indices are
/// introduced incrementally from left to right (see the tests for more).
#[allow(clippy::needless_range_loop)]
pub fn normalize_assignment(mut z: Vec<usize>) -> Vec<usize> {
    // XXX: I feel like there is a better way to do this, but this works...
    let mut should_be: usize = 0;
    let mut max_is = 0;
    for i in 0..z.len() {
        let is = z[i];
        if is > should_be {
            for j in i..z.len() {
                if z[j] == is {
                    z[j] = should_be;
                } else if z[j] == should_be {
                    z[j] = is;
                }
            }
            max_is = should_be;
            should_be += 1;
        } else {
            if max_is < is {
                max_is = is;
            }
            should_be = max_is + 1;
        }
    }
    z
}

use braid_stats::prior::{Ng, NigHyper};
use rand::Rng;
use rv::{dist::Gaussian, traits::Rv};

use crate::cc::{ColModel, Column, DataContainer};

pub fn build_features<R: Rng>(
    nrows: usize,
    ncols: usize,
    mut rng: &mut R,
) -> Vec<ColModel> {
    let g = Gaussian::standard();
    let prior = Ng::new(0.0, 1.0, 1.0, 1.0, NigHyper::default());
    (0..ncols)
        .map(|id| {
            let xs: Vec<f64> = g.sample(nrows, &mut rng);
            let data = DataContainer::new(xs);
            ColModel::Continuous(Column::new(id, data, prior.clone()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partitiion_to_ix_on_binary() {
        assert_eq!(partition_to_ix(&vec![0, 0]), 0);
        assert_eq!(partition_to_ix(&vec![1, 0]), 1);
        assert_eq!(partition_to_ix(&vec![0, 1]), 2);
        assert_eq!(partition_to_ix(&vec![1, 1]), 3);
    }

    #[test]
    fn test_partitiion_to_ix_on_trinary() {
        assert_eq!(partition_to_ix(&vec![0, 0, 0]), 0);
        assert_eq!(partition_to_ix(&vec![1, 0, 0]), 1);
        assert_eq!(partition_to_ix(&vec![2, 0, 0]), 2);
        assert_eq!(partition_to_ix(&vec![0, 1, 0]), 3);
        assert_eq!(partition_to_ix(&vec![1, 1, 0]), 4);
        assert_eq!(partition_to_ix(&vec![2, 1, 0]), 5);
        assert_eq!(partition_to_ix(&vec![0, 2, 0]), 6);
    }

    #[test]
    fn normalize_assignment_one_partition() {
        let z: Vec<usize> = vec![0, 0, 0, 0];
        assert_eq!(normalize_assignment(z.clone()), z);
    }

    #[test]
    fn normalize_assignment_should_not_change_normalize_assignment() {
        let z: Vec<usize> = vec![0, 1, 2, 1];
        assert_eq!(normalize_assignment(z.clone()), z);
    }

    #[test]
    fn normalize_assignment_should_fix_assignment_1() {
        let target: Vec<usize> = vec![0, 1, 2, 1];
        let unnormed: Vec<usize> = vec![1, 0, 2, 0];
        assert_eq!(normalize_assignment(unnormed.clone()), target);
    }

    #[test]
    fn normalize_assignment_should_fix_assignment_2() {
        let target: Vec<usize> = vec![0, 0, 1, 2];
        let unnormed: Vec<usize> = vec![0, 0, 2, 1];
        assert_eq!(normalize_assignment(unnormed.clone()), target);
    }

    #[test]
    fn normalize_assignment_should_fix_assignment_3() {
        let target: Vec<usize> = vec![0, 1, 1, 2, 1];
        let unnormed: Vec<usize> = vec![2, 1, 1, 0, 1];
        assert_eq!(normalize_assignment(unnormed.clone()), target);
    }
}
