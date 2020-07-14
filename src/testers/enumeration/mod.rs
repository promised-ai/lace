//! Enumeration tests
use crate::cc::{ColModel, Column, FType};
use braid_data::SparseContainer;
use rv::traits::Rv;

pub mod state;
pub mod view;

/// Convert a partition with to an integer index by converting a k-length
/// partition into a k-length base-k integer from left to right.
fn partition_to_ix(z: &[usize]) -> u64 {
    let k = z.len() as u64;
    z.iter()
        .enumerate()
        .fold(0_u64, |acc, (i, &zi)| acc + (zi as u64) * k.pow(i as u32))
}

/// Adjust the assignment for label switching. The resulting assignment will
/// have partition indices that start at zero; new partition indices are
/// introduced incrementally from left to right (see the tests for more).
#[allow(clippy::needless_range_loop)]
fn normalize_assignment(mut z: Vec<usize>) -> Vec<usize> {
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

fn gen_feature_ctor<R: rand::Rng>(
    ftype: FType,
) -> impl Fn(usize, usize, &mut R) -> ColModel {
    match ftype {
        FType::Continuous => {
            use braid_stats::prior::{Ng, NigHyper};
            use rv::dist::Gaussian;

            fn ctor<R: rand::Rng>(
                id: usize,
                nrows: usize,
                mut rng: &mut R,
            ) -> ColModel {
                let gauss = Gaussian::standard();
                let prior = Ng::new(0.0, 1.0, 1.0, 1.0, NigHyper::default());
                let xs: Vec<f64> = gauss.sample(nrows, &mut rng);
                let data = SparseContainer::new(xs);
                ColModel::Continuous(Column::new(id, data, prior))
            }
            ctor
        }
        FType::Categorical => {
            use braid_stats::prior::{Csd, CsdHyper};
            use rv::dist::Categorical;

            fn ctor<R: rand::Rng>(
                id: usize,
                nrows: usize,
                mut rng: &mut R,
            ) -> ColModel {
                let cat = Categorical::uniform(4);
                let prior = Csd::new(1.0, 4, CsdHyper::default());
                let xs: Vec<u8> = cat.sample(nrows, &mut rng);
                let data = SparseContainer::new(xs);
                ColModel::Categorical(Column::new(id, data, prior))
            }
            ctor
        }
        FType::Count => {
            use braid_stats::prior::{Pg, PgHyper};
            use rv::dist::Poisson;

            fn ctor<R: rand::Rng>(
                id: usize,
                nrows: usize,
                mut rng: &mut R,
            ) -> ColModel {
                let pois = Poisson::new(1.0).unwrap();
                let prior = Pg::new(3.0, 3.0, PgHyper::default());
                let xs: Vec<u32> = pois.sample(nrows, &mut rng);
                let data = SparseContainer::new(xs);
                ColModel::Count(Column::new(id, data, prior))
            }
            ctor
        }
        _ => panic!("unsupported ftype '{:?}' for view enum test", ftype),
    }
}

fn build_features<R: rand::Rng>(
    nrows: usize,
    ncols: usize,
    ftype: FType,
    mut rng: &mut R,
) -> Vec<ColModel> {
    let feature_ctor = gen_feature_ctor(ftype);
    (0..ncols)
        .map(|id| feature_ctor(id, nrows, &mut rng))
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
