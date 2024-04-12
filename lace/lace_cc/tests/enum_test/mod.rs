//! Enumeration tests
use lace_cc::feature::{ColModel, Column, FType};
use lace_consts::rv::experimental::stick_breaking::{
    StickBreaking, StickBreakingDiscrete,
};
use lace_data::SparseContainer;
use lace_stats::{prior::sbd::SbdHyper, rv::traits::Sampleable};

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

#[allow(dead_code)]
pub fn gen_feature_ctor<R: rand::Rng>(
    ftype: FType,
) -> impl Fn(usize, usize, &mut R) -> ColModel {
    match ftype {
        FType::Continuous => {
            use lace_stats::prior::nix::NixHyper;
            use lace_stats::rv::dist::{Gaussian, NormalInvChiSquared};

            fn ctor<R: rand::Rng>(
                id: usize,
                n_rows: usize,
                mut rng: &mut R,
            ) -> ColModel {
                let gauss = Gaussian::standard();
                let hyper = NixHyper::default();
                let prior =
                    NormalInvChiSquared::new_unchecked(0.0, 1.0, 1.0, 1.0);
                let xs: Vec<f64> = gauss.sample(n_rows, &mut rng);
                let data = SparseContainer::from(xs);
                ColModel::Continuous(Column::new(id, data, prior, hyper))
            }
            ctor
        }
        FType::Categorical => {
            use lace_stats::prior::csd::CsdHyper;
            use lace_stats::rv::dist::{Categorical, SymmetricDirichlet};

            fn ctor<R: rand::Rng>(
                id: usize,
                n_rows: usize,
                mut rng: &mut R,
            ) -> ColModel {
                let cat = Categorical::uniform(4);
                let hyper = CsdHyper::default();
                let prior = SymmetricDirichlet::new_unchecked(1.0, 4);
                let xs: Vec<u8> = cat.sample(n_rows, &mut rng);
                let data = SparseContainer::from(xs);
                ColModel::Categorical(Column::new(id, data, prior, hyper))
            }
            ctor
        }
        FType::Count => {
            use lace_stats::prior::pg::PgHyper;
            use lace_stats::rv::dist::{Gamma, Poisson};

            fn ctor<R: rand::Rng>(
                id: usize,
                n_rows: usize,
                mut rng: &mut R,
            ) -> ColModel {
                let pois = Poisson::new(1.0).unwrap();
                let hyper = PgHyper::default();
                let prior = Gamma::new_unchecked(3.0, 3.0);
                let xs: Vec<u32> = pois.sample(n_rows, &mut rng);
                let data = SparseContainer::from(xs);
                ColModel::Count(Column::new(id, data, prior, hyper))
            }
            ctor
        }
        FType::Index => {
            fn ctor<R: rand::Rng>(
                id: usize,
                n_rows: usize,
                mut rng: &mut R,
            ) -> ColModel {
                let prior = StickBreaking::from_alpha(0.5).unwrap();
                let hyper = SbdHyper::vague();
                let fx: StickBreakingDiscrete = prior.draw(rng);
                let xs: Vec<usize> = fx.sample(n_rows, &mut rng);
                let data = SparseContainer::from(xs);
                ColModel::StickBreakingDiscrete(Column::new(
                    id, data, prior, hyper,
                ))
            }
            ctor
        }
        _ => panic!("unsupported ftype '{:?}' for view enum test", ftype),
    }
}

#[allow(dead_code)]
pub fn build_features<R: rand::Rng>(
    n_rows: usize,
    n_cols: usize,
    ftype: FType,
    mut rng: &mut R,
) -> Vec<ColModel> {
    let feature_ctor = gen_feature_ctor(ftype);
    (0..n_cols)
        .map(|id| feature_ctor(id, n_rows, &mut rng))
        .collect()
}

/// A partition generator meant for testing. Partition of `n` points into
/// 1, ..., n partitions.
///
/// # Example
///
/// ```
/// # use lace::misc::Partition;
/// let partitions: Vec<Vec<usize>> = Partition::new(4).collect();
///
/// // Bell(4) = 15. There are 15 ways to partition 4 items.
/// assert_eq!(partitions.len(), 15);
///
/// assert_eq!(partitions[0], vec![0, 0, 0, 0]);
/// assert_eq!(partitions[14], vec![0, 1, 2, 3]);
/// ```
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct Partition {
    z: Vec<usize>,
    k: Vec<usize>,
    n: usize,
    fresh: bool,
}

impl Partition {
    /// Create a generator that partitions `n` items
    ///
    /// # Example
    ///
    /// ```
    /// # use lace::misc::Partition;
    /// let partitions: Vec<Vec<usize>> = Partition::new(4).collect();
    ///
    /// // Bell(4) = 15. There are 15 ways to partition 4 items.
    /// assert_eq!(partitions.len(), 15);
    /// ```
    pub fn new(n: usize) -> Self {
        Partition {
            z: vec![0; n],
            k: vec![0; n],
            n,
            fresh: true,
        }
    }
}

impl Iterator for Partition {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Vec<usize>> {
        if self.fresh {
            self.fresh = false;
            Some(self.z.clone())
        } else {
            for i in (1..self.n).rev() {
                if self.z[i] <= self.k[i - 1] {
                    self.z[i] += 1;

                    if self.k[i] <= self.z[i] {
                        self.k[i] = self.z[i];
                    }

                    for j in (i + 1)..self.n {
                        self.z[j] = self.z[0];
                        self.k[j] = self.k[i];
                    }
                    return Some(self.z.clone());
                }
            }
            None
        }
    }
}
