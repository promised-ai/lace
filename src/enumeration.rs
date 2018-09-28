//! Enumeration test
extern crate rand;
extern crate rv;

use std::collections::BTreeMap;

use self::rand::Rng;
use self::rv::dist::Gaussian;
use self::rv::traits::Rv;

use cc::assignment::lcrp;
use cc::transition::ViewTransition;
use cc::{
    AssignmentBuilder, ColModel, Column, DataContainer, Feature, RowAssignAlg,
    View, ViewBuilder,
};
use dist::prior::ng::{Ng, NigHyper};
use misc::{logsumexp, Partition};

/// Convert a partition with to an integer index
fn partition_to_ix(z: &Vec<usize>) -> u64 {
    let k = z.len() as u64;
    z.iter()
        .enumerate()
        .fold(0_u64, |acc, (i, &zi)| acc + (zi as u64) * k.pow(i as u32))
}

fn calc_partition_ln_posterior<R: Rng>(
    features: &Vec<ColModel>,
    alpha: f64,
    mut rng: &mut R,
) -> BTreeMap<u64, f64> {
    let n = features[0].len();
    let mut ln_posterior: BTreeMap<u64, f64> = BTreeMap::new();

    Partition::new(n).for_each(|z| {
        let ix = partition_to_ix(&z);

        let asgn = AssignmentBuilder::from_vec(z)
            .with_alpha(alpha)
            .build(&mut rng)
            .unwrap();

        let ln_pz = lcrp(n, &asgn.counts, alpha);

        let view: View = ViewBuilder::from_assignment(asgn)
            .with_features(features.clone())
            .build(&mut rng);

        ln_posterior.insert(ix, view.score() + ln_pz);
    });

    ln_posterior
}

fn norm_posterior(ln_posterior: &BTreeMap<u64, f64>) -> BTreeMap<u64, f64> {
    let logps: Vec<f64> = ln_posterior.values().map(|&p| p).collect();
    let z = logsumexp(&logps);
    let mut normed: BTreeMap<u64, f64> = BTreeMap::new();
    for (key, lp) in ln_posterior {
        normed.insert(*key, (lp - z).exp());
    }
    normed
}

fn normalize_assignment(mut z: Vec<usize>) -> Vec<usize> {
    let mut should_be: usize = 0;
    let mut max_is = 0;
    for i in 0..z.len() {
        let is = z[i];
        if is > should_be {
            for j in 0..z.len() {
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

fn build_features<R: Rng>(
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
        }).collect()
}

pub fn enum_test(
    nrows: usize,
    ncols: usize,
    n_runs: usize,
    n_iters: usize,
    row_alg: RowAssignAlg,
) -> f64 {
    let mut rng = rand::thread_rng();
    let features = build_features(nrows, ncols, &mut rng);
    let ln_posterior = calc_partition_ln_posterior(&features, 1.0, &mut rng);
    let posterior = norm_posterior(&ln_posterior);

    let transitions: Vec<ViewTransition> = vec![
        ViewTransition::RowAssignment,
        ViewTransition::ComponentParams,
    ];

    let mut est_posterior: BTreeMap<u64, f64> = BTreeMap::new();
    let inc: f64 = ((n_runs * n_iters) as f64).recip();

    for _ in 0..n_runs {
        let asgn = AssignmentBuilder::new(nrows)
            .with_alpha(1.0)
            .build(&mut rng)
            .unwrap();
        let mut view = ViewBuilder::from_assignment(asgn)
            .with_features(features.clone())
            .build(&mut rng);
        for _ in 0..n_iters {
            view.update(10, row_alg, &transitions, &mut rng);

            let normed = normalize_assignment(view.asgn.asgn.clone());
            let ix = partition_to_ix(&normed);

            if !posterior.contains_key(&ix) {
                println!("{:?}", view.asgn.asgn);
                println!("{:?}", normed);
                panic!("invalid index!");
            }

            *est_posterior.entry(ix).or_insert(0.0) += inc;
        }
    }

    //    println!("Posterior {:#?}", posterior);
    //    println!("Estimate {:#?}", est_posterior);

    posterior.iter().fold(0.0, |err, (key, &p)| {
        if est_posterior.contains_key(key) {
            err + (p - est_posterior[key]).abs()
        } else {
            err + p
        }
    })
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

    // TODO: Move enumeration test to integration tests
    #[test]
    fn view_enum_test_gibbs() {
        let err = enum_test(4, 1, 1, 5_000, RowAssignAlg::Gibbs);
        println!("Error: {}", err);
        assert!(err < 0.05);
    }

    #[test]
    fn view_enum_test_finite_cpu() {
        let err = enum_test(4, 1, 1, 5_000, RowAssignAlg::FiniteCpu);
        println!("Error: {}", err);
        assert!(err < 0.2);
    }

    #[test]
    fn view_enum_test_slice() {
        let err = enum_test(4, 1, 1, 5_000, RowAssignAlg::Slice);
        println!("Error: {}", err);
        assert!(err < 0.05);
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
