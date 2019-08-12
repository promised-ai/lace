use braid_stats::labeler::{
    sf_loglike, Label, Labeler, LabelerPrior, LabelerSuffStat,
};
use braid_stats::seq::SobolSeq;
use braid_stats::simplex::uvec_to_simplex;
use braid_utils::misc::logsumexp;
use maplit::hashmap;
use rand::{FromEntropy, Rng};
use rand_xoshiro::Xoshiro256Plus;
use rv::data::DataOrSuffStat;
use rv::traits::{ConjugatePrior, Rv};
use std::f64::NEG_INFINITY;

const N_LABELS: usize = 3;

fn qmc(stat: &LabelerSuffStat, n_iters: usize) -> f64 {
    let pr = LabelerPrior::standard(N_LABELS as u8);
    let loglikes: Vec<f64> = SobolSeq::new(N_LABELS + 1)
        .skip(100)
        .take(n_iters)
        .map(|mut c| {
            let a = c.pop().unwrap();
            let b = c.pop().unwrap();
            c.push(1.0);

            let labeler = Labeler::new(a, b, uvec_to_simplex(c));
            let lp = sf_loglike(&stat, &labeler) + pr.ln_f(&labeler);
            if lp.is_finite() {
                lp
            } else {
                NEG_INFINITY
            }
        })
        .collect();

    logsumexp(&loglikes) - (n_iters as f64).ln()
}

fn importance<R: Rng>(
    stat: &LabelerSuffStat,
    n_iters: usize,
    mut rng: &mut R,
) -> f64 {
    let pr = LabelerPrior::standard(N_LABELS as u8);
    let q = LabelerPrior {
        pr_k: rv::dist::Kumaraswamy::uniform(),
        pr_h: rv::dist::Kumaraswamy::uniform(),
        pr_world: rv::dist::SymmetricDirichlet::new(1.0, N_LABELS).unwrap(),
    };
    let loglikes: Vec<f64> = (0..n_iters)
        .map(|_| {
            let x = q.draw(&mut rng);
            sf_loglike(&stat, &x) + pr.ln_f(&x) - q.ln_f(&x)
        })
        .collect();

    logsumexp(&loglikes) - (n_iters as f64).ln()
}

fn mc<R: Rng>(stat: &LabelerSuffStat, n_iters: usize, mut rng: &mut R) -> f64 {
    let pr = LabelerPrior::standard(N_LABELS as u8);
    let loglikes: Vec<f64> = (0..n_iters)
        .map(|_| sf_loglike(&stat, &pr.draw(&mut rng)))
        .collect();

    logsumexp(&loglikes) - (n_iters as f64).ln()
}

// Whatever method is actually used
fn prod(stat: &LabelerSuffStat) -> f64 {
    LabelerPrior::standard(N_LABELS as u8).ln_m(&DataOrSuffStat::SuffStat(stat))
}

fn main() {
    let mut rng = Xoshiro256Plus::from_entropy();

    let stat = LabelerSuffStat {
        n: 17,
        counter: hashmap! {
            Label::new(0, Some(0)) => 10,
            Label::new(1, Some(0)) => 5,
            Label::new(0, Some(1)) => 2,
        },
    };

    println!("Production ln m(x): {}", prod(&stat));
    for n in [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000].iter() {
        println!(
            "{} samples - QMC: {} MC: {}, IS: {}",
            n,
            qmc(&stat, *n),
            mc(&stat, *n, &mut rng),
            importance(&stat, *n, &mut rng),
        );
    }
}
