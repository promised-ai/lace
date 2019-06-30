use braid_stats::labeler::{
    sf_loglike, Labeler, LabelerPrior, LabelerSuffStat,
};
use braid_stats::seq::HaltonSeq;
use braid_utils::misc::logsumexp;
use rand::{FromEntropy, Rng};
use rand_xoshiro::Xoshiro256Plus;
use rv::traits::Rv;

fn qmc(stat: &LabelerSuffStat, n_iters: usize) -> f64 {
    let loglikes: Vec<f64> = HaltonSeq::new(2)
        .zip(HaltonSeq::new(3))
        .zip(HaltonSeq::new(5))
        .take(n_iters)
        .map(|((a, b), c)| {
            let labeler = Labeler::new(a, b, c);
            sf_loglike(&stat, &labeler)
        })
        .collect();

    logsumexp(&loglikes) - (n_iters as f64).ln()
}

fn mc<R: Rng>(stat: &LabelerSuffStat, n_iters: usize, mut rng: &mut R) -> f64 {
    let pr = LabelerPrior::default();
    let loglikes: Vec<f64> = (0..n_iters)
        .map(|_| sf_loglike(&stat, &pr.draw(&mut rng)))
        .collect();

    logsumexp(&loglikes) - (n_iters as f64).ln()
}

fn main() {
    let mut rng = Xoshiro256Plus::from_entropy();

    let stat = LabelerSuffStat {
        n: 30,
        n_truth_tt: 20,
        n_truth_tf: 5,
        n_truth_ft: 1,
        n_truth_ff: 1,
        n_unk_t: 2,
        n_unk_f: 1,
    };

    for n in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        println!(
            "{} samples - QMC: {} MC: {}",
            n,
            qmc(&stat, *n),
            mc(&stat, *n, &mut rng)
        );
    }
}
