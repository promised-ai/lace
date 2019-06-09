use braid::integrate::mc_integral;
use braid::labeler::{Label, Labeler, LabelerSuffStat};
use braid_utils::misc::logsumexp;
use rand::{FromEntropy, Rng};
use rand_xoshiro::Xoshiro256Plus;
use rv::dist::Beta;
use rv::traits::*;

fn loglike(xs: &LabelerSuffStat, labeler: &Labeler) -> f64 {
    let mut logp = 0.0;

    if xs.n_truth_tt > 0 {
        logp +=
            xs.n_truth_tt as f64 * labeler.ln_f(&Label::new(true, Some(true)));
    }

    if xs.n_truth_tf > 0 {
        logp +=
            xs.n_truth_tf as f64 * labeler.ln_f(&Label::new(true, Some(false)));
    }

    if xs.n_truth_ft > 0 {
        logp +=
            xs.n_truth_ft as f64 * labeler.ln_f(&Label::new(false, Some(true)));
    }

    if xs.n_truth_ff > 0 {
        logp += xs.n_truth_ff as f64
            * labeler.ln_f(&Label::new(false, Some(false)));
    }

    if xs.n_unk_t > 0 {
        logp += xs.n_unk_t as f64 * labeler.ln_f(&Label::new(true, None));
    }

    if xs.n_unk_f > 0 {
        logp += xs.n_unk_f as f64 * labeler.ln_f(&Label::new(false, None));
    }

    logp
}

fn prior_draw<R: Rng>(mut rng: &mut R) -> (f64, f64, f64) {
    let beta = Beta::jeffreys();
    let p_k = beta.draw(&mut rng);
    let p_h = beta.draw(&mut rng);
    let p_world = beta.draw(&mut rng);

    (p_k, p_h, p_world)
}

fn main() {
    let labeler = Labeler::new(0.9, 0.9, 0.7);

    let xs = vec![
        Label::new(false, None),
        Label::new(true, None),
        Label::new(true, None),
        Label::new(true, Some(true)),
        Label::new(false, Some(true)),
        Label::new(true, Some(true)),
        Label::new(true, Some(true)),
        Label::new(true, Some(true)),
        Label::new(true, Some(false)),
    ];

    let mut stat = LabelerSuffStat::new();
    stat.observe_many(&xs);

    let logf = |params: (f64, f64, f64)| -> f64 {
        let lablr = Labeler::new(params.0, params.1, params.2);
        loglike(&stat, &lablr)
    };

    let mut rng = Xoshiro256Plus::from_entropy();
    let log_m = mc_integral(logf, prior_draw, 100_000, &mut rng);

    println!("    f(x) = {}", loglike(&stat, &labeler).exp());
    println!("log f(x) = {}", loglike(&stat, &labeler));
    println!("log m(x) ~ {}", log_m);
}
