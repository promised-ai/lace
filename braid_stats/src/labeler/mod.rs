mod label;
mod prior;

pub use label::Label;
pub use prior::{sf_loglike, LabelerPosterior, LabelerPrior};

use rand::Rng;
use rv::traits::{Entropy, HasSuffStat, KlDivergence, Rv, SuffStat};
use serde::{Deserialize, Serialize};

pub const ALL_LABELS: [Label; 4] = [
    Label {
        label: true,
        truth: Some(true),
    },
    Label {
        label: true,
        truth: Some(false),
    },
    Label {
        label: false,
        truth: Some(true),
    },
    Label {
        label: false,
        truth: Some(false),
    },
];

// FIXME: Currently designed for only binary worlds
/// An informant who provides binary labels.
///
/// # Notes
///
/// This data type cannot be used in the Gibbs row kernel because it is not a
/// conjugate model.
#[derive(Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Labeler {
    /// Probability knowledgeable
    p_k: f64,
    /// Probability helpful
    p_h: f64,
    /// Probability of hypotheses x = 1
    p_world: f64,
}

impl Labeler {
    pub fn new(p_k: f64, p_h: f64, p_world: f64) -> Self {
        assert!(0.0 <= p_k && p_k <= 1.0);
        assert!(0.0 <= p_h && p_h <= 1.0);
        assert!(0.0 <= p_world && p_world <= 1.0);
        Labeler { p_k, p_h, p_world }
    }

    /// Returns the probability that the informant is knowledgeable
    pub fn p_k(&self) -> f64 {
        self.p_k
    }

    /// Returns the probability that the informant is helpful
    pub fn p_h(&self) -> f64 {
        self.p_h
    }

    /// Returns the probability that the true label = 1
    pub fn p_world(&self) -> f64 {
        self.p_world
    }

    // Compute the probability of an informant label given no ground truth (world).
    // Optimized manual computation verified below in tests.
    pub fn f_truthless(&self, label: bool) -> f64 {
        let p_world = if label {
            self.p_world
        } else {
            1.0 - self.p_world
        };

        let one_minus_pk = 1.0 - self.p_k;
        let one_minus_ph = 1.0 - self.p_h;
        let one_minus_pw = 1.0 - p_world;

        self.p_k * (one_minus_ph * one_minus_pw + self.p_h * p_world)
            + (p_world * one_minus_pk * self.p_h) * (p_world + one_minus_pw)
            + one_minus_pw
                * one_minus_pk
                * one_minus_ph
                * (p_world + one_minus_pw)
    }

    // Compute the probability of an informant label given the ground truth
    // (world).  Optimized manual computation verified below in tests.
    pub fn f_truthful(&self, label: bool, world: bool) -> f64 {
        let p_world = if world {
            self.p_world
        } else {
            1.0 - self.p_world
        };

        let p = if label == world {
            // helpful and knowledgeable
            self.p_h * self.p_k
                // unknowledgeable and helpful (guess truth)
                + p_world * self.p_h * (1.0 - self.p_k)
                // knowledgeable and unhelpful (guess wrong, labels true)
                + (1.0 - p_world) * (1.0 - self.p_h) * (1.0 - self.p_k)
        } else {
            // unknowledgeable and helpful, guesses wrong and labels wrong
            (1.0 - p_world) * self.p_h * (1.0 - self.p_k)
                // unknowledgeable and unhelpful, guesses right and labels wrong
                + p_world * (1.0 - self.p_h) * (1.0 - self.p_k)
                // knowledgeable and unhelpful, labels wrong
                + (1.0 - self.p_h) * self.p_k
        };

        p * p_world
    }
}

impl HasSuffStat<Label> for Labeler {
    type Stat = LabelerSuffStat;
    fn empty_suffstat(&self) -> Self::Stat {
        LabelerSuffStat::default()
    }
}

impl Rv<Label> for Labeler {
    fn f(&self, x: &Label) -> f64 {
        // We want the joint probability of p(x, x*) not p(x|x*)
        match x.truth {
            Some(truth) => self.f_truthful(x.label, truth),
            None => self.f_truthless(x.label),
        }
    }

    fn ln_f(&self, x: &Label) -> f64 {
        self.f(&x).ln()
    }

    // Draws worlds/truths and labels
    fn draw<R: Rng>(&self, rng: &mut R) -> Label {
        let w = rng.gen::<f64>() < self.p_world;
        let h = rng.gen::<f64>() < self.p_h;
        let k = rng.gen::<f64>() < self.p_k;

        let b = if k {
            w
        } else {
            rng.gen::<f64>() < self.p_world
        };
        let a = if h { b } else { !b };

        Label {
            label: a,
            truth: Some(w),
        }
    }
}

impl Entropy for Labeler {
    fn entropy(&self) -> f64 {
        ALL_LABELS.iter().fold(0.0, |acc, x| {
            let p = self.f(&x);
            acc - p * p.ln()
        })
    }
}

impl KlDivergence for Labeler {
    fn kl(&self, other: &Self) -> f64 {
        ALL_LABELS.iter().fold(0.0, |acc, x| {
            let p = self.f(x);
            acc + p * (p.ln() - other.ln_f(x))
        })
    }
}

// TODO: The size of this suffstats is 224 bits, which is the same amount of
// data as 14 `Label`s. Not very efficient. It might be smarter to use a
// counter.

/// The sufficient statistic for the `Labeler`
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct LabelerSuffStat {
    // total number of data observed
    pub n: usize,
    // number ofvalues with truths where lable = true and truth = true
    pub n_truth_tt: usize,
    // number ofvalues with truths where lable = true and truth = false
    pub n_truth_tf: usize,
    // number ofvalues with truths where lable = false and truth = true
    pub n_truth_ft: usize,
    // number ofvalues with truths where lable = false and truth = false
    pub n_truth_ff: usize,
    // number of values without truth values, where label = true
    pub n_unk_t: usize,
    // number of values without truth values, where label = false
    pub n_unk_f: usize,
}

impl LabelerSuffStat {
    pub fn new() -> Self {
        LabelerSuffStat {
            n: 0,
            n_truth_tt: 0,
            n_truth_tf: 0,
            n_truth_ft: 0,
            n_truth_ff: 0,
            n_unk_t: 0,
            n_unk_f: 0,
        }
    }
}

impl Default for LabelerSuffStat {
    fn default() -> Self {
        LabelerSuffStat::new()
    }
}

impl SuffStat<Label> for LabelerSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn observe(&mut self, x: &Label) {
        let label = x.label;
        self.n += 1;
        match x.truth {
            None if label => self.n_unk_t += 1,
            None if !label => self.n_unk_f += 1,
            Some(truth) if truth && label => self.n_truth_tt += 1,
            Some(truth) if truth && !label => self.n_truth_ft += 1,
            Some(truth) if !truth && label => self.n_truth_tf += 1,
            Some(truth) if !(truth || label) => self.n_truth_ff += 1,
            _ => unreachable!(),
        }
    }

    fn forget(&mut self, x: &Label) {
        let label = x.label;
        self.n -= 1;
        match x.truth {
            None if label => self.n_unk_t -= 1,
            None if !label => self.n_unk_f -= 1,
            Some(truth) if truth && label => self.n_truth_tt -= 1,
            Some(truth) if truth && !label => self.n_truth_ft -= 1,
            Some(truth) if !truth && label => self.n_truth_tf -= 1,
            Some(truth) if !(truth || label) => self.n_truth_ff -= 1,
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    use itertools::iproduct;

    const TOL: f64 = 1E-8;
    const N_MH_SAMPLES: u32 = 2_500_000;

    fn f_truthful(labeler: &Labeler, label: bool, truth: bool) -> f64 {
        let beliefs = vec![true, false];
        let helpful = vec![true, false];
        let knowledgeable = vec![true, false];

        let states = iproduct!(knowledgeable, helpful, beliefs);

        // most states are impossible, should be a way to create an iterator
        // over states that doesn't consider impossible states
        states.fold(0.0, |acc, (k, h, b)| {
            if (k && b != truth) || (h && b != label) || (!h && b == label) {
                // impossible states. Knowledgeable informants always believe
                // the truth, helpful informants always label consistent with
                // their belief, and unhelpful informants never label
                // consistent with their belief.
                acc
            } else {
                // left in for clarity
                let p_a_given_h_b = 1.0;

                let p_k_given_b_w = if k {
                    1.0
                } else {
                    if b {
                        labeler.p_world
                    } else {
                        1.0 - labeler.p_world
                    }
                };

                let p_h = if h { labeler.p_h } else { 1.0 - labeler.p_h };
                let p_k = if k { labeler.p_k } else { 1.0 - labeler.p_k };
                let p_w = if truth {
                    labeler.p_world
                } else {
                    1.0 - labeler.p_world
                };

                acc + p_a_given_h_b * p_k_given_b_w * p_h * p_k * p_w
            }
        })
    }

    fn f_truthless(labeler: &Labeler, label: bool) -> f64 {
        let beliefs = vec![true, false];
        let worlds = vec![true, false];
        let helpful = vec![true, false];
        let knowledgeable = vec![true, false];

        let states = iproduct!(knowledgeable, helpful, beliefs, worlds);

        // most states are impossible, should be a way to create an iterator
        // over states that doesn't consider impossible states
        states.fold(0.0, |acc, (k, h, b, w)| {
            if (k && (b != w)) || (h && (b != label)) || (!h && (b == label)) {
                // impossible states. Knowledgeable informants always believe
                // the truth, helpful informants always label consistent with
                // their belief, and unhelpful informants never label
                // consistent with their belief.
                acc
            } else {
                // left in for clarity
                let p_a_given_h_b = 1.0;

                let p_k_given_b_w = if k {
                    1.0
                } else {
                    if b {
                        labeler.p_world
                    } else {
                        1.0 - labeler.p_world
                    }
                };

                let p_h = if h { labeler.p_h } else { 1.0 - labeler.p_h };
                let p_k = if k { labeler.p_k } else { 1.0 - labeler.p_k };
                let p_w = if w {
                    labeler.p_world
                } else {
                    1.0 - labeler.p_world
                };

                acc + p_a_given_h_b * p_k_given_b_w * p_h * p_k * p_w
            }
        })
    }

    // estimate the probability P(label, truth)
    fn mc_estimate(label: bool, truth: bool, labeler: &Labeler, n: u32) -> f64 {
        let mut rng = rand::thread_rng();

        let mut numer = 0.0;

        for _ in 0..n {
            let k = rng.gen::<f64>() < labeler.p_k();
            let h = rng.gen::<f64>() < labeler.p_h();
            let w = rng.gen::<f64>() < labeler.p_world();

            if w == truth {
                let b = if k {
                    w
                } else {
                    rng.gen::<f64>() < labeler.p_world()
                };
                let a = if h { b } else { !b };

                if a == label {
                    numer += 1.0;
                }
            }
        }
        numer / n as f64
    }

    // estimate the probability P(label)
    fn mc_estimate_truthless(label: bool, labeler: &Labeler, n: u32) -> f64 {
        let mut rng = rand::thread_rng();

        let mut numer = 0.0;

        for _ in 0..n {
            let k = rng.gen::<f64>() < labeler.p_k();
            let h = rng.gen::<f64>() < labeler.p_h();
            let w = rng.gen::<f64>() < labeler.p_world();

            let b = if k {
                w
            } else {
                rng.gen::<f64>() < labeler.p_world()
            };
            let a = if h { b } else { !b };

            if a == label {
                numer += 1.0;
            }
        }
        numer / f64::from(n)
    }

    #[test]
    fn ps_sum_to_one() {
        let labeler = Labeler::new(0.7, 0.8, 0.4);

        let sum_p = labeler.f_truthful(true, true)
            + labeler.f_truthful(true, false)
            + labeler.f_truthful(false, true)
            + labeler.f_truthful(false, false);

        assert_relative_eq!(sum_p, 1.0, epsilon = TOL);

    }

    #[test]
    fn p_truthful() {
        let labeler = Labeler::new(0.7, 0.8, 0.4);
        assert_relative_eq!(
            labeler.f_truthful(false, false),
            f_truthful(&labeler, false, false),
            epsilon = TOL
        );
        assert_relative_eq!(
            labeler.f_truthful(false, true),
            f_truthful(&labeler, false, true),
            epsilon = TOL
        );
        assert_relative_eq!(
            labeler.f_truthful(true, false),
            f_truthful(&labeler, true, false),
            epsilon = TOL
        );
        assert_relative_eq!(
            labeler.f_truthful(true, true),
            f_truthful(&labeler, true, true),
            epsilon = TOL
        );
    }

    #[test]
    fn p_truthless() {
        let labeler = Labeler::new(0.7, 0.8, 0.4);
        assert_relative_eq!(
            labeler.f_truthless(true),
            f_truthless(&labeler, true),
            epsilon = TOL
        );
        assert_relative_eq!(
            labeler.f_truthless(false),
            f_truthless(&labeler, false),
            epsilon = TOL
        );
    }

    #[test]
    fn p_truthful_vs_mc_estimate_truthful_tt() {
        let labeler = Labeler::new(0.7, 0.8, 0.4);
        assert_relative_eq!(
            labeler.f(&Label::new(true, Some(true))),
            mc_estimate(true, true, &labeler, N_MH_SAMPLES),
            epsilon = 1e-3,
        );
    }

    #[test]
    fn p_truthful_vs_mc_estimate_truthful_tf() {
        let labeler = Labeler::new(0.7, 0.8, 0.4);
        assert_relative_eq!(
            labeler.f(&Label::new(true, Some(false))),
            mc_estimate(true, false, &labeler, N_MH_SAMPLES),
            epsilon = 1e-3,
        );
    }

    #[test]
    fn p_truthful_vs_mc_estimate_truthful_ft() {
        let labeler = Labeler::new(0.7, 0.8, 0.4);
        assert_relative_eq!(
            labeler.f(&Label::new(false, Some(true))),
            mc_estimate(false, true, &labeler, N_MH_SAMPLES),
            epsilon = 1e-3,
        );
    }

    #[test]
    fn p_truthful_vs_mc_estimate_truthful_ff() {
        let labeler = Labeler::new(0.7, 0.8, 0.4);
        assert_relative_eq!(
            labeler.f(&Label::new(false, Some(false))),
            mc_estimate(false, false, &labeler, N_MH_SAMPLES),
            epsilon = 1e-3,
        );
    }

    #[test]
    fn p_truthful_vs_mc_estimate_truthless_t() {
        let labeler = Labeler::new(0.7, 0.8, 0.4);
        assert_relative_eq!(
            labeler.f(&Label::new(true, None)),
            mc_estimate_truthless(true, &labeler, N_MH_SAMPLES),
            epsilon = 1e-3,
        );
    }

    #[test]
    fn p_truthful_vs_mc_estimate_truthless_f() {
        let labeler = Labeler::new(0.7, 0.8, 0.4);
        assert_relative_eq!(
            labeler.f(&Label::new(false, None)),
            mc_estimate_truthless(false, &labeler, N_MH_SAMPLES),
            epsilon = 1e-3,
        );
    }

    #[test]
    fn labeler_suffstat_observe_forget_tt() {
        let x = Label::new(true, Some(true));

        let mut stat = LabelerSuffStat::new();
        stat.observe(&x);

        let target = LabelerSuffStat {
            n: 1,
            n_truth_tt: 1,
            n_truth_tf: 0,
            n_truth_ft: 0,
            n_truth_ff: 0,
            n_unk_t: 0,
            n_unk_f: 0,
        };

        assert_eq!(target, stat);

        stat.forget(&x);

        assert_eq!(LabelerSuffStat::new(), stat);
    }

    #[test]
    fn labeler_suffstat_observe_forget_tf() {
        let x = Label::new(true, Some(false));

        let mut stat = LabelerSuffStat::new();
        stat.observe(&x);

        let target = LabelerSuffStat {
            n: 1,
            n_truth_tt: 0,
            n_truth_tf: 1,
            n_truth_ft: 0,
            n_truth_ff: 0,
            n_unk_t: 0,
            n_unk_f: 0,
        };

        assert_eq!(target, stat);

        stat.forget(&x);

        assert_eq!(LabelerSuffStat::new(), stat);
    }

    #[test]
    fn labeler_suffstat_observe_forget_ft() {
        let x = Label::new(false, Some(true));

        let mut stat = LabelerSuffStat::new();
        stat.observe(&x);

        let target = LabelerSuffStat {
            n: 1,
            n_truth_tt: 0,
            n_truth_tf: 0,
            n_truth_ft: 1,
            n_truth_ff: 0,
            n_unk_t: 0,
            n_unk_f: 0,
        };

        assert_eq!(target, stat);

        stat.forget(&x);

        assert_eq!(LabelerSuffStat::new(), stat);
    }

    #[test]
    fn labeler_suffstat_observe_forget_ff() {
        let x = Label::new(false, Some(false));

        let mut stat = LabelerSuffStat::new();
        stat.observe(&x);

        let target = LabelerSuffStat {
            n: 1,
            n_truth_tt: 0,
            n_truth_tf: 0,
            n_truth_ft: 0,
            n_truth_ff: 1,
            n_unk_t: 0,
            n_unk_f: 0,
        };

        assert_eq!(target, stat);

        stat.forget(&x);

        assert_eq!(LabelerSuffStat::new(), stat);
    }

    #[test]
    fn labeler_suffstat_observe_forget_tn() {
        let x = Label::new(true, None);

        let mut stat = LabelerSuffStat::new();
        stat.observe(&x);

        let target = LabelerSuffStat {
            n: 1,
            n_truth_tt: 0,
            n_truth_tf: 0,
            n_truth_ft: 0,
            n_truth_ff: 0,
            n_unk_t: 1,
            n_unk_f: 0,
        };

        assert_eq!(target, stat);

        stat.forget(&x);

        assert_eq!(LabelerSuffStat::new(), stat);
    }

    #[test]
    fn labeler_suffstat_observe_forget_fn() {
        let x = Label::new(false, None);

        let mut stat = LabelerSuffStat::new();
        stat.observe(&x);

        let target = LabelerSuffStat {
            n: 1,
            n_truth_tt: 0,
            n_truth_tf: 0,
            n_truth_ft: 0,
            n_truth_ff: 0,
            n_unk_t: 0,
            n_unk_f: 1,
        };

        assert_eq!(target, stat);

        stat.forget(&x);

        assert_eq!(LabelerSuffStat::new(), stat);
    }
}
