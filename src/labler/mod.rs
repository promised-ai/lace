mod label;
mod prior;

pub use label::Label;
use rand::Rng;
use rv::traits::{HasSuffStat, Rv, SuffStat};
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize,
)]
pub struct LablerSuffStat {
    n: usize,
    n_truths: usize,
    n_correct: usize,
    n_unkown: usize,
}

impl Default for LablerSuffStat {
    fn default() -> Self {
        LablerSuffStat {
            n: 0,
            n_truths: 0,
            n_correct: 0,
            n_unkown: 0,
        }
    }
}

impl SuffStat<Label> for LablerSuffStat {
    fn n(&self) -> usize {
        self.n
    }

    fn observe(&mut self, x: &Label) {
        self.n += 1;
        match x.truth {
            Some(truth) => {
                self.n_truths += 1;
                if truth == x.label {
                    self.n_correct += 1;
                }
            }
            None => {
                self.n_unkown += 1;
            }
        }
    }

    fn forget(&mut self, x: &Label) {
        self.n -= 1;
        match x.truth {
            Some(truth) => {
                self.n_truths -= 1;
                if truth == x.label {
                    self.n_correct -= 1;
                }
            }
            None => {
                self.n_unkown -= 1;
            }
        }
    }
}

// FIXME: Currently designed for only binary worlds
#[derive(Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Labler {
    /// Probability knowledgeable
    p_k: f64,
    /// Probability helpful
    p_h: f64,
    /// Probability of hypotheses x = 1
    p_world: f64,
}

impl Labler {
    pub fn new(p_k: f64, p_h: f64, p_world: f64) -> Self {
        assert!(0.0 <= p_k && p_k <= 1.0);
        assert!(0.0 <= p_h && p_h <= 1.0);
        assert!(0.0 <= p_world && p_world <= 1.0);
        Labler { p_k, p_h, p_world }
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
    fn f_truthless(&self, label: bool) -> f64 {
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

    // Compute the probability of an informant label given the ground truth (world).
    // Optimized manual computation verified below in tests.
    fn f_truthful(&self, label: bool, world: bool) -> f64 {
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

impl HasSuffStat<Label> for Labler {
    type Stat = LablerSuffStat;
    fn empty_suffstat(&self) -> Self::Stat {
        LablerSuffStat::default()
    }
}

impl Rv<Label> for Labler {
    fn f(&self, x: &Label) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    use itertools::iproduct;

    const TOL: f64 = 1E-8;

    fn f_truthful(labler: &Labler, label: bool, truth: bool) -> f64 {
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
                        labler.p_world
                    } else {
                        1.0 - labler.p_world
                    }
                };

                let p_h = if h { labler.p_h } else { 1.0 - labler.p_h };
                let p_k = if k { labler.p_k } else { 1.0 - labler.p_k };
                let p_w = if truth {
                    labler.p_world
                } else {
                    1.0 - labler.p_world
                };

                acc + p_a_given_h_b * p_k_given_b_w * p_h * p_k * p_w
            }
        })
    }

    fn f_truthless(labler: &Labler, label: bool) -> f64 {
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
                        labler.p_world
                    } else {
                        1.0 - labler.p_world
                    }
                };

                let p_h = if h { labler.p_h } else { 1.0 - labler.p_h };
                let p_k = if k { labler.p_k } else { 1.0 - labler.p_k };
                let p_w = if w {
                    labler.p_world
                } else {
                    1.0 - labler.p_world
                };

                acc + p_a_given_h_b * p_k_given_b_w * p_h * p_k * p_w
            }
        })
    }

    #[test]
    fn p_truthful() {
        let labler = Labler::new(0.7, 0.8, 0.4);
        assert_relative_eq!(
            labler.f_truthful(false, false),
            f_truthful(&labler, false, false),
            epsilon = TOL
        );
        assert_relative_eq!(
            labler.f_truthful(false, true),
            f_truthful(&labler, false, true),
            epsilon = TOL
        );
        assert_relative_eq!(
            labler.f_truthful(true, false),
            f_truthful(&labler, true, false),
            epsilon = TOL
        );
        assert_relative_eq!(
            labler.f_truthful(true, true),
            f_truthful(&labler, true, true),
            epsilon = TOL
        );
    }

    #[test]
    fn p_truthless() {
        let labler = Labler::new(0.7, 0.8, 0.4);
        assert_relative_eq!(
            labler.f_truthless(true),
            f_truthless(&labler, true),
            epsilon = TOL
        );
        assert_relative_eq!(
            labler.f_truthless(false),
            f_truthless(&labler, false),
            epsilon = TOL
        );
    }
}
