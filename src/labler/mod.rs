mod label;
mod prior;

use itertools::iproduct;
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

    fn f_truthless(&self, label: bool) -> f64 {
        let beliefs = vec![true, false];
        let worlds = vec![true, false];
        let helpful = vec![true, false];
        let knowledgeable = vec![true, false];

        let states = iproduct!(knowledgeable, helpful, beliefs, worlds);

        // most states are impossible, should be a way to create an iterator
        // over states that doesn't consider impossible states
        states.fold(0.0, |acc, (k, h, b, w)| {
            if (k && b != w) || (h && b != label) || (!h && b == label) {
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
                        self.p_world
                    } else {
                        1.0 - self.p_world
                    }
                };

                let p_h = if h { self.p_h } else { 1.0 - self.p_h };
                let p_k = if k { self.p_k } else { 1.0 - self.p_k };
                let p_w = if w { self.p_world } else { 1.0 - self.p_world };

                acc + p_a_given_h_b * p_k_given_b_w * p_h * p_k * p_w
            }
        })
    }

    fn f_truthful(&self, label: bool, truth: bool) -> f64 {
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
                        self.p_world
                    } else {
                        1.0 - self.p_world
                    }
                };

                let p_h = if h { self.p_h } else { 1.0 - self.p_h };
                let p_k = if k { self.p_k } else { 1.0 - self.p_k };
                let p_w = if truth {
                    self.p_world
                } else {
                    1.0 - self.p_world
                };

                acc + p_a_given_h_b * p_k_given_b_w * p_h * p_k * p_w
            }
        })
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

// FIXME: tests!
