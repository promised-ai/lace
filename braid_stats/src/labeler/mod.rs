mod label;
mod prior;

pub use label::{Label, LabelIterator};
pub use prior::{sf_loglike, LabelerPosterior, LabelerPrior};

use crate::simplex::SimplexPoint;
use rand::Rng;
use rv::traits::{Entropy, HasSuffStat, KlDivergence, Mode, Rv, SuffStat};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An informant who provides Categorical labels.
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
    /// Probability of hypotheses
    p_world: SimplexPoint,
}

pub struct LabelerLikelihoodParts {
    pub p_hk: f64,
    pub p_uhk: f64,
    pub p_huk: f64,
    pub p_uhuk: f64,
}

impl LabelerLikelihoodParts {
    pub fn sum(&self) -> f64 {
        self.p_hk + self.p_uhk + self.p_huk + self.p_uhuk
    }

    pub fn p_helpful(&self) -> f64 {
        (self.p_hk + self.p_huk) / self.sum()
    }

    pub fn p_knowledgeable(&self) -> f64 {
        (self.p_hk + self.p_uhk) / self.sum()
    }
}

impl Labeler {
    pub fn new(p_k: f64, p_h: f64, p_world: SimplexPoint) -> Self {
        assert!(0.0 <= p_k && p_k <= 1.0);
        assert!(0.0 <= p_h && p_h <= 1.0);
        assert!(p_world.ndims() < std::u8::MAX as usize);
        Labeler { p_k, p_h, p_world }
    }

    /// Return the number of possible labels
    pub fn n_labels(&self) -> usize {
        self.p_world.ndims()
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
    pub fn p_world(&self) -> &SimplexPoint {
        &self.p_world
    }

    pub fn f_truthless_parts(&self, label: u8) -> LabelerLikelihoodParts {
        let init = LabelerLikelihoodParts {
            p_hk: 0.0,
            p_huk: 0.0,
            p_uhk: 0.0,
            p_uhuk: 0.0,
        };

        (0..self.n_labels()).fold(init, |acc, w| {
            let parts = self.f_truthful_parts(label, w as u8);
            LabelerLikelihoodParts {
                p_hk: acc.p_hk + parts.p_hk,
                p_huk: acc.p_huk + parts.p_huk,
                p_uhk: acc.p_uhk + parts.p_uhk,
                p_uhuk: acc.p_uhuk + parts.p_uhuk,
            }
        })
    }

    pub fn f_truthful_parts(
        &self,
        label: u8,
        world: u8,
    ) -> LabelerLikelihoodParts {
        let pl = self.p_world()[label];
        let pw = self.p_world()[world];

        // Probability if the informant is helpful but not knowledgeable
        let p_huk = pl * self.p_h() * (1.0 - self.p_k());

        // Probability if the informant is neither helpful nor knowledgeable
        let p_uhuk: f64 = (0..self.n_labels()).fold(0.0, |acc, b| {
            if b as u8 == label {
                acc
            } else {
                let pb = self.p_world()[b];
                acc + pb * pl / (1.0 - pb)
            }
        }) * (1.0 - self.p_h())
            * (1.0 - self.p_k());

        // Probability if the informant is unhelpful and knowledgeable
        let p_uhk = (1.0 - self.p_h()) * self.p_k() * pl / (1.0 - pw);

        if label == world {
            // Probability if the informant is helpful and knowledgeable
            let p_hk = self.p_h() * self.p_k();
            LabelerLikelihoodParts {
                p_hk: p_hk * pw,
                p_huk: p_huk * pw,
                p_uhuk: p_uhuk * pw,
                p_uhk: 0.0,
            }
        } else {
            LabelerLikelihoodParts {
                p_hk: 0.0,
                p_huk: p_huk * pw,
                p_uhuk: p_uhuk * pw,
                p_uhk: p_uhk * pw,
            }
        }
    }

    // Compute the probability of an informant label given no ground truth (world).
    // Optimized manual computation verified below in tests.
    pub fn f_truthless(&self, label: u8) -> f64 {
        // marginalize over states of the world
        (0..self.n_labels())
            .map(|world| self.f_truthful(label, world as u8))
            .sum()
    }

    // Compute the probability of an informant label given the ground truth
    // (world).  Optimized manual computation verified below in tests.
    pub fn f_truthful(&self, label: u8, world: u8) -> f64 {
        let parts = self.f_truthful_parts(label, world);
        parts.sum()
    }

    pub fn support_iter(&self) -> LabelIterator {
        LabelIterator::new(self.n_labels() as u8)
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
    #[allow(clippy::many_single_char_names)]
    fn draw<R: Rng>(&self, mut rng: &mut R) -> Label {
        let w = self.p_world().draw(&mut rng) as u8;
        let h = rng.gen::<f64>() < self.p_h;
        let k = rng.gen::<f64>() < self.p_k;

        let b = if k {
            w
        } else {
            self.p_world().draw(&mut rng) as u8
        };

        let a = if h {
            b
        } else {
            // unhelpful informant will not choose an action consistent with
            // their belief
            loop {
                let a_inner = self.p_world().draw(&mut rng) as u8;
                if a_inner != b {
                    break a_inner;
                }
            }
        };

        Label {
            label: a,
            truth: Some(w),
        }
    }
}

impl Mode<Label> for Labeler {
    fn mode(&self) -> Option<Label> {
        let n_labels = self.n_labels() as u8;

        let start_label = Label::new(0, Some(0));
        let start_f = self.f(&start_label);

        let label = (0..n_labels)
            .fold((start_label, start_f), |acc, obs| {
                (0..n_labels).fold(acc, |(label, f), world| {
                    let label_new = Label::new(obs, Some(world));
                    let f_new = self.f(&label_new);
                    if f_new > f {
                        (label_new, f_new)
                    } else {
                        (label, f)
                    }
                })
            })
            .0;

        Some(label)
    }
}

impl Entropy for Labeler {
    fn entropy(&self) -> f64 {
        self.support_iter().fold(0.0, |acc, x| {
            let p = self.f(&x);
            acc - p * p.ln()
        })
    }
}

impl KlDivergence for Labeler {
    fn kl(&self, other: &Self) -> f64 {
        self.support_iter().fold(0.0, |acc, x| {
            let p = self.f(&x);
            p.mul_add(p.ln() - other.ln_f(&x), acc)
        })
    }
}

// TODO: The size of this suffstats is 224 bits, which is the same amount of
// data as 14 `Label`s. Not very efficient. It might be smarter to use a
// counter.

/// The sufficient statistic for the `Labeler`
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LabelerSuffStat {
    // total number of data observed
    pub n: usize,
    pub counter: HashMap<Label, i32>,
}

impl LabelerSuffStat {
    pub fn new() -> Self {
        LabelerSuffStat {
            n: 0,
            counter: HashMap::new(),
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
        self.n += 1;

        if let Some(count) = self.counter.get_mut(x) {
            *count += 1;
        } else {
            self.counter.insert(x.to_owned(), 1);
        }
    }

    fn forget(&mut self, x: &Label) {
        self.n -= 1;
        match self.counter.get_mut(x) {
            Some(count) => {
                if *count == 1 {
                    self.counter.remove(x);
                } else {
                    *count += 1;
                }
            }
            None => panic!("Tried to forget something never observed"),
        }
    }
}

#[allow(clippy::many_single_char_names)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    use itertools::iproduct;
    use maplit::hashmap;

    const TOL: f64 = 1E-8;
    const N_MH_SAMPLES: u32 = 2_500_000;
    const N_LABELS: usize = 4;

    fn f_truthful(labeler: &Labeler, label: usize, truth: usize) -> f64 {
        let beliefs = 0..N_LABELS;
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
                let p_b_given_k_w: f64 =
                    if k { 1.0 } else { labeler.p_world[b] };

                // left in for clarity
                let p_a_given_h_b = if h {
                    1.0
                } else {
                    labeler.p_world[label] / (1.0 - labeler.p_world[b])
                };

                let p_h = if h { labeler.p_h } else { 1.0 - labeler.p_h };
                let p_k = if k { labeler.p_k } else { 1.0 - labeler.p_k };

                let p_w = labeler.p_world[truth];

                acc + p_a_given_h_b * p_b_given_k_w * p_h * p_k * p_w
            }
        })
    }

    fn f_truthless(labeler: &Labeler, label: usize) -> f64 {
        let beliefs = 0..N_LABELS;
        let worlds = 0..N_LABELS;
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
                let p_b_given_k_w: f64 =
                    if k { 1.0 } else { labeler.p_world[b] };

                // left in for clarity
                let p_a_given_h_b = if h {
                    1.0
                } else {
                    labeler.p_world[label] / (1.0 - labeler.p_world[b])
                };

                let p_h = if h { labeler.p_h } else { 1.0 - labeler.p_h };
                let p_k = if k { labeler.p_k } else { 1.0 - labeler.p_k };

                let p_w = labeler.p_world[w];

                acc + p_a_given_h_b * p_b_given_k_w * p_h * p_k * p_w
            }
        })
    }

    fn draw_anything_but<R: rand::Rng>(
        point: &SimplexPoint,
        b: usize,
        mut rng: &mut R,
    ) -> usize {
        loop {
            let ix: usize = point.draw(&mut rng);
            if ix != b {
                return ix;
            }
        }
    }

    // estimate the probability P(label, truth)
    fn mc_estimate(
        label: usize,
        truth: usize,
        labeler: &Labeler,
        n: u32,
    ) -> f64 {
        let mut rng = rand::thread_rng();

        let mut numer = 0.0;

        for _ in 0..n {
            let w = labeler.p_world().draw(&mut rng);

            if w == truth {
                let k = rng.gen::<f64>() < labeler.p_k();
                let h = rng.gen::<f64>() < labeler.p_h();

                let b: usize = if k {
                    w
                } else {
                    labeler.p_world().draw(&mut rng)
                };
                let a: usize = if h {
                    b
                } else {
                    draw_anything_but(labeler.p_world(), b, &mut rng)
                };

                if a == label {
                    numer += 1.0;
                }
            }
        }
        numer / n as f64
    }

    // estimate the probability P(label)
    fn mc_estimate_truthless(label: usize, labeler: &Labeler, n: u32) -> f64 {
        let mut rng = rand::thread_rng();

        let mut numer = 0.0;

        for _ in 0..n {
            let k = rng.gen::<f64>() < labeler.p_k();
            let h = rng.gen::<f64>() < labeler.p_h();
            let w = labeler.p_world().draw(&mut rng);

            let b: usize = if k {
                w
            } else {
                labeler.p_world().draw(&mut rng)
            };

            let a: usize = if h {
                b
            } else {
                draw_anything_but(labeler.p_world(), b, &mut rng)
            };

            if a == label {
                numer += 1.0;
            }
        }
        numer / f64::from(n)
    }

    fn test_labeler() -> Labeler {
        let p_world = SimplexPoint::new(vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        Labeler::new(0.7, 0.8, p_world)
    }

    #[test]
    fn mode_helpful_knowledgeable() {
        let labeler = test_labeler();
        let mode = labeler.mode().unwrap();
        assert_eq!(mode, Label::new(3, Some(3)));
    }

    #[test]
    fn mode_helpful_unknowledgeable() {
        let labeler = {
            let p_world = SimplexPoint::new(vec![0.1, 0.2, 0.3, 0.4]).unwrap();
            Labeler::new(0.2, 0.8, p_world)
        };
        let mode = labeler.mode().unwrap();
        assert_eq!(mode, Label::new(3, Some(3)));
    }

    #[test]
    fn mode_unhelpful_unknowledgeable() {
        let labeler = {
            let p_world = SimplexPoint::new(vec![0.1, 0.2, 0.3, 0.4]).unwrap();
            Labeler::new(0.2, 0.1, p_world)
        };
        let mode = labeler.mode().unwrap();
        assert_eq!(mode, Label::new(2, Some(3)));
    }

    #[test]
    fn mode_unhelpful_knowledgeable() {
        let labeler = {
            let p_world = SimplexPoint::new(vec![0.1, 0.2, 0.3, 0.4]).unwrap();
            Labeler::new(0.2, 0.1, p_world)
        };
        let mode = labeler.mode().unwrap();
        assert_eq!(mode, Label::new(2, Some(3)));
    }

    #[test]
    fn ps_sum_to_one() {
        let labeler = test_labeler();

        let states = iproduct!((0..N_LABELS), (0..N_LABELS));
        let sum_p: f64 = states
            .map(|(label, truth)| labeler.f_truthful(label as u8, truth as u8))
            .sum();

        assert_relative_eq!(sum_p, 1.0, epsilon = TOL);
    }

    #[test]
    fn p_truthful() {
        let labeler = test_labeler();

        let states = iproduct!((0..N_LABELS), (0..N_LABELS));
        states.for_each(|(label, truth)| {
            assert_relative_eq!(
                labeler.f_truthful(label as u8, truth as u8),
                f_truthful(&labeler, label, truth),
                epsilon = TOL
            );
        })
    }

    #[test]
    fn p_truthless() {
        let labeler = test_labeler();

        (0..N_LABELS).for_each(|label| {
            assert_relative_eq!(
                labeler.f_truthless(label as u8),
                f_truthless(&labeler, label),
                epsilon = TOL
            );
        })
    }

    #[test]
    fn p_truthful_vs_mc_estimate_truthful() {
        let labeler = test_labeler();
        let states = iproduct!((0..N_LABELS), (0..N_LABELS));
        states.for_each(|(label, truth)| {
            let x = Label::new(label as u8, Some(truth as u8));
            assert_relative_eq!(
                labeler.f(&x),
                mc_estimate(label, truth, &labeler, N_MH_SAMPLES),
                epsilon = 1e-3,
            );
        })
    }

    #[test]
    fn p_truthful_vs_mc_estimate_truthless_f() {
        let labeler = test_labeler();
        (0..N_LABELS).for_each(|label| {
            let x = Label::new(label as u8, None);
            assert_relative_eq!(
                labeler.f(&x),
                mc_estimate_truthless(label, &labeler, N_MH_SAMPLES),
                epsilon = 1e-3,
            );
        })
    }

    macro_rules! suffstat_obs_test {
        ($name:ident, $x:expr) => {
            #[test]
            fn $name() {
                let mut stat = LabelerSuffStat::new();
                stat.observe(&$x);

                let target = LabelerSuffStat {
                    n: 1,
                    counter: hashmap! {
                        $x.clone() => 1
                    },
                };

                assert_eq!(target, stat);

                stat.forget(&$x);

                assert_eq!(LabelerSuffStat::new(), stat);
            }
        };
    }

    suffstat_obs_test!(suffstat_observe_forget_0_0, Label::new(0, Some(0)));
    suffstat_obs_test!(suffstat_observe_forget_0_1, Label::new(0, Some(1)));
    suffstat_obs_test!(suffstat_observe_forget_0_2, Label::new(0, Some(2)));
    suffstat_obs_test!(suffstat_observe_forget_0_3, Label::new(0, Some(3)));
    suffstat_obs_test!(suffstat_observe_forget_1_0, Label::new(1, Some(0)));
    suffstat_obs_test!(suffstat_observe_forget_1_1, Label::new(1, Some(1)));
    suffstat_obs_test!(suffstat_observe_forget_1_2, Label::new(1, Some(2)));
    suffstat_obs_test!(suffstat_observe_forget_1_3, Label::new(1, Some(3)));
    suffstat_obs_test!(suffstat_observe_forget_2_0, Label::new(2, Some(0)));
    suffstat_obs_test!(suffstat_observe_forget_2_1, Label::new(2, Some(1)));
    suffstat_obs_test!(suffstat_observe_forget_2_2, Label::new(2, Some(2)));
    suffstat_obs_test!(suffstat_observe_forget_2_3, Label::new(2, Some(3)));
    suffstat_obs_test!(suffstat_observe_forget_3_0, Label::new(3, Some(0)));
    suffstat_obs_test!(suffstat_observe_forget_3_1, Label::new(3, Some(1)));
    suffstat_obs_test!(suffstat_observe_forget_3_2, Label::new(3, Some(2)));
    suffstat_obs_test!(suffstat_observe_forget_3_3, Label::new(3, Some(3)));

    suffstat_obs_test!(suffstat_observe_forget_0_n, Label::new(0, None));
    suffstat_obs_test!(suffstat_observe_forget_1_n, Label::new(1, None));
    suffstat_obs_test!(suffstat_observe_forget_2_n, Label::new(2, None));
    suffstat_obs_test!(suffstat_observe_forget_3_n, Label::new(3, None));
}
