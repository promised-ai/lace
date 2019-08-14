use braid_stats::labeler::{
    sf_loglike, Label, Labeler, LabelerPrior, LabelerSuffStat,
};
use braid_stats::seq::SobolSeq;
use braid_stats::simplex::{uvec_to_simplex, SimplexPoint};
use braid_utils::misc::logsumexp;
use maplit::hashmap;
use rand::{FromEntropy, Rng};
use rand_xoshiro::Xoshiro256Plus;
use rv::data::DataOrSuffStat;
use rv::traits::{ConjugatePrior, Rv};
use std::f64::NEG_INFINITY;

const N_LABELS: usize = 6;

use rv::dist::{Beta, Dirichlet};

struct LabelerQ {
    pw: Dirichlet,
    ph: Beta,
    pk: Beta,
}

impl Rv<Labeler> for LabelerQ {
    fn ln_f(&self, x: &Labeler) -> f64 {
        self.ph.ln_f(&x.p_h())
            + self.pk.ln_f(&x.p_k())
            + self.pw.ln_f(x.p_world().point())
    }

    fn draw<R: rand::Rng>(&self, mut rng: &mut R) -> Labeler {
        Labeler::new(
            self.pk.draw(&mut rng),
            self.ph.draw(&mut rng),
            SimplexPoint::new_unchecked(self.pw.draw(&mut rng)),
        )
    }
}

impl LabelerQ {
    fn build(stat: &LabelerSuffStat, n_labels: u8) -> LabelerQ {
        let ndims = n_labels as usize;
        let pw = (n_labels as f64).recip();
        let mut weights = vec![0.5; ndims];

        stat.counter.iter().for_each(|(key, ct)| {
            let k = *ct as f64;
            if let Label {
                truth: Some(ix), ..
            } = key
            {
                weights[(*ix) as usize] += k;
            } else {
                (0..ndims).for_each(|i| weights[i] += pw * k);
            }
        });

        let labeler = {
            let zw: f64 = weights.iter().sum();
            let inner = weights.iter().map(|&w| w / zw).collect();
            let p_world = SimplexPoint::new(inner).unwrap();
            Labeler::new(0.5, 0.5, p_world)
        };

        let mut ka: f64 = 0.5;
        let mut kb: f64 = 0.5;
        let mut ha: f64 = 0.5;
        let mut hb: f64 = 0.5;

        stat.counter.iter().for_each(|(key, ct)| {
            let k = *ct as f64;
            let parts = match key {
                Label {
                    label,
                    truth: Some(truth),
                } => labeler.f_truthful_parts(*label, *truth),
                Label { label, truth: None } => {
                    labeler.f_truthless_parts(*label)
                }
            };
            let ph = parts.p_helpful();
            let pk = parts.p_knowledgeable();

            ka += pk * k;
            kb += (1.0 - pk) * k;

            ha += ph * k;
            hb += (1.0 - ph) * k;
        });

        LabelerQ {
            pw: Dirichlet::new(weights).unwrap(),
            ph: Beta::new_unchecked(ha, hb),
            pk: Beta::new_unchecked(ka, kb),
        }
    }
}

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
    // let q = LabelerPrior {
    //     pr_k: rv::dist::Kumaraswamy::uniform(),
    //     pr_h: rv::dist::Kumaraswamy::uniform(),
    //     pr_world: rv::dist::SymmetricDirichlet::new(1.0, N_LABELS).unwrap(),
    // };
    let q = LabelerQ::build(&stat, N_LABELS as u8);
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
            Label::new(1, Some(0)) => 10,
            Label::new(1, Some(0)) => 5,
            Label::new(0, Some(1)) => 2,
        },
    };

    println!("Production ln m(x): {}", prod(&stat));
    for n in [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000].iter() {
        for _ in 0..5 {
            println!(
                "{} samples - QMC: {} MC: {}, IS: {}, Prod: {}",
                n,
                qmc(&stat, *n),
                mc(&stat, *n, &mut rng),
                importance(&stat, *n, &mut rng),
                prod(&stat),
            );
        }
    }
}
