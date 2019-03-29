//! Defines the `Feature` trait for cross-categorization columns
extern crate braid_stats;
extern crate num;
extern crate rand;
extern crate rv;
extern crate serde;
extern crate serde_yaml;

use std::collections::BTreeMap;

use self::rand::Rng;

use self::braid_stats::prior::{Csd, CsdHyper, Ng, NigHyper};
use self::rv::data::DataOrSuffStat;
use self::rv::dist::{Categorical, Gaussian};
use self::rv::traits::*;
use cc::assignment::Assignment;
use cc::container::DataContainer;
use cc::transition::ViewTransition;
use cc::ConjugateComponent;
use dist::traits::AccumScore;
use dist::{BraidDatum, BraidLikelihood, BraidPrior, BraidStat};
use geweke::traits::*;
use misc::{mean, std};

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(deserialize = "X: serde::de::DeserializeOwned"))]
/// A partitioned columns of data
pub struct Column<X, Fx, Pr>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Pr: BraidPrior<X, Fx>,
    Fx::Stat: BraidStat,
{
    pub id: usize,
    // TODO: Figure out a way to optionally serialize data
    pub data: DataContainer<X>,
    pub components: Vec<ConjugateComponent<X, Fx>>,
    pub prior: Pr,
    // TODO: pointers to data on GPU
}

impl<X, Fx, Pr> Column<X, Fx, Pr>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Pr: BraidPrior<X, Fx>,
    Fx::Stat: BraidStat,
{
    pub fn new(id: usize, data: DataContainer<X>, prior: Pr) -> Self {
        Column {
            id,
            data,
            components: Vec::new(),
            prior,
        }
    }

    pub fn len(&self) -> usize {
        // XXX: this will fail on features with dropped data
        self.data.len()
    }

    pub fn get_components(&self) -> Vec<Fx> {
        self.components.iter().map(|cpnt| cpnt.fx.clone()).collect()
    }
}

/// A Cross-Categorization feature/column
pub trait Feature {
    /// The feature id
    fn id(&self) -> usize;
    /// The number of rows
    fn len(&self) -> usize;
    /// The number of components
    fn k(&self) -> usize;

    /// score each datum under component `k` and add to the corresponding
    /// entries in `scores`
    fn accum_score(&self, scores: &mut Vec<f64>, k: usize);
    /// Draw `k` components from the prior
    fn init_components(&mut self, k: usize, rng: &mut impl Rng);
    /// Redraw the component parameters from the posterior distribution,
    /// f(θ|x<sub>k</sub>).
    fn update_components(&mut self, rng: &mut impl Rng);
    /// Create new components and assign data to them accoring to the
    /// assignment.
    fn reassign(&mut self, asgn: &Assignment, rng: &mut impl Rng);
    /// The log likelihood of the datum in the Feature under the current
    /// assignment
    fn score(&self) -> f64;
    /// The log likelihood of the datum in the Feature under a different
    /// assignment
    fn asgn_score(&self, asgn: &Assignment) -> f64;
    /// Draw new prior parameters from the posterior, p(φ|θ)
    fn update_prior_params(&mut self, rng: &mut impl Rng);
    /// Draw an empty component from the prior and append it to the components
    /// vector
    fn append_empty_component(&mut self, rng: &mut impl Rng);
    /// Remove the component at index `k`
    fn drop_component(&mut self, k: usize);
    /// The log likelihood of the datum at `row_ix` under the component at
    /// index `k`
    fn logp_at(&self, row_ix: usize, k: usize) -> Option<f64>;
    /// The log posterior predictive function of the datum at `row_ix` under
    /// the component at index `k`
    fn predictive_score_at(&self, row_ix: usize, k: usize) -> f64;
    /// The marginal likelihood of the datum on its own
    fn singleton_score(&self, row_ix: usize) -> f64;

    /// Have the component at index `k` observe the datum at row `row_ix`
    fn observe_datum(&mut self, row_ix: usize, k: usize);
    /// Have the component at index `k` forget the datum at row `row_ix`
    fn forget_datum(&mut self, row_ix: usize, k: usize);
}

fn draw_cpnts<X, Fx, Pr>(
    prior: &Pr,
    k: usize,
    mut rng: &mut impl Rng,
) -> Vec<ConjugateComponent<X, Fx>>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Pr: BraidPrior<X, Fx>,
    Fx::Stat: BraidStat,
{
    (0..k)
        .map(|_| ConjugateComponent::new(prior.draw(&mut rng)))
        .collect()
}

#[allow(dead_code)]
impl<X, Fx, Pr> Feature for Column<X, Fx, Pr>
where
    X: BraidDatum,
    Fx: BraidLikelihood<X>,
    Pr: BraidPrior<X, Fx>,
    Fx::Stat: BraidStat,
{
    fn id(&self) -> usize {
        self.id
    }

    fn accum_score(&self, mut scores: &mut Vec<f64>, k: usize) {
        // TODO: Decide when to use parallel or GPU
        self.components[k].accum_score(
            &mut scores,
            &self.data.data,
            &self.data.present,
        );
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn k(&self) -> usize {
        self.components.len()
    }

    fn init_components(&mut self, k: usize, mut rng: &mut impl Rng) {
        self.components = draw_cpnts(&self.prior, k, &mut rng);
    }

    fn update_components(&mut self, mut rng: &mut impl Rng) {
        let prior = self.prior.clone();
        self.components.iter_mut().for_each(|cpnt| {
            cpnt.fx = prior.posterior(&cpnt.obs()).draw(&mut rng);
        })
    }

    fn reassign(&mut self, asgn: &Assignment, mut rng: &mut impl Rng) {
        // re-draw empty k componants.
        // TODO: We should consider a way to do this without drawing from the
        // prior because we're just going to overwrite what we draw in a fe
        // lines. Wasted cycles.
        let mut components = draw_cpnts(&self.prior, asgn.ncats, &mut rng);

        // have the component obseve all their data
        self.data
            .zip()
            .zip(asgn.iter())
            .for_each(|((x, present), z)| {
                if *present {
                    components[*z].observe(x)
                }
            });

        // Set the components
        self.components = components;

        // update the component according to the posterior
        self.update_components(&mut rng);
    }

    fn score(&self) -> f64 {
        self.components
            .iter()
            .fold(0.0, |acc, cpnt| acc + self.prior.ln_m(&cpnt.obs()))
    }

    fn asgn_score(&self, asgn: &Assignment) -> f64 {
        let xks = self.data.group_by(asgn);
        xks.iter().fold(0.0, |acc, xk| {
            let data = DataOrSuffStat::Data(xk);
            acc + self.prior.ln_m(&data)
        })
    }

    fn update_prior_params(&mut self, mut rng: &mut impl Rng) {
        let components: Vec<&Fx> =
            self.components.iter().map(|cpnt| &cpnt.fx).collect();
        self.prior.update_prior(&components, &mut rng);
    }

    fn append_empty_component(&mut self, mut rng: &mut impl Rng) {
        let cpnt = ConjugateComponent::new(self.prior.draw(&mut rng));
        self.components.push(cpnt);
    }

    fn drop_component(&mut self, k: usize) {
        // cpnt goes out of scope and is dropped (Hopefully)
        let _cpnt = self.components.remove(k);
    }

    fn logp_at(&self, row_ix: usize, k: usize) -> Option<f64> {
        if self.data.present[row_ix] {
            let x = &self.data.data[row_ix];
            Some(self.components[k].ln_f(x))
        } else {
            None
        }
    }

    fn predictive_score_at(&self, row_ix: usize, k: usize) -> f64 {
        if self.data.present[row_ix] {
            self.prior
                .ln_pp(&self.data[row_ix], &self.components[k].obs())
        } else {
            0.0
        }
    }

    fn singleton_score(&self, row_ix: usize) -> f64 {
        if self.data.present[row_ix] {
            let mut stat = self.components[0].fx.empty_suffstat();
            stat.observe(&self.data.data[row_ix]);
            self.prior.ln_m(&DataOrSuffStat::SuffStat(&stat))
        } else {
            0.0
        }
    }

    fn observe_datum(&mut self, row_ix: usize, k: usize) {
        if self.data.present[row_ix] {
            let x = &self.data[row_ix];
            self.components[k].observe(x);
        }
    }

    fn forget_datum(&mut self, row_ix: usize, k: usize) {
        if self.data.present[row_ix] {
            let x = &self.data[row_ix];
            self.components[k].forget(x);
        }
    }
}

// Geweke implementations
// ======================
#[derive(Clone)]
pub struct ColumnGewekeSettings {
    asgn: Assignment,
    transitions: Vec<ViewTransition>,
    fixed_prior: bool,
}

impl ColumnGewekeSettings {
    pub fn new(asgn: Assignment, transitions: Vec<ViewTransition>) -> Self {
        let fixed_prior = transitions
            .iter()
            .find(|t| **t == ViewTransition::FeaturePriors)
            .is_none();

        ColumnGewekeSettings {
            asgn: asgn,
            transitions: transitions,
            fixed_prior: fixed_prior,
        }
    }
}

// Continuous
// ----------
impl GewekeModel for Column<f64, Gaussian, Ng> {
    fn geweke_from_prior(
        settings: &Self::Settings,
        mut rng: &mut impl Rng,
    ) -> Self {
        let f = Gaussian::new(0.0, 1.0).unwrap();
        let xs = f.sample(settings.asgn.len(), &mut rng);
        let data = DataContainer::new(xs); // initial data is re-sampled anyway
        let prior = if settings.fixed_prior {
            Ng::new(0.0, 1.0, 1.0, 1.0, NigHyper::geweke())
        } else {
            NigHyper::geweke().draw(&mut rng)
        };
        let mut col = Column::new(0, data, prior);
        col.init_components(settings.asgn.ncats, &mut rng);
        col
    }

    /// Update the state of the object by performing 1 MCMC transition
    fn geweke_step(
        &mut self,
        settings: &Self::Settings,
        mut rng: &mut impl Rng,
    ) {
        self.update_components(&mut rng);
        if !settings.fixed_prior {
            self.update_prior_params(&mut rng);
        }
    }
}

impl GewekeResampleData for Column<f64, Gaussian, Ng> {
    type Settings = ColumnGewekeSettings;
    fn geweke_resample_data(
        &mut self,
        settings: Option<&Self::Settings>,
        rng: &mut impl Rng,
    ) {
        let s = settings.unwrap();
        for (i, &k) in s.asgn.asgn.iter().enumerate() {
            self.data[i] = self.components[k].draw(rng);
        }
    }
}

impl GewekeSummarize for Column<f64, Gaussian, Ng> {
    fn geweke_summarize(
        &self,
        settings: &ColumnGewekeSettings,
    ) -> BTreeMap<String, f64> {
        let x_mean = mean(&self.data.data);
        let x_std = std(&self.data.data);

        let mus: Vec<f64> = self.components.iter().map(|c| c.fx.mu).collect();

        let sigmas: Vec<f64> =
            self.components.iter().map(|c| c.fx.sigma).collect();

        let mu_mean = mean(&mus);
        let sigma_mean = mean(&sigmas);

        let mut stats: BTreeMap<String, f64> = BTreeMap::new();

        stats.insert(String::from("x mean"), x_mean);
        stats.insert(String::from("x std"), x_std);
        stats.insert(String::from("mu mean"), mu_mean);
        stats.insert(String::from("sigma mean"), sigma_mean);
        if !settings.fixed_prior {
            stats.insert(String::from("NIG m"), self.prior.ng.m);
            stats.insert(String::from("NIG r"), self.prior.ng.r);
            stats.insert(String::from("NIG s"), self.prior.ng.s);
            stats.insert(String::from("NIG v"), self.prior.ng.v);
        }

        stats
    }
}

// Categorical
// -----------
impl GewekeModel for Column<u8, Categorical, Csd> {
    fn geweke_from_prior(
        settings: &Self::Settings,
        mut rng: &mut impl Rng,
    ) -> Self {
        let k = 5;
        let f = Categorical::uniform(k);
        let xs = f.sample(settings.asgn.len(), &mut rng);
        let data = DataContainer::new(xs); // initial data is resampled anyway
        let prior = if settings.fixed_prior {
            Csd::new(1.0, k, CsdHyper::geweke())
        } else {
            CsdHyper::geweke().draw(k, &mut rng)
        };
        let mut col = Column::new(0, data, prior);
        col.init_components(settings.asgn.ncats, &mut rng);
        col
    }

    /// Update the state of the object by performing 1 MCMC transition
    fn geweke_step(
        &mut self,
        settings: &Self::Settings,
        mut rng: &mut impl Rng,
    ) {
        self.update_components(&mut rng);
        if !settings.fixed_prior {
            self.update_prior_params(&mut rng);
        }
    }
}

// TODO: Make a macro for this
impl GewekeResampleData for Column<u8, Categorical, Csd> {
    type Settings = ColumnGewekeSettings;
    fn geweke_resample_data(
        &mut self,
        settings: Option<&Self::Settings>,
        rng: &mut impl Rng,
    ) {
        let s = settings.unwrap();
        for (i, &k) in s.asgn.asgn.iter().enumerate() {
            self.data[i] = self.components[k].draw(rng);
        }
    }
}

impl GewekeSummarize for Column<u8, Categorical, Csd> {
    fn geweke_summarize(
        &self,
        settings: &ColumnGewekeSettings,
    ) -> BTreeMap<String, f64> {
        let x_sum = self.data.data.iter().fold(0, |acc, x| acc + x);

        fn sum_sq(logws: &[f64]) -> f64 {
            logws.iter().fold(0.0, |acc, lw| acc + lw.exp().powi(2))
        }

        fn weight_mean(logws: &[f64]) -> f64 {
            let k = logws.len() as f64;
            logws.iter().fold(0.0, |acc, lw| acc + lw) / k
        }

        let k = self.components.len() as f64;
        let mean_hrm: f64 = self
            .components
            .iter()
            .fold(0.0, |acc, cpnt| acc + sum_sq(&cpnt.fx.ln_weights))
            / k;

        let mean_weight: f64 = self
            .components
            .iter()
            .fold(0.0, |acc, cpnt| acc + weight_mean(&cpnt.fx.ln_weights))
            / k;

        let mut stats: BTreeMap<String, f64> = BTreeMap::new();

        stats.insert(String::from("x sum"), x_sum as f64);
        stats.insert(String::from("weight sum squares"), mean_hrm as f64);
        stats.insert(String::from("weight mean"), mean_weight as f64);
        if !settings.fixed_prior {
            stats.insert(String::from("prior alpha"), self.prior.symdir.alpha);
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use self::rv::dist::Gaussian;
    use super::braid_stats::prior::ng::NigHyper;
    use super::*;
    use cc::AssignmentBuilder;

    #[test]
    fn score_and_asgn_score_equivalency() {
        let nrows = 100;
        let mut rng = rand::thread_rng();
        let g = Gaussian::standard();
        let prior = Ng::new(0.0, 1.0, 1.0, 1.0, NigHyper::default());
        for _ in 0..100 {
            let asgn = AssignmentBuilder::new(nrows).build(&mut rng).unwrap();
            let xs: Vec<f64> = g.sample(nrows, &mut rng);
            let data = DataContainer::new(xs);
            let mut feature = Column::new(0, data, prior.clone());
            feature.reassign(&asgn, &mut rng);

            assert_relative_eq!(
                feature.score(),
                feature.asgn_score(&asgn),
                epsilon = 1E-8
            );
        }
    }
}
