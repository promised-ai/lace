extern crate rand;

use std::collections::BTreeMap;
use self::rand::Rng;


/// The trait that allows samplers to be tested by `GewekeTester`.
pub trait GewekeModel: GewekeResampleData + GewekeSummarize {
    /// Draw a new object from the prior
    fn geweke_from_prior(settings: &Self::Settings, rng: &mut Rng) -> Self;

    /// Update the state of the object by performing 1 MCMC transition
    fn geweke_step(&mut self, settings: &Self::Settings, rng: &mut Rng);
}


pub trait GewekeResampleData {
    type Settings;
    fn geweke_resample_data(&mut self, s: Option<&Self::Settings>,
                            rng: &mut Rng);
}


pub trait GewekeSummarize {
    fn geweke_summarize(&self) -> BTreeMap<String, f64>;
}



/// Verifies the correctness of MCMC algorithms by way of the "joint
/// distribution test (Geweke FIXME: year).
pub struct GewekeTester<G>
    where G: GewekeModel + GewekeResampleData + GewekeSummarize
{
    rng        : rand::ThreadRng,
    settings   : G::Settings,
    f_chain_out: Vec<BTreeMap<String, f64>>,
    p_chain_out: Vec<BTreeMap<String, f64>>,
}


impl<G> GewekeTester<G>
    where G: GewekeModel + GewekeResampleData + GewekeSummarize
{
    pub fn new(settings: G::Settings) -> Self {
        GewekeTester{rng        : rand::thread_rng(),
                     settings   : settings,
                     f_chain_out: vec![],
                     p_chain_out: vec![]}
    }

    pub fn run(&mut self, n_iter: usize) {
        self.run_forward_chain(n_iter);
        self.run_posterior_chain(n_iter);
    }

    fn run_forward_chain(&mut self, n_iter: usize) {
        self.f_chain_out.reserve(n_iter);
        for _ in 0..n_iter {
            let mut model = G::geweke_from_prior(&self.settings, &mut self.rng);
            model.geweke_resample_data(Some(&self.settings), &mut self.rng);
            self.f_chain_out.push(model.geweke_summarize());
        }
    }

    fn run_posterior_chain(&mut self, n_iter: usize) {
        self.p_chain_out.reserve(n_iter);
        let mut model = G::geweke_from_prior(&self.settings, &mut self.rng);
        model.geweke_step(&self.settings, &mut self.rng);
        for _ in 0..n_iter {
            model.geweke_step(&self.settings, &mut self.rng);
            model.geweke_resample_data(Some(&self.settings), &mut self.rng);
            self.p_chain_out.push(model.geweke_summarize());
        }
    }
}
