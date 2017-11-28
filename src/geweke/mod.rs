extern crate rand;

use self::rand::Rng;



/// The trait that allows samplers to be tested by `GewekeTester`.
pub trait GewekeReady {
    /// The type of the summary output
    type Output;
    /// The type of object that contains sampler settings
    type Settings;

    fn from_prior(settings: &Self::Settings, rng: &mut Rng) -> Self;
    fn resample_data(&mut self, settings: &Self::Settings, rng: &mut Rng);
    fn resample_parameters(&mut self, settings: &Self::Settings, rng: &mut Rng);
    fn summarize(&self) -> Self::Output;
}



/// Verifies the correctness of MCMC algorithms by way of the "joint
/// distribution test (Gewekw FIXME: year).
pub struct GewekeTester<G: GewekeReady> {
    rng        : rand::ThreadRng,
    settings   : G::Settings,
    f_chain_out: Vec<G::Output>,
    p_chain_out: Vec<G::Output>,
}


impl<G: GewekeReady> GewekeTester<G> {
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
            let mut model = G::from_prior(&self.settings, &mut self.rng);
            model.resample_data(&self.settings, &mut self.rng);
            self.f_chain_out.push(model.summarize());
        }
    }

    fn run_posterior_chain(&mut self, n_iter: usize) {
        self.p_chain_out.reserve(n_iter);
        let mut model = G::from_prior(&self.settings, &mut self.rng);
        model.resample_data(&self.settings, &mut self.rng);
        for _ in 0..n_iter {
            model.resample_parameters(&self.settings, &mut self.rng);
            model.resample_data(&self.settings, &mut self.rng);
            self.p_chain_out.push(model.summarize());
        }
    }
}
