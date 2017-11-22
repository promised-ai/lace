extern crate rand;

use self::rand::Rng;

pub trait GewekeReady {
    type Output;

    fn from_prior(rng: &mut Rng) -> Self;
    fn resample_data(&mut self, rng: &mut Rng);
    fn resample_parameters(&mut self, rng: &mut Rng);
    fn summarize(&self) -> Self::Output;
}


pub struct GewekeTester<G: GewekeReady> {
    rng: rand::ThreadRng,
    f_chain_out: Vec<G::Output>,
    p_chain_out: Vec<G::Output>,
}


impl<G: GewekeReady> GewekeTester<G> {
    pub fn new() -> Self {
        GewekeTester{rng: rand::thread_rng(),
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
            let mut model = G::from_prior(&mut self.rng);
            model.resample_data(&mut self.rng);
            self.f_chain_out.push(model.summarize());
        }
    }

    fn run_posterior_chain(&mut self, n_iter: usize) {
        self.p_chain_out.reserve(n_iter);
        let mut model = G::from_prior(&mut self.rng);
        model.resample_data(&mut self.rng);
        for _ in 0..n_iter {
            model.resample_parameters(&mut self.rng);
            model.resample_data(&mut self.rng);
            self.p_chain_out.push(model.summarize());
        }
    }
}
