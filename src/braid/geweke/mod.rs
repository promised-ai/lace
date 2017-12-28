extern crate num;
extern crate serde_json;
extern crate rand;
extern crate progress;

use std::io::prelude::Write;
use std::fs::File;
use std::path::Path;
use std::collections::BTreeMap;
use self::num::traits::FromPrimitive;
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
    rng            : rand::ThreadRng,
    settings       : G::Settings,
    verbose        : bool,
    pub f_chain_out: Vec<BTreeMap<String, f64>>,
    pub p_chain_out: Vec<BTreeMap<String, f64>>,
}


impl<G> GewekeTester<G>
    where G: GewekeModel + GewekeResampleData + GewekeSummarize
{
    pub fn new(settings: G::Settings) -> Self {
        GewekeTester{rng        : rand::thread_rng(),
                     settings   : settings,
                     f_chain_out: vec![],
                     p_chain_out: vec![],
                     verbose: false}
    }

    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Output results as json
    pub fn save(&self, path: &Path) {
        if self.verbose {
            let path_str = path.to_str().unwrap();
            println!("Writing to '{}'.", path_str);
        }
        let mut res = BTreeMap::new();
        res.insert("forward", &self.f_chain_out);
        res.insert("posterior", &self.p_chain_out);

        let j = serde_json::to_string(&res).unwrap();
        let mut file = File::create(path).unwrap();
        let nbytes = file.write(j.as_bytes()).unwrap();
        if self.verbose {
            let mut bts = nbytes as f64;
            let mut bstr = "bytes";
            if bts > 1E9{
                bstr = "gb";
                bts /= 1E9;
            } else if bts > 1E6 {
                bstr = "mb";
                bts /= 1E6;
            } else if bts > 1E3 {
                bstr = "kb";
                bts /= 1E3;
            } 
            println!("Saved {} {}.", bts, bstr);
        }
    }

    pub fn run(&mut self, n_iter: usize) {
        self.run_forward_chain(n_iter);
        self.run_posterior_chain(n_iter);
    }

    fn run_forward_chain(&mut self, n_iter: usize) {
        let mut bar = progress::Bar::new();
        bar.set_job_title("Forward chain");
        self.f_chain_out.reserve(n_iter);
        for i in 0..n_iter {
            let mut model = G::geweke_from_prior(&self.settings, &mut self.rng);
            model.geweke_resample_data(Some(&self.settings), &mut self.rng);
            self.f_chain_out.push(model.geweke_summarize());
            let perc = (i as f32 + 1.0) / (n_iter as f32) * 100.0;
            bar.reach_percent(FromPrimitive::from_f32(perc).unwrap());
        }
        println!("Done.")
    }

    fn run_posterior_chain(&mut self, n_iter: usize) {
        let mut bar = progress::Bar::new();
        bar.set_job_title("Posterior chain");
        self.p_chain_out.reserve(n_iter);
        let mut model = G::geweke_from_prior(&self.settings, &mut self.rng);
        model.geweke_step(&self.settings, &mut self.rng);
        for i in 0..n_iter {
            model.geweke_step(&self.settings, &mut self.rng);
            model.geweke_resample_data(Some(&self.settings), &mut self.rng);
            self.p_chain_out.push(model.geweke_summarize());
            let perc = (i as f32 + 1.0) / (n_iter as f32) * 100.0;
            bar.reach_percent(FromPrimitive::from_f32(perc).unwrap());
        }
        println!("Done.")
    }
}
