extern crate num;
extern crate pbr;
extern crate rand;
extern crate serde_json;

use self::pbr::ProgressBar;
use self::rand::Rng;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::prelude::Write;
use std::path::Path;

/// The trait that allows samplers to be tested by `GewekeTester`.
pub trait GewekeModel: GewekeResampleData + GewekeSummarize {
    /// Draw a new object from the prior
    fn geweke_from_prior(settings: &Self::Settings, rng: &mut impl Rng)
        -> Self;

    /// Update the state of the object by performing 1 MCMC transition
    fn geweke_step(&mut self, settings: &Self::Settings, rng: &mut impl Rng);
}

pub trait GewekeResampleData {
    type Settings;
    fn geweke_resample_data(
        &mut self,
        s: Option<&Self::Settings>,
        rng: &mut impl Rng,
    );
}

pub trait GewekeSummarize {
    fn geweke_summarize(&self) -> BTreeMap<String, f64>;
}

/// Verifies the correctness of MCMC algorithms by way of the "joint
/// distribution test (Geweke FIXME: year).
pub struct GewekeTester<G>
where
    G: GewekeModel + GewekeResampleData + GewekeSummarize,
{
    settings: G::Settings,
    verbose: bool,
    pub f_chain_out: Vec<BTreeMap<String, f64>>,
    pub p_chain_out: Vec<BTreeMap<String, f64>>,
}

impl<G> GewekeTester<G>
where
    G: GewekeModel + GewekeResampleData + GewekeSummarize,
{
    pub fn new(settings: G::Settings) -> Self {
        GewekeTester {
            settings: settings,
            f_chain_out: vec![],
            p_chain_out: vec![],
            verbose: false,
        }
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
            if bts > 1E9 {
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

    pub fn run<R: Rng>(&mut self, n_iter: usize, mut rng: &mut R) {
        self.run_forward_chain(n_iter, &mut rng);
        self.run_posterior_chain(n_iter, &mut rng);
    }

    fn run_forward_chain<R: Rng>(&mut self, n_iter: usize, mut rng: &mut R) {
        if self.verbose {
            println!("Running forward chain...");
        }

        let mut bar = ProgressBar::new(n_iter as u64);
        bar.format("╢▌▌░╟");
        self.f_chain_out.reserve(n_iter);

        for _ in 0..n_iter {
            let mut model = G::geweke_from_prior(&self.settings, &mut rng);
            model.geweke_resample_data(Some(&self.settings), &mut rng);
            self.f_chain_out.push(model.geweke_summarize());
            bar.inc();
        }
        bar.finish_print("done.");
    }

    fn run_posterior_chain<R: Rng>(&mut self, n_iter: usize, mut rng: &mut R) {
        if self.verbose {
            println!("Running posterior chain...");
        }

        let mut bar = ProgressBar::new(n_iter as u64);
        bar.format("╢▌▌░╟");
        self.p_chain_out.reserve(n_iter);

        let mut model = G::geweke_from_prior(&self.settings, &mut rng);
        model.geweke_resample_data(Some(&self.settings), &mut rng);
        for _ in 0..n_iter {
            model.geweke_step(&self.settings, &mut rng);
            model.geweke_resample_data(Some(&self.settings), &mut rng);
            self.p_chain_out.push(model.geweke_summarize());
            bar.inc();
        }
        bar.finish_print("done.");
    }
}
