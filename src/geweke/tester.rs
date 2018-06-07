extern crate num;
extern crate indicatif;
extern crate rand;
extern crate serde_yaml;

use self::indicatif::ProgressBar;
use self::rand::Rng;
use geweke::traits::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::prelude::Write;
use std::path::Path;

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

#[derive(Serialize)]
pub struct GewekeResult {
    forward: Vec<BTreeMap<String, f64>>,
    posterior: Vec<BTreeMap<String, f64>>,
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

    pub fn result(&self) -> GewekeResult {
        GewekeResult {
            forward: self.f_chain_out.clone(),
            posterior: self.p_chain_out.clone(),
        }
    }

    /// Output results as json
    pub fn save(&self, path: &Path) {
        if self.verbose {
            let path_str = path.to_str().unwrap();
            println!("Writing to '{}'.", path_str);
        }
        let res = self.result();
        let j = serde_yaml::to_string(&res).unwrap();
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

        let pb = ProgressBar::new(n_iter as u64);
        self.f_chain_out.reserve(n_iter);

        for _ in 0..n_iter {
            let mut model = G::geweke_from_prior(&self.settings, &mut rng);
            model.geweke_resample_data(Some(&self.settings), &mut rng);
            self.f_chain_out.push(model.geweke_summarize());
            pb.inc(1);
        }
        pb.finish_and_clear();
    }

    fn run_posterior_chain<R: Rng>(&mut self, n_iter: usize, mut rng: &mut R) {
        if self.verbose {
            println!("Running posterior chain...");
        }

        let pb = ProgressBar::new(n_iter as u64);
        self.p_chain_out.reserve(n_iter);

        let mut model = G::geweke_from_prior(&self.settings, &mut rng);
        model.geweke_resample_data(Some(&self.settings), &mut rng);
        for _ in 0..n_iter {
            model.geweke_step(&self.settings, &mut rng);
            model.geweke_resample_data(Some(&self.settings), &mut rng);
            self.p_chain_out.push(model.geweke_summarize());
            pb.inc(1);
        }
        pb.finish_and_clear();
    }
}
