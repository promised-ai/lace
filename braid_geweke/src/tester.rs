use std::collections::BTreeMap;
use std::fs::File;
use std::io::prelude::Write;
use std::path::Path;

use braid_stats::EmpiricalCdf;
use braid_utils::misc::transpose_mapvec;
use indicatif::ProgressBar;
use rand::Rng;
use serde::Serialize;

use crate::traits::*;

/// Verifies the correctness of MCMC algorithms by way of the "joint
/// distribution test
pub struct GewekeTester<G>
where
    G: GewekeModel + GewekeResampleData + GewekeSummarize,
{
    settings: G::Settings,
    pub verbose: bool,
    pub f_chain_out: Vec<BTreeMap<String, f64>>,
    pub p_chain_out: Vec<BTreeMap<String, f64>>,
}

#[derive(Serialize, Debug, Clone)]
pub struct GewekeResult {
    forward: Vec<BTreeMap<String, f64>>,
    posterior: Vec<BTreeMap<String, f64>>,
}

impl GewekeResult {
    pub fn aucs(&self) -> BTreeMap<String, f64> {
        let forward_t = transpose_mapvec(&self.forward);
        let posterior_t = transpose_mapvec(&self.posterior);

        let mut aucs = BTreeMap::new();
        for key in forward_t.keys() {
            let k = key.clone();
            let cdf_f = EmpiricalCdf::new(&forward_t.get(&k).unwrap());
            let cdf_p = EmpiricalCdf::new(&posterior_t.get(&k).unwrap());
            let auc: f64 = cdf_f.auc(&cdf_p);
            aucs.insert(key.clone(), auc);
        }
        aucs
    }

    pub fn report(&self) {
        println!("Geweke AUCs\n-----------");
        self.aucs()
            .iter()
            .for_each(|(k, auc)| println!("  {}: {}", k, auc));
    }
}

impl<G> GewekeTester<G>
where
    G: GewekeModel + GewekeResampleData + GewekeSummarize,
{
    pub fn new(settings: G::Settings) -> Self {
        GewekeTester {
            settings,
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
        let res = self.result();
        let j = serde_yaml::to_string(&res).unwrap();
        let mut file = File::create(path).unwrap();
        let _nbytes = file.write(j.as_bytes()).unwrap();
    }

    pub fn run<R: Rng>(
        &mut self,
        n_iter: usize,
        lag: Option<usize>,
        mut rng: &mut R,
    ) {
        self.run_forward_chain(n_iter, &mut rng);
        self.run_posterior_chain(n_iter, lag.unwrap_or(1), &mut rng);
        if self.verbose {
            self.result().report()
        }
    }

    fn run_forward_chain<R: Rng>(&mut self, n_iter: usize, mut rng: &mut R) {
        let pb = ProgressBar::new(n_iter as u64);
        self.f_chain_out.reserve(n_iter);

        for _ in 0..n_iter {
            let mut model = G::geweke_from_prior(&self.settings, &mut rng);
            model.geweke_resample_data(Some(&self.settings), &mut rng);
            self.f_chain_out
                .push(model.geweke_summarize(&self.settings));
            pb.inc(1);
        }
        pb.finish_and_clear();
    }

    fn run_posterior_chain<R: Rng>(
        &mut self,
        n_iter: usize,
        lag: usize,
        mut rng: &mut R,
    ) {
        let pb = ProgressBar::new(n_iter as u64);
        self.p_chain_out.reserve(n_iter);

        let mut model = G::geweke_from_prior(&self.settings, &mut rng);
        model.geweke_resample_data(Some(&self.settings), &mut rng);
        for _ in 0..n_iter {
            for _ in 0..lag {
                model.geweke_step(&self.settings, &mut rng);
                model.geweke_resample_data(Some(&self.settings), &mut rng);
            }
            self.p_chain_out
                .push(model.geweke_summarize(&self.settings));
            pb.inc(1);
        }
        pb.finish_and_clear();
    }
}
