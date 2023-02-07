use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::prelude::Write;
use std::path::Path;

use lace_stats::EmpiricalCdf;
use lace_utils::transpose_mapvec;
use indicatif::ProgressBar;
use rand::Rng;
use serde::Serialize;

use crate::traits::*;

/// Verifies the correctness of MCMC algorithms by way of the "joint
/// distribution test
pub struct GewekeTester<G>
where
    G: GewekeModel + GewekeResampleData + GewekeSummarize,
    G::Summary: Into<BTreeMap<String, f64>> + Clone,
{
    settings: G::Settings,
    pub verbose: bool,
    pub f_chain_out: Vec<G::Summary>,
    pub p_chain_out: Vec<G::Summary>,
}

#[derive(Serialize, Debug, Clone)]
pub struct GewekeResult {
    pub forward: BTreeMap<String, Vec<f64>>,
    pub posterior: BTreeMap<String, Vec<f64>>,
}

impl fmt::Display for GewekeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Geweke Errors")?;
        write!(f, "━━━━━━━━━━━━━")?;
        let errs: BTreeMap<String, f64> = self.aucs().collect();
        let width = errs.keys().fold(0_usize, |len, k| len.max(k.len()));
        write!(f, "\n{:width$}  Value", "Stat", width = width)?;
        write!(f, "\n{:width$}  ━━━━━", "━━━━", width = width)?;
        errs.iter()
            .try_for_each(|(k, auc)| write!(f, "\n{k:width$}  {auc}"))
    }
}

impl GewekeResult {
    pub fn aucs<'a>(&'a self) -> Box<dyn Iterator<Item = (String, f64)> + 'a> {
        let iter = self.forward.keys().map(move |k| {
            let cdf_f = EmpiricalCdf::new(self.forward.get(k).unwrap());
            let cdf_p = EmpiricalCdf::new(self.posterior.get(k).unwrap());
            (String::from(k), cdf_f.auc(&cdf_p))
        });

        Box::new(iter)
    }

    pub fn ks(&self) -> BTreeMap<String, f64> {
        use lace_stats::rv::misc::{ks_two_sample, KsAlternative, KsMode};

        self.forward
            .keys()
            .map(|k| {
                let (_, p) = ks_two_sample(
                    self.forward.get(k).unwrap(),
                    self.posterior.get(k).unwrap(),
                    KsMode::Auto,
                    KsAlternative::TwoSided,
                )
                .unwrap();
                // TODO: return p value instead
                (k.clone(), p)
            })
            .collect()
    }

    pub fn report(&self) {
        println!("{self}")
    }
}

impl<G> GewekeTester<G>
where
    G: GewekeModel + GewekeResampleData + GewekeSummarize,
    G::Summary: Into<BTreeMap<String, f64>> + Clone,
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
        // TODO: would be nice if we didn't have to clone the summaries here
        let forward = transpose_mapvec(
            &self
                .f_chain_out
                .iter()
                .map(|val| val.to_owned().into())
                .collect::<Vec<_>>(),
        );

        let posterior = transpose_mapvec(
            &self
                .p_chain_out
                .iter()
                .map(|val| val.to_owned().into())
                .collect::<Vec<_>>(),
        );

        GewekeResult { forward, posterior }
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
