extern crate braid;
extern crate rand;
extern crate serde_yaml;

use rayon::prelude::*;
use std::collections::BTreeMap;

use self::braid::cc::config::EngineUpdateConfig;
use self::braid::cc::{ColAssignAlg, FType, RowAssignAlg};
use self::braid::data::DataSource;
use self::braid::{Codebook, Engine, EngineBuilder, Oracle};
use self::rand::{Rng, SeedableRng, XorShiftRng};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum PpcDataset {
    #[serde(rename = "animals")]
    Animals { nstates: usize, n_iters: usize },
    #[serde(rename = "satellites")]
    Satellites { nstates: usize, n_iters: usize },
    #[serde(rename = "satellites-normed")]
    SatellitesNormed { nstates: usize, n_iters: usize },
}

impl PpcDataset {
    fn name(&self) -> &str {
        match self {
            PpcDataset::Animals { .. } => "animals",
            PpcDataset::Satellites { .. } => "satellites",
            PpcDataset::SatellitesNormed { .. } => "satellites-normed",
        }
    }

    fn nstates(&self) -> usize {
        match self {
            PpcDataset::Animals { nstates, .. } => *nstates,
            PpcDataset::Satellites { nstates, .. } => *nstates,
            PpcDataset::SatellitesNormed { nstates, .. } => *nstates,
        }
    }

    fn n_iters(&self) -> usize {
        match self {
            PpcDataset::Animals { n_iters, .. } => *n_iters,
            PpcDataset::Satellites { n_iters, .. } => *n_iters,
            PpcDataset::SatellitesNormed { n_iters, .. } => *n_iters,
        }
    }

    fn engine<R: Rng>(&self, nstates: usize, mut rng: &mut R) -> Engine {
        let dir = format!("resources/datasets/{}", self.name());
        let data_src = format!("{}/{}.csv", dir, self.name());
        let cb_src = format!("{}/{}.codebook.yaml", dir, self.name());
        let codebook = Codebook::from_yaml(&cb_src);
        EngineBuilder::new(DataSource::Csv(data_src))
            .with_nstates(nstates)
            .with_codebook(codebook)
            .with_rng(XorShiftRng::from_rng(&mut rng).unwrap())
            .build()
            .expect(format!("Couldn't build {} Engine", self.name()).as_str())
    }
}

struct ConfidenceInterval {
    lower: f64,
    upper: f64,
}

impl ConfidenceInterval {
    fn new(upper: f64, lower: f64) -> Self {
        ConfidenceInterval {
            lower: lower,
            upper: upper,
        }
    }

    fn from_vec(xs: Vec<f64>, ci: f64) -> Self {
        if ci <= 0.0 || 1.0 < ci {
            panic!("ci must be in (0, 1]")
        }
        let n = xs.len() as f64;

        let ix_upper: usize = (n * (1.0 - ci) / 2.0).round() as usize;
        let ix_lower: usize = (n * (1.0 - (1.0 - ci) / 2.0)).round() as usize;

        let mut ys = xs.clone();
        ys.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        ConfidenceInterval::new(ys[ix_lower], ys[ix_upper])
    }

    fn contains(&self, x: f64) -> bool {
        self.lower < x && x < self.upper
    }
}

// The average of 1.0 - P(x) for all x in the categorical column
fn ctgrl_pred_dist(oracle: &Oracle, col_ix: usize) -> f64 {
    let nrows = oracle.nrows();
    let mut n: f64 = 0.0;
    let total_distance = (0..nrows).fold(0.0, |acc, row_ix| {
        let x = oracle.get_datum(row_ix, col_ix);
        // FIXME: The distance should be normalize for the number of categories
        if x.as_f64().is_some() {
            n += 1.0;
            let p = oracle.logp(&vec![col_ix], &vec![vec![x]], &None)[0].exp();
            let d = 1.0 - p;
            acc + d
        } else {
            acc
        }
    });
    total_distance / n
}

// proportion of observed continuous values w/in the 50% CI
fn postpred_dist<R: Rng>(
    oracle: &Oracle,
    col_ix: usize,
    n_samples: usize,
    mut rng: &mut R,
) -> f64 {
    let nrows = oracle.nrows();
    let mut n: f64 = 0.0;
    (0..nrows).fold(0.0, |acc, row_ix| {
        let x = oracle.get_datum(row_ix, col_ix);
        if x.is_continuous() {
            n += 1.0;
            let x_obs = x.as_f64().unwrap();
            let x_sim = oracle
                .draw(row_ix, col_ix, Some(n_samples), &mut rng)
                .iter()
                .map(|xi| xi.as_f64().unwrap())
                .collect();
            let ci = ConfidenceInterval::from_vec(x_sim, 0.5);
            if ci.contains(x_obs) {
                acc + 1.0
            } else {
                acc
            }
        } else {
            acc
        }
    }) / n
}

#[derive(Clone, Copy, Serialize)]
pub enum PpcDistance {
    Predictive(f64),
    PostPred(f64),
}

fn ppc<R: Rng>(
    dataset: PpcDataset,
    n_samples: usize,
    mut rng: &mut R,
) -> Vec<PpcDistance> {
    info!("Computing PPCs for {} dataset", dataset.name());
    let mut engine = dataset.engine(dataset.nstates(), &mut rng);
    let config = EngineUpdateConfig::new()
        .with_iters(dataset.n_iters())
        .with_row_alg(RowAssignAlg::FiniteCpu)
        .with_col_alg(ColAssignAlg::Gibbs);

    engine.update(config);

    let oracle = Oracle::from_engine(engine);
    let mut rngs: Vec<XorShiftRng> = (0..oracle.ncols())
        .map(|_| XorShiftRng::from_rng(&mut rng).unwrap())
        .collect();

    (0..oracle.ncols())
        .into_par_iter()
        .zip(rngs.par_iter_mut())
        .map(|(col_ix, mut trng)| match oracle.ftype(col_ix) {
            FType::Continuous => {
                let d = postpred_dist(&oracle, col_ix, n_samples, &mut trng);
                info!("Column ix {} ppc = {}", col_ix, d);
                PpcDistance::PostPred(d)
            }
            FType::Categorical => {
                let d = ctgrl_pred_dist(&oracle, col_ix);
                PpcDistance::Predictive(d)
            }
        }).collect()
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PpcRegressionConfig {
    pub datasets: Vec<PpcDataset>,
    pub n_samples: usize,
}

pub fn run_ppc<R: Rng>(
    config: &PpcRegressionConfig,
    mut rng: &mut R,
) -> BTreeMap<String, Vec<PpcDistance>> {
    let mut results: BTreeMap<String, Vec<PpcDistance>> = BTreeMap::new();
    config.datasets.iter().for_each(|&dataset| {
        let name = String::from(dataset.name());
        let res = ppc(dataset, config.n_samples, &mut rng);
        results.insert(name, res);
    });
    results
}
