extern crate braid;
extern crate braid_codebook;
extern crate braid_stats;
extern crate log;
extern crate maplit;
extern crate rand;
extern crate rv;
extern crate serde;
extern crate serde_yaml;

use std::collections::BTreeMap;
use std::f64::consts::PI;

use braid::cc::{ColModel, Column, DataContainer, State};
use braid::{Engine, Given, Oracle};
use braid_codebook::codebook::{Codebook, ColMetadata, ColType, SpecType};
use braid_stats::perm::gauss_perm_test;
use braid_stats::prior::Ng;
use log::info;
use maplit::btreemap;
use rand::distributions::{Normal, Uniform};
use rand::{Rng, SeedableRng, XorShiftRng};
use rv::dist::Gamma;
use serde::{Deserialize, Serialize};

const SHAPE_SCALE: f64 = 1_000.0;

pub struct Data2d {
    xs: DataContainer<f64>,
    ys: DataContainer<f64>,
}

impl Data2d {
    fn new(xs: Vec<f64>, ys: Vec<f64>) -> Self {
        if xs.len() != ys.len() {
            panic!("xs and ys must be same length");
        }

        Data2d {
            xs: DataContainer::new(xs),
            ys: DataContainer::new(ys),
        }
    }

    /// Convert to a vector of 2-length `Vec<f64>`s.
    fn to_vec(&self) -> Vec<Vec<f64>> {
        self.xs
            .data
            .iter()
            .zip(self.ys.data.iter())
            .map(|(x, y)| vec![*x, *y])
            .collect()
    }

    /// Scale the data by a constant factor, `c`.
    fn scale(&self, c: f64) -> Self {
        let xs: Vec<f64> = self.xs.data.iter().map(|x| x * c).collect();
        let ys: Vec<f64> = self.ys.data.iter().map(|y| y * c).collect();
        Data2d::new(xs, ys)
    }
}

fn gen_ring<R: Rng>(n: usize, rng: &mut R) -> Data2d {
    let unif = Uniform::new(-1.0, 1.0);
    let norm = Normal::new(0.0, 1.0 / 8.0);

    let rs: Vec<f64> = (0..n).map(|_| rng.sample(unif)).collect();

    let xs = rs
        .iter()
        .map(|&r| (r * PI).cos() + rng.sample(norm))
        .collect();
    let ys = rs
        .iter()
        .map(|&r| (r * PI).sin() + rng.sample(norm))
        .collect();

    Data2d::new(xs, ys)
}

fn gen_square<R: Rng>(n: usize, rng: &mut R) -> Data2d {
    let u = Uniform::new(-1.0, 1.0);
    let xs = (0..n).map(|_| rng.sample(u)).collect();
    let ys = (0..n).map(|_| rng.sample(u)).collect();
    Data2d::new(xs, ys)
}

fn gen_wave<R: Rng>(n: usize, rng: &mut R) -> Data2d {
    let u = Uniform::new(-1.0, 1.0);
    let xs: Vec<f64> = (0..n).map(|_| rng.sample(u)).collect();
    let ys = xs
        .iter()
        .map(|x| 4.0 * (x * x - 0.5).powi(2) + rng.sample(u) / 3.0)
        .collect();
    Data2d::new(xs, ys)
}

fn gen_x<R: Rng>(n: usize, rng: &mut R) -> Data2d {
    let u = Uniform::new(-1.0, 1.0);
    let jitter = Uniform::new(-0.1, 0.1);

    let xs: Vec<f64> = (0..n).map(|_| rng.sample(u)).collect();
    let ys = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| if i % 2 == 0 { x } else { -x } + rng.sample(jitter))
        .collect();

    Data2d::new(xs, ys)
}

fn gen_dots<R: Rng>(n: usize, mut rng: &mut R) -> Data2d {
    fn sample_dots_dim<R: Rng>(n: usize, rng: &mut R) -> Vec<f64> {
        let u = Uniform::new(0.0, 1.0);
        let norm_pos = Normal::new(3.0, 1.0);
        let norm_neg = Normal::new(-3.0, 1.0);
        (0..n)
            .map(|_| {
                if rng.sample(u) < 0.5 {
                    rng.sample(norm_pos)
                } else {
                    rng.sample(norm_neg)
                }
            })
            .collect()
    }

    let xs = sample_dots_dim(n, &mut rng);
    let ys = sample_dots_dim(n, &mut rng);
    Data2d::new(xs, ys)
}

fn xy_codebook() -> Codebook {
    Codebook {
        row_names: None,
        table_name: String::from("xy"),
        col_metadata: btreemap! {
            String::from("x") => ColMetadata {
                id: 0,
                name: String::from("x"),
                spec_type: SpecType::Other,
                coltype: ColType::Continuous { hyper: None },
                notes: None,
            },
            String::from("y") => ColMetadata {
                id: 1,
                name: String::from("y"),
                spec_type: SpecType::Other,
                coltype: ColType::Continuous { hyper: None },
                notes: None,
            },
        },
        view_alpha_prior: Some(Gamma::new(1.0, 1.0).unwrap()),
        state_alpha_prior: Some(Gamma::new(1.0, 1.0).unwrap()),
        comments: None,
    }
}

/// Fits a model to samples from `gen_data` and retuns samples from the fit
/// model.
fn exec_shape_fit<R: Rng>(
    shape: ShapeType,
    scale: f64,
    n: usize,
    nstates: usize,
    mut rng: &mut R,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let xy = shape.sample(n, &mut rng).scale(scale);
    let mut states: BTreeMap<usize, State> = BTreeMap::new();

    let alpha_prior = Gamma::new(1.0, 1.0).unwrap();

    (0..nstates).for_each(|i| {
        let prior_x = Ng::from_data(&xy.xs.data, &mut rng);
        let col_x = Column::new(0, xy.xs.clone(), prior_x);

        let prior_y = Ng::from_data(&xy.ys.data, &mut rng);
        let col_y = Column::new(1, xy.ys.clone(), prior_y);

        let ftrs =
            vec![ColModel::Continuous(col_x), ColModel::Continuous(col_y)];

        states.insert(
            i,
            State::from_prior(
                ftrs,
                alpha_prior.clone(),
                alpha_prior.clone(),
                &mut rng,
            ),
        );
    });

    let mut engine = Engine {
        states,
        codebook: xy_codebook(),
        rng: XorShiftRng::from_rng(&mut rng).unwrap(),
    };
    engine.run(500);

    let oracle = Oracle::from_engine(engine);

    let xy_sim: Vec<Vec<f64>> = oracle
        .simulate(&vec![0, 1], &Given::Nothing, n, None, &mut rng)
        .iter()
        .map(|xys| vec![xys[0].as_f64().unwrap(), xys[1].as_f64().unwrap()])
        .collect();

    (xy.to_vec(), xy_sim)
}

pub fn shape_perm<R: Rng>(
    shape: ShapeType,
    scale: f64,
    n: usize,
    n_perms: usize,
    nstates: usize,
    mut rng: &mut R,
) -> ShapeResultPerm {
    let (xy_src, xy_sim) = exec_shape_fit(shape, scale, n, nstates, &mut rng);
    let pval = gauss_perm_test(&xy_src, &xy_sim, n_perms, &mut rng);
    ShapeResultPerm {
        shape,
        n,
        n_perms,
        p: pval,
        observed: xy_src.to_vec(),
        simulated: xy_sim.to_vec(),
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum ShapeType {
    #[serde(rename = "ring")]
    Ring,
    #[serde(rename = "wave")]
    Wave,
    #[serde(rename = "square")]
    Square,
    #[serde(rename = "x")]
    X,
    #[serde(rename = "dots")]
    Dots,
}

impl ShapeType {
    /// Sample 2D continuous shape data
    pub fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Data2d {
        match &self {
            ShapeType::Ring => gen_ring(n, &mut rng),
            ShapeType::Wave => gen_wave(n, &mut rng),
            ShapeType::X => gen_x(n, &mut rng),
            ShapeType::Square => gen_square(n, &mut rng),
            ShapeType::Dots => gen_dots(n, &mut rng),
        }
    }

    /// Name String
    pub fn name(&self) -> String {
        let name = match &self {
            ShapeType::Ring => "ring",
            ShapeType::Wave => "wave",
            ShapeType::X => "x",
            ShapeType::Square => "square",
            ShapeType::Dots => "dots",
        };
        String::from(name)
    }
}

#[derive(Serialize)]
pub struct ShapeResultPerm {
    shape: ShapeType,
    n: usize,
    n_perms: usize,
    p: f64,
    observed: Vec<Vec<f64>>,
    simulated: Vec<Vec<f64>>,
}

#[derive(Serialize)]
pub struct ShapeResult {
    shape: ShapeType,
    perm_normal: ShapeResultPerm,
    perm_scaled: ShapeResultPerm,
}

fn do_shape_tests<R: Rng>(
    shape: ShapeType,
    n: usize,
    n_perms: usize,
    nstates: usize,
    mut rng: &mut R,
) -> ShapeResult {
    info!(
        "Executing NORMAL permutation test for '{}' ({} samples, {} perms)",
        shape.name(),
        n,
        n_perms
    );

    let perm_result_n = shape_perm(shape, 1.0, n, n_perms, nstates, &mut rng);

    info!(
        "Executing SCALED permutation test for '{}' ({} samples, {} perms)",
        shape.name(),
        n,
        n_perms
    );

    let perm_result_s =
        shape_perm(shape, SHAPE_SCALE, n, n_perms, nstates, &mut rng);

    ShapeResult {
        shape,
        perm_normal: perm_result_n,
        perm_scaled: perm_result_s,
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ShapesRegressionConfig {
    pub shapes: Vec<ShapeType>,
    pub n: usize,
    pub n_perms: usize,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nstates: Option<usize>,
}

pub fn run_shapes<R: Rng>(
    config: &ShapesRegressionConfig,
    mut rng: &mut R,
) -> Vec<ShapeResult> {
    let nstates: usize = config.nstates.unwrap_or(8);
    config
        .shapes
        .iter()
        .map(|shape| {
            do_shape_tests(*shape, config.n, config.n_perms, nstates, &mut rng)
        })
        .collect()
}
