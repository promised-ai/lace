use std::f64::consts::PI;

use braid::{Engine, Given, Oracle, OracleT};
use braid_cc::feature::{ColModel, Column};
use braid_cc::state::State;
use braid_codebook::{
    Codebook, ColMetadata, ColMetadataList, ColType, RowNameList,
};
use braid_data::{Container, SparseContainer};
use braid_stats::prior::crp::CrpPrior;
use braid_stats::test::gauss_perm_test;
use log::info;
use rand::{Rng, SeedableRng};
use rand_distr::{Normal, Uniform};
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use rv::dist::Gamma;
use serde::{Deserialize, Serialize};

const SHAPE_SCALE: f64 = 1_000.0;

pub struct Data2d {
    xs: SparseContainer<f64>,
    ys: SparseContainer<f64>,
}

impl Data2d {
    fn new(xs: Vec<f64>, ys: Vec<f64>) -> Self {
        assert_eq!(xs.len(), ys.len(), "xs and ys must be same length");

        Data2d {
            xs: SparseContainer::from(xs),
            ys: SparseContainer::from(ys),
        }
    }

    /// Convert to a vector of 2-length `Vec<f64>`s.
    fn to_vec(&self) -> Vec<(f64, f64)> {
        // correctness relies on add data being present in xs and ys
        self.xs
            .present_iter()
            .zip(self.ys.present_iter())
            .map(|(x, y)| (*x, *y))
            .collect()
    }

    /// Scale the data by a constant factor, `c`.
    fn scale(&self, c: f64) -> Self {
        let xs: Vec<f64> = self.xs.present_iter().map(|x| x * c).collect();
        let ys: Vec<f64> = self.ys.present_iter().map(|y| y * c).collect();
        Data2d::new(xs, ys)
    }
}

fn gen_ring<R: Rng>(n: usize, rng: &mut R) -> Data2d {
    let unif = Uniform::new(-1.0, 1.0);
    let norm = Normal::new(0.0, 1.0 / 8.0).unwrap();

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
        .map(|x| 4_f64.mul_add((x * x - 0.5).powi(2), rng.sample(u) / 3_f64))
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
        let norm_pos = Normal::new(3.0, 1.0).unwrap();
        let norm_neg = Normal::new(-3.0, 1.0).unwrap();
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

fn xy_codebook(n: usize) -> Codebook {
    Codebook {
        row_names: RowNameList::from_range(0..n),
        table_name: String::from("xy"),
        col_metadata: ColMetadataList::new(vec![
            ColMetadata {
                name: String::from("x"),
                coltype: ColType::Continuous {
                    hyper: None,
                    prior: None,
                },
                notes: None,
            },
            ColMetadata {
                name: String::from("y"),
                coltype: ColType::Continuous {
                    hyper: None,
                    prior: None,
                },
                notes: None,
            },
        ])
        .unwrap(),
        view_alpha_prior: Some(braid_consts::view_alpha_prior().into()),
        state_alpha_prior: Some(braid_consts::state_alpha_prior().into()),
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
) -> ShapeSamples {
    use braid_stats::prior::nix::NixHyper;
    let xy = shape.sample(n, &mut rng).scale(scale);

    let alpha_prior: CrpPrior = Gamma::new(1.0, 1.0).unwrap().into();

    let states: Vec<_> = (0..nstates)
        .map(|_| {
            let hyper_x = NixHyper::from_data(&xy.xs.present_cloned());
            let prior_x = hyper_x.draw(&mut rng);
            let col_x = Column::new(0, xy.xs.clone(), prior_x, hyper_x);

            let hyper_y = NixHyper::from_data(&xy.ys.present_cloned());
            let prior_y = hyper_y.draw(&mut rng);
            let col_y = Column::new(1, xy.ys.clone(), prior_y, hyper_y);

            let ftrs =
                vec![ColModel::Continuous(col_x), ColModel::Continuous(col_y)];

            State::from_prior(
                ftrs,
                alpha_prior.clone(),
                alpha_prior.clone(),
                &mut rng,
            )
        })
        .collect();

    let mut engine = Engine {
        states,
        state_ids: (0..nstates).collect(),
        codebook: xy_codebook(n),
        rng: Xoshiro256Plus::from_rng(&mut rng).unwrap(),
    };
    engine.run(500);

    let oracle = Oracle::from_engine(engine);

    let xy_sim: Vec<(f64, f64)> = oracle
        .simulate(&[0, 1], &Given::Nothing, n, None, &mut rng)
        .unwrap()
        .iter()
        .map(|xys| (xys[0].to_f64_opt().unwrap(), xys[1].to_f64_opt().unwrap()))
        .collect();

    ShapeSamples {
        observed: xy.to_vec(),
        simulated: xy_sim,
    }
}

pub fn shape_perm<R: Rng>(
    shape: ShapeType,
    scale: f64,
    n: usize,
    n_perms: usize,
    nstates: usize,
    save_samples: bool,
    mut rng: &mut R,
) -> ShapePermutationTestResult {
    let samples = exec_shape_fit(shape, scale, n, nstates, &mut rng);
    let pval = gauss_perm_test(
        samples.observed.clone(),
        samples.simulated.clone(),
        n_perms as u32,
        &mut rng,
    );
    ShapePermutationTestResult {
        shape,
        n,
        n_perms,
        p: pval,
        samples: if save_samples { Some(samples) } else { None },
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum ShapeType {
    Ring,
    Wave,
    Square,
    X,
    Dots,
}

impl ShapeType {
    /// Sample 2D continuous shape data
    pub fn sample<R: Rng>(self, n: usize, mut rng: &mut R) -> Data2d {
        match &self {
            ShapeType::Ring => gen_ring(n, &mut rng),
            ShapeType::Wave => gen_wave(n, &mut rng),
            ShapeType::X => gen_x(n, &mut rng),
            ShapeType::Square => gen_square(n, &mut rng),
            ShapeType::Dots => gen_dots(n, &mut rng),
        }
    }

    /// Name String
    pub fn name(self) -> String {
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
pub struct ShapeSamples {
    observed: Vec<(f64, f64)>,
    simulated: Vec<(f64, f64)>,
}

#[derive(Serialize)]
pub struct ShapePermutationTestResult {
    shape: ShapeType,
    n: usize,
    n_perms: usize,
    p: f64,
    samples: Option<ShapeSamples>,
}

#[derive(Serialize)]
pub struct ShapeResult {
    shape_type: ShapeType,
    perm_normal: ShapePermutationTestResult,
    perm_scaled: ShapePermutationTestResult,
}

fn do_shape_tests<R: Rng>(
    shape_type: ShapeType,
    n: usize,
    n_perms: usize,
    nstates: usize,
    save_samples: bool,
    mut rng: &mut R,
) -> ShapeResult {
    info!(
        "Executing NORMAL permutation test for '{}' ({} samples, {} perms)",
        shape_type.name(),
        n,
        n_perms
    );

    let perm_result_n = shape_perm(
        shape_type,
        1.0,
        n,
        n_perms,
        nstates,
        save_samples,
        &mut rng,
    );

    info!(
        "Executing SCALED permutation test for '{}' ({} samples, {} perms)",
        shape_type.name(),
        n,
        n_perms
    );

    let perm_result_s = shape_perm(
        shape_type,
        SHAPE_SCALE,
        n,
        n_perms,
        nstates,
        save_samples,
        &mut rng,
    );

    ShapeResult {
        shape_type,
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

pub fn run_shapes<R: Rng + SeedableRng + Send + Sync>(
    config: &ShapesRegressionConfig,
    save_samples: bool,
    mut rng: &mut R,
) -> Vec<ShapeResult> {
    let nstates: usize = config.nstates.unwrap_or(8);
    let mut rngs: Vec<_> = (0..config.shapes.len())
        .map(|_| R::from_rng(&mut rng).unwrap())
        .collect();

    config
        .shapes
        .par_iter()
        .zip(rngs.par_iter_mut())
        .map(|(shape, trng)| {
            do_shape_tests(
                *shape,
                config.n,
                config.n_perms,
                nstates,
                save_samples,
                trng,
            )
        })
        .collect()
}
