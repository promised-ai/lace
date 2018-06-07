extern crate braid;
extern crate rand;
extern crate serde_yaml;

use std::collections::BTreeMap;
use std::f64::consts::PI;

use self::braid::cc::codebook::{ColMetadata, MetaData, SpecType};
use self::braid::cc::{Codebook, ColModel, Column, DataContainer, State};
use self::braid::dist::prior::NormalInverseGamma;
use self::braid::stats::ks2sample;
use self::braid::stats::perm::gauss_perm_test;
use self::braid::{Engine, Oracle};
use self::rand::distributions::{Normal, Uniform};
use self::rand::Rng;

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

    fn to_vec(&self) -> Vec<Vec<f64>> {
        self.xs
            .data
            .iter()
            .zip(self.ys.data.iter())
            .map(|(x, y)| vec![*x, *y])
            .collect()
    }
}

// TODO: generators should take a scale parameter to make sure that our
// ability to fit data is not affected by their scale.
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
        .map(|x| 4.0 * (x * x - 0.5).powi(2) + rng.sample(u))
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
        let norm_pos = Normal::new(3.0, 1.0);
        let norm_neg = Normal::new(-3.0, 1.0);
        (0..n)
            .map(|i| {
                if i % 2 == 0 {
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
        table_name: String::from("xy"),
        metadata: vec![
            MetaData::Column {
                id: 0,
                name: String::from("x"),
                spec_type: SpecType::Other,
                colmd: ColMetadata::Continuous { hyper: None },
            },
            MetaData::Column {
                id: 1,
                name: String::from("y"),
                spec_type: SpecType::Other,
                colmd: ColMetadata::Continuous { hyper: None },
            },
            MetaData::StateAlpha { alpha: 1.0 },
            MetaData::ViewAlpha { alpha: 1.0 },
        ],
        row_names: None,
        comments: None,
    }
}

/// Fits a model to samples from `gen_data` and retuns samples from the fit
/// model.
fn exec_shape_fit<R: Rng>(
    shape: ShapeType,
    n: usize,
    mut rng: &mut R,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let xy = shape.sample(n, &mut rng);
    let mut states: BTreeMap<usize, State> = BTreeMap::new();

    (0..8).for_each(|i| {
        let prior_x = NormalInverseGamma::from_data(&xy.xs.data, &mut rng);
        let col_x = Column::new(0, xy.xs.clone(), prior_x);

        let prior_y = NormalInverseGamma::from_data(&xy.ys.data, &mut rng);
        let col_y = Column::new(1, xy.ys.clone(), prior_y);

        let ftrs =
            vec![ColModel::Continuous(col_x), ColModel::Continuous(col_y)];

        states.insert(i, State::from_prior(ftrs, 1.0, &mut rng));
    });

    let mut engine = Engine {
        states: states,
        codebook: xy_codebook(),
    };
    engine.run(500, 1);

    let oracle = Oracle::from_engine(engine);

    let xy_sim: Vec<Vec<f64>> = oracle
        .simulate(&vec![0, 1], &None, n, &mut rng)
        .iter()
        .map(|xys| vec![xys[0].as_f64().unwrap(), xys[1].as_f64().unwrap()])
        .collect();

    (xy.to_vec(), xy_sim)
}

fn unzip_2d_vec(mut xys: Vec<Vec<f64>>) -> (Vec<f64>, Vec<f64>) {
    let n = xys.len();
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    xys.drain(..).for_each(|xy| {
        xs.push(xy[0]);
        ys.push(xy[1]);
    });
    (xs, ys)
}

pub fn shape_ks<R: Rng>(
    shape: ShapeType,
    n: usize,
    mut rng: &mut R,
) -> (f64, f64) {
    let (xy_src, xy_sim) = exec_shape_fit(shape, n, &mut rng);
    let (x_src, y_src) = unzip_2d_vec(xy_src);
    let (x_sim, y_sim) = unzip_2d_vec(xy_sim);

    let ks_x = ks2sample(x_src, x_sim);
    let ks_y = ks2sample(y_src, y_sim);

    (ks_x, ks_y)
}

pub fn shape_perm<R: Rng>(
    shape: ShapeType,
    n: usize,
    n_perms: usize,
    mut rng: &mut R,
) -> f64 {
    let (xy_src, xy_sim) = exec_shape_fit(shape, n, &mut rng);
    gauss_perm_test(&xy_src, &xy_sim, n_perms, &mut rng)
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
    pub fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Data2d {
        match &self {
            ShapeType::Ring => gen_ring(n, &mut rng),
            ShapeType::Wave => gen_wave(n, &mut rng),
            ShapeType::X => gen_x(n, &mut rng),
            ShapeType::Square => gen_square(n, &mut rng),
            ShapeType::Dots => gen_dots(n, &mut rng),
        }
    }
}

#[derive(Serialize)]
pub struct ShapeResultKs {
    shape: ShapeType,
    n: usize,
    ks_x: f64,
    ks_y: f64,
}

#[derive(Serialize)]
pub struct ShapeResultPerm {
    shape: ShapeType,
    n: usize,
    n_perms: usize,
    p: f64,
}

#[derive(Serialize)]
pub struct ShapeResult {
    shape: ShapeType,
    ks: ShapeResultKs,
    perm: ShapeResultPerm,
}

fn do_shape_tests<R: Rng>(
    shape: ShapeType,
    n_ks: usize,
    n_perm: usize,
    n_perms: usize,
    mut rng: &mut R,
) -> ShapeResult {
    let perm_pval = shape_perm(shape, n_perm, n_perms, &mut rng);
    let (ks_x, ks_y) = shape_ks(shape, n_ks, &mut rng);

    let perm_result = ShapeResultPerm {
        shape: shape,
        n: n_perm,
        n_perms: n_perms,
        p: perm_pval,
    };

    let ks_result = ShapeResultKs {
        shape: shape,
        n: n_ks,
        ks_x: ks_x,
        ks_y: ks_y,
    };

    ShapeResult {
        shape: shape,
        ks: ks_result,
        perm: perm_result,
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ShapesRegressionConfig {
    pub shapes: Vec<ShapeType>,
    pub n_ks: usize,
    pub n_perm: usize,
    pub n_perms: usize,
}

pub fn run_shapes<R: Rng>(
    config: &ShapesRegressionConfig,
    mut rng: &mut R,
) -> Vec<ShapeResult> {
    config
        .shapes
        .iter()
        .map(|shape| {
            do_shape_tests(
                *shape,
                config.n_ks,
                config.n_perm,
                config.n_perms,
                &mut rng,
            )
        })
        .collect()
}
