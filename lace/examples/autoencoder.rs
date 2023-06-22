use lace::prelude::*;
use lace::stats::rv::dist::Gaussian;
use lace::stats::rv::traits::Rv;
use polars::prelude::*;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

pub const STATE_TRANSITIONS: [StateTransition; 5] = [
    StateTransition::StateAlpha,
    StateTransition::RowAssignment(RowAssignAlg::Slice),
    StateTransition::ComponentParams,
    StateTransition::ViewAlphas,
    StateTransition::FeaturePriors,
];

fn gen_line<R: Rng>(
    scale: f64,
    noise: f64,
    n: usize,
    rng: &mut R,
) -> DataFrame {
    let g = Gaussian::new_unchecked(0.0, noise);
    let (xs, ys): (Vec<f64>, Vec<f64>) = (0..n)
        .map(|_| {
            let x: f64 = rng.gen::<f64>() * scale - scale / 2.0;
            let y_noise: f64 = g.draw(rng);
            let x_noise: f64 = g.draw(rng);
            let y = x + y_noise;
            (x + x_noise, y)
        })
        .unzip();
    let zs: Vec<f64> = Gaussian::standard().sample(n, rng);
    df! {
        "index" =>(0..n as u64).collect::<Vec<u64>>(),
        "x" => xs,
        "y" => ys,
        "z_latent" => zs,
    }
    .unwrap()
}

fn main() {
    let n_sweeps = 100;
    let n_rows = 1_000;
    let mut rng = rand_xoshiro::Xoshiro256Plus::from_entropy();
    let line = gen_line(10.0, 0.05, n_rows, &mut rng);

    println!("{line}");

    let codebook = {
        let mut codebook = Codebook::from_df(&line, None, None, false).unwrap();
        codebook.col_metadata["z_latent"].latent = true;
        codebook
    };
    let mut engine =
        Engine::new(8, codebook, DataSource::Polars(line), 0, rng).unwrap();
    engine.flatten_cols();

    let mut rng = rand_xoshiro::Xoshiro256Plus::from_entropy();

    let update_config = EngineUpdateConfig {
        n_iters: 100,
        transitions: STATE_TRANSITIONS.into(),
        checkpoint: None,
        save_config: None,
    };
    let scale_contraint = Gaussian::standard();

    for i in 0..n_sweeps {
        let proposal = {
            let ln_constraint =
                |row_ix: usize, data: &HashMap<usize, Datum>| {
                    let x = engine.cell(row_ix, 0);
                    let y = engine.cell(row_ix, 1);
                    let conditions = data
                        .iter()
                        .map(|(&col_ix, datum)| (col_ix, datum.clone()))
                        .collect::<Vec<(usize, Datum)>>();

                    let scale = if let Datum::Continuous(z) = conditions[0].1 {
                        scale_contraint.ln_f(&z)
                    } else {
                        panic!("z should be continuous")
                    };
                    engine
                        .logp(
                            &[0, 1],
                            &[vec![x, y]],
                            &Given::Conditions(conditions),
                            None,
                        )
                        .unwrap()[0]
                        + scale
                };
            engine
                .propose_latent_values_with(&[2], &ln_constraint, 50, &mut rng)
                .unwrap()
        };

        engine.set_latent_values(proposal);
        engine.update(update_config.clone(), ()).unwrap();
        if i % 10 == 0 {
            println!("sweep {i} of {n_sweeps}");
        }
    }

    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    let mut zs: Vec<f64> = Vec::new();
    (0..n_rows).for_each(|row_ix| {
        let datum = engine.impute(row_ix, 2, None).unwrap().0;
        let z = datum.to_f64_opt().unwrap();
        let xys = engine
            .simulate(
                &[0, 1],
                &Given::Conditions(vec![(2, datum)]),
                1,
                None,
                &mut rng,
            )
            .unwrap();
        let x = xys[0][0].to_f64_opt().unwrap();
        let y = xys[0][1].to_f64_opt().unwrap();
        xs.push(x);
        ys.push(y);
        zs.push(z);
    });

    let synth = df! {
        "x" => xs,
        "y" => ys,
        "z_latent" => zs,
    }
    .unwrap();
    println!("{synth}")
}
