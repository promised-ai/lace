use clap::Parser;
use lace::prelude::*;
use lace::stats::rv::dist::{Gaussian, NormalInvChiSquared};
use lace::stats::rv::traits::Rv;
use polars::prelude::*;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[clap(rename_all = "kebab")]
struct Opt {
    #[clap(long, default_value = "16")]
    pub n_states: usize,
    #[clap(short, long, default_value = "1000")]
    pub n_rows: usize,
    #[clap(long, default_value = "100")]
    pub n_sweeps: usize,
    #[clap(long, default_value = "0.05")]
    pub noise: f64,
    #[clap(long, default_value = "10.0")]
    pub scale: f64,
    pub dst: PathBuf,
}

fn state_transitions() -> Vec<StateTransition> {
    let mut conditions = HashSet::new();
    conditions.insert(0);
    conditions.insert(1);

    vec![
        StateTransition::StateAlpha,
        StateTransition::RowAssignment(RowAssignAlg::ConditionalSlice(
            conditions,
        )),
        StateTransition::ComponentParams,
        StateTransition::ViewAlphas,
        StateTransition::FeaturePriors,
    ]
}

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
    df! {
        "index" =>(0..n as u64).collect::<Vec<u64>>(),
        "x" => xs,
        "y" => ys,
    }
    .unwrap()
}

fn main() {
    let Opt {
        n_states,
        n_rows,
        n_sweeps,
        noise,
        scale,
        dst,
    } = Opt::parse();
    let mut rng = rand_xoshiro::Xoshiro256Plus::from_entropy();
    let line = gen_line(scale, noise, n_rows, &mut rng);

    println!("{line}");
    let line_copy = line.clone();

    let codebook = {
        let mut codebook = Codebook::from_df(&line, None, None, false).unwrap();
        if let ColType::Continuous {
            ref mut hyper,
            ref mut prior,
        } = codebook.col_metadata[0].coltype
        {
            *hyper = None;
            *prior =
                Some(NormalInvChiSquared::new(0.0, 0.001, 1.0, 0.01).unwrap());
        } else {
            panic!("Supposed to be continuous");
        };
        if let ColType::Continuous {
            ref mut hyper,
            ref mut prior,
        } = codebook.col_metadata[1].coltype
        {
            *hyper = None;
            *prior =
                Some(NormalInvChiSquared::new(0.0, 0.001, 1.0, 0.01).unwrap());
        } else {
            panic!("Supposed to be continuous");
        };
        codebook
    };
    let mut engine =
        Engine::new(n_states, codebook, DataSource::Polars(line), 0, rng)
            .unwrap();
    engine
        .append_latent_column("z", LatentColumnType::Continuous)
        .unwrap();
    engine.flatten_cols();

    let mut rng = rand_xoshiro::Xoshiro256Plus::from_entropy();

    let update_config = EngineUpdateConfig {
        n_iters: 100,
        transitions: state_transitions(),
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
    let given_nothing: Given<usize> = Given::Nothing;
    (0..n_rows).for_each(|_| {
        let datum = engine
            .simulate(&[2], &given_nothing, 1, None, &mut rng)
            .unwrap()[0][0]
            .clone();
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

    let mut synth = df! {
        "xb" => xs,
        "yb" => ys,
        "z" => zs,
    }
    .unwrap();

    synth.with_column(line_copy["x"].clone()).unwrap();
    synth.with_column(line_copy["y"].clone()).unwrap();

    println!("{synth}");

    let mut file = std::fs::File::create(dst).unwrap();
    CsvWriter::new(&mut file).finish(&mut synth).unwrap();
}
