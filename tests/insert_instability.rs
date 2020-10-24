// This test reproduces behavior whereby inserting data into an engine without
// running an Engine update leaves the metadata in an invalid state.
use braid::cc::{ColAssignAlg, RowAssignAlg, StateTransition};
use braid::{EngineUpdateConfig, Given, Row};
use braid_codebook::{ColMetadata, ColMetadataList};

use std::convert::TryInto;

fn empty_engine() -> braid::Engine {
    use braid::data::DataSource;
    use braid_codebook::Codebook;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    braid::Engine::new(
        16,
        Codebook::default(),
        DataSource::Empty,
        0,
        Xoshiro256Plus::from_entropy(),
    )
    .unwrap()
}

fn default_transitions() -> Vec<StateTransition> {
    let mut transtions = vec![
        StateTransition::RowAssignment(RowAssignAlg::FiniteCpu),
        StateTransition::ComponentParams,
        StateTransition::FeaturePriors,
        StateTransition::ViewAlphas,
        StateTransition::ColumnAssignment(ColAssignAlg::FiniteCpu),
        StateTransition::StateAlpha,
    ];

    let finite_vec = vec![
        StateTransition::RowAssignment(RowAssignAlg::FiniteCpu),
        StateTransition::ComponentParams,
        StateTransition::FeaturePriors,
        StateTransition::ViewAlphas,
        StateTransition::ColumnAssignment(ColAssignAlg::FiniteCpu),
        StateTransition::StateAlpha,
    ];

    let n_finite_iters: usize = 4;

    transtions.reserve(6 * n_finite_iters);
    (0..n_finite_iters).for_each(|_| transtions.extend_from_slice(&finite_vec));
    transtions
}

fn update_config(n_iters: usize) -> EngineUpdateConfig {
    let transitions = default_transitions();

    EngineUpdateConfig {
        n_iters,
        transitions,
        timeout: None,
        save_path: None,
    }
}

fn gen_row<R: rand::Rng>(ix: u32, mut rng: &mut R) -> Row {
    use braid_stats::Datum;
    use rv::dist::Gaussian;
    use rv::traits::Rv;

    let g = Gaussian::default();
    let mut values = g
        .sample_stream(&mut rng)
        .enumerate()
        .take(14)
        .map(|(ix, x)| (ix.to_string(), Datum::Continuous(x)))
        .collect::<Vec<(String, Datum)>>();

    let label = (ix % 3) as u8;

    values.push((String::from("label"), Datum::Categorical(label)));

    (ix.to_string(), values).try_into().unwrap()
}

fn gen_col_metadata(col_name: &str) -> ColMetadata {
    use braid_codebook::ColType;
    use braid_stats::prior::{CsdHyper, NigHyper};

    if col_name != "label" {
        // temporal variables
        ColMetadata {
            name: String::from(col_name),
            coltype: ColType::Continuous {
                hyper: Some(NigHyper::default()),
            },
            notes: None,
        }
    } else {
        // label/action
        ColMetadata {
            name: String::from("label"),
            coltype: ColType::Categorical {
                k: 5,
                hyper: Some(CsdHyper::new(2.0, 3.0)),
                value_map: None,
            },
            notes: None,
        }
    }
}

fn gen_new_metadata(row: &Row) -> Option<ColMetadataList> {
    let colmds: Vec<ColMetadata> = row
        .values
        .iter()
        .map(|value| gen_col_metadata(value.col_name.as_str()))
        .collect();
    Some(colmds.try_into().unwrap())
}

#[test]
fn otacon_on_empty_table() {
    use braid::{HasData, OracleT, WriteMode};

    let mut rng = rand::thread_rng();
    let mut engine = empty_engine();

    let n_iters = 100;

    {
        let row = gen_row(0, &mut rng);
        let new_md = gen_new_metadata(&row);
        engine
            .insert_data(vec![row], new_md, None, WriteMode::unrestricted())
            .unwrap();
        engine.run(1);
    }

    let mut sum = 0.0;
    for i in 1..n_iters {
        let row = gen_row(i, &mut rng);
        engine
            .insert_data(vec![row], None, None, WriteMode::unrestricted())
            .unwrap();
        for ix in 0..15 {
            let vals = vec![vec![engine.cell(i as usize, ix)]];
            let logps = engine
                .logp_scaled(&vec![ix], &vals, &Given::Nothing, None)
                .unwrap();
            sum += logps[0];
        }
    }

    println!("{}", sum);
}
