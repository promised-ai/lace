use braid::data::DataSource;
use braid::{Builder, Given, OracleT};
use braid_data::Datum;
use braid_stats::rv::prelude::*;
use std::io::Write;

fn main() {
    let mut rng = rand::thread_rng();

    // Draw data from a mixture of Poisson
    let mixture = Mixture::uniform(vec![
        Poisson::new(3.0).unwrap(),
        Poisson::new(10.0).unwrap(),
    ])
    .unwrap();

    let mut engine = {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "id,x").unwrap();
        mixture
            .sample_stream(&mut rng)
            .take(500)
            .enumerate()
            .for_each(|(ix, x): (usize, u32)| {
                writeln!(file, "{},{}", ix, x).unwrap();
            });

        Builder::new(DataSource::Csv(file.path().into()))
            .with_nstates(2)
            .seed_from_u64(1337)
            .build()
            .unwrap()
    };

    engine.run(500).unwrap();

    let vals: Vec<_> = (0_u32..30).map(|x| vec![Datum::Count(x)]).collect();

    let fx: Vec<_> = engine
        .logp(&[0], &vals, &Given::<usize>::Nothing, None)
        .unwrap()
        .iter()
        .map(|ln_f| ln_f.exp())
        .collect();

    println!("x,fx_true,fx_braid");
    for (x, fx) in fx.iter().enumerate() {
        println!("{},{},{}", x, mixture.f(&(x as u32)), fx);
    }
}
