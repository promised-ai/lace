// Fit to a ring and plot the results
//
// To run:
//   cargo run --release --example -- <OPTIONS>
//
// Example:
//   cargo run --release --example -- -n 1000 --nstates 1
//
use rand::{Rng, SeedableRng};
use rand_distr::Uniform;
use std::io;
use structopt::StructOpt;
use tempfile::NamedTempFile;

use braid::{data::DataSource, Engine, Given, OracleT};

#[derive(Debug, StructOpt)]
struct Opt {
    /// The number of data to generate
    #[structopt(short, default_value = "1000")]
    n: usize,
    /// The number of states in the Engine
    #[structopt(long, default_value = "8")]
    nstates: usize,
    /// Scales the ring up or down from its default range, which is (-1, 1)
    #[structopt(long, default_value = "1.0")]
    scale: f64,
    /// Width of the ring
    #[structopt(long, default_value = "0.2")]
    width: f64,
}

fn gen_ring<R: Rng, W: io::Write>(
    n: usize,
    scale: f64,
    width: f64,
    f: &mut W,
    rng: &mut R,
) -> io::Result<(Vec<f64>, Vec<f64>)> {
    let unif = Uniform::new(-1.0, 1.0);

    let mut n_collected: usize = 0;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);

    write!(f, "ID,x,y\n")?;
    while n_collected < n {
        let x: f64 = rng.sample(unif) * scale;
        let y: f64 = rng.sample(unif) * scale;
        let r = (x * x + y * y).sqrt();
        if (1.0 - width) * scale <= r && r <= 1.0 * scale {
            write!(f, "{},{},{}\n", n_collected, x, y)?;
            xs.push(x);
            ys.push(y);
            n_collected += 1;
        }
    }
    Ok((xs, ys))
}

fn plot(xs_in: Vec<f64>, ys_in: Vec<f64>, xs_sim: Vec<f64>, ys_sim: Vec<f64>) {
    use plotly::common::Mode;
    use plotly::layout::{GridPattern, Layout, LayoutGrid};
    use plotly::{Plot, Scatter};

    let trace1 = Scatter::new(xs_in, ys_in).name("Input").mode(Mode::Markers);

    let trace2 = Scatter::new(xs_sim, ys_sim)
        .mode(Mode::Markers)
        .name("Simulated")
        .x_axis("x2")
        .y_axis("y2");

    let mut plot = Plot::new();
    plot.add_trace(trace1);
    plot.add_trace(trace2);

    let layout = Layout::new().grid(
        LayoutGrid::new()
            .rows(1)
            .columns(2)
            .pattern(GridPattern::Independent),
    );

    plot.set_layout(layout);
    plot.show();
}

fn main() {
    let opt = Opt::from_args();

    // generate csv data
    println!("Generating data");
    let mut rng = rand::thread_rng();
    let mut f = NamedTempFile::new().unwrap();
    let (xs_in, ys_in) =
        gen_ring(opt.n, opt.scale, opt.width, &mut f, &mut rng).unwrap();
    println!("Data written to {:?}", f.path());

    // generate codebook
    println!("Generating codebook");
    let codebook = braid_codebook::csv::codebook_from_csv(
        csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(f.path())
            .unwrap(),
        None,
        None,
    )
    .unwrap();

    // generate engine
    println!("Constructing Engine");
    let mut engine = Engine::new(
        opt.nstates,
        codebook,
        DataSource::Csv(f.path().into()),
        0,
        rand_xoshiro::Xoshiro256Plus::from_entropy(),
    )
    .unwrap();

    // run engine with default config. At the time of April 20 2021, the default
    // reassignment kernels for both rows and column is Gibbs.
    println!("Running Engine");
    engine.run(1_000);

    println!("Simulating data");
    let mut xs_sim = Vec::with_capacity(opt.n);
    let mut ys_sim = Vec::with_capacity(opt.n);
    engine
        .simulate(&[0, 1], &Given::Nothing, opt.n, None, &mut rng)
        .unwrap()
        .drain(..)
        .for_each(|xy| {
            xs_sim.push(xy[0].to_f64_opt().unwrap());
            ys_sim.push(xy[1].to_f64_opt().unwrap());
        });

    engine
        .states
        .iter()
        .for_each(|state| print!("{} ", state.views[0].asgn.n_cats));

    println!("\nPlotting");
    plot(xs_in, ys_in, xs_sim, ys_sim);
    println!("Done");
}
