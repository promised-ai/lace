use std::path::PathBuf;

use clap::Parser;
use plotly::common::Mode;
use plotly::layout::Layout;
use plotly::{Plot, Scatter};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use lace_cc::alg::{ColAssignAlg, RowAssignAlg};
use lace_cc::feature::FType;
use lace_cc::state::geweke::StateGewekeSettings;
use lace_cc::state::State;
use lace_cc::transition::StateTransition;
use lace_geweke::GewekeTester;
use lace_stats::prior_process::PriorProcessType;

#[derive(Parser, Debug)]
#[clap(rename_all = "kebab")]
struct Opt {
    #[clap(long, default_value = "gibbs")]
    pub row_alg: RowAssignAlg,
    #[clap(long, default_value = "gibbs")]
    pub col_alg: ColAssignAlg,
    #[clap(long, default_value = "50")]
    pub nrows: usize,
    #[clap(long, default_value = "5")]
    pub ncols: usize,
    #[clap(long)]
    pub no_row_reassign: bool,
    #[clap(long)]
    pub no_col_reassign: bool,
    #[clap(long)]
    pub no_view_alpha: bool,
    #[clap(long)]
    pub no_state_alpha: bool,
    #[clap(long)]
    pub no_priors: bool,
    #[clap(long)]
    pub plot_var: Option<String>,
    #[clap(long, short, default_value = "10000")]
    pub niters: usize,
    #[clap(long)]
    dst: Option<PathBuf>,
}

fn main() {
    let opt = Opt::parse();
    let mut rng = Xoshiro256Plus::from_entropy();

    // Some of each column type that is supported by Geweke (Labeler cannot be
    // used in Geweke tests)
    let ftypes: Vec<FType> = (0..opt.ncols)
        .map(|i| match i % 3 {
            0 => FType::Continuous,
            1 => FType::Categorical,
            2 => FType::Count,
            _ => unreachable!(),
        })
        .collect();

    // The state's Geweke test settings require the number of rows in the
    // state (50), and the types of each column. Everything else is filled out
    // automatically.
    let mut settings = StateGewekeSettings::new(
        opt.nrows,
        ftypes,
        PriorProcessType::Dirichlet,
        PriorProcessType::Dirichlet,
    );
    let mut transitions: Vec<StateTransition> = Vec::new();

    if !opt.no_col_reassign {
        transitions.push(StateTransition::ColumnAssignment(opt.col_alg));
    }

    if !opt.no_state_alpha {
        transitions.push(StateTransition::StatePriorProcessParams);
    }

    if !opt.no_row_reassign {
        transitions.push(StateTransition::RowAssignment(opt.row_alg));
    }

    if !opt.no_view_alpha {
        transitions.push(StateTransition::ViewPriorProcessParams);
    }

    if !opt.no_priors {
        transitions.push(StateTransition::FeaturePriors);
    }

    transitions.push(StateTransition::ComponentParams);

    settings.transitions = transitions;

    // Initialize a tester given the settings and run.
    let mut geweke: GewekeTester<State> = GewekeTester::new(settings);
    geweke.run(opt.niters, Some(5), &mut rng);

    // Reports the deviation from a perfect correspondence between the
    // forward and posterior CDFs. The best score is zero, the worst possible
    // score is 0.5.
    let res = geweke.result();
    res.report();

    if let Some(ref key) = opt.plot_var {
        use lace_stats::EmpiricalCdf;
        use lace_utils::minmax;
        let (min_f, max_f) = minmax(res.forward.get(key).unwrap());
        let (min_p, max_p) = minmax(res.posterior.get(key).unwrap());
        let x_min = min_f.min(min_p);
        let x_max = max_f.max(max_p);

        let step = (x_max - x_min) / 200.0;
        let xs: Vec<f64> =
            (0..200).map(|i| x_min + (i as f64) * step).collect();

        let cdf_f = EmpiricalCdf::new(res.forward.get(key).unwrap());
        let cdf_p = EmpiricalCdf::new(res.posterior.get(key).unwrap());

        let x_f: Vec<f64> = cdf_f.f(&xs);
        let x_p: Vec<f64> = cdf_p.f(&xs);

        let trace = Scatter::new(x_f, x_p).name("P-P").mode(Mode::Lines);
        let ideal = Scatter::new(vec![0_f64, 1_f64], vec![0_f64, 1_f64])
            .name("Reference")
            .mode(Mode::Lines);

        let mut plot = Plot::new();
        plot.add_trace(ideal);
        plot.add_trace(trace);
        plot.set_layout(Layout::new());
        plot.show();
    }

    if let Some(dst) = opt.dst {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(dst)
            .unwrap();
        serde_json::to_writer(f, &res).unwrap();
    }
}
