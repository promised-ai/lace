// Observe the labeler state over time as an informant labels perfectly
// correctly. States are saved to a csv.
//
// Use:
//
//     cargo run --example novel_label <N_LABELS> <OUTPUT.CSV>
//
// Example: A world where there are two labels, but only one is observed.
//
//     cargo run --example novel_label 1 novel-label.csv
//
// Example: A world where there are four labels, but only three are observed.
//
//     cargo run --example novel_label 3 novel-label.csv
//
// If you'd like to plot with python, the following code will do the trick:
//
//     import pandas as pd
//     import matplotlib.pyplot as plt
//     import seaborn as sns
//
//     df = pd.read_csv("novel-label.csv");
//     sns.lineplot(x="t", y="p", hue="measure", data=df)
//
//     plt.show()
//
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufWriter;

use rand;
use rv::data::DataOrSuffStat;
use rv::traits::*;

use braid_stats::labeler::*;

fn p_label_novel_correctly(labeler: &Labeler, n_labels: u8) -> f64 {
    let ps: Vec<f64> = (0..n_labels + 1)
        .map(|x| {
            let label = Label::new(x, Some(n_labels));
            labeler.f(&label)
        })
        .collect();

    ps.last().unwrap() / ps.iter().sum::<f64>()
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("Please provide N_LABELS and OUTPUT.CSV args.");
        std::process::exit(1)
    }

    let n_labels: u8 = args[1].parse().unwrap();
    let fileout = &args[2];

    let file = File::create(fileout).unwrap();
    let mut writer = BufWriter::new(file);

    writer.write("rep,measure,t,p\n".as_bytes()).unwrap();

    let n = 200;
    let n_reps = 20;

    let mut rng = rand::thread_rng();

    for rep in 0..n_reps {
        let prior = LabelerPrior::uniform(n_labels + 1);

        let mut stat = LabelerSuffStat::new();

        (1..=n).for_each(|t| {
            let label = t % n_labels;

            let x = Label::new(label, Some(label));
            stat.observe(&x);

            let posterior = prior.posterior(&DataOrSuffStat::SuffStat(&stat));
            let labeler = posterior.draw(&mut rng);

            {
                let line =
                    format!("{},{},{},{}\n", rep, "p(k)", t, labeler.p_k());
                writer.write(line.as_bytes()).unwrap();
            }
            {
                let line =
                    format!("{},{},{},{}\n", rep, "p(h)", t, labeler.p_h());
                writer.write(line.as_bytes()).unwrap();
            }
            {
                let line = format!(
                    "{},{},{},{}\n",
                    rep,
                    "p(w=2)",
                    t,
                    labeler.p_world().point()[n_labels as usize],
                );
                writer.write(line.as_bytes()).unwrap();
            }
            {
                let line = format!(
                    "{},{},{},{}\n",
                    rep,
                    "p(l=2|w=2)",
                    t,
                    p_label_novel_correctly(&labeler, n_labels),
                );

                writer.write(line.as_bytes()).unwrap();
            }
        });
    }
}
