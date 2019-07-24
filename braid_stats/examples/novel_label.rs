// Observe the labeler state over time. In this experiment, the true label is
// always 'false' and informant always labels 'false'. We then ask the labeler
// how the informant would label 'true'.
//
// The code outputs to a csv called 'novel-label.csv'.
//
// If you'd like to plot with python, the following code will do the trick:
//
//     import pandas as pd
//     import matplotlib.pyplot as plt
//
//     df = pd.read_csv("novel-label.csv");
//     df.plot(figsize=(8, 4), lw=1, alpha=0.5)
//     plt.savefig("plot.png", dpi=150)
//
use std::fs::File;
use std::io::prelude::*;
use std::io::BufWriter;

use rand;
use rv::data::DataOrSuffStat;
use rv::traits::*;

use braid_stats::labeler::*;

fn p_label_novel_correctly(labeler: &Labeler) -> f64 {
    let ft = Label::new(false, Some(true));
    let tt = Label::new(true, Some(true));

    let pt = labeler.f(&tt);
    let pf = labeler.f(&ft);

    pt / (pf + pt)
}

fn main() {
    let file = File::create("novel-label.csv").unwrap();
    let mut writer = BufWriter::new(file);

    writer.write("p_k,p_h,p_w,p_correct\n".as_bytes()).unwrap();

    let n = 1_000;

    let mut rng = rand::thread_rng();

    let prior = LabelerPrior::default();
    let mut stat = LabelerSuffStat::new();
    let x = Label::new(false, Some(false));

    (1..=n).for_each(|_| {
        stat.observe(&x);
        let posterior = prior.posterior(&DataOrSuffStat::SuffStat(&stat));
        let labeler = posterior.draw(&mut rng);

        let line = format!(
            "{},{},{},{}\n",
            labeler.p_k(),
            labeler.p_h(),
            labeler.p_world(),
            p_label_novel_correctly(&labeler),
        );

        writer.write(line.as_bytes()).unwrap();
    });
    println!("");
}
