//! Misc, generally useful helper functions
use braid_stats::rv::misc::pflip;
use braid_utils::Shape;
use indicatif::ProgressBar;
use rand::Rng;
use std::iter::Iterator;
use std::ops::Index;

/// Draw n categorical indices in {0,..,k-1} from an n-by-k vector of vectors
/// of un-normalized log probabilities.
///
/// Automatically chooses whether to use serial or parallel computing.
pub fn massflip<M>(logps: M, mut rng: &mut impl Rng) -> Vec<usize>
where
    M: Index<(usize, usize), Output = f64> + Shape + Sync,
{
    braid_flippers::massflip_mat_par(logps, &mut rng)
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub struct CrpDraw {
    pub asgn: Vec<usize>,
    pub counts: Vec<usize>,
    pub n_cats: usize,
}

/// Draw from Chinese Restaraunt Process
pub fn crp_draw<R: Rng>(n: usize, alpha: f64, rng: &mut R) -> CrpDraw {
    let mut n_cats = 0;
    let mut weights: Vec<f64> = vec![];
    let mut asgn: Vec<usize> = Vec::with_capacity(n);

    for _ in 0..n {
        weights.push(alpha);
        let k = pflip(&weights, 1, rng)[0];
        asgn.push(k);

        if k == n_cats {
            weights[n_cats] = 1.0;
            n_cats += 1;
        } else {
            weights.truncate(n_cats);
            weights[k] += 1.0;
        }
    }
    // convert weights to counts, correcting for possible floating point
    // errors
    let counts: Vec<usize> =
        weights.iter().map(|w| (w + 0.5) as usize).collect();

    CrpDraw {
        asgn,
        counts,
        n_cats,
    }
}

/// Simple progress bar for Engine runs run as a tokio task
pub fn progress_bar(
    total_iters: usize,
    mut rcvr: crate::Receiver<crate::StateProgress>,
) -> tokio::task::JoinHandle<crate::Receiver<crate::StateProgress>> {
    use indicatif::ProgressStyle;
    use std::time::{Duration, Instant};

    let style = ProgressStyle::default_bar().template(
        "Score {msg} {wide_bar:.white/white} │{pos}/{len}, Elapsed {elapsed_precise} ETA {eta_precise}│",
    ).unwrap().progress_chars("━╾ ");

    let pbar = ProgressBar::new(total_iters as u64);
    pbar.set_style(style);

    // update the output 4 times a second
    let update_duration = Duration::from_millis(250);
    let mut last_update = Instant::now();

    tokio::spawn(async move {
        let mut monitor = crate::StateProgressMonitor::new();

        while let Some(msg) = rcvr.recv().await {
            monitor.receive(&msg);
            let completed_iters = monitor.total_iters();

            if last_update.elapsed() > update_duration {
                pbar.set_position(completed_iters as u64);
                pbar.set_message(format!("{:.2}", monitor.mean_score()));
                last_update = Instant::now()
            }

            if msg.quit_now || total_iters <= completed_iters {
                pbar.finish();
                break;
            }
        }

        rcvr
    })
}
