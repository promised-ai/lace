//! Misc, generally useful helper functions
use crate::UpdateInformation;
use braid_utils::Shape;
use indicatif::{MultiProgress, ProgressBar};
use rand::Rng;
use rv::misc::pflip;
use std::iter::Iterator;
use std::ops::Index;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

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

pub async fn run_pbar(
    n_iters: usize,
    update_info: Arc<UpdateInformation>,
) -> (MultiProgress, Vec<tokio::task::JoinHandle<()>>) {
    use indicatif::ProgressStyle;
    use std::sync::atomic::Ordering;

    let update_duration = std::time::Duration::from_millis(250);

    let relaxed = Ordering::Relaxed;

    let total_iters = update_info.scores.len() * n_iters;

    let style = ProgressStyle::default_bar().template(
        "Score {msg} {wide_bar:.white/white} │{pos}/{len}, Elapsed {elapsed_precise} ETA {eta_precise}│",
    ).progress_chars("━╾ ");

    let m = MultiProgress::new();

    let pbar = m.add(ProgressBar::new(total_iters as u64));
    pbar.set_draw_rate(1);

    let mut pbars = Vec::new();

    for i in 0..update_info.scores.len() {
        let pb = m
            .add(ProgressBar::new(n_iters as u64))
            .with_style(style.clone());
        pb.set_draw_rate(1);
        let update_info_arc = Arc::clone(&update_info);
        let pb_proc = tokio::spawn(async move {
            loop {
                let is_done = update_info_arc.is_done.load(relaxed);
                let score = update_info_arc.scores[i].read().await;
                let iters = update_info_arc.iters[i].load(relaxed);
                if iters as usize == n_iters || is_done {
                    break;
                }
                if score.is_finite() {
                    pb.set_position(iters);
                    pb.set_message(format!("({}) {:.2}", i, score));
                }
                tokio::time::sleep(update_duration.clone()).await;
            }

            pb.finish_and_clear();
        });
        pbars.push(pb_proc);
    }

    let main = tokio::spawn(async move {
        let style = ProgressStyle::default_bar().template(
            "Score: {msg} {wide_bar:.red/white} │{pos}/{len}, Elapsed {elapsed_precise} ETA {eta_precise}│",
        ).progress_chars("━╾ ");
        pbar.set_style(style);
        pbar.set_position(0);
        while !update_info.is_done.load(relaxed) {
            // compute mean score
            let mut n_states: f64 = 0.0;
            let mut sum: f64 = 0.0;

            for score_lock in update_info.scores.iter() {
                let score = score_lock.read().await;
                if score.is_finite() {
                    n_states += 1.0;
                    sum += *score;
                }
            }
            // let (n_states, sum) = update_info.scores.iter().fold(
            //     (0.0, 0.0),
            //     |(n_states, sum), score| {
            //         score.read().await.map_or((n_states, sum), |s| {
            //         })
            //     },
            // );

            let mean_score = sum / n_states as f64;

            // compute number of iterations complete
            let pos = update_info
                .iters
                .iter()
                .map(|it| it.load(relaxed))
                .sum::<u64>();

            pbar.set_position(pos);
            pbar.set_message(format!("{mean_score:.2}"));

            tokio::time::sleep(update_duration).await;
        }

        if !update_info.quit_now.load(relaxed) {
            pbar.finish_at_current_pos();
        }
    });
    pbars.push(main);
    (m, pbars)
}

pub fn single_bar(
    n_iters: usize,
    update_info: Arc<UpdateInformation>,
) -> tokio::task::JoinHandle<()> {
    use indicatif::ProgressStyle;
    use std::sync::atomic::Ordering;

    let update_duration = std::time::Duration::from_millis(250);

    let relaxed = Ordering::Relaxed;

    let total_iters = update_info.scores.len() * n_iters;

    let style = ProgressStyle::default_bar().template(
        "Score {msg} {wide_bar:.white/white} │{pos}/{len}, Elapsed {elapsed_precise} ETA {eta_precise}│",
    ).progress_chars("━╾ ");

    let pbar = ProgressBar::new(total_iters as u64);
    pbar.set_style(style);

    tokio::spawn(async move {
        while !update_info.is_done.load(relaxed) {
            // compute mean score
            let mut n_states: f64 = 0.0;
            let mut sum: f64 = 0.0;

            for score_lock in update_info.scores.iter() {
                let score = score_lock.read().await;
                if score.is_finite() {
                    n_states += 1.0;
                    sum += *score;
                }
            }

            let mean_score = sum / n_states as f64;

            // compute number of iterations complete
            let pos = update_info
                .iters
                .iter()
                .map(|it| it.load(relaxed))
                .sum::<u64>();

            pbar.set_position(pos);
            pbar.set_message(format!("{mean_score:.2}"));

            if update_info.quit_now.load(relaxed) {
                break;
            }
            tokio::time::sleep(update_duration).await;
        }

        if !update_info.quit_now.load(relaxed) {
            pbar.finish_at_current_pos();
        }
    })
}
