/// Message for monitoring state run progress
#[derive(Clone, Debug)]
pub struct StateProgress {
    pub state_ix: usize,
    pub iter: usize,
    pub score: f64,
    pub quit_now: bool,
}

pub struct StateProgressMonitor {
    pub iters: Vec<Option<usize>>,
    pub scores: Vec<Option<f64>>,
}

impl Default for StateProgressMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl StateProgressMonitor {
    pub fn new() -> Self {
        Self {
            iters: Vec::new(),
            scores: Vec::new(),
        }
    }

    pub fn n_states(&self) -> usize {
        self.iters.len()
    }

    pub fn receive(&mut self, msg: &StateProgress) {
        let ix = msg.state_ix;
        if self.n_states() <= ix {
            self.iters.resize(ix + 1, None);
            self.scores.resize(ix + 1, None);
        }

        self.iters[ix] = Some(msg.iter);
        self.scores[ix] = Some(msg.score);
    }

    pub fn mean_score(&self) -> f64 {
        let mut n_scores: f64 = 0.0;
        self.scores
            .iter()
            .flat_map(|s| {
                n_scores += 1.0;
                s
            })
            .sum::<f64>()
            / n_scores
    }

    pub fn total_iters(&self) -> usize {
        self.iters.iter().flatten().sum::<usize>()
    }
}
