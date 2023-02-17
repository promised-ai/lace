use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
        Arc, Mutex,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use lace_cc::state::State;

use crate::EngineUpdateConfig;

/// Custom state inspector for `Engine::update`.
pub trait UpdateHandler: Clone + Send + Sync {
    /// Initialize the handler
    fn init(&mut self, config: &EngineUpdateConfig, states: &[State]);

    /// Handler for each state update.
    fn state_updated(&mut self, state_id: usize, state: &State);

    /// Should the `Engine` stop running.
    fn stop_running(&self) -> bool;

    /// Cleanup upon the end of updating.
    fn finish(&mut self);
}

macro_rules! impl_tuple {
($($idx:tt $t:tt),+) => {
    impl<$($t,)+> UpdateHandler for ($($t,)+)
    where
        $($t: UpdateHandler,)+
    {

        fn init(&mut self, config: &EngineUpdateConfig, states: &[State]) {
            $(
                self.$idx.init(config, states);
            )+
        }

        fn state_updated(&mut self, state_id: usize, state: &State) {
            $(
                self.$idx.state_updated(state_id, state);
            )+
        }

        fn stop_running(&self) -> bool {
            $(
                self.$idx.stop_running()
            )||+
        }

        fn finish(&mut self) {
            $(
                self.$idx.finish();
            )+
        }

    }
};
}

impl_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I);
impl_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H);
impl_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G);
impl_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F);
impl_tuple!(0 A, 1 B, 2 C, 3 D, 4 E);
impl_tuple!(0 A, 1 B, 2 C, 3 D);
impl_tuple!(0 A, 1 B, 2 C);
impl_tuple!(0 A, 1 B);
impl_tuple!(0 A);

impl<T> UpdateHandler for Vec<T>
where
    T: UpdateHandler,
{
    fn init(&mut self, config: &EngineUpdateConfig, states: &[State]) {
        self.iter_mut()
            .for_each(|handler| handler.init(config, states));
    }

    fn state_updated(&mut self, state_id: usize, state: &State) {
        self.iter_mut().for_each(|handler| {
            handler.state_updated(state_id, state);
        })
    }

    fn stop_running(&self) -> bool {
        self.iter().any(|handler| handler.stop_running())
    }

    fn finish(&mut self) {}
}

/// Handle Ctrl-C (sigint) signals by stopping the Engine.
#[derive(Clone)]
pub struct CtrlC {
    seen_sigint: Arc<AtomicBool>,
}

impl CtrlC {
    /// Create a new `CtrlCHandler`
    pub fn new() -> Self {
        let seen_sigint = Arc::new(AtomicBool::new(false));
        let r = seen_sigint.clone();

        ctrlc::set_handler(move || {
            r.store(true, Ordering::Relaxed);
        })
        .expect("Error setting Ctrl-C handler");

        Self { seen_sigint }
    }
}

impl UpdateHandler for CtrlC {
    fn init(&mut self, _config: &EngineUpdateConfig, _states: &[State]) {}

    fn state_updated(&mut self, _state_id: usize, _state: &State) {}

    fn stop_running(&self) -> bool {
        self.seen_sigint.load(Ordering::Relaxed)
    }

    fn finish(&mut self) {}
}

#[derive(Clone)]
/// An update handler which stops updates after a timeout limit.
pub enum Timeout {
    UnInitialized { timeout: Duration },
    Initialized { start: Instant, timeout: Duration },
}

impl Timeout {
    /// Create a new `TimeoutHandler` with `timeout` duration.
    pub fn new(timeout: Duration) -> Self {
        Self::UnInitialized { timeout }
    }
}

impl UpdateHandler for Timeout {
    fn init(&mut self, _config: &EngineUpdateConfig, _states: &[State]) {
        if let Self::UnInitialized { timeout } = self {
            *self = Self::Initialized {
                start: Instant::now(),
                timeout: *timeout,
            };
        };
    }

    fn state_updated(&mut self, _state_id: usize, _state: &State) {}

    fn stop_running(&self) -> bool {
        if let Self::Initialized { start, timeout } = self {
            start.elapsed() > *timeout
        } else {
            unreachable!()
        }
    }

    fn finish(&mut self) {}
}

/// Handler with no actions
#[derive(Clone)]
pub struct NoOp;

impl UpdateHandler for NoOp {
    fn init(&mut self, _config: &EngineUpdateConfig, _states: &[State]) {}

    fn state_updated(&mut self, _state_id: usize, _state: &State) {}

    fn stop_running(&self) -> bool {
        false
    }

    fn finish(&mut self) {}
}

/// Add a progress bar to the output
#[derive(Clone)]
pub enum ProgressBar {
    UnInitialized,
    Initialized {
        sender: Arc<Mutex<Sender<(usize, f64)>>>,
        handle: Arc<JoinHandle<()>>,
    },
}

impl ProgressBar {
    pub fn new() -> Self {
        Self::UnInitialized
    }
}

impl Default for ProgressBar {
    fn default() -> Self {
        Self::UnInitialized
    }
}

impl UpdateHandler for ProgressBar {
    fn init(&mut self, config: &EngineUpdateConfig, states: &[State]) {
        const UPDATE_INTERVAL: Duration = Duration::from_millis(250);

        let (sender, receiver) = std::sync::mpsc::channel();
        let total_iters = states.len() * config.n_iters;

        let handle = Arc::new(std::thread::spawn(move || {
            use indicatif::ProgressStyle;

            let style = ProgressStyle::default_bar().template(
    "Score {msg} {wide_bar:.white/white} │{pos}/{len}, Elapsed {elapsed_precise} ETA {eta_precise}│",
).unwrap().progress_chars("━╾ ");

            let progress_bar = indicatif::ProgressBar::new(total_iters as u64);
            progress_bar.set_style(style);
            let mut last_update = Instant::now();
            let mut completed_iters: usize = 0;
            let mut state_log_scores = HashMap::new();

            while let Ok((state_id, log_score)) = receiver.recv() {
                completed_iters += 1;
                state_log_scores.insert(state_id, log_score);

                if last_update.elapsed() > UPDATE_INTERVAL {
                    last_update = Instant::now();
                    progress_bar.set_length(completed_iters as u64);
                }
                let mean_log_score = state_log_scores.values().sum::<f64>()
                    / (state_log_scores.len() as f64);

                progress_bar.set_position(completed_iters as u64);
                progress_bar.set_message(format!("{:.2}", mean_log_score));
            }

            progress_bar.finish_and_clear();
        }));

        *self = Self::Initialized {
            sender: Arc::new(Mutex::new(sender)),
            handle,
        }
    }

    fn state_updated(&mut self, state_id: usize, state: &State) {
        if let Self::Initialized { sender, .. } = self {
            sender
                .lock()
                .unwrap()
                .send((state_id, state.log_prior + state.loglike))
                .unwrap();
        }
    }

    fn stop_running(&self) -> bool {
        false
    }

    fn finish(&mut self) {
        if let Self::Initialized { sender, handle } = std::mem::take(self) {
            std::mem::drop(sender);
            Arc::try_unwrap(handle).unwrap().join().unwrap();
        }
    }
}
