use std::{
    collections::HashMap,
    sync::{mpsc::Sender, Arc, Mutex, RwLock},
    thread::JoinHandle,
    time::{Duration, Instant},
};

#[cfg(feature = "ctrlc_handler")]
use std::sync::atomic::{AtomicBool, Ordering};

use lace_cc::state::State;

use crate::EngineUpdateConfig;

/// Custom state inspector for `Engine::update`.
///
/// This trait can be used to implement progress capture and early stopping.
///
///
/// # Example
/// The following example will store timing and then prints them to STDOUT.
///
/// ```
/// use std::sync::{Arc, Mutex};
/// use std::time::Instant;
///
/// use lace::update_handler::UpdateHandler;
/// use lace::EngineUpdateConfig;
/// use lace::cc::state::State;
/// use lace::examples::Example;
///
/// #[derive(Debug, Clone)]
/// pub struct TimingsHandler {
///     timings: Arc<Mutex<Vec<Instant>>>,
/// }
///
/// impl TimingsHandler {
///     pub fn new() -> Self {
///         Self { timings: Arc::new(Mutex::new(Vec::new())) }
///     }
/// }
///
/// impl UpdateHandler for TimingsHandler {
///     fn state_updated(&mut self, _state_id: usize, _state: &State) {
///         self.timings.lock().unwrap().push(Instant::now());
///     }
///
///     fn finalize(&mut self) {
///         let timings = self.timings.lock().unwrap();
///         let mean_time_between_updates =
///             timings.iter().zip(timings.iter().skip(1))
///             .map(|(&a, b)| b.duration_since(a).as_secs_f64())
///             .sum::<f64>() / (timings.len() as f64);
///
///         eprintln!("Mean time between updates = {mean_time_between_updates}");
///     }
/// }
/// let mut engine = Example::Animals.engine().unwrap();
///
/// engine.update(
///     &EngineUpdateConfig::with_default_transitions().n_iters(100),
///     TimingsHandler::new()
/// ).unwrap();
/// ```
pub trait UpdateHandler: Clone + Send + Sync {
    /// Initialize the handler, for all states (globally).
    ///
    /// This method is called after the states have been loaded but before any updating has
    /// occured.
    fn global_init(&mut self, _config: &EngineUpdateConfig, _states: &[State]) {
    }

    /// Initialize for a new state.
    ///
    /// This method is called after a specific state is loaded but before it's individual updates
    /// have occured. Other states may be initialized before this.
    fn new_state_init(&mut self, _state_id: usize, _state: &State) {}

    /// Handler for each state update.
    ///
    /// This method is called after each state update is complete.
    fn state_updated(&mut self, _state_id: usize, _state: &State) {}

    /// Handle complete state updates.
    ///
    /// This method is called after a state has completed all of its updates.
    fn state_complete(&mut self, _state_id: usize, _state: &State) {}

    /// Should the `Engine` stop running.
    ///
    /// The method is called after each state update.
    /// If a true is returned, all additional updates will be canceled.
    fn stop_engine(&self) -> bool {
        false
    }

    /// Should the `State` stop updating.
    ///
    /// The method is called after each state update.
    /// If a true is returned, all additional updates for the specified state will be canceled.
    fn stop_state(&self, _state_id: usize) -> bool {
        false
    }

    /// Cleanup upon the end of updating.
    ///
    /// This method is called when all updating is complete.
    /// Uses for this method include cleanup, report generation, etc.
    fn finalize(&mut self) {}
}

macro_rules! impl_tuple {
($($idx:tt $t:tt),+) => {
    impl<$($t,)+> UpdateHandler for ($($t,)+)
    where
        $($t: UpdateHandler,)+
    {

        fn global_init(&mut self, config: &EngineUpdateConfig, states: &[State]) {
            $(
                self.$idx.global_init(config, states);
            )+
        }

        fn new_state_init(&mut self, state_id: usize, state: &State) {
            $(
                self.$idx.new_state_init(state_id, state);
            )+
        }

        fn state_updated(&mut self, state_id: usize, state: &State) {
            $(
                self.$idx.state_updated(state_id, state);
            )+
        }
        fn state_complete(&mut self, state_id: usize, state: &State) {
            $(
                self.$idx.state_complete(state_id, state);
            )+
        }

        fn stop_engine(&self) -> bool {
            $(
                self.$idx.stop_engine()
            )||+
        }

        fn stop_state(&self, state_id: usize) -> bool {
            $(
                self.$idx.stop_state(state_id)
            )||+
        }

        fn finalize(&mut self) {
            $(
                self.$idx.finalize();
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
    fn global_init(&mut self, config: &EngineUpdateConfig, states: &[State]) {
        self.iter_mut()
            .for_each(|handler| handler.global_init(config, states));
    }

    fn state_updated(&mut self, state_id: usize, state: &State) {
        self.iter_mut().for_each(|handler| {
            handler.state_updated(state_id, state);
        })
    }

    fn stop_engine(&self) -> bool {
        self.iter().any(|handler| handler.stop_engine())
    }

    fn new_state_init(&mut self, state_id: usize, state: &State) {
        self.iter_mut()
            .for_each(|handler| handler.new_state_init(state_id, state));
    }

    fn state_complete(&mut self, state_id: usize, state: &State) {
        self.iter_mut()
            .for_each(|handler| handler.state_complete(state_id, state));
    }

    fn stop_state(&self, _state_id: usize) -> bool {
        false
    }

    fn finalize(&mut self) {
        self.iter_mut().for_each(|handler| handler.finalize());
    }
}

/// Handle Ctrl-C (sigint) signals by stopping the Engine.
#[cfg(feature = "ctrlc_handler")]
#[derive(Clone)]
pub struct CtrlC {
    seen_sigint: Arc<AtomicBool>,
}

#[cfg(feature = "ctrlc_handler")]
impl Default for CtrlC {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "ctrlc_handler")]
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

#[cfg(feature = "ctrlc_handler")]
impl UpdateHandler for CtrlC {
    fn global_init(&mut self, _config: &EngineUpdateConfig, _states: &[State]) {
    }

    fn state_updated(&mut self, _state_id: usize, _state: &State) {}

    fn stop_engine(&self) -> bool {
        self.seen_sigint.load(Ordering::Relaxed)
    }
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
    fn global_init(&mut self, _config: &EngineUpdateConfig, _states: &[State]) {
        if let Self::UnInitialized { timeout } = self {
            *self = Self::Initialized {
                start: Instant::now(),
                timeout: *timeout,
            };
        };
    }

    fn state_updated(&mut self, _state_id: usize, _state: &State) {}

    fn stop_engine(&self) -> bool {
        if let Self::Initialized { start, timeout } = self {
            start.elapsed() > *timeout
        } else {
            unreachable!()
        }
    }

    fn finalize(&mut self) {}
}

/// Limit the time each state can run for during an `Engine::update`.
#[derive(Clone)]
pub enum StateTimeout {
    UnInitialized {
        timeout: Duration,
    },
    Initialized {
        timeout: Duration,
        state_start: Arc<RwLock<HashMap<usize, Instant>>>,
    },
}

impl StateTimeout {
    /// Create a new `StateTimeout` with the per-state timeout given.
    pub fn new(timeout: Duration) -> Self {
        Self::UnInitialized { timeout }
    }
}

impl UpdateHandler for StateTimeout {
    fn global_init(&mut self, _config: &EngineUpdateConfig, _states: &[State]) {
        if let Self::UnInitialized { timeout } = *self {
            *self = Self::Initialized {
                timeout,
                state_start: Arc::new(RwLock::new(HashMap::new())),
            };
        }
    }

    fn new_state_init(&mut self, state_id: usize, _state: &State) {
        if let Self::Initialized { state_start, .. } = self {
            let mut state_start = state_start
                .write()
                .expect("Shoule be able to lock the state_start for writing.");
            state_start.insert(state_id, Instant::now());
        }
    }

    fn stop_state(&self, state_id: usize) -> bool {
        if let Self::Initialized {
            timeout,
            state_start,
            ..
        } = self
        {
            let state_start = state_start
                .read()
                .expect("Shoule be able to lock the state_start for reading.");
            if let Some(start_time) = state_start.get(&state_id) {
                start_time.elapsed() > *timeout
            } else {
                unreachable!()
            }
        } else {
            unreachable!()
        }
    }
}

impl UpdateHandler for () {}

/// Add a progress bar to the output
#[derive(Clone)]
pub enum ProgressBar {
    UnInitialized,
    Initialized {
        sender: Arc<Mutex<Sender<(usize, f64)>>>,
        handle: Arc<Mutex<Option<JoinHandle<()>>>>,
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
    fn global_init(&mut self, config: &EngineUpdateConfig, states: &[State]) {
        const UPDATE_INTERVAL: Duration = Duration::from_millis(250);

        let (sender, receiver) = std::sync::mpsc::channel();
        let total_iters = states.len() * config.n_iters;

        let handle = std::thread::spawn(move || {
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
                    progress_bar.set_position(completed_iters as u64);
                    let mean_log_score = state_log_scores.values().sum::<f64>()
                        / (state_log_scores.len() as f64);
                    progress_bar.set_message(format!("{:.2}", mean_log_score));
                }
            }

            progress_bar.finish_and_clear();
        });

        *self = Self::Initialized {
            sender: Arc::new(Mutex::new(sender)),
            handle: Arc::new(Mutex::new(Some(handle))),
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

    fn stop_engine(&self) -> bool {
        false
    }

    fn finalize(&mut self) {
        if let Self::Initialized { sender, handle } = std::mem::take(self) {
            std::mem::drop(sender);

            if let Some(handle) = handle.lock().unwrap().take() {
                handle.join().unwrap();
            }
        }
    }
}
