use super::State;
use super::StateDiagnostics;
use super::StateScoreComponents;

use crate::config::StateUpdateConfig;
use crate::feature::geweke::gen_geweke_col_models;
use crate::feature::FType;
use crate::transition::StateTransition;
use crate::view;
use crate::view::geweke::GewekeViewSummary;
use crate::view::geweke::ViewGewekeSettings;
use crate::view::View;

use lace_consts::geweke_alpha_prior;
use lace_geweke::{GewekeModel, GewekeResampleData, GewekeSummarize};
use lace_stats::prior_process;
use lace_stats::prior_process::PriorProcessType;
use lace_stats::prior_process::Process;
use rand::Rng;
use serde::Deserialize;
use serde::Serialize;
use std::collections::BTreeMap;
use std::convert::TryInto;

#[derive(Clone, Serialize, Deserialize)]
pub struct StateGewekeSettings {
    /// The number of columns/features in the state
    pub n_cols: usize,
    /// The number of rows in the state
    pub n_rows: usize,
    /// Column Model types
    pub cm_types: Vec<FType>,
    /// Which transitions to do
    pub transitions: Vec<StateTransition>,
    /// Which prior process to use for the State assignment
    pub state_process_type: PriorProcessType,
    /// Which prior process to use for the View assignment
    pub view_process_type: PriorProcessType,
}

impl StateGewekeSettings {
    pub fn new(
        n_rows: usize,
        cm_types: Vec<FType>,
        state_process_type: PriorProcessType,
        view_process_type: PriorProcessType,
    ) -> Self {
        use crate::transition::DEFAULT_STATE_TRANSITIONS;

        StateGewekeSettings {
            n_cols: cm_types.len(),
            n_rows,
            cm_types,
            transitions: DEFAULT_STATE_TRANSITIONS.into(),
            state_process_type,
            view_process_type,
        }
    }

    pub fn new_dirichlet_process(n_rows: usize, cm_types: Vec<FType>) -> Self {
        use crate::transition::DEFAULT_STATE_TRANSITIONS;

        StateGewekeSettings {
            n_cols: cm_types.len(),
            n_rows,
            cm_types,
            transitions: DEFAULT_STATE_TRANSITIONS.into(),
            state_process_type: PriorProcessType::Dirichlet,
            view_process_type: PriorProcessType::Dirichlet,
        }
    }

    pub fn do_col_asgn_transition(&self) -> bool {
        self.transitions
            .iter()
            .any(|t| matches!(t, StateTransition::ColumnAssignment(_)))
    }

    pub fn do_row_asgn_transition(&self) -> bool {
        self.transitions
            .iter()
            .any(|t| matches!(t, StateTransition::RowAssignment(_)))
    }

    pub fn do_process_params_transition(&self) -> bool {
        self.transitions
            .iter()
            .any(|t| matches!(t, StateTransition::StatePriorProcessParams))
    }
}

impl GewekeResampleData for State {
    type Settings = StateGewekeSettings;

    fn geweke_resample_data(
        &mut self,
        settings: Option<&StateGewekeSettings>,
        mut rng: &mut impl Rng,
    ) {
        let s = settings.unwrap();
        // XXX: View.geweke_resample_data only needs the transitions
        let view_settings = ViewGewekeSettings {
            n_rows: 0,
            n_cols: 0,
            cm_types: vec![],
            transitions: s
                .transitions
                .iter()
                .filter_map(|&st| st.try_into().ok())
                .collect(),
            process_type: s.view_process_type,
        };
        for view in &mut self.views {
            view.geweke_resample_data(Some(&view_settings), &mut rng);
        }
    }
}

/// The State summary for Geweke
#[derive(Clone, Debug)]
pub struct GewekeStateSummary {
    /// The number of views
    pub n_views: Option<usize>,
    /// CRP alpha
    pub alpha: Option<f64>,
    /// The summary for each view
    pub views: Vec<GewekeViewSummary>,
}

impl From<&GewekeStateSummary> for BTreeMap<String, f64> {
    fn from(value: &GewekeStateSummary) -> Self {
        let mut map: BTreeMap<String, f64> = BTreeMap::new();

        if let Some(n_views) = value.n_views {
            map.insert("n views".into(), n_views as f64);
        }

        if let Some(alpha) = value.alpha {
            map.insert("crp alpha".into(), alpha);
        }

        value.views.iter().for_each(|view_summary| {
            let view_map: BTreeMap<String, f64> = view_summary.into();
            view_map.iter().for_each(|(key, val)| {
                map.insert(key.clone(), *val);
            });
        });
        map
    }
}

impl From<GewekeStateSummary> for BTreeMap<String, f64> {
    fn from(value: GewekeStateSummary) -> Self {
        Self::from(&value)
    }
}

impl GewekeSummarize for State {
    type Summary = GewekeStateSummary;

    fn geweke_summarize(
        &self,
        settings: &StateGewekeSettings,
    ) -> GewekeStateSummary {
        // Dummy settings. the only thing the view summarizer cares about is the
        // transitions.
        let view_settings = ViewGewekeSettings {
            n_cols: 0,
            n_rows: 0,
            cm_types: vec![],
            transitions: settings
                .transitions
                .iter()
                .filter_map(|&st| st.try_into().ok())
                .collect(),
            process_type: settings.view_process_type,
        };

        GewekeStateSummary {
            n_views: if settings.do_col_asgn_transition() {
                Some(self.asgn().n_cats)
            } else {
                None
            },
            alpha: if settings.do_process_params_transition() {
                Some(match self.prior_process.process {
                    Process::Dirichlet(ref inner) => inner.alpha,
                    Process::PitmanYor(ref inner) => inner.alpha,
                })
            } else {
                None
            },
            views: self
                .views
                .iter()
                .map(|view| view.geweke_summarize(&view_settings))
                .collect(),
        }
    }
}

// XXX: Note that the only Geweke is only guaranteed to return turn results if
// all transitions are on. For example, we can turn off the view alphas
// transition, but the Gibbs column transition will create new views with
// alpha drawn from the prior. As of now, the State has no way of knowing that
// the View alphas are 'turned off', so it initializes new Views from the
// prior. So yeah, make sure that all transitions work, and maybe later we'll
// build knowledge of the transition set into the state.
impl GewekeModel for State {
    fn geweke_from_prior(
        settings: &StateGewekeSettings,
        mut rng: &mut impl Rng,
    ) -> Self {
        use lace_stats::prior_process::Dirichlet as PDirichlet;

        let has_transition = |t: StateTransition, s: &StateGewekeSettings| {
            s.transitions.iter().any(|&ti| ti == t)
        };
        // TODO: Generate new rng from randomly-drawn seed
        // TODO: Draw features properly depending on the transitions
        let do_ftr_prior_transition =
            has_transition(StateTransition::FeaturePriors, settings);

        let do_state_process_transition =
            has_transition(StateTransition::StatePriorProcessParams, settings);

        let do_view_process_transition =
            has_transition(StateTransition::ViewPriorProcessParams, settings);

        let do_col_asgn_transition = settings.do_col_asgn_transition();
        let do_row_asgn_transition = settings.do_row_asgn_transition();

        let mut ftrs = gen_geweke_col_models(
            &settings.cm_types,
            settings.n_rows,
            do_ftr_prior_transition,
            &mut rng,
        );

        let n_cols = ftrs.len();

        let state_prior_process = {
            let process = if do_state_process_transition {
                Process::Dirichlet(PDirichlet::from_prior(
                    geweke_alpha_prior(),
                    &mut rng,
                ))
            } else {
                Process::Dirichlet(PDirichlet {
                    alpha_prior: geweke_alpha_prior(),
                    alpha: 1.0,
                })
            };

            if do_col_asgn_transition {
                prior_process::Builder::new(n_cols)
            } else {
                prior_process::Builder::new(n_cols).flat()
            }
            .with_process(process.clone())
            .seed_from_rng(&mut rng)
            .build()
            .unwrap()
        };

        let view_asgn_bldr = if do_row_asgn_transition {
            prior_process::Builder::new(settings.n_rows)
        } else {
            prior_process::Builder::new(settings.n_rows).flat()
        };

        let mut views: Vec<View> = (0..state_prior_process.asgn.n_cats)
            .map(|_| {
                // may need to redraw the process params from the prior many
                // times, so Process construction must be a generating function
                let process = if do_view_process_transition {
                    Process::Dirichlet(PDirichlet::from_prior(
                        geweke_alpha_prior(),
                        &mut rng,
                    ))
                } else {
                    Process::Dirichlet(PDirichlet {
                        alpha_prior: geweke_alpha_prior(),
                        alpha: 1.0,
                    })
                };

                let asgn = view_asgn_bldr
                    .clone()
                    .seed_from_rng(&mut rng)
                    .with_process(process.clone())
                    .build()
                    .unwrap();
                view::Builder::from_prior_process(asgn)
                    .seed_from_rng(&mut rng)
                    .build()
            })
            .collect();

        for (&v, ftr) in
            state_prior_process.asgn.asgn.iter().zip(ftrs.drain(..))
        {
            views[v].geweke_init_feature(ftr, &mut rng);
        }

        State {
            views,
            weights: state_prior_process.weight_vec(false),
            prior_process: state_prior_process,
            score: StateScoreComponents::default(),
            diagnostics: StateDiagnostics::default(),
        }
    }

    fn geweke_step(
        &mut self,
        settings: &StateGewekeSettings,
        mut rng: &mut impl Rng,
    ) {
        let config = StateUpdateConfig {
            transitions: settings.transitions.clone(),
            n_iters: 1,
        };

        self.refresh_suffstats(&mut rng);
        self.update(config, &mut rng);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::alg::ColAssignAlg;
    use crate::alg::RowAssignAlg;

    struct StateFlatnessResult {
        pub rows_always_flat: bool,
        pub cols_always_flat: bool,
        pub state_alpha_1: bool,
        pub view_alphas_1: bool,
    }

    fn test_asgn_flatness(
        settings: &StateGewekeSettings,
        n_runs: usize,
        mut rng: &mut impl Rng,
    ) -> StateFlatnessResult {
        let mut cols_always_flat = true;
        let mut rows_always_flat = true;
        let mut state_alpha_1 = true;
        let mut view_alphas_1 = true;

        let basically_one = |x: f64| (x - 1.0).abs() < 1E-12;

        for _ in 0..n_runs {
            let state = State::geweke_from_prior(settings, &mut rng);

            // Column assignment is not flat
            if state.asgn().asgn.iter().any(|&zi| zi != 0) {
                cols_always_flat = false;
            }

            let alpha = match state.prior_process.process {
                Process::Dirichlet(ref inner) => inner.alpha,
                Process::PitmanYor(ref inner) => inner.alpha,
            };

            if !basically_one(alpha) {
                state_alpha_1 = false;
            }

            // 2. Check the column priors
            for view in state.views.iter() {
                // Check the view assignment priors
                // Check the view assignments aren't flat
                if view.asgn().asgn.iter().any(|&zi| zi != 0) {
                    rows_always_flat = false;
                }
                let view_alpha = match view.prior_process.process {
                    Process::Dirichlet(ref inner) => inner.alpha,
                    Process::PitmanYor(ref inner) => inner.alpha,
                };

                if !basically_one(view_alpha) {
                    view_alphas_1 = false;
                }
            }
        }

        StateFlatnessResult {
            rows_always_flat,
            cols_always_flat,
            state_alpha_1,
            view_alphas_1,
        }
    }

    #[test]
    fn geweke_from_prior_all_transitions() {
        let settings = StateGewekeSettings::new_dirichlet_process(
            50,
            vec![FType::Continuous; 40],
        );
        let mut rng = rand::thread_rng();
        let result = test_asgn_flatness(&settings, 10, &mut rng);
        assert!(!result.rows_always_flat);
        assert!(!result.cols_always_flat);
        assert!(!result.view_alphas_1);
        assert!(!result.state_alpha_1);
    }

    #[test]
    fn geweke_from_prior_no_row_transition() {
        let settings = StateGewekeSettings {
            n_cols: 20,
            n_rows: 50,
            cm_types: vec![FType::Continuous; 20],
            transitions: vec![
                StateTransition::ColumnAssignment(ColAssignAlg::FiniteCpu),
                StateTransition::StatePriorProcessParams,
                StateTransition::ViewPriorProcessParams,
                StateTransition::FeaturePriors,
            ],
            state_process_type: PriorProcessType::Dirichlet,
            view_process_type: PriorProcessType::Dirichlet,
        };
        let mut rng = rand::thread_rng();
        let result = test_asgn_flatness(&settings, 100, &mut rng);
        assert!(result.rows_always_flat);
        assert!(!result.cols_always_flat);
        assert!(!result.view_alphas_1);
        assert!(!result.state_alpha_1);
    }

    #[test]
    fn geweke_from_prior_no_col_transition() {
        let settings = StateGewekeSettings {
            n_cols: 20,
            n_rows: 50,
            cm_types: vec![FType::Continuous; 20],
            transitions: vec![
                StateTransition::RowAssignment(RowAssignAlg::FiniteCpu),
                StateTransition::StatePriorProcessParams,
                StateTransition::ViewPriorProcessParams,
                StateTransition::FeaturePriors,
            ],
            state_process_type: PriorProcessType::Dirichlet,
            view_process_type: PriorProcessType::Dirichlet,
        };
        let mut rng = rand::thread_rng();
        let result = test_asgn_flatness(&settings, 100, &mut rng);
        assert!(!result.rows_always_flat);
        assert!(result.cols_always_flat);
        assert!(!result.view_alphas_1);
        assert!(!result.state_alpha_1);
    }

    #[test]
    fn geweke_from_prior_no_row_or_col_transition() {
        let settings = StateGewekeSettings {
            n_cols: 20,
            n_rows: 50,
            cm_types: vec![FType::Continuous; 20],
            transitions: vec![
                StateTransition::StatePriorProcessParams,
                StateTransition::ViewPriorProcessParams,
                StateTransition::FeaturePriors,
            ],
            state_process_type: PriorProcessType::Dirichlet,
            view_process_type: PriorProcessType::Dirichlet,
        };
        let mut rng = rand::thread_rng();
        let result = test_asgn_flatness(&settings, 100, &mut rng);
        assert!(result.rows_always_flat);
        assert!(result.cols_always_flat);
        assert!(!result.view_alphas_1);
        assert!(!result.state_alpha_1);
    }

    #[test]
    fn geweke_from_prior_no_alpha_transition() {
        let settings = StateGewekeSettings {
            n_cols: 20,
            n_rows: 50,
            cm_types: vec![FType::Continuous; 20],
            transitions: vec![
                StateTransition::ColumnAssignment(ColAssignAlg::FiniteCpu),
                StateTransition::RowAssignment(RowAssignAlg::FiniteCpu),
                StateTransition::FeaturePriors,
            ],
            state_process_type: PriorProcessType::Dirichlet,
            view_process_type: PriorProcessType::Dirichlet,
        };
        let mut rng = rand::thread_rng();
        let result = test_asgn_flatness(&settings, 100, &mut rng);
        assert!(!result.rows_always_flat);
        assert!(!result.cols_always_flat);
        assert!(result.state_alpha_1);
        assert!(result.view_alphas_1);
    }
}
