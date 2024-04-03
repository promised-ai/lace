use super::View;

use crate::alg::RowAssignAlg;
use crate::feature::geweke::GewekeColumnSummary;
use crate::feature::geweke::{gen_geweke_col_models, ColumnGewekeSettings};
use crate::feature::FType;
use crate::feature::Feature;
use crate::transition::ViewTransition;

use lace_geweke::{GewekeModel, GewekeResampleData, GewekeSummarize};
use lace_stats::prior_process;
use lace_stats::prior_process::PriorProcess;
use lace_stats::prior_process::PriorProcessType;
use lace_stats::prior_process::Process;
use rand::Rng;
use std::collections::BTreeMap;

pub struct ViewGewekeSettings {
    /// The number of columns/features in the view
    pub n_cols: usize,
    /// The number of rows in the view
    pub n_rows: usize,
    /// Column model types
    pub cm_types: Vec<FType>,
    /// Which transitions to run
    pub transitions: Vec<ViewTransition>,
    /// Which prior process to use
    pub process_type: PriorProcessType,
}

impl ViewGewekeSettings {
    pub fn new(n_rows: usize, cm_types: Vec<FType>) -> Self {
        ViewGewekeSettings {
            n_rows,
            n_cols: cm_types.len(),
            cm_types,
            // XXX: You HAVE to run component params update explicitly for gibbs
            // and SAMS reassignment kernels because these algorithms do not do
            // parameter updates explicitly (they marginalize over the component
            // parameters) and the data resample relies on the component
            // parameters.
            process_type: PriorProcessType::Dirichlet,
            transitions: vec![
                ViewTransition::RowAssignment(RowAssignAlg::Slice),
                ViewTransition::FeaturePriors,
                ViewTransition::ComponentParams,
                ViewTransition::PriorProcessParams,
            ],
        }
    }

    pub fn with_pitman_yor_process(mut self) -> Self {
        self.process_type = PriorProcessType::PitmanYor;
        self
    }

    pub fn with_dirichlet_process(mut self) -> Self {
        self.process_type = PriorProcessType::Dirichlet;
        self
    }

    pub fn do_row_asgn_transition(&self) -> bool {
        self.transitions
            .iter()
            .any(|t| matches!(t, ViewTransition::RowAssignment(_)))
    }

    pub fn do_process_params_transition(&self) -> bool {
        self.transitions
            .iter()
            .any(|t| matches!(t, ViewTransition::PriorProcessParams))
    }
}

fn view_geweke_asgn<R: Rng>(
    n_rows: usize,
    do_process_params_transition: bool,
    do_row_asgn_transition: bool,
    process_type: PriorProcessType,
    rng: &mut R,
) -> (prior_process::Builder, Process) {
    use lace_consts::geweke_alpha_prior;
    let process = match process_type {
        PriorProcessType::Dirichlet => {
            use lace_stats::prior_process::Dirichlet;
            let inner = if do_process_params_transition {
                Dirichlet::from_prior(geweke_alpha_prior(), rng)
            } else {
                Dirichlet {
                    alpha: 1.0,
                    alpha_prior: geweke_alpha_prior(),
                }
            };
            Process::Dirichlet(inner)
        }
        PriorProcessType::PitmanYor => {
            use lace_stats::prior_process::PitmanYor;
            use lace_stats::rv::dist::Beta;
            let inner = if do_process_params_transition {
                PitmanYor::from_prior(
                    geweke_alpha_prior(),
                    Beta::jeffreys(),
                    rng,
                )
            } else {
                PitmanYor {
                    alpha: 1.0,
                    d: 0.2,
                    alpha_prior: geweke_alpha_prior(),
                    d_prior: Beta::jeffreys(),
                }
            };
            Process::PitmanYor(inner)
        }
    };
    let mut bldr =
        prior_process::Builder::new(n_rows).with_process(process.clone());

    if !do_row_asgn_transition {
        bldr = bldr.flat();
    }

    (bldr, process)
}

impl GewekeModel for View {
    fn geweke_from_prior(
        settings: &ViewGewekeSettings,
        mut rng: &mut impl Rng,
    ) -> View {
        let do_ftr_prior_transition = settings
            .transitions
            .iter()
            .any(|&t| t == ViewTransition::FeaturePriors);

        // FIXME: Redundant! asgn_builder builds a PriorProcess
        let (asgn_builder, process) = view_geweke_asgn(
            settings.n_rows,
            settings.do_process_params_transition(),
            settings.do_row_asgn_transition(),
            settings.process_type,
            rng,
        );
        let asgn = asgn_builder.seed_from_rng(&mut rng).build().unwrap();

        // this function sets up dummy features that we can properly populate with
        // Feature.geweke_init in the next loop
        let mut ftrs = gen_geweke_col_models(
            &settings.cm_types,
            settings.n_rows,
            do_ftr_prior_transition,
            &mut rng,
        );

        let ftrs: BTreeMap<_, _> = ftrs
            .drain(..)
            .enumerate()
            .map(|(id, mut ftr)| {
                ftr.geweke_init(&asgn.asgn, &mut rng);
                (id, ftr)
            })
            .collect();

        let prior_process = PriorProcess {
            process,
            asgn: asgn.asgn,
        };

        View {
            ftrs,
            weights: prior_process.weight_vec(false),
            prior_process,
        }
    }

    fn geweke_step(
        &mut self,
        settings: &ViewGewekeSettings,
        mut rng: &mut impl Rng,
    ) {
        self.step(&settings.transitions, &mut rng);
    }
}

impl GewekeResampleData for View {
    type Settings = ViewGewekeSettings;
    fn geweke_resample_data(
        &mut self,
        settings: Option<&ViewGewekeSettings>,
        rng: &mut impl Rng,
    ) {
        let s = settings.unwrap();
        let col_settings = ColumnGewekeSettings::new(
            self.asgn().clone(),
            s.transitions.clone(),
        );
        for ftr in self.ftrs.values_mut() {
            ftr.geweke_resample_data(Some(&col_settings), rng);
        }
    }
}

/// The View summary for Geweke
#[derive(Clone, Debug)]
pub struct GewekeViewSummary {
    /// The number of categories
    pub n_cats: Option<usize>,
    /// The CRP alpha
    pub alpha: Option<f64>,
    /// The summary for each column/feature.
    pub cols: Vec<(usize, GewekeColumnSummary)>,
}

impl From<&GewekeViewSummary> for BTreeMap<String, f64> {
    fn from(value: &GewekeViewSummary) -> BTreeMap<String, f64> {
        let mut map: BTreeMap<String, f64> = BTreeMap::new();
        if let Some(n_cats) = value.n_cats {
            map.insert("n_cats".into(), n_cats as f64);
        }

        if let Some(alpha) = value.alpha {
            map.insert("crp alpha".into(), alpha);
        }

        value.cols.iter().for_each(|(id, col_summary)| {
            let summary_map: BTreeMap<String, f64> = col_summary.into();
            summary_map.iter().for_each(|(key, value)| {
                let new_key = format!("Col {id} {key}");
                map.insert(new_key, *value);
            });
        });
        map
    }
}

impl From<GewekeViewSummary> for BTreeMap<String, f64> {
    fn from(value: GewekeViewSummary) -> BTreeMap<String, f64> {
        Self::from(&value)
    }
}

impl GewekeSummarize for View {
    type Summary = GewekeViewSummary;

    fn geweke_summarize(&self, settings: &ViewGewekeSettings) -> Self::Summary {
        let col_settings = ColumnGewekeSettings::new(
            self.asgn().clone(),
            settings.transitions.clone(),
        );

        GewekeViewSummary {
            n_cats: if settings.do_row_asgn_transition() {
                Some(self.n_cats())
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
            cols: self
                .ftrs
                .values()
                .map(|ftr| (ftr.id(), ftr.geweke_summarize(&col_settings)))
                .collect(),
        }
    }
}
