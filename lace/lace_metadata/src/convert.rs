//! Conversion traits
use crate::latest;
use crate::versions::{v1, v2};
use lace_cc::component::ConjugateComponent;
use lace_data::label::Label;
use lace_stats::labeler::{Labeler, LabelerPrior};
use lace_stats::prior::csd::CsdHyper;
use lace_stats::prior::nix::NixHyper;
use lace_stats::prior::pg::PgHyper;
use lace_stats::rv::dist::{
    Categorical, Gamma, Gaussian, NormalInvChiSquared, Poisson,
    SymmetricDirichlet,
};
use once_cell::sync::OnceCell;
use std::collections::BTreeMap;
use std::convert::TryInto;

/// ===========================================================================
///                                  V1 -> Lace
/// ===========================================================================
impl From<v1::PgHyper> for PgHyper {
    fn from(h: v1::PgHyper) -> Self {
        PgHyper {
            pr_shape: h.pr_shape,
            pr_rate: lace_stats::rv::dist::InvGamma::new_unchecked(
                h.pr_rate.shape(),
                h.pr_rate.rate(),
            ),
        }
    }
}

impl From<v1::ColType> for lace_codebook::ColType {
    fn from(ct: v1::ColType) -> Self {
        match ct {
            v1::ColType::Continuous { hyper, prior } => {
                Self::Continuous { hyper, prior }
            }
            v1::ColType::Categorical {
                k,
                hyper,
                prior,
                value_map,
            } => Self::Categorical {
                k,
                hyper,
                prior,
                value_map,
            },
            v1::ColType::Count { hyper, prior } => Self::Count {
                hyper: hyper.map(PgHyper::from),
                prior,
            },
            v1::ColType::Labeler {
                n_labels,
                pr_h,
                pr_k,
                pr_world,
            } => Self::Labeler {
                n_labels,
                pr_h,
                pr_k,
                pr_world,
            },
        }
    }
}

/// ===========================================================================
///                                  V1 -> V2
/// ===========================================================================
impl From<v1::ColMetadata> for v2::ColMetadata {
    fn from(md: v1::ColMetadata) -> Self {
        Self {
            name: md.name,
            coltype: md.coltype.into(),
            notes: md.notes,
        }
    }
}

impl From<v1::Codebook> for v2::Codebook {
    fn from(mut codebook: v1::Codebook) -> Self {
        Self {
            table_name: codebook.table_name,
            state_alpha_prior: codebook.state_alpha_prior,
            view_alpha_prior: codebook.view_alpha_prior,
            col_metadata: {
                let mds: Vec<_> = codebook
                    .col_metadata
                    .drain(..)
                    .map(v2::ColMetadata::from)
                    .collect();
                mds
            },
            comments: codebook.comments,
            row_names: codebook.row_names.try_into().unwrap(),
        }
    }
}

macro_rules! from_component {
    ($x:ty, $fx:ty, $pr: ty) => {
        impl From<v1::ConjugateComponent<$x, $fx>>
            for ConjugateComponent<$x, $fx, $pr>
        {
            fn from(cpnt: v1::ConjugateComponent<$x, $fx>) -> Self {
                Self {
                    fx: cpnt.fx.into(),
                    stat: cpnt.stat.into(),
                    ln_pp_cache: OnceCell::new(),
                }
            }
        }
    };
}

macro_rules! from_dataless_column {
    ($x:ty, $fx:ty, $pr:ty, $h1:ty, $h2:ty) => {
        impl From<v1::DatalessColumn<$x, $fx, $pr, $h1>>
            for v2::DatalessColumn<$x, $fx, $pr, $h2>
        {
            fn from(mut col: v1::DatalessColumn<$x, $fx, $pr, $h1>) -> Self {
                Self {
                    id: col.id,
                    components: col
                        .components
                        .drain(..)
                        .map(|cpnt| cpnt.into())
                        .collect(),
                    prior: col.prior,
                    hyper: col.hyper.into(),
                    ignore_hyper: col.ignore_hyper,
                }
            }
        }
    };
    ($x:ty, $fx:ty, $pr:ty, $h:ty) => {
        from_dataless_column!($x, $fx, $pr, $h, $h);
    };
}

from_component!(f64, Gaussian, NormalInvChiSquared);
from_component!(u8, Categorical, SymmetricDirichlet);
from_component!(Label, Labeler, LabelerPrior);
from_component!(u32, Poisson, Gamma);

from_dataless_column!(f64, Gaussian, NormalInvChiSquared, NixHyper);
from_dataless_column!(u8, Categorical, SymmetricDirichlet, CsdHyper);
from_dataless_column!(Label, Labeler, LabelerPrior, ());
from_dataless_column!(u32, Poisson, Gamma, v1::PgHyper, PgHyper);

impl From<v1::DatalessColModel> for v2::DatalessColModel {
    fn from(col: v1::DatalessColModel) -> Self {
        match col {
            v1::DatalessColModel::Continuous(c) => Self::Continuous(c.into()),
            v1::DatalessColModel::Categorical(c) => Self::Categorical(c.into()),
            v1::DatalessColModel::Count(c) => Self::Count(c.into()),
            v1::DatalessColModel::Labeler(c) => Self::Labeler(c.into()),
        }
    }
}

impl From<v1::DatalessView> for v2::DatalessView {
    fn from(mut view: v1::DatalessView) -> Self {
        Self {
            ftrs: {
                let mut btree = BTreeMap::new();
                let ids: Vec<usize> = view.ftrs.keys().cloned().collect();
                for id in ids.iter() {
                    if let Some(ftr) = view.ftrs.remove(id) {
                        btree.insert(*id, ftr.into());
                    } else {
                        unreachable!();
                    }
                }
                btree
            },
            asgn: view.asgn,
            weights: view.weights,
        }
    }
}

impl From<v1::DatalessState> for v2::DatalessState {
    fn from(mut state: v1::DatalessState) -> Self {
        Self {
            views: state.views.drain(..).map(|v| v.into()).collect(),
            asgn: state.asgn,
            weights: state.weights,
            view_alpha_prior: state.view_alpha_prior,
            loglike: state.loglike,
            log_prior: state.log_prior,
            log_view_alpha_prior: state.log_view_alpha_prior,
            log_state_alpha_prior: state.log_state_alpha_prior,
            diagnostics: state.diagnostics,
        }
    }
}

impl From<v1::Metadata> for v2::Metadata {
    fn from(mut md: v1::Metadata) -> Self {
        Self {
            states: md.states.drain(..).map(|s| s.into()).collect(),
            state_ids: md.state_ids,
            codebook: md.codebook.into(),
            data: md.data,
            rng: md.rng,
        }
    }
}

/// ===========================================================================
///                                  V2 -> V3
/// ===========================================================================
impl From<v2::ColMetadata> for lace_codebook::ColMetadata {
    fn from(md: v2::ColMetadata) -> Self {
        Self {
            name: md.name,
            coltype: md.coltype,
            notes: md.notes,
            missing_not_at_random: false,
        }
    }
}

impl From<v2::Codebook> for latest::Codebook {
    fn from(mut codebook: v2::Codebook) -> Self {
        Self(lace_codebook::Codebook {
            table_name: codebook.table_name,
            state_alpha_prior: codebook.state_alpha_prior,
            view_alpha_prior: codebook.view_alpha_prior,
            col_metadata: codebook
                .col_metadata
                .drain(..)
                .map(|md| md.into())
                .collect::<Vec<_>>()
                .try_into()
                .expect("Failed to convert metadata vec to ColMetadataList"),
            comments: codebook.comments,
            row_names: codebook.row_names,
        })
    }
}

impl From<v2::Metadata> for latest::Metadata {
    fn from(md: v2::Metadata) -> Self {
        Self {
            states: md.states,
            state_ids: md.state_ids,
            codebook: md.codebook.into(),
            data: md.data,
            rng: md.rng,
        }
    }
}
