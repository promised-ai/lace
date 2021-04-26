use braid_codebook::ColType;
use braid_stats::labeler::{Label, Labeler, LabelerPrior};
use braid_stats::prior::crp::CrpPrior;
use braid_stats::prior::csd::CsdHyper;
use braid_stats::prior::nix::NixHyper;
use braid_stats::prior::pg::PgHyper;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rv::dist::{Categorical, Gamma, Gaussian, NormalInvChiSquared, Poisson};
use rv::traits::*;
use thiserror::Error;

use crate::cc::{
    AssignmentBuilder, ColModel, Column, Feature, State, ViewBuilder,
};
use braid_data::SparseContainer;

/// Builds a dummy state with a given size and structure
#[derive(Debug, Clone)]
pub struct StateBuilder {
    pub nrows: Option<usize>,
    pub nviews: Option<usize>,
    pub ncats: Option<usize>,
    pub col_configs: Option<Vec<ColType>>,
    pub ftrs: Option<Vec<ColModel>>,
    pub seed: Option<u64>,
}

impl Default for StateBuilder {
    fn default() -> Self {
        StateBuilder {
            nrows: None,
            nviews: None,
            ncats: None,
            col_configs: None,
            ftrs: None,
            seed: None,
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum BuildStateError {
    #[error("Supply either features or column configs; not both")]
    BothColumnConfigsAndFeaturesPresent,
    #[error("No column configs or features supplied")]
    NeitherColumnConfigsAndFeaturesPresent,
}

/// Builds a state with a given complexity for benchmarking and testing purposes
impl StateBuilder {
    pub fn new() -> Self {
        StateBuilder::default()
    }

    /// Set the number of rows
    pub fn with_rows(mut self, nrows: usize) -> Self {
        self.nrows = Some(nrows);
        self
    }

    /// Set the number of views -- must be less than or equal to the number
    /// of columns
    pub fn with_views(mut self, nviews: usize) -> Self {
        self.nviews = Some(nviews);
        self
    }

    /// Set the number of categories -- must be less than or equal to the
    /// number of rows.
    pub fn with_cats(mut self, ncats: usize) -> Self {
        self.ncats = Some(ncats);
        self
    }

    /// Use a specific set of features.
    pub fn add_features(mut self, ftrs: Vec<ColModel>) -> Self {
        self.ftrs = Some(ftrs);
        self
    }

    /// Push a column configuration, adding one additional column.
    pub fn add_column_config(mut self, col_config: ColType) -> Self {
        if let Some(ref mut col_configs) = self.col_configs {
            col_configs.push(col_config);
        } else {
            self.col_configs = Some(vec![col_config]);
        }
        self
    }

    /// Push a number of column configurations
    pub fn add_column_configs(mut self, n: usize, col_config: ColType) -> Self {
        if let Some(ref mut col_configs) = self.col_configs {
            col_configs.append(&mut vec![col_config; n]);
        } else {
            self.col_configs = Some(vec![col_config; n]);
        }
        self
    }

    /// Seed from an RNG
    pub fn seed_from_rng<R: rand::Rng>(mut self, rng: &mut R) -> Self {
        self.seed = Some(rng.next_u64());
        self
    }

    /// With an RNG seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build the `State`
    pub fn build(self) -> Result<State, BuildStateError> {
        let mut rng = match self.seed {
            Some(seed) => Xoshiro256Plus::seed_from_u64(seed),
            None => Xoshiro256Plus::from_entropy(),
        };

        // TODO: this is gross and indicates a lot of issues that should be
        // fixed in the next patch-version release
        let nrows = self
            .ftrs
            .as_ref()
            .map(|ftrs| ftrs[0].len())
            .or(self.nrows)
            .unwrap_or(100);

        let nviews = self.nviews.unwrap_or(1);
        let ncats = self.ncats.unwrap_or(1);

        if self.col_configs.is_some() && self.ftrs.is_some() {
            return Err(BuildStateError::BothColumnConfigsAndFeaturesPresent);
        } else if self.col_configs.is_none() && self.ftrs.is_none() {
            return Err(
                BuildStateError::NeitherColumnConfigsAndFeaturesPresent,
            );
        }

        let mut ftrs = if self.col_configs.is_some() {
            self.col_configs
                .unwrap()
                .iter()
                .enumerate()
                .map(|(id, col_config)| {
                    gen_feature(id, col_config.clone(), nrows, ncats, &mut rng)
                })
                .collect()
        } else {
            // XXX: might want to clone if using the builder repeatedly
            self.ftrs.unwrap()
        };

        let mut col_asgn: Vec<usize> = vec![];
        let mut col_counts: Vec<usize> = vec![];
        let ftrs_per_view = ftrs.len() / nviews;
        let views = (0..nviews)
            .map(|view_ix| {
                let ftrs_left = ftrs.len();
                let to_drain = if view_ix == nviews - 1 {
                    ftrs_left
                } else {
                    ftrs_per_view
                };

                col_asgn.append(&mut vec![view_ix; to_drain]);
                col_counts.push(to_drain);
                let ftrs_view = ftrs.drain(0..to_drain).map(|f| f).collect();
                let asgn = AssignmentBuilder::new(nrows)
                    .with_ncats(ncats)
                    .unwrap()
                    .seed_from_rng(&mut rng)
                    .build()
                    .unwrap();
                ViewBuilder::from_assignment(asgn)
                    .with_features(ftrs_view)
                    .seed_from_rng(&mut rng)
                    .build()
            })
            .collect();

        assert_eq!(ftrs.len(), 0);

        let asgn = AssignmentBuilder::from_vec(col_asgn)
            .seed_from_rng(&mut rng)
            .build()
            .unwrap();
        let alpha_prior: CrpPrior = braid_consts::state_alpha_prior().into();
        Ok(State::new(views, asgn, alpha_prior))
    }
}

fn gen_feature<R: rand::Rng>(
    id: usize,
    col_config: ColType,
    nrows: usize,
    ncats: usize,
    mut rng: &mut R,
) -> ColModel {
    match col_config {
        ColType::Continuous { .. } => {
            let hyper = NixHyper::default();
            let prior = NormalInvChiSquared::new_unchecked(0.0, 1.0, 4.0, 4.0);
            let g = Gaussian::standard();
            let xs: Vec<f64> = g.sample(nrows, &mut rng);
            let data = SparseContainer::from(xs);
            let col = Column::new(id, data, prior, hyper);
            ColModel::Continuous(col)
        }
        ColType::Count { .. } => {
            let hyper = PgHyper::default();
            let prior = Gamma::new_unchecked(1.0, 1.0);
            let pois = Poisson::new_unchecked(1.0);
            let xs: Vec<u32> = pois.sample(nrows, &mut rng);
            let data = SparseContainer::from(xs);
            let col = Column::new(id, data, prior, hyper);
            ColModel::Count(col)
        }
        ColType::Categorical { k, .. } => {
            let hyper = CsdHyper::vague(k);
            let prior = braid_stats::prior::csd::vague(k, &mut rng);
            let components: Vec<Categorical> =
                (0..ncats).map(|_| prior.draw(&mut rng)).collect();
            let xs: Vec<u8> = (0..nrows)
                .map::<u8, _>(|i| components[i % ncats].draw::<R>(&mut rng))
                .collect();
            let data = SparseContainer::from(xs);
            let col = Column::new(id, data, prior, hyper);
            ColModel::Categorical(col)
        }
        ColType::Labeler { n_labels, .. } => {
            let prior = LabelerPrior::standard(n_labels);
            let components: Vec<Labeler> =
                (0..ncats).map(|_| prior.draw(&mut rng)).collect();
            let xs: Vec<Label> = components[0].sample(nrows, &mut rng);
            let data = SparseContainer::from(xs);
            let col = Column::new(id, data, prior, ());
            ColModel::Labeler(col)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cc::config::StateUpdateConfig;

    #[test]
    fn test_dimensions() {
        let state = StateBuilder::new()
            .add_column_configs(
                10,
                ColType::Continuous {
                    hyper: None,
                    prior: None,
                },
            )
            .with_rows(50)
            .build()
            .expect("Failed to build state");

        assert_eq!(state.nrows(), 50);
        assert_eq!(state.ncols(), 10);
    }

    #[test]
    fn built_state_should_update() {
        let mut rng = Xoshiro256Plus::from_entropy();
        let mut state = StateBuilder::new()
            .add_column_configs(
                10,
                ColType::Continuous {
                    hyper: None,
                    prior: None,
                },
            )
            .with_rows(50)
            .seed_from_rng(&mut rng)
            .build()
            .expect("Failed to build state");

        let config = StateUpdateConfig {
            n_iters: 5,
            ..Default::default()
        };
        state.update(config, &mut rng);
    }

    #[test]
    fn seeding_state_works() {
        let state_1 = {
            let mut rng = Xoshiro256Plus::seed_from_u64(122445);
            StateBuilder::new()
                .add_column_configs(
                    10,
                    ColType::Continuous {
                        hyper: None,
                        prior: None,
                    },
                )
                .with_rows(50)
                .seed_from_rng(&mut rng)
                .build()
                .expect("Failed to build state")
        };

        let state_2 = {
            let mut rng = Xoshiro256Plus::seed_from_u64(122445);
            StateBuilder::new()
                .add_column_configs(
                    10,
                    ColType::Continuous {
                        hyper: None,
                        prior: None,
                    },
                )
                .with_rows(50)
                .seed_from_rng(&mut rng)
                .build()
                .expect("Failed to build state")
        };

        assert_eq!(state_1.asgn.asgn, state_2.asgn.asgn);

        for (view_1, view_2) in state_1.views.iter().zip(state_2.views.iter()) {
            assert_eq!(view_1.asgn.asgn, view_2.asgn.asgn);
        }
    }

    #[test]
    fn nrows_overriden_by_features() {
        let ncols = 5;
        let col_models = {
            let state = StateBuilder::new()
                .add_column_configs(
                    ncols,
                    ColType::Continuous {
                        hyper: None,
                        prior: None,
                    },
                )
                .with_rows(11)
                .build()
                .unwrap();

            (0..ncols)
                .map(|ix| state.feature(ix))
                .cloned()
                .collect::<Vec<_>>()
        };

        let state = StateBuilder::new()
            .add_features(col_models)
            .with_rows(101)
            .build()
            .unwrap();

        assert_eq!(state.nrows(), 11);
    }
}