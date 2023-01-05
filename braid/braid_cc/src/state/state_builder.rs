use braid_codebook::ColType;
use braid_data::label::Label;
use braid_data::SparseContainer;
use braid_stats::labeler::{Labeler, LabelerPrior};
use braid_stats::prior::crp::CrpPrior;
use braid_stats::prior::csd::CsdHyper;
use braid_stats::prior::nix::NixHyper;
use braid_stats::prior::pg::PgHyper;
use braid_stats::rv::dist::{
    Categorical, Gamma, Gaussian, NormalInvChiSquared, Poisson,
};
use braid_stats::rv::traits::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use thiserror::Error;

use crate::assignment::AssignmentBuilder;
use crate::feature::{ColModel, Column, Feature};
use crate::state::State;

/// Builds a dummy state with a given size and structure
#[derive(Debug, Clone, Default)]
pub struct Builder {
    pub n_rows: Option<usize>,
    pub n_views: Option<usize>,
    pub n_cats: Option<usize>,
    pub col_configs: Option<Vec<ColType>>,
    pub ftrs: Option<Vec<ColModel>>,
    pub seed: Option<u64>,
}

#[derive(Debug, Error, PartialEq)]
pub enum BuildStateError {
    #[error("Supply either features or column configs; not both")]
    BothColumnConfigsAndFeaturesPresent,
    #[error("No column configs or features supplied")]
    NeitherColumnConfigsAndFeaturesPresent,
}

/// Builds a state with a given complexity for benchmarking and testing purposes
impl Builder {
    pub fn new() -> Self {
        Builder::default()
    }

    /// Set the number of rows
    #[must_use]
    pub fn n_rows(mut self, n_rows: usize) -> Self {
        self.n_rows = Some(n_rows);
        self
    }

    /// Set the number of views -- must be less than or equal to the number
    /// of columns
    #[must_use]
    pub fn n_views(mut self, n_views: usize) -> Self {
        self.n_views = Some(n_views);
        self
    }

    /// Set the number of categories -- must be less than or equal to the
    /// number of rows.
    #[must_use]
    pub fn n_cats(mut self, n_cats: usize) -> Self {
        self.n_cats = Some(n_cats);
        self
    }

    /// Use a specific set of features.
    #[must_use]
    pub fn features(mut self, ftrs: Vec<ColModel>) -> Self {
        self.ftrs = Some(ftrs);
        self
    }

    /// Push a column configuration, adding one additional column.
    #[must_use]
    pub fn column_config(mut self, col_config: ColType) -> Self {
        if let Some(ref mut col_configs) = self.col_configs {
            col_configs.push(col_config);
        } else {
            self.col_configs = Some(vec![col_config]);
        }
        self
    }

    /// Push a number of column configurations
    #[must_use]
    pub fn column_configs(mut self, n: usize, col_config: ColType) -> Self {
        if let Some(ref mut col_configs) = self.col_configs {
            col_configs.append(&mut vec![col_config; n]);
        } else {
            self.col_configs = Some(vec![col_config; n]);
        }
        self
    }

    /// Seed from an RNG
    #[must_use]
    pub fn seed_from_rng<R: rand::Rng>(mut self, rng: &mut R) -> Self {
        self.seed = Some(rng.next_u64());
        self
    }

    /// With an RNG seed
    #[must_use]
    pub fn seed_from_u64(mut self, seed: u64) -> Self {
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
        let n_rows = self
            .ftrs
            .as_ref()
            .map(|ftrs| ftrs[0].len())
            .or(self.n_rows)
            .unwrap_or(100);

        let n_views = self.n_views.unwrap_or(1);
        let n_cats = self.n_cats.unwrap_or(1);

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
                    gen_feature(
                        id,
                        col_config.clone(),
                        n_rows,
                        n_cats,
                        &mut rng,
                    )
                })
                .collect()
        } else {
            // XXX: might want to clone if using the builder repeatedly
            self.ftrs.unwrap()
        };

        let mut col_asgn: Vec<usize> = vec![];
        let mut col_counts: Vec<usize> = vec![];
        let ftrs_per_view = ftrs.len() / n_views;
        let views = (0..n_views)
            .map(|view_ix| {
                let ftrs_left = ftrs.len();
                let to_drain = if view_ix == n_views - 1 {
                    ftrs_left
                } else {
                    ftrs_per_view
                };

                col_asgn.append(&mut vec![view_ix; to_drain]);
                col_counts.push(to_drain);
                let ftrs_view = ftrs.drain(0..to_drain).collect();
                let asgn = AssignmentBuilder::new(n_rows)
                    .with_n_cats(n_cats)
                    .unwrap()
                    .seed_from_rng(&mut rng)
                    .build()
                    .unwrap();
                crate::view::Builder::from_assignment(asgn)
                    .features(ftrs_view)
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
    n_rows: usize,
    n_cats: usize,
    rng: &mut R,
) -> ColModel {
    match col_config {
        ColType::Continuous { .. } => {
            let hyper = NixHyper::default();
            let prior = NormalInvChiSquared::new_unchecked(0.0, 1.0, 4.0, 4.0);
            let g = Gaussian::standard();
            let xs: Vec<f64> = g.sample(n_rows, rng);
            let data = SparseContainer::from(xs);
            let col = Column::new(id, data, prior, hyper);
            ColModel::Continuous(col)
        }
        ColType::Count { .. } => {
            let hyper = PgHyper::default();
            let prior = Gamma::new_unchecked(1.0, 1.0);
            let pois = Poisson::new_unchecked(1.0);
            let xs: Vec<u32> = pois.sample(n_rows, rng);
            let data = SparseContainer::from(xs);
            let col = Column::new(id, data, prior, hyper);
            ColModel::Count(col)
        }
        ColType::Categorical { k, .. } => {
            let hyper = CsdHyper::vague(k);
            let prior = hyper.draw(k, rng);
            let components: Vec<Categorical> =
                (0..n_cats).map(|_| prior.draw(rng)).collect();
            let xs: Vec<u8> = (0..n_rows)
                .map::<u8, _>(|i| components[i % n_cats].draw::<R>(rng))
                .collect();
            let data = SparseContainer::from(xs);
            let col = Column::new(id, data, prior, hyper);
            ColModel::Categorical(col)
        }
        ColType::Labeler { n_labels, .. } => {
            let prior = LabelerPrior::standard(n_labels);
            let components: Vec<Labeler> =
                (0..n_cats).map(|_| prior.draw(rng)).collect();
            let xs: Vec<Label> = components[0].sample(n_rows, rng);
            let data = SparseContainer::from(xs);
            let col = Column::new(id, data, prior, ());
            ColModel::Labeler(col)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StateUpdateConfig;

    #[test]
    fn test_dimensions() {
        let state = Builder::new()
            .column_configs(
                10,
                ColType::Continuous {
                    hyper: None,
                    prior: None,
                },
            )
            .n_rows(50)
            .build()
            .expect("Failed to build state");

        assert_eq!(state.n_rows(), 50);
        assert_eq!(state.n_cols(), 10);
    }

    #[test]
    fn built_state_should_update() {
        let mut rng = Xoshiro256Plus::from_entropy();
        let mut state = Builder::new()
            .column_configs(
                10,
                ColType::Continuous {
                    hyper: None,
                    prior: None,
                },
            )
            .n_rows(50)
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
            Builder::new()
                .column_configs(
                    10,
                    ColType::Continuous {
                        hyper: None,
                        prior: None,
                    },
                )
                .n_rows(50)
                .seed_from_rng(&mut rng)
                .build()
                .expect("Failed to build state")
        };

        let state_2 = {
            let mut rng = Xoshiro256Plus::seed_from_u64(122445);
            Builder::new()
                .column_configs(
                    10,
                    ColType::Continuous {
                        hyper: None,
                        prior: None,
                    },
                )
                .n_rows(50)
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
    fn n_rows_overriden_by_features() {
        let n_cols = 5;
        let col_models = {
            let state = Builder::new()
                .column_configs(
                    n_cols,
                    ColType::Continuous {
                        hyper: None,
                        prior: None,
                    },
                )
                .n_rows(11)
                .build()
                .unwrap();

            (0..n_cols)
                .map(|ix| state.feature(ix))
                .cloned()
                .collect::<Vec<_>>()
        };

        let state = Builder::new()
            .features(col_models)
            .n_rows(101)
            .build()
            .unwrap();

        assert_eq!(state.n_rows(), 11);
    }
}
