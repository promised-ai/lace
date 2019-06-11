use braid_codebook::codebook::ColType;
use braid_stats::prior::{Csd, Ng, NigHyper};
use rand::{FromEntropy, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use rv::{
    dist::{Categorical, Gamma, Gaussian},
    traits::*,
};

use crate::cc::{
    AssignmentBuilder, ColModel, Column, DataContainer, State, ViewBuilder,
};

use crate::result;

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

/// Builds a state with a given complexity for benchmarking and testing purposes
impl StateBuilder {
    pub fn new() -> Self {
        StateBuilder {
            nrows: None,
            nviews: None,
            ncats: None,
            col_configs: None,
            ftrs: None,
            seed: None,
        }
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
    pub fn from_rng<R: rand::Rng>(mut self, rng: &mut R) -> Self {
        self.seed = Some(rng.next_u64());
        self
    }

    /// With an RNG seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build the `State`
    pub fn build(self) -> result::Result<State> {
        let mut rng = match self.seed {
            Some(seed) => Xoshiro256Plus::seed_from_u64(seed),
            None => Xoshiro256Plus::from_entropy(),
        };
        let nrows = self.nrows.unwrap_or(100);
        let nviews = self.nviews.unwrap_or(1);
        let ncats = self.ncats.unwrap_or(1);

        if self.col_configs.is_some() && self.ftrs.is_some() {
            let err = result::Error::new(
                result::ErrorKind::InvalidConfigError,
                String::from("Only one of col_configs or ftrs may be present"),
            );
            return Err(err);
        } else if self.col_configs.is_none() && self.ftrs.is_none() {
            let err = result::Error::new(
                result::ErrorKind::InvalidConfigError,
                String::from("No column configs or features supplied"),
            );
            return Err(err);
        }

        let mut ftrs = if self.col_configs.is_some() {
            self.col_configs
                .clone()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(id, col_config)| {
                    gen_feature(id, col_config.clone(), nrows, ncats, &mut rng)
                })
                .collect()
        } else {
            self.ftrs.clone().unwrap()
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
                // println!("N: {}, to drain: {}", ftrs_left, to_drain);
                col_asgn.append(&mut vec![view_ix; to_drain]);
                col_counts.push(to_drain);
                let ftrs_view = ftrs.drain(0..to_drain).map(|f| f).collect();
                let asgn = AssignmentBuilder::new(nrows)
                    .with_ncats(ncats)
                    .unwrap()
                    .from_rng(&mut rng)
                    .build()
                    .unwrap();
                ViewBuilder::from_assignment(asgn)
                    .with_features(ftrs_view)
                    .from_rng(&mut rng)
                    .build()
            })
            .collect();

        assert_eq!(ftrs.len(), 0);

        let asgn = AssignmentBuilder::from_vec(col_asgn)
            .from_rng(&mut rng)
            .build()
            .unwrap();
        Ok(State::new(views, asgn, Gamma::new(1.0, 1.0).unwrap()))
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
            let hyper = NigHyper::default();
            let prior = Ng::new(0.0, 1.0, 4.0, 4.0, hyper);
            let g = Gaussian::standard();
            let xs: Vec<f64> = g.sample(nrows, &mut rng);
            let data = DataContainer::new(xs);
            let col = Column::new(id, data, prior);
            ColModel::Continuous(col)
        }
        ColType::Categorical { k, .. } => {
            let prior = Csd::vague(k, &mut rng);
            let components: Vec<Categorical> =
                (0..ncats).map(|_| prior.draw(&mut rng)).collect();
            let xs: Vec<u8> = (0..nrows)
                .map(|i| components[i % ncats].draw(&mut rng))
                .collect();
            let data = DataContainer::new(xs);
            let col = Column::new(id, data, prior);
            ColModel::Categorical(col)
        }
        _ => panic!("Unsupported feature type"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cc::config::StateUpdateConfig;

    #[test]
    fn test_dimensions() {
        let state = StateBuilder::new()
            .add_column_configs(10, ColType::Continuous { hyper: None })
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
            .add_column_configs(10, ColType::Continuous { hyper: None })
            .with_rows(50)
            .from_rng(&mut rng)
            .build()
            .expect("Failed to build state");

        let config = StateUpdateConfig::new().with_iters(5);
        state.update(config, &mut rng);
    }
}
