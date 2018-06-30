extern crate rand;

use self::rand::Rng;
use cc::codebook::ColMetadata;
use cc::{
    Assignment, AssignmentBuilder, ColModel, Column, DataContainer, State,
    View, ViewBuilder,
};
use dist::prior::csd::CatSymDirichlet;
use dist::prior::nig::{NigHyper, NormalInverseGamma};
use dist::prior::Prior;
use dist::traits::RandomVariate;
use dist::{Categorical, Gaussian};
use std::io;

pub struct StateBuilder {
    pub nrows: Option<usize>,
    pub nviews: Option<usize>,
    pub ncats: Option<usize>,
    pub col_configs: Vec<ColMetadata>,
}

/// Builds a state with a given complexity for benchmarking and testing purposes
impl StateBuilder {
    pub fn new() -> Self {
        StateBuilder {
            nrows: None,
            nviews: None,
            ncats: None,
            col_configs: vec![],
        }
    }

    pub fn with_rows(mut self, nrows: usize) -> Self {
        self.nrows = Some(nrows);
        self
    }

    pub fn with_views(mut self, nviews: usize) -> Self {
        self.nviews = Some(nviews);
        self
    }

    pub fn with_cats(mut self, ncats: usize) -> Self {
        self.ncats = Some(ncats);
        self
    }

    pub fn add_column(mut self, col_config: ColMetadata) -> Self {
        self.col_configs.push(col_config);
        self
    }

    pub fn add_columns(mut self, n: usize, col_config: ColMetadata) -> Self {
        self.col_configs.append(&mut vec![col_config; n]);
        self
    }

    pub fn build(&self, mut rng: &mut impl Rng) -> io::Result<State> {
        let nrows = self.nrows.unwrap_or(100);
        let nviews = self.nviews.unwrap_or(1);
        let ncats = self.ncats.unwrap_or(1);
        let mut col_configs = if !self.col_configs.is_empty() {
            self.col_configs.clone()
        } else {
            let err =
                io::Error::new(io::ErrorKind::InvalidData, "No column configs");
            return Err(err);
        };
        let mut ftrs: Vec<ColModel> = col_configs
            .drain(..)
            .enumerate()
            .map(|(id, col_config)| {
                gen_feature(id, col_config, nrows, ncats, &mut rng)
            })
            .collect();
        // println!("N: {}", ftrs.len());

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
                    .expect("Failed to create asgn")
                    .build(&mut rng);
                ViewBuilder::from_assignment(asgn)
                    .with_features(ftrs_view)
                    .build(&mut rng)
            })
            .collect();

        assert_eq!(ftrs.len(), 0);

        let asgn = AssignmentBuilder::from_vec(col_asgn).build(&mut rng);
        Ok(State::new(views, asgn, 1.0))
    }
}

fn gen_feature(
    id: usize,
    col_config: ColMetadata,
    nrows: usize,
    ncats: usize,
    mut rng: &mut impl Rng,
) -> ColModel {
    match col_config {
        ColMetadata::Continuous { .. } => {
            let hyper = NigHyper::default();
            let prior = NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0, hyper);
            let components: Vec<Gaussian> =
                (0..ncats).map(|_| prior.prior_draw(&mut rng)).collect();
            let xs: Vec<f64> = (0..nrows)
                .map(|i| components[i % ncats].draw(&mut rng))
                .collect();
            let data = DataContainer::new(xs);
            let col = Column::new(id, data, prior);
            ColModel::Continuous(col)
        }
        ColMetadata::Categorical { k, .. } => {
            let prior = CatSymDirichlet::vague(k, &mut rng);
            let components: Vec<Categorical<u8>> =
                (0..ncats).map(|_| prior.prior_draw(&mut rng)).collect();
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

    #[test]
    fn test_dimensions() {
        let mut rng = rand::thread_rng();
        let state = StateBuilder::new()
            .add_columns(10, ColMetadata::Continuous { hyper: None })
            .with_rows(50)
            .build(&mut rng)
            .expect("Failed to build state");

        assert_eq!(state.nrows(), 50);
        assert_eq!(state.ncols(), 10);
    }

    #[test]
    fn built_state_should_update() {
        let mut rng = rand::thread_rng();
        let mut state = StateBuilder::new()
            .add_columns(10, ColMetadata::Continuous { hyper: None })
            .with_rows(50)
            .build(&mut rng)
            .expect("Failed to build state");
        state.update(5, None, None, None, &mut rng);
    }
}
