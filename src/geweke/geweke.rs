extern crate rand;

use rand::Rng;

pub trait GewkeReady {
    fn from_prior(rng: &mut Rng) -> Self;
    fn resample_data(&mut self, rng: &mut Rng);
    fn resample_params(&mut self, rng: &mut Rng);
}
