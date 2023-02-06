use crate::rv::traits::Rv;
use rand::Rng;

struct FlatImproper;

impl<X> Rv<X> for FlatImproper {
    fn f(&self, _x: &X) -> f64 {
        1.0
    }

    fn ln_f(&self, _x: &X) -> f64 {
        0.0
    }

    fn draw<R: Rng>(&self, _rng: &mut R) -> X {
        panic!("Cannot draw from flat prior")
    }
}
