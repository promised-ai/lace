use crate::rv::traits::HasDensity;

struct FlatImproper;

impl<X> HasDensity<X> for FlatImproper {
    fn f(&self, _x: &X) -> f64 {
        1.0
    }

    fn ln_f(&self, _x: &X) -> f64 {
        0.0
    }
}
