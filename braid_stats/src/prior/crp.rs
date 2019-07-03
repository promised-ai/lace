use rv::dist::{Gamma, InvGamma};
use rv::traits::Rv;

/// Provides distribution options for CRP row/column prior
pub enum CrpPrior {
    /// Inverse Gamma for long tails (long tails may cause problems with the
    /// slice algorithm).
    InvGamma(InvGamma),
    /// Gamma for short tails
    Gamma(Gamma),
}

impl Rv<f64> for CrpPrior {
    fn ln_f(&self, x: &f64) -> f64 {
        match self {
            CrpPrior::Gamma(inner) => inner.ln_f(x),
            CrpPrior::InvGamma(inner) => inner.ln_f(x),
        }
    }

    fn draw<R: rand::Rng>(&self, mut rng: &mut R) -> f64 {
        match self {
            CrpPrior::Gamma(inner) => inner.draw(&mut rng),
            CrpPrior::InvGamma(inner) => inner.draw(&mut rng),
        }
    }
}

impl From<Gamma> for CrpPrior {
    fn from(gamma: Gamma) -> CrpPrior {
        CrpPrior::Gamma(gamma)
    }
}

impl From<InvGamma> for CrpPrior {
    fn from(inv_gamma: InvGamma) -> CrpPrior {
        CrpPrior::InvGamma(inv_gamma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draw_gamma() {
        let mut rng = rand::thread_rng();
        let crp_prior: CrpPrior = Gamma::new(1.0, 1.0).unwrap().into();
        let x: f64 = crp_prior.draw(&mut rng);
        assert!(x > 0.0);
    }
}
