use rv::dist::{Gamma, InvGamma};
use rv::traits::Rv;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Provides distribution options for CRP row/column prior
#[derive(Serialize, Deserialize, PartialOrd, PartialEq, Debug, Clone)]
pub enum CrpPrior {
    /// Inverse Gamma for long tails (long tails may cause problems with the
    /// slice algorithm).
    InvGamma(InvGamma),
    /// Gamma for short tails
    Gamma(Gamma),
}

macro_rules! re_to_crp_prior {
    ($s: ident, $re: ident, $variant: ident) => {{
        $re.captures($s)
            .ok_or_else(|| format!("could not parse {}", $s))
            .and_then(|m| {
                let shape = f64::from_str(m.get(1).unwrap().as_str()).map_err(
                    |_| String::from("Could not parse shape as f64"),
                )?;
                let rate = f64::from_str(m.get(2).unwrap().as_str())
                    .map_err(|_| String::from("Could not parse rate as f64"))?;
                let inner = $variant::new(shape, rate).map_err(|_| {
                    format!(
                        "Invalid shape ({}) and rate ({}) paramters",
                        shape, rate
                    )
                })?;
                Ok(CrpPrior::$variant(inner))
            })
    }};
}

impl FromStr for CrpPrior {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let re = regex::Regex::new(r"\((\d+\.\d+),\s*(\d+\.\d+)\)").unwrap();
        if s.starts_with("Gamma(") {
            re_to_crp_prior!(s, re, Gamma)
        } else if s.starts_with("InvGamma(") {
            re_to_crp_prior!(s, re, InvGamma)
        } else {
            Err(format!("could not parse \"{}\" into Gamma or InvGamma", s))
        }
    }
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
    fn draw_gamma() {
        let mut rng = rand::thread_rng();
        let crp_prior: CrpPrior = Gamma::new(1.0, 1.0).unwrap().into();
        let x: f64 = crp_prior.draw(&mut rng);
        assert!(x > 0.0);
    }

    #[test]
    fn parse_str_to_gamma() {
        let prior = CrpPrior::from_str("Gamma(1.2, 3.4)").unwrap();
        if let CrpPrior::Gamma(g) = prior {
            assert_eq!(g, Gamma::new(1.2, 3.4).unwrap());
        } else {
            panic!("Failed to parse");
        }
    }

    #[test]
    fn parse_str_to_inv_gamma() {
        let prior = CrpPrior::from_str("InvGamma(1.2,3.4)").unwrap();
        if let CrpPrior::InvGamma(g) = prior {
            assert_eq!(g, InvGamma::new(1.2, 3.4).unwrap());
        } else {
            panic!("Failed to parse");
        }
    }

    #[test]
    fn parse_str_to_bad_name_fails() {
        assert!(CrpPrior::from_str("TimGamma(1.2,3.4)").is_err());
    }

    #[test]
    fn parse_str_to_bad_params_fails() {
        assert!(CrpPrior::from_str("Gamma(0.0, 3.4)").is_err());
    }
}
