mod dataless;
pub mod error;
mod traits;
pub mod utils;
mod validation;

pub use dataless::DatalessOracle;
pub use traits::OracleT;

use std::path::Path;

use braid_cc::state::State;
use braid_codebook::Codebook;
use braid_data::{DataStore, Datum, SummaryStatistics};
use braid_metadata::latest::Metadata;
use serde::{Deserialize, Serialize};

use crate::{Engine, HasData, HasStates};

/// Mutual Information Type
#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
)]
#[serde(rename_all = "snake_case")]
pub enum MiType {
    /// The Standard, un-normalized variant
    UnNormed,
    /// Normalized by the max MI, which is `min(H(A), H(B))`
    Normed,
    /// Linfoot information Quantity. Derived by computing the mutual
    /// information between the two components of a bivariate Normal with
    /// covariance rho, and solving for rho.
    Linfoot,
    /// Variation of Information. A version of mutual information that
    /// satisfies the triangle inequality.
    Voi,
    /// Jaccard distance between X an Y. Jaccard(X, Y) is in [0, 1].
    Jaccard,
    /// Information Quality Ratio:  the amount of information of a variable
    /// based on another variable against total uncertainty.
    Iqr,
    /// Mutual Information normed the with square root of the product of the
    /// components entropies. Akin to the Pearson correlation coefficient.
    Pearson,
}

/// Holds the components required to compute mutual information
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, PartialOrd)]
pub struct MiComponents {
    /// The entropy of column a, H(A)
    pub h_a: f64,
    /// The entropy of column b, H(B)
    pub h_b: f64,
    /// The joint entropy of columns a and b, H(A, B)
    pub h_ab: f64,
}

impl MiComponents {
    #[inline]
    pub fn compute(&self, mi_type: MiType) -> f64 {
        let mi = (self.h_a + self.h_b - self.h_ab).max(0.0);

        match mi_type {
            MiType::UnNormed => mi,
            MiType::Normed => mi / self.h_a.min(self.h_b),
            MiType::Voi => 2.0_f64.mul_add(-mi, self.h_a + self.h_b),
            MiType::Pearson => mi / (self.h_a * self.h_b).sqrt(),
            MiType::Iqr => mi / self.h_ab,
            MiType::Jaccard => 1.0 - mi / self.h_ab,
            MiType::Linfoot => (1.0 - (-2.0 * mi).exp()).sqrt(),
        }
    }
}

/// The type of uncertainty to use for `Oracle.impute`
#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
)]
#[serde(rename_all = "snake_case")]
pub enum ImputeUncertaintyType {
    /// Given a set of distributions Θ = {Θ<sub>1</sub>, ..., Θ<sub>n</sub>},
    /// return the mean of KL(Θ<sub>i</sub> || Θ<sub>i</sub>)
    PairwiseKl,
    /// The Jensen-Shannon divergence in nats divided by ln(n), where n is the
    /// number of distributions
    JsDivergence,
}

/// The type of uncertainty to use for `Oracle.predict`
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum PredictUncertaintyType {
    /// The Jensen-Shannon divergence in nats divided by ln(n), where n is the
    /// number of distributions
    JsDivergence,
}

//
#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
)]
pub enum ConditionalEntropyType {
    /// Normal conditional entropy
    UnNormed,
    /// IP(X; Y), The proportion of information in X accounted for by Y
    InfoProp,
}

/// Oracle answers questions
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "Metadata", into = "Metadata")]
pub struct Oracle {
    /// Vector of states
    pub states: Vec<State>,
    /// Metadata for the rows and columns
    pub codebook: Codebook,
    pub data: DataStore,
}

impl Oracle {
    // TODO: just make this a From trait impl
    /// Convert an `Engine` into an `Oracle`
    pub fn from_engine(engine: Engine) -> Self {
        let data = {
            let data_map = engine.states.get(0).unwrap().clone_data();
            DataStore::new(data_map)
        };

        // TODO: would be nice to have a draining iterator on the states
        // rather than cloning them
        let states: Vec<State> = engine
            .states
            .iter()
            .map(|state| {
                let mut state_clone = state.clone();
                state_clone.drop_data();
                state_clone
            })
            .collect();

        Self {
            data,
            states,
            codebook: engine.codebook,
        }
    }

    /// Load an Oracle from a .braid file
    pub fn load<P: AsRef<Path>>(
        path: P,
    ) -> Result<Self, braid_metadata::Error> {
        use std::convert::TryInto;

        let metadata = braid_metadata::load_metadata(path, None)?;
        metadata
            .try_into()
            .map_err(|err| braid_metadata::Error::Other(format!("{}", err)))
    }
}

impl HasStates for Oracle {
    #[inline]
    fn states(&self) -> &Vec<State> {
        &self.states
    }

    #[inline]
    fn states_mut(&mut self) -> &mut Vec<State> {
        &mut self.states
    }
}

impl HasData for Oracle {
    #[inline]
    fn summarize_feature(&self, ix: usize) -> SummaryStatistics {
        self.data.0[&ix].summarize()
    }

    #[inline]
    fn cell(&self, row_ix: usize, col_ix: usize) -> Datum {
        self.data.get(row_ix, col_ix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Given;
    use crate::{Oracle, OracleT};
    use approx::*;
    use braid_cc::feature::Feature;
    use braid_stats::MixtureType;
    use rand::Rng;
    use rv::dist::{Categorical, Gaussian, Mixture};
    use rv::traits::Rv;
    use std::collections::BTreeMap;
    use std::path::Path;

    fn oracle_from_yaml<P: AsRef<Path>>(filenames: Vec<P>) -> Oracle {
        let states = utils::load_states(filenames);
        let data = DataStore::new(states[0].clone_data());
        Oracle {
            states,
            codebook: Codebook::default(),
            data,
        }
    }

    const TOL: f64 = 1E-8;
    fn get_single_continuous_oracle_from_yaml() -> Oracle {
        let filenames = vec!["resources/test/single-continuous.yaml"];
        oracle_from_yaml(filenames)
    }

    fn get_duplicate_single_continuous_oracle_from_yaml() -> Oracle {
        let filenames = vec![
            "resources/test/single-continuous.yaml",
            "resources/test/single-continuous.yaml",
        ];
        oracle_from_yaml(filenames)
    }

    fn get_oracle_from_yaml() -> Oracle {
        let filenames = vec![
            "resources/test/small/small-state-1.yaml",
            "resources/test/small/small-state-2.yaml",
            "resources/test/small/small-state-3.yaml",
        ];

        oracle_from_yaml(filenames)
    }

    fn get_entropy_oracle_from_yaml() -> Oracle {
        let filenames = vec![
            "resources/test/entropy/entropy-state-1.yaml",
            "resources/test/entropy/entropy-state-2.yaml",
        ];
        oracle_from_yaml(filenames)
    }

    #[test]
    fn single_continuous_column_logp() {
        let oracle = get_single_continuous_oracle_from_yaml();

        let vals = vec![vec![Datum::Continuous(-1.0)]];
        let logp = oracle.logp(&[0], &vals, &Given::Nothing, None).unwrap()[0];

        assert_relative_eq!(logp, -2.794_105_164_665_195_3, epsilon = TOL);
    }

    #[test]
    fn single_continuous_column_logp_state_0() {
        let oracle = get_oracle_from_yaml();

        let vals = vec![vec![Datum::Continuous(-1.0)]];
        let logp = oracle
            .logp(&[0], &vals, &Given::Nothing, Some(&[0]))
            .unwrap()[0];

        assert_relative_eq!(logp, -1.223_532_985_437_053, epsilon = TOL);
    }

    #[test]
    fn single_continuous_column_logp_duplicated_states() {
        let oracle = get_duplicate_single_continuous_oracle_from_yaml();

        let vals = vec![vec![Datum::Continuous(-1.0)]];
        let logp = oracle.logp(&[0], &vals, &Given::Nothing, None).unwrap()[0];

        assert_relative_eq!(logp, -2.794_105_164_665_195_3, epsilon = TOL);
    }

    #[test]
    #[ignore]
    fn mutual_information_smoke() {
        let oracle = get_oracle_from_yaml();

        let mi_01 = oracle.mi(0, 1, 10_000, MiType::Normed).unwrap();
        let mi_02 = oracle.mi(0, 2, 10_000, MiType::Normed).unwrap();
        let mi_12 = oracle.mi(1, 2, 10_000, MiType::Normed).unwrap();

        assert!(mi_01 > 0.0);
        assert!(mi_02 > 0.0);
        assert!(mi_12 > 0.0);
    }

    #[test]
    fn surpisal_value_1() {
        let oracle = get_oracle_from_yaml();
        let s = oracle
            .surprisal(&Datum::Continuous(1.2), 3, 1, None)
            .unwrap()
            .unwrap();
        assert_relative_eq!(s, 1.773_919_580_331_675_8, epsilon = 10E-7);
    }

    #[test]
    fn surpisal_value_2() {
        let oracle = get_oracle_from_yaml();
        let s = oracle
            .surprisal(&Datum::Continuous(0.1), 1, 0, None)
            .unwrap()
            .unwrap();
        assert_relative_eq!(s, 0.620_843_253_052_312_7, epsilon = 10E-7);
    }

    #[test]
    fn kl_impute_uncertainty_smoke() {
        let oracle = get_oracle_from_yaml();
        let u =
            oracle.impute_uncertainty(0, 1, ImputeUncertaintyType::PairwiseKl);
        assert!(u > 0.0);
    }

    #[test]
    fn js_impute_uncertainty_smoke() {
        let oracle = get_oracle_from_yaml();
        let u = oracle.impute_uncertainty(
            0,
            1,
            ImputeUncertaintyType::JsDivergence,
        );
        assert!(u > 0.0);
    }

    #[test]
    fn predict_uncertainty_smoke_no_given() {
        let oracle = get_oracle_from_yaml();
        let u = oracle.predict_uncertainty(0, &Given::Nothing, None);
        assert!(u > 0.0);
    }

    #[test]
    fn predict_uncertainty_smoke_with_given() {
        let oracle = get_oracle_from_yaml();
        let given = Given::Conditions(vec![(1, Datum::Continuous(2.5))]);
        let u = oracle.predict_uncertainty(0, &given, None);
        assert!(u > 0.0);
    }

    #[test]
    fn mixture_and_oracle_logp_equivalence_categorical() {
        let oracle = get_entropy_oracle_from_yaml();

        let mm: Mixture<Categorical> = {
            let mixtures: Vec<_> = oracle
                .states
                .iter()
                .map(|s| s.feature_as_mixture(2))
                .collect();
            match MixtureType::combine(mixtures) {
                MixtureType::Categorical(mm) => mm,
                _ => panic!("bad mixture type"),
            }
        };

        for x in 0..4 {
            let y = Datum::Categorical(x as u8);
            let logp_mm = mm.ln_f(&(x as usize));
            let logp_or = oracle
                .logp(&[2], &[vec![y]], &Given::Nothing, None)
                .unwrap()[0];
            assert_relative_eq!(logp_or, logp_mm, epsilon = 1E-12);
        }
    }

    #[test]
    fn mixture_and_oracle_logp_equivalence_gaussian() {
        let oracle = get_oracle_from_yaml();
        let mut rng = rand::thread_rng();

        let mm: Mixture<Gaussian> = {
            let mixtures: Vec<_> = oracle
                .states
                .iter()
                .map(|s| s.feature_as_mixture(1))
                .collect();
            match MixtureType::combine(mixtures) {
                MixtureType::Gaussian(mm) => mm,
                _ => panic!("bad mixture type"),
            }
        };

        for _ in 0..1000 {
            let x: f64 = {
                let u: f64 = rng.gen();
                u * 3.0
            };
            let y = Datum::Continuous(x);
            let logp_mm = mm.ln_f(&x);
            let logp_or = oracle
                .logp(&[1], &[vec![y]], &Given::Nothing, None)
                .unwrap()[0];
            assert_relative_eq!(logp_or, logp_mm, epsilon = 1E-12);
        }
    }

    #[test]
    fn recreate_doctest_mi_failure() {
        use crate::examples::animals::Column;
        use crate::examples::Example;
        use crate::MiType;

        let oracle = Example::Animals.oracle().unwrap();

        let mi_flippers = oracle
            .mi(
                Column::Swims.into(),
                Column::Flippers.into(),
                1000,
                MiType::Iqr,
            )
            .unwrap();

        let mi_fast = oracle
            .mi(Column::Swims.into(), Column::Fast.into(), 1000, MiType::Iqr)
            .unwrap();

        assert!(mi_flippers > mi_fast);
    }

    #[test]
    fn mixture_and_oracle_logp_equivalence_animals_single_state() {
        use crate::examples::Example;

        let oracle = Example::Animals.oracle().unwrap();

        for (ix, state) in oracle.states.iter().enumerate() {
            for col_ix in 0..oracle.n_cols() {
                let mm = match state.feature_as_mixture(col_ix) {
                    MixtureType::Categorical(mm) => mm,
                    _ => panic!("Invalid MixtureType"),
                };
                for val in 0..2 {
                    let logp_mm = mm.ln_f(&(val as usize));
                    let datum = Datum::Categorical(val as u8);
                    let logp_or = oracle
                        .logp(
                            &[col_ix],
                            &[vec![datum]],
                            &Given::Nothing,
                            Some(&[ix]),
                        )
                        .unwrap()[0];
                    assert_relative_eq!(logp_or, logp_mm, epsilon = 1E-12);
                }
            }
        }
    }

    #[test]
    fn pw_and_conditional_entropy_equivalence_animals() {
        use crate::examples::Example;
        let oracle = Example::Animals.oracle().unwrap();

        let n_cols = oracle.n_cols();
        let mut col_pairs: Vec<(usize, usize)> = Vec::new();
        let mut entropies: Vec<f64> = Vec::new();
        for col_a in 0..n_cols {
            for col_b in 0..n_cols {
                if col_a != col_b {
                    col_pairs.push((col_a, col_b));
                    let ce = oracle
                        .conditional_entropy(col_a, &[col_b], 1000)
                        .unwrap();
                    entropies.push(ce);
                }
            }
        }

        let entropies_pw = oracle
            .conditional_entropy_pw(
                &col_pairs,
                1000,
                ConditionalEntropyType::UnNormed,
            )
            .unwrap();

        entropies
            .iter()
            .zip(entropies_pw.iter())
            .enumerate()
            .for_each(|(ix, (h, h_pw))| {
                println!("{ix}");
                assert_relative_eq!(h, h_pw, epsilon = 1E-12);
            })
    }

    #[test]
    fn pw_and_info_prop_equivalence_animals() {
        use crate::examples::Example;
        let oracle = Example::Animals.oracle().unwrap();

        let n_cols = oracle.n_cols();
        let mut col_pairs: Vec<(usize, usize)> = Vec::new();
        let mut entropies: Vec<f64> = Vec::new();
        for col_a in 0..n_cols {
            for col_b in 0..n_cols {
                if col_a != col_b {
                    col_pairs.push((col_a, col_b));
                    let ce =
                        oracle.info_prop(&[col_a], &[col_b], 1000).unwrap();
                    entropies.push(ce);
                }
            }
        }

        let entropies_pw = oracle
            .conditional_entropy_pw(
                &col_pairs,
                1000,
                ConditionalEntropyType::InfoProp,
            )
            .unwrap();

        entropies
            .iter()
            .zip(entropies_pw.iter())
            .for_each(|(h, h_pw)| {
                assert_relative_eq!(h, h_pw, epsilon = 1E-12);
            })
    }

    #[test]
    fn mi_pw_and_normal_equivalence() {
        use crate::examples::Example;
        let oracle = Example::Animals.oracle().unwrap();

        let n_cols = oracle.n_cols();
        let mut col_pairs: Vec<(usize, usize)> = Vec::new();
        let mut mis: Vec<f64> = Vec::new();
        for col_a in 0..n_cols {
            for col_b in 0..n_cols {
                if col_a != col_b {
                    col_pairs.push((col_a, col_b));
                    let mi = oracle
                        .mi(col_a, col_b, 1000, MiType::UnNormed)
                        .unwrap();
                    mis.push(mi);
                }
            }
        }

        let mis_pw = oracle.mi_pw(&col_pairs, 1000, MiType::UnNormed).unwrap();

        mis.iter().zip(mis_pw.iter()).for_each(|(mi, mi_pw)| {
            assert_relative_eq!(mi, mi_pw, epsilon = 1E-12);
        })
    }

    // pre v0.20.0 simulate code ripped straight from simulate_unchecked
    fn old_simulate(
        oracle: &Oracle,
        col_ixs: &[usize],
        given: &Given,
        n: usize,
        states_ixs_opt: Option<Vec<usize>>,
        mut rng: &mut impl Rng,
    ) -> Vec<Vec<Datum>> {
        let state_ixs: Vec<usize> = match states_ixs_opt {
            Some(state_ixs) => state_ixs,
            None => (0..oracle.n_states()).collect(),
        };

        let states: Vec<&State> =
            state_ixs.iter().map(|&ix| &oracle.states()[ix]).collect();
        let state_ixer = Categorical::uniform(state_ixs.len());
        let weights = utils::given_weights(&states, col_ixs, given);

        (0..n)
            .map(|_| {
                // choose a random state
                let draw_ix: usize = state_ixer.draw(&mut rng);
                let state = states[draw_ix];

                // for each view
                //   choose a random component from the weights
                let mut cpnt_ixs: BTreeMap<usize, usize> = BTreeMap::new();
                for (view_ix, view_weights) in &weights[draw_ix] {
                    // TODO: use Categorical::new_unchecked when rv 0.9.3 drops.
                    // from_ln_weights checks that the input logsumexp's to 0
                    let component_ixer =
                        Categorical::from_ln_weights(view_weights.clone())
                            .unwrap();
                    let k = component_ixer.draw(&mut rng);
                    cpnt_ixs.insert(*view_ix, k);
                }

                // for eacch column
                //   draw from appropriate component from that view
                let mut xs: Vec<Datum> = Vec::with_capacity(col_ixs.len());
                col_ixs.iter().for_each(|col_ix| {
                    let view_ix = state.asgn.asgn[*col_ix];
                    let k = cpnt_ixs[&view_ix];
                    let x = state.views[view_ix].ftrs[col_ix].draw(k, &mut rng);
                    xs.push(x);
                });
                xs
            })
            .collect()
    }

    fn simulate_equivalence(
        col_ixs: &[usize],
        given: &Given,
        state_ixs_opt: Option<Vec<usize>>,
    ) {
        use crate::examples::Example;
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256Plus;

        let n: usize = 100;
        let oracle = Example::Satellites.oracle().unwrap();

        let xs_simulator: Vec<Vec<Datum>> = {
            let mut rng = Xoshiro256Plus::seed_from_u64(1337);
            old_simulate(
                &oracle,
                col_ixs,
                given,
                n,
                state_ixs_opt.clone(),
                &mut rng,
            )
        };

        let xs_standard: Vec<Vec<Datum>> = {
            let mut rng = Xoshiro256Plus::seed_from_u64(1337);
            oracle
                .simulate(col_ixs, given, n, state_ixs_opt, &mut rng)
                .unwrap()
        };

        for (x, y) in xs_simulator.iter().zip(xs_standard.iter()) {
            assert_eq!(x, y)
        }
    }

    #[test]
    fn seeded_simulate_and_simulator_agree() {
        let col_ixs = [0_usize, 5, 6];
        let given = Given::Nothing;
        simulate_equivalence(&col_ixs, &given, None);
    }

    #[test]
    fn seeded_simulate_and_simulator_agree_state_ixs() {
        let col_ixs = [0_usize, 5, 6];
        let given = Given::Nothing;
        simulate_equivalence(&col_ixs, &given, Some(vec![3, 6]));
    }

    #[test]
    fn seeded_simulate_and_simulator_agree_given() {
        let col_ixs = [0_usize, 5, 6];
        let given = Given::Conditions(vec![(8, Datum::Continuous(100.0))]);
        simulate_equivalence(&col_ixs, &given, None);
    }

    #[test]
    fn seeded_simulate_and_simulator_agree_given_state_ixs() {
        let col_ixs = [0_usize, 5, 6];
        let given = Given::Conditions(vec![(8, Datum::Continuous(100.0))]);
        simulate_equivalence(&col_ixs, &given, Some(vec![3, 6]));
    }
}
