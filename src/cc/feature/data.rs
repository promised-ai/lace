use crate::cc::DataContainer;
use crate::cc::SummaryStatistics;
use braid_stats::labeler::Label;
use braid_stats::Datum;
use serde::{Deserialize, Serialize};

/// Used when pulling data from features for saving
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum FeatureData {
    /// Univariate continuous data
    Continuous(DataContainer<f64>),
    /// Categorical data
    Categorical(DataContainer<u8>),
    /// Categorical data
    Labeler(DataContainer<Label>),
}

impl FeatureData {
    /// Get the datum at [row_ix, col_ix] as a `Datum`
    pub fn get(&self, ix: usize) -> Datum {
        // TODO: DataContainer index get (xs[i]) should return an option
        match self {
            FeatureData::Continuous(xs) => {
                if xs.present[ix] {
                    Datum::Continuous(xs[ix])
                } else {
                    Datum::Missing
                }
            }
            FeatureData::Categorical(xs) => {
                if xs.present[ix] {
                    Datum::Categorical(xs[ix])
                } else {
                    Datum::Missing
                }
            }
            FeatureData::Labeler(xs) => {
                if xs.present[ix] {
                    Datum::Label(xs[ix])
                } else {
                    Datum::Missing
                }
            }
        }
    }

    /// Get the summary statistic for a column
    pub fn summarize(&self) -> SummaryStatistics {
        match self {
            FeatureData::Continuous(ref container) => {
                summarize_continuous(&container)
            }
            FeatureData::Categorical(ref container) => {
                summarize_categorical(&container)
            }
            FeatureData::Labeler(..) => {
                unimplemented!("cannot summarize labeler column")
            }
        }
    }
}

pub fn summarize_continuous(
    container: &DataContainer<f64>,
) -> SummaryStatistics {
    use braid_utils::{mean, var};
    let mut xs: Vec<f64> =
        container.zip().filter(|xp| *xp.1).map(|xp| *xp.0).collect();

    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = xs.len();
    SummaryStatistics::Continuous {
        min: xs[0],
        max: xs[n - 1],
        mean: mean(&xs),
        variance: var(&xs),
        median: if n % 2 == 0 {
            (xs[n / 2] + xs[n / 2 - 1]) / 2.0
        } else {
            xs[n / 2]
        },
    }
}

pub fn summarize_categorical(
    container: &DataContainer<u8>,
) -> SummaryStatistics {
    use braid_utils::{bincount, minmax};
    let xs: Vec<u8> =
        container.zip().filter(|xp| *xp.1).map(|xp| *xp.0).collect();

    let (min, max) = minmax(&xs);
    let counts = bincount(&xs, (max + 1) as usize);
    let max_ct = counts
        .iter()
        .fold(0_usize, |acc, &ct| if ct > acc { ct } else { acc });
    let mode = counts
        .iter()
        .enumerate()
        .filter(|(_, &ct)| ct == max_ct)
        .map(|(ix, _)| ix as u8)
        .collect();

    SummaryStatistics::Categorical { min, max, mode }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cc::DataContainer;
    use approx::*;

    fn get_continuous() -> FeatureData {
        let dc1: DataContainer<f64> = DataContainer {
            data: vec![4.0, 3.0, 2.0, 1.0, 0.0],
            present: vec![true, false, true, true, true],
        };

        FeatureData::Continuous(dc1)
    }

    fn get_categorical() -> FeatureData {
        let dc2: DataContainer<u8> = DataContainer {
            data: vec![5, 3, 2, 1, 4],
            present: vec![true, true, true, false, true],
        };

        FeatureData::Categorical(dc2)
    }

    #[test]
    fn gets_present_continuous_data() {
        let ds = get_continuous();
        assert_eq!(ds.get(0), Datum::Continuous(4.0));
        assert_eq!(ds.get(2), Datum::Continuous(2.0));
    }

    #[test]
    fn gets_present_categorical_data() {
        let ds = get_categorical();
        assert_eq!(ds.get(0), Datum::Categorical(5));
        assert_eq!(ds.get(4), Datum::Categorical(4));
    }

    #[test]
    fn gets_missing_continuous_data() {
        let ds = get_continuous();
        assert_eq!(ds.get(1), Datum::Missing);
    }

    #[test]
    fn gets_missing_categorical_data() {
        let ds = get_categorical();
        assert_eq!(ds.get(3), Datum::Missing);
    }

    #[test]
    fn summarize_categorical_works_with_fixture() {
        let summary = get_categorical().summarize();
        match summary {
            SummaryStatistics::Categorical { min, max, mode } => {
                assert_eq!(min, 2);
                assert_eq!(max, 5);
                assert_eq!(mode, vec![2, 3, 4, 5]);
            }
            _ => panic!("Unexpected summary type"),
        }
    }

    #[test]
    fn summarize_categorical_works_one_mode() {
        let container: DataContainer<u8> = DataContainer {
            data: vec![5, 3, 2, 2, 1, 4],
            present: vec![true, true, true, true, true, true],
        };
        let summary = summarize_categorical(&container);
        match summary {
            SummaryStatistics::Categorical { min, max, mode } => {
                assert_eq!(min, 1);
                assert_eq!(max, 5);
                assert_eq!(mode, vec![2]);
            }
            _ => panic!("Unexpected summary type"),
        }
    }

    #[test]
    fn summarize_categorical_works_two_modes() {
        let container: DataContainer<u8> = DataContainer {
            data: vec![5, 3, 2, 2, 3, 4],
            present: vec![true, true, true, true, true, true],
        };
        let summary = summarize_categorical(&container);
        match summary {
            SummaryStatistics::Categorical { min, max, mode } => {
                assert_eq!(min, 2);
                assert_eq!(max, 5);
                assert_eq!(mode, vec![2, 3]);
            }
            _ => panic!("Unexpected summary type"),
        }
    }

    #[test]
    fn summarize_continuous_works_with_fixture() {
        let summary = get_continuous().summarize();
        match summary {
            SummaryStatistics::Continuous {
                min,
                max,
                mean,
                median,
                variance,
            } => {
                assert_relative_eq!(min, 0.0, epsilon = 1E-10);
                assert_relative_eq!(max, 4.0, epsilon = 1E-10);
                assert_relative_eq!(mean, 1.75, epsilon = 1E-10);
                assert_relative_eq!(median, 1.5, epsilon = 1E-10);
                assert_relative_eq!(variance, 2.1875, epsilon = 1E-10);
            }
            _ => panic!("Unexpected summary type"),
        }
    }

    #[test]
    fn summarize_continuous_works_with_odd_number_data() {
        let container: DataContainer<f64> = DataContainer {
            data: vec![4.0, 3.0, 2.0, 1.0, 0.0],
            present: vec![true, true, true, true, true],
        };
        let summary = summarize_continuous(&container);
        match summary {
            SummaryStatistics::Continuous { median, .. } => {
                assert_relative_eq!(median, 2.0, epsilon = 1E-10);
            }
            _ => panic!("Unexpected summary type"),
        }
    }
}
