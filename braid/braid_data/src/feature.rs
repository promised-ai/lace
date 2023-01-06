use crate::label::Label;
use crate::Datum;
use crate::{Container, SparseContainer};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SummaryStatistics {
    #[serde(rename = "binary")]
    Binary {
        n: usize,
        pos: usize,
    },
    #[serde(rename = "continuous")]
    Continuous {
        min: f64,
        max: f64,
        mean: f64,
        median: f64,
        variance: f64,
    },
    #[serde(rename = "categorical")]
    Categorical {
        min: u8,
        max: u8,
        mode: Vec<u8>,
    },
    #[serde(rename = "labeler")]
    Labeler {
        n: usize,
        n_true: usize,
        n_false: usize,
        n_labeled: usize,
        n_correct: usize,
    },
    #[serde(rename = "count")]
    Count {
        min: u32,
        max: u32,
        median: f64,
        mean: f64,
        mode: Vec<u32>,
    },
    None,
}

/// Used when pulling data from features for saving
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum FeatureData {
    /// Binary data
    Binary(SparseContainer<bool>),
    /// Univariate continuous data
    Continuous(SparseContainer<f64>),
    /// Categorical data
    Categorical(SparseContainer<u8>),
    /// Categorical data
    Labeler(SparseContainer<Label>),
    /// Count data
    Count(SparseContainer<u32>),
}

impl FeatureData {
    pub fn len(&self) -> usize {
        match self {
            Self::Binary(xs) => xs.len(),
            Self::Continuous(xs) => xs.len(),
            Self::Categorical(xs) => xs.len(),
            Self::Count(xs) => xs.len(),
            Self::Labeler(xs) => xs.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the datum at [row_ix, col_ix] as a `Datum`
    pub fn is_present(&self, ix: usize) -> bool {
        match self {
            Self::Binary(xs) => xs.is_present(ix),
            Self::Continuous(xs) => xs.is_present(ix),
            Self::Categorical(xs) => xs.is_present(ix),
            Self::Count(xs) => xs.is_present(ix),
            Self::Labeler(xs) => xs.is_present(ix),
        }
    }

    /// Get the datum at [row_ix, col_ix] as a `Datum`
    pub fn get(&self, ix: usize) -> Datum {
        // TODO: SparseContainer index get (xs[i]) should return an option
        match self {
            FeatureData::Binary(xs) => {
                xs.get(ix).map(Datum::Binary).unwrap_or(Datum::Missing)
            }
            FeatureData::Continuous(xs) => {
                xs.get(ix).map(Datum::Continuous).unwrap_or(Datum::Missing)
            }
            FeatureData::Categorical(xs) => {
                xs.get(ix).map(Datum::Categorical).unwrap_or(Datum::Missing)
            }
            FeatureData::Labeler(xs) => {
                xs.get(ix).map(Datum::Label).unwrap_or(Datum::Missing)
            }
            FeatureData::Count(xs) => {
                xs.get(ix).map(Datum::Count).unwrap_or(Datum::Missing)
            }
        }
    }

    /// Get the summary statistic for a column
    pub fn summarize(&self) -> SummaryStatistics {
        match self {
            FeatureData::Binary(ref container) => SummaryStatistics::Binary {
                n: container.n_present(),
                pos: container
                    .get_slices()
                    .iter()
                    .map(|(_, xs)| xs.len())
                    .sum::<usize>(),
            },
            FeatureData::Continuous(ref container) => {
                summarize_continuous(container)
            }
            FeatureData::Categorical(ref container) => {
                summarize_categorical(container)
            }
            FeatureData::Count(ref container) => summarize_count(container),
            FeatureData::Labeler(..) => {
                unimplemented!("cannot summarize labeler column")
            }
        }
    }
}

pub fn summarize_continuous(
    container: &SparseContainer<f64>,
) -> SummaryStatistics {
    use braid_utils::{mean, var};
    let mut xs: Vec<f64> = container.present_cloned();

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
    container: &SparseContainer<u8>,
) -> SummaryStatistics {
    use braid_utils::{bincount, minmax};
    let xs: Vec<u8> = container.present_cloned();

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

pub fn summarize_count(container: &SparseContainer<u32>) -> SummaryStatistics {
    use braid_utils::{bincount, minmax};
    let xs: Vec<usize> = {
        let mut xs: Vec<usize> =
            container.present_iter().map(|&x| x as usize).collect();
        xs.sort_unstable();
        xs
    };

    let n = xs.len();
    let nf = n as f64;

    let (min, max) = {
        let (min, max) = minmax(&xs);
        (min as u32, max as u32)
    };

    let counts = bincount(&xs, (max + 1) as usize);

    let max_ct = counts
        .iter()
        .fold(0_usize, |acc, &ct| if ct > acc { ct } else { acc });

    let mode = counts
        .iter()
        .enumerate()
        .filter(|(_, &ct)| ct == max_ct)
        .map(|(ix, _)| ix as u32)
        .collect();

    let mean = xs.iter().sum::<usize>() as f64 / nf;

    let median = if n % 2 == 0 {
        (xs[n / 2] + xs[n / 2 - 1]) as f64 / 2.0
    } else {
        xs[n / 2] as f64
    };

    SummaryStatistics::Count {
        min,
        max,
        median,
        mean,
        mode,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    fn get_continuous() -> FeatureData {
        let dc1: SparseContainer<f64> = SparseContainer::from(vec![
            (4.0, true),
            (3.0, false),
            (2.0, true),
            (1.0, true),
            (0.0, true),
        ]);

        FeatureData::Continuous(dc1)
    }

    fn get_categorical() -> FeatureData {
        let dc2: SparseContainer<u8> = SparseContainer::from(vec![
            (5, true),
            (3, true),
            (2, true),
            (1, false),
            (4, true),
        ]);

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
        let container: SparseContainer<u8> = SparseContainer::from(vec![
            (5, true),
            (3, true),
            (2, true),
            (2, true),
            (1, true),
            (4, true),
        ]);

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
        let container: SparseContainer<u8> = SparseContainer::from(vec![
            (5, true),
            (3, true),
            (2, true),
            (2, true),
            (3, true),
            (4, true),
        ]);

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
        let container: SparseContainer<f64> = SparseContainer::from(vec![
            (4.0, true),
            (3.0, true),
            (2.0, true),
            (1.0, true),
            (0.0, true),
        ]);

        let summary = summarize_continuous(&container);
        match summary {
            SummaryStatistics::Continuous { median, .. } => {
                assert_relative_eq!(median, 2.0, epsilon = 1E-10);
            }
            _ => panic!("Unexpected summary type"),
        }
    }
}
