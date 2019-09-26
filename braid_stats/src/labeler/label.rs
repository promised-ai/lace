use regex::Regex;
use serde::{Deserialize, Serialize};
use std::iter::Iterator;
use std::str::FromStr;

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
    Serialize,
    Deserialize,
)]
pub struct Label {
    pub label: u8,
    pub truth: Option<u8>,
}

impl Label {
    pub fn new(label: u8, truth: Option<u8>) -> Self {
        Label { label, truth }
    }
}

impl Default for Label {
    fn default() -> Self {
        Label {
            label: 0,
            truth: Some(0),
        }
    }
}

pub struct LabelIterator {
    n_labels: u8,
    label: u8,
    truth: u8,
}

impl LabelIterator {
    pub fn new(n_labels: u8) -> Self {
        LabelIterator {
            n_labels,
            label: 0,
            truth: 0,
        }
    }
}

impl Iterator for LabelIterator {
    type Item = Label;
    fn next(&mut self) -> Option<Self::Item> {
        if self.label == self.n_labels {
            None
        } else {
            let output = Label::new(self.label, Some(self.truth));
            if self.truth == self.n_labels - 1 {
                self.label += 1;
                self.truth = 0;
            } else {
                self.truth += 1;
            };
            Some(output)
        }
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct ParseLabelError(String);

impl ParseLabelError {
    pub fn new<S>(input: S) -> Self
    where
        S: std::fmt::Display,
    {
        let msg = format!(
            "Could not parse '{}', format should be 'IL(0, 1)' for a label of \
             0, with truth value of 1, or  'IL(1, None)' for a label of 1 \
             with no truth value",
            input
        );
        ParseLabelError(msg)
    }
}

impl FromStr for Label {
    type Err = ParseLabelError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let re = Regex::new(r"IL\(\s*(\d+),\s*(\d+|None)\s*\)").unwrap();

        let captures =
            re.captures(&s).ok_or_else(|| ParseLabelError::new(s))?;

        let label_str = captures
            .get(1)
            .ok_or_else(|| ParseLabelError::new(s))?
            .as_str();

        let truth_str = captures
            .get(2)
            .ok_or_else(|| ParseLabelError::new(s))?
            .as_str();

        let label =
            u8::from_str(label_str).map_err(|_| ParseLabelError::new(s))?;

        let truth = if truth_str == "None" {
            Ok(None)
        } else {
            u8::from_str(truth_str)
                .map_err(|_| ParseLabelError::new(s))
                .map(Some)
        }?;

        Ok(Label { label, truth })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_from_string_with_truth() {
        assert_eq!(
            Label::from_str("IL(0, 0)").unwrap(),
            Label {
                label: 0,
                truth: Some(0)
            }
        );

        assert_eq!(
            Label::from_str("IL(0, 1)").unwrap(),
            Label {
                label: 0,
                truth: Some(1)
            }
        );

        assert_eq!(
            Label::from_str("IL(1, 0)").unwrap(),
            Label {
                label: 1,
                truth: Some(0)
            }
        );

        assert_eq!(
            Label::from_str("IL(1, 12)").unwrap(),
            Label {
                label: 1,
                truth: Some(12)
            }
        );
    }

    #[test]
    fn label_from_string_wacky_whitespace() {
        assert_eq!(
            Label::from_str("IL(0,0)").unwrap(),
            Label {
                label: 0,
                truth: Some(0)
            }
        );

        assert_eq!(
            Label::from_str("IL(    0,\t1)").unwrap(),
            Label {
                label: 0,
                truth: Some(1)
            }
        );

        assert_eq!(
            Label::from_str("IL( 1, 0 )").unwrap(),
            Label {
                label: 1,
                truth: Some(0)
            }
        );
    }

    #[test]
    fn label_from_string_without_truth() {
        assert_eq!(
            Label::from_str("IL(0, None)").unwrap(),
            Label {
                label: 0,
                truth: None
            }
        );

        assert_eq!(
            Label::from_str("IL(1, None)").unwrap(),
            Label {
                label: 1,
                truth: None
            }
        );
    }

    #[test]
    #[should_panic]
    fn label_from_string_with_truth_bad_label_should_panic() {
        // label must be < 256
        let _label = Label::from_str("IL(256, 1)").unwrap();
    }

    #[test]
    #[should_panic]
    fn label_from_string_with_truth_bad_truth_should_panic() {
        // truth must be 0, 1, or "None"
        let _label = Label::from_str("IL(0, 267)").unwrap();
    }

    #[test]
    fn label_iterator_3_should_cover_0_to_2() {
        let mut iter = LabelIterator::new(3);
        assert_eq!(iter.next(), Some(Label { label: 0, truth: Some(0) }));
        assert_eq!(iter.next(), Some(Label { label: 0, truth: Some(1) }));
        assert_eq!(iter.next(), Some(Label { label: 0, truth: Some(2) }));
        assert_eq!(iter.next(), Some(Label { label: 1, truth: Some(0) }));
        assert_eq!(iter.next(), Some(Label { label: 1, truth: Some(1) }));
        assert_eq!(iter.next(), Some(Label { label: 1, truth: Some(2) }));
        assert_eq!(iter.next(), Some(Label { label: 2, truth: Some(0) }));
        assert_eq!(iter.next(), Some(Label { label: 2, truth: Some(1) }));
        assert_eq!(iter.next(), Some(Label { label: 2, truth: Some(2) }));
        assert_eq!(iter.next(), None);
    }
}
