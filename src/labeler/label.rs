use crate::result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(
    Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize,
)]
pub struct Label {
    pub label: bool,
    pub truth: Option<bool>,
}

impl Label {
    pub fn new(label: bool, truth: Option<bool>) -> Self {
        Label { label, truth }
    }
}

impl FromStr for Label {
    type Err = result::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let msg = String::from(
            "Could not parse '{}', format should be 'IL(0, 1)' for a label of \
             0, with truth value of 1, or  'IL(1, None)' for a label of 1 \
             with no truth value",
        );
        let re = Regex::new(r"IL\(\s*(0|1),\s*(0|1|None)\s*\)").unwrap();

        let captures = re.captures(&s).ok_or_else(|| {
            let err_kind = result::ErrorKind::ParseError;
            result::Error::new(err_kind, msg.clone())
        })?;

        let label_str = captures
            .get(1)
            .ok_or_else(|| {
                let err_kind = result::ErrorKind::ParseError;
                result::Error::new(err_kind, msg.clone())
            })?
            .as_str();

        let truth_str = captures
            .get(2)
            .ok_or_else(|| {
                let err_kind = result::ErrorKind::ParseError;
                result::Error::new(err_kind, msg.clone())
            })?
            .as_str();

        let label = match label_str {
            "0" => Ok(false),
            "1" => Ok(true),
            _ => {
                let err_kind = result::ErrorKind::ParseError;
                Err(result::Error::new(err_kind, msg.clone()))
            }
        }?;

        let truth = match truth_str {
            "0" => Ok(Some(false)),
            "1" => Ok(Some(true)),
            "None" => Ok(None),
            _ => {
                let err_kind = result::ErrorKind::ParseError;
                Err(result::Error::new(err_kind, msg.clone()))
            }
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
                label: false,
                truth: Some(false)
            }
        );

        assert_eq!(
            Label::from_str("IL(0, 1)").unwrap(),
            Label {
                label: false,
                truth: Some(true)
            }
        );

        assert_eq!(
            Label::from_str("IL(1, 0)").unwrap(),
            Label {
                label: true,
                truth: Some(false)
            }
        );

        assert_eq!(
            Label::from_str("IL(1, 1)").unwrap(),
            Label {
                label: true,
                truth: Some(true)
            }
        );
    }

    #[test]
    fn label_from_string_wacky_whitespace() {
        assert_eq!(
            Label::from_str("IL(0,0)").unwrap(),
            Label {
                label: false,
                truth: Some(false)
            }
        );

        assert_eq!(
            Label::from_str("IL(    0,\t1)").unwrap(),
            Label {
                label: false,
                truth: Some(true)
            }
        );

        assert_eq!(
            Label::from_str("IL( 1, 0 )").unwrap(),
            Label {
                label: true,
                truth: Some(false)
            }
        );
    }

    #[test]
    fn label_from_string_without_truth() {
        assert_eq!(
            Label::from_str("IL(0, None)").unwrap(),
            Label {
                label: false,
                truth: None
            }
        );

        assert_eq!(
            Label::from_str("IL(1, None)").unwrap(),
            Label {
                label: true,
                truth: None
            }
        );
    }

    #[test]
    #[should_panic]
    fn label_from_string_with_truth_bad_label_should_panic() {
        // label must be 0 or 1
        let _label = Label::from_str("IL(2, 1)").unwrap();
    }

    #[test]
    #[should_panic]
    fn label_from_string_with_truth_bad_truth_should_panic() {
        // truth must be 0, 1, or "None"
        let _label = Label::from_str("IL(0, 11)").unwrap();
    }
}
