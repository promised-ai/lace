/// A type of data
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DType {
    #[serde(rename = "continuous")]
    Continuous(f64),
    #[serde(rename = "categorical")]
    Categorical(u8),
    #[serde(rename = "binary")]
    Binary(bool),
    #[serde(rename = "missing")]
    Missing, // Should carry an error message?
}

// XXX: What happens when we add vector types? Error?
impl DType {
    /// Unwraps the datum as an `f64` if possible
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            DType::Continuous(x) => Some(*x),
            DType::Categorical(x) => Some(*x as f64),
            DType::Binary(x) => {
                if *x {
                    Some(1.0)
                } else {
                    Some(0.0)
                }
            }
            DType::Missing => None,
        }
    }

    /// Unwraps the datum as an `u8` if possible
    pub fn as_u8(&self) -> Option<u8> {
        match self {
            DType::Continuous(..) => None,
            DType::Categorical(x) => Some(*x),
            DType::Binary(x) => {
                if *x {
                    Some(1)
                } else {
                    Some(0)
                }
            }
            DType::Missing => None,
        }
    }

    /// Returns the datum as a string
    pub fn as_string(&self) -> String {
        match self {
            DType::Continuous(x) => format!("{}", *x),
            DType::Categorical(x) => format!("{}", *x),
            DType::Binary(x) => format!("{}", *x),
            DType::Missing => String::from("NaN"),
        }
    }

    /// Returns `true` if the `DType` is continuous
    pub fn is_continuous(&self) -> bool {
        match self {
            DType::Continuous(_) => true,
            _ => false,
        }
    }

    /// Returns `true` if the `DType` is categorical
    pub fn is_categorical(&self) -> bool {
        match self {
            DType::Categorical(_) => true,
            _ => false,
        }
    }

    /// Returns `true` if the `DType` is binary
    pub fn is_binary(&self) -> bool {
        match self {
            DType::Binary(_) => true,
            _ => false,
        }
    }

    /// Returns `true` if the `DType` is missing
    pub fn is_missing(&self) -> bool {
        match self {
            DType::Missing => true,
            _ => false,
        }
    }
}
