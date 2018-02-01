use std::f64::NAN;


// TODO: Should this go with ColModel?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DType {
    Continuous(f64),
    Categorical(u8),
    Binary(bool),
    Missing, // Should carry an error message?
}


// XXX: What happens when we add vector types? Error?
impl DType {
    pub fn as_f64(&self) -> f64 {
        match self {
            DType::Continuous(x)  => *x,
            DType::Categorical(x) => *x as f64,
            DType::Binary(x)      => if *x {1.0} else {0.0},
            DType::Missing        => NAN,
        }
    }

    pub fn as_string(&self) -> String {
        match self {
            DType::Continuous(x)  => format!("{}", *x),
            DType::Categorical(x) => format!("{}", *x),
            DType::Binary(x)      => format!("{}", *x),
            DType::Missing        => String::from("NaN")
        }
    }

    pub fn is_continuous(&self) -> bool {
        match self {
            DType::Continuous(_) => true,
            _ => false
        }
    }

    pub fn is_categorical(&self) -> bool {
        match self {
            DType::Categorical(_) => true,
            _ => false
        }
    }

    pub fn is_binary(&self) -> bool {
        match self {
            DType::Binary(_) => true,
            _ => false
        }
    }

    pub fn is_missing(&self) -> bool {
        match self {
            DType::Missing => true,
            _ => false
        }
    }
}
