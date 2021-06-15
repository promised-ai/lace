use std::f64;

/// Types that have a default missing (Null) value for sql
pub trait SqlNull {
    fn sql_null() -> Self;
}

impl SqlNull for f64 {
    fn sql_null() -> Self {
        Self::NAN
    }
}

impl SqlNull for u8 {
    fn sql_null() -> Self {
        Self::max_value()
    }
}

/// Specify the default filler value for missing entries
pub trait SqlDefault {
    fn sql_default() -> Self;
}

impl SqlDefault for f64 {
    fn sql_default() -> Self {
        0.0
    }
}

impl SqlDefault for u8 {
    fn sql_default() -> Self {
        0
    }
}
