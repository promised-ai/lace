use std::{io, error, fmt};

#[derive(Clone, Debug)]
pub enum OracleError {
    RowIndexOutOfBounds { row_ix: usize, nrows: usize},
    ColumnIndexOutOfBounds { col_ix: usize, ncols: usize},
    InvalidDType { col_ix: usize, dtype: String, expected: String },
    GivenQueryColumnOverlap { col_ix: usize },
    NegativeEntropy { col_ix: usize },
    ZeroSamples,
}


impl OracleError {
    pub fn to_error(&self) -> io::Error {
        match &self {
            OracleError::RowIndexOutOfBounds { row_ix, nrows} => {
                io::Error::new(io::ErrorKind::InvalidInput, self.clone())
            },
            OracleError::ColumnIndexOutOfBounds { col_ix, ncols} => {
                io::Error::new(io::ErrorKind::InvalidInput, self.clone())
            },
            OracleError::InvalidDType { col_ix, dtype, expected } => {
                io::Error::new(io::ErrorKind::InvalidData, self.clone())
            },
            OracleError::GivenQueryColumnOverlap { col_ix} => {
                io::Error::new(io::ErrorKind::InvalidInput, self.clone())
            },
            OracleError::NegativeEntropy { col_ix } => {
                io::Error::new(io::ErrorKind::Other, self.clone())
            },
            OracleError::ZeroSamples => {
                io::Error::new(io::ErrorKind::InvalidInput, self.clone())
            },
        }
    }

    pub fn to_string(&self) -> String {
        match &self {
            OracleError::RowIndexOutOfBounds { row_ix, nrows} => {
                format!("Row index {} out of bounds for oracle with {} rows",
                        row_ix, nrows)
            },
            OracleError::ColumnIndexOutOfBounds { col_ix, ncols} => {
                format!("Column index {} out of bounds for oracle with {} rows",
                        col_ix, ncols)
            },
            OracleError::InvalidDType { col_ix, dtype, expected } => {
                format!("Column {} expects {} but recieved {}",
                        col_ix, dtype, expected)
            },
            OracleError::GivenQueryColumnOverlap { col_ix } => {
                format!("Column {} appears in query and given conditions",
                        col_ix)
            },
            OracleError::NegativeEntropy {col_ix } => {
                format!("Column {} has negative entropy", col_ix)
            },
            OracleError::ZeroSamples => {
                String::from("Number of samples must be greater than zero")
            },
        }
    }
}


// Traits
// ------
impl fmt::Display for OracleError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}


impl error::Error for OracleError {
    fn description(&self) -> &str {
        match &self {
            OracleError::RowIndexOutOfBounds {..} => {
                "row index out of state bounds"
            },
            OracleError::ColumnIndexOutOfBounds {..} => {
                "column index out of state bounds"
            },
            OracleError::InvalidDType {..} => {
                "invalid data type given to feature"
            },
            OracleError::GivenQueryColumnOverlap {..} => {
                "column appeared in query and given condition"
            },
            OracleError::NegativeEntropy {..} => {
                "negative entropy detected"
            },
            OracleError::ZeroSamples => {
                "number of samples must be greater than 0"
            },
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        None
    }
}
