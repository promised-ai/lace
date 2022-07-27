use derive_more::{Display, From};
use serde::Serialize;
use utoipa::Component;

/// An error for which the user is at fault.
#[derive(Debug, Clone, Display, From, Component, Serialize)]
pub struct UserError(pub String);

impl UserError {
    pub fn from_error<E>(e: E) -> Self
    where
        E: std::error::Error,
    {
        Self(e.to_string())
    }
}

impl From<&str> for UserError {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// An error for which the server is at fault.
#[derive(Debug, Clone, Display, From, Component, Serialize)]
pub struct InternalError(pub String);

/// Errors producable from user or server activity
#[derive(Clone, Debug, Display, From, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Error {
    /// User Caused Error
    User(
        /// The error itself
        UserError,
    ),
    /// Internal Server Error
    Internal(
        /// The error itself
        InternalError,
    ),
}

impl warp::reject::Reject for Error {}

impl From<tokio::task::JoinError> for Error {
    fn from(e: tokio::task::JoinError) -> Self {
        Self::Internal(InternalError(e.to_string()))
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Internal(InternalError(e.to_string()))
    }
}
