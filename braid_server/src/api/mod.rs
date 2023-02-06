// pub mod obj;
pub mod v1;

/// For determining whether an API call will take too long (in that it could jam
/// up the server) and should be rejected with status 429.
pub trait TooLong {
    /// Returns `true` if the query is too long
    fn too_long(&self) -> bool;
    /// Returns a note containing info on how to break queries up so they're not
    /// too long.
    fn too_long_msg(&self) -> String;
}
