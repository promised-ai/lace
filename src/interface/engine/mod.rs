mod builder;
mod engine;
pub mod error;

pub use builder::EngineBuilder;
pub use engine::Engine;
pub use engine::EngineSaver;

/// How to enfore row alignment when appending columns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowAlignmentStrategy {
    /// Do not check row names. Only checks that the new columns are the same
    /// length as the existing columns. This is the only strategy that will
    /// work when there are no row names in the codebook.
    Ignore,
    /// Check that the row names match and are in the same order.
    CheckNames,
}
