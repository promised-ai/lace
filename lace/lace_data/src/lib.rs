#![warn(unused_extern_crates)]
#![warn(
    clippy::all,
    clippy::imprecise_flops,
    clippy::suboptimal_flops,
    clippy::unseparated_literal_suffix,
    clippy::unreadable_literal,
    clippy::option_option,
    clippy::implicit_clone
)]

mod category;
mod data_store;
mod datum;
mod feature;
mod sparse;
mod traits;

pub use category::Category;
pub use data_store::DataStore;
pub use datum::{Datum, DatumConversionError};
pub use feature::{FeatureData, SummaryStatistics};
pub use sparse::SparseContainer;
pub use traits::AccumScore;
pub use traits::Container;
pub use traits::TranslateContainer;
pub use traits::TranslateDatum;
