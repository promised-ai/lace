//! The `Codebook` is a YAML file used to associate metadata with the dataset.
//! The user can set the priors on the structure of each state, can identify
//! the model for each columns, and set hyper priors.
//!
//! Often the data has too many columns to write a codebook manually, so there
//! are functions to guess at a default codebook given a dataset. The user can
//! then edit the default file.
//!
//! # Example
//!
//! An Example codebook for a two-column dataset.
//!
//! ```
//! # use braid_codebook::Codebook;
//! use indoc::indoc;
//!
//! let codebook_str = indoc!("
//!     ---
//!     table_name: two column dataset
//!     state_alpha_prior:
//!       InvGamma:
//!         shape: 1.0
//!         scale: 1.0
//!     view_alpha_prior:
//!       Gamma:
//!         shape: 1.0
//!         rate: 1.0
//!     col_metadata:
//!       - name: col_1
//!         notes: first column with all fields filled in
//!         coltype:
//!           Categorical:
//!             k: 3
//!             hyper:
//!               pr_alpha:
//!                 shape: 1.0
//!                 scale: 1.0
//!             prior:
//!                 k: 3
//!                 alpha: 0.5
//!             value_map:
//!               0: red
//!               1: green
//!               2: blue
//!       - name: col_2
//!         notes: A binary column with optional fields left out
//!         coltype:
//!           Categorical:
//!             k: 2
//!     comments: An example codebook
//!     row_names:
//!       - A
//!       - B
//!       - C
//!       - D
//!       - E");
//!
//! let codebook: Codebook = serde_yaml::from_str(&codebook_str).unwrap();
//!
//! assert_eq!(codebook.col_metadata.len(), 2);
//! ```
#![warn(
    clippy::all,
    clippy::imprecise_flops,
    clippy::suboptimal_flops,
    clippy::unseparated_literal_suffix,
    clippy::unreadable_literal,
    clippy::option_option,
    clippy::implicit_clone
)]
mod codebook;
pub mod csv;
mod error;

pub use codebook::*;
pub use error::*;
