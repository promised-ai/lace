use thiserror::Error;

#[derive(Clone, Debug, Error)]
pub enum ProposeLatentValuesError {
    #[error(
        "Attempted to propose latent values for non-latent columns: {0:?}"
    )]
    NonLatentColumns(Vec<usize>),
}
