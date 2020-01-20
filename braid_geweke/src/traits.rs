use rand::Rng;

/// The trait that allows samplers to be tested by `GewekeTester`.
pub trait GewekeModel: GewekeResampleData + GewekeSummarize {
    /// Draw a new object from the prior
    fn geweke_from_prior(settings: &Self::Settings, rng: &mut impl Rng)
        -> Self;

    /// Update the state of the object by performing 1 MCMC transition
    fn geweke_step(&mut self, settings: &Self::Settings, rng: &mut impl Rng);
}

/// Allow the data to be re-sampled within the component
pub trait GewekeResampleData {
    /// Any settings needed to manage the resample
    type Settings;

    /// re-sample the data from the current parameters
    fn geweke_resample_data(
        &mut self,
        s: Option<&Self::Settings>,
        rng: &mut impl Rng,
    );
}

/// Summarize the state of the model
pub trait GewekeSummarize: GewekeResampleData {
    /// The type of the summary. Must be convertible into `Map<String, f64>`
    type Summary;

    /// Summarize the model.
    fn geweke_summarize(&self, settings: &Self::Settings) -> Self::Summary;
}
