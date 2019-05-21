use rand::Rng;
use std::collections::BTreeMap;

/// The trait that allows samplers to be tested by `GewekeTester`.
pub trait GewekeModel: GewekeResampleData + GewekeSummarize {
    /// Draw a new object from the prior
    fn geweke_from_prior(settings: &Self::Settings, rng: &mut impl Rng)
        -> Self;

    /// Update the state of the object by performing 1 MCMC transition
    fn geweke_step(&mut self, settings: &Self::Settings, rng: &mut impl Rng);
}

pub trait GewekeResampleData {
    type Settings;
    fn geweke_resample_data(
        &mut self,
        s: Option<&Self::Settings>,
        rng: &mut impl Rng,
    );
}

pub trait GewekeSummarize: GewekeResampleData {
    fn geweke_summarize(
        &self,
        settings: &Self::Settings,
    ) -> BTreeMap<String, f64>;
}
