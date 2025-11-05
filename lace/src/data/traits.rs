use crate::data::Datum;
use crate::data::FeatureData;
use crate::data::SparseContainer;

pub trait AccumScore<T> {
    /// Compute scores on the data using `score_fn` and add them to `scores`
    fn accum_score<F: Fn(&T) -> f64>(&self, scores: &mut [f64], score_fn: &F);
}

pub trait TranslateDatum<X> {
    fn extract(x: Datum) -> X;
    fn pack(x: X) -> Datum;
}

pub trait TranslateContainer<X: Clone> {
    fn extract_container(xs: FeatureData) -> SparseContainer<X>;
    fn pack_container(xs: SparseContainer<X>) -> FeatureData;
}

/// A data container
pub trait Container<T: Clone> {
    /// get the data slices and the start indices
    fn get_slices(&self) -> Vec<(usize, &[T])>;

    /// Get the entry at ix if it exists
    fn get(&self, ix: usize) -> Option<T>;

    /// Get the entry at ix if it exists as a Datum
    fn get_datum<Fx: TranslateDatum<T>>(&self, ix: usize) -> Datum {
        self.get(ix).map_or(Datum::Missing, Fx::pack)
    }

    /// Insert or overwrite an entry at ix
    fn insert_overwrite(&mut self, ix: usize, x: T);

    /// Append a new datum to the end of the container
    fn push(&mut self, xopt: Option<T>);

    /// Append n new empty entries to the container
    fn upsize(&mut self, n: usize) {
        (0..n).for_each(|_| self.push(None));
    }

    /// Get as cloned vector containing only the present data
    fn present_cloned(&self) -> Vec<T>;

    /// Remove and return the entry at ix if it exists. Used to mark a present
    /// datum as missing, not to completely remove a record. Does not decrease
    /// the length.
    fn remove(&mut self, ix: usize) -> Option<T>;

    // TODO: should return result type
    fn push_datum<Fx: TranslateDatum<T>>(&mut self, x: Datum) {
        match x {
            Datum::Missing => self.push(None),
            _ => {
                let x: T = Fx::extract(x);
                self.push(Some(x))
            }
        }
    }

    fn insert_datum<Fx: TranslateDatum<T>>(&mut self, row_ix: usize, x: Datum) {
        match x {
            Datum::Missing => {
                self.remove(row_ix);
            }
            _ => {
                let x: T = Fx::extract(x);
                self.insert_overwrite(row_ix, x)
            }
        }
    }
}

impl TranslateDatum<u32> for crate::consts::rv::dist::Poisson {
    fn extract(x: Datum) -> u32 {
        match x {
            Datum::Count(y) => y,
            _ => panic!("Count not extract {x:?} into Poisson"),
        }
    }

    fn pack(x: u32) -> Datum {
        Datum::Count(x)
    }
}

impl TranslateDatum<u32> for crate::consts::rv::dist::Categorical {
    fn extract(x: Datum) -> u32 {
        match x {
            Datum::Categorical(crate::Category::UInt(y)) => y,
            Datum::Categorical(crate::Category::Bool(y)) => y as u32,
            _ => panic!("Count not extract {x:?} into Categorical"),
        }
    }

    fn pack(x: u32) -> Datum {
        Datum::Categorical(crate::Category::UInt(x))
    }
}

impl TranslateDatum<f64> for crate::consts::rv::dist::Gaussian {
    fn extract(x: Datum) -> f64 {
        match x {
            Datum::Continuous(y) => y,
            _ => panic!("Count not extract {x:?} into Gaussian"),
        }
    }

    fn pack(x: f64) -> Datum {
        Datum::Continuous(x)
    }
}

impl TranslateDatum<bool> for crate::consts::rv::dist::Bernoulli {
    fn extract(x: Datum) -> bool {
        match x {
            Datum::Binary(y) => y,
            _ => panic!("Count not extract {x:?} into Bernoulli"),
        }
    }

    fn pack(x: bool) -> Datum {
        Datum::Binary(x)
    }
}

macro_rules! impl_trainslate_container {
    ($X: ty, $Fx: ident, $variant: ident) => {
        impl TranslateContainer<$X> for crate::consts::rv::dist::$Fx {
            fn extract_container(xs: FeatureData) -> SparseContainer<$X> {
                match xs {
                    FeatureData::$variant(ctr) => ctr,
                    _ => panic!("Failed to extract"),
                }
            }

            fn pack_container(xs: SparseContainer<$X>) -> FeatureData {
                FeatureData::$variant(xs)
            }
        }
    };
}

impl_trainslate_container!(u32, Categorical, Categorical);
impl_trainslate_container!(u32, Poisson, Count);
impl_trainslate_container!(f64, Gaussian, Continuous);
impl_trainslate_container!(bool, Bernoulli, Binary);
