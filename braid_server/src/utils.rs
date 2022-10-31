use serde::Serialize;
use warp::Filter;

/// Error response structure
#[derive(Serialize, Debug)]
struct ErrorResp {
    func_name: String,
    args: String,
    error: String,
}

// Composition rules which fix an issue with deeply nested filters in warp
//#[cfg(debug_assertions)]
macro_rules! compose {
    ($x:expr $(,)?) => (
        $x
    );
    ($x0:expr, $($x:expr),+ $(,)?) => (
        $x0$(.or($x).boxed())+
    );
}

/*
#[cfg(not(debug_assertions))]
macro_rules! compose {
    ($x:expr $(,)?) => (
        $x
    );
    ($x0:expr, $($x:expr),+ $(,)?) => (
        $x0$(.or($x))+
    );
}
*/

/// Include an object in the arguments of a handler for warp
pub fn with<T>(
    t: T,
) -> impl Filter<Extract = (T,), Error = std::convert::Infallible> + Clone
where
    T: Clone + Send,
{
    warp::any().map(move || t.clone())
}

pub(crate) use compose;

// NOTE: The items below are copied from the unstable `iter_intersperse`
// feature so we don't have to use nightly just for this.
pub(crate) trait Weave: Iterator {
    fn weave(self, separator: Self::Item) -> Intersperse<Self>
    where
        Self: Sized,
        Self::Item: Clone,
    {
        Intersperse::new(self, separator)
    }
}

impl<I> Weave for I
where
    I: Iterator + Sized,
    I::Item: Clone,
{
}

#[derive(Debug, Clone)]
pub(crate) struct Intersperse<I: Iterator>
where
    I::Item: Clone,
{
    separator: I::Item,
    iter: std::iter::Peekable<I>,
    needs_sep: bool,
}

impl<I: Iterator> Intersperse<I>
where
    I::Item: Clone,
{
    fn new(iter: I, separator: I::Item) -> Self {
        Self {
            iter: iter.peekable(),
            separator,
            needs_sep: false,
        }
    }
}

impl<I> Iterator for Intersperse<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        if self.needs_sep && self.iter.peek().is_some() {
            self.needs_sep = false;
            Some(self.separator.clone())
        } else {
            self.needs_sep = true;
            self.iter.next()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        intersperse_size_hint(&self.iter, self.needs_sep)
    }
}

fn intersperse_size_hint<I>(iter: &I, needs_sep: bool) -> (usize, Option<usize>)
where
    I: Iterator,
{
    let (lo, hi) = iter.size_hint();
    let next_is_elem = !needs_sep;
    (
        lo.saturating_sub(next_is_elem as usize).saturating_add(lo),
        hi.and_then(|hi| {
            hi.saturating_sub(next_is_elem as usize).checked_add(hi)
        }),
    )
}
