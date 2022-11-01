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
