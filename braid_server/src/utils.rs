use serde::Serialize;
use warp::reply::Response;
use warp::{Filter, Reply};

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

pub(crate) fn gzip_accepted(accept_encoding: Option<&String>) -> bool {
    if let Some(values) = accept_encoding {
        // FIXME: this can pic up things in an invalid header
        values.contains("gzip")
    } else {
        false
    }
}

// A GZipped JSON response
pub(crate) struct JsonGz {
    inner: Result<Vec<u8>, ()>,
    gzipped: bool,
}

pub(crate) fn jsongz<T: Serialize>(
    obj: &T,
    accept_encoding: Option<&String>,
) -> JsonGz {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    let mut gzipped = false;
    let inner = serde_json::to_vec(obj)
        .map_err(|err| tracing::error!("jsongz error: {}", err))
        .and_then(|bytes| {
            if gzip_accepted(accept_encoding) {
                let mut encoder =
                    GzEncoder::new(Vec::new(), Compression::fast());
                let res = encoder.write_all(bytes.as_slice());
                res.map_err(|err| tracing::error!("jsongz error: {}", err))
                    .and_then(|_| {
                        gzipped = true;
                        encoder.finish().map_err(|err| {
                            tracing::error!("jsongz error: {}", err)
                        })
                    })
            } else {
                Ok(bytes)
            }
        });

    JsonGz { inner, gzipped }
}

impl Reply for JsonGz {
    #[inline]
    fn into_response(self) -> Response {
        use warp::http::header::{HeaderValue, CONTENT_ENCODING, CONTENT_TYPE};
        use warp::http::StatusCode;
        match self.inner {
            Ok(body) => {
                let mut res = Response::new(body.into());
                res.headers_mut().insert(
                    CONTENT_TYPE,
                    HeaderValue::from_static("application/json"),
                );
                if self.gzipped {
                    res.headers_mut().insert(
                        CONTENT_ENCODING,
                        HeaderValue::from_static("gzip"),
                    );
                }

                res
            }
            Err(()) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        }
    }
}
