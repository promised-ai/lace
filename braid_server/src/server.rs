use crate::api::v1::{
    self, AssignmentResponse, CodebookResponse, DepprobRequest,
    DepprobResponse, DrawRequest, DrawResponse, EntropyRequest,
    EntropyResponse, FTypeResponse, FTypesResponse, FeatureErrorResponse,
    GetDataRequest, GetDataResponse, ImputeRequest, ImputeResponse,
    InfoPropRequest, InfoPropResponse, InsertDataRequest, InsertDataResponse,
    LogpRequest, LogpResponse, LogpScaledRequest, LogpScaledResponse,
    MiRequest, MiResponse, NStatesResponse, NoveltyRequest, NoveltyResponse,
    PredictRequest, PredictResponse, RequestLimitsResponse, RowsimRequest,
    RowsimResponse, ShapeResponse, SimulateRequest, SimulateResponse,
    StateDiagnosticsResponse, SummarizeColumnsRequest,
    SummarizeColumnsResponse, SurprisalRequest, SurprisalResponse,
    UpdateRequest, VersionResponse,
};
use crate::api::TooLong;
use crate::result::UserError;
use crate::result::{self, Error};
use crate::utils::{compose, gzip_accepted, jsongz, with, JsonGz};
use braid::{Datum, Engine, OracleT, UserInfo};
use braid::{HasStates, NameOrIndex};
use serde::Serialize;
use std::convert::Infallible;
use std::path::Path;
use std::sync::Arc;
use warp::http::header::ACCEPT_ENCODING;
use warp::hyper::StatusCode;
use warp::{Filter, Rejection};

#[cfg(feature = "download")]
use crate::result::InternalError;
#[cfg(feature = "download")]
use std::fs::File;
#[cfg(feature = "download")]
use std::path::PathBuf;
#[cfg(feature = "download")]
use tokio_util::io::ReaderStream;
#[cfg(feature = "download")]
use warp::hyper::header::CONTENT_TYPE;

#[derive(Debug, Clone, Copy)]
pub enum ServerMutability {
    Mutable,
    Immutable,
}

impl ServerMutability {
    fn is_immutable(self) -> bool {
        match self {
            Self::Mutable => false,
            Self::Immutable => true,
        }
    }

    fn from_mutable_bool(mutable: bool) -> Self {
        if mutable {
            Self::Mutable
        } else {
            Self::Immutable
        }
    }
}

macro_rules! check_too_long {
    ($query: expr) => {
        if $query.too_long() {
            let err = result::UserError($query.too_long_msg());
            Err(Error::User(err))
        } else {
            Ok(())
        }
    };
}

macro_rules! no_zip_if_output_lt {
    ($req: ident, $field: ident, $len: expr, $accept_encoding: ident) => {{
        if $req.$field.len() < $len {
            None
        } else {
            $accept_encoding
        }
    }};
}

// Hold a tempdir with a path to a tarball inside. This is a way to use tempfile
// with rocket's NamedFile responder. NamedFile, doesn't allo wthe user to
// manually set the content type, but instead infers the content type from the
// file extension. Tempfile doesn't allow you to change the file extension, so
// here we are...
#[cfg(feature = "download")]
struct TempTarball {
    // needs to own the tempdir so it isn't destroyed
    _dir: tempfile::TempDir,
    path: PathBuf,
}

#[cfg(feature = "download")]
impl TempTarball {
    fn new() -> Result<Self, UserError> {
        let dir = tempfile::tempdir().map_err(UserError::from_error)?;
        let path = dir.path().join("metadata.tar.gz");
        Ok(Self { _dir: dir, path })
    }
}

// Serializes engine, writes it to a tarball, and returns the file handle
#[cfg(feature = "download")]
fn save_metadata(engine: &Engine) -> Result<TempTarball, UserError> {
    use flate2::write::GzEncoder;
    use flate2::Compression;

    let save_config = braid::metadata::SaveConfig {
        serialized_type: braid::metadata::SerializedType::Bincode,
        ..Default::default()
    };

    let engine_dir = tempfile::tempdir().map_err(UserError::from_error)?;

    engine
        .save(engine_dir.path(), &save_config)
        .map_err(UserError::from_error)?;

    let tarball = TempTarball::new()?;
    let tar_gz =
        File::create(tarball.path.as_path()).map_err(UserError::from_error)?;

    let enc = GzEncoder::new(tar_gz, Compression::default());
    let mut tar = tar::Builder::new(enc);
    tar.append_dir_all(".", engine_dir.path())
        .map_err(UserError::from_error)?;

    tar.finish().map_err(UserError::from_error)?;

    Ok(tarball)
}

#[utoipa::path(get, path="/request_limits", responses(
    (status = 200, description = "Get the limits on requests", body = RequestLimitsResponse)
))]
async fn request_limits(
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    Ok(jsongz(
        &RequestLimitsResponse {
            limits: v1::request_limits(),
        },
        accept_encoding.as_ref(),
    ))
}

#[utoipa::path(get, path="/version", responses(
    (status = 200, description = "Version of the server", body = VersionResponse)
))]
async fn version() -> Result<impl warp::Reply, Rejection> {
    let r = VersionResponse {
        version: String::from(crate::CRATE_VERSION),
    };
    Ok(warp::reply::json(&r))
}

#[utoipa::path(get, path="/nstates", responses(
    (status = 200, description = "The number of states in the current model", body = NStatesResponse)
))]
async fn nstates(
    state: State,
    _accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    let n_states = state.engine.read().await.n_states();
    let r = NStatesResponse { nstates: n_states };
    Ok(jsongz(&r, None))
}

#[utoipa::path(get, path="/shape", responses(
    (status = 200, description = "The shape of the current dataframe", body = ShapeResponse)
))]
async fn shape(
    state: State,
    _accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    let engine = state.engine.read().await;

    let n_rows = engine.n_rows();
    let n_cols = engine.n_cols();

    let r = ShapeResponse { n_rows, n_cols };
    Ok(jsongz(&r, None))
}

#[utoipa::path(get, path="/codebook", responses(
    (status = 200, description = "The codebook for the current model", body = CodebookResponse)
))]
async fn codebook(
    state: State,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    let codebook = state.engine.read().await.codebook.clone();
    let r = CodebookResponse { codebook };
    Ok(jsongz(&r, accept_encoding.as_ref()))
}

#[utoipa::path(get, path="/ftype/{col_ix}", responses(
    (status = 200, description = "The type for a given column", body = FTypeResponse),
    (status = 400, description = "The requested column does not exist", body = UserError),
))]
async fn ftype(
    col_ix: usize,
    state: State,
    _accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    state
        .engine
        .read()
        .await
        .ftype(col_ix)
        .map_err(|e| Error::User(UserError::from_error(e)))
        .map(|ftype| jsongz(&FTypeResponse { ftype }, None))
        .map_err(warp::reject::custom)
}

#[utoipa::path(get, path="/assignments/{state_ix}", responses(
    (status = 200, description = "Assignments for a state", body = AssignmentResponse),
    (status = 400, description = "The requested state does not exist", body = UserError),
))]
pub async fn assignments(
    state_ix: usize,
    state: State,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    state
        .engine
        .read()
        .await
        .states
        .get(state_ix)
        .ok_or_else(|| {
            Error::User(UserError(format!("State {} does not exist", state_ix)))
        })
        .map(|state| {
            let column_assignment = state.asgn.asgn.clone();
            let row_assignments = state
                .views
                .iter()
                .map(|view| view.asgn.asgn.clone())
                .collect();

            jsongz(
                &AssignmentResponse {
                    column_assignment,
                    row_assignments,
                },
                accept_encoding.as_ref(),
            )
        })
        .map_err(warp::reject::custom)
}

#[utoipa::path(get, path="/ftypes", responses(
    (status = 200, description = "FTypes for each column.", body = FTypesResponse),
))]
pub async fn ftypes(
    state: State,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    let ftypes = state.engine.read().await.ftypes();
    let r = FTypesResponse { ftypes };

    let accept_encoding = no_zip_if_output_lt!(r, ftypes, 5, accept_encoding);
    Ok(jsongz(&r, accept_encoding.as_ref()))
}

#[utoipa::path(post, path="/depprob", request_body=DepprobRequest, responses(
    (status = 200, description = "FTypes for each column.", body = DepprobResponse),
    (status = 400, description = "The request is not compatable with the given model", body = UserError),
))]
pub async fn depprob(
    state: State,
    mut req: DepprobRequest,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    check_too_long!(req)?;
    let engine = state.engine.read().await;

    let accept_encoding =
        no_zip_if_output_lt!(req, col_pairs, 10, accept_encoding);

    engine
        .depprob_pw(&req.col_pairs)
        .map_err(|e| Error::User(UserError::from_error(e)))
        .map(|depprobs| {
            let depprob = depprobs
                .iter()
                .zip(req.col_pairs.drain(..))
                .map(|(depprob, (col_a, col_b))| (col_a, col_b, *depprob))
                .collect();
            DepprobResponse { depprob }
        })
        .map(|res| jsongz(&res, accept_encoding.as_ref()))
        .map_err(warp::reject::custom)
}

#[utoipa::path(post, path="/rowsim", request_body=RowsimRequest, responses(
    (status = 200, description = "Row simularity for the given request.", body = RowsimResponse),
    (status = 400, description = "The request is not compatable with the given model", body = UserError),
))]
pub async fn rowsim(
    state: State,
    mut req: RowsimRequest,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    check_too_long!(req)?;

    let wrt_opt = if req.wrt.is_empty() {
        None
    } else {
        Some(req.wrt.as_slice())
    };
    let engine = state.engine.read().await;

    let rowsim_variant = if req.col_weighted {
        braid::RowSimilarityVariant::ColumnWeighted
    } else {
        braid::RowSimilarityVariant::ViewWeighted
    };

    engine
        .rowsim_pw(&req.row_pairs, wrt_opt, rowsim_variant)
        .map_err(|e| Error::User(UserError::from_error(e)))
        .map(|sims| {
            let rowsim = sims
                .iter()
                .zip(req.row_pairs.drain(..))
                .map(|(rowsim, (row_a, row_b))| (row_a, row_b, *rowsim))
                .collect();
            RowsimResponse { rowsim }
        })
        .map(|res| jsongz(&res, accept_encoding.as_ref()))
        .map_err(warp::reject::custom)
}

#[utoipa::path(post, path="/novelty", request_body=NoveltyRequest, responses(
    (status = 200, description = "Novelty for the given request.", body = NoveltyResponse),
    (status = 400, description = "The request is not compatable with the given model", body = UserError),
))]
pub async fn novelty(
    state: State,
    mut req: NoveltyRequest,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    check_too_long!(req)?;
    let wrt_opt = if req.wrt.is_empty() {
        None
    } else {
        Some(req.wrt.as_slice())
    };

    let engine = state.engine.read().await;

    let novelty_res: Result<Vec<(NameOrIndex, f64)>, _> = req
        .row_ixs
        .drain(..)
        .map(|row_ix| {
            engine
                .novelty(&row_ix, wrt_opt)
                .map_err(|e| Error::User(UserError::from_error(e)))
                .map(|novelty| (row_ix, novelty))
        })
        .collect();

    let accept_encoding =
        no_zip_if_output_lt!(req, row_ixs, 10, accept_encoding);

    novelty_res
        .map(|novelty| NoveltyResponse { novelty })
        .map(|res| jsongz(&res, accept_encoding.as_ref()))
        .map_err(warp::reject::custom)
}

#[utoipa::path(post, path="/novelty", request_body=MiRequest, responses(
    (status = 200, description = "Mutual Information for the given request.", body = MiResponse),
    (status = 400, description = "The request is not compatable with the given model", body = UserError),
))]
pub async fn mi(
    state: State,
    mut req: MiRequest,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    check_too_long!(req)?;
    let engine = state.engine.read().await;

    let accept_encoding =
        no_zip_if_output_lt!(req, col_pairs, 10, accept_encoding);

    engine
        .mi_pw(&req.col_pairs, req.n, req.mi_type)
        .map_err(|e| Error::User(UserError::from_error(e)))
        .map(|mis| {
            let mi = mis
                .iter()
                .zip(req.col_pairs.drain(..))
                .map(|(mi, (col_a, col_b))| (col_a, col_b, *mi))
                .collect();
            MiResponse { mi }
        })
        .map(|res| jsongz(&res, accept_encoding.as_ref()))
        .map_err(warp::reject::custom)
}

#[utoipa::path(post, path="/entropy", request_body=EntropyRequest, responses(
    (status = 200, description = "Entropy for the given request.", body = EntropyResponse),
    (status = 400, description = "The request is not compatable with the given model", body = UserError),
))]
pub async fn entropy(
    state: State,
    req: EntropyRequest,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    check_too_long!(req)?;
    let engine = state.engine.read().await;

    engine
        .entropy(&req.col_ixs, req.n)
        .map_err(|e| Error::User(UserError::from_error(e)))
        .map(|entropy| EntropyResponse { entropy })
        .map(|res| jsongz(&res, accept_encoding.as_ref()))
        .map_err(warp::reject::custom)
}

#[utoipa::path(post, path="/info_prop", request_body=InfoPropRequest, responses(
    (status = 200, description = "Information Proportion for the given request.", body = InfoPropResponse),
    (status = 400, description = "The request is not compatable with the given model", body = UserError),
))]
pub async fn info_prop(
    state: State,
    req: InfoPropRequest,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    check_too_long!(req)?;
    let engine = state.engine.read().await;

    engine
        .info_prop(&req.target_ixs, &req.predictor_ixs, req.n)
        .map_err(|e| Error::User(UserError::from_error(e)))
        .map(|info_prop| InfoPropResponse { info_prop })
        .map(|res| jsongz(&res, accept_encoding.as_ref()))
        .map_err(warp::reject::custom)
}

#[utoipa::path(post, path="/simulate", request_body=SimulateRequest, responses(
    (status = 200, description = "Simulation Result", body = SimulateResponse),
    (status = 400, description = "The request is not compatable with the given model", body = UserError),
))]
pub async fn simulate(
    state: State,
    req: SimulateRequest,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    check_too_long!(req)?;
    let engine = state.engine.read().await;

    let mut rng = rand::thread_rng();
    engine
        .simulate(&req.col_ixs, &req.given, req.n, None, &mut rng)
        .map_err(|e| Error::User(UserError::from_error(e)))
        .map(|mut values| SimulateResponse {
            values: {
                values
                    .drain(..)
                    .map(|mut xs| xs.drain(..).map(Datum::from).collect())
                    .collect()
            },
        })
        .map(|res| jsongz(&res, accept_encoding.as_ref()))
        .map_err(warp::reject::custom)
}

#[utoipa::path(post, path="/draw", request_body=DrawRequest, responses(
    (status = 200, description = "Draws from the posterior.", body = DrawResponse),
    (status = 400, description = "The request is not compatable with the given model", body = UserError),
))]
pub async fn draw(
    state: State,
    req: DrawRequest,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    check_too_long!(req)?;
    let engine = state.engine.read().await;

    let mut rng = rand::thread_rng();
    engine
        .draw(req.row_ix, req.col_ix, req.n, &mut rng)
        .map_err(|e| Error::User(UserError::from_error(e)))
        .map_err(warp::reject::custom)
        .map(|mut values| DrawResponse {
            values: { values.drain(..).map(Datum::from).collect() },
        })
        .map(|res| jsongz(&res, accept_encoding.as_ref()))
}

#[utoipa::path(post, path="/logp", request_body=LogpRequest, responses(
    (status = 200, description = "Log Probability", body = LogpResponse),
    (status = 400, description = "The request is not compatable with the given model", body = UserError),
))]
pub async fn logp(
    state: State,
    mut req: LogpRequest,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    check_too_long!(req)?;
    let engine = state.engine.read().await;

    let values: Vec<Vec<braid::Datum>> = req
        .values
        .drain(..)
        .map(|mut xs| xs.drain(..).map(braid::Datum::from).collect())
        .collect();

    engine
        .logp(&req.col_ixs, &values, &req.given, req.state_ixs.as_deref())
        .map_err(|e| Error::User(UserError::from_error(e)))
        .map_err(warp::reject::custom)
        .map(|logp| LogpResponse { logp })
        .map(|res| jsongz(&res, accept_encoding.as_ref()))
}

#[utoipa::path(post, path="/logp_scaled", request_body=LogpScaledRequest, responses(
    (status = 200, description = "Scaled Log Probability", body = LogpScaledResponse),
    (status = 400, description = "The request is not compatable with the given model", body = UserError),
))]
pub async fn logp_scaled(
    state: State,
    mut req: LogpScaledRequest,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    check_too_long!(req)?;

    let engine = state.engine.read().await;

    let values: Vec<Vec<braid::Datum>> = req
        .values
        .drain(..)
        .map(|mut xs| xs.drain(..).map(braid::Datum::from).collect())
        .collect();

    engine
        .logp_scaled(
            &req.col_ixs,
            &values,
            &req.given,
            req.state_ixs.as_deref(),
            None,
        )
        .map(|logp| LogpScaledResponse { logp })
        .map(|res| jsongz(&res, accept_encoding.as_ref()))
        .map_err(|e| Error::User(UserError::from_error(e)))
        .map_err(warp::reject::custom)
}

#[utoipa::path(post, path="/surprisal", request_body=SurprisalRequest, responses(
    (status = 200, description = "Surprisal", body = SurprisalResponse),
    (status = 400, description = "The request is not compatable with the given model", body = UserError),
))]
pub async fn surprisal(
    state: State,
    req: SurprisalRequest,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    let col_ix = req.col_ix;
    let row_ixs = req.row_ixs;
    let engine = state.engine.read().await;

    if req.target_data.is_some()
        && req.target_data.as_ref().unwrap().len() != row_ixs.len()
    {
        return Err(Error::User(UserError(
            "Number of data must match the number of rows".to_owned(),
        ))
        .into());
    }

    let surprisal_res: Result<Vec<Option<f64>>, _> =
        match req.target_data.as_ref() {
            None => row_ixs
                .iter()
                .map(|row_ix| {
                    engine
                        .self_surprisal(row_ix, &col_ix, req.state_ixs.clone())
                        .map_err(UserError::from_error)
                })
                .collect(),
            Some(data) => row_ixs
                .iter()
                .zip(data.iter())
                .map(|(row_ix, datum)| {
                    engine
                        .surprisal(
                            &(datum.clone()),
                            row_ix,
                            &col_ix,
                            req.state_ixs.clone(),
                        )
                        .map_err(UserError::from_error)
                })
                .collect(),
        };

    surprisal_res
        .and_then(|surprisal| {
            if let Some(values) = req.target_data {
                let resp = SurprisalResponse {
                    col_ix,
                    row_ixs,
                    values,
                    surprisal,
                };
                Ok(resp)
            } else {
                let values_res: Result<Vec<Datum>, _> = row_ixs
                    .iter()
                    .map(|row_ix| engine.datum(row_ix, &col_ix))
                    .collect();

                values_res
                    .map(|mut values| SurprisalResponse {
                        col_ix,
                        row_ixs,
                        values: values.drain(..).collect(),
                        surprisal,
                    })
                    .map_err(UserError::from_error)
            }
        })
        .map_err(Error::from)
        .map_err(warp::reject::custom)
        .map(|res| jsongz(&res, accept_encoding.as_ref()))
}

#[utoipa::path(get, path="/diagnostics", responses(
    (status = 200, description = "Diagnostics", body = StateDiagnosticsResponse),
))]
pub async fn diagnostics(
    state: State,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    let engine = state.engine.read().await;
    let mut diagnostics = engine.state_diagnostics();
    let resp = StateDiagnosticsResponse {
        diagnostics: diagnostics.drain(..).collect(),
    };
    Ok(jsongz(&resp, accept_encoding.as_ref()))
}

#[utoipa::path(post, path="/impute", request_body=ImputeRequest, responses(
    (status = 200, description = "Imputed response", body = ImputeResponse),
    (status = 400, description = "The request is not compatable with the given model", body = UserError),
))]
pub async fn impute(
    state: State,
    req: ImputeRequest,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    let engine = state.engine.read().await;

    let row_ix = req.row_ix;
    let col_ix = req.col_ix;

    let unc_type = req.uncertainty_type;

    engine
        .impute(&row_ix, &col_ix, unc_type)
        .map_err(|e| Error::User(UserError::from_error(e)))
        .map_err(warp::reject::custom)
        .map(|(value, unc)| ImputeResponse {
            row_ix,
            col_ix,
            value,
            uncertainty: unc,
        })
        .map(|res| jsongz(&res, accept_encoding.as_ref()))
}

#[utoipa::path(post, path="/predict", request_body=PredictRequest, responses(
    (status = 200, description = "Predict response", body = PredictResponse),
    (status = 400, description = "The request is not compatable with the given model", body = UserError),
))]
pub async fn predict(
    state: State,
    req: PredictRequest,
    _accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    let engine = state.engine.read().await;

    let col_ix = req.col_ix;
    let unc_type = req.uncertainty_type;

    engine
        .predict(
            col_ix,
            &req.given.clone(),
            unc_type,
            req.state_ixs.as_deref(),
        )
        .map(|(value, uncertainty)| PredictResponse { value, uncertainty })
        .map(|res| jsongz(&res, None))
        .map_err(|e| Error::User(UserError::from_error(e)))
        .map_err(warp::reject::custom)
}

#[utoipa::path(post, path="/get_data", request_body=GetDataRequest, responses(
    (status = 200, description = "Data within the datatable", body = GetDataResponse),
    (status = 400, description = "The request is not compatable with the given model", body = UserError),
))]
pub async fn get_data(
    state: State,
    req: GetDataRequest,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    check_too_long!(req)?;
    let engine = state.engine.read().await;

    let accept_encoding = no_zip_if_output_lt!(req, ixs, 10, accept_encoding);

    let values_res: Result<Vec<(usize, usize, Datum)>, _> = req
        .ixs
        .iter()
        .map(|(row_ix, col_ix)| {
            engine
                .datum(*row_ix, *col_ix)
                .map(|datum| (*row_ix, *col_ix, datum))
        })
        .collect();

    values_res
        .map(|mut values| GetDataResponse {
            values: values
                .drain(..)
                .map(|(row_ix, col_ix, x)| (row_ix, col_ix, x))
                .collect(),
        })
        .map(|res| jsongz(&res, accept_encoding.as_ref()))
        .map_err(|e| Error::User(UserError::from_error(e)))
        .map_err(warp::reject::custom)
}

#[utoipa::path(post, path="/summary", request_body=SummarizeColumnsRequest, responses(
    (status = 200, description = "Data within the datatable", body = SummarizeColumnsResponse),
    (status = 400, description = "The request is not compatable with the given model", body = UserError),
))]
pub async fn summary(
    state: State,
    mut req: SummarizeColumnsRequest,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    check_too_long!(req)?;

    let engine = state.engine.read().await;

    let summaries_res: Result<Vec<_>, _> = req
        .col_ixs
        .drain(..)
        .map(|col_ix| {
            engine
                .summarize_col(&col_ix)
                .map(|summary| (col_ix, summary))
        })
        .collect();

    summaries_res
        .map(|summaries| SummarizeColumnsResponse { summaries })
        .map(|res| jsongz(&res, accept_encoding.as_ref()))
        .map_err(|e| Error::User(UserError::from_error(e)))
        .map_err(warp::reject::custom)
}

#[utoipa::path(post, path="/insert", request_body=InsertDataRequest, responses(
    (status = 200, description = "Result of inserting data", body = InsertDataResponse),
    (status = 400, description = "The request is not compatable with the given model", body = UserError),
))]
pub async fn insert_data(
    state: State,
    mut req: InsertDataRequest,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    if state.mutability.is_immutable() {
        return Err(Error::User(UserError::from(
            "Called insert_data on immutable server",
        ))
        .into());
    }

    let write_mode = req.write_mode.unwrap_or_default();

    // TODO: Allow support extension
    let res = state.engine.write().await.insert_data(
        req.rows.drain(..).map(braid::Row::from).collect(),
        req.new_col_metadata,
        None,
        write_mode,
    );

    res.map(|r| InsertDataResponse {
        new_rows: r.new_rows().map_or(0, |r| r.len()),
        new_cols: r.new_cols().map_or(0, |c| c.len()),
    })
    .map(|res| jsongz(&res, accept_encoding.as_ref()))
    .map_err(|e| Error::User(UserError::from_error(e)))
    .map_err(warp::reject::custom)
}

#[utoipa::path(post, path="/update", request_body=UpdateRequest, responses(
    (status = 200, description = "Nothing is returned if successful"),
    (status = 400, description = "The request is not compatable with the given model", body = UserError),
))]
pub async fn update(
    state: State,
    req: UpdateRequest,
    _accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    if state.mutability.is_immutable() {
        return Err(Error::User(UserError::from(
            "Called update on immutable server",
        ))
        .into());
    }

    let config = braid::EngineUpdateConfig::from(req);
    state
        .engine
        .write()
        .await
        .update(config, None, None)
        .expect("update failed");

    Ok(jsongz(&(), None))
}

#[cfg(not(feature = "download"))]
pub async fn download(_state: State) -> Result<String, Rejection> {
    Err(warp::reject::not_found())
}

#[cfg(feature = "download")]
#[utoipa::path(get, path="/download", responses(
    (status = 200, description = "A Tarball of the model's state"),
    (status = 400, description = "Failed to create the tarball", body = UserError),
))]
pub async fn download(state: State) -> Result<impl warp::Reply, Rejection> {
    let engine = state.engine.read_owned().await;
    let temp_tarball =
        tokio::task::spawn_blocking(move || save_metadata(&engine))
            .await
            .map_err(Error::from)?
            .map_err(Error::from)?;
    let stream = ReaderStream::new(
        tokio::fs::File::open(temp_tarball.path)
            .await
            .map_err(Error::from)?,
    );

    warp::hyper::Response::builder()
        .header(CONTENT_TYPE, "application/gzip")
        .body(warp::hyper::body::Body::wrap_stream(stream))
        .map_err(|e| Error::Internal(InternalError(e.to_string())))
        .map_err(warp::reject::custom)
}

#[utoipa::path(get, path="/feature_error/{col_ix}", responses(
    (status = 200, description = "Error relative to a feature", body = FeatureErrorResponse),
    (status = 400, description = "The privided column index does not exist", body = UserError),
))]
pub async fn feature_error(
    state: State,
    col_ix: usize,
    _accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    state
        .engine
        .read()
        .await
        .feature_error(col_ix)
        .map(|(error, centroid)| FeatureErrorResponse { error, centroid })
        .map(|res| jsongz(&res, None))
        .map_err(|e| Error::User(UserError::from_error(e)))
        .map_err(warp::reject::custom)
}

#[utoipa::path(get, path="/csv", responses(
    (status = 200, description = "Error relative to a feature", body = FeatureErrorResponse),
))]
pub async fn csv(
    state: State,
    accept_encoding: Option<String>,
) -> Result<impl warp::Reply, Rejection> {
    use async_compression::tokio::bufread::GzipEncoder;
    use async_compression::Level;
    use braid::codebook::ColType;
    use braid::HasData;

    let stream = async_stream::stream! {
        let engine = state
            .engine
            .read()
            .await;

        let codebook = &engine.codebook;
        let n_rows = engine.n_rows();

        let header_iter: std::iter::Once<Vec<u8>> = std::iter::once({
            let record = std::iter::once(String::from("ID"))
                .chain(codebook.col_metadata.iter().map(|md| md.name.clone()))
                .collect::<Vec<String>>();

            // NOTE: csv::StringRecord::as_slice does not return properly
            // encoded values
            let mut buf = Vec::<u8>::new();
            {
                let mut writer = csv::Writer::from_writer(&mut buf);
                writer.write_record(record)
                    .expect("Should be able to write to csv::Writer");
            }
            buf
        });

        let body_iter = codebook.row_names.iter().enumerate().map(|(row_ix, (row_name, _))| {
            let record = std::iter::once(row_name.to_owned())
                .chain({
                    codebook.col_metadata.iter().enumerate().map(|(col_ix, colmd)| {
                        let datum = engine.cell(row_ix, col_ix);
                        match datum {
                            Datum::Continuous(x) => x.to_string(),
                            Datum::Categorical(x) => {
                                match colmd.coltype {
                                    ColType::Categorical { value_map: None , .. } => x.to_string(),
                                    ColType::Categorical { value_map: Some(ref value_map) , .. } => {
                                        value_map[&usize::from(x)].to_owned()
                                    }
                                    _ => panic!("Expeted categorical column"),
                                }
                            }
                            Datum::Missing => String::from(""),
                            Datum::Count(x) => x.to_string(),
                            Datum::Label(_) => panic!("Label not supported"),
                        }
                    })
                })
                .collect::<Vec<String>>();

            let mut buf = Vec::<u8>::new();
            {
                let mut writer = csv::Writer::from_writer(&mut buf);
                writer.write_record(record)
                    .expect("Should be able to write to csv::Writer");
            }

            // remove the final newline
            if row_ix == n_rows - 1 {
                let _last = buf.pop();
            }

            buf
        });

        let csv_iter = header_iter.chain(body_iter);

        use bytes::Bytes;
        for item in csv_iter {
            let res: Result<Bytes, std::io::Error> = Ok(Bytes::from(item));
            yield res
        }
    };

    // We need to convert a stream into a async reader for gzip compression,
    // then we need to convert the encoder back from an async reader to a normal
    // stream for warp. Yikes.
    let body = if gzip_accepted(accept_encoding.as_ref()) {
        let encoder = GzipEncoder::with_quality(
            tokio_util::io::StreamReader::new(stream),
            Level::Fastest,
        );
        let reader_stream = tokio_util::io::ReaderStream::new(encoder);
        hyper::body::Body::wrap_stream(reader_stream)
    } else {
        hyper::body::Body::wrap_stream(stream)
    };

    Ok(warp::http::Response::builder()
        .header("Content-Type", "text/csv")
        .header("Content-Encoding", "gzip")
        .body(body))
}

/// Server State
#[derive(Debug, Clone)]
pub struct State {
    engine: Arc<tokio::sync::RwLock<Engine>>,
    mutability: ServerMutability,
}

impl State {
    pub fn load(
        path: &Path,
        encryption_key: Option<String>,
        mutable: bool,
    ) -> Self {
        use std::str::FromStr;

        let user_info = UserInfo {
            encryption_key: encryption_key
                .map(|s| braid::metadata::EncryptionKey::from_str(&s).unwrap()),
            profile: None,
        };

        let engine = Engine::load(
            Path::new(path),
            user_info.encryption_key().unwrap().as_ref(),
        )
        .unwrap();
        Self {
            engine: Arc::new(tokio::sync::RwLock::new(engine)),
            mutability: ServerMutability::from_mutable_bool(mutable),
        }
    }
}

/// Build a Warp filter
///
/// # Arguments
/// - path: string path to the braid metadata
/// - encryption_key: optional 32-byte hex private key for use with encrypted
///   metadata
/// - mutable: choose if the engine is mutible or not.
/// - json_limit: Limit in bytes of uploaded json content.
pub fn warp(
    path: &Path,
    mutable: bool,
    encryption_key: Option<String>,
    json_limit: u64,
) -> impl warp::Filter<Extract = impl warp::Reply, Error = Infallible> + Clone {
    let state = State::load(path, encryption_key, mutable);

    macro_rules! post_handler {
        ($name:literal, $handler:ident) => {
            warp::path($name)
                .and(warp::post())
                .and(with(state.clone()))
                .and(warp::filters::body::content_length_limit(json_limit))
                .and(warp::body::json())
                .and(warp::header::optional::<String>(ACCEPT_ENCODING.as_str()))
                .and_then($handler)
        };
        ($handler:ident) => {{
            warp::path(stringify!($handler))
                .and(warp::post())
                .and(with(state.clone()))
                .and(warp::filters::body::content_length_limit(json_limit))
                .and(warp::body::json())
                .and(warp::header::optional::<String>(ACCEPT_ENCODING.as_str()))
                .and_then($handler)
        }};
    }

    macro_rules! get_handler {
        ($name:literal, $handler:ident) => {
            warp::path($name)
                .and(warp::get())
                .and(with(state.clone()))
                .and(warp::header::optional::<String>(ACCEPT_ENCODING.as_str()))
                .and_then($handler)
        };
        ($handler:ident) => {{
            warp::path(stringify!($handler))
                .and(warp::get())
                .and(with(state.clone()))
                .and(warp::header::optional::<String>(ACCEPT_ENCODING.as_str()))
                .and_then($handler)
        }};
    }

    warp::path("api")
        .and(warp::path("v1"))
        .and(compose!(
            warp::path!("version").and(warp::get()).and_then(version),
            warp::path!("request_limits")
                .and(warp::get())
                .and(warp::header::optional::<String>(ACCEPT_ENCODING.as_str()))
                .and_then(request_limits),
            get_handler!(shape),
            get_handler!(ftypes),
            get_handler!(nstates),
            get_handler!(codebook),
            get_handler!(diagnostics),
            get_handler!(csv),
            post_handler!(depprob),
            post_handler!(rowsim),
            post_handler!(mi),
            post_handler!(surprisal),
            post_handler!(simulate),
            post_handler!(draw),
            post_handler!(logp),
            post_handler!(logp_scaled),
            post_handler!(impute),
            post_handler!(predict),
            post_handler!(get_data),
            post_handler!(summary),
            post_handler!(novelty),
            post_handler!(info_prop),
            post_handler!(entropy),
            post_handler!("insert", insert_data),
            post_handler!(update),
            post_handler!(feature_error),
            warp::path!("download")
                .and(warp::get())
                .and(with(state.clone()))
                .and_then(download),
            warp::path!("assignments" / usize)
                .and(warp::get())
                .and(with(state.clone()))
                .and(warp::header::optional::<String>(ACCEPT_ENCODING.as_str()))
                .and_then(assignments),
            warp::path!("ftype" / usize)
                .and(warp::get())
                .and(with(state))
                .and(warp::header::optional::<String>(ACCEPT_ENCODING.as_str()))
                .and_then(ftype),
        ))
        .recover(error_handler)
}

async fn error_handler(
    error: Rejection,
) -> Result<warp::reply::WithStatus<JsonGz>, Infallible> {
    let code;
    let message: String;

    if error.is_not_found() {
        code = StatusCode::NOT_FOUND;
        message = "Not Found".to_string();
    } else if let Some(error) = error.find::<Error>() {
        match error {
            Error::User(user_error) => {
                code = StatusCode::BAD_REQUEST;
                message = user_error.0.clone();
            }
            Error::Internal(internal_error) => {
                code = StatusCode::INTERNAL_SERVER_ERROR;
                message = internal_error.0.clone();
            }
        }
    } else {
        code = StatusCode::INTERNAL_SERVER_ERROR;
        message = "UNHANDLED_REJECTION".to_string();
        dbg!(error);
    }

    #[derive(Serialize)]
    struct ErrorMessage {
        error: String,
    }
    let json = jsongz(&ErrorMessage { error: message }, None);

    Ok(warp::reply::with_status(json, code))
}
