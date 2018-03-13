extern crate rand;

use std::io;
use interface::oracle::{MiType, Oracle};
use interface::server::{utils, validate};
use interface::server::validate::Dim;
use cc::{Codebook, DType, FType};
use cc::state::StateDiagnostics;

#[derive(Deserialize, Debug)]
pub struct NoReq {}

// Number of states
// ----------------
#[derive(Serialize, Debug)]
pub struct ShapeResp {
    rows: usize,
    cols: usize,
}

pub fn shape_req(oracle: &Oracle, _req: &NoReq) -> io::Result<String> {
    let resp = ShapeResp {
        rows: oracle.nrows(),
        cols: oracle.ncols(),
    };
    utils::serialize_resp(&resp)
}

// Number of states
// ----------------
#[derive(Serialize, Debug)]
pub struct NStatesResp {
    nstates: usize,
}

pub fn nstates_req(oracle: &Oracle, _req: &NoReq) -> io::Result<String> {
    let resp = NStatesResp {
        nstates: oracle.nstates(),
    };
    utils::serialize_resp(&resp)
}

// Get data types
// --------------
#[derive(Serialize, Debug)]
pub struct CodebookResp {
    codebook: Codebook,
}

pub fn codebook_req(oracle: &Oracle, _req: &NoReq) -> io::Result<String> {
    let codebook = oracle.codebook.clone();
    let resp = CodebookResp { codebook: codebook };
    utils::serialize_resp(&resp)
}

// Get data types
// --------------
#[derive(Serialize, Debug)]
pub struct FTypesResp {
    ftypes: Vec<FType>,
}

pub fn ftypes_req(oracle: &Oracle, _req: &NoReq) -> io::Result<String> {
    let ftypes = oracle.ftypes();
    let resp = FTypesResp { ftypes: ftypes };
    utils::serialize_resp(&resp)
}

// Dependence probability
// ----------------------
#[derive(Deserialize, Debug)]
pub struct DepprobReq {
    pub col_a: usize,
    pub col_b: usize,
}

#[derive(Serialize, Debug)]
pub struct DepprobResp {
    col_a: usize,
    col_b: usize,
    depprob: f64,
}

pub fn depprob_req(oracle: &Oracle, req: &DepprobReq) -> io::Result<String> {
    let col_a = req.col_a;
    let col_b = req.col_b;

    validate::validate_ix(col_a, oracle.ncols(), Dim::Columns)?;
    validate::validate_ix(col_b, oracle.ncols(), Dim::Columns)?;

    let ans = oracle.depprob(col_a, col_b);
    let resp = DepprobResp {
        col_a: col_a,
        col_b: col_b,
        depprob: ans,
    };
    utils::serialize_resp(&resp)
}

// Row similarity
// --------------
#[derive(Deserialize, Debug)]
pub struct RowsimReq {
    pub row_a: usize,
    pub row_b: usize,
    pub wrt: Vec<usize>,
}

#[derive(Serialize)]
pub struct RowsimResp {
    row_a: usize,
    row_b: usize,
    rowsim: f64,
}

pub fn rowsim_req(oracle: &Oracle, req: &RowsimReq) -> io::Result<String> {
    let row_a = req.row_a;
    let row_b = req.row_b;
    let wrt_opt = if req.wrt.is_empty() {
        None
    } else {
        Some(&req.wrt)
    };

    validate::validate_ix(row_a, oracle.nrows(), Dim::Rows)?;
    validate::validate_ix(row_b, oracle.nrows(), Dim::Rows)?;
    validate::validate_wrt(&wrt_opt, oracle.ncols())?;

    let ans = oracle.rowsim(row_a, row_b, wrt_opt);
    let resp = RowsimResp {
        row_a: row_a,
        row_b: row_b,
        rowsim: ans,
    };
    utils::serialize_resp(&resp)
}

// Mutual information (MI)
// -----------------------
#[derive(Deserialize, Debug)]
pub struct MiReq {
    pub col_a: usize,
    pub col_b: usize,
    pub n: usize,
    pub mi_type: MiType,
}

#[derive(Serialize)]
pub struct MiResp {
    col_a: usize,
    col_b: usize,
    mi: f64,
}

pub fn mi_req(oracle: &Oracle, req: &MiReq) -> io::Result<String> {
    let col_a = req.col_a;
    let col_b = req.col_b;
    let mi_type = req.mi_type.clone();

    validate::validate_ix(col_a, oracle.ncols(), Dim::Columns)?;
    validate::validate_ix(col_b, oracle.ncols(), Dim::Columns)?;

    let mut rng = rand::thread_rng();
    let ans = oracle.mi(col_a, col_b, req.n, mi_type, &mut rng);
    let resp = MiResp {
        col_a: col_a,
        col_b: col_b,
        mi: ans,
    };
    utils::serialize_resp(&resp)
}

// Simulate
// ---------
#[derive(Deserialize, Debug)]
pub struct SimulateReq {
    pub col_ixs: Vec<usize>,
    pub given: Vec<(usize, DType)>,
    pub n: usize,
}

#[derive(Serialize)]
pub struct SimulateResp {
    values: Vec<Vec<DType>>,
}

pub fn simulate_req(oracle: &Oracle, req: &SimulateReq) -> io::Result<String> {
    let given_opt = if req.given.is_empty() {
        None
    } else {
        Some(req.given.clone())
    };

    validate::validate_ixs(&req.col_ixs, oracle.ncols(), Dim::Columns)?;
    validate::validate_given(&oracle, &given_opt)?;

    let mut rng = rand::thread_rng();
    let values = oracle.simulate(&req.col_ixs, &given_opt, req.n, &mut rng);
    let resp = SimulateResp { values: values };
    utils::serialize_resp(&resp)
}

// logp
// ----
#[derive(Deserialize, Debug)]
pub struct LogpReq {
    pub col_ixs: Vec<usize>,
    pub values: Vec<Vec<DType>>,
    pub given: Vec<(usize, DType)>,
}

#[derive(Serialize)]
pub struct LogpResp {
    logp: Vec<f64>,
}

pub fn logp_req(oracle: &Oracle, req: &LogpReq) -> io::Result<String> {
    let given_opt = if req.given.is_empty() {
        None
    } else {
        Some(req.given.clone())
    };

    validate::validate_ixs(&req.col_ixs, oracle.ncols(), Dim::Columns)?;
    validate::validate_given(&oracle, &given_opt)?;

    let logp = oracle.logp(&req.col_ixs, &req.values, &given_opt);
    let resp = LogpResp { logp: logp };
    utils::serialize_resp(&resp)
}

// Surprisal
// ---------
#[derive(Deserialize, Debug)]
pub struct SurprisalReq {
    pub col_ix: usize,
    pub row_ix: usize,
}

#[derive(Serialize)]
pub struct SurprisalResp {
    col_ix: usize,
    row_ix: usize,
    value: DType,
    surprisal: Option<f64>,
}

pub fn surprisal_req(
    oracle: &Oracle,
    req: &SurprisalReq,
) -> io::Result<String> {
    let col_ix = req.col_ix;
    let row_ix = req.row_ix;
    validate::validate_ixs(&vec![col_ix], oracle.ncols(), Dim::Columns)?;
    validate::validate_ixs(&vec![row_ix], oracle.nrows(), Dim::Rows)?;

    let surpisal_opt = oracle.self_surprisal(row_ix, col_ix);
    let value = oracle.get_datum(row_ix, col_ix);
    let resp = SurprisalResp {
        col_ix: col_ix,
        row_ix: row_ix,
        value: value,
        surprisal: surpisal_opt,
    };
    utils::serialize_resp(&resp)
}

// Diagnostics
// -----------
#[derive(Deserialize, Debug)]
pub struct DiagnosticsReq {}

#[derive(Serialize)]
pub struct DiagnosticsResp {
    diagnostics: Vec<StateDiagnostics>,
}

pub fn diagnostics_req(
    oracle: &Oracle,
    _req: &DiagnosticsReq,
) -> io::Result<String> {
    let diagnostics = oracle.state_diagnostics();
    let resp = DiagnosticsResp {
        diagnostics: diagnostics,
    };
    utils::serialize_resp(&resp)
}

// Impute
// ------
#[derive(Deserialize, Debug)]
pub struct ImputeReq {
    pub row_ix: usize,
    pub col_ix: usize,
    pub with_uncertainty: bool,
}

#[derive(Serialize)]
pub struct ImputeResp {
    row_ix: usize,
    col_ix: usize,
    value: DType,
    uncertainty: f64,
}

pub fn impute_req(oracle: &Oracle, req: &ImputeReq) -> io::Result<String> {
    validate::validate_ix(req.row_ix, oracle.nrows(), Dim::Rows)?;
    validate::validate_ix(req.col_ix, oracle.ncols(), Dim::Columns)?;
    let row_ix = req.row_ix;
    let col_ix = req.col_ix;
    let (value, unc) = oracle.impute(row_ix, col_ix, req.with_uncertainty);
    let resp = ImputeResp {
        row_ix: row_ix,
        col_ix: col_ix,
        value: value,
        uncertainty: unc,
    };
    utils::serialize_resp(&resp)
}
