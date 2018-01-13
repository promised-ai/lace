extern crate rand;
extern crate hyper;
extern crate futures;
extern crate serde;
extern crate serde_json;

use std::fmt::Debug;
use std::sync::Arc;
use self::serde::{Serialize, Deserialize};
use self::futures::{Future, Stream};
use self::hyper::header::ContentLength;
use self::hyper::server::{Http, Service, Request, Response};
use interface::{Oracle, MiType};
use cc::DType;


const VERSION: &str = "braid version 0.0.1";

/// Deseralize the request body (as bytes)
fn deserialize_req<'de, T: Deserialize<'de>>(req: &'de [u8]) -> T {
    serde_json::from_slice(&req).unwrap()
}


/// Deralize the a struct into a json response (as String)
fn serialize_resp<T: Serialize>(resp: &T) -> String {
    serde_json::to_string(resp).unwrap()
}

fn serialize_error<Q: Debug>(func_name: &str, req: &Q, err: String) -> String {
    let resp = ErrorResp {
        func_name: String::from(func_name),
        args: format!("{:?}", req),
        error: err
    };
    serialize_resp(&resp)
}


/// Error response structure
#[derive(Serialize, Debug)]
struct ErrorResp {
    func_name: String,
    args: String,
    error: String,
}


// Dependent probability
// ---------------------
#[derive(Deserialize, Debug)]
struct DepprobReq {
    pub col_a: usize,
    pub col_b: usize,
}

#[derive(Serialize, Debug)]
struct DepprobResp {
    col_a: usize,
    col_b: usize,
    depprob: f64,
}


fn depprob_req(oracle: &Oracle, req: &DepprobReq) -> String {
    let col_a = req.col_a;
    let col_b = req.col_b;
    let result = oracle.depprob(col_a, col_b);
    match result {
        Ok(ans) => {
            let resp = DepprobResp { col_a: col_a, col_b: col_b, depprob: ans };
            serialize_resp(&resp)
        },
        Err(err) => serialize_error("depprob", &req, err)
    }
}


// Row similarity
// --------------
#[derive(Deserialize, Debug)]
struct RowsimReq {
    pub row_a: usize,
    pub row_b: usize,
    pub wrt: Vec<usize>,
}

#[derive(Serialize)]
struct RowsimResp {
    row_a: usize,
    row_b: usize,
    rowsim: f64,
}

fn rowsim_req(oracle: &Oracle, req: &RowsimReq) -> String {
    let row_a = req.row_a;
    let row_b = req.row_b;
    let wrt = if req.wrt.is_empty() {None} else {Some(&req.wrt)};
    let result = oracle.rowsim(row_a, row_b, wrt);
    match result {
        Ok(ans) => {
            let resp = RowsimResp {row_a: row_a, row_b: row_b, rowsim: ans};
            serialize_resp(&resp)
        },
        Err(err) => serialize_error("rowsim", &req, err)
    }
}


// Mutual information (MI)
// -----------------------
#[derive(Deserialize, Debug)]
struct MiReq {
    pub col_a: usize,
    pub col_b: usize,
    pub n: usize,
    pub mi_type: MiType,
}

#[derive(Serialize)]
struct MiResp {
    col_a: usize,
    col_b: usize,
    mi: f64,
}

fn mi_req(oracle: &Oracle, req: &MiReq) -> String {
    let col_a = req.col_a;
    let col_b = req.col_b;
    let mut rng = rand::thread_rng();
    let result = oracle.mutual_information(
        col_a, col_b, req.n, req.mi_type.clone(), &mut rng);
    match result {
        Ok(ans) => {
            let resp = MiResp {col_a: col_a, col_b: col_b, mi: ans};
            serialize_resp(&resp)
        },
        Err(err) => serialize_error("mutual_information", &req, err)
    }
}


// Simulate
#[derive(Deserialize, Debug)]
struct SimulateReq {
    pub col_ixs: Vec<usize>,
    pub given: Vec<(usize, DType)>,
    pub n: usize
}

#[derive(Serialize)]
struct SimulateResp {
    values: Vec<Vec<DType>>
}

fn simulate_req(oracle: &Oracle, req: &SimulateReq) -> String {
    let given_opt = if req.given.is_empty() {
        None
    } else {
        Some(req.given.clone())
    };
    let mut rng = rand::thread_rng();
    let result = oracle.simulate(&req.col_ixs, &given_opt, req.n, &mut rng);
    match result {
        Ok(values) => {
            let resp = SimulateResp { values: values };
            serialize_resp(&resp)
        },
        Err(err) => serialize_error("simulate", &req, err)
    }
}


// logp
#[derive(Deserialize, Debug)]
struct LogpReq {
    pub col_ixs: Vec<usize>,
    pub values: Vec<Vec<DType>>,
    pub given: Vec<(usize, DType)>,
}

#[derive(Serialize)]
struct LogpResp {
    logp: Vec<f64>
}

fn logp_req(oracle: &Oracle, req: &LogpReq) -> String {
    let given_opt = if req.given.is_empty() {
        None
    } else {
        Some(req.given.clone())
    };
    let result = oracle.logp(&req.col_ixs, &req.values, &given_opt);
    match result {
        Ok(logp) => {
            let resp = LogpResp { logp: logp };
            serialize_resp(&resp)
        },
        Err(err) => serialize_error("logp", &req, err)
    }
}


// Server
// ------
#[derive(Clone)]
struct OraclePt {
    arc: Arc<Oracle>,
}

impl OraclePt {
    fn new(oracle: Oracle) -> Self {
        OraclePt { arc: Arc::new(oracle) }
    }

    fn clone_arc(&self) -> Arc<Oracle> {
        self.arc.clone()
    }
}


impl Service for OraclePt {
    type Request = Request;
    type Response = Response;
    type Error = hyper::Error;
    type Future = Box<Future<Item = Self::Response, Error = Self::Error>>;

    fn call(&self, req: Request) -> Self::Future {
         match (req.method(), req.path()) {
            (&hyper::Method::Get, _) => {
                println!("{:?}", req.uri().query());
                let response = Response::new().with_body(VERSION);
                Box::new(futures::future::ok(response))
            },
            (&hyper::Method::Post, "/shape") => {
                // get number of rows and columns
                println!("\t - REQUEST: shape");
                let oracle = self.clone_arc();
                let nrows = oracle.nrows();
                let ncols = oracle.ncols();
                let resp = format!("{{\"rows\": {}, \"cols\": {}}}", nrows, ncols);
                println!("\t   + Copmuted response.");
                let response = Response::new().with_body(resp);
                Box::new(futures::future::ok(response))
            },
            (&hyper::Method::Post, "/nstates") => {
                // get number of rows and columns
                let oracle = self.clone_arc();
                let resp = format!("{{\"nstates\": {}}}", oracle.nstates());
                let response = Response::new().with_body(resp);
                Box::new(futures::future::ok(response))
            },
            (&hyper::Method::Post, "/depprob") => {
                println!("\t - REQUEST: depprob");
                let oracle = self.clone_arc();
                Box::new(req.body().concat2().map(move |b| {
                    let query: DepprobReq = deserialize_req(&b.as_ref());
                    let ans = depprob_req(&*oracle, &query);
                    Response::new()
                        .with_header(ContentLength(ans.len() as u64))
                        .with_body(ans)
                }))
            },
            (&hyper::Method::Post, "/rowsim") => {
                println!("\t - REQUEST: rowsim");
                let oracle = self.clone_arc();
                Box::new(req.body().concat2().map(move |b| {
                    let query: RowsimReq = deserialize_req(&b.as_ref());
                    let ans = rowsim_req(&*oracle, &query);
                    Response::new()
                        .with_header(ContentLength(ans.len() as u64))
                        .with_body(ans)
                }))
            },
            (&hyper::Method::Post, "/mutual_information") => {
                println!("\t - REQUEST: mutual_information");
                let oracle = self.clone_arc();
                Box::new(req.body().concat2().map(move |b| {
                    let query: MiReq = deserialize_req(&b.as_ref());
                    let ans = mi_req(&*oracle, &query);
                    Response::new()
                        .with_header(ContentLength(ans.len() as u64))
                        .with_body(ans)
                }))
            },
            (&hyper::Method::Post, "/surprisal") => {
                // surprisal
                let response = Response::new()
                    .with_status(hyper::StatusCode::NotFound);
                Box::new(futures::future::ok(response))
            },
            (&hyper::Method::Post, "/simulate") => {
                // simulate
                println!("\t - REQUEST: simualte");
                let oracle = self.clone_arc();
                Box::new(req.body().concat2().map(move |b| {
                    let query: SimulateReq = deserialize_req(&b.as_ref());
                    let ans = simulate_req(&*oracle, &query);
                    Response::new()
                        .with_header(ContentLength(ans.len() as u64))
                        .with_body(ans)
                }))
            },
            (&hyper::Method::Post, "/logp") => {
                // log probability
                println!("\t - REQUEST: logp");
                let oracle = self.clone_arc();
                Box::new(req.body().concat2().map(move |b| {
                    let query: LogpReq = deserialize_req(&b.as_ref());
                    let ans = logp_req(&*oracle, &query);
                    Response::new()
                        .with_header(ContentLength(ans.len() as u64))
                        .with_body(ans)
                }))
            },
            (&hyper::Method::Post, "/predict") => {
                // predict with uncertainty
                let response = Response::new()
                    .with_status(hyper::StatusCode::NotFound);
                Box::new(futures::future::ok(response))
            },
            _ => {
                let response = Response::new()
                    .with_status(hyper::StatusCode::NotFound);
                Box::new(futures::future::ok(response))
            },
        }
    }
}


pub fn run_oracle_server(oracle: Oracle, _port: &str) {
    let addr = "127.0.0.1:3000".parse().unwrap();

    let arc = OraclePt::new(oracle);

    println!("\n  Running {} server on {}", VERSION, addr);
    println!("  Shut down with ^c\n");

    let server = Http::new().bind(&addr, move || Ok(arc.clone())).unwrap();
    server.run().unwrap();
}
