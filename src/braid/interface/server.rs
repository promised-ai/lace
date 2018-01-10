extern crate rand;
extern crate hyper;
extern crate futures;
extern crate serde;
extern crate serde_json;

use std::sync::Arc;
use self::serde::{Serialize, Deserialize};
use self::futures::{Future, Stream, Sink};
use self::futures::sink::Wait;
use self::hyper::header::ContentLength;
use self::hyper::server::{Http, Service, Request, Response};
use Oracle;


const VERSION: &str = "braid version 0.0.1";



/// Deseralize the request body (as bytes)
fn deserialize_req<'de, T: Deserialize<'de>>(req: &'de [u8]) -> T {
    serde_json::from_slice(&req).unwrap()
}


/// Deralize the a struct into a json response (as String)
fn serialize_resp<T: Serialize>(resp: &T) -> String {
    serde_json::to_string(resp).unwrap()
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
    let depprob = oracle.depprob(col_a, col_b);
    let ans = DepprobResp { col_a: col_a, col_b: col_b, depprob: depprob };
    serialize_resp(&ans)
}


// Row similarity
// --------------
#[derive(Deserialize)]
struct RowsimReq {
    pub row_a: usize,
    pub row_b: usize,
}

#[derive(Serialize)]
struct RowsimResp {
    row_a: usize,
    row_b: usize,
    rowsim: f64,
}

fn rowsim_req(_oracle: &Oracle, req: &RowsimReq) -> String {
    // TODO: There is going to be some work to be done in order to make wrt
    // work correctly...
   unimplemented!();
}


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
                // dependence probability
                println!("\t - REQUEST: depprob");
                // let b = req.body().concat2().wait();
                // println!("\t   + In depprob future.");
                let oracle = self.clone_arc();
                Box::new(req.body().concat2().map(move |b| {
                    let query: DepprobReq = deserialize_req(&b.as_ref());
                    let ans = depprob_req(&*oracle, &query);
                    println!("{:?}", ans);
                    Response::new()
                        .with_header(ContentLength(ans.len() as u64))
                        .with_body(ans)
                }))
            },
            (&hyper::Method::Post, "/rowsim") => {
                // row simlarity
                let response = Response::new()
                    .with_status(hyper::StatusCode::NotFound);
                Box::new(futures::future::ok(response))
            },
            (&hyper::Method::Post, "/mutual_information") => {
                // dependence probability
                let response = Response::new()
                    .with_status(hyper::StatusCode::NotFound);
                Box::new(futures::future::ok(response))
            },
            (&hyper::Method::Post, "/surprisal") => {
                // surprisal
                let response = Response::new()
                    .with_status(hyper::StatusCode::NotFound);
                Box::new(futures::future::ok(response))
            },
            (&hyper::Method::Post, "/simulate") => {
                // simulate
                let response = Response::new()
                    .with_status(hyper::StatusCode::NotFound);
                Box::new(futures::future::ok(response))
            },
            (&hyper::Method::Post, "/logp") => {
                // log probability
                let response = Response::new()
                    .with_status(hyper::StatusCode::NotFound);
                Box::new(futures::future::ok(response))
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
