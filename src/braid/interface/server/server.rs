extern crate rand;
extern crate hyper;
extern crate futures;
extern crate serde;
extern crate serde_json;

use std::fmt::Debug;
use std::io;
use std::sync::Arc;

use self::serde::Deserialize;
use self::futures::{Future, Stream};
use self::hyper::Chunk;
use self::hyper::header::{ContentType, ContentLength};
use self::hyper::server::{Http, Service, Request, Response};

use interface::Oracle;
use interface::server::api;
use interface::server::utils;


const VERSION: &str = "braid version 0.0.1";


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
                let response = Response::new()
                    .with_header(ContentLength(resp.len() as u64))
                    .with_header(ContentType::json())
                    .with_body(resp);

                Box::new(futures::future::ok(response))
            },
            (&hyper::Method::Post, "/nstates") => {
                // get number of rows and columns
                let oracle = self.clone_arc();
                let resp = format!("{{\"nstates\": {}}}", oracle.nstates());
                let response = Response::new()
                    .with_header(ContentLength(resp.len() as u64))
                    .with_header(ContentType::json())
                    .with_body(resp);

                Box::new(futures::future::ok(response))
            },
            (&hyper::Method::Post, "/ftypes") => {
                println!("\t - REQUEST: ftypes");
                let oracle = self.clone_arc();
                Box::new(req.body().concat2().map(move |b| {
                    do_func::<api::FTypesReq, _>(
                        "DTypes", &b, &oracle, api::ftypes_req)
                }))
            },
            (&hyper::Method::Post, "/codebook") => {
                println!("\t - REQUEST: codebook");
                let oracle = self.clone_arc();
                Box::new(req.body().concat2().map(move |b| {
                    do_func::<api::CodebookReq, _>(
                        "Codebook", &b, &oracle, api::codebook_req)
                }))
            },
            (&hyper::Method::Post, "/diagnostics") => {
                // get number of rows and columns
                println!("\t - REQUEST: diagnostics");
                let oracle = self.clone_arc();
                Box::new(req.body().concat2().map(move |b| {
                    do_func::<api::DiagnosticsReq, _>(
                        "Diagnostics", &b, &oracle, api::diagnostics_req)
                }))
            },
            (&hyper::Method::Post, "/depprob") => {
                println!("\t - REQUEST: depprob");
                let oracle = self.clone_arc();
                Box::new(req.body().concat2().map(move |b| {
                    do_func::<api::DepprobReq, _>(
                        "Depprob", &b, &oracle, api::depprob_req)
                }))
            },
            (&hyper::Method::Post, "/rowsim") => {
                println!("\t - REQUEST: rowsim");
                let oracle = self.clone_arc();
                Box::new(req.body().concat2().map(move |b| {
                    do_func::<api::RowsimReq, _>(
                        "Rowsim", &b, &oracle, api::rowsim_req)
                }))
            },
            (&hyper::Method::Post, "/mutual_information") => {
                println!("\t - REQUEST: mutual_information");
                let oracle = self.clone_arc();
                Box::new(req.body().concat2().map(move |b| {
                    do_func::<api::MiReq, _>(
                        "mutual_information", &b, &oracle, api::mi_req)
                }))
            },
            (&hyper::Method::Post, "/surprisal") => {
                // surprisal
                println!("\t - REQUEST: surprisal");
                let oracle = self.clone_arc();
                Box::new(req.body().concat2().map(move |b| {
                    do_func::<api::SurprisalReq, _>(
                        "surprisal", &b, &oracle, api::surprisal_req)
                }))
            },
            (&hyper::Method::Post, "/simulate") => {
                // simulate
                println!("\t - REQUEST: simulate");
                let oracle = self.clone_arc();
                Box::new(req.body().concat2().map(move |b| {
                    do_func::<api::SimulateReq, _>(
                        "simulate", &b, &oracle, api::simulate_req)
                }))
            },
             (&hyper::Method::Post, "/impute") => {
                // simulate
                println!("\t - REQUEST: impute");
                let oracle = self.clone_arc();
                Box::new(req.body().concat2().map(move |b| {
                    do_func::<api::ImputeReq, _>(
                        "impute", &b, &oracle, api::impute_req)
                }))
           },
            (&hyper::Method::Post, "/logp") => {
                // log probability
                println!("\t - REQUEST: logp");
                let oracle = self.clone_arc();
                Box::new(req.body().concat2().map(move |b| {
                    do_func::<api::LogpReq, _>(
                        "logp", &b, &oracle, api::logp_req)
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


fn do_func<'de, T, F>(func_name: &str, b: &'de Chunk, oracle: &Oracle, f: F)
    -> Response 
    where T: Deserialize<'de> + Debug,
          F: Fn(&Oracle, &T) -> io::Result<String>
{
    let msg = match utils::deserialize_req::<T>(&b.as_ref()) {
        Ok(query) => {
            match f(&oracle, &query) {
                Ok(resp) => resp,
                Err(err) => utils::serialize_error(func_name, &query, err)
            }
        },
        Err(err) => {
            utils::serialize_error(func_name, &"json-err", err)
        }
    };
    Response::new()
        .with_header(ContentLength(msg.len() as u64))
        .with_header(ContentType::json())
        .with_body(msg)

}


pub fn run_oracle_server(oracle: Oracle, port: &str) {
    let addr = format!("127.0.0.1:{}", port).parse().unwrap();

    let arc = OraclePt::new(oracle);

    println!("\n  Running {} server on {}", VERSION, addr);
    println!("  Shut down with ^c\n");

    let server = Http::new().bind(&addr, move || Ok(arc.clone())).unwrap();
    server.run().unwrap();
}
