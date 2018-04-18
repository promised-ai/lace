extern crate futures;
extern crate hyper;
extern crate rand;
extern crate serde;
extern crate serde_json;

use std::fmt::Debug;
use std::io;
use std::sync::Arc;

use self::futures::{Future, Stream};
use self::hyper::Chunk;
use self::hyper::header::{ContentLength, ContentType};
use self::hyper::server::{Http, Request, Response, Service};
use self::serde::Deserialize;

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
        OraclePt {
            arc: Arc::new(oracle),
        }
    }

    fn clone_arc(&self) -> Arc<Oracle> {
        self.arc.clone()
    }
}

macro_rules! router_func {
    ($name:expr, $req:expr, $oracle:expr, $req_type:ty, $req_func:expr) => {
        Box::new($req.body().concat2().map(move |b| {
            do_func::<$req_type, _>($name, &b, &$oracle, $req_func)
        }))
    };
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
            }

            (&hyper::Method::Post, "/shape") => {
                let oracle = self.clone_arc();
                println!("\t - REQUEST: shape");
                router_func!("Shape", req, oracle, api::NoReq, api::shape_req)
            }

            (&hyper::Method::Post, "/nstates") => {
                let oracle = self.clone_arc();
                println!("\t - REQUEST: nstates");
                router_func!(
                    "NStates",
                    req,
                    oracle,
                    api::NoReq,
                    api::nstates_req
                )
            }

            (&hyper::Method::Post, "/ftypes") => {
                let oracle = self.clone_arc();
                println!("\t - REQUEST: ftypes");
                router_func!(
                    "DTypes",
                    req,
                    oracle,
                    api::NoReq,
                    api::ftypes_req
                )
            }

            (&hyper::Method::Post, "/codebook") => {
                let oracle = self.clone_arc();
                println!("\t - REQUEST: codebook");
                router_func!(
                    "Codebook",
                    req,
                    oracle,
                    api::NoReq,
                    api::codebook_req
                )
            }

            (&hyper::Method::Post, "/diagnostics") => {
                let oracle = self.clone_arc();
                println!("\t - REQUEST: diagnostics");
                router_func!(
                    "Diagnostics",
                    req,
                    oracle,
                    api::DiagnosticsReq,
                    api::diagnostics_req
                )
            }

            (&hyper::Method::Post, "/depprob") => {
                let oracle = self.clone_arc();
                println!("\t - REQUEST: depprob");
                router_func!(
                    "Depprob",
                    req,
                    oracle,
                    api::DepprobReq,
                    api::depprob_req
                )
            }

            (&hyper::Method::Post, "/rowsim") => {
                let oracle = self.clone_arc();
                println!("\t - REQUEST: rowsim");
                router_func!(
                    "Rowsim",
                    req,
                    oracle,
                    api::RowsimReq,
                    api::rowsim_req
                )
            }

            (&hyper::Method::Post, "/mutual_information") => {
                let oracle = self.clone_arc();
                println!("\t - REQUEST: mutual_information");
                router_func!(
                    "mutual_information",
                    req,
                    oracle,
                    api::MiReq,
                    api::mi_req
                )
            }

            (&hyper::Method::Post, "/surprisal") => {
                let oracle = self.clone_arc();
                println!("\t - REQUEST: surprisal");
                router_func!(
                    "surprisal",
                    req,
                    oracle,
                    api::SurprisalReq,
                    api::surprisal_req
                )
            }

            (&hyper::Method::Post, "/simulate") => {
                let oracle = self.clone_arc();
                println!("\t - REQUEST: simulate");
                router_func!(
                    "Simulate",
                    req,
                    oracle,
                    api::SimulateReq,
                    api::simulate_req
                )
            }

            (&hyper::Method::Post, "/draw") => {
                let oracle = self.clone_arc();
                println!("\t - REQUEST: draw");
                router_func!(
                    "Draw",
                    req,
                    oracle,
                    api::DrawReq,
                    api::draw_req
                )
            }

            (&hyper::Method::Post, "/impute") => {
                let oracle = self.clone_arc();
                println!("\t - REQUEST: impute");
                router_func!(
                    "Impute",
                    req,
                    oracle,
                    api::ImputeReq,
                    api::impute_req
                )
            }

            (&hyper::Method::Post, "/logp") => {
                let oracle = self.clone_arc();
                println!("\t - REQUEST: logp");
                router_func!(
                    "LogP",
                    req,
                    oracle,
                    api::LogpReq,
                    api::logp_req
                    )
            }

            (&hyper::Method::Post, "/predict") => {
                // predict with uncertainty
                let response =
                    Response::new().with_status(hyper::StatusCode::NotFound);
                Box::new(futures::future::ok(response))
            }

            _ => {
                let response =
                    Response::new().with_status(hyper::StatusCode::NotFound);
                Box::new(futures::future::ok(response))
            }
        }
    }
}

fn do_func<'de, T, F>(
    func_name: &str,
    b: &'de Chunk,
    oracle: &Oracle,
    f: F,
) -> Response
where
    T: Deserialize<'de> + Debug,
    F: Fn(&Oracle, &T) -> io::Result<String>,
{
    let msg = match utils::deserialize_req::<T>(&b.as_ref()) {
        Ok(query) => match f(&oracle, &query) {
            Ok(resp) => resp,
            Err(err) => utils::serialize_error(func_name, &query, err),
        },
        Err(err) => utils::serialize_error(func_name, &"json-err", err),
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

    let server = Http::new()
        .bind(&addr, move || Ok(arc.clone()))
        .unwrap();
    server.run().unwrap();
}
