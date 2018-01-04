extern crate rand;
extern crate hyper;
extern crate futures;

use self::rand::XorShiftRng;
use self::futures::future::Future;
use self::hyper::header::ContentLength;
use self::hyper::server::{Http, Request, Response, Service};
use Oracle;


const PHRASE: &str = "braid version 0.0.1";

pub struct OracleServer {
    oracle: Oracle,
    rng: XorShiftRng,
}


impl OracleServer {
    pub fn new(oracle: Oracle, rng: XorShiftRng) -> Self {
        OracleServer { oracle: oracle, rng: rng }
    }
}


impl Service for OracleServer {
    type Request = Request;
    type Response = Response;
    type Error = hyper::Error;

    type Future = Box<Future<Item=Self::Response, Error=Self::Error>>;

    fn call(&self, _req: Request) -> Self::Future {
        // We're currently ignoring the Request
        // And returning an 'ok' Future, which means it's ready
        // immediately, and build a Response with the 'PHRASE' body.
        Box::new(futures::future::ok(
            Response::new()
                .with_header(ContentLength(PHRASE.len() as u64))
                .with_body(PHRASE)
        ))
    }
}

pub fn run_server(oracle: Oracle, port: &str) {
    unimplemented!();
}
