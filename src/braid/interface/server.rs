extern crate rand;
extern crate hyper;
extern crate futures;

use self::futures::future::Future;
use self::hyper::header::ContentLength;
use self::hyper::server::{Http, Service, Request, Response};
use Oracle;


const PHRASE: &str = "braid version 0.0.1";


impl<'a> Service for &'a Oracle {
    type Request = Request;
    type Response = Response;
    type Error = hyper::Error;
    type Future = Box<Future<Item = Self::Response, Error = Self::Error>>;

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


pub fn run_oracle_server(oracle: Oracle, _port: &str) {
    let addr = "127.0.0.1:3000".parse().unwrap();
    println!("\n  Running {} server on {}", PHRASE, addr);
    println!("  Shut down with ^c\n");
    let server = Http::new().bind(&addr, move || Ok(&oracle)).unwrap();
    server.run().unwrap();
}
