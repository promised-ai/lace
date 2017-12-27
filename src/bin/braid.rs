extern crate jsonrpc_core;
extern crate jsonrpc_http_server;

use self::jsonrpc_core::{IoHandler, Params, Value};
use self::jsonrpc_http_server::{ServerBuilder};


pub fn build_server() -> ServerBuilder {
    let mut io = IoHandler::new();
    io.add_method("version", |_: Params| {
        Ok(Value::String("Braid version 0.0.1dev".into()))
    });

    ServerBuilder::new(io)
}


fn main() {
    let _server = build_server()
        .start_http(&"127.0.0.1:2723".parse().unwrap());
}
