extern crate serde;
extern crate serde_json;

use std::fmt::Debug;
use std::io;
use std::error::Error;
use self::serde::{Serialize, Deserialize};


/// Error response structure
#[derive(Serialize, Debug)]
struct ErrorResp {
    func_name: String,
    args: String,
    error: String,
}


/// Deseralize the request body (as bytes)
pub fn deserialize_req<'de, T: Deserialize<'de>>(req: &'de [u8])
    -> io::Result<T>
{
    match serde_json::from_slice(&req) {
        Ok(obj)   => Ok(obj),
        Err(err)  => Err(err.into()),
    }
}


/// Deralize the a struct into a json response (as String)
pub fn serialize_resp<T: Serialize>(resp: &T) -> io::Result<String> {
    match serde_json::to_string(resp) {
        Ok(ser)  => Ok(ser),
        Err(err) => Err(err.into()),
    }
}


pub fn serialize_error<Q: Debug>(func_name: &str, req: &Q, err: io::Error)
    -> String {
    let resp = ErrorResp {
        func_name: String::from(func_name),
        args: format!("{:?}", req),
        error: String::from(err.description())
    };
    serialize_resp(&resp).unwrap()
}
