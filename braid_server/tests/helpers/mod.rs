use std::convert::{Infallible, TryFrom};
use std::path::{Path, PathBuf};
use std::process::Command;
use warp::http::HeaderValue;
use warp::hyper::header::{CONTENT_ENCODING, CONTENT_TYPE};
use warp::hyper::StatusCode;
use warp::Filter;

const BRAIDFILE: &str = "resources/test/animals.braid";
const DATAFILE: &str = "resources/test/animals.csv";
const ENCRYPTED_DATAFILE: &str = "resources/test/animals-encrypted.csv";

const ENCRYPTION_KEY: &str =
    "93d3c14316ba2c5375532ddb725a20f75773cf3b7f9b92542abb3da4fe14a18b";

pub fn fixture(
    encrypted: bool,
) -> impl warp::Filter<Extract = impl warp::Reply, Error = Infallible> + Clone {
    // build a braidfile if none exists
    // XXX: If the system `braid` creates metadata that is not compatible
    // with the `braid_server` version of `braid`, the tests will fail.
    if !Path::new(BRAIDFILE).exists() {
        let output = if encrypted {
            Command::new("braid")
                .arg("run")
                .args(["-s", "4", "-n", "100"])
                .arg("--csv")
                .arg(ENCRYPTED_DATAFILE)
                .arg("-k")
                .arg(ENCRYPTION_KEY)
                .output()
                .expect("failed to execute process")
        } else {
            Command::new("braid")
                .arg("run")
                .args(["-s", "4", "-n", "100"])
                .arg("--csv")
                .arg(DATAFILE)
                .output()
                .expect("failed to execute process")
        };

        assert!(output.status.success());
    }

    braid_server::server::warp(
        &PathBuf::try_from(BRAIDFILE).expect("should parse"),
        true,
        None,
        1024_u64.pow(3),
    )
    .with(warp::log("braid_server"))
}

#[allow(dead_code)]
pub async fn post_resp_client<F, R>(
    filter: &F,
    route: &str,
    data: Option<&str>,
    exp_status: Option<StatusCode>,
) -> String
where
    F: warp::Filter<Extract = R, Error = Infallible> + 'static,
    R: warp::Reply,
{
    let req = warp::test::request().path(route).method("POST");
    let req = match data {
        Some(data) => req.body(data).header(CONTENT_TYPE, "application/json"),
        None => req,
    };

    let response = req.reply(filter).await;
    let (parts, body) = response.into_parts();

    assert_eq!(parts.status, exp_status.unwrap_or(StatusCode::OK));
    assert_eq!(
        parts.headers.get(CONTENT_TYPE),
        Some(&HeaderValue::from_static("application/json"))
    );

    let encoding = parts.headers.get(CONTENT_ENCODING);
    let body: Vec<u8> = body.into_iter().collect();

    let resp: Vec<u8> = if encoding == Some(&HeaderValue::from_static("gzip")) {
        use flate2::read::GzDecoder;
        use std::io::Read;
        let mut decoder = GzDecoder::new(body.as_slice());
        let mut buf: Vec<u8> = Vec::new();
        decoder.read_to_end(&mut buf).unwrap();
        buf
    } else {
        body
    };

    String::from_utf8(resp).expect("Body should be UTF8")
}

#[allow(dead_code)]
pub async fn post_resp(
    route: &str,
    data: Option<&str>,
    exp_status: Option<StatusCode>,
) -> String {
    let client = fixture(false);
    post_resp_client(&client, route, data, exp_status).await
}

#[allow(dead_code)]
pub async fn get_resp_client<F, R>(
    filter: &F,
    route: &str,
    exp_status: Option<StatusCode>,
) -> String
where
    F: warp::Filter<Extract = R, Error = Infallible> + 'static,
    R: warp::Reply,
{
    let req = warp::test::request().path(route).method("GET");
    let response = req.reply(filter).await;
    let (parts, body) = response.into_parts();

    assert_eq!(parts.status, exp_status.unwrap_or(StatusCode::OK));
    assert_eq!(
        parts.headers.get(CONTENT_TYPE),
        Some(&HeaderValue::from_static("application/json"))
    );

    let body = body.into_iter().collect();
    String::from_utf8(body).expect("Body should be UTF8")
}

#[allow(dead_code)]
pub async fn get_resp(uri: &str, exp_status: Option<StatusCode>) -> String {
    let client = fixture(false);
    get_resp_client(&client, uri, exp_status).await
}

#[allow(dead_code)]
pub async fn encrypted_get_resp(
    uri: &str,
    exp_status: Option<StatusCode>,
) -> String {
    let client = fixture(true);
    get_resp_client(&client, uri, exp_status).await
}
