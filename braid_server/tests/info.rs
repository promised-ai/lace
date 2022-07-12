mod helpers;

use braid_server::api::v1;
use helpers::{encrypted_get_resp, get_resp};

#[test_log::test(tokio::test)]
async fn get_version() {
    let body = get_resp("/api/v1/version", None).await;
    let _version = {
        let res: v1::VersionResponse =
            serde_json::from_str(body.as_str()).unwrap();
        semver::Version::parse(&res.version).unwrap()
    };
}

#[test_log::test(tokio::test)]
async fn get_version_encrypted() {
    let body = encrypted_get_resp("/api/v1/version", None).await;
    let _version = {
        let res: v1::VersionResponse =
            serde_json::from_str(body.as_str()).unwrap();
        semver::Version::parse(&res.version).unwrap()
    };
}

#[test_log::test(tokio::test)]
async fn get_request_limits() {
    let body = get_resp("/api/v1/request_limits", None).await;
    let res: v1::RequestLimitsResponse =
        serde_json::from_str(body.as_str()).unwrap();
    println!("{:?}", res);
}
