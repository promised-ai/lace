mod helpers;

use helpers::post_resp;

use braid_server::api::v1;
use warp::hyper::StatusCode;

#[test_log::test(tokio::test)]
async fn draw() {
    let query = r#"{
        "col_ix": 0,
        "row_ix": 1,
        "n": 12
    }"#;
    let body = post_resp("/api/v1/draw", Some(query), None).await;
    let res: v1::DrawResponse = serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.values.len(), 12);
}

#[test_log::test(tokio::test)]
async fn draw_oob_col_returns_status_400() {
    let query = r#"{
        "col_ix": 1000,
        "row_ix": 1,
        "n": 12
    }"#;
    let body =
        post_resp("/api/v1/draw", Some(query), Some(StatusCode::BAD_REQUEST))
            .await;
    assert!(body.contains("error"));
}

#[test_log::test(tokio::test)]
async fn draw_oob_row_returns_status_400() {
    let query = r#"{
        "col_ix": 0,
        "row_ix": 10000,
        "n": 12
    }"#;
    let body =
        post_resp("/api/v1/draw", Some(query), Some(StatusCode::BAD_REQUEST))
            .await;
    assert!(body.contains("error"));
}
