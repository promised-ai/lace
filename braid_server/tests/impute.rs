mod helpers;

use helpers::post_resp;

use braid_server::api::v1;
use warp::hyper::StatusCode;

#[test_log::test(tokio::test)]
async fn impute_no_uncertainty() {
    let query = r#"{
        "col_ix": 0,
        "row_ix": 1
    }"#;
    let body = post_resp("/api/v1/impute", Some(query), None).await;

    let res: v1::ImputeResponse = serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.col_ix, 0);
    assert_eq!(res.row_ix, 1);
    assert!(res.value.is_categorical());
    assert!(res.uncertainty.is_none());
}

#[test_log::test(tokio::test)]
async fn impute_with_uncertainty() {
    let query = r#"{
        "col_ix": 0,
        "row_ix": 1,
        "uncertainty_type": "js_divergence"
    }"#;
    let body = post_resp("/api/v1/impute", Some(query), None).await;

    let res: v1::ImputeResponse = serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.col_ix, 0);
    assert_eq!(res.row_ix, 1);
    assert!(res.value.is_categorical());
    assert!(res.uncertainty.is_some());
}

#[test_log::test(tokio::test)]
async fn impute_oob_col_returns_status_400() {
    let query = r#"{
        "col_ix": 999,
        "row_ix": 1
    }"#;
    post_resp("/api/v1/impute", Some(query), Some(StatusCode::BAD_REQUEST))
        .await;
}

#[test_log::test(tokio::test)]
async fn impute_oob_row_returns_status_400() {
    let query = r#"{
        "col_ix": 0,
        "row_ix": 8888,
        "uncertainty_type": "js_divergence"
    }"#;
    post_resp("/api/v1/impute", Some(query), Some(StatusCode::BAD_REQUEST))
        .await;
}
