mod helpers;

use helpers::post_resp;

use braid_server::api::v1;
use warp::hyper::StatusCode;

#[test_log::test(tokio::test)]
async fn surprisal() {
    let query = r#"{
        "col_ix": 0,
        "row_ixs": [1]
    }"#;
    let body = post_resp("/api/v1/surprisal", Some(query), None).await;

    let res: v1::SurprisalResponse =
        serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.col_ix, 0);
    assert_eq!(res.row_ixs, vec![1]);

    assert_eq!(res.values.len(), 1);
    assert!(res.values[0].is_categorical());

    assert_eq!(res.surprisal.len(), 1);
    assert!(res.surprisal[0].is_some());
}

#[test_log::test(tokio::test)]
async fn surprisal_with_state_ixs() {
    let query = r#"{
        "col_ix": 0,
        "row_ixs": [1],
        "state_ixs": [0, 3]
    }"#;
    let body = post_resp("/api/v1/surprisal", Some(query), None).await;

    let res: v1::SurprisalResponse =
        serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.col_ix, 0);
    assert_eq!(res.row_ixs, vec![1]);

    assert_eq!(res.values.len(), 1);
    assert!(res.values[0].is_categorical());

    assert_eq!(res.surprisal.len(), 1);
    assert!(res.surprisal[0].is_some());
}

#[test_log::test(tokio::test)]
async fn surprisal_oob_col_returns_status_400() {
    let query = r#"{
        "col_ix": 10000,
        "row_ixs": [1]
    }"#;
    post_resp(
        "/api/v1/surprisal",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
}

#[test_log::test(tokio::test)]
async fn surprisal_oob_row_returns_status_400() {
    let query = r#"{
        "col_ix": 0,
        "row_ixs": [999]
    }"#;
    post_resp(
        "/api/v1/surprisal",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
}
