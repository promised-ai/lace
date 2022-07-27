mod helpers;

use helpers::post_resp;

use braid_server::api::v1;
use warp::hyper::StatusCode;

#[test_log::test(tokio::test)]
async fn one_col() {
    let query = r#"{
        "col_ixs": [0],
        "n": 1000
    }"#;
    let body = post_resp("/api/v1/entropy", Some(query), None).await;
    let res: v1::EntropyResponse = serde_json::from_str(body.as_str()).unwrap();
    // The examples server uses the animals dataset, so all entropies will be
    // positive (proper).
    assert!(res.entropy > 0.0);
}

#[test_log::test(tokio::test)]
async fn multi_cols() {
    let query = r#"{
        "col_ixs": [0, 4],
        "n": 1000
    }"#;
    let body = post_resp("/api/v1/entropy", Some(query), None).await;
    let res: v1::EntropyResponse = serde_json::from_str(body.as_str()).unwrap();
    assert!(res.entropy > 0.0);
}

#[test_log::test(tokio::test)]
async fn more_cols_more_entropy() {
    let h1 = {
        let query = r#"{
            "col_ixs": [75],
            "n": 1000
        }"#;
        let body = post_resp("/api/v1/entropy", Some(query), None).await;
        let res: v1::EntropyResponse =
            serde_json::from_str(body.as_str()).unwrap();
        res.entropy
    };

    let h2 = {
        let query = r#"{
            "col_ixs": [75, 65],
            "n": 1000
        }"#;
        let body = post_resp("/api/v1/entropy", Some(query), None).await;
        let res: v1::EntropyResponse =
            serde_json::from_str(body.as_str()).unwrap();
        res.entropy
    };
    assert!(h1 < h2);
}

#[test_log::test(tokio::test)]
async fn no_columns_status_400() {
    let query = r#"{
        "col_ixs": [],
        "n": 1000
    }"#;
    let resp = post_resp(
        "/api/v1/entropy",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    assert!(resp.contains("No target columns provided"));
}

#[test_log::test(tokio::test)]
async fn oob_column_status_400() {
    let query = r#"{
        "col_ixs": [122],
        "n": 1000
    }"#;
    let resp = post_resp(
        "/api/v1/entropy",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    assert!(resp.contains("ix is 122 but there"));
}

#[test_log::test(tokio::test)]
async fn no_samples_status_400() {
    let query = r#"{
        "col_ixs": [0],
        "n": 0
    }"#;
    let resp = post_resp(
        "/api/v1/entropy",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    assert!(resp.contains("more than zero samples"));
}
