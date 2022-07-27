mod helpers;

use helpers::post_resp;

use braid_server::api::v1;
use warp::hyper::StatusCode;

#[test_log::test(tokio::test)]
async fn get_data_single() {
    let query = r#"{
        "ixs": [
            [0, 1]
        ]
    }"#;

    let body = post_resp("/api/v1/get_data", Some(query), None).await;
    let res: v1::GetDataResponse = serde_json::from_str(body.as_str()).unwrap();

    assert_eq!(res.values.len(), 1);
    assert_eq!(res.values[0].0, 0);
    assert_eq!(res.values[0].1, 1);
    assert!(res.values[0].2.is_categorical());
}

#[test_log::test(tokio::test)]
async fn get_data_multi() {
    let query = r#"{
        "ixs": [
            [0, 1],
            [2, 3],
            [4, 3]
        ]
    }"#;

    let body = post_resp("/api/v1/get_data", Some(query), None).await;
    let res: v1::GetDataResponse = serde_json::from_str(body.as_str()).unwrap();

    assert_eq!(res.values.len(), 3);
    assert!(res.values.iter().all(|v| v.2.is_categorical()));
}

#[test_log::test(tokio::test)]
async fn get_data_oob_row_returns_status_400() {
    let query = r#"{
        "ixs": [
            [1, 1],
            [200, 3],
            [4, 3]
        ]
    }"#;

    post_resp(
        "/api/v1/get_data",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
}

#[test_log::test(tokio::test)]
async fn get_data_oob_col_returns_status_400() {
    let query = r#"{
        "ixs": [
            [1, 1],
            [2, 3],
            [4, 200]
        ]
    }"#;

    post_resp(
        "/api/v1/get_data",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
}

#[test_log::test(tokio::test)]
async fn get_data_too_many_reqs_returns_400() {
    let mut query = String::from("{\"ixs\":[[1,1],[2,3],[4,0],");
    (0..99_999).for_each(|_| query.push_str("[0,0],"));
    query.push_str("[1,1]]}");

    post_resp(
        "/api/v1/get_data",
        Some(query.as_str()),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
}
