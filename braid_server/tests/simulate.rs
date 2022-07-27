mod helpers;

use helpers::post_resp;

use braid_server::api::v1;
use warp::hyper::StatusCode;

#[test_log::test(tokio::test)]
async fn simulate_no_given() {
    let query = r#"{"col_ixs": [2, 3, 1], "n": 112}"#;
    let body = post_resp("/api/v1/simulate", Some(query), None).await;
    let res: v1::SimulateResponse =
        serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.values.len(), 112);
    assert!(res.values.iter().all(|vals| vals.len() == 3));
}

#[test_log::test(tokio::test)]
async fn simulate_empty_given() {
    let query = r#"{
        "col_ixs": [2, 3, 1],
        "given": "nothing",
        "n": 112
    }"#;
    let body = post_resp("/api/v1/simulate", Some(query), None).await;
    let res: v1::SimulateResponse =
        serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.values.len(), 112);
    assert!(res.values.iter().all(|vals| vals.len() == 3));
}

#[test_log::test(tokio::test)]
async fn simulate_given() {
    let query = r#"{
        "col_ixs": [2, 3, 1],
        "given": {
            "conditions": [
                [0, {"categorical": 0}]
            ]
        },
        "n": 112
    }"#;
    let body = post_resp("/api/v1/simulate", Some(query), None).await;
    let res: v1::SimulateResponse =
        serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.values.len(), 112);
    assert!(res.values.iter().all(|vals| vals.len() == 3));
}

#[test_log::test(tokio::test)]
async fn simulate_given_with_redundant_col_should_raise_status_400() {
    let query = r#"{
        "col_ixs": [2, 3, 1],
        "given": {
            "conditions": [
                [2, {"categorical": 0}]
            ]
        },
        "n": 112
    }"#;
    let body = post_resp(
        "/api/v1/simulate",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    println!("{:?}", body);
}

#[test_log::test(tokio::test)]
async fn simulate_given_with_oob_col_should_raise_status_400() {
    let query = r#"{
        "col_ixs": [2, 3, 1],
        "given": {
            "conditions": [
                [900, {"categorical": 0}]
            ]
        },
        "n": 112
    }"#;
    let body = post_resp(
        "/api/v1/simulate",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    println!("{:?}", body);
}

#[test_log::test(tokio::test)]
async fn simulate_with_oob_target_should_return_status_400() {
    let query = r#"{
        "col_ixs": [2, 3, 900],
        "n": 112
    }"#;
    let body = post_resp(
        "/api/v1/simulate",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    println!("{:?}", body);
}

#[test_log::test(tokio::test)]
async fn simulate_given_with_wrong_data_type_should_raise_status_400() {
    let query = r#"{
        "col_ixs": [2, 3, 1],
        "given": {
            "conditions": [
                [0, {"continuous": 0.1}]
            ]
        },
        "n": 112
    }"#;
    let body = post_resp(
        "/api/v1/simulate",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    println!("{:?}", body);
}
