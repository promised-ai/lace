mod helpers;

use helpers::post_resp;

use braid_server::api::v1;
use warp::hyper::StatusCode;

#[test_log::test(tokio::test)]
async fn one_target_one_predictor() {
    let query = r#"{
        "target_ixs": [0],
        "predictor_ixs": [1],
        "n": 1000
    }"#;
    let body = post_resp("/api/v1/info_prop", Some(query), None).await;
    let res: v1::InfoPropResponse =
        serde_json::from_str(body.as_str()).unwrap();
    assert!(res.info_prop > 0.0);
    assert!(res.info_prop < 1.0);
}

#[test_log::test(tokio::test)]
async fn multi_target_one_predictor() {
    let query = r#"{
        "target_ixs": [0, 2],
        "predictor_ixs": [1],
        "n": 1000
    }"#;
    let body = post_resp("/api/v1/info_prop", Some(query), None).await;
    let res: v1::InfoPropResponse =
        serde_json::from_str(body.as_str()).unwrap();
    assert!(res.info_prop > 0.0);
    assert!(res.info_prop < 1.0);
}

#[test_log::test(tokio::test)]
async fn multi_target_multi_predictor() {
    let query = r#"{
        "target_ixs": [0, 2],
        "predictor_ixs": [1, 4],
        "n": 1000
    }"#;
    let body = post_resp("/api/v1/info_prop", Some(query), None).await;
    let res: v1::InfoPropResponse =
        serde_json::from_str(body.as_str()).unwrap();
    assert!(res.info_prop > 0.0);
    assert!(res.info_prop < 1.0);
}

#[test_log::test(tokio::test)]
async fn adding_predictors_increases_info_prop() {
    let res_1: v1::InfoPropResponse = {
        let query = r#"{
            "target_ixs": [18],
            "predictor_ixs": [36],
            "n": 1000
        }"#;
        let body = post_resp("/api/v1/info_prop", Some(query), None).await;
        serde_json::from_str(body.as_str()).unwrap()
    };

    let res_2: v1::InfoPropResponse = {
        let query = r#"{
            "target_ixs": [18],
            "predictor_ixs": [36, 65],
            "n": 1000
        }"#;
        let body = post_resp("/api/v1/info_prop", Some(query), None).await;
        serde_json::from_str(body.as_str()).unwrap()
    };

    assert!(res_1.info_prop < res_2.info_prop);
}

#[test_log::test(tokio::test)]
async fn no_targets_status_400() {
    let query = r#"{
        "target_ixs": [],
        "predictor_ixs": [1, 4],
        "n": 1000
    }"#;
    let resp = post_resp(
        "/api/v1/info_prop",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    assert!(resp.contains("no target columns"));
}

#[test_log::test(tokio::test)]
async fn no_predictors_status_400() {
    let query = r#"{
        "target_ixs": [0],
        "predictor_ixs": [],
        "n": 1000
    }"#;
    let resp = post_resp(
        "/api/v1/info_prop",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    assert!(resp.contains("no predictor columns"));
}

#[test_log::test(tokio::test)]
async fn no_samples_status_400() {
    let query = r#"{
        "target_ixs": [0],
        "predictor_ixs": [1],
        "n": 0
    }"#;
    let resp = post_resp(
        "/api/v1/info_prop",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    assert!(resp.contains("more than zero samples"));
}

#[test_log::test(tokio::test)]
async fn oob_target_status_400() {
    let query = r#"{
        "target_ixs": [113],
        "predictor_ixs": [1],
        "n": 100
    }"#;
    let resp = post_resp(
        "/api/v1/info_prop",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    assert!(resp.contains("index 113 but"));
}

#[test_log::test(tokio::test)]
async fn oob_predictor_status_400() {
    let query = r#"{
        "target_ixs": [13],
        "predictor_ixs": [1121],
        "n": 100
    }"#;
    let resp = post_resp(
        "/api/v1/info_prop",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    assert!(resp.contains("index 1121 but"));
}
