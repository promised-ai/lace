mod helpers;

use helpers::post_resp;

use braid_server::api::v1;
use warp::hyper::StatusCode;

#[test_log::test(tokio::test)]
async fn predict_no_given_no_uncertainty() {
    let query = r#"{
        "col_ix": 0
    }"#;
    let body = post_resp("/api/v1/predict", Some(query), None).await;
    let res: v1::PredictResponse = serde_json::from_str(body.as_str()).unwrap();
    assert!(res.value.is_categorical());
    assert!(res.uncertainty.is_none());
}

#[test_log::test(tokio::test)]
async fn predict_no_given_with_uncertainty() {
    let query = r#"{
        "col_ix": 0,
        "uncertainty_type": "js_divergence"
    }"#;
    let body = post_resp("/api/v1/predict", Some(query), None).await;
    let res: v1::PredictResponse = serde_json::from_str(body.as_str()).unwrap();
    assert!(res.value.is_categorical());
    assert!(res.uncertainty.is_some());
}

#[test_log::test(tokio::test)]
async fn predict_empty_given_no_uncertainty() {
    let query = r#"{
        "col_ix": 0,
        "given": "nothing"
    }"#;
    let body = post_resp("/api/v1/predict", Some(query), None).await;
    let res: v1::PredictResponse = serde_json::from_str(body.as_str()).unwrap();
    assert!(res.value.is_categorical());
    assert!(res.uncertainty.is_none());
}

#[test_log::test(tokio::test)]
async fn predict_empty_given_with_uncertainty() {
    let query = r#"{
        "col_ix": 0,
        "given": "nothing",
        "uncertainty_type": "js_divergence"
    }"#;
    let body = post_resp("/api/v1/predict", Some(query), None).await;
    let res: v1::PredictResponse = serde_json::from_str(body.as_str()).unwrap();
    assert!(res.value.is_categorical());
    assert!(res.uncertainty.is_some());
}

#[test_log::test(tokio::test)]
async fn predict_given_no_uncertainty() {
    let query = r#"{
        "col_ix": 0,
        "given": {
            "conditions" : [
                [1, {"categorical": 0}],
                [2, {"categorical": 0}]
            ]
        },
        "uncertainty_type": null
    }"#;
    let body = post_resp("/api/v1/predict", Some(query), None).await;
    let res: v1::PredictResponse = serde_json::from_str(body.as_str()).unwrap();
    assert!(res.value.is_categorical());
    assert!(res.uncertainty.is_none());
}

#[test_log::test(tokio::test)]
async fn predict_given_with_state_ixs() {
    let query = r#"{
        "col_ix": 0,
        "given": {
            "conditions" : [
                [1, {"categorical": 0}],
                [2, {"categorical": 0}]
            ]
        },
        "uncertainty_type": null,
        "state_ixs": [ 1 ]
    }"#;
    let body = post_resp("/api/v1/predict", Some(query), None).await;
    let res: v1::PredictResponse = serde_json::from_str(body.as_str()).unwrap();
    assert!(res.value.is_categorical());
    assert!(res.uncertainty.is_none());
}

#[test_log::test(tokio::test)]
async fn predict_given_with_uncertainty() {
    let query = r#"{
        "col_ix": 0,
        "given": {
            "conditions" : [
                [1, {"categorical": 0}],
                [2, {"categorical": 0}]
            ]
        },
        "uncertainty_type": "js_divergence"
    }"#;
    let body = post_resp("/api/v1/predict", Some(query), None).await;
    let res: v1::PredictResponse = serde_json::from_str(body.as_str()).unwrap();
    assert!(res.value.is_categorical());
    assert!(res.uncertainty.is_some());
}

#[test_log::test(tokio::test)]
async fn predict_oob_col_ix_returns_status_400() {
    let query = r#"{
        "col_ix": 5001,
        "given": {
            "conditions" : [
                [1, {"categorical": 0}],
                [2, {"categorical": 0}]
            ]
        }
    }"#;
    post_resp(
        "/api/v1/predict",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
}

#[test_log::test(tokio::test)]
async fn predict_oob_given_ix_returns_status_400() {
    let query = r#"{
        "col_ix": 0,
        "given": {
            "conditions" : [
                [1, {"categorical": 0}],
                [2001, {"categorical": 0}]
            ]
        }
    }"#;
    post_resp(
        "/api/v1/predict",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
}

#[test_log::test(tokio::test)]
async fn predict_oob_given_data_type_returns_status_400() {
    let query = r#"{
        "col_ix": 0,
        "given": {
            "conditions" : [
                [1, {"categorical": 0}],
                [2, {"continuous": 10.3}]
            ]
        }
    }"#;
    post_resp(
        "/api/v1/predict",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
}
