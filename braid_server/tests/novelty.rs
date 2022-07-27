mod helpers;
use helpers::post_resp;

use braid_server::api::v1;
use warp::hyper::StatusCode;

#[test_log::test(tokio::test)]
async fn novelty_one_row_no_wrt() {
    let query = r#"{
        "row_ixs": [12],
        "wrt": []
    }"#;
    let body = post_resp("/api/v1/novelty", Some(query), None).await;
    let res: v1::NoveltyResponse = serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.novelty.len(), 1);
    assert_eq!(res.novelty[0].0, 12);
    assert!(res.novelty[0].1 > 0.0);
    assert!(res.novelty[0].1 < 1.0);
}

#[test_log::test(tokio::test)]
async fn novelty_no_rows_no_wrt_return_empty() {
    let query = r#"{
        "row_ixs": [],
        "wrt": []
    }"#;
    let body = post_resp("/api/v1/novelty", Some(query), None).await;
    let res: v1::NoveltyResponse = serde_json::from_str(body.as_str()).unwrap();
    assert!(res.novelty.is_empty());
}

#[test_log::test(tokio::test)]
async fn novelty_no_rows_with_wrt_return_empty() {
    let query = r#"{
        "row_ixs": [],
        "wrt": [11]
    }"#;
    let body = post_resp("/api/v1/novelty", Some(query), None).await;
    let res: v1::NoveltyResponse = serde_json::from_str(body.as_str()).unwrap();
    assert!(res.novelty.is_empty());
}

#[test_log::test(tokio::test)]
async fn novelty_multi_row_no_wrt() {
    let query = r#"{
        "row_ixs": [0, 12, 2],
        "wrt": []
    }"#;
    let body = post_resp("/api/v1/novelty", Some(query), None).await;
    let res: v1::NoveltyResponse = serde_json::from_str(body.as_str()).unwrap();

    assert_eq!(res.novelty.len(), 3);
    assert_eq!(res.novelty[0].0, 0);
    assert_eq!(res.novelty[1].0, 12);
    assert_eq!(res.novelty[2].0, 2);

    for (_, val) in res.novelty {
        assert!(val > 0.0);
        assert!(val < 1.0);
    }
}

#[test_log::test(tokio::test)]
async fn novelty_multi_row_no_wrt_empty_same_as_absent() {
    let res_empty: v1::NoveltyResponse = {
        let query = r#"{
            "row_ixs": [0, 12, 2],
            "wrt": []
        }"#;
        let body = post_resp("/api/v1/novelty", Some(query), None).await;
        serde_json::from_str(body.as_str()).unwrap()
    };

    let res_absent: v1::NoveltyResponse = {
        let query = r#"{
            "row_ixs": [0, 12, 2]
        }"#;
        let body = post_resp("/api/v1/novelty", Some(query), None).await;
        serde_json::from_str(body.as_str()).unwrap()
    };

    assert_eq!(res_empty.novelty.len(), 3);
    assert_eq!(res_absent.novelty.len(), 3);

    for ((_, val_empty), (_, val_absent)) in
        res_empty.novelty.iter().zip(res_absent.novelty.iter())
    {
        assert!(val_empty == val_absent);
    }
}

#[test_log::test(tokio::test)]
async fn novelty_one_row_wrt() {
    let query = r#"{
        "row_ixs": [12],
        "wrt": [5]
    }"#;
    let body = post_resp("/api/v1/novelty", Some(query), None).await;
    let res: v1::NoveltyResponse = serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.novelty.len(), 1);
    assert_eq!(res.novelty[0].0, 12);
    assert!(res.novelty[0].1 > 0.0);
    assert!(res.novelty[0].1 < 1.0);
}

#[test_log::test(tokio::test)]
async fn novelty_multi_row_wrt() {
    let query = r#"{
        "row_ixs": [0, 12, 2],
        "wrt": [5]
    }"#;
    let body = post_resp("/api/v1/novelty", Some(query), None).await;
    let res: v1::NoveltyResponse = serde_json::from_str(body.as_str()).unwrap();

    assert_eq!(res.novelty.len(), 3);
    assert_eq!(res.novelty[0].0, 0);
    assert_eq!(res.novelty[1].0, 12);
    assert_eq!(res.novelty[2].0, 2);

    for (_, val) in res.novelty {
        assert!(val > 0.0);
        assert!(val < 1.0);
    }
}

#[test_log::test(tokio::test)]
async fn oob_row_causes_status_400() {
    let query = r#"{
        "row_ixs": [2, 120]
    }"#;
    let resp = post_resp(
        "/api/v1/novelty",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    println!("{}", resp);
    assert!(resp.contains("ix is 120 but there"));
}

#[test_log::test(tokio::test)]
async fn oob_wrt_causes_status_400() {
    let query = r#"{
        "row_ixs": [2],
        "wrt": [199]
    }"#;
    let resp = post_resp(
        "/api/v1/novelty",
        Some(query),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    assert!(resp.contains("ix is 199 but there"));
}
