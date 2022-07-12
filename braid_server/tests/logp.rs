mod helpers;

use helpers::post_resp;

use braid_server::api::v1;

mod logp {
    use warp::hyper::StatusCode;

    use super::*;
    #[test_log::test(tokio::test)]
    async fn logp_no_given() {
        let query = r#"{
            "values": [
                [{"categorical": 0}, {"categorical": 0}],
                [{"categorical": 1}, {"categorical": 0}],
                [{"categorical": 0}, {"categorical": 1}]
            ],
            "col_ixs": [0, 1]
        }"#;
        let body = post_resp("/api/v1/logp", Some(query), None).await;
        let res: v1::LogpResponse =
            serde_json::from_str(body.as_str()).unwrap();
        assert_eq!(res.logp.len(), 3);
    }

    #[test_log::test(tokio::test)]
    async fn logp_empty_given() {
        let query = r#"{
            "values": [
                [{"categorical": 0}, {"categorical": 0}],
                [{"categorical": 1}, {"categorical": 0}],
                [{"categorical": 0}, {"categorical": 1}]
            ],
            "given": "nothing",
            "col_ixs": [0, 1]
        }"#;
        let body = post_resp("/api/v1/logp", Some(query), None).await;
        let res: v1::LogpResponse =
            serde_json::from_str(body.as_str()).unwrap();
        assert_eq!(res.logp.len(), 3);
    }

    #[test_log::test(tokio::test)]
    async fn logp_given() {
        let query = r#"{
            "values": [
                [{"categorical": 0}, {"categorical": 0}],
                [{"categorical": 1}, {"categorical": 0}],
                [{"categorical": 0}, {"categorical": 1}]
            ],
            "given": {
                "conditions": [
                    [2, {"categorical": 0}],
                    [3, {"categorical": 1}]
                ]
            },
            "col_ixs": [0, 1]
        }"#;
        let body = post_resp("/api/v1/logp", Some(query), None).await;
        let res: v1::LogpResponse =
            serde_json::from_str(body.as_str()).unwrap();
        assert_eq!(res.logp.len(), 3);
    }

    #[test_log::test(tokio::test)]
    async fn logp_oob_col_returns_status_400() {
        let query = r#"{
            "values": [
                [{"categorical": 0}, {"categorical": 0}],
                [{"categorical": 1}, {"categorical": 0}],
                [{"categorical": 0}, {"categorical": 1}]
            ],
            "col_ixs": [0, 1000]
        }"#;
        post_resp("/api/v1/logp", Some(query), Some(StatusCode::BAD_REQUEST))
            .await;
    }

    #[test_log::test(tokio::test)]
    async fn logp_incorrect_data_type_returns_status_400() {
        let query = r#"{
            "values": [
                [{"categorical": 0}, {"categorical": 0}],
                [{"categorical": 1}, {"continuous": 0.12}],
                [{"categorical": 0}, {"categorical": 1}]
            ],
            "col_ixs": [0, 1]
        }"#;
        post_resp("/api/v1/logp", Some(query), Some(StatusCode::BAD_REQUEST))
            .await;
    }

    #[test_log::test(tokio::test)]
    async fn logp_oob_given_col_returns_status_400() {
        let query = r#"{
            "values": [
                [{"categorical": 0}, {"categorical": 0}],
                [{"categorical": 1}, {"categorical": 0}],
                [{"categorical": 0}, {"categorical": 1}]
            ],
            "given": {
                "conditions": [
                    [2, {"categorical": 0}],
                    [3000, {"categorical": 1}]
                ]
            },
            "col_ixs": [0, 1]
        }"#;
        post_resp("/api/v1/logp", Some(query), Some(StatusCode::BAD_REQUEST))
            .await;
    }
}

mod logp_scaled {
    use warp::hyper::StatusCode;

    use super::*;
    #[test_log::test(tokio::test)]
    async fn logp_no_given() {
        let query = r#"{
            "values": [
                [{"categorical": 0}, {"categorical": 0}],
                [{"categorical": 1}, {"categorical": 0}],
                [{"categorical": 0}, {"categorical": 1}]
            ],
            "col_ixs": [0, 1]
        }"#;
        let body = post_resp("/api/v1/logp_scaled", Some(query), None).await;
        let res: v1::LogpScaledResponse =
            serde_json::from_str(body.as_str()).unwrap();
        assert_eq!(res.logp.len(), 3);
    }

    #[test_log::test(tokio::test)]
    async fn logp_empty_given() {
        let query = r#"{
            "values": [
                [{"categorical": 0}, {"categorical": 0}],
                [{"categorical": 1}, {"categorical": 0}],
                [{"categorical": 0}, {"categorical": 1}]
            ],
            "given": "nothing",
            "col_ixs": [0, 1]
        }"#;
        let body = post_resp("/api/v1/logp_scaled", Some(query), None).await;
        let res: v1::LogpScaledResponse =
            serde_json::from_str(body.as_str()).unwrap();
        assert_eq!(res.logp.len(), 3);
    }

    #[test_log::test(tokio::test)]
    async fn logp_given() {
        let query = r#"{
            "values": [
                [{"categorical": 0}, {"categorical": 0}],
                [{"categorical": 1}, {"categorical": 0}],
                [{"categorical": 0}, {"categorical": 1}]
            ],
            "given": {
                "conditions": [
                    [2, {"categorical": 0}],
                    [3, {"categorical": 1}]
                ]
            },
            "col_ixs": [0, 1]
        }"#;
        let body = post_resp("/api/v1/logp_scaled", Some(query), None).await;
        let res: v1::LogpScaledResponse =
            serde_json::from_str(body.as_str()).unwrap();
        assert_eq!(res.logp.len(), 3);
    }

    #[test_log::test(tokio::test)]
    async fn logp_oob_col_returns_status_400() {
        let query = r#"{
            "values": [
                [{"categorical": 0}, {"categorical": 0}],
                [{"categorical": 1}, {"categorical": 0}],
                [{"categorical": 0}, {"categorical": 1}]
            ],
            "col_ixs": [0, 1000]
        }"#;
        post_resp(
            "/api/v1/logp_scaled",
            Some(query),
            Some(StatusCode::BAD_REQUEST),
        )
        .await;
    }

    #[test_log::test(tokio::test)]
    async fn logp_incorrect_data_type_returns_status_400() {
        let query = r#"{
            "values": [
                [{"categorical": 0}, {"categorical": 0}],
                [{"categorical": 1}, {"continuous": 0.12}],
                [{"categorical": 0}, {"categorical": 1}]
            ],
            "col_ixs": [0, 1]
        }"#;
        post_resp(
            "/api/v1/logp_scaled",
            Some(query),
            Some(StatusCode::BAD_REQUEST),
        )
        .await;
    }

    #[test_log::test(tokio::test)]
    async fn logp_oob_given_col_returns_status_400() {
        let query = r#"{
            "values": [
                [{"categorical": 0}, {"categorical": 0}],
                [{"categorical": 1}, {"categorical": 0}],
                [{"categorical": 0}, {"categorical": 1}]
            ],
            "given": {
                "conditions": [
                    [2, {"categorical": 0}],
                    [3000, {"categorical": 1}]
                ]
            },
            "col_ixs": [0, 1]
        }"#;
        post_resp(
            "/api/v1/logp_scaled",
            Some(query),
            Some(StatusCode::BAD_REQUEST),
        )
        .await;
    }
}
