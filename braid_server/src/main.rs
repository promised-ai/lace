use clap::Parser;
use std::{net::Ipv4Addr, path::PathBuf, sync::Arc};
use tracing_subscriber::fmt::format::FmtSpan;
use utoipa::OpenApi;
use warp::{hyper::StatusCode, Filter};

const BYTES_PER_MB: u64 = 1024 * 1024;

#[cfg(feature = "idlock")]
const MACHINE_ID: &str = env!("BRAID_MACHINE_ID");

#[cfg(feature = "idlock")]
const LOCK_DATE: &str = env!("BRAID_LOCK_DATE", "");

#[cfg(feature = "idlock")]
fn validate_date() {
    use chrono::NaiveDate;

    if LOCK_DATE.is_empty() {
        return;
    }
    let lock_date = NaiveDate::parse_from_str(LOCK_DATE, "%Y-%m-%d").unwrap();
    let today = chrono::offset::Local::today().naive_local();

    if today > lock_date {
        panic!("License expired")
    }
}

#[derive(Parser, Debug)]
#[clap(name = "braid-server", about = "Braid oracle server", author, version)]
struct Args {
    /// Port number
    #[clap(short, long, default_value = "8000")]
    pub port: u16,
    /// Max allowable size of JSON in megabytes
    #[clap(long, default_value = "100")]
    pub json_limit: u64,
    /// Path to braidfile
    #[clap(parse(from_os_str))]
    pub path: PathBuf,
    /// Optional 32-byte hex encryption key for use with encrypted metadata
    #[clap(short, long)]
    pub encryption_key: Option<String>,
    /// Make the internal engine mutable
    #[clap(long)]
    pub mutable: bool,
}

#[cfg(not(feature = "idlock"))]
fn validate_machine_id() {}

#[cfg(feature = "idlock")]
fn validate_machine_id() {
    use rp_machine_id::{check_id, MachineIdVersion};
    check_id(MACHINE_ID, MachineIdVersion::V1).unwrap();
    validate_date()
}

#[tokio::main]
async fn main() {
    // Set up Logging
    let filter = std::env::var("RUST_LOG")
        .unwrap_or_else(|_| "tracing=info,warp=warn".to_owned());

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_span_events(FmtSpan::CLOSE)
        .init();

    validate_machine_id();

    let utoipa_config =
        Arc::new(utoipa_swagger_ui::Config::from("/api-doc.json"));

    #[derive(OpenApi)]
    #[openapi(
        handlers(),
        components(),
        tags(
            (name = "braid_server", description = "Braid as an HTTP endpoint")
        )
    )]
    struct ApiDoc;

    let args = Args::parse();

    let api_doc = warp::path("api-doc.json")
        .and(warp::get())
        .map(|| warp::reply::json(&ApiDoc::openapi()));

    let swagger_ui = warp::path("swagger-ui")
        .and(warp::get())
        .and(warp::path::tail())
        .and(warp::any().map(move || utoipa_config.clone()))
        .and_then(serve_swagger);

    warp::serve(
        api_doc
            .or(swagger_ui)
            .or(braid_server::server::warp(
                &args.path,
                args.mutable,
                args.encryption_key,
                args.json_limit * BYTES_PER_MB,
            ))
            .with(warp::trace::request()),
    )
    .run((Ipv4Addr::UNSPECIFIED, args.port))
    .await
}

async fn serve_swagger(
    tail: warp::path::Tail,
    config: Arc<utoipa_swagger_ui::Config<'static>>,
) -> Result<Box<dyn warp::Reply + 'static>, warp::Rejection> {
    let path = tail.as_str();
    match utoipa_swagger_ui::serve(path, config) {
        Ok(file) => {
            if let Some(file) = file {
                Ok(Box::new(
                    warp::hyper::Response::builder()
                        .header("Content-Type", file.content_type)
                        .body(file.bytes),
                ))
            } else {
                Ok(Box::new(StatusCode::NOT_FOUND))
            }
        }
        Err(error) => Ok(Box::new(
            warp::hyper::Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(error.to_string()),
        )),
    }
}
