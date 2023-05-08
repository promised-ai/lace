use clap::{Arg, ArgAction, ArgMatches, Command};
use env_logger::Env;
use lace_preprocess_mdbook_yaml::YamlTester;
use log::{warn, LevelFilter};
use mdbook::errors::Error;
use mdbook::preprocess::{CmdPreprocessor, Preprocessor};
use semver::{Version, VersionReq};
use std::io;
use std::process;
use std::str::FromStr;

pub fn make_app() -> Command {
    Command::new("yaml-check-preprocessor")
        .about("A mdbook preprocessor which performs special tests on YAML code blocks with a specific attribute")
        .arg(Arg::new("log-level").short('l').long("log-level").help("Set the log level (overrides RUST_LOG)").action(ArgAction::Set).value_name("LOG_LEVEL"))
        .subcommand(
            Command::new("supports")
                .arg(Arg::new("renderer").required(true))
                .about("Check whether a renderer is supported by this preprocessor"),
        )
}

fn main() {
    let matches = make_app().get_matches();

    let env = Env::default().default_filter_or("info");
    let mut builder = env_logger::Builder::from_env(env);
    if let Some(level) = matches.get_one::<String>("log-level") {
        let level = LevelFilter::from_str(level).expect("Invalid log level");
        builder.filter(None, level);
    }
    builder.init();

    let preprocessor = YamlTester::new();

    if let Some(sub_args) = matches.subcommand_matches("supports") {
        handle_supports(&preprocessor, sub_args);
    } else if let Err(e) = handle_preprocessing(&preprocessor) {
        eprintln!("{}", e);
        process::exit(1);
    }
}

fn handle_preprocessing(pre: &dyn Preprocessor) -> Result<(), Error> {
    let (ctx, book) = CmdPreprocessor::parse_input(io::stdin())?;

    let book_version = Version::parse(&ctx.mdbook_version)?;
    let version_req = VersionReq::parse(mdbook::MDBOOK_VERSION)?;

    if !version_req.matches(&book_version) {
        warn!(
            "The {} plugin was built against version {} of mdbook, \
             but we're being called from version {}",
            pre.name(),
            mdbook::MDBOOK_VERSION,
            ctx.mdbook_version
        );
    }

    let processed_book = pre.run(&ctx, book)?;
    serde_json::to_writer(io::stdout(), &processed_book)?;

    Ok(())
}

fn handle_supports(pre: &dyn Preprocessor, sub_args: &ArgMatches) -> ! {
    let renderer = sub_args
        .get_one::<String>("renderer")
        .expect("Required argument");
    let supported = pre.supports_renderer(renderer);

    // Signal whether the renderer is supported by exiting with 1 or 0.
    if supported {
        process::exit(0);
    } else {
        process::exit(1);
    }
}
