use std::str::FromStr;
use clap::ArgMatches;

pub fn parse_arg<T: FromStr>(arg_name: &str, matches: &ArgMatches) -> T {
    match matches.value_of(arg_name).unwrap().parse::<T>() {
        Ok(x) => x,
        Err(_) => panic!("Could not parse {}", arg_name),
    }
}
