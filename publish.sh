#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

crate_version() {
    toml_path=$1/Cargo.toml
    version=`grep -oEi "^version = \"(.+)\"" ${toml_path} | grep -Po "\d+\.\d+\.\d+"`
    echo $version
}

publish_crate() {
    crate_name=$1
    crate_path=$2
    version=`crate_version $crate_path`

    echo ""
    echo "==Publishing ${crate_name} v${version}=="
    cd $crate_path
    cargo package -q
    cloudsmith push cargo \
        redpoll/crates \
        $DIR/target/package/${crate_name}-${version}.crate
}

publish_subcrate() {
    start_dir=`pwd`
    crate_name=$1
    crate_path=$DIR/$crate_name
    publish_crate $crate_name $crate_path
    cd $start_dir
}

publish_subcrate braid_consts
publish_subcrate braid_utils
publish_subcrate braid_flippers
publish_subcrate braid_stats
publish_subcrate braid_geweke
publish_subcrate braid_codebook

publish_crate braid $DIR
