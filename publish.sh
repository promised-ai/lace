#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

publish_crate() {
    crate_name=$1
    crate_path=$2

    echo "== Publishing ${crate_name} =="
    cd $crate_path
    cargo publish --registry redpoll-crates
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
publish_subcrate braid_data
publish_subcrate braid_stats
publish_subcrate braid_geweke
publish_subcrate braid_codebook
publish_subcrate braid_cc
publish_subcrate braid_metadata

publish_crate braid $DIR
