#!/bin/bash
set -e

version() {
    crate_path=$1
    grep -E "version\s?=\s?\"(.+)\"" ${crate_path}/Cargo.toml | grep -oE "\d+\.\d+\.\d+"
}

publish_subcrate() {
    crate_name=$1
    crate_version=$(version $crate_name)
    search_result=$(cloudsmith list packages redpoll/crates -k $CLOUDSMITH_API_KEY -q "${crate_name}" | grep ${crate_version})

    if [-z "$search_result"]
    then
        cd $crate_name
        cargo package
        cd ../
        cloudsmith push cargo -k $CLOUDSMITH_API_KEY $crate_name target/package/${crate_name}-${crate_version}.crate
    fi
}

publish_crate() {
    crate_name=$1
    crate_version=$(version .)
    search_result=$(cloudsmith list packages redpoll/crates -k $CLOUDSMITH_API_KEY -q "${crate_name}" | grep ${crate_version})

    if [-z "$search_result"]
    then
        cargo package
        cloudsmith push cargo -k $CLOUDSMITH_API_KEY $crate_name target/package/${crate_name}-${crate_version}.crate
    fi
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# publish_crate() {
#     crate_name=$1
#     crate_path=$2
# 
#     echo "== Publishing ${crate_name} =="
#     cd $crate_path
#     cargo publish --registry redpoll-crates
# }
# 
# publish_subcrate() {
#     start_dir=`pwd`
#     crate_name=$1
#     crate_path=$DIR/$crate_name
#     publish_crate $crate_name $crate_path
#     cd $start_dir
#     # I guess things in cloudsmith somethings take a while to update, so here
#     # we give them a bit of time
#     sleep 5
# }

publish_subcrate braid_consts
publish_subcrate braid_utils
publish_subcrate braid_flippers
publish_subcrate braid_data
publish_subcrate braid_stats
publish_subcrate braid_geweke
publish_subcrate braid_codebook
publish_subcrate braid_cc
publish_subcrate braid_metadata

publish_crate braid
