#!/bin/bash
set -e

version() {
    crate_path=$1
    grep -E "version\s?=\s?\"(.+)\"" ${crate_path}/Cargo.toml | grep -oE "\d+\.\d+\.\d+"
}

publish_subcrate() {
    crate_name=$1
    subcrate_name=$2
    subcrate_version=$(version $subcrate_name)
    search_result=$(cloudsmith list packages redpoll/crates -k $CLOUDSMITH_API_KEY -q "${subcrate_name}" | grep ${subcrate_version})

    if [-z "$search_result"]
    then
        cd $crate_name/$subcrate_name
        cargo package
        cd ../
        cloudsmith push cargo -k $CLOUDSMITH_API_KEY $subcrate_name target/package/${subcrate_name}-${subcrate_version}.crate
        cd ../
    else
        echo "${crate_name}/${subcrate_name} v${crate_version} exists. Skipping."
    fi
}

publish_crate() {
    crate_name=$1
    crate_version=$(version .)
    search_result=$(cloudsmith list packages redpoll/crates -k $CLOUDSMITH_API_KEY -q "${crate_name}" | grep ${crate_version})

    if [-z "$search_result"]
    then
        cd $crate_name
        cargo package
        cloudsmith push cargo -k $CLOUDSMITH_API_KEY $crate_name target/package/${crate_name}-${crate_version}.crate
        cd ../
    else
        echo "${crate_name} v${crate_version} exists. Skipping."
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

publish_subcrate braid braid_consts
publish_subcrate braid braid_utils
publish_subcrate braid braid_flippers
publish_subcrate braid braid_data
publish_subcrate braid braid_stats
publish_subcrate braid braid_geweke
publish_subcrate braid braid_codebook
publish_subcrate braid braid_cc
publish_subcrate braid braid_metadata

publish_crate braid
publish_crate braid_server
