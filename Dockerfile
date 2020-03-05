FROM rust:1.41 as base
WORKDIR /rp
# ARG CLOUDSMITH_KEY
# ENV CLOUDSMITH_KEY=${CLOUDSMITH_KEY}
ADD . .
ENV CARGO_HOME=/cargo

RUN apt update && apt install -y sqlite3 && \
    mkdir $CARGO_HOME && touch ${CARGO_HOME}/config && \
    echo "[registries]" >> ${CARGO_HOME}/config && \
    echo "redpoll-crates = { index = \"https://dl.cloudsmith.io/${CLOUDSMITH_KEY}/redpoll/crates/cargo/index.git\" }" >> ${CARGO_HOME}/config

RUN cargo build --release


FROM rust:1.41
COPY --from=base /rp/target/release/braid /bin/braid
ENTRYPOINT ["/bin/braid"]
