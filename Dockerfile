FROM rust:1.42-alpine as base
WORKDIR /rp
# ARG CLOUDSMITH_KEY
# ENV CLOUDSMITH_KEY=${CLOUDSMITH_KEY}
ADD . .
ENV CARGO_HOME=/cargo
ENV RUSTFLAGS="-C target-feature=-crt-static"

RUN apk add --no-cache sqlite-dev musl-dev && \
    mkdir $CARGO_HOME && touch ${CARGO_HOME}/config && \
    echo "[registries]" >> ${CARGO_HOME}/config && \
    echo "redpoll-crates = { index = \"https://dl.cloudsmith.io/${CLOUDSMITH_KEY}/redpoll/crates/cargo/index.git\" }" >> ${CARGO_HOME}/config

RUN cargo build --release


FROM alpine:3.11
RUN apk add --no-cache libgcc
COPY --from=base /rp/target/release/braid /bin/braid
ENTRYPOINT ["/bin/braid"]
