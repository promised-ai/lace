# Braid Server

The server component of the [`braid`](https://gitlab.com/redpoll/braid) library.

## Install

Installation is unnecessary if you wish to just run the server via pre-prepared
docker image (instructions below). To build `braid_server`, install it either
from the Cloudsmith repository or build it from source.

### Install from Redpoll-Crates Repository

To install braid_server pre-built from the `redpoll-crates` repository, you may
run the following:

```bash
cargo install braid_server --registry redpoll-crates
```

If you do not have this registry configured on your system, you will need
to follow the instructions
[here on Cloudsmith.io](https://cloudsmith.io/~redpoll/repos/crates/setup/#formats-cargo)
in order to do so. If this link does not work, you will need to have your
user added to Redpoll's Cloudsmith repository.

### Build from Source

To build braid_server and its documentation from source, you may do the following:

Build `braid_server`

```console
$ cargo build --release
```

Build documentation

```console
$ cargo doc --all --no-deps
```

Run tests

```console
$ cargo test --all
```

Install binary to system

```console
$ cargo install --path .
```


### Locking braid to a specific machine

To ensure that braid is only run on a specific machine, you may generate a hardware ID.

```console
$ cargo install rp-machine-id --registry redpoll-crates --features cli
$ rp-machine-id
UlAB8wQc6srvhu98uGsRflTZUrnQpseDrkp_9zN91482HYE
```

To lock the binary, pass the ID via the `BRAID_MACHINE_ID` env arg to the
`idlock` feature during compilation

```console
$ BRAID_MACHINE_ID=UlAB8wQc6srvhu98uGsRflTZUrnQpseDrkp_9zN91482HYp cargo build --features idlock
```

Now that binary will only work on the machine with the above ID.


## Running `braid_server` locally

Navigate to the location of the prepared .braid file and run:

```console
$ braid_server <BRAIDFILE>
```

You can also use an encryption key with the `-k/--encryption-key` argument if
you have used encrypted metadata.

```console
$ braid_server -k $MY_KEY <BRAIDFILE>
```

## Using the Container

Building the server can take a very long time. Why not download a pre-built
docker image?

1. Create a Gitlab [personal access token](https://gitlab.com/profile/personal_access_tokens)
   with `read_registry` access
2. run `docker login registry.gitlab.com/redpoll/braid_server`
    - Username is your gitlab username
    - Passowrd is you personal access token
3. cd into the directory where your .braid file is
4. run the following command:

```console
docker run -v `pwd`:/usr/share registry.gitlab.com/redpoll/braid_server:<TAG> /usr/share/<BRAIDFILE>
```

- If you don't have the image, it will be downloaded
- You should be able to access the server on localhost port 8000
- the `-v` command shares the current directory with the container at
  `/user/share`
- NOTE: version `v0.10.0` of the container also requires the
  `--allowed-org="*"` argument

**Example**: v0.10.1 on pybraid animals example

```bash
$ cd ~/pybraid/examples/animals
$ docker run -v `pwd`:/usr/share registry.gitlab.com/redpoll/braid_server:v0.10.1 --no-auth /usr/share/animals.braid
```
