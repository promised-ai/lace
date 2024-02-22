import hashlib
import json
import os
from pathlib import Path
from shutil import rmtree

import polars

from lace.engine import Engine

HERE = Path(__file__).resolve().parent
DATASETS_PATH = Path(HERE, "resources", "datasets")
ANIMALS = "animals"
SATELLITES = "satellites"
DATA_FILE = "data.csv"
HASH_FILE = "checksum.json"
CODEBOOK_FILE = "codebook.yaml"
METADATA_DIR = "metadata.lace"
ANIMALS_PATH = Path(DATASETS_PATH, ANIMALS)
SATELLITES_PATH = Path(DATASETS_PATH, SATELLITES)
EXAMPLE_PATHS = {SATELLITES: SATELLITES_PATH, ANIMALS: ANIMALS_PATH}

QUIET_VAR = "PYLACE_EXAMPLES_QUIET"
AUTOREGEN_VAR = "PYLACE_EXAMPLES_AUTO_REGEN"


def md5checksum(filename: Path) -> str:
    BUF_SIZE = 65536

    md5 = hashlib.md5()

    with open(filename, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)

    return md5.hexdigest()


def generate_metadata(data_src, dst, codebook, quiet=False):
    if not quiet:
        print(f'Generating metadata for data file "{data_src}"')

    df = polars.read_csv(data_src)
    engine = Engine.from_df(
        df,
        codebook,
        n_states=16,
        rng_seed=1337,
    )
    engine.update(5000, quiet=quiet)
    engine.save(dst)


def write_hashes(hashes: dict, dst: Path):
    with open(dst, "w") as f:
        json.dump(hashes, f)


def read_hashes(src: Path):
    if src.exists():
        with open(src) as f:
            return json.load(f)
    else:
        return None


def need_to_regen(hashes, current_hashes) -> bool:
    if current_hashes is None:
        return True

    return any(hashes[k] != current_hashes[k] for k in hashes)


def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question + " (y/n): ")).lower().strip()
        if reply[:1] == "y":
            return True
        if reply[:1] == "n":
            return False


class ExamplePaths:
    def __init__(self, name: str):
        if name not in EXAMPLE_PATHS:
            raise ValueError(
                f"Invalid example `{name}`. Valid names are: \
                {EXAMPLE_PATHS.keys()}"
            )
        base = EXAMPLE_PATHS[name]
        self.base = base
        self.hash = Path(base, HASH_FILE)
        self.data = Path(base, DATA_FILE)
        self.codebook = Path(base, CODEBOOK_FILE)
        self.metadata = Path(base, METADATA_DIR)

        quiet = bool(int(os.environ.get(QUIET_VAR, 0)))
        auto_regen = bool(int(os.environ.get(AUTOREGEN_VAR, 0)))

        # generate codebook and data hashes to monitor whether the examples have
        # changed. If the examples have changed, the user will need to recreate
        # the metadata.
        hashes = {
            "codebook": md5checksum(self.codebook),
            "data": md5checksum(self.data),
        }

        current_hashes = read_hashes(self.hash)

        if not self.metadata.exists():
            generate_metadata(self.data, self.metadata, self.codebook, quiet)
            write_hashes(hashes, self.hash)
        else:
            if need_to_regen(hashes, current_hashes):
                if auto_regen:
                    generate_metadata(
                        self.data, self.metadata, self.codebook, quiet
                    )
                    write_hashes(hashes, self.hash)
                elif yes_or_no(f"{name} metadata is out of date. Regenerate?"):
                    generate_metadata(
                        self.data, self.metadata, self.codebook, quiet
                    )
                    write_hashes(hashes, self.hash)


def delete_metadata(name: str):
    """
    Delete the metadata associated with a specific example.

    Parameters
    ----------
    name: str
        The example name

    Examples
    --------
    Delete the animals example metadata

    >>> from lace.examples import delete_metadata
    >>> delete_metadata("animals")

    """
    if name not in EXAMPLE_PATHS:
        raise ValueError(
            f"Invalid example `{name}`. Valid names are: \
            {EXAMPLE_PATHS.keys()}"
        )
    metadata_path = Path(EXAMPLE_PATHS[name], METADATA_DIR)
    if metadata_path.exists():
        rmtree(metadata_path)


class Example(Engine):
    def __init__(self, name: str):
        paths = ExamplePaths(name)
        engine = Engine.load(paths.metadata)
        super().__init__(engine.engine)
        self.df = polars.read_csv(paths.data)


class Animals(Example):
    """A dataset about animals (rows) and their features (columns)."""

    def __init__(self):
        super().__init__(ANIMALS)


class Satellites(Example):
    """A dataset about Earth-orbiting satellites."""

    def __init__(self):
        super().__init__(SATELLITES)
