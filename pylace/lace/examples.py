from pathlib import Path
from shutil import rmtree
import polars
from .engine import Engine


HERE = Path(__file__).resolve().parent
DATASETS_PATH = Path(HERE, "resources", "datasets")
ANIMALS = "animals"
SATELLITES = "satellites"
DATA_FILE = "data.csv"
CODEBOOK_FILE = "codebook.yaml"
METADATA_DIR = "metadata.lace"
ANIMALS_PATH = Path(DATASETS_PATH, ANIMALS)
SATELLITES_PATH = Path(DATASETS_PATH, SATELLITES)
EXAMPLE_PATHS = {SATELLITES: SATELLITES_PATH, ANIMALS: ANIMALS_PATH}


class ExamplePaths:
    def __init__(self, name: str):
        if name not in EXAMPLE_PATHS:
            raise ValueError(
                f"Invalid example `{name}`. Valid names are: \
                {EXAMPLE_PATHS.keys()}"
            )
        base = EXAMPLE_PATHS[name]
        self.base = base
        self.data = Path(base, DATA_FILE)
        self.codebook = Path(base, CODEBOOK_FILE)
        self.metadata = Path(base, METADATA_DIR)

        if not self.metadata.exists():
            print(f'Generating metadata for data file "{self.data}"')
            engine = Engine(
                data_source=self.data,
                codebook=self.codebook,
                n_states=16,
                rng_seed=1337,
            )
            engine.update(5000)
            engine.save(self.metadata)


def delete_metadata(name: str):
    """
    Delete the metadata associated with a specific example

    Parameters
    ----------
    name: str
        The example name

    Examples
    --------
    Delete the animals example metadata

    >>> from lace.examples import delete_metadata
    >>> delete_metadata('animals')
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
        super().__init__(metadata=paths.metadata)
        self.df = polars.read_csv(paths.data)


class Animals(Example):
    """
    A dataset about animals (rows) and their features (columns)
    """

    def __init__(self):
        super().__init__(ANIMALS)


class Satellites(Example):
    """
    A dataset about Earth-orbiting satellites
    """

    def __init__(self):
        super().__init__(SATELLITES)
