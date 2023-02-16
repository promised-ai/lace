from pathlib import Path
import polars
from .engine import Engine


RESOURCES = Path("resources")
ANIMALS_PATH = Path(RESOURCES, "animals")
SATELLITES_PATH = Path(RESOURCES, "animals")


class __ExamplePaths:
    def __init__(self, base: Path):
        self.base = base
        self.data = Path(base, "data.csv.gz")
        self.metadata = Path(base, "metadata.lace")


class Example(Engine):
    def __init__(self, base: Path):
        paths = __ExamplePaths(base)
        super().__init__(metadata=paths.metadata)
        self.df = polars.read_csv(paths.data)


class Animals(Engine):
    """
    A dataset about animals (rows) and their features (columns)
    """

    def __init__(self):
        super().__init__(ANIMALS_PATH)


class Satellites(Engine):
    """
    A dataset about Earth-orbiting satellites
    """

    def __init__(self):
        super().__init__(SATELLITES_PATH)
