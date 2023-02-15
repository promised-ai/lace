from pathlib import Path
import polars
from .engine import Engine 


HERE = Path(__file__).resolve().parent
DATASETS = Path(HERE, 'resources', 'datasets')
ANIMALS_PATH = Path(DATASETS, 'animals')
SATELLITES_PATH = Path(DATASETS, 'satellites')


class ExamplePaths:
    def __init__(self, base: Path):
        self.base = base
        self.data = Path(base, 'data.csv')
        self.codebook = Path(base, 'codebook.yaml')
        self.metadata = Path(base, 'metadata.lace')

        if not self.metadata.exists():
            print(f'Generating metadata for data file "{self.data}"')
            engine = Engine(
                data_source=self.data,
                codebook=self.codebook,
                n_states=16,
                rng_seed=1337,
            )
            engine.update(5000, save_path=self.metadata)


class Example(Engine):
    def __init__(self, base: Path):
        paths = ExamplePaths(base)
        super().__init__(metadata=paths.metadata)
        self.df = polars.read_csv(paths.data)


class Animals(Example):
    '''
    A dataset about animals (rows) and their features (columns)
    '''
    def __init__(self):
        super().__init__(ANIMALS_PATH)


class Satellites(Example):
    '''
    A dataset about Earth-orbiting satellites
    '''
    def __init__(self):
        super().__init__(SATELLITES_PATH)
