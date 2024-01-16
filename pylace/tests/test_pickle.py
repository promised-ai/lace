import pickle

from lace import examples


def test_pickle_engine():
    engine = examples.Animals().engine
    s = pickle.dumps(engine)
    engine_b = pickle.loads(s)

    sim_a = engine.simulate(["swims", "flys"], n=10)
    sim_b = engine_b.simulate(["swims", "flys"], n=10)

    assert sim_a.equals(sim_b)
