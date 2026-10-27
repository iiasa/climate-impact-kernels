import dask
import pytest

from .world import World

# The kernels on the small made-up inputs are tiny; dask's threaded scheduler spends most of
# its time on thread hand-offs. The synchronous scheduler returns identical values.
dask.config.set(scheduler="synchronous")


@pytest.fixture(scope="session")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    return World(tmp_path_factory.mktemp("regression_world"))
