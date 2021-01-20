from pymultifracs.simul import fbm, mrw
import pytest


@pytest.fixture(scope='session')
def fbm_generate(tmpdir_factory):

    with open('tests/fbm_config.json', 'rb') as f:
        config_list = json.load(f)

    for i, config in enumerate(config_list):

        X = fbm(**config)
        