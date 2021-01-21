from pymultifracs.simul import fbm, mrw
import pytest
import json


@pytest.fixture(scope='session')
def fbm_file(tmpdir_factory):

    with open('tests/fbm_config.json', 'rb') as f:
        config_list = json.load(f)

    for i, config in enumerate(config_list):

        config['shape'] = (config['shape'], 100)
        X = fbm(**config)
        X.