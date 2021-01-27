import pytest
import json

import numpy as np

from pymultifracs.simul import fbm, mrw


@pytest.fixture(scope="session")
def fbm_file(tmp_path_factory):

    fbm_directory = tmp_path_factory.mktemp("fbm")

    with open('tests/fbm_config.json', 'rb') as f:
        config_list = json.load(f)

    path_list = []

    for i, config in enumerate(config_list):

        config['shape'] = (config['shape'], 100)
        X = fbm(**config)
        path_list.append(fbm_directory / f'fbm_{i}.npy')
        with open(path_list[-1], 'wb') as f:
            np.save(f, X)

    return path_list


@pytest.fixture(scope="session")
def mrw_file(tmp_path_factory):

    mrw_directory = tmp_path_factory.mktemp("mrw")

    with open('tests/mrw_config.json', 'rb') as f:
        config_list = json.load(f)

    path_list = []

    for i, config in enumerate(config_list):

        config['shape'] = (65536, 100)
        config['L'] = 65536
        X = mrw(**config)
        path_list.append(mrw_directory / f'mrw_{i}.npy')
        with open(path_list[-1], 'wb') as f:
            np.save(f, X)

    return path_list
