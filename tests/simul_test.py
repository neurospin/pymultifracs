import pytest
import json

from pymultifracs.simul import fbm, mrw


def test_fbm():

    with open('tests/fbm_config.json', 'rb') as f:
        config_list = json.load(f)

    for param_set in config_list:
        fbm(**param_set)


def test_fbm_failure():

    with pytest.raises(ValueError):
        fbm(100, H=-1.0)
    with pytest.raises(ValueError):
        fbm(100, H=2.0)


def test_fbm_shape():

    for shape in [1000, 2345, 4096]:
        X = fbm(shape, 0.5)
        assert X.shape[0] == shape

    for shape in [(100, 2), (100, 3), (400, 2)]:
        X = fbm(shape, 0.5)
        assert X.shape == shape


def test_mrw():

    with open('tests/mrw_config.json', 'rb') as f:
        config_list = json.load(f)

    for param_set in config_list:
        mrw(shape=100, L=100, **param_set)


def test_mrw_failure():

    with pytest.raises(ValueError):
        mrw(100, H=-1.0, lam=0.1, L=100)
    with pytest.raises(ValueError):
        mrw(100, H=2.0, lam=0.1, L=100)

    with pytest.raises(ValueError):
        mrw(100, H=0.5, lam=0.1, L=200)


def test_mrw_shape():

    for shape in [100, 1000, 4096]:
        X = mrw(shape, 0.5, 0.1, shape)
        assert X.shape[0] == shape

    for shape in [(100, 2), (100, 3), (400, 2)]:
        X = mrw(shape, 0.5, 0.1, shape[0])
        assert X.shape == shape
