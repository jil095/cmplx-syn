import os
import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import settings

MODELPATH = settings.MODELPATH


def save_config(config, path):
    config['save_path'] = str(config['save_path'])  # from Path to plain string
    os.makedirs(MODELPATH / path, exist_ok=True)
    with open(MODELPATH / path / 'config.json', 'w') as f:
        json.dump(config, f)


def _load_config(path):
    """Load config with absolute path to config."""
    with open(path, 'r') as f:
        config = json.load(f)
    config['save_path'] = Path(config['save_path'])
    return config


def load_config(path):
    path = Path(path)
    # Search consecutively
    paths = []
    if str(path)[-4:] == 'json':
        paths.append(path)
    paths += [path / 'config.json', MODELPATH / path / 'config.json']
    for p in paths:
        try:
            return _load_config(p)
        except (FileNotFoundError, IsADirectoryError):
            continue
    raise FileNotFoundError


def vary_config(base_config, config_ranges, mode, name_keys=None):
    """Return configurations.

    Args:
        base_config: dict, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }
        mode: str, can take 'combinatorial', 'sequential', and 'control'

    Return:
        configs: a list of config dict [config1, config2, ...]
    """
    if mode == 'combinatorial':
        _vary_config = _vary_config_combinatorial
    elif mode == 'sequential':
        _vary_config = _vary_config_sequential
    elif mode == 'control':
        _vary_config = _vary_config_control
    else:
        raise ValueError('Unknown mode {}'.format(str(mode)))
    configs, config_diffs = _vary_config(base_config, config_ranges)
    # Automatic set names for configs
    configs = autoname(configs, config_diffs, name_keys=name_keys)
    return configs


def autoname(configs, config_diffs, name_keys=None):
    """Helper function for automatically naming models based on configs."""
    new_configs = list()
    for config, config_diff in zip(configs, config_diffs):
        name = 'model'
        for key, val in config_diff.items():
            if isinstance(val,list) or isinstance(val,tuple):
                str_val = ''
                for cur in val:
                    str_val += str(cur)
            else:
                str_val = str(val)
            if (name_keys is None) or (key in name_keys):
                str_key = str(key)
                name += '_' + str_key[:4] + '-' + str_val
        config['save_path'] = Path(config['save_path']) / name
        new_configs.append(config)
    return new_configs


def _vary_config_combinatorial(base_config, config_ranges):
    """Return combinatorial configurations.

    Args:
        base_config: dict, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }

    Return:
        configs: a list of config dict [config1, config2, ...]
            Loops over all possible combinations of hp1, hp2, ...
        config_diffs: a list of config diff from base_config
    """
    # Unravel the input index
    keys = config_ranges.keys()
    dims = [len(config_ranges[k]) for k in keys]
    n_max = int(np.prod(dims))

    configs, config_diffs = list(), list()
    for i in range(n_max):
        config_diff = dict()
        indices = np.unravel_index(i, shape=dims)
        # Set up new config
        for key, index in zip(keys, indices):
            config_diff[key] = config_ranges[key][index]
        config_diffs.append(config_diff)

        new_config = deepcopy(base_config)
        new_config.update(config_diff)
        configs.append(new_config)
    return configs, config_diffs


def _vary_config_sequential(base_config, config_ranges):
    """Return sequential configurations.

    Args:
        base_config: dict, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }

    Return:
        configs: a list of config dict [config1, config2, ...]
            Loops over all hyperparameters hp1, hp2 together sequentially
        config_diffs: a list of config diff from base_config
    """
    keys = config_ranges.keys()
    dims = [len(config_ranges[k]) for k in keys]
    n_max = dims[0]

    configs, config_diffs = list(), list()
    for i in range(n_max):
        config_diff = dict()
        for key in keys:
            config_diff[key] = config_ranges[key][i]
        config_diffs.append(config_diff)

        new_config = deepcopy(base_config)
        new_config.update(config_diff)
        configs.append(new_config)

    return configs, config_diffs


def _vary_config_control(base_config, config_ranges):
    """Return sequential configurations.

    Args:
        base_config: dict, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }

    Return:
        configs: a list of config dict [config1, config2, ...]
            Loops over all hyperparameters hp1, hp2 independently
        config_diffs: a list of config diff from base_config
    """
    # Unravel the input index
    keys = list(config_ranges.keys())
    dims = [len(config_ranges[k]) for k in keys]
    n_max = int(np.sum(dims))

    configs, config_diffs = list(), list()
    for i in range(n_max):
        index = i
        for j, dim in enumerate(dims):
            if index >= dim:
                index -= dim
            else:
                break

        config_diff = dict()
        key = keys[j]
        config_diff[key] = config_ranges[key][i]
        config_diffs.append(config_diff)

        new_config = deepcopy(base_config)
        new_config.update(config_diff)
        configs.append(new_config)
    return configs, config_diffs
