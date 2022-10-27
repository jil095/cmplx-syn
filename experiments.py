"""
All experiments for the paper.
config: the baseline config.
config_ranges: the config items to be varied.
"""
from utils import vary_config
import numpy as np
import pprint
#from birdseye import eye

def co_vary9():
    # only for b=9 N=1024; otherwise taking too much time!
    beakers = [9]
    seed_num = 10
    config={
        'save_path':'co_vary',
        'isdiscrete': True,
        'burnin_num': 2*10**6,
        'fillin_num': 2*10**6,
        'sample_num': 400,
    }
    configs=[]
    for b in beakers:
        config_ranges = {
            'dim_num': [2**(b+1)],
            'beaker_num': [b],
            'pattern_type': ['face',
                             'rand'],
            'seed': range(seed_num),
        }
        configs += vary_config(config, config_ranges, mode='combinatorial')

    return configs


def co_vary_10_sparse_weight():
    seed_num = 15
    b=10
    config={
        'save_path':'sparse_weight',
        'isdiscrete': True,
        'burnin_num': 2*10**6,
        'fillin_num': 2*10**6,
        'sample_num': 140,
    }
    configs=[]

    config_ranges = {
        'dim_num': [2**(b+1)],
        'beaker_num': [b],
        'pre_feature_ratio': [0.1],
        'pattern_type': ['face',
                     'rand'
        ],
        'seed': range(seed_num),
    }
    configs += vary_config(config, config_ranges, mode='combinatorial')

    return configs

def co_vary10():
    # only for b=10 N=2048; otherwise taking too much time!
    beakers = [10]
    seed_num = 30
    config={
        'save_path':'co_vary',
        'isdiscrete': True,
        'burnin_num': 2*10**6,
        'fillin_num': 2*10**6,
        'sample_num': 70,
    }
    configs=[]
    for b in beakers:
        config_ranges = {
            'dim_num': [2**(b+1)],
            'beaker_num': [b],
            'pattern_type': ['face',
                             'rand'],
            'seed': range(10, seed_num),
        }
        configs += vary_config(config, config_ranges, mode='combinatorial')

    return configs


def co_vary_6_facefillin():
    beakers = [6]
    seed_num = 5
    config={
        'save_path':'co_vary_6_facefillin',
        'isdiscrete': True,
        'burnin_num': 2*10**5,
        'fillin_num': 2*10**5,
        'sample_num': 2000,
        'face_fillin_pattern': True,

    }
    configs=[]
    for b in beakers:
        config_ranges = {
            'dim_num': [2**(b+1)],
            'beaker_num': [b],
            'pattern_type': ['face',
                             'rand'
                             ],
            'seed': range(seed_num),
        }
        configs += vary_config(config, config_ranges, mode='combinatorial')

    return configs


def co_vary_6to1():
    beakers = [6, 5, 4, 3, 2, 1]
    seed_num = 5
    config={
        'save_path':'co_vary',
        'isdiscrete': True,
        'burnin_num': 2*10**5,
        'fillin_num': 2*10**5,
        'sample_num': 2000,
    }
    configs=[]
    for b in beakers:
        config_ranges = {
            'dim_num': [2**(b+1)],
            'beaker_num': [b],
            'pattern_type': ['face',
                             'rand'],
            'seed': range(seed_num),
        }
        configs += vary_config(config, config_ranges, mode='combinatorial')

    return configs

def big_simple():
    seed_num = 15
    config={
        'save_path':'big_simple',
        'isdiscrete': True,
        'beaker_num': 1,
        'burnin_num': 8*10**5,
        'fillin_num': 8*10**5,
        'sample_num': 200,
        'dim_num': 1448,
    }
    config_ranges = {
        'pattern_type': ['face',
                         'rand'],
        'seed': range(seed_num),
    }
    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs

def big_simple2048_prob():
    seed_num = 15
    config={
        'save_path':'big_simple2048_prob',
        'isdiscrete': True,
        'beaker_num': 1,
        'burnin_num': 10*10**5,
        'fillin_num': 10*10**5,
        'sample_num': 140,
        'dim_num': 2048,
    }
    config_ranges = {
        'pattern_type': ['face',
                         'rand',
        ],
        'prob_encoding': [
            1, 0.128, 0.01,
            ],
        'seed': range(seed_num),
    }
    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs

def big_simple_prob():
    seed_num = 15
    config={
        'save_path':'big_simple_prob',
        'isdiscrete': True,
        'beaker_num': 1,
        'burnin_num': 8*10**5,
        'fillin_num': 8*10**5,
        'sample_num': 200,
        'dim_num': 1448,
    }
    config_ranges = {
        'pattern_type': ['face',
                         'rand'],
        'prob_encoding': [
            0.128, 0.01, 0.005,
            ],
        'seed': range(seed_num),
    }
    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs

def big_simple_all():
    return big_simple()+big_simple_prob()

def vary_len_b7():
    seed_num = 6
    config={
        'save_path':'vary_len_b7',
        'dim_num': 256,
        'isdiscrete': True,
        'beaker_num': 7,
        'burnin_num': 2*10**5,
        'fillin_num': 2*10**5,
        'sample_num': 2500,
    }

    config_ranges = {
        'beaker_num':[7, 6, 5, 4, 3, 2, 1],
        'pattern_type': ['face', 'rand'],
        'seed': range(seed_num),
    }

    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs

def vary_len_b8():
    seed_num = 15
    config={
        'save_path':'vary_len_b8',
        'dim_num': 512,
        'isdiscrete': True,
        'beaker_num': 8,
        'burnin_num': 8*10**5,
        'fillin_num': 8*10**5,
        'sample_num': 1000,
    }

    config_ranges = {
        'beaker_num':[8, 7, 6, 5, 4, 3, 2, 1],
        'pattern_type': ['face', 'rand'],
        'seed': range(seed_num),
    }

    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs

def vary_dim():
    seed_num = 3
    config={
        'save_path':'vary_dim',
        'isdiscrete': True,
        'beaker_num': 8,
        'burnin_num': 4*10**5,
        'fillin_num': 2*10**5,
        'sample_num': 2500,
    }

    config_ranges = {
        'dim_num':[256, 128, 64, 32, 16, 8, 4],
        'pattern_type': ['face', 'rand'],
        'seed': range(seed_num),
    }
    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs

def vary_len():
    return vary_len_b7()+vary_len_b8()



def monitor_snr():
    config = {
        'save_path': 'monitor_snr',
        'dim_num': 64,
        'isdiscrete': True,
        'burnin_num': 20000,
        'rpt': 30, # target re-present time
        # 'fillin_num': 200000,
        'sample_num': 100,
    }

    config_ranges = {
        'beaker_num': [7,6,5,1],
        'pattern_type': ['rand'],
        'seed': range(10),
    }

    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs


def monitor_snr_500_128levels():
    config = {
        'save_path': 'monitor_snr_500_128levels',
        'dim_num': 64,
        'isdiscrete': True,
        'burnin_num': 40000,
        'level_num': 128,
        'rpt': 500, # target re-present time
        # 'fillin_num': 200000,
        'sample_num': 100,
        'monitor_threshold':0.5,
    }

    config_ranges = {
        'beaker_num': [8,7,6,5,4,3,2,1],
        'pattern_type': ['rand'],
        'seed': range(10),
    }

    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs

def monitor_snr_300():
    config = {
        'save_path': 'monitor_snr_300',
        'dim_num': 64,
        'isdiscrete': True,
        'burnin_num': 40000,
        'rpt': 300, # target re-present time
        # 'fillin_num': 200000,
        'sample_num': 100,
        'monitor_threshold':0.5,
    }

    config_ranges = {
        'beaker_num': [8,7,6,5,4,3,2,1],
        'pattern_type': ['rand'],
        'seed': range(10),
    }

    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs


def monitor_snr_50():
    config = {
        'save_path': 'monitor_snr_50',
        'dim_num': 64,
        'isdiscrete': True,
        'burnin_num': 40000,
        'rpt': 50, # target re-present time
        # 'fillin_num': 200000,
        'sample_num': 100,
        'monitor_threshold':0.5,
    }

    config_ranges = {
        'beaker_num': [8,7,6,5,1],
        'pattern_type': ['rand'],
        'seed': range(10),
    }

    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs


if __name__ == '__main__':
    pass
