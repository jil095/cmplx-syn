"""
Analyzing all experiments for the paper.
"""
import numpy as np
import time
import roblib
import experiments
import os.path
import settings
from pathlib import Path
import re
from analysis import compute_perf

def analysis_module(model_paths,verbose=True, force_params=None):
    def seed2none(model_name):
        return re.sub(r'seed-\d{1,2}', 'seed', str(model_name))
    for model_path in model_paths:
        if 'seed-0' in str(model_path):
            model_path_template = seed2none(model_path)
            model_paths_same_seed = []
            for model_path_cmp in model_paths:
                if seed2none(model_path_cmp) == model_path_template:
                    model_paths_same_seed.append(model_path_cmp)
            # analyses
            compute_perf(Path(model_path_template), model_paths_same_seed, verbose=verbose, force_params=force_params)

def satisfy_filters(name_filter, x):
    x=str(x)
    if '+' in name_filter:
        name_filters=name_filter.split('+')
    else:
        name_filters=[name_filter]
    for nf in name_filters:
        if nf not in x:
            return False
    return True


def get_model_path_in_exp(exp_name, name_filter=''):
    path = settings.MODELPATH / Path(exp_name)
    model_paths = [x for x in path.iterdir() if x.is_dir() and satisfy_filters(name_filter, str(x))]
    if len(model_paths)==0:
        print('No models found at ',path)
        raise ValueError()
    return model_paths


def analyze_path(exp_name, name_filter, force_params=None):
    analysis_module(get_model_path_in_exp(exp_name, name_filter), force_params=force_params)

def vary_len_b7(name_filter=''):
    analyze_path('vary_len_b7', name_filter)

def vary_len_b8(name_filter=''):
    analyze_path('vary_len_b8', name_filter)

def monitor_snr_500_128levels(name_filter=''):
    analyze_path('monitor_snr_500_128levels', name_filter)

def vary_dim(name_filter=''):
    analyze_path('vary_dim', name_filter)

def big_simple(name_filter=''):
    force_params = {'SNR_thre': 0.5}
    analyze_path('big_simple', name_filter, force_params)
    analyze_path('big_simple_prob', name_filter, force_params)

def fair_comp_2048(name_filter=''):
    force_params = {'SNR_thre': 0.5}
    analyze_path('sparse_weight', name_filter, force_params)
    analyze_path('big_simple2048_prob', name_filter, force_params)

def co_vary(name_filter=''):
    force_params = {'SNR_thre': 0.5}
    analyze_path('co_vary', name_filter, force_params)
    analyze_path('vary_len_b7', name_filter, force_params)
    analyze_path('vary_len_b8', name_filter, force_params)

def co_vary_6_facefillin(name_filter=''):
    force_params = {'SNR_thre': 0.5}
    analyze_path('co_vary_6_facefillin', name_filter, force_params)

def big_simple_prob(name_filter=''):
    analyze_path('big_simple_prob', name_filter)

def all(name_filter=''):
    path = settings.MODELPATH
    [analyze_path(os.path.split(x)[-1], name_filter) for x in path.iterdir() if x.is_dir()]