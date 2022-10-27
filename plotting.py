"""
Plotting all experiments for the paper.
"""

import numpy as np
import roblib
import settings
from pathlib import Path
import pandas as pd
from utils import load_config
from plot_maker import plot_module

def naming_unfolding(str_template, unfolding_specs):
    cur_strings = [str_template]
    for idx, spec in enumerate(unfolding_specs):
        if len(spec) == 2:
            re_match, var_range = spec
            colors = [None]*len(var_range)
        else:
            assert idx+1==len(unfolding_specs) # only the last variable can specify colors
            re_match, var_range, colors = spec
        assert re_match[-2:]=='-?'
        re_replace = re_match[:-1]
        new_strings = []
        for cur_str in cur_strings:
            for var, color in zip(var_range, colors):
                s = cur_str.replace(re_match, re_replace+str(var))
                new_strings.append(s if color is None else (s, var, color))
        cur_strings = new_strings
    print(cur_strings)
    return cur_strings

def model_iter_beakers(str_template, pattern_types, b_max):
    compared_models = []
    for f in pattern_types:
        for b in range(1,b_max+1):
            compared_models.append([str_template.replace('beak-?', f'beak-{b}').replace('patt-?', f'patt-{f}'),
                b, 1-b/b_max]) # path, id, color
    return compared_models


def vary_len_b8(name_filter=''): # Vary synaptic complexity
    plot_module(
        'vary_len_b8',
        naming_unfolding('vary_len_b8/model_beak-?_patt-?_seed',
                         [('patt-?', ['face', 'rand']),
                          ('beak-?', np.arange(1, 9), 1 - np.arange(1, 9) / 8)]),
        {'legendtitle': 'Synaptic \ncomplexity','tstar_type': 'linear',
         'no_regr_models': {
             ('face', 'same'): [7,8],
             ('face', 'noisy'): [6,7, 8],
             ('rand', 'same'): [7, 8],
         }
         },
    )

def co_vary_6_facefillin(name_filter=''): # Compare filling effects of real and artificial face patterns
    plot_module(
        'face_fillin',
        [
            ('co_vary/model_dim_-128_beak-6_patt-face_seed', 'recent random', 1 / 7),
            ('co_vary_6_facefillin/model_dim_-128_beak-6_patt-face_seed', 'recent face', 6 / 7),
         ],
        {'legendtitle': '',
         },
    )


def vary_dim(name_filter=''): # Vary number of neurons
    plot_module(
        'vary_dim',
        [
            ('vary_dim/model_dim_-16_patt-face_seed', 16, 6/6),
            ('vary_dim/model_dim_-32_patt-face_seed', 32, 5/6),
            ('vary_dim/model_dim_-64_patt-face_seed', 64, 4/6),
            ('vary_dim/model_dim_-128_patt-face_seed', 128, 3/6),
            ('vary_dim/model_dim_-256_patt-face_seed', 256, 2/6),
            ('vary_len_b8/model_beak-8_patt-face_seed',512,1/6),
            ('vary_dim/model_dim_-16_patt-rand_seed', 16, 6/6),
            ('vary_dim/model_dim_-32_patt-rand_seed', 32, 5/6),
            ('vary_dim/model_dim_-64_patt-rand_seed', 64, 4/6),
            ('vary_dim/model_dim_-128_patt-rand_seed', 128, 3/6),
            ('vary_dim/model_dim_-256_patt-rand_seed', 256, 2/6),
            ('vary_len_b8/model_beak-8_patt-rand_seed',512,1/6),
         ],
        {'legendtitle': 'Number of\nneurons','tstar_type': 'log',
         },
    )

def co_vary(name_filter=''): # Vary number of neurons and synaptic complexity
    plot_module(
        'co_vary',
        [
            ('co_vary/model_dim_-16_beak-3_patt-rand_seed', 16,  7 / 7),
            ('co_vary/model_dim_-32_beak-4_patt-rand_seed', 32,  6 / 7),
            ('co_vary/model_dim_-64_beak-5_patt-rand_seed', 64,  5 / 7),
            ('co_vary/model_dim_-128_beak-6_patt-rand_seed', 128,  4 / 7),
            ('vary_len_b7/model_beak-7_patt-rand_seed', 256, 3 / 7),
            ('vary_len_b8/model_beak-8_patt-rand_seed', 512, 2 / 7),
            ('co_vary/model_dim_-1024_beak-9_patt-rand_seed', 1024, 1 / 7),
            ('co_vary/model_dim_-2048_beak-10_patt-rand_seed', 2048, 0 / 7),

            ('co_vary/model_dim_-16_beak-3_patt-face_seed', 16, 7 / 7),
            ('co_vary/model_dim_-32_beak-4_patt-face_seed', 32, 6 / 7),
            ('co_vary/model_dim_-64_beak-5_patt-face_seed', 64, 5 / 7),
            ('co_vary/model_dim_-128_beak-6_patt-face_seed', 128, 4 / 7),
            ('vary_len_b7/model_beak-7_patt-face_seed', 256, 3 / 7),
            ('vary_len_b8/model_beak-8_patt-face_seed', 512, 2 / 7),
            ('co_vary/model_dim_-1024_beak-9_patt-face_seed', 1024, 1 / 7),
            ('co_vary/model_dim_-2048_beak-10_patt-face_seed', 2048, 0 / 7),
         ],
        {'legendtitle': 'Number of\nneurons', 'tstar_type': 'log', 'init_type': 'log',
         #'additional': ['ioSNR_seedwise','rSNR_seedwise','FC_seedwise','FD_seedwise'],
         #'skip_default_metric': True,
         'no_regr_models': {('face', 'noisy'): [2048]},
         #'has_legend': False,
         #'no_curve_plot': True,
         }
    )


def fair_comp(name_filter=''): # The first fair comparison
    compared_models = [['big_simple/model_patt-face_seed', 'Simple, q=1', 1],
                       ['big_simple/model_patt-rand_seed', 'Simple, q=1', 1],
                       ['big_simple_prob/model_patt-face_prob-0.128_seed', 'Simple, q=0.128', 0.7],
                       ['big_simple_prob/model_patt-rand_prob-0.128_seed', 'Simple, q=0.128', 0.7],
                       ['big_simple_prob/model_patt-face_prob-0.01_seed', 'Simple, q=0.01', 0.4],
                       ['big_simple_prob/model_patt-rand_prob-0.01_seed', 'Simple, q=0.01', 0.4],
                       ['vary_len_b8/model_beak-8_patt-face_seed', 'Complex, q=1', 0.0],
                       ['vary_len_b8/model_beak-8_patt-rand_seed', 'Complex, q=1', 0.0],]
    plot_params = {'legendtitle': '', 'tstar_type': 'bar'}
    plot_module('fair_comp', compared_models, plot_params)


def fair_comp_2048(name_filter=''):  # The second fair comparison
    compared_models = [
                       ['big_simple2048_prob/model_patt-face_prob-1_seed', 'Simple, q=1', 1],
                       ['big_simple2048_prob/model_patt-rand_prob-1_seed', 'Simple, q=1', 1],
                       ['big_simple2048_prob/model_patt-face_prob-0.128_seed', 'Simple, q=0.128', 0.7],
                       ['big_simple2048_prob/model_patt-rand_prob-0.128_seed', 'Simple, q=0.128', 0.7],
                       ['big_simple2048_prob/model_patt-face_prob-0.01_seed', 'Simple, q=0.01', 0.4],
                       ['big_simple2048_prob/model_patt-rand_prob-0.01_seed', 'Simple, q=0.01', 0.4],
                       ['sparse_weight/model_dim_-2048_beak-10_pre_-0.1_patt-face_seed', 'Complex, q=1', 0.0],
                       ['sparse_weight/model_dim_-2048_beak-10_pre_-0.1_patt-rand_seed', 'Complex, q=1', 0.0],
                    ]
    plot_params = {'legendtitle': '', 'tstar_type': 'bar', 'has_legend': False}
    plot_module('fair_comp_2048', compared_models, plot_params)
