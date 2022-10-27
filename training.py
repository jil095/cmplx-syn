"""
General functions for training an experiment.
"""
import numpy as np
import torch
import torch.nn as nn
import time
import settings
import roblib
from models import BCSLayer, Memorynet, set_seed
from datasets import load_aux_patterns, load_traintest_patterns
import experiments
import os.path
import utils
import pprint
import multiprocessing as mp


def train_experiment(experiment,device='cuda'):
    '''
    Training without monitoring SNR
    '''
    print('Training {:s} experiment'.format(experiment))
    configs = getattr(experiments, experiment)()
    for config in configs:
        train(config,device=device)
        if device == 'cuda':
            torch.cuda.empty_cache()


def monitor_process_one_config(config, device):
    monitor(config,device=device)
    if device == 'cuda':
        torch.cuda.empty_cache()


def monitor_experiment(experiment,n_jobs=1, device='cuda'):
    '''
    Training with monitoring SNR
    '''
    print('Training & mornitoring {:s} experiment'.format(experiment))
    configs = getattr(experiments, experiment)()
    if n_jobs>1:
        processes_num = n_jobs
        with mp.Pool(processes=processes_num) as pool:
            mid_results = [
                pool.apply_async(monitor_process_one_config, args=(config,device)) for config in configs]
            [p.get() for p in mid_results]
    else:
        for config in configs:
            monitor_process_one_config(config,device)



def get_test_time_point(max_time_point, inval_type='log'):
    L = []
    if inval_type == 'log':
        st = 0
        i = 0
        while(True):
            L.append(np.arange(st, st + 10 * 2 ** i, 2 ** i))
            st += 10 * 2 ** i
            i += 1
            if st>max_time_point:
                break
        L = np.concatenate(L, 0)
        L=L[L<max_time_point]
    elif inval_type == 'lin':
        L = np.arange(max_time_point)
    neg_times= -L[::-1] - 1
    pos_times = L
    return neg_times, pos_times
    # array([...,
    # -311287, -294903, -278519, -262135, -245751, -229367, -212983,...
    # -3,      -2,      -1,]       [0,       1,       2,       3,
    # 229366,  245750,  262134,  278518,  294902,  311286,
    # ...])

def train(config, device='cuda'):
    path=settings.MODELPATH / config['save_path']
    os.makedirs(path, exist_ok=True)
    filename = path / 'simulation.bk'
    if os.path.exists(filename):
        return

    utils.save_config(config, config['save_path'])
    pprint.pprint(config)
    print('Simulating ',filename)
    burnin_num = 196608 if 'burnin_num' not in config else config['burnin_num']
    sample_num = 2000 if 'sample_num' not in config else config['sample_num']
    fillin_num = 196608 if 'fillin_num' not in config else config['fillin_num']
    pattern_type = config['pattern_type']
    dim_num = config['dim_num']
    # set_seed(config['seed'])
    np.random.seed(config['seed'])
    if 'sparse_coding' in config and config['sparse_coding']:
        sparse_coding = config['sparse_coding']
        coding_f = 1/config['inv_coding_f']
    else:
        sparse_coding = False
        coding_f = 0.5
    if 'face_fillin_pattern' in config:
        face_fillin_pattern = config['face_fillin_pattern']
    else:
        face_fillin_pattern = False
    traintest_features, test_type = load_traintest_patterns(sample_num, dim_num, pattern_type=pattern_type, sparse_coding=sparse_coding, coding_f=coding_f)
    # [3, sample_num, feature_num]
    burnin_features = load_aux_patterns(burnin_num, dim_num, aug_pattern_type=pattern_type, sparse_coding=sparse_coding, coding_f=coding_f)
    if face_fillin_pattern:
        print('face_fillin_pattern',face_fillin_pattern)
        fillin_features = load_aux_patterns(fillin_num, dim_num, aug_pattern_type=pattern_type,
                                            sparse_coding=sparse_coding, coding_f=coding_f, face_fillin_pattern=face_fillin_pattern, real_face_start_from=sample_num)
    else:
        fillin_features = load_aux_patterns(fillin_num, dim_num, aug_pattern_type=pattern_type, sparse_coding=sparse_coding, coding_f=coding_f)
    memorynet = Memorynet(device=device, **config)
    print('Memorynet created')
    neg_times, pos_times = get_test_time_point(fillin_num, 'log')
    pt = 2  # better use protocol 2 in order to compute the task performance; or use 3

    save_weight_num = 0
    start = time.time()
    num_per_block = burnin_num
    cur_num = 0
    while cur_num<burnin_num:
        next_num=min(cur_num+num_per_block, burnin_num)
        print(f'Training burnin samples from {cur_num} to {next_num}')
        all_weight = memorynet.train_all_neurons(
            burnin_features[cur_num:next_num], save_weight=save_weight_num, burnin_stage=True)
        # memorynet.show_neurons_coef_distribution()
        cur_num = next_num

    print(f'Training burnin done. second passed', time.time() - start)
    r_signal_famil, io_signal_famil, times_famil, \
    r_signal_novel, io_signal_novel, times_novel = memorynet.build_dataset_protocol_during_training(
        traintest_features, fillin_features, neg_times, pos_times, protocol=pt)
    print(f'Training sample+fillin done. second passed', time.time() - start)

    roblib.dump({'r_signal_famil': r_signal_famil,
                 'io_signal_famil': io_signal_famil,
                 'times_famil': times_famil,
                 'r_signal_novel': r_signal_novel,
                 'io_signal_novel': io_signal_novel,
                 'times_novel': times_novel,
                 'test_type': test_type,
                 }, filename)

    print('second passed', time.time() - start)


def monitor(config, device='cuda'):
    '''
    Monitor SNR to re-present patterns
    '''
    path=settings.MODELPATH / config['save_path']
    os.makedirs(path, exist_ok=True)
    filename = path / 'monitor_simulation.bk'
    if os.path.exists(filename):
        return

    utils.save_config(config, config['save_path'])
    pprint.pprint(config)
    print('Simulating & monitoring',filename)
    burnin_num = 8000 if 'burnin_num' not in config else config['burnin_num']
    sample_num = 500 if 'sample_num' not in config else config['sample_num']
    pattern_type = config['pattern_type']
    dim_num = config['dim_num']
    np.random.seed(config['seed'])
    if 'sparse_coding' in config and config['sparse_coding']:
        sparse_coding = config['sparse_coding']
        coding_f = 1/config['inv_coding_f']
    else:
        sparse_coding = False
        coding_f = 0.5
    if 'monitor_threshold' in config:
        monitored_signal_thre = config['monitor_threshold']
    else:
        monitored_signal_thre = 0.1
    traintest_features, test_type = load_traintest_patterns(sample_num, dim_num, pattern_type=pattern_type, sparse_coding=sparse_coding, coding_f=coding_f)
    # [3, sample_num, feature_num]
    burnin_features = load_aux_patterns(burnin_num, dim_num, aug_pattern_type=pattern_type, sparse_coding=sparse_coding, coding_f=coding_f)

    memorynet = Memorynet(device=device, **config)
    print('Memorynet created')
    save_weight_num = 0
    start = time.time()
    num_per_block = burnin_num
    cur_num = 0
    while cur_num<burnin_num:
        next_num=min(cur_num+num_per_block, burnin_num)
        print(f'Training burnin samples from {cur_num} to {next_num}')
        all_weight = memorynet.train_all_neurons(
            burnin_features[cur_num:next_num], save_weight=save_weight_num, burnin_stage=True)
        # memorynet.show_neurons_coef_distribution()
        cur_num = next_num

    print(f'Training burnin done. second passed', time.time() - start)
    r_signal_all = []
    io_signal_all = []
    present_time_all = []
    monitor_sample_num = sample_num # 100
    print('monitored samples:')
    for idx in range(monitor_sample_num):
        print(idx,end=',')
        r_signal, io_signal, present_time = memorynet.monitor_snr_during_training(
            traintest_features[0, idx:idx+1],
            monitored_signal_thre=monitored_signal_thre, config=config)
        r_signal_all.append(r_signal)
        io_signal_all.append(io_signal)
        present_time_all.append(present_time)
    print(f'\nTraining sample+fillin done. second passed', time.time() - start)

    d = {'r_signal_famil': r_signal_all,
                 'io_signal_famil': io_signal_all,
                 'present_time': present_time_all,
                 'test_type': test_type,
                 }
    roblib.dump(d, filename)

    print('second passed', time.time() - start)
