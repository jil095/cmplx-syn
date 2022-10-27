"""
General functions for analyzing the memory neural system
"""
import numpy as np
import roblib
import settings
from pathlib import Path
import pandas as pd
from utils import load_config
from sklearn.metrics import roc_auc_score
from training import get_test_time_point
import os

def get_intersect(x, y, thre):
    if (y >= thre).sum()<=1:
        return np.nan, np.nan
    below = np.where(y < thre)[0]
    if len(below) == 0:
        below = above = np.where(y >= thre)[0][-1]
    else:
        below = below[0]
        above = below - 1
    return (x[below] + x[above]) / 2, (y[below] + y[above]) / 2


def get_tstar(x, y, thre):
    t_star, _ = get_intersect(np.log10(x + 1), y, thre)
    t_star = 10 ** t_star - 1
    return t_star


def accuracy_at_decision_point(famil_patterns, novel_patterns, thre):
    FD_TPR = (famil_patterns > thre).mean(-1)
    FD_TNR = (novel_patterns <= thre).mean(-1)
    acc = (FD_TPR + FD_TNR) / 2
    return acc, FD_TPR, FD_TNR


def get_linear_interpolation_mean(acc_curve, t_axis):
    assert acc_curve.shape[0] == t_axis.shape[0]
    expand_acc = []
    if len(t_axis) == 1:
        return acc_curve.mean()
    for i in range(len(t_axis) - 1):
        expand_acc.append(np.linspace(acc_curve[i], acc_curve[i + 1], t_axis[i + 1] - t_axis[i],
                                      endpoint=False))
    expand_acc = np.concatenate(expand_acc, 0)
    return expand_acc.mean()


def optimize_decision_point(famil_patterns, novel_patterns, t_axis):
    # time steps, pattern num
    thre_list = np.arange(0, 1, 0.005)
    perf = []
    for thre in thre_list:
        acc_curve = accuracy_at_decision_point(famil_patterns, novel_patterns, thre)[0]
        test_acc = get_linear_interpolation_mean(acc_curve, t_axis)
        perf.append(test_acc)
    return thre_list[np.argmax(perf)]


def mix_novel_patterns(novel_patterns):
    novel_patterns = np.random.permutation(novel_patterns.T).T
    return novel_patterns

def expand_in_time(x, T=None):
    assert len(x.shape)==1
    T = len(get_test_time_point(2*10**6, 'log')[1])
    if len(x)<T:
        print('Expanding ', len(x),end=',')
        rep_time = T//len(x)+1
        new_x = [x]
        for idx in range(1, rep_time):
            new_x.append(np.random.permutation(x))
        x = np.concatenate(new_x, 0)[:T]
        print(len(x))
    return x

def compute_perf(model_path_agg, model_paths_same_seed, verbose=True, force_params=None):
    # shape: [probe_time, test_num, ref_sample]
    r_signal_famil_allseeds = []
    io_signal_famil_allseeds = []
    r_signal_novel_allseeds = []
    io_signal_novel_allseeds = []
    for model_path in model_paths_same_seed:
        if os.path.exists(model_path / 'simulation.bk'):
            analytics = roblib.load(model_path / 'simulation.bk')
        else:
            return
        r_signal_famil_allseeds.append(analytics['r_signal_famil'])
        io_signal_famil_allseeds.append(analytics['io_signal_famil'])
        if analytics['r_signal_novel'] is not None:
            r_signal_novel_allseeds.append(analytics['r_signal_novel'])
        if analytics['io_signal_novel'] is not None:
            io_signal_novel_allseeds.append(analytics['io_signal_novel'])
    seed_num = len(r_signal_famil_allseeds)
    r_signal_famil = np.concatenate(r_signal_famil_allseeds, axis=-1)
    io_signal_famil = np.concatenate(io_signal_famil_allseeds, axis=-1)
    r_signal_novel = np.concatenate(r_signal_novel_allseeds, axis=-1)
    io_signal_novel = np.concatenate(io_signal_novel_allseeds, axis=-1)
    config = load_config(model_path)
    model_stats = pd.DataFrame(columns=['model_name', 'is_pos_time', 'pattern_type', 'test_type',
                                        'metric_type', 'perf_thre', 'tstar', 'perf', 'time'])
    SNR_thre = 0.5
    FC_thre = 0.6
    FD_thre = 0.6
    AUC_thre = 0.6
    TNR_thre = TPR_thre = 0.8413 # 0.8413 for 1 sigma 0.9772 for 2 sigma
    if force_params is not None:
        if 'SNR_thre' in force_params:
            SNR_thre = force_params['SNR_thre']
        if 'FC_thre' in force_params:
            FC_thre = force_params['FC_thre']
        if 'AUC_thre' in force_params:
            AUC_thre = force_params['AUC_thre']
        if 'FD_thre' in force_params:
            FD_thre = force_params['FD_thre']
        if 'TNR_thre' in force_params:
            TNR_thre = force_params['TNR_thre']
        if 'TPR_thre' in force_params:
            TPR_thre = force_params['TPR_thre']

    show_posneg_time = False
    t_axis = analytics['times_famil']
    mixed_signal_novel = mix_novel_patterns(r_signal_famil[:, -1]) # [T, N]->[T, perm(N)]
    for ttype_idx, ttype in enumerate(analytics['test_type'][:-1]): #  ignore the last 'new' type
        def wrap_snr(signal):
            s = signal.mean(-1)
            n = signal.std(-1)
            snr = s / n
            return s, n, snr
        ioSignal, ioNoise, ioSNR = wrap_snr(io_signal_famil[:, ttype_idx])
        rSignal, rNoise, rSNR = wrap_snr(r_signal_famil[:, ttype_idx])


        SNRratio = rSNR / ioSNR
        SNRratio[ioSNR<=1] = 1
        FC = (r_signal_famil[:, ttype_idx] > mixed_signal_novel).mean(-1)+ \
             (r_signal_famil[:, ttype_idx] == mixed_signal_novel).mean(-1)/2
        TNR_pattern_thre = np.quantile(mixed_signal_novel, TNR_thre)
        TPR = np.mean(r_signal_famil[:, ttype_idx] > TNR_pattern_thre, -1)
        probe_time_num, test_num, ref_sample_num = r_signal_famil.shape
        AUC = np.zeros(probe_time_num)
        for t_idx in range(probe_time_num):
            y_true = np.array([1] * ref_sample_num + [0] * ref_sample_num)
            y_score = np.concatenate([r_signal_famil[t_idx, ttype_idx], mixed_signal_novel[t_idx]], 0)
            AUC[t_idx] = roc_auc_score(y_true, y_score)

        ioSNR_tstar = get_tstar(t_axis, ioSNR, SNR_thre)
        rSNR_tstar = get_tstar(t_axis, rSNR, SNR_thre)
        FC_tstar = get_tstar(t_axis, FC, FC_thre)
        TPR_tstar = get_tstar(t_axis, TPR, TPR_thre)
        AUC_tstar = get_tstar(t_axis, AUC, AUC_thre)

        FD_T_thre = t_axis<=ioSNR_tstar
        if np.sum(FD_T_thre) >0:
            FD_decision_point = optimize_decision_point(
                r_signal_famil[FD_T_thre, ttype_idx], mixed_signal_novel[FD_T_thre], t_axis[FD_T_thre])
            FD, FD_TPR, FD_TNR = accuracy_at_decision_point(r_signal_famil[:, ttype_idx], mixed_signal_novel, FD_decision_point)
            FD_tstar = get_tstar(t_axis, FD, FD_thre)
            FD_TPR_tstar =  get_tstar(t_axis, FD_TPR, FD_thre)
        else:
            FD = np.nan * FC
            FD_TPR = FD_TNR = FD
            FD_tstar = np.nan
            FD_TPR_tstar = np.nan

        ioSNR_seedwise = []
        rSNR_seedwise = []
        FC_seedwise = []
        FD_seedwise = []
        for i in range(seed_num):
            equal_sample_size = 200
            ios = io_signal_famil_allseeds[i][...,:equal_sample_size]
            rs = r_signal_famil_allseeds[i][...,:equal_sample_size]
            ioSNR_seedwise.append(wrap_snr(ios[:, ttype_idx])[-1])
            rSNR_seedwise.append(wrap_snr(rs[:, ttype_idx])[-1])
            mixed_signal_novel_seed = mix_novel_patterns(rs[:, -1])  # [T, N]->[T, perm(N)]
            FCs = (rs[:, ttype_idx] > mixed_signal_novel_seed).mean(-1) + \
                 (rs[:, ttype_idx] == mixed_signal_novel_seed).mean(-1) / 2
            if np.sum(FD_T_thre) >0:
                FD_decision_point = optimize_decision_point(
                    rs[FD_T_thre, ttype_idx], mixed_signal_novel_seed[FD_T_thre], t_axis[FD_T_thre])
                FDs, _, _ = accuracy_at_decision_point(rs[:, ttype_idx], mixed_signal_novel_seed,
                                                            FD_decision_point)
            else:
                FDs = np.nan
            FC_seedwise.append(FCs)
            FD_seedwise.append(FDs)


        if show_posneg_time: # for debugging
            ioSignal, ioNoise, ioSNR = wrap_snr(np.concatenate([io_signal_novel,io_signal_famil],0)[:, ttype_idx])
            rSignal, rNoise, rSNR = wrap_snr(np.concatenate([r_signal_novel,r_signal_famil],0)[:, ttype_idx])
            t_axis = np.concatenate([analytics['times_novel'], analytics['times_famil']],0)
            ioSNR_tstar = rSNR_tstar = FC_tstar = TPR_tstar = FD_tstar = AUC_tstar = 0

        model_stat = pd.DataFrame({
            'metric_type': pd.Series(['ioSignal', 'ioNoise', 'ioSNR', 'rSignal', 'rNoise', 'rSNR', 'FC', 'TPR', 'FD', 'SNRratio','AUC', 'FD_TPR', 'FD_TNR', 'ioSNR_seedwise','rSNR_seedwise','FC_seedwise','FD_seedwise',]),
            'perf': pd.Series([ioSignal, ioNoise, ioSNR, rSignal, rNoise, rSNR, FC, TPR, FD, SNRratio, AUC, FD_TPR, expand_in_time(FD_TNR), ioSNR_seedwise,rSNR_seedwise, FC_seedwise,FD_seedwise]),
            'perf_thre': pd.Series([np.nan, np.nan, SNR_thre, np.nan, np.nan, SNR_thre, FC_thre, TPR_thre, FD_thre, np.nan, AUC_thre, FD_thre, np.nan, SNR_thre, SNR_thre,FC_thre,FD_thre]),
            'tstar': pd.Series([np.nan, np.nan, ioSNR_tstar, np.nan, np.nan, rSNR_tstar, FC_tstar, TPR_tstar, FD_tstar, np.nan, AUC_tstar, FD_TPR_tstar, np.nan, np.nan,np.nan,np.nan,np.nan]),
        })
        model_stat['model_name'] = str(model_path_agg)
        model_stat['is_pos_time'] = True
        model_stat['pattern_type'] = config['pattern_type']
        model_stat['test_type'] = ttype
        model_stat['time'] = pd.Series([t_axis]*len(model_stat))

        model_stats = model_stats.append(model_stat, ignore_index=True)
    if verbose:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        print(model_stats)
        print('Saving at ', model_path_agg)
    roblib.dump(model_stats, model_path_agg)