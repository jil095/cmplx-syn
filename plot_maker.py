"""
General plot functions for making figures
"""
import numpy as np
import roblib
import settings
import matplotlib.pyplot as plt
import os
import matplotlib
cmap = matplotlib.cm.get_cmap('plasma')
from pathlib import Path
import pandas as pd
from sklearn import linear_model
from scipy.ndimage import gaussian_filter
from training import get_test_time_point
show_posneg_time = False # for debugging

def set_mpl():
    import matplotlib as mpl

    mpl.rcParams['font.size'] = 6
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'arial'
    mpl.rcParams['savefig.dpi'] = 480


def plot_start(square=True,figsize=None,ticks_pos=True):
    '''
    unified plot params
    '''
    set_mpl()
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    elif square:
        fig = plt.figure(figsize=(1.5, 1.5))
    else:
        fig = plt.figure(figsize=(1.5, 0.8))
    ax = fig.add_axes((0.1,0.1,0.8,0.8))
    if ticks_pos:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    return fig,ax

def metric_plot_params(metric_type):
    plot = {}
    plot['metric_save_name'] = metric_type
    plot['smooth'] = lambda x: x
    plot['xticks'] = [10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6,]
    if metric_type in ['ioSNR', 'rSNR', 'ioSignal', 'ioNoise']:
        plot['plot'] = plt.loglog
        plot['xlabel'] = 'Memory age'
        plot['ylabel'] = metric_type
        plot['xlim'] = [1,2*10**6]
        plot['ylim'] = [0.05, 10**3]
        plot['yticks'] = [0.1, 1, 10, 100,10**3]
        if show_posneg_time:
            plot['plot'] = plt.plot
            plot['xlabel'] = 'Memory age'
            plot['ylabel'] = metric_type
            plot['xlim'] = [-10**3,10**3]
            plot['ylim'] = [0.0, 2.3]
            plot['yticks'] = [0.1, 1]#, 10]#, 100]
    elif metric_type in ['ioSNR_seedwise','rSNR_seedwise']:
        def plot_seed(x, y, label,color):
            dt = np.array(y)
            y = dt.mean(0)
            std = dt.std(0)
            plt.loglog(x, y, label=label,color=color)
            plt.fill_between(x, np.maximum(y-std,10**-10), y + std, facecolor=color, alpha=0.3,edgecolor=None)
        plot['plot'] = plot_seed
        plot['xlabel'] = 'Memory age'
        plot['ylabel'] = metric_type[:-9]
        plot['xlim'] = [1,2*10**6]
        plot['ylim'] = [0.05, 10**3]
        plot['yticks'] = [0.1, 1, 10, 100,10**3]
    elif metric_type in ['TPR']:
        plot['plot'] = plt.semilogx
        plot['xlabel'] = 'Memory age'
        plot['ylabel'] = 'True positive rate'
        plot['xlim'] = [1,2*10**6]
        plot['ylim'] = [1-0.8413, 1.02]
        plot['yticks'] = [0.2, 0.4, 0.6, 0.8, 1]
    elif metric_type in ['FC', 'FD', 'AUC']:
        plot['plot'] = plt.semilogx
        plot['xlabel'] = 'Memory age'
        plot['ylabel'] = metric_type + ' performance'
        plot['xlim'] = [1,2*10**6]
        plot['ylim'] = [0.5, 1.02]
        plot['yticks'] = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
        plot['smooth'] = lambda x: gaussian_filter(x, 1, mode='nearest')
    elif metric_type in ['FC_seedwise','FD_seedwise']:
        def plot_seed(x, y, label,color):
            dt = np.array(y)
            y = dt.mean(0)
            std = dt.std(0)
            plt.semilogx(x, y, label=label,color=color)
            plt.fill_between(x, np.maximum(y-std,10**-10), y + std,  facecolor=color, alpha=0.3,edgecolor=None)
        plot['plot'] = plot_seed
        plot['xlabel'] = 'Memory age'
        plot['ylabel'] = metric_type[:-9] + ' performance'
        plot['xlim'] = [1,2*10**6]
        plot['ylim'] = [0.5, 1.02]
        plot['yticks'] = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    elif metric_type in ['FD_TPR', 'FD_TNR']:
        plot['plot'] = plt.semilogx
        plot['xlabel'] = 'Memory age'
        plot['ylabel'] = {'FD_TPR': 'FD true positive rate', 'FD_TNR': 'FD true negative rate'}[metric_type]
        plot['xlim'] = [1,2*10**6]
        plot['ylim'] = [0.5, 1.02]
        plot['yticks'] = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
        plot['smooth'] = lambda x: gaussian_filter(x, 1, mode='nearest')
    elif metric_type in ['SNRratio']:
        plot['plot'] = plt.loglog
        plot['xlabel'] = 'Memory age'
        plot['ylabel'] = 'rSNR/ioSNR'
        plot['xlim'] = [1,2*10**6]
        plot['ylim'] = [1, 1000]
        plot['yticks'] = [0.1, 1, 10, 100]

    return plot


def pattern_test_type_plot_params(pattern_type, test_type):
    plot = {}
    if pattern_type == 'face':
        if test_type == 'same':
            plot['title'] = 'Same pose'
        elif test_type == 'noisy':
            plot['title'] = 'Different pose'
    elif pattern_type == 'rand':
        if test_type == 'same':
            plot['title'] = 'Random pattern'
    plot['pattern_save_name'] = pattern_type + '-' + test_type
    return plot


def tstar_vs_var_all():
    ...


def get_reg_coef(x,y,get_reg=False):
    regr = linear_model.LinearRegression()
    x = np.array(x)
    regr.fit(x.reshape(-1, 1),y)
    coef=regr.coef_[0]
    if get_reg:
        return coef,lambda x: regr.predict(np.reshape(x,(-1, 1))),regr.score(x.reshape(-1, 1),y)
    return coef


def plot_module(save_folder_name, compared_models, comparison_plot_params, save=True):
    plt.ion()
    all_model_stats = pd.DataFrame()
    for model_path, model_id, model_color in compared_models:
        if not os.path.exists(settings.MODELPATH / model_path):
            continue
        model_stats = roblib.load(settings.MODELPATH / model_path)
        model_stats['id'] = model_id
        model_stats['color'] = model_color
        all_model_stats = all_model_stats.append(model_stats, ignore_index=True)
    if 'skip_default_metric' not in comparison_plot_params or not comparison_plot_params['skip_default_metric']:
        metric_types = list(set.intersection({'ioSNR', 'rSNR', 'FC', 'TPR', 'FD', 'AUC', 'FD_TPR', 'FD_TNR'},
                                         set(pd.unique(all_model_stats['metric_type']))))
    else:
        metric_types = []
    pattern_types = ['rand', 'face']
    if show_posneg_time:
        metric_types = ['ioSNR', 'rSNR', 'ioSignal', 'ioNoise']
    if 'additional' in comparison_plot_params:
        metric_types += comparison_plot_params['additional']
    if save:
        save_path = settings.FIGPATH / save_folder_name
        os.makedirs(save_path, exist_ok=True)

    def summary_curve_params(pattern_type, test_type):
        if pattern_type == 'face' and test_type == 'same':
            tstar_curve_params = (1, 'SP', '--', 'C2', 'o', 5)
        elif pattern_type == 'face' and test_type == 'noisy':
            tstar_curve_params = (2, 'DP', '--', 'C1', 'v', 5)
        else:  # pattern_type == 'rand' and test_type == 'same':
            tstar_curve_params = (0, 'RD', '--', 'C0', 'x', 6)
        return tstar_curve_params


    for metric_type in metric_types:
        models_same_m = all_model_stats[all_model_stats['metric_type']==metric_type]
        if 'no_curve_plot' not in comparison_plot_params or not comparison_plot_params['no_curve_plot']:
            for pattern_type in pattern_types:
                models_same_mp = models_same_m[models_same_m['pattern_type']==pattern_type]
                if len(models_same_mp) == 0:
                    continue
                for test_type in pd.unique(models_same_mp['test_type']):
                    models_same_mpt = models_same_mp[models_same_mp['test_type']==test_type]
                    if len(models_same_mpt) == 0: continue
                    print('Making plots for ', metric_type, pattern_type, test_type)
                    fig, ax = plot_start(square=True)
                    plot = {**comparison_plot_params,
                            **metric_plot_params(metric_type),
                            **pattern_test_type_plot_params(pattern_type, test_type)}
                    assert len(pd.unique(models_same_mpt['perf_thre']))==1
                    for idx, row in models_same_mpt.iterrows():
                        perf = row['perf']
                        perf = plot['smooth'](perf)
                        perf = np.maximum(perf, 10**-10)
                        x_time = row['time']
                        if len(x_time)<len(perf):
                            x_time = get_test_time_point(2*10**6, 'log')[1]
                        plot['plot'](x_time+1, perf, label=row['id'],color=cmap(row['color']))
                    plt.xlabel(plot['xlabel'])
                    plt.ylabel(plot['ylabel'])
                    plt.xlim(plot['xlim'])
                    plt.ylim(plot['ylim'])
                    plt.yticks(plot['yticks'])
                    plt.xticks(plot['xticks'])
                    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1, 0.1), numticks=1+len(plot['xticks']))
                    ax.xaxis.set_minor_locator(locmin)
                    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                    ax.tick_params(which='major', width=0.75)
                    ax.tick_params(which='minor', width=0.25)
                    plt.title(plot['title'])
                    plt.plot(plot['xlim'], [row['perf_thre'], row['perf_thre']], '--', color='gray')
                    if 'has_legend' not in plot or plot['has_legend']:
                        leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=False, ncol=1)
                        leg.set_title(plot['legendtitle'])
                    if save:
                        filename = save_path / f"{plot['metric_save_name']}_{plot['pattern_save_name']}.pdf"
                        plt.show()
                        plt.savefig(filename, bbox_inches="tight")
                        print('Saved ', filename)
                    plt.close()

        if 'tstar_type' in comparison_plot_params:
            if np.isnan(models_same_m.iloc[0]['tstar']):
                continue
            fig, ax = plot_start(square=True)
            tstar_type = comparison_plot_params['tstar_type']
            for pattern_type in pattern_types:
                models_same_mp = models_same_m[models_same_m['pattern_type']==pattern_type]
                for test_type in pd.unique(models_same_mp['test_type']):
                    models_same_mpt = models_same_mp[models_same_mp['test_type']==test_type]
                    idx, label, linestyle, color, marker, msize = summary_curve_params(pattern_type, test_type)
                    var = models_same_mpt['id'].to_numpy()
                    t_star_list = models_same_mpt['tstar'].to_numpy()

                    if tstar_type == 'log':
                        locs = np.logical_not(np.isnan(t_star_list))
                        var = var[locs]
                        t_star_list = t_star_list[locs]

                        reg_indices = np.ones(var.shape).astype(np.bool)
                        if 'no_regr_models' in comparison_plot_params and \
                            (pattern_type, test_type) in comparison_plot_params['no_regr_models']:
                            no_regr_models = comparison_plot_params['no_regr_models'][(pattern_type, test_type)]
                            for model_var in no_regr_models:
                                reg_indices[var == model_var] = False
                        coef, regr, score = get_reg_coef(np.log2(var)[reg_indices], (np.log2(t_star_list) + np.log2(np.log2(var)))[reg_indices], get_reg=True)
                        y_pred = 2 ** (regr(np.log2(var)) - np.log2(np.log2(var)))
                        plt.loglog(var, t_star_list, label='%s: %.2f' % (label, coef), alpha=0.7, linestyle='None',
                                   color=color, marker=marker, markersize=msize)
                        plt.loglog(var, y_pred, alpha=1, linestyle=linestyle, color=color, marker=None)
                        plt.xlim([10, 3000])
                        plt.xticks([10, 100, 1000])
                        plt.xlabel('Number of neurons')
                        plt.ylim([1, 10 ** 6])
                        plt.yticks([1, 10 ** 2, 10 ** 4, 10 ** 6])
                        plt.ylabel(metric_type+r' $t^*$')
                        if 'has_legend' not in plot or plot['has_legend']:
                            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
                    elif tstar_type == 'linear':
                        locs = np.logical_not(np.isnan(t_star_list))
                        var = var[locs]
                        t_star_list = t_star_list[locs]

                        reg_indices = np.ones(var.shape).astype(np.bool)
                        if 'no_regr_models' in comparison_plot_params and \
                            (pattern_type, test_type) in comparison_plot_params['no_regr_models']:
                            no_regr_models = comparison_plot_params['no_regr_models'][(pattern_type, test_type)]
                            for model_var in no_regr_models:
                                reg_indices[var == model_var] = False
                        coef, regr, score = get_reg_coef(var[reg_indices],
                                                         np.log2(t_star_list)[reg_indices],
                                                         get_reg=True)
                        y_pred = 2 ** (regr(var))
                        plt.semilogy(var, t_star_list, label='%s: %.2f' % (label, coef), alpha=0.7, linestyle='None',
                                   color=color, marker=marker, markersize=msize)
                        plt.semilogy(var, y_pred, alpha=1, linestyle=linestyle, color=color, marker=None)
                        plt.xlim([0, 9])
                        plt.xticks([2,4,6,8])
                        plt.xlabel('Synaptic complexity')
                        plt.ylim([1, 10 ** 6])
                        plt.yticks([1, 10 ** 2, 10 ** 4, 10 ** 6])
                        plt.ylabel(metric_type+r' $t^*$')
                        if 'has_legend' not in plot or plot['has_legend']:
                            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
                    elif tstar_type == 'bar':
                        cmp_num=len(var)
                        coef=t_star_list[-1]/t_star_list[0]
                        print('best/worse ratio:',coef)
                        dimw=0.75/3
                        plt.bar(np.arange(cmp_num) + (idx-1) * dimw, t_star_list, dimw, bottom=1,label=label)
                        plt.xticks(range(cmp_num),var,rotation=90)
                        ax.set_yscale('log')
                        plt.xlim([-1, cmp_num])
                        plt.ylim([1, 10 ** 6])
                        plt.yticks([1, 10 ** 2, 10 ** 4, 10 ** 6])
                        plt.ylabel(metric_type+r' $t^*$')
                        if 'has_legend' not in plot or plot['has_legend']:
                            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
                    else:
                        raise NotImplementedError
            if save:
                filename = save_path / f"{metric_type}_tstar_{tstar_type}.pdf"
                plt.savefig(filename, bbox_inches="tight")
                print('Saved ', filename)
            plt.show()
            plt.close()

        if 'init_type' in comparison_plot_params:
            fig, ax = plot_start(square=True)
            init_type = comparison_plot_params['init_type']
            for pattern_type in pattern_types:
                models_same_mp = models_same_m[models_same_m['pattern_type']==pattern_type]
                for test_type in pd.unique(models_same_mp['test_type']):
                    models_same_mpt = models_same_mp[models_same_mp['test_type']==test_type]
                    idx, label, linestyle, color, marker, msize = summary_curve_params(pattern_type, test_type)
                    var = models_same_mpt['id'].to_numpy()
                    initperf_list = models_same_mpt['perf'].map(lambda x: x[0]).to_numpy()

                    if init_type == 'log':
                        locs = np.logical_not(np.isnan(initperf_list))
                        var = var[locs]
                        initperf_list = initperf_list[locs]
                        plt.loglog(var, initperf_list, label=label, alpha=0.7, linestyle='None',
                                   color=color, marker=marker, markersize=msize)
                        plt.xlim([10, 2200])
                        plt.xticks([10, 100, 1000])
                        plt.xlabel('Number of neurons')
                        plt.ylabel(metric_type+r' $initial perf$')
                        if 'has_legend' not in plot or plot['has_legend']:
                            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
                    elif init_type == 'linear':
                        ...
                    elif init_type == 'bar':
                        ...
                    else:
                        raise NotImplementedError
            if save:
                filename = save_path / f"{metric_type}_initperf_{init_type}.pdf"
                plt.savefig(filename, bbox_inches="tight")
                print('Saved ', filename)
            plt.show()
            plt.close()
    plt.ioff()