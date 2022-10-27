import numpy as np
import torch
import torch.nn as nn
import random
import scipy
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from datasets import load_aux_patterns


def set_seed(manualSeed):
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_phi_coef(x, y):
    n = x.shape[0]
    xis1 = x==1
    yis1 = y==1
    n11 = torch.sum(torch.logical_and(xis1, yis1))
    n1_ = torch.sum(xis1)
    n_1 = torch.sum(yis1)
    denominator = n1_*n_1*(n-n1_)*(n-n_1)
    if denominator>0:
        phi = (n*n11-n1_*n_1)/torch.sqrt(denominator.to(torch.float64))
    else:
        phi = 0
    return phi


class BCSLayer(nn.Module):
    def __init__(self, beaker_num=1, level_num=32, beakers=None, feature_num=63, neuron_num=64,
                 n=2, alpha=0.25, leaky_coef=1,
                 isdiscrete=True, prob_encoding=1, sparse_coding=False, inv_coding_f=2, isbounded=True, pre_feature_ratio=1.0, **kw):
        super().__init__()
        assert beakers is None
        beakers = [level_num] * beaker_num
        # otherwise should use minimum rather than torch.clamp in truncate
        self.beaker_num = beaker_num
        self.beakers = beakers
        self.feature_num = feature_num
        self.neuron_num = neuron_num
        self.n = n
        self.alpha = alpha
        self.leaky_coef = leaky_coef
        self.isdiscrete = isdiscrete
        self.prob_encoding = prob_encoding
        self.burnin_mode = True
        self.sparse_coding = sparse_coding
        self.isbounded = isbounded
        self.pre_feature_ratio = pre_feature_ratio
        allowed_feature_num = int(np.round(neuron_num * self.pre_feature_ratio)) - 1 # -1 due to intercept
        feature_mask = torch.zeros([feature_num, neuron_num])
        if self.pre_feature_ratio < 1:
            for n_idx in range(neuron_num):
                idx = np.random.choice(range(feature_num), size=allowed_feature_num, replace=False)
                feature_mask[idx, n_idx] = 1
        else:
            feature_mask = torch.ones([feature_num, neuron_num])
        if sparse_coding:
            self.coding_f = 1/inv_coding_f
        else:
            self.coding_f = 0.5
        if self.isdiscrete:
            offset = torch.tensor([(level_num % 2 == 0) * 0.5 for level_num in self.beakers])
            offset = offset.reshape([-1, 1, 1])
            self.register_buffer('offset', offset)
        transition_mat = torch.zeros([self.beaker_num, self.beaker_num])
        upper_bound = torch.zeros([self.beaker_num, 1, 1])
        lower_bound = torch.zeros([self.beaker_num, 1, 1])
        coef_ = torch.zeros([self.beaker_num, self.feature_num, self.neuron_num])
        intercept_ = torch.zeros([self.beaker_num, self.neuron_num])

        self.register_buffer('transition_mat', transition_mat)
        self.register_buffer('upper_bound', upper_bound)
        self.register_buffer('lower_bound', lower_bound)
        self.register_buffer('coef_', coef_)
        self.register_buffer('intercept_', intercept_)
        self.register_buffer('feature_mask', feature_mask)

        for beaker_idx in range(self.beaker_num):
            i = beaker_idx + 1
            self.upper_bound[beaker_idx, 0, 0] = (self.beakers[beaker_idx] - self.isdiscrete) / 2
            self.lower_bound[beaker_idx, 0, 0] = -self.upper_bound[beaker_idx, 0, 0]
            self.transition_mat[beaker_idx, beaker_idx] = 1
            to_next = self.n ** (-2 * i + 1) * alpha
            to_prev = self.n ** (-2 * i + 2) * alpha

            if i > 1:  # has previous beaker
                self.transition_mat[beaker_idx, beaker_idx - 1] = to_prev
                self.transition_mat[beaker_idx, beaker_idx] -= to_prev
            if i < self.beaker_num:  # has next beaker
                self.transition_mat[beaker_idx, beaker_idx + 1] = to_next
                self.transition_mat[beaker_idx, beaker_idx] -= to_next
            if i == self.beaker_num:  # leak to envir
                self.transition_mat[beaker_idx, beaker_idx] -= to_next * leaky_coef

    def set_burnin_mode(self):
        self.burnin_mode = True

    def set_eval_mode(self):
        self.burnin_mode = False

    def transition_dynamics(self, coef, intercept):
        new_intercept = torch.matmul(self.transition_mat, intercept)
        new_coef = torch.matmul(
            self.transition_mat,
            coef.reshape([self.beaker_num, self.feature_num * self.neuron_num])
        ).reshape([self.beaker_num, self.feature_num, self.neuron_num])
        return new_coef, new_intercept

    def truncate(self, coef, intercept):
        lb = self.lower_bound[0, 0, 0]
        ub = self.upper_bound[0, 0, 0]
        intercept = torch.clamp(intercept, lb, ub)
        coef = torch.clamp(coef, lb, ub)
        return coef, intercept

    def rollback_assist(self, temp_size, mat, mat_backup):
        step_back = torch.rand(size=(temp_size,), device=mat.device) >= self.prob_encoding
        mat.reshape([self.beaker_num, temp_size])[:, step_back] = mat_backup.reshape(
            [self.beaker_num, temp_size])[:, step_back]

    def prob_rollback(self, coef_backup, intercept_backup):
        self.rollback_assist(self.feature_num * self.neuron_num, self.coef_, coef_backup)
        self.rollback_assist(self.neuron_num, self.intercept_, intercept_backup)

    @staticmethod
    def discretize(x, offset):
        x_floor = torch.floor(x - offset) + offset
        x_ran = torch.rand(*x.shape, device=x.device)
        x_mask = (x_ran <= x - x_floor)
        x = x_mask + x_floor
        return x

    def fit(self, features, labels):
        # features: batch_size,feature_num,neuron_num
        # labels: batch_size,neuron_num
        assert features.shape[0] == 1 and labels.shape[0] == 1  # batch size=1
        device = self.coef_.device
        delta_coef, delta_intercept = self.local_learning_update(
            features.to(device), labels.to(device), sparse_coding=self.sparse_coding, coding_f=self.coding_f)
        delta_coef, delta_intercept = delta_coef[0], delta_intercept[0]
        self.update(delta_coef, delta_intercept)

    def update(self, delta_coef, delta_intercept):
        # delta_coef: feature_num, neuron_num
        # delta_intercept: neuron_num
        new_coef, new_intercept = self.transition_dynamics(self.coef_, self.intercept_)
        new_intercept[0] += delta_intercept
        new_coef[0, :] += delta_coef
        if self.isdiscrete:
            new_intercept = self.discretize(new_intercept, self.offset[:, 0, :])
            new_coef = self.discretize(new_coef, self.offset)
        if self.isbounded:
            new_coef, new_intercept = self.truncate(new_coef, new_intercept)

        coef_backup, intercept_backup = self.coef_, self.intercept_
        self.coef_, self.intercept_ = new_coef, new_intercept

        if self.prob_encoding < 1 and not self.burnin_mode:
            self.prob_rollback(coef_backup, intercept_backup)


    def forward_linearity(self, features):
        # feature: batch_size,dim_num,neuron_num
        # coef[0]: dim_num,neuron_num
        # intercept[0]: neuron_num
        assert len(features.shape)==3
        features = features - self.pre_pattern_mean
        return torch.sum(features * (self.feature_mask * self.coef_[0]), 1) + self.intercept_[0]
        # batch_size,neuron_num


    def forward_thresholding(self, memory_pre):
        if self.sparse_coding:
            thre = torch.quantile(memory_pre, 1-self.coding_f, axis=1,keepdim=True)
        else:
            thre = 0
        return 2 * (memory_pre >= thre) - 1
        # batch_size,neuron_num


    def forward(self, features):
        memory_pre = self.forward_linearity(features)
        memories = self.forward_thresholding(memory_pre)
        return memories

    def predict(self, features):
        return self.forward(features)

    def local_learning_update(self, features, labels, sparse_coding=False, coding_f=0.5):
        # features: batch_size,feature_num,neuron_num
        # labels: batch_size,neuron_num
        batch_size, feature_num, neuron_num = features.shape
        lbatch_size, lneuron_num = labels.shape
        assert neuron_num == lneuron_num and batch_size == lbatch_size

        if sparse_coding:
            assert torch.sum(features==-1)>0
            pre_pattern_mean = 2*coding_f-1
            post_pattern_mean = 2*coding_f-1
            std_delta_w = 1
            intercept_learning_rate = 0
            coef_learning_rate = 1/std_delta_w
            delta_weight_mean = 0
        else:
            pre_pattern_mean = 0
            post_pattern_mean = 0
            intercept_learning_rate = 1
            coef_learning_rate = 1
            delta_weight_mean = 0
        features = features - pre_pattern_mean
        labels = labels - post_pattern_mean
        delta_intercept = intercept_learning_rate * labels #
        delta_coef = coef_learning_rate * (self.feature_mask * features * labels[:, None, :] - delta_weight_mean)
        self.pre_pattern_mean = pre_pattern_mean
        self.post_pattern_mean = post_pattern_mean
        return delta_coef, delta_intercept # batch_size,feature_num,neuron_num; batch_size,neuron_num


class Circularize(object):
    def __init__(self, dim, device='cpu'):
        # x:      [0 1 2 3 4]
        # h:    [[0, 1, 2, 3, 4],
        #        [1, 2, 3, 4, 0],
        #        [2, 3, 4, 0, 1],
        #        [3, 4, 0, 1, 2],
        #        [4, 0, 1, 2, 3]]
        self.dim = dim
        h = np.arange(dim)
        h = scipy.linalg.hankel(h, np.concatenate([h[-1:], h[:-1]], 0))
        self.h = torch.from_numpy(h).long().to(device) # dim, dim

    def get_feature_label_ensemble_from_samples(self, x):
        assert len(x.shape) == 2
        sample_num, feature_num = x.shape
        assert self.dim == feature_num
        new_x = x[:, self.h] # sample_num, dim_num, dim_num
        labels = new_x[:,0] # sample_num, dim_num
        features = new_x[:,1:] #sample_num,dim_num-1,dim_num
        return features, labels


class Memorynet(object):
    def __init__(self, dim_num=100, device='cpu', **config):
        self.device=device
        if 'sparse_coding' in config and config['sparse_coding']:
            self.sparse_coding = config['sparse_coding']
            self.coding_f = 1/config['inv_coding_f']
        else:
            self.sparse_coding = False
            self.coding_f = 0.5
        if 'pre_feature_ratio' in config:
            self.pre_feature_ratio = config['pre_feature_ratio']
        else:
            self.pre_feature_ratio = 1
        self.neurons = BCSLayer(feature_num=dim_num - 1, neuron_num=dim_num, **config).to(device)
        self.circ = Circularize(dim_num, device)
        self.neuron_num = dim_num
        self.beaker_num = self.neurons.beaker_num

    def train_all_neurons(self, features, save_weight=0, burnin_stage=False):
        assert len(features.shape) == 2
        sample_num = features.shape[0]
        features= torch.from_numpy(features)#.to(self.device)
        if save_weight != 0:
            weight_history = torch.zeros([save_weight, self.beaker_num, self.neuron_num,
                                       self.neuron_num],device=self.device)  # last dim is ensemble size [T,seq_num,neuron_num,neuron_num]
        #features, labels = self.circ.get_feature_label_ensemble_from_samples(features)
        if burnin_stage:
            self.neurons.set_burnin_mode()
        else:
            self.neurons.set_eval_mode()
        for i in range(sample_num):
            #self.neurons.fit(features[i:i+1], labels[i:i+1])
            feature0, label0 = self.circ.get_feature_label_ensemble_from_samples(features[i:i + 1].to(self.device))
            self.neurons.fit(feature0, label0)
            idx = i - (sample_num - save_weight)
            if save_weight!=0 and idx >= 0:
                weight_history[idx, :, :-1, :] = self.neurons.coef_
                weight_history[idx, :, -1, :] = self.neurons.intercept_
        if save_weight!=0:
            return weight_history.permute(0, 3, 1, 2)
            # [T,seq_num,fea_num+1,neuron_num] -> [T,neuron_num,seq_num,fea_num+1]
        else:
            return None

    def train_all_neurons_probe(self, train_features, probe_pattern, pattern_probing_times, neg_times, pos_times):
        # probe_pattern: [2/3, sample_num, neuron_num]
        dtype = torch.float16#int8
        assert len(probe_pattern.shape)==3
        test_num, sample_num, neuron_num = probe_pattern.shape
        ref_sample, probe_time = pattern_probing_times.shape
        assert sample_num == ref_sample
        feature_cont = np.unique(np.concatenate([np.unique(train_features),np.unique(probe_pattern)],0))
        if self.sparse_coding:
            assert feature_cont[0] == 0 and feature_cont[1] == 1
        else:
            assert feature_cont[0] == -1 and feature_cont[1] == 1
        probe_pattern = torch.from_numpy(probe_pattern).to(dtype=dtype).to(device=self.device)
        io_signal = torch.zeros([probe_time, test_num, ref_sample], device=self.device)
        r_signal = torch.zeros([probe_time, test_num, ref_sample], device=self.device)
        #memory_dataset = torch.zeros([probe_time, ref_sample, self.neuron_num],device=self.device)

        probe_features, probe_labels = self.circ.get_feature_label_ensemble_from_samples(probe_pattern.reshape((test_num*sample_num, neuron_num)))
        coef_delta, _ = self.neurons.local_learning_update(probe_features, probe_labels, sparse_coding=self.sparse_coding, coding_f=self.coding_f)
        # sample,neuron_num-1, num_neuron,

        probe_features = probe_features.reshape((
            test_num, sample_num, neuron_num-1, neuron_num)).to(dtype=dtype).to(device=self.device)
        probe_labels = probe_labels.reshape((test_num, sample_num, neuron_num))
        coef_delta = coef_delta.reshape((
            test_num, sample_num, neuron_num-1, neuron_num)).to(dtype=dtype).to(device=self.device)
        assert not np.any(pattern_probing_times < 0), pattern_probing_times
        run_sample_num = np.min(pattern_probing_times[pattern_probing_times >= 0])  # ATTENTION: >=0
        sample_idx = run_sample_num
        self.train_all_neurons(train_features[:sample_idx + 1])
        pattern_probing_times -= run_sample_num
        zero_loc = np.argwhere(pattern_probing_times == 0)
        rate = 0
        memory_pre_distribution = [[] for t in range(probe_time)]
        memory_coding_level = [[] for t in range(probe_time)]
        probe_coding_level = [[] for t in range(probe_time)]
        while True:
            if 0 and rate % 1000 == 0:
                print(f'while {rate}: {sample_idx}-th sample', end=',')
            rate += 1
            coef = self.neurons.coef_[0, :, :] # neuron_num-1, num_neuron
            for probe_s, probe_t in zero_loc:
                memory_pre = self.neurons.forward_linearity(probe_features[:,probe_s])# test_num, neuron_num
                memory = self.neurons.forward_thresholding(memory_pre)# test_num, neuron_num
                temp_probe_pattern = probe_pattern[:,probe_s]#.to(self.device)
                if probe_s>ref_sample//2: # < for familiar; > for novel
                    memory_pre_distribution[probe_t].append((memory_pre[0].cpu().detach().numpy()))
                    memory_coding_level[probe_t].append(torch.mean((memory[0]==1)*1.0).cpu().detach().numpy()[()])
                    probe_coding_level[probe_t].append(torch.mean((temp_probe_pattern[0]==1)*1.0).cpu().detach().numpy()[()])
                if self.sparse_coding:
                    assert self.pre_feature_ratio == 1.0
                    temp_io_signal = coef * coef_delta[:, probe_s, :, :]
                    for test_idx in range(test_num):
                        postis1_idx = temp_probe_pattern[test_idx] == 1
                        postis0_idx = torch.logical_not(postis1_idx)
                        r_signal[probe_t, test_idx, probe_s] = get_phi_coef(memory[test_idx], temp_probe_pattern[test_idx])
                        # io signal is not correct
                        io_signal[probe_t, test_idx, probe_s] = \
                            0.5*torch.mean(temp_io_signal[test_idx, :, postis1_idx]) +\
                            0.5*torch.mean(temp_io_signal[test_idx, :, postis0_idx])
                else:
                    r_signal[probe_t, :, probe_s] = 1-torch.mean(((memory - temp_probe_pattern).abs()), 1)
                    io_signal[probe_t, :, probe_s] = torch.mean(self.neurons.feature_mask * coef * coef_delta[:, probe_s, :, :], (1,2))/self.pre_feature_ratio #.to(self.device)
                    # 0.45 ms for this block in a loop; 2 ms for following fit 1 pattern (256 neurons)
            if not np.any(pattern_probing_times > 0):
                break
            run_sample_num = np.min(pattern_probing_times[pattern_probing_times > 0])  # ATTENTION: >0
            self.train_all_neurons(train_features[sample_idx + 1:sample_idx + 1 + run_sample_num])
            sample_idx += run_sample_num
            pattern_probing_times -= run_sample_num
            zero_loc = np.argwhere(pattern_probing_times == 0)
        self.train_all_neurons(train_features[sample_idx + 1:])

        return r_signal.cpu().numpy(), io_signal.cpu().numpy() #[probe_time, test_num, ref_sample]


    def show_memory_distribution(self, memory_pre_distribution, memory_coding_level, probe_coding_level, pos_times):

        memory_pre_distribution = [np.concatenate(x, 0) for x in memory_pre_distribution]
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.hist(np.concatenate(memory_pre_distribution, 0), bins=500)
        plt.xlabel('sum of inputs')
        plt.subplot(1, 2, 2)
        ub = [np.percentile(x, (1 - self.coding_f) * 100) for x in memory_pre_distribution]
        lb = [np.percentile(x, self.coding_f * 100) for x in memory_pre_distribution]
        std = [np.std(x) for x in memory_pre_distribution]
        plt.plot(pos_times, ub, label=f'ub {1 - self.coding_f}')
        plt.plot(pos_times, lb, label=f'lb {self.coding_f}')
        plt.title(f'coding level {self.coding_f}')
        plt.xlabel('test time')
        plt.ylabel('sum of inputs')
        plt.legend()
        plt.show()
        ub = np.mean(ub)
        std = np.mean(std)
        ppf = norm.ppf(1-self.coding_f)
        with open('coding_level_distribution.txt', 'a+') as f:
            f.write(f'[{self.coding_f},{ub},{std},{ppf},{std*ppf}],\n')

        memory_coding_level = [np.array(x) for x in memory_coding_level]
        probe_coding_level = [np.array(x) for x in probe_coding_level]
        memory_coding_level = np.concatenate(memory_coding_level)
        probe_coding_level = np.concatenate(probe_coding_level)
        plt.rc('figure', figsize=(3 * 2, 3 * 1))
        plt.subplot(1, 2, 1)
        plt.hist(memory_coding_level, color=f'C0', density=True,bins=20)
        # sns.histplot(memory_coding_level, color=f'C0', kde=True, stat='density')
        mean = np.mean(memory_coding_level)
        plt.vlines(mean, 0, 1, 'k')
        plt.title(f'level:{self.coding_f:.4f} mem mean:{mean:.4f}')

        plt.subplot(1, 2, 2)
        print(probe_coding_level)
        plt.hist(probe_coding_level, color=f'C1', density=True,bins=20)
        # sns.histplot(probe_coding_level, color=f'C1', kde=True, stat='density')
        mean = np.mean(probe_coding_level)
        plt.vlines(mean, 0, 1, 'k')
        plt.title(f'level:{self.coding_f:.4f} inpt mean:{mean:.4f}')
        plt.show()

    def build_dataset_protocol_during_training(self, traintest_features, fillin_features, neg_times, pos_times,
                                               protocol=2):
        test_num, sample_size, feature_num =traintest_features.shape
        probe_max_time = fillin_features.shape[0]

        if protocol == 2:
            # run the first half of traintest_features, test during pos_times
            # run fillin_features
            # run the second half of traintest_features, test during neg_times
            half_size = sample_size//2
            ref_sample2 = np.arange(0, half_size)
            pattern_probing_times2 = ref_sample2[:, None] + pos_times[None, :]

            ref_sample1 = np.arange(half_size + probe_max_time, probe_max_time + sample_size)
            pattern_probing_times1 = ref_sample1[:, None] + neg_times[None, :]
            assert len(pos_times) == len(neg_times)
            pattern_probing_times = np.concatenate([pattern_probing_times2, pattern_probing_times1], 0)
            train_features = np.concatenate((
                traintest_features[0, :half_size],fillin_features, traintest_features[0, half_size:]),0)
            r_signal, io_signal = self.train_all_neurons_probe(train_features, traintest_features, pattern_probing_times, neg_times, pos_times,)
            # [probe_time, test_num, ref_sample]
            r_signal_famil = r_signal[:, :, :half_size] # first half
            r_signal_novel =r_signal[:, :, half_size:] # second half
            io_signal_famil = io_signal[:, :, :half_size]
            io_signal_novel =io_signal[:, :, half_size:]
            times_famil = pos_times
            times_novel = neg_times
        elif protocol == 3:
            # run traintest_features, test during pos_times
            # run fillin_features
            ref_sample = np.arange(0, sample_size)
            probe_pattern = traintest_features
            pattern_probing_times = ref_sample[:, None] + pos_times[None, :]
            train_features = np.concatenate((traintest_features[0], fillin_features), 0)
            r_signal_famil, io_signal_famil = self.train_all_neurons_probe(train_features, probe_pattern, pattern_probing_times)
            r_signal_novel, io_signal_novel = None, None
            times_famil = pos_times
            times_novel = None
        else:
            raise ValueError()
        return r_signal_famil, io_signal_famil, times_famil, r_signal_novel, io_signal_novel, times_novel

    def show_neurons_coef_distribution(self):
        all_coef = self.neurons.coef_.cpu().detach().numpy() #[seq_num, fea_num, neuron_num]
        all_intercept = self.neurons.intercept_.cpu().detach().numpy() #[seq_num, neuron_num]
        assert len(all_coef.shape) == 3 and len(all_intercept.shape) == 2
        seq_num = all_coef.shape[0]
        max_coef, min_coef = 10, -10 #np.max(all_coef), np.min(all_coef)
        max_intercept, min_intercept = 10, -10 #np.max(all_intercept), np.min(all_intercept)
        plt.rc('figure', figsize=(3 * seq_num, 3 * 2))
        for seq_idx in range(seq_num):
            coef = all_coef[seq_idx].flatten()
            intercept = all_intercept[seq_idx].flatten()
            plt.subplot(2, seq_num, seq_idx + 1)
            sns.histplot(coef, color=f'C{seq_idx}', kde=True, stat='density')
            plt.vlines(np.mean(coef), 0, 1, 'k')
            plt.title(f'coef beakder idx: {seq_idx}')
            plt.xlim([min_coef, max_coef])

            plt.subplot(2, seq_num, seq_num+seq_idx + 1)
            sns.histplot(intercept, color=f'C{seq_idx}', kde=True, stat='density')
            plt.vlines(np.mean(intercept), 0, 1, 'k')
            plt.title(f'intercept beakder idx: {seq_idx}')
            plt.xlim([min_intercept, max_intercept])
        plt.show()


    def monitor_snr_during_training(self, probe_pattern,
                                    monitored_signal_thre=None, config=None):
        # probe_pattern: [1, neuron_num]
        sample_num, neuron_num = probe_pattern.shape
        target_present_time = config['rpt']
        assert sample_num == 1

        dtype = torch.float16#int8
        feature_cont = np.unique(probe_pattern)
        if self.sparse_coding:
            raise NotImplementedError
        else:
            assert feature_cont[0] == -1 and feature_cont[1] == 1
        probe_pattern_numpy = probe_pattern
        probe_pattern = torch.from_numpy(probe_pattern).to(dtype=dtype).to(device=self.device)
        io_signal = []
        r_signal = []

        probe_features, probe_labels = self.circ.get_feature_label_ensemble_from_samples(probe_pattern.reshape((1, neuron_num)))
        coef_delta, _ = self.neurons.local_learning_update(probe_features, probe_labels, sparse_coding=self.sparse_coding, coding_f=self.coding_f)
        # 1,neuron_num-1, num_neuron
        probe_features = probe_features.reshape((
            1, neuron_num-1, neuron_num)).to(dtype=dtype).to(device=self.device)
        probe_labels = probe_labels.reshape((1, neuron_num))
        coef_delta = coef_delta.reshape((
            neuron_num-1, neuron_num)).to(dtype=dtype).to(device=self.device)

        monitored_signal = 0
        present_time_collect = []
        present_time_count = 0
        sample_idx = 0
        cycle_sample_num = 1000
        while present_time_count < target_present_time:
            if sample_idx % cycle_sample_num == 0:
                train_features = load_aux_patterns(cycle_sample_num, config['dim_num'],
                   aug_pattern_type=config['pattern_type'], verbose=False)
            if monitored_signal <= monitored_signal_thre:
                self.train_all_neurons(probe_pattern_numpy)
                present_time_collect.append(sample_idx)
                present_time_count += 1
            else:
                idx = sample_idx % cycle_sample_num
                self.train_all_neurons(train_features[idx:idx+1])

            coef = self.neurons.coef_[0, :, :]  # neuron_num-1, num_neuron
            memory = self.neurons.forward(probe_features)# 1, neuron_num

            r_signal.append((1-torch.mean(((memory - probe_pattern).abs()))).cpu().numpy())
            io_signal.append((torch.mean(self.neurons.feature_mask * coef * coef_delta)/self.pre_feature_ratio).cpu().numpy())
            monitored_signal = io_signal[-1]
            sample_idx += 1
        return np.array(r_signal), np.array(io_signal), np.array(present_time_collect)

if __name__ == '__main__':
    neuron_num = 2048
    feature_num = neuron_num - 1
    set_seed(0)
    device = 'cuda'
    bcs1 = BCSLayer(beaker_num=4, feature_num=feature_num, neuron_num=neuron_num, isdiscrete=True,
                    ).to(device)
    features1 = torch.from_numpy(np.random.choice([-1, 1], size=(1, feature_num, neuron_num))).to(device)
    labels1 = torch.from_numpy(np.random.choice([-1, 1], size=(1, neuron_num))).to(device)
    bcs1.set_eval_mode()
    bcs1.fit(features1, labels1)
