"""
General functions for loading data from disk and generating random patterns
"""
import roblib
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
import pickle
import os.path
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import sys
import matplotlib.pyplot as plt
import settings

def preprocess_pipeline(x, pl_des):
    for k, v in pl_des:
        assert k in ['pca', 'lda']
        print(k, v, end=',')
        if k == 'pca':
            pca = PCA(v)
            pca_mean = x.mean(0)
            x = pca.fit_transform(x - pca_mean)
        elif k == 'lda':
            lda = LDA(n_components=v)
            x = lda.fit_transform(x)
    print('')
    return x


def vgg_preprocess(pl_des=None):
    if pl_des is None:
        pl_des = [('pca', 2048)]  # [('pca',1024)]

    # load original vggface
    file = settings.DATAPATH / "name_and_pattern.pkl"
    with open(file, "rb") as f:
        [name_list, face_list] = pickle.load(f, encoding='bytes')
    labels_name = [x[:7] for x in name_list]
    features = np.array(face_list)
    ct = labels_name.count(labels_name[-1])
    labels_name = labels_name[:-ct]
    features = features[:-ct]

    # pca & binarize
    np.random.seed(0)
    random_index = np.random.choice(features.shape[0], size=features.shape[0], replace=False)
    features = preprocess_pipeline(features, pl_des)[random_index]
    labels = LabelEncoder().fit_transform(np.array(labels_name))[random_index]
    identities = np.arange(len(np.unique(labels_name)))

    final_features=[]
    poses_loc_of_ids = [np.where(labels == id)[0] for id in identities]
    # locations of poses for every identity
    for pose_idx in range(50): # 2
        pose_locs = [poses_loc_of_id[pose_idx] for poses_loc_of_id in poses_loc_of_ids]
        # location of one specific pose for every identity
        final_features.append(features[pose_locs])
    roblib.dump(final_features, settings.DATAPATH / "vgg_pca2048_face.pkl")

    norm_mean = np.mean(features, 0)
    norm_cov = np.cov(features, rowvar=False)
    sign_keep_ratio = np.mean(bin(final_features[0]) == bin(final_features[1]), 0)
    roblib.dump([norm_mean, norm_cov, sign_keep_ratio], settings.DATAPATH / "vgg_pca2048_genface_par.pkl")


def load_random_pattern(sample_num, feature_num, sparse_coding=False, coding_f=0.5):
    if sparse_coding:
        choice_pool = [0, 1]
        Pr = [1-coding_f, coding_f]
    else:
        choice_pool = [-1, 1]
        Pr = [0.5, 0.5]
    features = np.zeros([sample_num, feature_num], dtype=np.float16)
    assert sample_num % 10 == 0
    sample_num_over10 = sample_num // 10
    for i in range(10):
        features[i * sample_num_over10: (i + 1) * sample_num_over10] = np.random.choice(
            choice_pool, size=(sample_num_over10, feature_num), p=Pr)
    return features


def load_genface_patterns(sample_num, feature_num):
    norm_mean, norm_cov, sign_keep_ratio = roblib.load(settings.DATAPATH / "vgg_pca2048_genface_par.pkl")
    max_feature_num = len(norm_mean)
    assert feature_num<=max_feature_num, (feature_num, max_feature_num)

    print(f'    Loading generated {sample_num} faces of size {feature_num}')
    features = np.zeros([sample_num, feature_num], dtype=np.float16)
    assert sample_num%10==0
    sample_num_over10 = sample_num // 10
    for i in range(10):
        features[i*sample_num_over10: (i+1)*sample_num_over10] = np.random.multivariate_normal(
            norm_mean[:feature_num], norm_cov[:feature_num, :feature_num], size=sample_num_over10)
    return bin(features)

def bin(x):
    return 2 * (x > np.median(x, 0)).astype(np.float16) - 1


def load_realface_patterns(sample_num, feature_num, pose_num):
    final_features=roblib.load(settings.DATAPATH / "vgg_pca2048_face.pkl")
    max_sample_num, max_feature_num = final_features[0].shape
    assert sample_num<=max_sample_num, (sample_num, max_sample_num)
    assert feature_num<=max_feature_num, (feature_num, max_feature_num)
    return [bin(features1pose)[:sample_num,:feature_num] for features1pose in final_features[:pose_num]]
    # median computed on all samples


def load_aux_patterns(sample_num, feature_num, aug_pattern_type=None, sparse_coding=False, coding_f=0.5, verbose=True, face_fillin_pattern=False, real_face_start_from=2000):
    if sparse_coding:
        assert aug_pattern_type!='face'
    if verbose:
        print(f'Loading burnin/fillin {sample_num} {aug_pattern_type} patterns of {feature_num} dimension')
    if aug_pattern_type=='face':
        if face_fillin_pattern:
            pt = load_realface_patterns(8621, feature_num, 50)
            pt = np.array(pt)[:,real_face_start_from:,:] # 50, sample_num-real_face_start_from, feature_num
            np.random.seed(2020)
            pt = pt.reshape([-1, feature_num])
            random_index = np.random.choice(pt.shape[0], size=pt.shape[0], replace=False)
            return pt[random_index][:sample_num]
        return load_genface_patterns(sample_num, feature_num)
    elif aug_pattern_type=='rand':
        return load_random_pattern(sample_num, feature_num, sparse_coding=sparse_coding, coding_f=coding_f)
    else:
        return


def load_traintest_patterns(sample_num, feature_num, pattern_type=None, sparse_coding=False, coding_f=0.5):
    if sparse_coding:
        assert pattern_type!='face'
    print(f'Loading training/testing {sample_num} {pattern_type} patterns of {feature_num} dimension')
    if pattern_type=='face':
        final_features = load_realface_patterns(sample_num * 2, feature_num, 2)
        features_pose1=final_features[0]
        features_pose2=final_features[1]
        features = np.array([features_pose1[:sample_num],
                             features_pose2[:sample_num],
                             features_pose1[sample_num:],]) # [3, sample_num, feature_num]
        test_type = ['same', 'noisy', 'new']

    elif pattern_type=='rand':
        features=load_random_pattern(sample_num * 2, feature_num, sparse_coding=sparse_coding, coding_f=coding_f)
        features = np.array([features[:sample_num],
                             features[sample_num:],]) # [2, sample_num, feature_num]
        test_type = ['same', 'new']
    return features, test_type


if __name__ == '__main__':
    pass