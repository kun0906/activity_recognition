import os
import time
from collections import Counter, OrderedDict
from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import feature

#
# def _generate(n=100):
#     np.random.seed(42)
#     X = np.concatenate((np.random.normal(0, 1, int(0.3 * n)),
#                         np.random.normal(5, 1, int(0.7 * n))))[:, np.newaxis]
#
#     return X

#
# def _generate_data(n=1000):
#     # ----------------------------------------------------------------------
#     # Plot a 1D density example
#     X = _generate(n=n)
#     X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
#
#     true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
#                  + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))
#
#     fig, ax = plt.subplots()
#     ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
#             label='input distribution')
#     # colors = ['navy', 'cornflowerblue', 'darkorange']
#     # kernels = ['gaussian', 'tophat', 'epanechnikov']
#     # lw = 2
#     #
#     # for color, kernel in zip(colors, kernels):
#     #     kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
#     #     log_dens = kde.score_samples(X_plot)
#     #     ax.plot(X_plot[:, 0], np.exp(log_dens), color=color, lw=lw,
#     #             linestyle='-', label="kernel = '{0}'".format(kernel))
#
#     ax.text(6, 0.38, "N={0} points".format(n))
#
#     ax.legend(loc='upper left')
#     ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')
#
#     ax.set_xlim(-4, 9)
#     ax.set_ylim(-0.02, 0.4)
#     plt.show()
#
#     y = ''
#     return X, y
from features.feature import extract_feature_average, generate_data


class AnomalyDetector:

    def __init__(self, model_name='GMM', model_parameters= {}, random_state=42):
        self.model_name = model_name
        self.random_state = random_state
        self.model_parameters = model_parameters

    def fit(self, X_train, y_train=None):
        # 1. preprocessing
        # 2. build models
        if self.model_name == 'KDE':
            self.model = KernelDensity(kernel='gaussian', bandwidth=0.5)
            self.model.fit(X_train)
        elif self.model_name =='GMM':
            pass
        elif self.model_name == 'DT':
            self.model = DecisionTreeClassifier(random_state=self.random_state)
            self.model.fit(X_train, y_train)
        elif self.model_name == 'RF':
            n_estimators = self.model_parameters['n_estimators']
            self.model = RandomForestClassifier(n_estimators, random_state=self.random_state)
            self.model.fit(X_train, y_train)
        elif self.model_name == 'SVM':
            kernel = self.model_parameters['kernel']
            self.model = sklearn.svm.SVC(kernel=kernel, random_state=self.random_state)
            self.model.fit(X_train, y_train)


    def get_threshold(self, X_train, q=0.95):
        # 3. anomaly theadhold: t
        log_dens = self.model.score_samples(X_train)
        self.thres = np.quantile(np.exp(log_dens), q=q)

    def predict_prob(self, X):
        log_dens = self.model.score_samples(X)

        return np.exp(log_dens)

    def predict(self, X):
        dens = self.predict_prob(X)
        dens[dens < self.thres] = 1
        dens[dens >= self.thres] = 0

        return dens

#
# def generate_anomaly(n=100, name='same'):
#     if name == 'same':
#         X = _generate(n=1000)
#         X = np.asarray([v for v in X if (-1 > v or 1 < v < 4 or 7 < v)][:n])
#         y = [1] * len(X)
#     else:
#         np.random.seed(42)
#         X = np.concatenate((np.random.normal(-2, 3, int(0.3 * n)),
#                             np.random.normal(8, 3, int(0.7 * n))))[:, np.newaxis]
#
#         y = [1] * n
#     return X, y

#
# def generate_data(files):
#
#     X = []
#     for features in files:
#         name, f = features.split()
#         X.append(np.load(f+'.npy'))
#
#     print(X)
#     X_train  = np.asarray([v.flatten() for v in X] * 100)
#     y_train = [0] * 100 + [1] * 100
#
#     X_test =   np.asarray([v.flatten() for v in X] * 100)
#     y_test = [0] * 100 + [1] * 100
#
#     #
#     # n = 1000
#     # X, y = _generate_data(n=n)  # includes normal and novelty.
#     #
#     # X_normal = []
#     # y_normal = []
#     # X_novelty = []
#     # y_novelty = []
#     # for v in X:
#     #     if (-2 < v < 2 or 3 < v < 7):
#     #         X_normal.append(v)
#     #         y_normal.append(0)
#     #     else:
#     #         X_novelty.append(v)
#     #         y_novelty.append(1)
#     #
#     # n_test_normal = 30
#     # X_normal = np.asarray(X_normal)
#     # X_novelty = np.asarray(X_novelty)
#     # X_train_normal, X_test_normal = train_test_split(X_normal, test_size=n_test_normal)
#     # X_train = X_train_normal[:500, :]
#     # # X = np.asarray(X_normal  + X_novelty)[:, np.newaxis]
#     # # y = y_normal+y_novelty
#     # y_train = [0] * len(X_train)
#     # y_test_normal = [0] * n_test_normal
#     # # X_test_novel, y_test_novel = generate_anomaly(n=n_test_normal)
#     # _, X_test_novel = train_test_split(X_novelty, test_size=n_test_normal)
#     # y_test_novel = [1] * n_test_normal
#     # X_test = np.concatenate([X_test_normal, X_test_novel], axis=0)
#     # y_test = y_test_normal + y_test_novel
#     #
#     # X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state=42)
#     # X_test, y_test = sklearn.utils.shuffle(X_test, y_test, random_state=42)
#
#
#     return X_train, y_train, X_test, y_test
#

# def get_data(random_state=42, case = 'comb'):
#     # 1. get data points from 1-D distribution.
#     # X_train, y_train, X_test, y_test = generate_data(files=open('out/video_feature_list.txt').readlines())
#     if case == 'idv':
#
#     elif case == 'comb':
#
#
#     return X_train, X_test, y_train, y_test, mp

def get_shape(X, Y, Z=[]) -> tuple:
	return len(X), len(Y), len(Z)


def get_X_y(Xs, ys):
	X = []
	Y = []
	for f, y in zip(Xs, ys):
		x = extract_feature_average(f)
		X.extend(x)
		Y.append(y)

	return np.asarray(X), np.asarray(Y)


def main(random_state=42):
    in_dir = 'out/data/data-clean/refrigerator'
    # in_dir = 'out/data/trimmed/data-clean/refrigerator'
    in_file = f'{in_dir}/Xy-mkv.dat'
    in_file, video_type = f'{in_dir}/Xy-mp4.dat', 'mp4'
    # in_file = f'{in_dir}/Xy-comb.dat'
    # if not os.path.exists(in_file):
    #     if 'mkv' in in_file:
    #         in_dir = 'out/output_mkv'
    #         out_file = 'out/Xy-mkv.dat'
    #     elif 'mp4' in in_file:
    #         in_dir = f'{in_dir}'
    #         out_file = f'{in_dir}/Xy-mp4.dat'
    #     elif 'comb' in in_file:
    #         in_dir = ['out/output_mp4', 'out/output_mkv']
    #         out_file = 'out/Xy-comb.dat'
    #     else:
    #         raise NotImplementedError
    if os.path.exists(in_file):
        os.remove(in_file)
    generate_data(in_dir, in_file, video_type=video_type)  # get file_path and label
    meta = feature.load_data(in_file)
    X, y = meta['X'], meta['y']
    X, y = get_X_y(X, y)
    print(meta['in_dir'], ', its shape:', meta['shape'])
    print(f'mapping-(activity:(label, cnt)): ', '\n\t' + '\n\t'.join([f'{k}:{v}' for k, v in meta['labels'].items()]))
    mp = {v[0]: k for k, v in meta['labels'].items()}  # idx -> activity name
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    print(f'X_train: {X_train.shape}\nX_test: {X_test.shape}')
    print(f'X_train: {X_train.shape}, y_train: {sorted(Counter(y_train).items(), key=lambda x: x[0])}')
    print(f'X_train: {X_test.shape}, y_test: {sorted(Counter(y_test).items(), key=lambda x: x[0])}')

    # print(X_train[:10])
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    # X = np.concatenate(X, axis=0)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    res = []
    for n_estimators in [10, 50, 100, 200, 300, 400, 500, 700, 900, 1000]:
        print(f'\nn_estimators: {n_estimators}')
        # 2. build the kde models
        # detector = AnomalyDetector(model_name='KDE', model_parameters = {'bandwidth': 0.1, 'kernel': 'gussisan'})
        # detector = AnomalyDetector(model_name='DT', model_parameters={}, random_state=random_state)
        # detector = AnomalyDetector(model_name='RF', model_parameters={'n_estimators': n_estimators},
        #                            random_state=random_state)
        # detector = AnomalyDetector(model_name='SVM', model_parameters={'kernel':'rbf'}, random_state=random_state)
        # detector = AnomalyDetector(model_name='SVM', model_parameters={'kernel': 'linear'}, random_state=random_state)
        detector = AnomalyDetector(model_name='OvRLogReg', model_parameters={'C': 1}, random_state=random_state)
        detector.fit(X_train, y_train)
        #
        # # 3. compute the threshold
        # detector.get_threshold(X_train, q=0.01)
        # # print(detector.predict_prob(X_train))

        # 4. evaulation
        y_preds = detector.model.predict(X_test)
        # y_preds = []
        # X_test = np.asarray(X_test)[:, np.newaxis]
        #
        # for i, x in enumerate(X_test):
        #     # time.sleep(1)
        #     y_prob = detector.predict_prob(x).item()
        #     # y_pred = detector.predict(x).item()
        #     if y_prob < detector.thres:
        #         print(f'{i}, x: {x}-> ***novelty')
        #         y_pred = 1
        #     else:
        #         print(f'{i}, x: {x}-> normal ')
        #         y_pred = 0
        #     y_preds.append(y_pred)

        print('y_test (label, cnt): ', sorted(Counter(y_test).items(), key=lambda x:x[0]))
        acc = sklearn.metrics.accuracy_score(y_test, y_preds)
        print(f'accuracy: {acc}')
        res.append((acc, n_estimators))
        cm = sklearn.metrics.confusion_matrix(y_test, y_preds)
        # print(cm)
        # labels = list(mp.keys())
        w = 15   # width
        # print()
        labels = [f'{v[:w]:<{w}}' for k,v in mp.items()]
        # for v in zip_longest(*labels, fillvalue=' '):
        #     print(' '.join(v))
        # print(' '* 15 + ' '.join(labels)+f'(predicted)')
        print(' '* 40 + '(predicted)')
        print(' '*(w+1)  + '\t' + '\t\t'.join([f'({k})' for k, v in mp.items()]))
        for i,vs in enumerate(list(cm)):
            print(f'{mp[i][:w]:<{w}} ({i})\t', '\t\t'.join([f'{v}' for v in list(vs)]))


        # # 5 get misclassification
        # err_mp = {}
        # for y_t, y_p in zip(y_test, y_preds):
        #     if y_t != y_p:
        #         name = f'{mp[y_t]}({y_t})'
        #         if name not in err_mp.keys():
        #             err_mp[name] = [f'{mp[y_p]}({y_p})']
        #         else:
        #             err_mp[name].append(f'{mp[y_p]}({y_p})')
        #
        #         # print(f'{mp[y_t]} -> {mp[y_p]}')
        # print('***misclassified classes:')
        # print('\t'+'\n\t'.join([ f'{k}->{Counter(vs)}' for k, vs in err_mp.items()]))
    print(res)

if __name__ == '__main__':
    main()
