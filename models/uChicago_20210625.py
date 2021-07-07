import os
import time
from collections import Counter, OrderedDict
from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KernelDensity
from sklearn.tree import DecisionTreeClassifier

from features import feature

from features.feature import extract_feature_average, generate_data, extract_feature_uChicago, \
    extract_feature_avg_uChicago, extract_feature_sliding_window


class AnomalyDetector:

    def __init__(self, model_name='GMM', model_parameters={}, random_state=42):
        self.model_name = model_name
        self.random_state = random_state
        self.model_parameters = model_parameters

    def fit(self, X_train, y_train=None):
        # 1. preprocessing
        # 2. build models
        if self.model_name == 'KDE':
            self.model = KernelDensity(kernel='gaussian', bandwidth=0.5)
            self.model.fit(X_train)
        elif self.model_name == 'GMM':
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
        elif self.model_name == 'OvRLogReg':
            C = self.model_parameters['C']
            self.model = OneVsRestClassifier(
                LogisticRegression(C=C, random_state=self.random_state, solver='liblinear'))
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


def get_X_y(Xs, ys):
    X = []
    Y = []
    for f, y in zip(Xs, ys):
        x = extract_feature_uChicago(f)
        x = extract_feature_avg_uChicago(f)
        X.extend(x)
        Y.append(y)

    return np.asarray(X), np.asarray(Y)


def main(random_state=42):
    # in_dir = 'out/uChicago/20210625/output_mp4/'
    in_dir = 'out/data/data-clean/refrigerator'
    # in_file, video_type = f'{in_dir}/Xy-mkv.dat', 'mkv
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
    # if os.path.exists(in_file):
    #     os.remove(in_file)
    generate_data(in_dir, in_file, video_type=video_type)  # get file_path and label
    meta = feature.load_data(in_file)
    print(in_file)
    X, y = meta['X'], meta['y']
    X, y = get_X_y(X, y)  # get feature data
    print(meta['in_dir'], ', its shape:', meta['shape'])
    print(f'mapping-(activity:(label, cnt)): ', '\n\t' + '\n\t'.join([f'{k}:{v}' for k, v in meta['labels'].items()]))
    mp = {v[0]: k for k, v in meta['labels'].items()}  # idx -> activity name
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    print(f'X_train: {X_train.shape}\nX_test: {X_test.shape}')
    print(f'X_train: {X_train.shape}, y_train: {sorted(Counter(y_train).items(), key=lambda x: x[0])}')
    print(f'X_train: {X_test.shape}, y_test: {sorted(Counter(y_test).items(), key=lambda x: x[0])}')

    # print(X_train[:10])

    res = []
    for n_estimators in [10, 50, 100, 200, 300, 400, 500, 700, 900, 1000]:
        print(f'\nn_estimators: {n_estimators}')
        # 2. build the kde models
        # detector = AnomalyDetector(model_name='KDE', model_parameters = {'bandwidth': 0.1, 'kernel': 'gussisan'})
        # detector = AnomalyDetector(model_name='DT', model_parameters={}, random_state=random_state)
        detector = AnomalyDetector(model_name='RF', model_parameters={'n_estimators': n_estimators},
                                   random_state=random_state)
        # detector = AnomalyDetector(model_name='SVM', model_parameters={'kernel':'rbf'}, random_state=random_state)
        # detector = AnomalyDetector(model_name='SVM', model_parameters={'kernel':'linear'}, random_state=random_state)
        # detector = AnomalyDetector(model_name='OvRLogReg', model_parameters={'C':1}, random_state=random_state)

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

        print('y_test (label, cnt): ', sorted(Counter(y_test).items(), key=lambda x: x[0]))
        acc = sklearn.metrics.accuracy_score(y_test, y_preds)
        print(f'accuracy: {acc}')
        res.append((acc, n_estimators))
        cm = sklearn.metrics.confusion_matrix(y_test, y_preds)
        # print(cm)
        # labels = list(mp.keys())
        w = 15  # width
        # print()
        labels = [f'{v[:w]:<{w}}' for k, v in mp.items()]
        # for v in zip_longest(*labels, fillvalue=' '):
        #     print(' '.join(v))
        # print(' '* 15 + ' '.join(labels)+f'(predicted)')
        print(' ' * 40 + '(predicted)')
        print(' ' * (w + 1) + '\t' + '\t\t'.join([f'({k})' for k, v in mp.items()]))
        for i, vs in enumerate(list(cm)):
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
