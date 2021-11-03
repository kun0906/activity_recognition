import os
from collections import Counter

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ar.features import feature
from ar.features.feature import extract_feature_sliding_window, extract_feature_average, generate_data


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


#
# def generate_data(in_dir ='out/output_mp4', out_file = 'out/Xy.dat'):
#     if type(in_dir) == str:
#         in_dir = [in_dir]
#     elif type(in_dir) == list:
#         pass
#     else:
#         raise NotImplementedError()
#
#     meta = generate.form_X_y(in_dir)
#     generate.dump_data(meta, out_file)
#     return meta
#
# def get_video_file(in_dir = ['refrigerator/']):
#     mp = OrderedDict()
#     i = 0
#     c = 0
#     X = []
#     Y = []
#     for _in_dir in in_dir:
#         for f in sorted(os.listdir(_in_dir)):
#             pth = os.path.join(_in_dir, f)
#             y = generate.extract_label(f)
#             if y == 'no_interaction' or '.txt' in pth: continue
#             if y not in mp.keys():
#                 mp[y] = (i, 1)  # (idx, cnt)
#                 i += 1
#             else:
#                 mp[y] = (mp[y][0], mp[y][1] + 1)
#             X.append(pth)
#             Y.append(mp[y][0])
#             c += 1
#     print(f'{in_dir}: total videos: {c}, and classes: {i}')
#     print(f'{in_dir}: Labels: {mp.items()}')
#     mp2 = {v[0]:k for k, v in mp.items()}
#     meta = {'X': X, 'y': Y, 'shape': (c, ), 'label2idx': mp, 'idx2label': mp2, 'in_dir': in_dir}
#     return meta


def get_X_y(Xs, ys, augment=True):
    X = []
    Y = []
    for f, y in zip(Xs, ys):
        if augment:
            xs = extract_feature_sliding_window(f)
        else:
            xs = extract_feature_average(f)
        X.extend(xs)
        Y.extend([y] * len(xs))

    return Xs, np.asarray(X), np.asarray(Y)


#
# def augment_data(files, labels, idx2label, augment = True):
#     mp = OrderedDict()
#     i = 0
#     c = 0
#     video_id = []
#     X = []
#     Y = []
#     for pth, y in zip(files, labels):
#         # print(pth, y)
#         if augment:
#             xs = generate.extract_feature_sliding_window(pth)
#         else:
#             xs = generate.extract_feature_average(pth)
#         name = idx2label[y]
#         if name == 'no_interaction': continue
#         if name not in mp.keys():
#             mp[name] = (y, len(xs))  # (idx, cnt)
#         else:
#             mp[name] = (mp[name][0], mp[name][1] + 1)
#         video_id.extend([pth]*len(xs))
#         X.extend(xs)
#         Y.extend([mp[name][0]] * len(xs))
#         c += 1
#     print(f'total videos: {len(X)}, and classes: {len(mp.keys())}')
#     print(f'Labels: {mp.items()}')
#     X = np.asarray(X)
#     Y = np.asarray(Y)
#     meta = {'X': X, 'y': Y, 'shape': (c, len(X[0])), 'label2idx': mp, 'idx2label': idx2label, 'in_dir': ''}
#     return video_id, X, Y, meta


def main(random_state=42):
    in_dir = '../out/data/data-clean/refrigerator'
    # in_dir = 'out/data/trimmed/data-clean/refrigerator'
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

    X_train, X_test, y_train, y_test = train_test_split(meta['X'], meta['y'], test_size=0.3, random_state=random_state)
    mp = meta['idx2label']

    # augment train set by sliding window
    # video_id_train, X_train, y_train, meta_train = augment_data(X_train, y_train, mp, augment=True)
    # video_id_test, X_test, y_test, meta_test = augment_data(X_test, y_test, mp, augment= False)
    video_id_train, X_train, y_train = get_X_y(X_train, y_train, augment=True)
    video_id_test, X_test, y_test = get_X_y(X_test, y_test, augment=False)
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
        detector = AnomalyDetector(model_name='SVM', model_parameters={'kernel': 'linear'}, random_state=random_state)
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

        # 5 get misclassification
        err_mp = {}
        for v_id, y_t, y_p in zip(video_id_test, y_test, y_preds):
            if y_t != y_p:
                name = f'{mp[y_t]}({y_t}):{v_id}'
                if name not in err_mp.keys():
                    err_mp[name] = [f'{mp[y_p]}({y_p})']
                else:
                    err_mp[name].append(f'{mp[y_p]}({y_p})')

                # print(f'{mp[y_t]} -> {mp[y_p]}')
        print('***misclassified classes:')
        print('\t' + '\n\t'.join([f'{k}->{Counter(vs)}' for k, vs in err_mp.items()]))
    print(res)


if __name__ == '__main__':
    main()
