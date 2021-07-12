import os
from collections import Counter
from shutil import copyfile

import cv2
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import feature
from features.feature import extract_feature_average, generate_data, extract_video_feature


def _mirror_video(in_file, out_dir):
    """
    https://stackoverflow.com/questions/29317262/opencv-video-saving-in-python
    https://docs.opencv.org/4.5.2/dd/d43/tutorial_py_video_display.html
    https://stackoverflow.com/questions/61659346/how-to-get-4-character-codec-code-for-videocapture-object-in-opencv
    Parameters
    ----------
    in_file
    out_dir

    Returns
    -------

    """
    _, file_name = os.path.split(in_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    copyfile(in_file, os.path.join(out_dir, file_name))
    file_name, ent = file_name.split('.')
    # capture video
    cap = cv2.VideoCapture(in_file)
    out_file = os.path.join(out_dir, file_name + '-mirrored.' + ent)
    # Get the Default resolutions
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(
    #     *f"{fourcc & 255:c},{(fourcc >> 8) & 255:c}, {(fourcc >> 16) & 255:c}, {(fourcc >> 24) & 255:c}")
    fourcc = cv2.VideoWriter_fourcc(*(chr(fourcc & 0xff) + chr((fourcc >> 8) & 0xff) + chr((fourcc >> 16) & 0xff)
                                      + chr((fourcc >> 24) & 0xff)))
    # print(fourcc)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # Define the codec and filename.
    fps = cap.get(cv2.CAP_PROP_FPS)
    # out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
    out = cv2.VideoWriter(out_file, fourcc, fps, (frame_width, frame_height), isColor=True)
    while True:
        ret, img = cap.read()
        # print(ret)
        if ret:
            # cv2.imshow('Original Video',img)
            # flip for truning(fliping) frames of video
            img2 = cv2.flip(img, 1)  # Horizontal
            # cv2.imshow('Flipped video',img2)
            out.write(img2)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(out_file)


def mirror_video(in_dir, device_type='refrigerator', video_type='mp4', out_dir=None):
    # # mirror video
    # for device_dir in in_dir:  # 'data/data-clean/refrigerator
    #     out_dir_sub = ''
    #     if device_type not in device_dir: continue
    #     for activity_dir in os.listdir(device_dir):
    #         out_dir_activity = activity_dir
    #         activity_dir = os.path.join(device_dir, activity_dir)
    #         if not os.path.exists(activity_dir) or '.DS_Store' in activity_dir or not os.path.isdir(
    #                 activity_dir): continue
    #         for participant_dir in os.listdir(activity_dir):
    #             out_dir_participant = participant_dir
    #             out_dir_sub = os.path.join(participant_dir)
    #             participant_dir = os.path.join(activity_dir, participant_dir)
    #             if not os.path.exists(participant_dir) or '.DS_Store' in participant_dir: continue
    #             # print(participant_dir)
    #             for f in sorted(os.listdir(participant_dir)):
    #                 if video_type == 'mp4'  in f:
    #                     x = os.path.join(participant_dir, f)
    #                 elif video_type == 'mkv' in f:
    #                     x = os.path.join(participant_dir, f)
    #                 else:
    #                     continue
    #                 _mirror_video(x, out_dir=os.path.join(out_dir, out_dir_activity, out_dir_participant))

    extract_video_feature(in_dir=os.path.dirname(out_dir))


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
        x = extract_feature_average(f)
        X.extend(x)
        Y.append(y)

    return np.asarray(X), np.asarray(Y)


def split_train_test_npy(meta, test_size=0.3, is_mirror_test_set=False, random_state=42):
    X = []  # doesn't include 'mirrored' npy
    y = []  # doesn't include 'mirrored' npy
    X_mirrored = []
    y_mirrored = []
    for x_, y_ in zip(meta['X'], meta['y']):
        if 'mirrored_vgg.npy' not in x_:
            X.append(x_)
            y.append(y_)
            # to make X and X_mirriored have the same order.
            ent = '_vgg.npy'
            new_x_ = x_[:-len(ent)] + '-mirrored' + ent
            # print(x_, new_x_)
            X_mirrored.append(new_x_)
            y_mirrored.append(y_)

    X, y = get_X_y(X, y)  # extract features from 'npy' files
    # print(meta['in_dir'], ', its shape:', meta['shape'])
    # print(f'mapping-(activity:(label, cnt)): ', '\n\t' + '\n\t'.join([f'{k}:{v}' for k, v in meta['labels'].items()]))
    # mp = {v[0]: k for k, v in meta['labels'].items()}  # idx -> activity name
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_mirrored, y_mirrored = get_X_y(X_mirrored, y_mirrored)  # extract features from 'npy' files
    X_mirrored_train, X_mirrored_test, y_mirrored_train, y_mirrored_test = \
        train_test_split(X_mirrored, y_mirrored, test_size=test_size, random_state=random_state)

    X_train = np.concatenate([X_train, X_mirrored_train], axis=0)
    y_train = np.concatenate([y_train, y_mirrored_train], axis=0)

    if is_mirror_test_set:
        X_test = np.concatenate([X_test, X_mirrored_test], axis=0)
        y_test = np.concatenate([y_test, y_mirrored_test], axis=0)

    print(f'X_train: {X_train.shape}\nX_test: {X_test.shape}')
    print(f'X_train: {X_train.shape}, y_train: {sorted(Counter(y_train).items(), key=lambda x: x[0])}')
    print(f'X_train: {X_test.shape}, y_test: {sorted(Counter(y_test).items(), key=lambda x: x[0])}')

    return X_train, X_test, y_train, y_test


def main(random_state=42):
    # extract feature data (npy) by CNN
    # in_dir_raw = 'data/data-clean/refrigerator'
    # in_dir = 'data/mirrored/data-clean/refrigerator'
    # mirror_video([in_dir_raw], out_dir = in_dir)  # mirror videos

    in_dir = 'out/data/mirrored/data-clean/refrigerator'
    # in_dir = 'out/data/data-clean/refrigerator'
    # in_dir = 'out/data/trimmed/data-clean/refrigerator'
    # in_dir = 'out/data/mirror/data-clean/refrigerator'
    # in_file = f'{in_dir}/Xy-mkv.dat'
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
    generate_data(in_dir, in_file, video_type=video_type)  # get file_path (npy) and label
    meta = feature.load_data(in_file)  # load 'npy' files
    # X, y = meta['X'], meta['y']
    # X, y = get_X_y(X, y)    # extract features from 'npy' files
    # print(meta['in_dir'], ', its shape:', meta['shape'])
    # print(f'mapping-(activity:(label, cnt)): ', '\n\t' + '\n\t'.join([f'{k}:{v}' for k, v in meta['labels'].items()]))
    mp = {v[0]: k for k, v in meta['labels'].items()}  # idx -> activity name
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    # print(f'X_train: {X_train.shape}\nX_test: {X_test.shape}')
    # print(f'X_train: {X_train.shape}, y_train: {sorted(Counter(y_train).items(), key=lambda x: x[0])}')
    # print(f'X_train: {X_test.shape}, y_test: {sorted(Counter(y_test).items(), key=lambda x: x[0])}')

    X_train, X_test, y_train, y_test = split_train_test_npy(meta, test_size=0.3,
                                                            is_mirror_test_set=True, random_state=random_state)
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
        detector = AnomalyDetector(model_name='RF', model_parameters={'n_estimators': n_estimators},
                                   random_state=random_state)
        # detector = AnomalyDetector(model_name='SVM', model_parameters={'kernel':'rbf'}, random_state=random_state)
        # detector = AnomalyDetector(model_name='SVM', model_parameters={'kernel': 'linear'}, random_state=random_state)
        # detector = AnomalyDetector(model_name='OvRLogReg', model_parameters={'C': 1}, random_state=random_state)
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
