"""

run the below command under 'activity_recognition' folder:
    PYTHONPATH=../:./ python3 models/classical/detector_feature_A_mirror-2.py



"""

import os
import shutil
from collections import Counter
from shutil import copyfile

import cv2
import numpy as np
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import feature
from features.feature import extract_feature_average, generate_data, extract_video_feature, \
    extract_feature_sliding_window, extract_feature_sampling, load_data, dump_data
from features.video.model_tf import CNN_tf
from features.video.utils import load_video
from features.video.video import trim


def _extract_video_feature(model, in_file, out_dir):
    # in_file = 'data/data-clean/refrigerator/open_close_fridge/1/open_close_fridge_3_1615392727_2.mkv'
    video_name = os.path.splitext(os.path.basename(in_file))[0]
    out_file = os.path.join(out_dir, '{}_{}.npy'.format(video_name, model.net_name))
    if os.path.exists(out_file):
        return out_file

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    batch_sz = 32
    # sampling: only extract the first frame in each second from the video.
    video_tensor = load_video(in_file, model.desired_size)
    # extract features
    features = model.extract(video_tensor, batch_sz)
    # save features
    np.save(os.path.splitext(out_file)[0], features)

    return out_file


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
     # in_file = 'data/data-clean/refrigerator/open_close_fridge/1/open_close_fridge_3_1615392727_2.mkv'
    """
    _, file_name = os.path.split(in_file)
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    # copyfile(in_file, os.path.join(out_dir, file_name))
    file_name, ent = file_name.split('.')
    out_file = os.path.join(out_dir, file_name + '-mirrored.' + ent)
    if os.path.exists(out_file):
        return out_file

    # capture video
    cap = cv2.VideoCapture(in_file)
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
    try:
        out = cv2.VideoWriter(out_file, fourcc, fps, (frame_width, frame_height), isColor=True)
    except Exception as e:
        print(e)
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
    print(f'_mirror_video: {out_file}')
    return out_file


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

    return Xs, np.asarray(X), np.asarray(Y)


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


def augment_train(train_meta, augment_type='camera_1+camera_2+camera_3', is_mirror=False):
    X_meta = []
    X = []
    Y = []
    if augment_type == 'camera_1+camera_2+camera_3':
        # combine all camera data, but without mirrored data
        for name, train in train_meta.items():
            cnt = 0
            for vs in train:
                video_path, cnn_feature, y_label, y_idx, x = vs
                if len(x) == 0: continue
                if not is_mirror and '-mirrored' in video_path: continue
                X.extend(x)
                Y.extend(len(x) * [y_idx])
                X_meta.extend(len(x) * [video_path])
                cnt += len(x)
            print(f'{name}_train: {cnt}')
    elif augment_type == 'camera_1+camera_2':
        # combine all camera data, but without mirrored data
        for name, train in train_meta.items():
            if name == 'camera_3': continue
            cnt = 0
            for vs in train:
                video_path, cnn_feature, y_label, y_idx, x = vs
                if len(x) == 0: continue
                if not is_mirror and '-mirrored' in video_path: continue
                X.extend(x)
                Y.extend(len(x) * [y_idx])
                X_meta.extend(len(x) * [video_path])
                cnt += len(x)
            print(f'{name}_train: {cnt}')
    elif augment_type == 'camera_1' or augment_type == 'camera_2' or augment_type == 'camera_3':
        # only use camera_i data
        for name, train in train_meta.items():
            if name != augment_type: continue
            for vs in train:
                video_path, cnn_feature, y_label, y_idx, x = vs
                if len(x) == 0: continue
                if not is_mirror and '-mirrored' in video_path: continue
                X.extend(x)
                Y.extend(len(x) * [y_idx])
                X_meta.extend(len(x) * [video_path])
    else:
        msg = augment_type
        raise ValueError(msg)

    return X_meta, np.asarray(X), np.asarray(Y)


def augment_test(test_meta, augment_type='camera_1+camera_2+camera_3', is_mirror=False):
    X_meta = []
    X = []
    Y = []
    if augment_type == 'camera_1+camera_2+camera_3':
        # combine all camera data, but without mirrored data
        for name, test in test_meta.items():
            for vs in test:
                video_path, cnn_feature, y_label, y_idx, x = vs
                if len(x) == 0: continue
                if not is_mirror and '-mirrored' in video_path: continue
                X.extend(x)
                Y.extend(len(x) * [y_idx])
                X_meta.extend(len(x) * [video_path])
    elif augment_type == 'camera_1' or augment_type == 'camera_2' or augment_type == 'camera_3':
        # only use camera_i data
        for name, test in test_meta.items():
            if name != augment_type: continue
            for vs in test:
                video_path, cnn_feature, y_label, y_idx, x = vs
                if len(x) == 0: continue
                if not is_mirror and '-mirrored' in video_path: continue
                X.extend(x)
                Y.extend(len(x) * [y_idx])
                X_meta.extend(len(x) * [video_path])
    else:
        msg = augment_type
        raise ValueError(msg)

    return X_meta, np.asarray(X), np.asarray(Y)


def tsne_plot(X, y, y_label, random_state=42):
    """

    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    tsne_plot(X[:, :2], X[:, 2])


    Parameters
    ----------
    X
    y

    Returns
    -------

    """
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import ListedColormap

    X_embedded = TSNE(n_components=2, random_state=random_state).fit_transform(X)
    # df = pd.DataFrame(np.concatenate([X_embedded, np.reshape(y, (-1, 1))], axis=1), columns=['x1', 'x2', 'y'])
    # print(df.head(5))
    # g = sns.scatterplot(data=df, x="x1", y="x2", hue="y", palette="deep")
    # # g.set(xticklabels=[])
    # # g.set(yticklabels=[])
    # plt.show()

    df = pd.DataFrame(np.concatenate([X_embedded, np.reshape(y, (-1, 1)), np.reshape(y_label, (-1, 1))], axis=1),
                      columns=['x1', 'x2', 'y', 'y_label'])
    df = df.astype({"x1": float, "x2": float, 'y': int, 'y_label': str})
    print(df.info())
    print(df.head(5))
    print(df.describe())
    g = sns.scatterplot(data=df, x="x1", y="x2", hue="y_label", palette='deep', s=50, alpha=0.3)
    g.set_title('Refrigerator')
    # Put the legend out of the figure
    # Note that the (1.05, 1) coordinates correspond to the (x, y) coordinates where the legend should be placed
    # and the borderaxespad specifies the padding between the axes and the border legend.
    #  bbox (x, y, width, height)
    g.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), ncol=1, borderaxespad=0,
             fancybox=False, shadow=False, fontsize=8, title='classes')
    # g.legend(loc='center left', bbox_to_anchor=(1.25, 1), ncol=1, borderaxespad=0.)
    plt.tight_layout()
    plt.show()

    ### FacetGrid
    grid = sns.FacetGrid(df, col="y_label", hue="y_label", hue_order=list(sorted(set(y_label))), col_wrap=3)
    grid.map(sns.scatterplot, "x1", "x2", s=100, alpha=0.3)
    grid.add_legend()
    plt.show()

    ### 3D
    X_embedded = TSNE(n_components=3, random_state=random_state).fit_transform(X)
    df = pd.DataFrame(np.concatenate([X_embedded, np.reshape(y, (-1, 1)), np.reshape(y_label, (-1, 1))], axis=1),
                      columns=['x1', 'x2', 'x3', 'y', 'y_label'])
    df = df.astype({"x1": float, "x2": float, "x3": float, 'y': int, 'y_label': str})
    sns.set(style="white")
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection = '3d')
    # ax.scatter(df['x1'], df['x2'], df['x3'])
    # plt.show()
    # axes instance
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    # plot
    # get colormap from seaborn
    cmap = ListedColormap(sns.color_palette("deep", 5).as_hex())
    sc = ax.scatter(df['x1'], df['x2'], df['x3'], s=40, c=df['y'].values, marker='o', cmap=cmap, alpha=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.legend(*sc.legend_elements())
    #  plt.tight_layout(pad = 2.0, rect=(0.3, 0.3, 0.8, 0.8))   # not works with axes. savefig works.
    # save
    plt.savefig("refirgerator", bbox_inches='tight')
    plt.show()


def split_train_test_video(meta, video_type='_1.mp4', test_size=0.3, random_state=42):
    # only get camera_1 data  (_1.mp4)
    camera_1 = [(f, feat, y) for f, feat, y in meta['data'] if video_type in f]  # '_1.mp4'
    camera_2 = [(f, feat, y) for f, feat, y in meta['data'] if '_2.mkv' in f]
    camera_3 = [(f, feat, y) for f, feat, y in meta['data'] if '_3.mp4' in f]
    print(f'camera_1 ({len(camera_1)}): {Counter([y for f, feat, y in camera_1])}')
    print(f'camera_2 ({len(camera_2)}): {Counter([y for f, feat, y in camera_2])}')
    print(f'camera_3 ({len(camera_3)}): {Counter([y for f, feat, y in camera_3])}')

    train1, test1 = train_test_split(camera_1, test_size=test_size, random_state=random_state)

    # add '-mirror' into train1 and test1
    # camera_1
    new_train1 = []
    for (f, feat, y) in train1:
        f = f.replace('_1.mp4', '_1-mirrored.mp4')
        feat = feat.replace('_1_vgg.npy', '_1-mirrored_vgg.npy')
        if not os.path.exists(f) or not os.path.exists(feat):
            print(f'train1: {f} does not exist')
            continue
        new_train1.append((f, feat, y))
    new_test1 = []
    for (f, feat, y) in test1:
        f = f.replace('_1.mp4', '_1-mirrored.mp4')
        feat = feat.replace('_1_vgg.npy', '_1-mirrored_vgg.npy')
        if not os.path.exists(f) or not os.path.exists(feat):
            print(f'test1: {f} does not exist')
            continue
        new_test1.append((f, feat, y))
    train1.extend(new_train1)
    test1.extend(new_test1)

    # camera_2
    train2 = []
    for (f, feat, y) in train1:
        if '-mirrored.mp4' not in f:
            f = f.replace('_1.mp4', '_2.mkv')
            feat = feat.replace('_1_vgg.npy', '_2_vgg.npy')
        else:
            f = f.replace('_1-mirrored.mp4', '_2-mirrored.mkv')
            feat = feat.replace('_1-mirrored_vgg.npy', '_2-mirrored_vgg.npy')
        if not os.path.exists(f) or not os.path.exists(feat):
            print(f'train2: {f} does not exist')
            continue
        train2.append((f, feat, y))
    test2 = []
    for (f, feat, y) in test1:
        if '-mirrored.mp4' not in f:
            f = f.replace('_1.mp4', '_2.mkv')
            feat = feat.replace('_1_vgg.npy', '_2_vgg.npy')
        else:
            f = f.replace('_1-mirrored.mp4', '_2-mirrored.mkv')
            feat = feat.replace('_1-mirrored_vgg.npy', '_2-mirrored_vgg.npy')
        if not os.path.exists(f) or not os.path.exists(feat):
            print(f'test2: {f} does not exist')
            continue
        test2.append((f, feat, y))

    # camera_3
    train3 = []
    for (f, feat, y) in train1:
        if '-mirrored.mp4' not in f:
            f = f.replace('_1.mp4', '_3.mp4')
            feat = feat.replace('_1_vgg.npy', '_3_vgg.npy')
        else:
            f = f.replace('_1-mirrored.mp4', '_3-mirrored.mp4')
            feat = feat.replace('_1-mirrored_vgg.npy', '_3-mirrored_vgg.npy')
        if not os.path.exists(f) or not os.path.exists(feat):
            # print(f'{f} or {feat} does not exist')
            continue
        train3.append((f, feat, y))
    test3 = []
    for (f, feat, y) in test1:
        if '-mirrored.mp4' not in f:
            f = f.replace('_1.mp4', '_3.mp4')
            feat = feat.replace('_1_vgg.npy', '_3_vgg.npy')
        else:
            f = f.replace('_1-mirrored.mp4', '_3-mirrored.mp4')
            feat = feat.replace('_1-mirrored_vgg.npy', '_3-mirrored_vgg.npy')
        if not os.path.exists(f) or not os.path.exists(feat):
            # print(f'{f} or {feat} does not exist')
            continue
        test3.append((f, feat, y))

    # # camera_2
    # train2 = []
    # for (f, feat, y) in train1:
    #     f = f.replace('_1.mp4', '_2.mkv')
    #     feat = feat.replace('_1_vgg.npy', '_2_vgg.npy')
    #     if not os.path.exists(f) or not os.path.exists(feat): continue
    #     train2.append((f, feat, y))
    # test2 = []
    # for (f, feat, y) in test1:
    #     f = f.replace('_1.mp4', '_2.mkv')
    #     feat = feat.replace('_1_vgg.npy', '_2_vgg.npy')
    #     if not os.path.exists(f) or not os.path.exists(feat): continue
    #     test2.append((f, feat, y))
    #
    # # camera_3
    # train3 = []
    # for (f, feat, y) in train1:
    #     f = f.replace('_1.mp4', '_2.mkv')
    #     feat = feat.replace('_1_vgg.npy', '_2_vgg.npy')
    #     if not os.path.exists(f) or not os.path.exists(feat): continue
    #     train3.append((f, feat, y))
    # test3 = []
    # for (f, feat, y) in test1:
    #     f = f.replace('_1.mp4', '_3.mp4')
    #     feat = feat.replace('_1_vgg.npy', '_3_vgg.npy')
    #     if not os.path.exists(f) or not os.path.exists(feat): continue
    #     test3.append((f, feat, y))

    train_meta = {'camera_1': train1, 'camera_2': train2, 'camera_3': train3}
    test_meta = {'camera_1': test1, 'camera_2': test2, 'camera_3': test3}

    return train_meta, test_meta


def get_activity_info(in_file, log):
    start_time, end_time = 0, 0

    return start_time, end_time


def parse_logs(in_dir='data/data-clean/log'):
    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in os.listdir(in_dir).sort()])
    # # export to csv
    # combined_csv.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')

    video_logs = {}  # f: v
    for line in combined_csv.values:
        video_logs[line] = ''
    return video_logs


def change_label2idx(train_meta, label2idx={}):
    """ change label to index

    Parameters
    ----------
    train_meta
    label2idx

    Returns
    -------

    """
    for name, train in train_meta.items():
        for i, vs in enumerate(train):
            train_meta[name][i] = (vs[0], vs[1], vs[2], label2idx[vs[2]])  # (video_path, feature, y_label, y_idx)

    return train_meta


def cnn_feature2final_feature(train_meta, feature_type='mean', is_test=False):
    """

    Parameters
    ----------
    train_meta
    feature_type

    Returns
    -------

    """
    for name, train in train_meta.items():
        for i, vs in enumerate(train):
            f = vs[1]  # (video_path, feature, y_label, y_idx )
            if not os.path.exists(f):
                x = ''
                train_meta[name][i] = (vs[0], vs[1], vs[2], vs[3], x)  # (video_path, feature, y_label, y_idx, X)
                continue
            if not is_test:
                if feature_type == 'mean':
                    x = extract_feature_average(f)
                    # x = extract_feature_sampling(f)
                elif feature_type == 'sampling':
                    x = extract_feature_sampling(f)
            elif is_test:
                if feature_type == 'mean':
                    x = extract_feature_average(f)
            train_meta[name][i] = (vs[0], vs[1], vs[2], vs[3], x)  # (video_path, feature, y_label, y_idx, X)

    return train_meta


def gen_Xy(in_dir, out_dir, is_subclip=False, is_mirror=True, is_cnn_feature=True, device_type='refrigerator'):
    """Mirror videos and extract features by CNN

    Parameters
    ----------
    in_dir:  ['data/data-clean/refrigerator]
    out_dir:
    is_subclip: cut video
    is_mirror
    is_cnn_feature

    Returns
    -------
        meta: dictionary
    """
    if is_cnn_feature:
        # deep neural network model
        model_file = './features/video/slim/vgg_16.ckpt'
        model = CNN_tf('vgg', model_file)
    else:
        model = None

    data = []  # [(video_path, cnn_feature, y)]

    # list device folders (e.g., refrigerator or camera)
    i = 0
    cnt_3 = 0  # camera_3
    cnt_32 = 0  # camera_32: backup
    for device_dir in sorted(in_dir):
        out_dir_sub = ''
        if device_type not in device_dir: continue
        # list activity folders (e.g., open_close or take_out )
        for activity_dir in sorted(os.listdir(device_dir)):
            activity_label = activity_dir
            out_dir_activity = activity_dir
            activity_dir = os.path.join(device_dir, activity_dir)
            if not os.path.exists(activity_dir) or '.DS_Store' in activity_dir or not os.path.isdir(
                    activity_dir): continue
            # list participant folders (e.g., participant 1 or participant 2)
            for participant_dir in sorted(os.listdir(activity_dir)):
                out_dir_participant = participant_dir
                out_dir_sub = os.path.join(participant_dir)
                participant_dir = os.path.join(activity_dir, participant_dir)
                if not os.path.exists(participant_dir) or '.DS_Store' in participant_dir: continue
                # print(participant_dir)
                # list videos (e.g., 'no_interaction_1_1614038765_1.mp4')
                for f in sorted(os.listdir(participant_dir)):
                    if ('mp4' not in f) and ('mkv' not in f): continue  # only process video file.
                    x = os.path.join(participant_dir, f)
                    if '_3.mp4' in f: cnt_3 += 1
                    if '_3 2.mp4' in f: cnt_32 += 1
                    print(f'i: {i}, {x}')
                    try:
                        out_dir_tmp = os.path.join(out_dir, out_dir_activity, out_dir_participant)

                        if is_cnn_feature:
                            x_feat = _extract_video_feature(model, x, out_dir=out_dir_tmp)
                        else:
                            x_feat = ''
                        data.append((os.path.join(out_dir_tmp, f), x_feat, activity_label))
                        if is_mirror:
                            mirrored_x = _mirror_video(x, out_dir=out_dir_tmp)
                            if is_cnn_feature:
                                mirrored_x_feat = _extract_video_feature(model, mirrored_x, out_dir=out_dir_tmp)
                            else:
                                mirrored_x_feat = ''
                            data.append((mirrored_x, mirrored_x_feat, activity_label))
                    except Exception as e:
                        msg = f'error: {e} on {x}'
                        raise ValueError(msg)
                    i += 1
    print(f'camera_3: {cnt_3}, camera_32 (backup): {cnt_32}')
    meta = {'data': data, 'is_mirror': is_mirror, 'is_cnn_feature': is_cnn_feature}
    return meta


def main(random_state=42):
    ###############################################################################################################
    # Step 1. check if the Xy.dat exists. Xy.dat includes the train and test set.
    in_dir = 'out/data/data-clean/refrigerator'
    Xy_train_test_file = f'{in_dir}/Xy_train_test.dat'
    if os.path.exists(Xy_train_test_file):
        os.remove(Xy_train_test_file)

    if not os.path.exists(Xy_train_test_file):
        ###############################################################################################################
        # Step 2. Get all cnn_features from videos
        Xy_cnn_features_file = f'{in_dir}/Xy_cnn_features.dat'
        # if os.path.exists(Xy_cnn_features_file): os.remove(Xy_cnn_features_file)
        if not os.path.exists(Xy_cnn_features_file):
            in_raw_dir = 'data/data-clean/refrigerator'
            # mirror the data and extract the features by CNN
            meta = gen_Xy([in_raw_dir], out_dir=in_dir, is_mirror=True, is_cnn_feature=True)
            dump_data(meta, out_file=Xy_cnn_features_file)

        ###############################################################################################################
        # Step 3. Split the features to train and test set according to camera 1 (i.e., front view-'_1.mp4') .
        if not os.path.exists(Xy_train_test_file):
            meta = load_data(Xy_cnn_features_file)
            # video_type = {'_1.mp4': front view,'_3.mp4': side view and mirrored (up to down) view, 'mkv': side view}
            video_type = '_1.mp4'  # split the train and test based on 'camera_1' (i.e, '_1.mp4')
            test_size = 0.3
            label2idx = {'no_interaction': 0, 'open_close_fridge': 1, 'put_back_item': 2, 'screen_interaction': 3,
                         'take_out_item': 4}
            idx2label = {0: 'no_interaction', 1: 'open_close_fridge', 2: 'put_back_item', 3: 'screen_interaction',
                         4: 'take_out_item'}
            ###############################################################################################################
            # Step 3.1. Split the videos capured by camera 1 (i.e., front view-'_1.mp4') to train and test set.
            train_meta, test_meta = split_train_test_video(meta, video_type, test_size=test_size,
                                                           random_state=random_state)

            ###############################################################################################################
            # Step 3.2. change label to idx
            train_meta = change_label2idx(train_meta, label2idx)
            test_meta = change_label2idx(test_meta, label2idx)

            ###############################################################################################################
            # Step 3.3. dump all data to disk
            meta = {'train_meta': train_meta, 'test_meta': test_meta, 'test_size': test_size,
                    'label2idx': label2idx, 'idx2label': idx2label}
            dump_data(meta, out_file=Xy_train_test_file)

    ###############################################################################################################
    # Step 4. load Xy_train_test.dat
    meta = load_data(Xy_train_test_file)

    train_meta = meta['train_meta']
    test_meta = meta['test_meta']
    test_size = meta['test_size']

    ###############################################################################################################
    # Step 5. obtain final feature data (X_train and X_test) from CNN features with different methods
    train_meta = cnn_feature2final_feature(train_meta, feature_type='mean', is_test=False)
    test_meta = cnn_feature2final_feature(test_meta, feature_type='mean', is_test=True)

    ###############################################################################################################
    # Step 6. if augment data or not
    # X_train_meta, X_train, y_train = augment_train(train_meta, augment_type='camera_1',
    #                                                is_mirror=False)
    X_train_meta, X_train, y_train = augment_train(train_meta, augment_type='camera_1+camera_2+camera_3',
                                                   is_mirror=False)
    X_test_meta, X_test, y_test = augment_test(test_meta, augment_type='camera_1', is_mirror=False)

    print(f'X_train: {X_train.shape}\nX_test: {X_test.shape}')
    print(f'X_train: {X_train.shape}, y_train: {sorted(Counter(y_train).items(), key=lambda x: x[0])}')
    print(f'X_test: {X_test.shape}, y_test: {sorted(Counter(y_test).items(), key=lambda x: x[0])}')

    # print(X_train[:10])
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    # X = np.concatenate(X, axis=0)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    is_plot = True
    if is_plot:
        tsne_plot(X_train, y_train, [idx2label[i] for i in y_train])

    res = []
    # for n_estimators in [10, 50, 100, 200, 300, 400, 500, 700, 900, 1000]:
    for model_name in ['OvRLogReg', 'SVM(linear)', 'RF']:
        print(f'\n\n***{model_name}')
        if model_name == 'OvRLogReg':
            detector = AnomalyDetector(model_name='OvRLogReg', model_parameters={'C': 1}, random_state=random_state)
        elif model_name == 'SVM(linear)':
            # detector = AnomalyDetector(model_name='SVM', model_parameters={'kernel':'rbf'}, random_state=random_state)
            detector = AnomalyDetector(model_name='SVM', model_parameters={'kernel': 'linear'},
                                       random_state=random_state)
        elif model_name == 'RF':
            detector = AnomalyDetector(model_name='RF', model_parameters={'n_estimators': 100},
                                       random_state=random_state)
        else:
            # 2. build the kde models
            # detector = AnomalyDetector(model_name='KDE', model_parameters = {'bandwidth': 0.1, 'kernel': 'gussisan'})
            detector = AnomalyDetector(model_name='DT', model_parameters={}, random_state=random_state)

        detector.fit(X_train, y_train)
        #
        # # 3. compute the threshold
        # detector.get_threshold(X_train, q=0.01)
        # # print(detector.predict_prob(X_train))

        # 4. evaulation
        y_preds = detector.model.predict(X_test)

        print('y_test (label, cnt): ', sorted(Counter(y_test).items(), key=lambda x: x[0]))
        acc = sklearn.metrics.accuracy_score(y_test, y_preds)
        print(f'accuracy: {acc}')
        # res.append((acc, n_estimators))
        cm = sklearn.metrics.confusion_matrix(y_test, y_preds)
        # print(cm)
        # labels = list(mp.keys())
        w = 15  # width
        # print()
        # labels = [f'{v[:w]:<{w}}' for k, v in mp.items()]
        # for v in zip_longest(*labels, fillvalue=' '):
        #     print(' '.join(v))
        # print(' '* 15 + ' '.join(labels)+f'(predicted)')

        print(' ' * 40 + '(predicted)')
        # print(' ' * (w + 1) + '\t' + '\t\t'.join([f'({k})' for k, v in mp.items()]))
        for i, vs in enumerate(list(cm)):
            print(f'{idx2label[i][:w]:<{w}} ({i})\t', '\t\t'.join([f'{v}' for v in list(vs)]))

        # 5 get misclassification
        err_mp = {}
        misclassified_dir = 'out/misclassified'
        if os.path.exists(misclassified_dir):
            shutil.rmtree(misclassified_dir)
        for x_test_, y_t, y_p in zip(X_test_meta, y_test, y_preds):
            if y_t != y_p:
                name = f'{idx2label[y_t]}({y_t}):{x_test_}'
                if name not in err_mp.keys():
                    err_mp[name] = [f'{idx2label[y_p]}({y_p})']
                else:
                    err_mp[name].append(f'{idx2label[y_p]}({y_p})')
                # copy misclassified videos to dst
                # print(x_test_)
                tmp_out_dir = os.path.join(misclassified_dir, model_name, idx2label[y_t] + '->')
                if not os.path.exists(tmp_out_dir):
                    os.makedirs(tmp_out_dir)
                copyfile(x_test_, os.path.join(tmp_out_dir, os.path.split(x_test_)[-1]))
                # print(f'{mp[y_t]} -> {mp[y_p]}')
        print(f'***misclassified classes: {len(err_mp.keys())}')
        # print('\t' + '\n\t'.join([f'{k}->{Counter(vs)}' for k, vs in sorted(err_mp.items())]))
        # for label_ in y_test:
        #     label_ = idx2label[label_]
        for k, vs in sorted(err_mp.items()):
            print('\t' + '\n\t'.join([f'{k}->{vs}']))
    print('\n\n', res)
    print(sorted(res, key=lambda x: x[0], reverse=True))


if __name__ == '__main__':
    main()
