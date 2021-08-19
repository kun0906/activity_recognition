import os
import pickle
from collections import OrderedDict

import numpy as np
from future.utils import lrange

from features.video.model_tf import CNN_tf
from features.video.utils import load_video


def write2disk(file_lst, out_file):
    with open(out_file, 'w') as f:
        s = '\n'.join(file_lst)
        f.write(s)


def feature_extraction_videos(model, batch_sz, video_file, output_path):
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    # sampling: only extract the first frame in each second from the video.
    video_tensor = load_video(video_file, model.desired_size)
    # extract features
    features = model.extract(video_tensor, batch_sz)
    path = os.path.join(output_path, '{}_{}'.format(video_name, model.net_name))
    # save features
    np.save(path, features)
    print(path)


def extract_video_feature(in_dir='data/data-clean', video_type='mkv'):
    mp4_lst = []  # camera 1
    mkv_lst = []  # camera 2
    for device_dir in os.listdir(in_dir):  # devices
        pth = os.path.join(in_dir, device_dir)
        if not os.path.isdir(pth): continue
        if 'refrigerator' not in pth: continue  # only focus on refrigerator
        for activity_dir in os.listdir(pth):  # activity:
            pth1 = os.path.join(pth, activity_dir)
            if not os.path.isdir(pth1): continue
            for part_id_dir in os.listdir(pth1):  # participate_id
                pth2 = os.path.join(pth1, part_id_dir)
                if not os.path.isdir(pth2): continue
                for f in os.listdir(pth2):
                    # if 'no_interaction' in f: continue
                    if f.endswith('.mp4'):
                        mp4_lst.append(os.path.join(pth2, f))
                    elif f.endswith('.mkv'):
                        mkv_lst.append(os.path.join(pth2, f))

    out_dir = 'out'
    mp4_file = f'{out_dir}/{in_dir}/camera1_mp4.txt'
    if not os.path.exists(os.path.dirname(mp4_file)):
        os.makedirs((os.path.dirname(mp4_file)))
    write2disk(mp4_lst, mp4_file)
    mkv_file = f'{out_dir}/{in_dir}/camera2_mkv.txt'
    write2disk(mkv_lst, mkv_file)

    # # for camera, video_file in [('camera1_mp4', mp4_file), ('camera2_mkv', mkv_file)]:
    # for camera, video_file, video_lst in [('camera1_mp4', mp4_file, mp4_lst)]:
    #     _out_dir = os.path.basename(video_file)
    #     if not os.path.exists(_out_dir):
    #         os.makedirs(_out_dir)
    #     cmd = f'python3.7 features/video/feature_extraction.py --video_list {video_lst}  ' \
    #           f'--network vgg --framework tensorflow ' \
    #           f'--output_path {_out_dir} --tf_model ./features/video/slim/vgg_16.ckpt'
    #     print(cmd)
    #     os.system(cmd)

    # for camera, video_file in [('camera1_mp4', mp4_file), ('camera2_mkv', mkv_file)]:

    # deep neural network model
    model_file = './features/video/slim/vgg_16.ckpt'
    model = CNN_tf('vgg', model_file)
    batch_sz = 32

    if video_type == 'mp4':
        file_lst = mp4_lst
    elif video_type == 'mkv':
        file_lst = mkv_lst

    tot = len(file_lst)
    print('\nNumber of videos: ', tot)

    for i, video_file in enumerate(file_lst):
        print(f'{i + 1}/{tot}, {video_file}')
        _out_dir = os.path.join(out_dir, os.path.dirname(video_file))
        if not os.path.exists(_out_dir):
            os.makedirs(_out_dir)
        feature_extraction_videos(model, batch_sz, video_file, _out_dir)



def extract_feature(file_path):
    x = np.load(file_path)
    return [x.flatten()]


def extract_feature_average(file_path):
    x = np.load(file_path)
    if len(x) == 0:
        x = np.sum(x, axis=0)
    else:
        x = np.sum(x, axis=0) / len(x)
    return x.reshape(1, -1)


def extract_feature_sliding_window(file_path):
    x = np.load(file_path)

    # sliding window
    w = 10
    stride = 2
    res = []
    for i in range(0, len(x), stride):
        _x = x[i:i + w]
        tmp = np.sum(_x, axis=0) / len(_x)
        res.append(tmp)

    return res


def extract_feature_sampling_old(file_path):
    x = np.load(file_path)

    res = []
    for step in range(1, 10 + 1):
        tmp = []
        c = 0
        for i in range(0, len(x), step):
            tmp.append(x[i])
            c += 1
        if c == 0: continue
        tmp = np.sum(np.asarray(tmp), axis=0) / c
        res.append(tmp.flatten())

    return res


def extract_feature_sampling(file_path, steps=[1, 2, 3, 4, 5]):
    x = np.load(file_path)

    res = []
    for step in steps:
        tmp = []
        c = 0
        for i in range(0, len(x), step):
            tmp.extend(x[i].tolist())
            c += 1
        if c == 0: continue
        res.append(np.asarray(tmp))

    return res


def extract_feature_sampling_mean(file_path, window_size=5):
    x = np.load(file_path)
    res = []
    for i in range(0, len(x), window_size):
        if i == 0:
            res = np.mean(x[i:i + window_size], axis=0)
        else:
            res = np.concatenate([res, np.mean(x[i:i + window_size], axis=0)])

    return res.reshape(1, -1)


def extract_feature_fixed_segments(file_path, dim=5):
    x = np.load(file_path)
    if len(x) == 0:
        return []
    if len(x) == dim:
        res = x
    elif len(x) < dim:
        print(len(x), dim, file_path)
        res = np.concatenate([x, np.zeros(shape=(dim - len(x), len(x[0])))], axis=0)  # row first
    else:
        res = []
        window_size = len(x) // dim
        lst = [ws for ws in range(0, len(x), window_size)]
        for idx, i in enumerate(lst):
            if i == 0:
                res = np.mean(x[i:i + window_size], axis=0)
            else:
                if idx == dim - 1:
                    res = np.concatenate([res, np.mean(x[i:], axis=0)])
                    break
                else:
                    res = np.concatenate([res, np.mean(x[i:i + window_size], axis=0)])
    return res.reshape(1, -1)

def extract_feature_uChicago(file_path):
    arr = np.load(file_path)

    res = np.zeros((arr.shape[1],))
    for i in range(arr.shape[0]):
        res += arr[i] - arr[0]

    res = np.clip(res, -2, 2)  # lower and upper bound
    res = [res]
    return res


def extract_feature_avg_uChicago(file_path):
    x = np.load(file_path)
    x = np.sum(x, axis=0) / len(x)
    return [x.flatten()]


def extract_label(file_name):
    arr = file_name.split('_')
    label = ''
    for v in arr:
        if v.isdigit():
            break
        label += v + '_'

    return label[:-1]


# def form_x_y(in_dir=''):
#     # act_label = {"frg_no_interaction": 0,
#     #              "frg_open_close_fridge": 1,
#     #              "frg_put_back_item": 2,
#     #              "frg_screen_interaction": 3,
#     #              "frg_take_out_item": 4,
#     #              "alx_no_interaction": 5,
#     #              "alx_ask_weather": 6,
#     #              "alx_play_music": 7,
#     #              "nstc_no_interaction": 8,
#     #              "nstc_ask_time": 9,
#     #              "nstc_trigger_motion_detection": 10}
#     mp = OrderedDict()
#     # print(f'tot: {len(os.listdir(in_dir))}')
#     X = []
#     Y = []
#     i = 0
#     c = 0
#     for f in sorted(os.listdir(in_dir)):
#         x = extract_feature(os.path.join(in_dir, f))
#         y = extract_label(f)
#         if y == 'no_interaction': continue
#
#         if y not in mp.keys():
#             mp[y] = (i, 1)  # (idx, cnt)
#             i += 1
#         else:
#             mp[y] = (mp[y][0], mp[y][1] + 1)
#         X.append(x)
#         Y.append(mp[y][0])
#         c += 1
#     print(f'total videos: {c}, and classes: {i}')
#     print(f'Labels: {mp.items()}')
#     meta = {'X':np.asarray(X), 'y': np.asarray(Y), 'shape': (c, len(X[0])), 'labels':mp, 'in_dir': in_dir}
#     return meta

def form_X_y_old(in_dir):
    mp = OrderedDict()
    i = 0
    c = 0
    X = []
    Y = []
    for _in_dir in in_dir:
        for f in sorted(os.listdir(_in_dir)):
            xs = extract_feature(os.path.join(_in_dir, f))
            y = extract_label(f)
            # if y == 'no_interaction': continue
            if y not in mp.keys():
                mp[y] = (i, 1)  # (idx, cnt)
                i += 1
            else:
                mp[y] = (mp[y][0], mp[y][1] + 1)
            X.extend(xs)
            Y.extend([mp[y][0]] * len(xs))
            c += 1
    print(f'{in_dir}: total videos: {c}, and classes: {i}')
    print(f'{in_dir}: Labels: {mp.items()}')
    meta = {'X': np.asarray(X), 'y': np.asarray(Y), 'shape': (c, len(X[0])), 'labels': mp, 'in_dir': in_dir}
    return meta


def form_X_y(in_dir, device_type='refrigerator', video_type='mp4'):
    mp = OrderedDict()
    i = 0
    c = 0
    X = []
    Y = []
    for device_dir in in_dir:  # 'data/data-clean/refrigerator
        if device_type not in device_dir: continue
        for activity_dir in os.listdir(device_dir):
            activity_dir = os.path.join(device_dir, activity_dir)
            if not os.path.exists(activity_dir) or '.DS_Store' in activity_dir or not os.path.isdir(
                activity_dir): continue
            for participant_dir in os.listdir(activity_dir):
                participant_dir = os.path.join(activity_dir, participant_dir)
                if not os.path.exists(participant_dir) or '.DS_Store' in participant_dir: continue
                # print(participant_dir)
                for f in sorted(os.listdir(participant_dir)):
                    if '.npy' not in f: continue
                    if video_type == '_1.mp4' and '_1_vgg.npy' in f:  # ('1_vgg.npy' in f or '1-mirrored_vgg.npy' in f):
                        x = os.path.join(participant_dir, f)
                    elif video_type == '_3.mp4':
                        if '_3_vgg.npy' in f or '_3 2_vgg.npy' in f:  # side view and # up to down
                            x = os.path.join(participant_dir, f)
                        else:
                            # x = os.path.join(participant_dir, f)
                            # print(f'not exist: {x}')
                            continue
                    elif video_type == 'mkv' and '_2_vgg.npy' in f:  # ('2_vgg.npy' in f or '2-mirrored_vgg.npy' in f):
                        x = os.path.join(participant_dir, f)
                    else:
                        continue
                    y = extract_label(f)
                    # if y == 'no_interaction': continue
                    if y not in mp.keys():
                        mp[y] = (i, 1)  # (idx, cnt)
                        i += 1
                    else:
                        mp[y] = (mp[y][0], mp[y][1] + 1)
                    X.append(x)
                    Y.append(mp[y][0])
                    c += 1
    print(f'{in_dir}: total videos: {c}, and classes: {i}')
    print(f'{in_dir}: Labels: {mp.items()}')
    idx2label = {v[0]: k for k, v in mp.items()}  # idx -> activity name
    meta = {'X': X, 'y': Y, 'shape': (c,), 'labels': mp, 'idx2label': idx2label, 'in_dir': in_dir}
    return meta


def dump_data(data, out_file):
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)


def load_data(in_file):
    with open(in_file, 'rb') as f:
        data = pickle.load(f)
    return data


def generate_data(in_dir='out/output_mp4', out_file='out/Xy.dat', video_type='mp4'):
    if type(in_dir) == str:
        in_dir = [in_dir]
    elif type(in_dir) == list:
        pass
    else:
        raise NotImplementedError()

    meta = form_X_y(in_dir, video_type=video_type)
    dump_data(meta, out_file)
    return meta


#

if __name__ == '__main__':
    # # in_dir = 'out/output_mkv'
    # in_dir = ['out/output_mp4']
    # meta = form_X_y(in_dir)
    # out_file = 'out/Xy-mp4.dat'
    # dump_data(meta, out_file)
    # extract_video_feature(in_dir='data/data-clean')
    extract_video_feature(in_dir='data/trimmed/data-clean')
