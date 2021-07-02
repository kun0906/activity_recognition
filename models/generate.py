import os
import pickle
from collections import OrderedDict

import numpy as np


def write2disk(file_lst, out_file):
    with open(out_file, 'w') as f:
        s = '\n'.join(file_lst)
        f.write(s)


def extract_video_feature(in_dir='data/data-clean'):
    mp4_lst = []  # camera 1
    mkv_lst = []  # camera 2
    for device_dir in os.listdir(in_dir):  # devices
        pth = os.path.join(in_dir, device_dir)
        if not os.path.isdir(pth): continue
        for activity_dir in os.listdir(pth):  # activity:
            pth1 = os.path.join(pth, activity_dir)
            if not os.path.isdir(pth1): continue
            for part_id_dir in os.listdir(pth1):  # participate_id
                pth2 = os.path.join(pth1, part_id_dir)
                if not os.path.isdir(pth2): continue
                for f in os.listdir(pth2):
                    if 'no_interaction' in f: continue
                    if f.endswith('.mp4'):
                        mp4_lst.append(os.path.join(pth2, f))
                    elif f.endswith('.mkv'):
                        mkv_lst.append(os.path.join(pth2, f))

    mp4_file = 'out/camera1_mp4.txt'
    write2disk(mp4_lst, mp4_file)
    mkv_file = 'out/camera2_mkv.txt'
    write2disk(mkv_lst, mkv_file)

    out_dir = 'out'
    for camera, video_file in [('camera1_mp4', mp4_file), ('camera2_mkv', mkv_file)]:
        _out_dir = os.path.join(out_dir, camera)
        if not os.path.exists(_out_dir):
            os.makedirs(_out_dir)
        cmd = f'python3.7 feature_extraction.py --video_list {video_file}  --network vgg --framework tensorflow ' \
              f'--output_path {_out_dir} --tf_model ./slim/vgg_16.ckpt'
        print(cmd)
        os.system(cmd)




def extract_feature(file_path):
    x = np.load(file_path)
    return [x.flatten()]



def extract_feature3(file_path):
    x = np.load(file_path)
    x = np.sum(x, axis=0) / len(x)
    return [x.flatten()]


def extract_feature2(file_path):
    x = np.load(file_path)

    # sliding window
    w = 10
    stride = 2
    res = []
    for i in range(0, len(x), stride):
        if i > w:
            # tmp -= x[i-w]
            # tmp += x[i]
            _x = x[i-w:i]
            tmp = np.sum(_x, axis=0) / len(_x)
            res.append(tmp)
        else:
            _x = x[:w]
            tmp = np.sum(x[:w], axis=0) / len(_x)
            res.append(tmp)

    return res



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

def form_X_y(in_dir):
    mp = OrderedDict()
    i = 0
    c = 0
    X = []
    Y = []
    for _in_dir in in_dir:
        for f in sorted(os.listdir(_in_dir)):
            xs = extract_feature(os.path.join(_in_dir, f))
            y = extract_label(f)
            if y == 'no_interaction': continue
            if y not in mp.keys():
                mp[y] = (i, 1)  # (idx, cnt)
                i += 1
            else:
                mp[y] = (mp[y][0], mp[y][1] + 1)
            X.extend(xs)
            Y.extend([mp[y][0]]*len(xs))
            c += 1
    print(f'{in_dir}: total videos: {c}, and classes: {i}')
    print(f'{in_dir}: Labels: {mp.items()}')
    meta = {'X': np.asarray(X), 'y': np.asarray(Y), 'shape': (c, len(X[0])), 'labels': mp, 'in_dir': in_dir}
    return meta


def dump_data(data, out_file):
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)


def load_data(in_file):
    with open(in_file, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    # # in_dir = 'out/output_mkv'
    # in_dir = ['out/output_mp4']
    # meta = form_X_y(in_dir)
    # out_file = 'out/Xy-mp4.dat'
    # dump_data(meta, out_file)
    extract_video_feature(in_dir='data/data-clean')
