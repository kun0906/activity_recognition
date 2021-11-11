""" Install odet first:
    https://github.com/kun0906/odet

"""

import os
import pickle

import pandas as pd

from odet.pparser.parser import _pcap2flows, _get_IAT_SIZE, _get_STATS
from odet.utils.tool import dump_data, check_path

import numpy as np

""" Analyze IOT datasets (data-clean.zip: 20GB, 20210714) collected on 2021.

"""
import collections
import os
import subprocess
from odet.pparser.parser import PCAP
import numpy as np

RANDOM_STATE = 42


def check_path(in_dir):
    if not os.path.exists(in_dir):
        os.makedirs(in_dir)


def dump_data(data, out_file=''):
    """Save data to file

    Parameters
    ----------
    data: any data

    out_file: str
        out file path
    verbose: int (default is 1)
        a print level is to control what information should be printed according to the given value.
        The higher the value is, the more info is printed.

    Returns
    -------

    """
    # save results
    with open(out_file, 'wb') as out_hdl:
        pickle.dump(data, out_hdl)


class IOT2021(PCAP):
    def get_flows(self, in_file='xxx.pcap'):
        # flows: [(fid, arrival times list, packet sizes list)]
        self.flows = _pcap2flows(in_file, flow_pkts_thres=2)

    def keep_ip(self, pcap_file, kept_ips=[], output_file=''):

        if output_file == '':
            output_file = os.path.splitext(pcap_file)[0] + 'kept_ips.pcap'  # Split a path in root and extension.
        # only keep srcIPs' traffic
        # srcIP_str = " or ".join([f'ip.src=={srcIP}' for srcIP in kept_ips])
        # filter by mac srcIP address
        srcIP_str = " or ".join([f'eth.src=={srcIP}' for srcIP in kept_ips])
        cmd = f"tshark -r {pcap_file} -w {output_file} {srcIP_str}"

        print(f'{cmd}')
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
        except Exception as e:
            print(f'{e}, {result}')
            return -1

        return output_file


def get_pcaps(in_dir, file_type='normal'):
    files = collections.defaultdict(list)
    for activity_name in sorted(os.listdir(in_dir)):
        if activity_name.startswith('.'): continue
        activity_dir = os.path.join(in_dir, activity_name)
        for partcipant_id in sorted(os.listdir(activity_dir)):
            if partcipant_id.startswith('.'): continue
            partcipant_dir = os.path.join(activity_dir, partcipant_id)
            for f in sorted(os.listdir(partcipant_dir)):
                if f.startswith('.'): continue
                if f.endswith('pcap'):
                    f = os.path.join(partcipant_dir, f)
                    files[activity_name].append(f)
                    # files.append(f)
                else:
                    pass
    return files


def get_mac_ip(flows):
    ips = []
    macs = []
    # [(fid, arrival times list, packet sizes list)]
    for i, (fid, pkt_times, pkts) in enumerate(flows):
        macs.append(pkts[0].src)
        ips.append(fid[0])
    print(set(ips))
    return macs, ips


def get_durations(flows):
    durations = []
    # [(fid, arrival times list, packet sizes list)]
    for i, (fid, pkt_times, pkts) in enumerate(flows):
        start = min(pkt_times)
        end = max(pkt_times)
        durations.append(end - start)

    return durations


# IP has changed (dynamic IPs) in different collection process, so please use mac to filter packets.
ip2device = {'192.168.143.152': 'refrigerator', }
device2ip = {'refrigerator': '192.168.143.43', 'nestcam': '192.168.143.104', 'alexa': '192.168.143.74'}
device2mac = {'refrigerator': '70:2c:1f:39:25:6e', 'nestcam': '18:b4:30:8a:9f:b2', 'alexa': '4c:ef:c0:0b:91:b3'}


#
# def main(device='refrigerator'):
#     in_dir = f'../Datasets/UCHI/IOT_2021/data-clean/{device}'
#     out_dir = f'examples/datasets/IOT2021/data-clean/{device}'
#     device_meta_file = os.path.join(out_dir, f'{device}.dat')
#     device_meta = {}
#     if not os.path.exists(device_meta_file):
#         device_files = get_pcaps(in_dir, file_type='normal')
#         for i, (activity_name, files) in enumerate(device_files.items()):
#             activity_flows = []
#             for j, f in enumerate(files):
#                 print(j, f)
#                 # create the PCAP object
#                 pp = IOT2021()
#
#                 # filter unnecesarry IP addresses
#                 filtered_f = os.path.join(out_dir,
#                                           os.path.splitext(os.path.relpath(f, start=in_dir))[0] + '-filtered.pcap')
#                 check_path(filtered_f)
#                 # pp.keep_ip(f, kept_ips=[device2ip[device]], output_file=filtered_f)
#                 pp.keep_ip(f, kept_ips=[device2mac[device]], output_file=filtered_f)
#
#                 # parse pcap and get the flows (only forward flows (sent from src IP))
#                 pp.get_flows(filtered_f)
#
#                 # concatenated the flows to the total flows
#                 device_files[activity_name][j] = (filtered_f, pp.flows)
#                 activity_flows += pp.flows
#                 # break
#
#             # activity_flows = sum(len(flows_) for flows_ in ])
#             print(f'activity_flows: {len(activity_flows)}')
#             device_meta[activity_name] = (activity_flows, device_files[activity_name])
#         check_path(device_meta_file)
#         print(device_meta_file)
#         dump_data(device_meta, output_file=device_meta_file)
#     else:
#         device_meta = load_data(device_meta_file)
#
#     ips = set()
#     macs = set()
#     for i, (activity_name, vs_) in enumerate(device_meta.items()):
#         activity_flows, file_flows = vs_
#         print(i, activity_name, len(activity_flows))
#         macs_, ips_ = get_mac_ip(activity_flows)
#         # print strange IP and pcap_file
#         for v_, (f_, _) in zip(ips_, file_flows):
#             if v_ == '0.0.0.0':
#                 print(activity_name, v_, f_)
#         macs.update(macs_)
#         ips.update(ips_)
#
#     print(f'MAC: {macs}, IP: {ips}')
#     # get normal_durations
#     normal_flows = device_meta['no_interaction'][0]
#     normal_durations = get_durations(normal_flows)
#
#     # get subflow_interval
#     q_flow_dur = 0.9
#     subflow_interval = np.quantile(normal_durations, q=q_flow_dur)  # median  of flow_durations
#     print(f'---subflow_interval: ', subflow_interval, f', q_flow_dur: {q_flow_dur}')
#
#     subflow_device_meta = {'q_flow_dur': q_flow_dur, 'subflow_interval': subflow_interval,
#                            'normal_durations': normal_durations}
#     for i, (activity_name, vs_) in enumerate(device_meta.items()):
#         activity_flows, file_flows = vs_
#         subflows = []
#         for file_, flows_ in file_flows:
#             subflows_ = flow2subflows(flows_, interval=subflow_interval, num_pkt_thresh=2, verbose=False)
#             subflows += subflows_
#         print(i, activity_name, len(activity_flows), len(subflows))
#         subflow_device_meta[activity_name] = subflows[:]
#
#     print('\n')
#     # print subflow results
#     for i, (key, vs_) in enumerate(sorted(subflow_device_meta.items())):
#         if type(vs_) == list:
#             print(i, key, len(vs_))
#         else:
#             print(i, key, vs_)


def _extract_pcap_feature(pcap_file, out_dir, feat_type='IAT+SIZE', device='refrigerator'):
    # # filter ip by macaddress
    # filtered_pcap_file = os.path.join(out_dir, os.path.basename(pcap_file))
    # keep_ip(pcap_file, kept_ips= [device2mac[device]], output_file= filtered_pcap_file)

    # create the PCAP object
    pp = IOT2021()

    # filter unnecesarry IP addresses
    filtered_f = os.path.join(out_dir, os.path.basename(pcap_file))
    check_path(os.path.dirname(filtered_f))
    # pp.keep_ip(f, kept_ips=[device2ip[device]], output_file=filtered_f)
    pp.keep_ip(pcap_file, kept_ips=[device2mac[device]], output_file=filtered_f)

    # parse pcap and get the flows (only forward flows (sent from src IP))
    pp.get_flows(filtered_f)
    pp.flows = [(fid, pkts) for fid, pkts in pp.flows if '0.0.0.0' not in fid[0] and '0.0.0.0' not in fid[1]]
    check_path(out_dir)
    out_file = os.path.join(out_dir, os.path.basename(pcap_file) + f'-flows.dat')
    dump_data(pp.flows, out_file)

    # get features
    if feat_type == 'IAT+SIZE':
        features, fids = _get_IAT_SIZE(pp.flows)
    elif feat_type == 'STATS':
        features, fids = _get_STATS(pp.flows)
    else:
        msg = f'{feat_type}'
        raise NotImplementedError(msg)
    feature_file = os.path.join(out_dir, os.path.basename(pcap_file) + f'-{feat_type}.dat')
    dump_data((features, fids), feature_file)
    return out_file, feature_file, 0


def pcap2feature(in_dir, out_dir, is_subclip=True, is_mirror=False, is_cnn_feature=False, feat_type='IAT+SIZE',
                 device_type='refrigerator'):
    """ preprocessing the videos:
            e.g., trim and mirror videos,  extract features by CNN

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

    # video_logs = parse_logs(in_dir='data/data-clean/log')

    # issued_videos = pd.read_csv(os.path.join('data/data-clean/refrigerator', 'issued_videos.csv'), header=None).values[
    #                 :, -1].tolist()
    issued_videos = []
    data = []  # [(video_path, cnn_feature, y)]

    durations = {'camera1': [], 'camera2': [], 'camera3': []}
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
                    print(f)
                    if f.startswith('.'): continue
                    if not f.endswith('.pcap'): continue
                    issued_flg = False
                    for _issued_f in issued_videos:
                        if f in _issued_f + '.npy':
                            issued_flg = True
                            break
                    if issued_flg:
                        continue  # issued videos, skip
                    x = os.path.join(participant_dir, f)
                    try:
                        # vd_info = get_info(x)
                        out_dir_tmp = os.path.join(out_dir, out_dir_activity, out_dir_participant)
                        x_flows, x_feat, kept_durations = _extract_pcap_feature(x, out_dir=out_dir_tmp,
                                                                                feat_type=feat_type)

                        data.append((x, x_feat, activity_label))

                    except Exception as e:
                        msg = f'error: {e} on {x}'
                        raise ValueError(msg)
                    i += 1
    print(f'tot pcaps: {i}')
    meta = {'data': data, 'is_mirror': is_mirror, 'is_cnn_feature': is_cnn_feature}
    return meta


if __name__ == '__main__':
    pcap2feature(in_dir=['data/data-clean/refrigerator'], out_dir='out/data/data-clean/refrigerator',
                 feat_type='IAT+SIZE', device_type='refrigerator')
