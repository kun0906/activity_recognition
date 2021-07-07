import os
from logging import warning

import numpy as np

from features.video.utils import trim
import pandas as pd


def demo():
    in_dir = ''
    out_dir = ''
    video_list_file = os.path.join(in_dir, 'video_list.txt')
    video_list = []
    durations = []
    # get info from video_list_file
    with open(video_list_file, 'r') as f:
        line = f.readline().split(',')
        while line != ['']:
            video_path = os.path.join(in_dir, line[0])
            # print(line, video_path)
            start_time = int(line[1])
            end_time = int(line[2])
            out_file = os.path.join(out_dir, video_path)
            video_list.append([video_path, start_time, end_time, out_file])
            durations.append(end_time - start_time)
            line = f.readline().split(',')

    # get fixed duration
    fixed_duration = int(np.quantile(durations, q=0.9))
    print(f'fixed_duration: {fixed_duration}')

    # obtain the fixed size of all videos
    for vs in video_list:
        video_path = vs[0]
        start_time = vs[1]
        end_time = start_time + fixed_duration
        out_file = vs[3]
        trim(video_path, start_time, end_time, out_file)


#

def main():
    in_dir = 'data/data-clean/'
    out_dir = 'data/trimmed/data-clean'
    xlsx_file = f'{in_dir}/refrigerator/description.xlsx'
    xls = pd.ExcelFile(xlsx_file)
    # to read all sheets to a map
    df_mp = {}
    video_list = []
    durations = []
    for sheet_name in xls.sheet_names:
        df_mp[sheet_name] = xls.parse(sheet_name)
        for line in df_mp[sheet_name].values.tolist():
            try:
                if not line or str(line[0]) == 'nan' or line[0].startswith('full'): continue
                video_path = os.path.join(in_dir, line[0])
                out_file = os.path.join(out_dir, line[0])
                print(line, video_path)
                start_time = int(line[1])
                end_time = int(line[2])
                video_list.append([video_path, start_time, end_time, out_file])
                durations.append(end_time - start_time)
            except Exception as e:
                warning(f"{line}, {e}")

    # total videos
    tot = len(durations)
    print(f'total videos: {tot}')
    # get fixed duration
    fixed_duration = int(np.quantile(durations, q=0.9))
    print(f'fixed_duration: {fixed_duration}')

    # obtain the fixed size of all videos
    c = 0
    for vs in video_list:
        try:
            video_path = vs[0]
            start_time = vs[1]
            end_time = start_time + fixed_duration
            out_file = vs[3]
            _out_dir = os.path.dirname(out_file)
            if not os.path.exists(_out_dir):
                os.makedirs(_out_dir)
            trim(video_path, start_time, end_time, out_file)
            c += 1
        except Exception as e:
            warning(f"{out_file}, {e}")
            # print(Fore.RED + f"{out_file}, {e}")

    print(f'trimmed {c}/{tot} videos successfully!')
    #


if __name__ == '__main__':
    main()
