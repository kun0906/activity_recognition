import os

import numpy as np

from features.video.utils import trim


def main():
    in_dir = 'data/data-clean'
    out_dir = 'out'
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


if __name__ == '__main__':
    main()
