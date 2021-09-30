import numpy as np


def npz(in_file):
    data = np.load(in_file, allow_pickle=True)
    print(data.files)


if __name__ == '__main__':
    in_file = 'videopose3d/inference/out/ask_time_1_1614904536_1.mp4.mp4.mp4.npz'
    in_file = 'videopose3d/inference/data/data_2d_custom_2d_keypoints.npz'  # 2d keypoints
    npz(in_file)
