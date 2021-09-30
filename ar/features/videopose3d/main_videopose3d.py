import os
from glob import glob

import numpy as np


def main():
    # step 1: use detectron2 to get 2d_keypoints
    """ Be careful with directory path
    # generate 2D keypoint predictions from videos
    cd inference
    # python3.7 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir out_directory --image-ext mp4 input_directory
    # python3.7 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir out2d --image-ext mp4 ../../data/demo
    python3.7 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir out2d --image-ext mp4 ../../data/data-clean/refrigerator
    python3.7 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir out2d --image-ext mkv ../../data/data-clean/refrigerator

    # create customized dataset "data/data_2d_custom_myvideos.npz"
    cd ../data
    #python3.7 prepare_data_2d_custom.py -i ../inference/out2d/data/data-clean/ -o myvideos
    python3.7 prepare_data_2d_custom.py -i ../inference/out2d/data/data-clean/ -o 2d_keypoints
    cd ..

    """
    # mp4
    # cmd = 'python3.7 inference/infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir inference/out2d --image-ext mp4 ../data/data-clean/refrigerator'
    # mkv
    cmd = 'python3.7 inference/infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir inference/out2d --image-ext mkv ../data/data-clean/refrigerator'
    print(f'\n\nstep1: {cmd}')
    os.system(cmd)

    # step 2: generate 2d_keypoints dataset by prepare_data_2d_custom.py
    # (Note that the output also includes 2d_keypoints of demo videos. Be careful of the input and output directory)
    cmd = 'python3.7 data/prepare_data_2d_custom.py -i inference/out2d/data/ -o data/2d_keypoints'
    print(f'\n\nstep2: {cmd}')
    os.system(cmd)

    keypoints_2d_file = 'data/data_2d_custom_2d_keypoints.npz'
    keypoints_2d = np.load(keypoints_2d_file, allow_pickle=True)
    print(len(keypoints_2d['positions_2d'].tolist()))
    im_list = list(keypoints_2d['positions_2d'].tolist().keys())
    camera1 = [f for f in im_list if '1.mp4' in f]
    camera2 = [f for f in im_list if '2.mkv' in f]
    camera3 = [f for f in im_list if '3.mp4' in f]
    camera32 = [f for f in im_list if '3 2.mp4' in f]
    print(f'total videos: {len(im_list)}, in which, camera1={len(camera1)}, camera2={len(camera2)}, '
          f'camera3={len(camera3)}, camera32={len(camera32)}')

    # step 3: use videopose3d to get 3d_keypoints
    in_dir = '../data/data-clean/refrigerator/'  # mp4
    camera1 = glob(in_dir + '/**/*1.mp4', recursive=True)
    camera2 = glob(in_dir + '/**/*2.mkv', recursive=True)
    camera3 = glob(in_dir + '/**/*3.mp4', recursive=True)
    camera32 = glob(in_dir + '/**/*3 2.mp4', recursive=True)
    print(f'camera1: {len(camera1)},camera2: {len(camera2)}, camera3: {len(camera3)}, and camera32: {len(camera32)}')
    for file_list in [camera1, camera2, camera3]:
        for i, f in enumerate(file_list):
            print(i, f)
            f_name = os.path.basename(f)
            out_file = os.path.join('out3d_pred', os.path.relpath(f, '../'))
            out_dir = os.path.dirname(out_file)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            if os.path.exists(out_file + '.npy'):
                print(out_file + '.npy')
                continue
            # cmd = 'python3.7 run.py -d custom -k 2d_keypoints -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject ask_time_1_1614904536_1.mp4 --viz-action custom --viz-camera 0 --viz-video ../data/demo/ask_time_1_1614904536_1.mp4 --viz-output output.mp4 --viz-size 6'

            # output the video with predicted 3d keypoints
            # cmd = f'python3.7 run.py -d custom -k 2d_keypoints -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject \'{f_name}\' --viz-action custom --viz-camera 0 --viz-video \'{f}\' --viz-output \'{out_file}\' --viz-size 6'

            # save the predicted 3d keypoints to out_file
            cmd = f'python3.7 run.py -d custom -k 2d_keypoints -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject \'{f_name}\' --viz-action custom --viz-camera 0 --viz-video \'{f}\' --viz-export \'{out_file}\' --viz-size 6'
            os.system(cmd)
    print('finished')


if __name__ == '__main__':
    main()
