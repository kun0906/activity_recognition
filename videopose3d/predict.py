import os
from glob import glob


def main():
    in_dir = '../data/data-clean/refrigerator/'  # mp4
    file_list = glob(in_dir + '/**/*.mp4', recursive=True)
    for i, f in enumerate(file_list):
        print(i, f)
        f_name = os.path.basename(f)
        out_file = os.path.join('out3d_pred', os.path.relpath(f, '../'))
        out_dir = os.path.dirname(out_file)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # cmd = 'python3.7 run.py -d custom -k 2d_keypoints -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject ask_time_1_1614904536_1.mp4 --viz-action custom --viz-camera 0 --viz-video ../data/demo/ask_time_1_1614904536_1.mp4 --viz-output output.mp4 --viz-size 6'
        cmd = f'python3.7 run.py -d custom -k 2d_keypoints -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject \'{f_name}\' --viz-action custom --viz-camera 0 --viz-video \'{f}\' --viz-output \'{out_file}\' --viz-size 6'
        os.system(cmd)


if __name__ == '__main__':
    main()

    # afdafadfaf
