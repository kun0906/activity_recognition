""" Using the "videopose3d" to get the 3d keypoints' features.

	Instructions:
		1. PYTHONPATH=. python3.7 ar/features/videopose3d_feature.py

"""
# Email: kun.bj@outlook.com
# License: xxx

#
# def main():
#     # step 1: use detectron2 to get 2d_keypoints
#     """ Be careful with directory path
#     # generate 2D keypoint predictions from videos
#     cd inference
#     # python3.7 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir out_directory --image-ext mp4 input_directory
#     # python3.7 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir out2d --image-ext mp4 ../../data/demo
#     python3.7 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir out2d --image-ext mp4 ../../data/data-clean/refrigerator
#     python3.7 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir out2d --image-ext mkv ../../data/data-clean/refrigerator
#
#     # create customized dataset "data/data_2d_custom_myvideos.npz"
#     cd ../data
#     #python3.7 prepare_data_2d_custom.py -i ../inference/out2d/data/data-clean/ -o myvideos
#     python3.7 prepare_data_2d_custom.py -i ../inference/out2d/data/data-clean/ -o 2d_keypoints
#     cd ..
#
#     """
#     root_dir = 'ar/features/videopose3d'
#
#     # step 3: use videopose3d to get 3d_keypoints
#     in_dir = 'examples/data/data-clean/refrigerator/'  # mp4
#     camera1 = glob(in_dir + '/**/*1.mp4', recursive=True)
#     camera2 = glob(in_dir + '/**/*2.mkv', recursive=True)
#     camera3 = glob(in_dir + '/**/*3.mp4', recursive=True)
#     camera32 = glob(in_dir + '/**/*3 2.mp4', recursive=True)
#     print(f'camera1: {len(camera1)},camera2: {len(camera2)}, camera3: {len(camera3)}, and camera32: {len(camera32)}')
#     for file_list in [camera1, camera2, camera3]:
#         for i, f in enumerate(sorted(file_list)):
#             print(i, f)
#             f_name = os.path.basename(f)
#             # out_file = os.path.join('out3d_pred', os.path.relpath(f, '../'))
#             out_file = os.path.join('examples/out/keypoints3d/', os.path.relpath(f, 'examples'))
#             out_dir = os.path.dirname(out_file)
#             if not os.path.exists(out_dir):
#                 os.makedirs(out_dir)
#
#             if os.path.exists(out_file + '.npy'):
#                 print(out_file + '.npy')
#                 continue
#             # cmd = 'python3.7 run.py -d custom -k 2d_keypoints -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject ask_time_1_1614904536_1.mp4 --viz-action custom --viz-camera 0 --viz-video ../data/demo/ask_time_1_1614904536_1.mp4 --viz-output output.mp4 --viz-size 6'
#
#             # output the video with predicted 3d keypoints
#             # cmd = f'python3.7 run.py -d custom -k 2d_keypoints -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject \'{f_name}\' --viz-action custom --viz-camera 0 --viz-video \'{f}\' --viz-output \'{out_file}\' --viz-size 6'
#
#             # save the predicted 3d keypoints to out_file
#             cmd = f'python3.7 {root_dir}/run.py -d custom -k 2d_keypoints -arc 3,3,3,3,3 -c checkpoint --evaluate {root_dir}/checkpoint/pretrained_h36m_detectron_coco.bin --render --viz-subject \'{f_name}\' --viz-action custom --viz-camera 0 --viz-video \'{f}\' --viz-export \'{out_file}\' --viz-size 6'
#             os.system(cmd)
#             # break
#     print('finished')
import step1_video2keypoints2d
import step2_zip_2d
import step3_2d_to_3d
from ar.utils.utils import timer


@timer
def gen_3dkeypoints(in_dir, out_dir):
	# step1: video to 2d keypoints
	out_2d_dir = f'{out_dir}/keypoints2d'  # individual kypoints2d
	step1_video2keypoints2d.main(in_dir, out_2d_dir, image_type='mp4')
	step1_video2keypoints2d.main(in_dir, out_2d_dir, image_type='mkv')
	# step 2: aggregate all the 2d keypoints and only extract the needed info (such as (x,y) coordinates and pred_prob)
	out_agg_2d_dir = f'{out_dir}/keypoints2d_agg'
	step2_zip_2d.main(in_dir=out_2d_dir, out_dir=out_agg_2d_dir)  #
	# step3: 2d to 3d keypoints
	out_3d_dir = f'{out_dir}/keypoints3d'
	step3_2d_to_3d.main(in_dir=in_dir, out_dir=out_3d_dir)


if __name__ == '__main__':
	in_dir = 'examples/datasets/data-clean/refrigerator'
	out_dir = 'examples/out'
	gen_3dkeypoints(in_dir, out_dir)
