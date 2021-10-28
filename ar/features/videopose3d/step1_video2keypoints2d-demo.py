# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Perform inference on a single video or all videos with a certain extension
(e.g., .mp4) in a folder.
"""
import cv2
import detectron2
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import subprocess as sp
import numpy as np
import time
import argparse
import sys
import os
import glob

from detectron2.utils.visualizer import Visualizer

from ar.utils.utils import check_path


def parse_args():
	parser = argparse.ArgumentParser(description='End-to-end inference')
	parser.add_argument(
		'--cfg',
		dest='cfg',
		help='cfg model file (/path/to/model_config.yaml)',
		default=None,
		type=str
	)
	parser.add_argument(
		'--output-dir',
		dest='output_dir',
		help='directory for visualization pdfs (default: /tmp/infer_simple)',
		default='/tmp/infer_simple',
		type=str
	)
	parser.add_argument(
		'--image-ext',
		dest='image_ext',
		help='image file name extension (default: mp4)',
		default='mp4',
		type=str
	)
	parser.add_argument(
		'--im_or_folder',
		dest='im_or_folder',
		help='image or folder of images',
		default=''
	)
	# if len(sys.argv) == 1:
	#     parser.print_help()
	#     sys.exit(1)
	return parser.parse_args()


def get_resolution(filename):
	command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
	           '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
	pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
	for line in pipe.stdout:
		w, h = line.decode().strip().split(',')
		return int(w), int(h)


def read_video(filename):
	# w, h = get_resolution(filename)
	#
	# command = ['ffmpeg',
	#            '-i', filename,
	#            '-f', 'image2pipe',
	#            '-pix_fmt', 'bgr24',
	#            '-vsync', '0',
	#            '-vcodec', 'rawvideo', '-']
	#
	# pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
	# while True:
	#     data = pipe.stdout.read(w * h * 3)
	#     if not data:
	#         break
	#     yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))

	cap = cv2.VideoCapture(filename)
	fps = cap.get(cv2.CAP_PROP_FPS)
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	duration = frame_count / fps

	imgs = []
	i = 0
	while i < frame_count:
		ret, img = cap.read()
		if not ret: break
		imgs.append(img)
		i += 1
		print(i, ret)

	cap.release()
	cv2.destroyAllWindows()
	# https://github.com/opencv/opencv/issues/4362
	# Python cv2.VideoCapture.read() does not read all frames
	print(f'total frames: {len(imgs)}, fps: {fps}, frame_count: {frame_count}, duration: {duration}s')
	return imgs


def _main(args):
	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file(args.cfg))
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.cfg)
	print(cfg)
	predictor = DefaultPredictor(cfg)  # detectron2 API

	if os.path.isdir(args.im_or_folder):
		im_list = glob.iglob(args.im_or_folder + '/**/*.' + args.image_ext, recursive=True)
	else:
		im_list = [args.im_or_folder]
	im_list = list(im_list)
	camera1 = [f for f in im_list if '1.mp4' in f]
	camera2 = [f for f in im_list if '2.mkv' in f]
	camera3 = [f for f in im_list if '3.mp4' in f]
	camera32 = [f for f in im_list if '3 2.mp4' in f]
	print(f'total videos: {len(im_list)}, in which, camera1={len(camera1)}, camera2={len(camera2)}, '
	      f'camera3={len(camera3)}, camera32={len(camera32)}')
	for i, video_name in enumerate(sorted(im_list, reverse=True)):
		# out_name = os.path.join(args.output_dir, os.path.basename(video_name))
		out_name = os.path.join(args.output_dir, os.path.relpath(video_name, 'examples/'))
		print(f'i: {i}, out_file: ', out_name + '.npz')
		if os.path.exists(out_name + '.npz'): continue

		tmp_dir = os.path.dirname(out_name)
		if not os.path.exists(tmp_dir):
			os.makedirs(tmp_dir)
		print('Processing {}'.format(video_name))

		boxes = []
		segments = []
		keypoints = []
		predicted_imgs = []

		for frame_i, im in enumerate(read_video(video_name)):
			t = time.time()
			outputs = predictor(im)['instances'].to('cpu')
			# print(outputs)
			print('Frame {} processed in {:.3f}s'.format(frame_i, time.time() - t))

			has_bbox = False
			if outputs.has('pred_boxes'):
				bbox_tensor = outputs.pred_boxes.tensor.numpy()
				if len(bbox_tensor) > 0:
					has_bbox = True
					scores = outputs.scores.numpy()[:, None]
					bbox_tensor = np.concatenate((bbox_tensor, scores), axis=1)
			if has_bbox:
				kps = outputs.pred_keypoints.numpy()
				kps_xy = kps[:, :, :2]
				kps_prob = kps[:, :, 2:3]
				kps_logit = np.zeros_like(kps_prob)  # Dummy
				kps = np.concatenate((kps_xy, kps_logit, kps_prob), axis=2)
				kps = kps.transpose(0, 2, 1)
			else:
				kps = []
				bbox_tensor = []

			# Mimic Detectron1 format
			cls_boxes = [[], bbox_tensor]
			cls_keyps = [[], kps]

			boxes.append(cls_boxes)
			segments.append(None)
			keypoints.append(cls_keyps)

			# draw 2d keypoint
			# https://towardsdatascience.com/understanding-detectron2-demo-bc648ea569e5
			v = Visualizer(im[:, :, ::-1],
			               MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
			               scale=1.0)
			result = v.draw_instance_predictions(outputs.to("cpu"))
			result_image = result.get_image()[:, :, ::-1]
			# get file name without extension, -1 to remove "." at the end
			# out_file_name = f"examples/~tmp/{frame_i}.png"
			# check_path(out_file_name)
			# cv2.imwrite(out_file_name, result_image)
			predicted_imgs.append(result_image)

		# out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
		cap = cv2.VideoCapture(video_name)
		fourcc = cv2.VideoWriter_fourcc(*'mp4v'.lower())
		fourcc = cv2.VideoWriter_fourcc(*(chr(fourcc & 0xff) + chr((fourcc >> 8) & 0xff) + chr((fourcc >> 16) & 0xff)
		                                  + chr((fourcc >> 24) & 0xff)))
		# print(fourcc)
		frame_width = int(cap.get(3))
		frame_height = int(cap.get(4))
		# Define the codec and filename.
		fps = cap.get(cv2.CAP_PROP_FPS)
		out = cv2.VideoWriter(out_name + '.mp4', fourcc, fps, (frame_width, frame_height), isColor=True)
		print(out_name, f', fps: {fps}')
		for i in range(len(predicted_imgs)):
			out.write(predicted_imgs[i])
		cap.release()
		out.release()
		cv2.destroyAllWindows()

		# Video resolution
		metadata = {
			'w': im.shape[1],
			'h': im.shape[0],
		}

		np.savez_compressed(out_name, boxes=boxes, segments=segments, keypoints=keypoints, metadata=metadata)
		print(f'total videos: {len(im_list)}, in which, camera1={len(camera1)}, camera2={len(camera2)}, '
		      f'camera3={len(camera3)}, camera32={len(camera32)}')
		break


def main(in_dir='examples/datasets/data-clean/refrigerator', out_dir='examples/out/~tmp/', image_type='mp4'):
	setup_logger()
	args = parse_args()
	args.cfg = 'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'
	args.output_dir = out_dir
	# args.image_ext = 'mp4'
	args.image_ext = image_type
	args.im_or_folder = in_dir  # 'examples/data/data-clean/refrigerator'

	"""
		python3.7 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir out3d --image-ext mp4
   ../../data/data-clean/refrigerator 

	"""
	print(args)
	_main(args)


if __name__ == '__main__':
	main()
