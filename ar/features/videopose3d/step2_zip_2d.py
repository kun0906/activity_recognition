# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from glob import glob
import os
import sys

import argparse
from ar.features.videopose3d.data.data_utils import suggest_metadata

output_prefix_2d = 'data_2d_custom_'


def decode(filename):
	# Latin1 encoding because Detectron runs on Python 2.7
	print('Processing {}'.format(filename))
	data = np.load(filename, encoding='latin1', allow_pickle=True)
	bb = data['boxes']
	kp = data['keypoints']
	metadata = data['metadata'].item()
	results_bb = []
	results_kp = []
	results_bb_probs = []
	results_kp_probs = []
	for i in range(len(bb)):
		if len(bb[i][1]) == 0 or len(kp[i][1]) == 0:
			# No bbox/keypoints detected for this frame -> will be interpolated
			results_bb.append(np.full(4, np.nan, dtype=np.float32))  # 4 bounding box coordinates
			results_kp.append(np.full((17, 4), np.nan, dtype=np.float32))  # 17 COCO keypoints
			results_bb_probs.append(np.nan)
			results_kp_probs.append(np.full((17,), np.nan, dtype=np.float32))  # probs
			continue
		best_match = np.argmax(bb[i][1][:, 4])
		best_bb = bb[i][1][best_match, :4]
		best_kp = kp[i][1][best_match].T.copy()
		results_bb.append(best_bb)
		results_kp.append(best_kp)
		results_bb_probs.append(bb[i][1][best_match, 4])
		results_kp_probs.append(kp[i][1][best_match].T.copy()[:, 3])  # probs

	bb = np.array(results_bb, dtype=np.float32)
	bb_probs = np.array(results_bb_probs, dtype=np.float32)
	kp = np.array(results_kp, dtype=np.float32)
	kp = kp[:, :, :2]  # Extract (x, y)
	# kp_probs = kp[:, :, 3]  # Extract the probablity of the predicted point: (x, y)
	kp_probs = np.array(results_kp_probs, dtype=np.float32)

	# without mask
	# Fix missing bboxes/keypoints by linear interpolation
	mask = ~np.isnan(bb[:, 0])
	indices = np.arange(len(bb))
	for i in range(4):
		bb[:, i] = np.interp(indices, indices[mask], bb[mask, i])
	for i in range(17):
		for j in range(2):
			kp[:, i, j] = np.interp(indices, indices[mask], kp[mask, i, j])
	print('{} total frames processed'.format(len(bb)))
	print('{} frames were interpolated'.format(np.sum(~mask)))

	# # fill nan to 0
	# bb[np.isnan(bb)] = 0
	# bb_probs[np.isnan(bb_probs)] = 0
	# kp[np.isnan(kp)] = 0
	# kp_probs[np.isnan(kp_probs)] = 0
	print('----------')

	return [{
		'start_frame': 0,  # Inclusive
		'end_frame': len(kp),  # Exclusive
		'bounding_boxes': bb,
		'bounding_boxes_probs': bb_probs,
		'keypoints': kp,
		'keypoints_probs': kp_probs
	}], metadata


def main(in_dir='examples/out/keypoints2d', out_dir='examples/out/2d_keypoints'):
	# if os.path.basename(os.getcwd()) != 'data':
	#     print('This script must be launched from the "data" directory')
	#     exit(0)

	parser = argparse.ArgumentParser(description='Custom dataset creator')
	parser.add_argument('-i', '--input', type=str, default='', metavar='PATH', help='detections directory')
	parser.add_argument('-o', '--output', type=str, default='', metavar='PATH', help='output suffix for 2D detections')
	args = parser.parse_args()
	args.input = in_dir
	args.output = out_dir

	if not args.input:
		print('Please specify the input directory')
		exit(0)

	if not args.output:
		print('Please specify an output suffix (e.g. detectron_pt_coco)')
		exit(0)

	print('Parsing 2D detections from', args.input)

	metadata = suggest_metadata('coco')
	metadata['video_metadata'] = {}

	output = {}
	file_list = glob(args.input + '/**/*.npz', recursive=True)
	camera1 = [f for f in file_list if '1.mp4' in f]
	camera2 = [f for f in file_list if '2.mpk' in f]
	camera3 = [f for f in file_list if '3.mp4' in f]
	camera32 = [f for f in file_list if '3 2.mp4' in f]
	print(f'total videos: {len(file_list)}, in which, camera1={len(camera1)}, camera2={len(camera2)}, '
	      f'camera3={len(camera3)}, camera32={len(camera32)}')
	error_lst = []
	camera1 = 0
	camera2 = 0
	camera3 = 0
	camera32 = 0
	keypoints2d = {}
	for i, f in enumerate(sorted(file_list)):
		# print(i, f)
		try:
			canonical_name = os.path.splitext(os.path.basename(f))[0]
			data, video_metadata = decode(f)
			output[canonical_name] = {}
			output[canonical_name]['custom'] = [data[0]['keypoints'].astype('float32')]
			metadata['video_metadata'][canonical_name] = video_metadata
			keypoints2d[canonical_name] = data
			if '1.mp4' in f:
				camera1 += 1
			elif '2.mkv' in f:
				camera2 += 1
			elif '3.mp4' in f:
				camera3 += 1
			elif '3 2.mp4' in f:
				camera32 += 1
			else:
				pass
		except Exception as e:
			print(f'Error: {e}, i={i}, {f}')
			error_lst.append((i, f, e))

	print(f'len(error_lst) ={len(error_lst)}, {error_lst}.')
	print('Saving...')
	basename = output_prefix_2d + os.path.basename(args.output)
	out_file = os.path.join(os.path.dirname(args.output), basename + '-with-probs')
	# out_file = os.path.join(os.path.dirname(args.output), basename )
	print(os.path.abspath(out_file) + '.npz')
	np.savez_compressed(out_file, positions_2d=output, metadata=metadata, keypoints2d=keypoints2d)
	print(f'total videos: {camera1 + camera2 + camera3 + camera32}, in which, camera1={camera1}, '
	      f'camera2={camera2}, '
	      f'camera3={camera3}, camera32={camera32}')
	print('Done.')
	return out_file + '.npz'


if __name__ == '__main__':
	main()
