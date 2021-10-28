"""
    Run the command under root directory (i.e., project directory)
    cmd = f'python3.7 {root_dir}/run.py -d custom -k 2d_keypoints -arc 3,3,3,3,3 -c checkpoint --evaluate {root_dir}/checkpoint/pretrained_h36m_detectron_coco.bin --render --viz-subject \'{f_name}\' --viz-action custom --viz-camera 0 --viz-video \'{f}\' --viz-export \'{out_file}\' --viz-size 6'
    os.system(cmd)

    python3.7 ar/features/videopose3d/run.py -d custom -k 2d_keypoints -arc 3,3,3,3,3 -c checkpoint --evaluate ar/features/videopose3d/checkpoint/pretrained_h36m_detectron_coco.bin --render --viz-subject 'take_out_item_2_1616179391_2.mkv' --viz-action custom --viz-camera 0 --viz-video 'examples/data/data-clean/refrigerator/take_out_item/4/take_out_item_2_1616179391_2.mkv' --viz-export 'examples/out/keypoints3d/data/data-clean/refrigerator/take_out_item/4/take_out_item_2_1616179391_2.mkv' --viz-size 6

"""
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random

import argparse


def parse_args():
	parser = argparse.ArgumentParser(description='Training script')

	# General arguments
	parser.add_argument('-d', '--dataset', default='custom', type=str, metavar='NAME',
	                    help='target dataset')  # h36m or humaneva
	parser.add_argument('-k', '--keypoints', default='2d_keypoints', type=str, metavar='NAME',
	                    help='2D detections to use')
	parser.add_argument('-str', '--subjects-train', default='S1,S5,S6,S7,S8', type=str, metavar='LIST',
	                    help='training subjects separated by comma')
	parser.add_argument('-ste', '--subjects-test', default='S9,S11', type=str, metavar='LIST',
	                    help='test subjects separated by comma')
	parser.add_argument('-sun', '--subjects-unlabeled', default='', type=str, metavar='LIST',
	                    help='unlabeled subjects separated by comma for self-supervision')
	parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
	                    help='actions to train/test on, separated by comma, or * for all')
	parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
	                    help='checkpoint directory')
	parser.add_argument('--checkpoint-frequency', default=10, type=int, metavar='N',
	                    help='create a checkpoint every N epochs')
	parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
	                    help='checkpoint to resume (file name)')
	parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
	                    help='checkpoint to evaluate (file name)')
	parser.add_argument('--render', action='store_true', help='visualize a particular video')
	parser.add_argument('--by-subject', action='store_true', help='break down error by subject (on evaluation)')
	parser.add_argument('--export-training-curves', action='store_true', help='save training curves as .png images')

	# Model arguments
	parser.add_argument('-s', '--stride', default=1, type=int, metavar='N', help='chunk size to use during training')
	parser.add_argument('-e', '--epochs', default=60, type=int, metavar='N', help='number of training epochs')
	parser.add_argument('-b', '--batch-size', default=1024, type=int, metavar='N',
	                    help='batch size in terms of predicted frames')
	parser.add_argument('-drop', '--dropout', default=0.25, type=float, metavar='P', help='dropout probability')
	parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
	parser.add_argument('-lrd', '--lr-decay', default=0.95, type=float, metavar='LR',
	                    help='learning rate decay per epoch')
	parser.add_argument('-no-da', '--no-data-augmentation', dest='data_augmentation', action='store_false',
	                    help='disable train-time flipping')
	parser.add_argument('-no-tta', '--no-test-time-augmentation', dest='test_time_augmentation', action='store_false',
	                    help='disable test-time flipping')
	parser.add_argument('-arc', '--architecture', default='3,3,3,3,3', type=str, metavar='LAYERS',
	                    help='filter widths separated by comma')
	parser.add_argument('--causal', action='store_true', help='use causal convolutions for real-time processing')
	parser.add_argument('-ch', '--channels', default=1024, type=int, metavar='N',
	                    help='number of channels in convolution layers')

	# Experimental
	parser.add_argument('--subset', default=1, type=float, metavar='FRACTION', help='reduce dataset size by fraction')
	parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR',
	                    help='downsample frame rate by factor (semi-supervised)')
	parser.add_argument('--warmup', default=1, type=int, metavar='N', help='warm-up epochs for semi-supervision')
	parser.add_argument('--no-eval', action='store_true',
	                    help='disable epoch evaluation while training (small speed-up)')
	parser.add_argument('--dense', action='store_true', help='use dense convolutions instead of dilated convolutions')
	parser.add_argument('--disable-optimizations', action='store_true',
	                    help='disable optimized model for single-frame predictions')
	parser.add_argument('--linear-projection', action='store_true',
	                    help='use only linear coefficients for semi-supervised projection')
	parser.add_argument('--no-bone-length', action='store_false', dest='bone_length_term',
	                    help='disable bone length term in semi-supervised settings')
	parser.add_argument('--no-proj', action='store_true', help='disable projection for semi-supervised setting')

	# Visualization
	parser.add_argument('--viz-subject', type=str, metavar='STR', help='subject to render')
	parser.add_argument('--viz-action', type=str, metavar='STR', help='action to render')
	parser.add_argument('--viz-camera', type=int, default=0, metavar='N', help='camera to render')
	parser.add_argument('--viz-video', type=str, metavar='PATH', help='path to input video')
	parser.add_argument('--viz-skip', type=int, default=0, metavar='N', help='skip first N frames of input video')
	parser.add_argument('--viz-output', type=str, metavar='PATH', help='output file name (.gif or .mp4)')
	parser.add_argument('--viz-export', type=str, metavar='PATH', help='output file name for coordinates')
	parser.add_argument('--viz-bitrate', type=int, default=3000, metavar='N', help='bitrate for mp4 videos')
	parser.add_argument('--viz-no-ground-truth', action='store_true', help='do not show ground-truth poses')
	parser.add_argument('--viz-limit', type=int, default=-1, metavar='N', help='only render first N frames')
	parser.add_argument('--viz-downsample', type=int, default=1, metavar='N', help='downsample FPS by a factor N')
	parser.add_argument('--viz-size', type=int, default=5, metavar='N', help='image size')

	parser.set_defaults(bone_length_term=True)
	parser.set_defaults(data_augmentation=True)
	parser.set_defaults(test_time_augmentation=True)

	args = parser.parse_args()
	# Check invalid configuration
	if args.resume and args.evaluate:
		print('Invalid flags: --resume and --evaluate cannot be set at the same time')
		exit()

	if args.export_training_curves and args.no_eval:
		print('Invalid flags: --export-training-curves and --no-eval cannot be set at the same time')
		exit()

	return args


args = parse_args()
args.dataset = 'custom'
args.keypoints = '2d_keypoints'
args.architecture = '3,3,3,3,3'
args.evaluate = 'ar/features/videopose3d/checkpoint/pretrained_h36m_detectron_coco.bin'
args.render = True
args.viz_action = 'custom'
args.viz_camera = 0
# no_interaction is predicted as 'open_close_fridge'
# examples/classical_ml/out/keypoints3d-20210907/keypoints3d/data/data-clean/refrigerator/no_interaction/4/no_interaction_4_1616182615_1.mp4.npy
args.viz_subject = 'no_interaction_1_1625875733_1.mp4'
# only use for rendering the final mp4. only the 2d keypoints are used to pred the 3d keypoints
args.viz_video = f'examples/datasets/data-clean/refrigerator/no_interaction/2/{args.viz_subject}'
args.viz_export = f'examples/strange/data/data-clean/refrigerator/open_close_fridge/4/{args.viz_subject}'  # 3d .npy
args.viz_output = f'examples/strange/data/data-clean/refrigerator/open_close_fridge/4/{args.viz_subject}'  # visualization
args.viz_size = 6
print(args)
tmp_dir = os.path.dirname(args.viz_export)
if not os.path.exists(tmp_dir):
	os.makedirs(tmp_dir)

try:
	# Create checkpoint directory if it does not exist
	os.makedirs(args.checkpoint)
except OSError as e:
	if e.errno != errno.EEXIST:
		raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

print('Loading dataset...')
if args.dataset == 'h36m':
	from common.h36m_dataset import Human36mDataset

	keypoints2d_path = 'ar/features/videopose3d/data/data_3d_' + args.dataset + '.npz'
	# keypoints2d_path = 'examples/out/data_2d_custom_keypoints2d_agg.npz'
	print(keypoints2d_path)
	dataset = Human36mDataset(keypoints2d_path)
elif args.dataset.startswith('humaneva'):
	from common.humaneva_dataset import HumanEvaDataset

	keypoints2d_path = 'ar/features/videopose3d/data/data_3d_' + args.dataset + '.npz'
	# keypoints2d_path = 'examples/out/data_2d_custom_keypoints2d_agg.npz'
	print(keypoints2d_path)
	dataset = HumanEvaDataset(keypoints2d_path)
elif args.dataset.startswith('custom'):
	from common.custom_dataset import CustomDataset

	keypoints2d_path = 'examples/out/data_2d_custom_keypoints2d_agg-with-probs.npz'
	print(keypoints2d_path)
	# dataset = CustomDataset('ar/features/videopose3d/data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
	dataset = CustomDataset(keypoints2d_path)
else:
	raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in dataset.subjects():
	for action in dataset[subject].keys():
		anim = dataset[subject][action]

		if 'positions' in anim:
			positions_3d = []
			for cam in anim['cameras']:
				pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
				pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
				positions_3d.append(pos_3d)
			anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
# keypoints = np.load('ar/features/videopose3d/data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz',
#                     allow_pickle=True)
keypoints = np.load(keypoints2d_path, allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

for subject in dataset.subjects():
	assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
	for action in dataset[subject].keys():
		assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(
			action, subject)
		if 'positions_3d' not in dataset[subject][action]:
			continue

		for cam_idx in range(len(keypoints[subject][action])):

			# We check for >= instead of == because some videos in H3.6M contain extra frames
			mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
			assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

			if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
				# Shorten sequence
				keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

		assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

for subject in keypoints.keys():
	for action in keypoints[subject]:
		for cam_idx, kps in enumerate(keypoints[subject][action]):
			# Normalize camera frame
			cam = dataset.cameras()[subject][cam_idx]
			kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
			keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(',')
subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
if not args.render:
	subjects_test = args.subjects_test.split(',')
else:
	subjects_test = [args.viz_subject]

semi_supervised = len(subjects_semi) > 0
if semi_supervised and not dataset.supports_semi_supervised():
	raise RuntimeError('Semi-supervised training is not implemented for this dataset')


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
	out_poses_3d = []
	out_poses_2d = []
	out_camera_params = []
	for subject in subjects:
		for action in keypoints[subject].keys():
			if action_filter is not None:
				found = False
				for a in action_filter:
					if action.startswith(a):
						found = True
						break
				if not found:
					continue

			poses_2d = keypoints[subject][action]
			for i in range(len(poses_2d)):  # Iterate across cameras
				out_poses_2d.append(poses_2d[i])

			if subject in dataset.cameras():
				cams = dataset.cameras()[subject]
				assert len(cams) == len(poses_2d), 'Camera count mismatch'
				for cam in cams:
					if 'intrinsic' in cam:
						out_camera_params.append(cam['intrinsic'])

			if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
				poses_3d = dataset[subject][action]['positions_3d']
				assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
				for i in range(len(poses_3d)):  # Iterate across cameras
					out_poses_3d.append(poses_3d[i])

	if len(out_camera_params) == 0:
		out_camera_params = None
	if len(out_poses_3d) == 0:
		out_poses_3d = None

	stride = args.downsample
	if subset < 1:
		for i in range(len(out_poses_2d)):
			n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
			start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
			out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
			if out_poses_3d is not None:
				out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
	elif stride > 1:
		# Downsample as requested
		for i in range(len(out_poses_2d)):
			out_poses_2d[i] = out_poses_2d[i][::stride]
			if out_poses_3d is not None:
				out_poses_3d[i] = out_poses_3d[i][::stride]

	return out_camera_params, out_poses_3d, out_poses_2d


action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
	print('Selected actions:', action_filter)

cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

filter_widths = [int(x) for x in args.architecture.split(',')]
if not args.disable_optimizations and not args.dense and args.stride == 1:
	# Use optimized model for single-frame predictions
	model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
	                                           dataset.skeleton().num_joints(),
	                                           filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
	                                           channels=args.channels)
else:
	# When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
	model_pos_train = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
	                                dataset.skeleton().num_joints(),
	                                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
	                                channels=args.channels,
	                                dense=args.dense)

model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                          filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                          dense=args.dense)

receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2  # Padding on each side
if args.causal:
	print('INFO: Using causal convolutions')
	causal_shift = pad
else:
	causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
	model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
	model_pos = model_pos.cuda()
	model_pos_train = model_pos_train.cuda()

if args.resume or args.evaluate:
	# chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
	chk_filename = args.resume if args.resume else args.evaluate
	print('Loading checkpoint', chk_filename)
	checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
	print('This model was trained for {} epochs'.format(checkpoint['epoch']))
	model_pos_train.load_state_dict(checkpoint['model_pos'])
	model_pos.load_state_dict(checkpoint['model_pos'])  # load model's parameters

	if args.evaluate and 'model_traj' in checkpoint:
		# Load trajectory model if it contained in the checkpoint (e.g. for inference in the wild)
		model_traj = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
		                           filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
		                           channels=args.channels,
		                           dense=args.dense)
		if torch.cuda.is_available():
			model_traj = model_traj.cuda()
		model_traj.load_state_dict(checkpoint['model_traj'])
	else:
		model_traj = None

test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                    joints_right=joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

if not args.evaluate:  # training
	cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)

	lr = args.learning_rate
	if semi_supervised:
		cameras_semi, _, poses_semi_2d = fetch(subjects_semi, action_filter, parse_3d_poses=False)

		if not args.disable_optimizations and not args.dense and args.stride == 1:
			# Use optimized model for single-frame predictions
			model_traj_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
			                                            filter_widths=filter_widths, causal=args.causal,
			                                            dropout=args.dropout, channels=args.channels)
		else:
			# When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
			model_traj_train = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
			                                 filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
			                                 channels=args.channels,
			                                 dense=args.dense)

		model_traj = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
		                           filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
		                           channels=args.channels,
		                           dense=args.dense)
		if torch.cuda.is_available():
			model_traj = model_traj.cuda()
			model_traj_train = model_traj_train.cuda()
		optimizer = optim.Adam(list(model_pos_train.parameters()) + list(model_traj_train.parameters()),
		                       lr=lr, amsgrad=True)

		losses_2d_train_unlabeled = []
		losses_2d_train_labeled_eval = []
		losses_2d_train_unlabeled_eval = []
		losses_2d_valid = []

		losses_traj_train = []
		losses_traj_train_eval = []
		losses_traj_valid = []
	else:
		optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)

	lr_decay = args.lr_decay

	losses_3d_train = []
	losses_3d_train_eval = []
	losses_3d_valid = []

	epoch = 0
	initial_momentum = 0.1
	final_momentum = 0.001

	train_generator = ChunkedGenerator(args.batch_size // args.stride, cameras_train, poses_train, poses_train_2d,
	                                   args.stride,
	                                   pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
	                                   kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
	                                   joints_right=joints_right)
	train_generator_eval = UnchunkedGenerator(cameras_train, poses_train, poses_train_2d,
	                                          pad=pad, causal_shift=causal_shift, augment=False)
	print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))
	if semi_supervised:
		semi_generator = ChunkedGenerator(args.batch_size // args.stride, cameras_semi, None, poses_semi_2d,
		                                  args.stride,
		                                  pad=pad, causal_shift=causal_shift, shuffle=True,
		                                  random_seed=4321, augment=args.data_augmentation,
		                                  kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
		                                  joints_right=joints_right,
		                                  endless=True)
		semi_generator_eval = UnchunkedGenerator(cameras_semi, None, poses_semi_2d,
		                                         pad=pad, causal_shift=causal_shift, augment=False)
		print('INFO: Semi-supervision on {} frames'.format(semi_generator_eval.num_frames()))

	if args.resume:
		epoch = checkpoint['epoch']
		if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
			optimizer.load_state_dict(checkpoint['optimizer'])
			train_generator.set_random_state(checkpoint['random_state'])
		else:
			print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

		lr = checkpoint['lr']
		if semi_supervised:
			model_traj_train.load_state_dict(checkpoint['model_traj'])
			model_traj.load_state_dict(checkpoint['model_traj'])
			semi_generator.set_random_state(checkpoint['random_state_semi'])

	print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
	print('** The final evaluation will be carried out after the last training epoch.')

	# Pos model only
	while epoch < args.epochs:
		start_time = time()
		epoch_loss_3d_train = 0
		epoch_loss_traj_train = 0
		epoch_loss_2d_train_unlabeled = 0
		N = 0
		N_semi = 0
		model_pos_train.train()
		if semi_supervised:
			# Semi-supervised scenario
			model_traj_train.train()
			for (_, batch_3d, batch_2d), (cam_semi, _, batch_2d_semi) in \
					zip(train_generator.next_epoch(), semi_generator.next_epoch()):

				# Fall back to supervised training for the first epoch (to avoid instability)
				skip = epoch < args.warmup

				cam_semi = torch.from_numpy(cam_semi.astype('float32'))
				inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
				if torch.cuda.is_available():
					cam_semi = cam_semi.cuda()
					inputs_3d = inputs_3d.cuda()

				inputs_traj = inputs_3d[:, :, :1].clone()
				inputs_3d[:, :, 0] = 0

				# Split point between labeled and unlabeled samples in the batch
				split_idx = inputs_3d.shape[0]

				inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
				inputs_2d_semi = torch.from_numpy(batch_2d_semi.astype('float32'))
				if torch.cuda.is_available():
					inputs_2d = inputs_2d.cuda()
					inputs_2d_semi = inputs_2d_semi.cuda()
				inputs_2d_cat = torch.cat((inputs_2d, inputs_2d_semi), dim=0) if not skip else inputs_2d

				optimizer.zero_grad()

				# Compute 3D poses
				predicted_3d_pos_cat = model_pos_train(inputs_2d_cat)

				loss_3d_pos = mpjpe(predicted_3d_pos_cat[:split_idx], inputs_3d)
				epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
				N += inputs_3d.shape[0] * inputs_3d.shape[1]
				loss_total = loss_3d_pos

				# Compute global trajectory
				predicted_traj_cat = model_traj_train(inputs_2d_cat)
				w = 1 / inputs_traj[:, :, :, 2]  # Weight inversely proportional to depth
				loss_traj = weighted_mpjpe(predicted_traj_cat[:split_idx], inputs_traj, w)
				epoch_loss_traj_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_traj.item()
				assert inputs_traj.shape[0] * inputs_traj.shape[1] == inputs_3d.shape[0] * inputs_3d.shape[1]
				loss_total += loss_traj

				if not skip:
					# Semi-supervised loss for unlabeled samples
					predicted_semi = predicted_3d_pos_cat[split_idx:]
					if pad > 0:
						target_semi = inputs_2d_semi[:, pad:-pad, :, :2].contiguous()
					else:
						target_semi = inputs_2d_semi[:, :, :, :2].contiguous()

					projection_func = project_to_2d_linear if args.linear_projection else project_to_2d
					reconstruction_semi = projection_func(predicted_semi + predicted_traj_cat[split_idx:], cam_semi)

					loss_reconstruction = mpjpe(reconstruction_semi, target_semi)  # On 2D poses
					epoch_loss_2d_train_unlabeled += predicted_semi.shape[0] * predicted_semi.shape[
						1] * loss_reconstruction.item()
					if not args.no_proj:
						loss_total += loss_reconstruction

					# Bone length term to enforce kinematic constraints
					if args.bone_length_term:
						dists = predicted_3d_pos_cat[:, :, 1:] - predicted_3d_pos_cat[:, :,
						                                         dataset.skeleton().parents()[1:]]
						bone_lengths = torch.mean(torch.norm(dists, dim=3), dim=1)
						penalty = torch.mean(torch.abs(torch.mean(bone_lengths[:split_idx], dim=0) \
						                               - torch.mean(bone_lengths[split_idx:], dim=0)))
						loss_total += penalty

					N_semi += predicted_semi.shape[0] * predicted_semi.shape[1]
				else:
					N_semi += 1  # To avoid division by zero

				loss_total.backward()

				optimizer.step()
			losses_traj_train.append(epoch_loss_traj_train / N)
			losses_2d_train_unlabeled.append(epoch_loss_2d_train_unlabeled / N_semi)
		else:
			# Regular supervised scenario
			for _, batch_3d, batch_2d in train_generator.next_epoch():
				inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
				inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
				if torch.cuda.is_available():
					inputs_3d = inputs_3d.cuda()
					inputs_2d = inputs_2d.cuda()
				inputs_3d[:, :, 0] = 0

				optimizer.zero_grad()

				# Predict 3D poses
				predicted_3d_pos = model_pos_train(inputs_2d)
				loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
				epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
				N += inputs_3d.shape[0] * inputs_3d.shape[1]

				loss_total = loss_3d_pos
				loss_total.backward()

				optimizer.step()

		losses_3d_train.append(epoch_loss_3d_train / N)

		# End-of-epoch evaluation
		with torch.no_grad():
			model_pos.load_state_dict(model_pos_train.state_dict())
			model_pos.eval()
			if semi_supervised:
				model_traj.load_state_dict(model_traj_train.state_dict())
				model_traj.eval()

			epoch_loss_3d_valid = 0
			epoch_loss_traj_valid = 0
			epoch_loss_2d_valid = 0
			N = 0

			if not args.no_eval:
				# Evaluate on test set
				for cam, batch, batch_2d in test_generator.next_epoch():
					inputs_3d = torch.from_numpy(batch.astype('float32'))
					inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
					if torch.cuda.is_available():
						inputs_3d = inputs_3d.cuda()
						inputs_2d = inputs_2d.cuda()
					inputs_traj = inputs_3d[:, :, :1].clone()
					inputs_3d[:, :, 0] = 0

					# Predict 3D poses
					predicted_3d_pos = model_pos(inputs_2d)
					loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
					epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
					N += inputs_3d.shape[0] * inputs_3d.shape[1]

					if semi_supervised:
						cam = torch.from_numpy(cam.astype('float32'))
						if torch.cuda.is_available():
							cam = cam.cuda()

						predicted_traj = model_traj(inputs_2d)
						loss_traj = mpjpe(predicted_traj, inputs_traj)
						epoch_loss_traj_valid += inputs_traj.shape[0] * inputs_traj.shape[1] * loss_traj.item()
						assert inputs_traj.shape[0] * inputs_traj.shape[1] == inputs_3d.shape[0] * inputs_3d.shape[1]

						if pad > 0:
							target = inputs_2d[:, pad:-pad, :, :2].contiguous()
						else:
							target = inputs_2d[:, :, :, :2].contiguous()
						reconstruction = project_to_2d(predicted_3d_pos + predicted_traj, cam)
						loss_reconstruction = mpjpe(reconstruction, target)  # On 2D poses
						epoch_loss_2d_valid += reconstruction.shape[0] * reconstruction.shape[
							1] * loss_reconstruction.item()
						assert reconstruction.shape[0] * reconstruction.shape[1] == inputs_3d.shape[0] * \
						       inputs_3d.shape[1]

				losses_3d_valid.append(epoch_loss_3d_valid / N)
				if semi_supervised:
					losses_traj_valid.append(epoch_loss_traj_valid / N)
					losses_2d_valid.append(epoch_loss_2d_valid / N)

				# Evaluate on training set, this time in evaluation mode
				epoch_loss_3d_train_eval = 0
				epoch_loss_traj_train_eval = 0
				epoch_loss_2d_train_labeled_eval = 0
				N = 0
				for cam, batch, batch_2d in train_generator_eval.next_epoch():
					if batch_2d.shape[1] == 0:
						# This can only happen when downsampling the dataset
						continue

					inputs_3d = torch.from_numpy(batch.astype('float32'))
					inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
					if torch.cuda.is_available():
						inputs_3d = inputs_3d.cuda()
						inputs_2d = inputs_2d.cuda()
					inputs_traj = inputs_3d[:, :, :1].clone()
					inputs_3d[:, :, 0] = 0

					# Compute 3D poses
					predicted_3d_pos = model_pos(inputs_2d)
					loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
					epoch_loss_3d_train_eval += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
					N += inputs_3d.shape[0] * inputs_3d.shape[1]

					if semi_supervised:
						cam = torch.from_numpy(cam.astype('float32'))
						if torch.cuda.is_available():
							cam = cam.cuda()
						predicted_traj = model_traj(inputs_2d)
						loss_traj = mpjpe(predicted_traj, inputs_traj)
						epoch_loss_traj_train_eval += inputs_traj.shape[0] * inputs_traj.shape[1] * loss_traj.item()
						assert inputs_traj.shape[0] * inputs_traj.shape[1] == inputs_3d.shape[0] * inputs_3d.shape[1]

						if pad > 0:
							target = inputs_2d[:, pad:-pad, :, :2].contiguous()
						else:
							target = inputs_2d[:, :, :, :2].contiguous()
						reconstruction = project_to_2d(predicted_3d_pos + predicted_traj, cam)
						loss_reconstruction = mpjpe(reconstruction, target)
						epoch_loss_2d_train_labeled_eval += reconstruction.shape[0] * reconstruction.shape[
							1] * loss_reconstruction.item()
						assert reconstruction.shape[0] * reconstruction.shape[1] == inputs_3d.shape[0] * \
						       inputs_3d.shape[1]

				losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)
				if semi_supervised:
					losses_traj_train_eval.append(epoch_loss_traj_train_eval / N)
					losses_2d_train_labeled_eval.append(epoch_loss_2d_train_labeled_eval / N)

				# Evaluate 2D loss on unlabeled training set (in evaluation mode)
				epoch_loss_2d_train_unlabeled_eval = 0
				N_semi = 0
				if semi_supervised:
					for cam, _, batch_2d in semi_generator_eval.next_epoch():
						cam = torch.from_numpy(cam.astype('float32'))
						inputs_2d_semi = torch.from_numpy(batch_2d.astype('float32'))
						if torch.cuda.is_available():
							cam = cam.cuda()
							inputs_2d_semi = inputs_2d_semi.cuda()

						predicted_3d_pos_semi = model_pos(inputs_2d_semi)
						predicted_traj_semi = model_traj(inputs_2d_semi)
						if pad > 0:
							target_semi = inputs_2d_semi[:, pad:-pad, :, :2].contiguous()
						else:
							target_semi = inputs_2d_semi[:, :, :, :2].contiguous()
						reconstruction_semi = project_to_2d(predicted_3d_pos_semi + predicted_traj_semi, cam)
						loss_reconstruction_semi = mpjpe(reconstruction_semi, target_semi)

						epoch_loss_2d_train_unlabeled_eval += reconstruction_semi.shape[0] * reconstruction_semi.shape[
							1] \
						                                      * loss_reconstruction_semi.item()
						N_semi += reconstruction_semi.shape[0] * reconstruction_semi.shape[1]
					losses_2d_train_unlabeled_eval.append(epoch_loss_2d_train_unlabeled_eval / N_semi)

		elapsed = (time() - start_time) / 60

		if args.no_eval:
			print('[%d] time %.2f lr %f 3d_train %f' % (
				epoch + 1,
				elapsed,
				lr,
				losses_3d_train[-1] * 1000))
		else:
			if semi_supervised:
				print('[%d] time %.2f lr %f 3d_train %f 3d_eval %f traj_eval %f 3d_valid %f '
				      'traj_valid %f 2d_train_sup %f 2d_train_unsup %f 2d_valid %f' % (
					      epoch + 1,
					      elapsed,
					      lr,
					      losses_3d_train[-1] * 1000,
					      losses_3d_train_eval[-1] * 1000,
					      losses_traj_train_eval[-1] * 1000,
					      losses_3d_valid[-1] * 1000,
					      losses_traj_valid[-1] * 1000,
					      losses_2d_train_labeled_eval[-1],
					      losses_2d_train_unlabeled_eval[-1],
					      losses_2d_valid[-1]))
			else:
				print('[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f' % (
					epoch + 1,
					elapsed,
					lr,
					losses_3d_train[-1] * 1000,
					losses_3d_train_eval[-1] * 1000,
					losses_3d_valid[-1] * 1000))

		# Decay learning rate exponentially
		lr *= lr_decay
		for param_group in optimizer.param_groups:
			param_group['lr'] *= lr_decay
		epoch += 1

		# Decay BatchNorm momentum
		momentum = initial_momentum * np.exp(-epoch / args.epochs * np.log(initial_momentum / final_momentum))
		model_pos_train.set_bn_momentum(momentum)
		if semi_supervised:
			model_traj_train.set_bn_momentum(momentum)

		# Save checkpoint if necessary
		if epoch % args.checkpoint_frequency == 0:
			chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
			print('Saving checkpoint to', chk_path)

			torch.save({
				'epoch': epoch,
				'lr': lr,
				'random_state': train_generator.random_state(),
				'optimizer': optimizer.state_dict(),
				'model_pos': model_pos_train.state_dict(),
				'model_traj': model_traj_train.state_dict() if semi_supervised else None,
				'random_state_semi': semi_generator.random_state() if semi_supervised else None,
			}, chk_path)

		# Save training curves after every epoch, as .png images (if requested)
		if args.export_training_curves and epoch > 3:
			if 'matplotlib' not in sys.modules:
				import matplotlib

				matplotlib.use('Agg')
				import matplotlib.pyplot as plt

			plt.figure()
			epoch_x = np.arange(3, len(losses_3d_train)) + 1
			plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
			plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
			plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
			plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
			plt.ylabel('MPJPE (m)')
			plt.xlabel('Epoch')
			plt.xlim((3, epoch))
			plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))

			if semi_supervised:
				plt.figure()
				plt.plot(epoch_x, losses_traj_train[3:], '--', color='C0')
				plt.plot(epoch_x, losses_traj_train_eval[3:], color='C0')
				plt.plot(epoch_x, losses_traj_valid[3:], color='C1')
				plt.legend(['traj. train', 'traj. train (eval)', 'traj. valid (eval)'])
				plt.ylabel('Mean distance (m)')
				plt.xlabel('Epoch')
				plt.xlim((3, epoch))
				plt.savefig(os.path.join(args.checkpoint, 'loss_traj.png'))

				plt.figure()
				plt.plot(epoch_x, losses_2d_train_labeled_eval[3:], color='C0')
				plt.plot(epoch_x, losses_2d_train_unlabeled[3:], '--', color='C1')
				plt.plot(epoch_x, losses_2d_train_unlabeled_eval[3:], color='C1')
				plt.plot(epoch_x, losses_2d_valid[3:], color='C2')
				plt.legend(
					['2d train labeled (eval)', '2d train unlabeled', '2d train unlabeled (eval)', '2d valid (eval)'])
				plt.ylabel('MPJPE (2D)')
				plt.xlabel('Epoch')
				plt.xlim((3, epoch))
				plt.savefig(os.path.join(args.checkpoint, 'loss_2d.png'))
			plt.close('all')


# Evaluate
def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False):
	epoch_loss_3d_pos = 0
	epoch_loss_3d_pos_procrustes = 0
	epoch_loss_3d_pos_scale = 0
	epoch_loss_3d_vel = 0
	with torch.no_grad():
		if not use_trajectory_model:
			model_pos.eval()
		else:
			model_traj.eval()
		N = 0
		for _, batch, batch_2d in test_generator.next_epoch():
			inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
			if torch.cuda.is_available():
				inputs_2d = inputs_2d.cuda()

			# Positional model
			if not use_trajectory_model:
				predicted_3d_pos = model_pos(inputs_2d)
			else:
				predicted_3d_pos = model_traj(inputs_2d)

			# Test-time augmentation (if enabled)
			if test_generator.augment_enabled():
				# Undo flipping and take average with non-flipped version
				predicted_3d_pos[1, :, :, 0] *= -1
				if not use_trajectory_model:
					predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :,
					                                                     joints_right + joints_left]
				predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

			if return_predictions:
				return predicted_3d_pos.squeeze(0).cpu().numpy()

			inputs_3d = torch.from_numpy(batch.astype('float32'))
			if torch.cuda.is_available():
				inputs_3d = inputs_3d.cuda()
			inputs_3d[:, :, 0] = 0
			if test_generator.augment_enabled():
				inputs_3d = inputs_3d[:1]

			error = mpjpe(predicted_3d_pos, inputs_3d)
			epoch_loss_3d_pos_scale += inputs_3d.shape[0] * inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos,
			                                                                             inputs_3d).item()

			epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
			N += inputs_3d.shape[0] * inputs_3d.shape[1]

			inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
			predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

			epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

			# Compute velocity error
			epoch_loss_3d_vel += inputs_3d.shape[0] * inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

	if action is None:
		print('----------')
	else:
		print('----' + action + '----')
	e1 = (epoch_loss_3d_pos / N) * 1000
	e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
	e3 = (epoch_loss_3d_pos_scale / N) * 1000
	ev = (epoch_loss_3d_vel / N) * 1000
	print('Test time augmentation:', test_generator.augment_enabled())
	print('Protocol #1 Error (MPJPE):', e1, 'mm')
	print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
	print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
	print('Velocity Error (MPJVE):', ev, 'mm')
	print('----------')

	return e1, e2, e3, ev


if args.render:
	print('Rendering...')

	input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
	ground_truth = None
	if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
		if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
			ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
	if ground_truth is None:
		print('INFO: this action is unlabeled. Ground truth will not be rendered.')

	gen = UnchunkedGenerator(None, None, [input_keypoints],
	                         pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
	                         kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
	prediction = evaluate(gen, return_predictions=True)
	prediction[:, 16, :] = 0  # set the right wrist = 0 (red color) # for debug
	if model_traj is not None and ground_truth is None:
		prediction_traj = evaluate(gen, return_predictions=True, use_trajectory_model=True)
		prediction += prediction_traj

	if args.viz_export is not None:
		print('Exporting joint positions to', os.path.abspath(args.viz_export + '.npy'))
		# Predictions are in camera space
		np.save(args.viz_export, prediction)
	# if you also want to generated mp4. just comment exit(0)
	# exit(0)

	if args.viz_output is not None:
		print(args.viz_output)
		if ground_truth is not None:
			# Reapply trajectory
			trajectory = ground_truth[:, :1]
			ground_truth[:, 1:] += trajectory
			prediction += trajectory

		# Invert camera transformation
		cam = dataset.cameras()[args.viz_subject][args.viz_camera]
		if ground_truth is not None:
			prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
			ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
		else:
			# If the ground truth is not available, take the camera extrinsic params from a random subject.
			# They are almost the same, and anyway, we only need this for visualization purposes.
			for subject in dataset.cameras():
				if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
					rot = dataset.cameras()[subject][args.viz_camera]['orientation']
					break
			prediction = camera_to_world(prediction, R=rot, t=0)
			# We don't have the trajectory, but at least we can rebase the height
			prediction[:, :, 2] -= np.min(prediction[:, :, 2])

		anim_output = {'Reconstruction': prediction}
		if ground_truth is not None and not args.viz_no_ground_truth:
			anim_output['Ground truth'] = ground_truth

		input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

		from common.visualization import render_animation

		render_animation(input_keypoints, keypoints_metadata, anim_output,
		                 dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
		                 limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
		                 input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
		                 input_video_skip=args.viz_skip)
		print(f'out_video: {args.viz_output}')
else:
	print('Evaluating...')
	all_actions = {}
	all_actions_by_subject = {}
	for subject in subjects_test:
		if subject not in all_actions_by_subject:
			all_actions_by_subject[subject] = {}

		for action in dataset[subject].keys():
			action_name = action.split(' ')[0]
			if action_name not in all_actions:
				all_actions[action_name] = []
			if action_name not in all_actions_by_subject[subject]:
				all_actions_by_subject[subject][action_name] = []
			all_actions[action_name].append((subject, action))
			all_actions_by_subject[subject][action_name].append((subject, action))


	def fetch_actions(actions):
		out_poses_3d = []
		out_poses_2d = []

		for subject, action in actions:
			poses_2d = keypoints[subject][action]
			for i in range(len(poses_2d)):  # Iterate across cameras
				out_poses_2d.append(poses_2d[i])

			poses_3d = dataset[subject][action]['positions_3d']
			assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
			for i in range(len(poses_3d)):  # Iterate across cameras
				out_poses_3d.append(poses_3d[i])

		stride = args.downsample
		if stride > 1:
			# Downsample as requested
			for i in range(len(out_poses_2d)):
				out_poses_2d[i] = out_poses_2d[i][::stride]
				if out_poses_3d is not None:
					out_poses_3d[i] = out_poses_3d[i][::stride]

		return out_poses_3d, out_poses_2d


	def run_evaluation(actions, action_filter=None):
		errors_p1 = []
		errors_p2 = []
		errors_p3 = []
		errors_vel = []

		for action_key in actions.keys():
			if action_filter is not None:
				found = False
				for a in action_filter:
					if action_key.startswith(a):
						found = True
						break
				if not found:
					continue

			poses_act, poses_2d_act = fetch_actions(actions[action_key])
			gen = UnchunkedGenerator(None, poses_act, poses_2d_act,
			                         pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
			                         kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
			                         joints_right=joints_right)
			e1, e2, e3, ev = evaluate(gen, action_key)
			errors_p1.append(e1)
			errors_p2.append(e2)
			errors_p3.append(e3)
			errors_vel.append(ev)

		print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
		print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
		print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
		print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')


	if not args.by_subject:
		run_evaluation(all_actions, action_filter)
	else:
		for subject in all_actions_by_subject.keys():
			print('Evaluating on subject', subject)
			run_evaluation(all_actions_by_subject[subject], action_filter)
			print('')

print('finished!')
