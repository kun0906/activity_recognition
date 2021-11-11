""" Using the 'cnn' library (or directory) to get the CNN features
	run the below command under 'activity_recognition' folder (root folder):
		PYTHONPATH=. python3 ar/features/cnn_feature.py

"""
import glob
import os

import numpy as np

from ar.features.cnn.model_tf import CNN_tf
from ar.features.cnn.utils import load_video
from ar.utils.utils import timer


def _extract_video_feature(model, in_file, out_dir):
	# in_file = 'data/data-clean/refrigerator/open_close_fridge/1/open_close_fridge_3_1615392727_2.mkv'
	video_name = os.path.splitext(os.path.basename(in_file))[0]
	out_file = os.path.join(out_dir, '{}_{}.npy'.format(video_name, model.net_name))
	if os.path.exists(out_file):
		return out_file

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	batch_sz = 32
	# sampling: only extract the first frame in each second from the video.
	video_tensor = load_video(in_file, model.desired_size)
	# extract features
	features = model.extract(video_tensor, batch_sz)
	# save features
	np.save(os.path.splitext(out_file)[0], features)

	return out_file


@timer
def gen_cnn_based_features(in_dir='data/data-clean/refrigerator',
                           out_dir='examples/classical_ml/out/cnn_based_features',
                           device_type='refrigerator'):
	"""

	Parameters
	----------
	in_dir:  ['data/data-clean/refrigerator]
	out_dir:

	Returns
	-------
		meta: dictionary
	"""
	# deep neural network model
	model_file = './ar/features/cnn/slim/vgg_16.ckpt'
	model = CNN_tf('vgg', model_file)

	for camera_type in ['_1.mp4', '_2.mkv', '_3.mp4']:
		files = glob.glob(f'{in_dir}/data-clean/{device_type}/*/*/*{camera_type}')
		for i, f in enumerate(files):
			print(f'{i}/{len(files)}')
			sub_dir = os.path.relpath(f, in_dir)
			out_dir_tmp = os.path.join(out_dir, os.path.dirname(sub_dir))
			cnn_feature_file = _extract_video_feature(model, f, out_dir=out_dir_tmp)
			print(cnn_feature_file)


# data = []  # [(video_path, cnn_feature, y)]
#
# # list device folders (e.g., refrigerator or camera)
# i = 0
# cnt_3 = 0  # camera_3
# cnt_32 = 0  # camera_32: backup
# for device_dir in sorted(in_dir):
# 	out_dir_sub = ''
# 	if device_type not in device_dir: continue
# 	# list activity folders (e.g., open_close or take_out )
# 	for activity_dir in sorted(os.listdir(device_dir)):
# 		activity_label = activity_dir
# 		out_dir_activity = activity_dir
# 		activity_dir = os.path.join(device_dir, activity_dir)
# 		if not os.path.exists(activity_dir) or '.DS_Store' in activity_dir or not os.path.isdir(
# 				activity_dir): continue
# 		# list participant folders (e.g., participant 1 or participant 2)
# 		for participant_dir in sorted(os.listdir(activity_dir)):
# 			out_dir_participant = participant_dir
# 			out_dir_sub = os.path.join(participant_dir)
# 			participant_dir = os.path.join(activity_dir, participant_dir)
# 			if not os.path.exists(participant_dir) or '.DS_Store' in participant_dir: continue
# 			# print(participant_dir)
# 			# list videos (e.g., 'no_interaction_1_1614038765_1.mp4')
# 			for f in sorted(os.listdir(participant_dir)):
# 				if f.startswith('.'): continue
# 				if ('mp4' not in f) and ('mkv' not in f): continue  # only process video file.
# 				x = os.path.join(participant_dir, f)
# 				if '_3 2.mp4' in f:  # ignore _3 2.mp4 data.
# 					cnt_32 += 1
# 					continue
# 				print(f'i: {i}, {x}')
# 				try:
# 					out_dir_tmp = os.path.join(out_dir, out_dir_activity, out_dir_participant)
# 					cnn_feature_file = _extract_video_feature(model, x, out_dir=out_dir_tmp)
# 					print(cnn_feature_file)
# 				except Exception as e:
# 					msg = f'error: {e} on {x}'
# 					raise ValueError(msg)


if __name__ == '__main__':
	in_dir = 'examples/datasets'
	out_dir = 'examples/classical_ml/out/cnn_based_features'
	gen_cnn_based_features(in_dir, out_dir, device_type='refrigerator')
