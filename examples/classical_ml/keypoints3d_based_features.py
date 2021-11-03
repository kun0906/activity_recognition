"""

"""
import copy
import glob
import os
import shutil
from collections import defaultdict, Counter
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sliced import SlicedInverseRegression

from ar.features.feature import _get_fft
from ar.utils.utils import make_confusion_matrix, load, dump, check_path, timer
from examples.classical_ml.misc.fft_feature_inverse_regression_users import pca_plot2

# 2d keypoints output format
# https://github.com/facebookresearch/VideoPose3D/issues/57
coco_keypoints = [
	"nose",
	"left_eye",
	"right_eye",
	"left_ear",
	"right_ear",
	"left_shoulder",
	"right_shoulder",
	"left_elbow",
	"right_elbow",
	"left_wrist",
	"right_wrist",
	"left_hip",
	"right_hip",
	"left_knee",
	"right_knee",
	"left_ankle",
	"Right_ankle"]


# 3d keypoints output format
# The inputs are in COCO layout, the outputs are in Human3.6M.
class Keypoints3d:
	HIP = 0
	R_HIP = 1
	R_KNEE = 2
	R_FOOT = 3
	L_HIP = 4
	L_KNEE = 5
	L_FOOT = 6
	SPINE = 7
	THORAX = 8
	NOSE = 9
	HEAD = 10
	L_SHOULDER = 11
	L_ELBOW = 12
	L_WRIST = 13
	R_SHOULDER = 14
	R_ELBOW = 15
	R_WRIST = 16


keypoints = [
	'HIP',
	'R_HIP',
	'R_KNEE',
	'R_FOOT',
	'L_HIP',
	'L_KNEE',
	'L_FOOT',
	'SPINE',
	'THORAX',
	'NOSE',
	'HEAD',
	'L_SHOULDER',
	'L_ELBOW',
	'L_WRIST',
	'R_SHOULDER',
	'R_ELBOW',
	'R_WRIST'
]
features_names = []
for i, k in enumerate(keypoints):
	for x in ['x', 'y', 'z']:
		features_names += [f'{i}_{k}_{x}_{v}' for v in ['mean', 'std', 'skew', 'kurtosis', 'min', 'max']]
# [np.mean(data), np.std(data), skew(data), kurtosis(data), np.min(data), np.max(data)]


def get_missclassified(cm, test_y, pred_y, test_raw_files, model_name, idx2label):
	try:
		shutil.rmtree(f'examples/classical_ml/out/misclassified/{model_name}')
	except Exception as e:
		print(e)
	res = defaultdict(list)
	# n, m = cm.shape
	for i, (y_, y_2) in enumerate(zip(test_y, pred_y)):
		if y_ != y_2:
			# print(y_, y_2, test_raw_files[i])
			in_file = os.path.relpath(test_raw_files[i], 'examples/classical_ml/out/keypoints3d-20210907/keypoints3d/')
			# tmp_file = os.path.relpath(in_file, f'data/data-clean/refrigerator/{idx2label[y_]}')
			out_file = f'examples/classical_ml/out/misclassified/{model_name}/{idx2label[y_]}->/{idx2label[y_2]}/{in_file}'[
			           :-4]
			check_path(out_file)
			try:
				# copyfile(os.path.join('examples', in_file)[:-4], out_file)
				copyfile(os.path.join('ar/features/videopose3d/out3d_pred', in_file)[:-4], out_file)
			except Exception as e:
				print(f'Error: {e}')
			res[y_].append(test_raw_files[i])

	res = sorted(res.items(), key=lambda x: x[0], reverse=False)
	for vs in res:
		label, lst = vs
		print(f'label: {label}, misclassified_num: {len(lst)}, {lst}')
	return


def _get_fft2(vs, fft_type='magnitude'):
	out = np.fft.fft2(vs)
	if fft_type == 'phase':
		out = np.angle(out)  # phase
	else:
		# fft_type =='magnitude':
		out = np.abs(out)
	n, _ = out.shape
	d = 51
	if n % 2 == 0:
		n = 1 + (n - 1) // 2 + 1
	else:
		n = 1 + (n - 1) // 2
	out = np.sort(out[:n, :1 + (d - 1) // 2], axis=1)  # [:, -2:]  # return the top 2 for each row
	return list(out.flatten())


def trim(X):
	diff = X[1:] - X[:-1]
	n = len(diff)
	start = 0
	end = n
	thres = 1e-5
	for i in range(1, n // 2):
		if abs(diff[i]) < thres:
			start = i
	for i in range(n - 1, n // 2, -1):
		if abs(diff[i]) > thres:
			end = i
			break
	print(f'n: {n}, (start: {start}, end: {end}), {n - (end - start)}')
	return start, end


def stats(d, dim=3):
	tmp = []
	for i in range(dim):
		tmp += [np.mean(d[:, i]), np.std(d[:, i]), np.min(d[:, i]), np.max(d[:, i]), np.max(d[:, i]) - np.min(d[:, i]),
		        skew(d[:, i]), kurtosis(d[:, i]), np.sum(d[:, i])] + list(
			np.quantile(d[:, i], q=[v * 0.01 for v in range(0, 100, 10)]))
	# tmp += list(np.quantile(d[:, i], q=[ v *0.01 for v in range(0, 100, 10)]))
	# tmp +=[np.mean(d[:, i]), np.std(d[:, i]), np.min(d[:, i]), np.max(d[:, i]), skew(d[:,i]), kurtosis(d[:, i])]
	# tmp += d[:70, i].tolist()
	# tmp += [np.mean(d[:, i]), np.std(d[:, i])]
	# tmp += [skew(d[:, i]), kurtosis(d[:, i])]
	return tmp


def stats1(d):
	tmp = stats(d[:, 0:1] - d[:, 1:2], dim=1) + stats(d[:, 1:2] - d[:, 2:3], dim=1) + stats(d[:, 2:3] - d[:, 0:1],
	                                                                                        dim=1)
	return tmp


def get_fft_features(raw_data='', m=84, keypoint=7, file=''):
	""" Get features (fft or stats)

	Parameters
	----------
	npy_file
	m:
	   without trimmming: m = 84
	   with trimming: m = 51
	keypoint
	coordinate

	Returns
	-------

	"""
	# print(file)

	flg = 'hand_direction1'
	if flg == 'hand_direction':
		# start, end = trim(raw_data[:, -3])
		x, y = 0, 3
		head = raw_data[:, 10, x:y]
		left_shoulder = raw_data[:, 11, x:y]
		right_shoulder = raw_data[:, 14, x:y]
		left_elbow = raw_data[:, 12, x:y]
		right_elbow = raw_data[:, 15, x:y]
		left_wrist = raw_data[:, 13, x:y]
		right_wrist = raw_data[:, 16, x:y]

		d = right_wrist - left_wrist
		d2 = right_shoulder - left_shoulder
		d3 = right_elbow - left_elbow

		d11 = right_wrist[1:, :] - right_wrist[:-1, :]
		d22 = right_shoulder[1:, :] - right_shoulder[:-1, :]
		d33 = right_elbow[1:, :] - right_elbow[:-1, :]

		d111 = left_wrist[1:, :] - left_wrist[:-1, :]
		d222 = left_shoulder[1:, :] - left_shoulder[:-1, :]
		d333 = left_elbow[1:, :] - left_elbow[:-1, :]

		# res = [np.mean(d), np.std(d), np.min(d), np.max(d)] + [np.mean(d2), np.std(d2), np.min(d2), np.max(d2)]
		# res = stats(head) + stats(d) + stats(d2) + stats(d3)  #+ stats(d11)  + stats(d22) + stats(d33) + stats(d111)  + stats(d222) + stats(d333)
		# res = stats(d**2) + stats(d2**2) + stats(d3**2) + stats(d11**2) + stats(d22**2) + stats(d33**2) + stats(d111**2) + stats(d222**2) + stats(d333**2)
		p = 3
		# res = stats(d ** p) + stats(d2 ** p) + stats(d3 ** p) + stats(d11 ** p) + stats(d22 ** p) + stats(d33 ** p) + stats(d111 ** p) + stats(d222 ** p) + stats(d333 ** p)
		res = [len(raw_data)]
		# data = np.sum(raw_data, axis=2)
		# res += np.mean(data, axis=0).tolist() + np.std(data, axis=0).tolist() +  np.min(data, axis=0).tolist() + np.max(data, axis=0).tolist()
		# data = np.sum(raw_data, axis=1)
		# res += np.mean(data, axis=0).tolist() + np.std(data, axis=0).tolist() + np.min(data, axis=0).tolist() + np.max(
		# 	data, axis=0).tolist()
		for i in range(11, 17):
			# 	# res += stats(raw_data[:, i, x:y], dim=1) #+ stats(raw_data[1:, i, x:y]-raw_data[:-1, i, x:y])
			res += stats(raw_data[:, i, x:y])
			# for s in range(1, 50, 5):
			# 	# res+= stats(raw_data[::s, i, x:y])
			# 	res += stats(raw_data[s:, i, x:y] - raw_data[:-s, i, x:y])
			for j in range(i + 1, 17):
				# res += stats(raw_data[:, j, x:y]-raw_data[0:1, j, x:y])
				if i == j: continue
				res += stats((raw_data[:, i, x:y] - raw_data[:, j, x:y]) ** 1)
		# # # 	# res += stats(raw_data[:, j, x:y] - raw_data[:, i, x:y])

		# res = stats(right_wrist) +stats(left_wrist) + stats(d) + stats(d2) + stats(d3) #+ stats(d11)  + stats(d22) + stats(d33) + stats(d111)  + stats(d222) + stats(d333)
		# res = [np.max(right_wrist) - np.max(left_wrist), np.max(right_elbow) - np.max(left_elbow)]
		# res = np.concatenate([np.max(np.abs(right_wrist), axis=0), np.max(np.abs(left_wrist), axis=0)], axis=0)
		# t = m
		# # res = np.max(right_wrist[t:, :]-right_wrist[:-t, :], axis=0).tolist() + np.max(right_elbow[t:, :]-right_elbow[:-t, :], axis=0).tolist()
		# d = right_wrist[t:, :] - right_wrist[:-t, :]
		# d2 = right_elbow[t:, :] - right_elbow[:-t, :]
		# res = [np.mean(d), np.std(d), np.mean(d2), np.std(d2)]
		# res = (np.max(right_wrist, axis=0)- np.min(right_wrist, axis=0)).tolist()
		# (np.max(right_elbow, axis=0) - np.min(right_elbow, axis=0)).tolist() + \
		# (np.max(right_shoulder, axis=0) - np.min(right_shoulder, axis=0)).tolist()
		# (np.max(left_wrist, axis=0) - np.min(left_wrist, axis=0)).tolist() + \
		# (np.max(left_elbow, axis=0) - np.min(left_elbow, axis=0)).tolist() + \
		# (np.max(left_shoulder, axis=0) - np.min(left_shoulder, axis=0)).tolist()

		feature_name = []
		return res

	n = raw_data.shape[0]
	# only right keypoints
	# raw_data = raw_data[:, [11, 12, 13, 14, 15, 16], :].reshape((n, -1))
	raw_data = raw_data[:, :, :].reshape((n, -1))
	# raw_data = raw_data[:, [14, 16], :].reshape((n, -1))

	# raw_data = raw_data.reshape((n, 51))
	res = []

	for i in range(raw_data.shape[1]):
		data = raw_data[:, i]
		# data = raw_data[:, i]
		# # data = data[1:] - data[:-1] # difference
		# # print(data[:10], data[-10:])
		# data = data[start: end]
		flg = 'stats1'
		if flg == 'fft':
			fft_features = _get_fft(data, fft_bin=m)
			fft_features = fft_features[0:1 + int(np.ceil((m - 1) / 2))]  # half of fft values
		elif flg == 'std':
			fft_features = [np.mean(data), np.std(data)]
			# fft_features = list(np.quantile(data, q=[0, 0.25, 0.5, 0.75, 1]))
			# fft_features = list(np.quantile(data, q = [0, 0.25, 0.5, 0.75, 1])) + [np.mean(data), np.std(data)]
			fft_features = list(np.quantile(data, q=[v * 0.01 for v in range(0, 100, 5)])) + [np.mean(data),
			                                                                                  np.std(data), skew(data),
			                                                                                  kurtosis(data),
			                                                                                  np.min(data),
			                                                                                  np.max(data)]
		elif flg == 'stats':
			# fft_features = [np.min(data), np.max(data)]
			# fft_features = [np.mean(data), np.std(data)]
			# fft_features = [skew(data)]
			# fft_features = [np.std(data)]
			# fft_features = [np.min(data), np.mean(data), np.std(data),  np.max(data)]
			fft_features = [np.min(data), np.max(data), np.mean(data), np.std(data), skew(data), kurtosis(data)]
		elif flg == 'hand_direction':
			fft_features = [np.max(data) - np.min(data), np.mean(data), np.std(data)]
		elif flg == 'top10':
			fft_features = sorted(data, reverse=True)[:50]
		# fft_features = sorted(data, key=lambda x:abs(x), reverse=True)[:50]
		else:
			n = len(data)
			step = int(np.ceil(n / m))
			fft_features = []
			for i in range(0, len(data), step):
				vs = data[i:i + step]
				flg2 = 'stats'  # stats (per window)
				if flg2 == 'stats':
					# tmp = list(np.quantile(vs, q = [0, 0.5, 1] )) # [0, 0.25, 0.5, 0.75, 1]+ [np.mean(vs), np.std(vs)]
					# tmp = list(np.quantile(vs, q=[0, 0.5, 1]))
					tmp = [np.mean(data), np.std(data), skew(data), kurtosis(data), np.min(data), np.max(data)]
					# tmp = [np.mean(vs)]
					n_feat = len(tmp)
				elif flg2:
					tmp = _get_fft(vs)
					tmp = sorted(tmp[0:1 + int(np.ceil((m - 1) / 2))], reverse=True)
					n_feat = 2
				fft_features.extend(tmp[:n_feat])
			fft_features = fft_features + [0] * (n_feat * m - len(fft_features))
		res.append(fft_features)
	return np.asarray(res).reshape(-1, )


@timer
def _get_data(root_dir='examples/out/keypoints3d-20210907/keypoints3d/data', camera_type='_1.mp4', classes=[], m=10):
	"""

	Parameters
	----------
	root_dir: input directory
	camera_type: camera type
	classes: activities
	m: number of windows for FFT or STATS(per windows)

	Returns
	-------
	dataset:
	users:
	activities:
	raw_files: list of file paths
	"""
	dataset = []
	users = []
	activities = []
	raw_files = []
	duration = []
	for act in classes:
		files = glob.glob(f'{root_dir}/data-clean/refrigerator/' + act + f'/*/*{camera_type}.npy')
		# files = [f'{root_dir}/data-clean/refrigerator/take_out_item/4/take_out_item_2_1616179391_1.mp4.npy',
		#          f'{root_dir}/data-clean/refrigerator/take_out_item/4/take_out_item_2_1616179391_1.mp4.npy']
		# print(f'***{camera_type}: {len(files)}')
		for file in files:
			# print('Processing file', file)
			user = int(file.split('/')[-2])

			data = []
			raw_data = np.load(file)
			duration.append(len(raw_data))
			# if camera_type == '_1.mp4':
			# 	fps = 10
			# 	min_frames = 286  # number of frames
			# 	max_frames = 477  # number of frames
			# 	if raw_data.shape[0] < min_frames or raw_data.shape[0] > max_frames:
			# 		print(len(raw_data), file)
			# 		continue
			# elif camera_type == '_2.mkv':  # fps = 30
			# 	fps = 30
			# 	min_frames = 466  # number of frames
			# 	max_frames = 708  # number of frames
			# 	if raw_data.shape[0] < min_frames or raw_data.shape[0] > max_frames:
			# 		print(len(raw_data), file)
			# 		continue
			# elif camera_type == '_3.mp4':  # fps = 3
			# 	min_frames = 76  # number of frames
			# 	max_frames = 95  # number of frames
			# 	if raw_data.shape[0] < min_frames or raw_data.shape[0] > max_frames:
			# 		print(len(raw_data), file)
			# 		continue

			# for keypoint_idx, _ in enumerate(keypoints):
			# 	tmp = get_fft_features(raw_data, keypoint=keypoint_idx)
			# 	# data = special_keypoints(np.load(file))
			# 	data.extend(list(tmp))
			data = get_fft_features(raw_data, m=m, file=file)
			data = np.asarray(data)

			dataset.append(data)
			users.append(user)
			activities.append(classes.index(act))
			raw_files.append(file)
		print(act, len(files))
	print(f'***{camera_type}, n_frames: {np.quantile(duration, q=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1])}\n')

	return np.array(dataset), np.array(users), np.array(activities), raw_files


def pca_plot(dataset, activities, classes, title):
	from sklearn.decomposition import PCA
	pca = PCA(n_components=2)
	X = pca.fit_transform(dataset)
	for i in range(5):
		plt.scatter(X[activities == i, 0], X[activities == i, 1])
		plt.title(title)
	plt.legend(classes)
	plt.show()


def split_train_test(history, data_type='', random_state=42):
	if data_type in ['_1.mp4', '_2.mkv', '_3.mp4']:
		dataset, users, activities, raw_files = history[data_type]
		# train_x, test_x, train_y, test_y = train_test_split(dataset, users, test_size=0.3, random_state=42)
		train_x, test_x, train_y, test_y, train_raw_files, test_raw_files = train_test_split(dataset, activities,
		                                                                                     raw_files, test_size=0.3,
		                                                                                     random_state=random_state)
	elif data_type in ['random']:
		# data_type == 'all':
		dataset1, users1, activities1, raw_files1 = history['_1.mp4']
		dataset2, users2, activities2, raw_files2 = history['_2.mkv']
		dataset3, users3, activities3, raw_files3 = history['_3.mp4']
		dataset = np.concatenate([dataset1, dataset2, dataset3], axis=0)
		users = np.concatenate([users1, users2, users3], axis=0)
		activities = np.concatenate([activities1, activities2, activities3], axis=0)
		raw_files = raw_files1 + raw_files2 + raw_files3

		train_x, test_x, train_y, test_y, train_raw_files, test_raw_files = train_test_split(dataset, activities,
		                                                                                     raw_files, test_size=0.3,
		                                                                                     random_state=random_state)
	else:
		# data_type == 'all':
		dataset1, users1, activities1, raw_files1 = history['_1.mp4']
		dataset2, users2, activities2, raw_files2 = history['_2.mkv']
		dataset3, users3, activities3, raw_files3 = history['_3.mp4']
		# raw_files3 = []
		# raw_files2 = []

		train_x, test_x, train_y, test_y, train_raw_files, test_raw_files = train_test_split(dataset1, activities1,
		                                                                                     raw_files1, test_size=0.3,
		                                                                                     random_state=random_state)

		train_ = []
		activities_ = []
		users_ = []
		raw_files_ = []
		y_ = []
		for idx, f in enumerate(train_raw_files):
			for i, f2 in enumerate(raw_files2):
				if f.replace('_1.mp4.npy', '_2.mkv.npy') == f2:
					train_.append(dataset2[i, :])
					activities_.append(activities2[i])
					users_.append(users2[i])
					raw_files_.append(raw_files2[i])
					y_.append(train_y[idx])
					break
			for j, f3 in enumerate(raw_files3):
				if f.replace('_1.mp4.npy', '_3.mp4.npy') == f3:
					train_.append(dataset3[j, :])
					activities_.append(activities3[j])
					users_.append(users3[j])
					raw_files_.append(raw_files3[j])
					y_.append(train_y[idx])
					break

		train_x = np.concatenate([train_x, np.asarray(train_)], axis=0)
		train_y = np.concatenate([train_y, np.asarray(y_)], axis=0)
		train_raw_files.extend(raw_files_)

		test_ = []
		activities_ = []
		users_ = []
		raw_files_ = []
		y_ = []
		for idx, f in enumerate(test_raw_files):
			for i, f2 in enumerate(raw_files2):
				if f.replace('_1.mp4.npy', '_2.mkv.npy') == f2:
					test_.append(dataset2[i, :])
					activities_.append(activities2[i])
					users_.append(users2[i])
					raw_files_.append(raw_files2[i])
					y_.append(test_y[idx])
					break

			for j, f3 in enumerate(raw_files3):
				if f.replace('_1.mp4.npy', '_3.mp4.npy') == f3:
					test_.append(dataset3[j, :])
					activities_.append(activities3[j])
					users_.append(users3[j])
					raw_files_.append(raw_files3[j])
					y_.append(test_y[idx])
					break

		test_x = np.concatenate([test_x, np.asarray(test_)], axis=0)
		test_y = np.concatenate([test_y, np.asarray(y_)], axis=0)
		test_raw_files.extend(raw_files_)

	return train_x, train_y, test_x, test_y, train_raw_files, test_raw_files


@timer
def get_data(root_dir, cameras=[], classes=[], m=10):
	history = {}
	for i, camera in enumerate(cameras):
		dataset, users, activities, raw_files = _get_data(root_dir, camera_type=camera, classes=classes, m=m)
		history[camera] = (dataset, users, activities, raw_files)

	return history


def merge_label(train_y):
	res = []
	for v in train_y:
		if v == 0:
			res.append(v)
		elif v == 4:
			res.append(2)
		else:
			res.append(1)

	return np.asarray(res)


def add_new_features(train_x, train_y, idx2label):
	# new_train_x  = []
	# # add direction of operations [operate the device or not, put_back direction, take_out_direction]
	# for x_, y_ in zip(train_x, train_y):
	# 	if idx2label[y_] == 'no_interaction':
	# 		new_train_x.append(np.asarray(x_.tolist() + [0, 0, 0]))
	# 	elif idx2label[y_] == 'open_close_fridge':
	# 		new_train_x.append(np.asarray(x_.tolist() + [1, 0, 0]))
	# 	elif idx2label[y_] == 'put_back_item':
	# 		new_train_x.append(np.asarray(x_.tolist() + [1, 1, 0]))
	# 	elif idx2label[y_] == 'take_out_item':
	# 		new_train_x.append(np.asarray(x_.tolist() + [1, 0, 1]))
	# 	elif idx2label[y_] == 'screen_interaction':
	# 		new_train_x.append(np.asarray(x_.tolist() + [1, 0, 0]))

	# new_train_x = []
	# # add direction of operations [operate the device or not, hand direction (in to out (), out to in, mid to out, mid)]
	# for x_, y_ in zip(train_x, train_y):
	# 	if idx2label[y_] == 'no_interaction':
	# 		new_train_x.append(np.asarray(x_.tolist() + [0, 0]))
	# 	elif idx2label[y_] == 'open_close_fridge':
	# 		new_train_x.append(np.asarray(x_.tolist() + [1, 0]))
	# 	elif idx2label[y_] == 'put_back_item':
	# 		new_train_x.append(np.asarray(x_.tolist() + [1, 1]))
	# 	elif idx2label[y_] == 'take_out_item':
	# 		new_train_x.append(np.asarray(x_.tolist() + [1, -1]))
	# 	elif idx2label[y_] == 'screen_interaction':
	# 		new_train_x.append(np.asarray(x_.tolist() + [1, 0]))

	new_train_x = []
	# add direction of operations [operate the device or not, hand direction (in to out (), out to in, mid to out, mid)]
	for x_, y_ in zip(train_x, train_y):
		if idx2label[y_] == 'no_interaction':
			new_train_x.append([0, 0])
		elif idx2label[y_] == 'open_close_fridge':
			new_train_x.append([1, 0])
		elif idx2label[y_] == 'put_back_item':
			new_train_x.append([1, 1])
		elif idx2label[y_] == 'take_out_item':
			new_train_x.append([1, -1])
		elif idx2label[y_] == 'screen_interaction':
			new_train_x.append([1, 0])

	# new_train_x = []
	# # add direction of operations [operate the device or not, hand direction (in to out (), out to in, mid to out, mid)]
	# for x_, y_ in zip(train_x, train_y):
	# 	if idx2label[y_] == 'no_interaction':
	# 		new_train_x.append([0])
	# 	elif idx2label[y_] == 'open_close_fridge':
	# 		new_train_x.append([0])
	# 	elif idx2label[y_] == 'put_back_item':
	# 		new_train_x.append([1])
	# 	elif idx2label[y_] == 'take_out_item':
	# 		new_train_x.append([-1])
	# 	elif idx2label[y_] == 'screen_interaction':
	# 		new_train_x.append([0])

	return np.asarray(new_train_x), train_y


def _main(m=10, random_state=10):
	ROOT = 'examples/classical_ml'
	all_results = {}
	###############################################################################################################
	# 1. get each camera data
	# root_dir: the location of 3d keypoints data
	root_dir = f'{ROOT}/out/keypoints3d-20210907/keypoints3d/data'
	# root_dir = 'examples/out/keypoints3d/datasets'
	classes = ['no_interaction', 'open_close_fridge', 'put_back_item', 'take_out_item', 'screen_interaction']
	label2idx = {v: i for i, v in enumerate(classes)}
	idx2label = {i: v for i, v in enumerate(classes)}
	print('label2idx: ', label2idx)
	idx2camera = {'_1.mp4': 'camera1', '_2.mkv': 'camera2', '_3.mp4': 'camera3', 'all': 'all', 'random': 'random'}
	# history_file: the location where I save the fft_feature to
	# history_file = f'{ROOT}/out/fft_feature/datasets_6stats-entire.dat'
	# history_file = f'{ROOT}/out/fft_feature/datasets_6stats-windows.dat'
	history_file = f'{ROOT}/out/fft_feature/datasets_6stats11-.dat'
	if os.path.exists(history_file): os.remove(history_file)
	if not os.path.exists(history_file):
		history = get_data(root_dir, cameras=['_1.mp4', '_2.mkv', '_3.mp4'], classes=classes, m=m)
		# history = get_data(root_dir, cameras=['_1.mp4', '_2.mkv'], classes=classes, m=m)
		check_path(history_file)
		dump(history, history_file)
	else:
		history = load(history_file)
	# # plot
	is_plot = False
	if is_plot:
		for camera, vs in history.items():
			dataset, users, activities, raw_files = vs
			pca_plot(dataset, activities, classes, title=f'{camera}')

	###############################################################################################################
	# 2. combine all cameras' data
	data_type = 'all'  # '_2.mkv', 'random', 'all'
	train_x, train_y, test_x, test_y, train_raw_files, test_raw_files = split_train_test(history, data_type,
	                                                                                     random_state)

	# # add new features
	# train_x, train_y = add_new_features(train_x, train_y, idx2label)
	# test_x, test_y = add_new_features(test_x, test_y, idx2label)

	is_merge_label = False
	if is_merge_label:
		# classes = ['no_interaction', 'rest']
		classes = ['no_interaction', 'open_put_take', 'screen_interaction']
		label2idx = {v: i for i, v in enumerate(classes)}
		train_y = merge_label(train_y)
		test_y = merge_label(test_y)
		is_plot = True
		if is_plot:
			pca_plot2(train_x, train_y, classes, title=f'')

	print(
		f'before sampling, train: {sorted(Counter(train_y).items(), key=lambda x: x[0], reverse=False)}, tot: {len(train_y)}')

	# from imblearn.over_sampling import RandomOverSampler
	# ros = RandomOverSampler(random_state=42)
	# train_x, train_y = ros.fit_resample(train_x, train_y)
	# print(sorted(Counter(train_y).items()))

	###############################################################################################################
	# 3. normalization
	print(f'train: {sorted(Counter(train_y).items(), key=lambda x: x[0], reverse=False)}, tot: {len(train_y)}\n'
	      f'test: {sorted(Counter(test_y).items(), key=lambda x: x[0], reverse=False)},  tot: {len(test_y)}')
	print(f'train_raw_files.shape: {len(train_raw_files)}, test_raw_files.shape: {len(test_raw_files)}')
	ss = StandardScaler()
	# ss = RobustScaler()
	# ss = MinMaxScaler()
	ss.fit(train_x)
	train_x = ss.transform(train_x)
	test_x = ss.transform(test_x)

	###############################################################################################################
	# 4. Feature reduction
	print(f'before: X_train: {train_x.shape}, y_train.shape: {train_y.shape}, X_test: {test_x.shape}')
	# n_comps = [1, 5, 10] + list(range(20, 301, 20)) + [train_x.shape[1]]
	raw_train_x = copy.deepcopy(train_x)
	raw_test_x = copy.deepcopy(test_x)
	res = {}
	# n_comps = [v * 0.01 for v in range(80, 100 + 1, 5)]
	n_comps = [1]
	print(n_comps)

	for n_comp in n_comps:
		train_x = copy.deepcopy(raw_train_x)
		test_x = copy.deepcopy(raw_test_x)
		reduction_method = 'pca1'
		if reduction_method == 'pca':
			pca = PCA(n_components=n_comp)  # min(len(np.unique(train_y)) - 1, train_x.shape[1]))
			pca.fit(train_x)
			train_x = pca.transform(train_x)
			test_x = pca.transform(test_x)
		elif reduction_method == 'sir':  # sliced inverse regression
			# # Set the options for SIR
			sir = SlicedInverseRegression(n_directions=n_comp, n_slices=len(np.unique(train_y)))
			# fit the model
			sir.fit(train_x, train_y)
			# transform into the new subspace
			train_x = sir.transform(train_x)
			test_x = sir.transform(test_x)
		elif reduction_method == 'lda':
			lda = LinearDiscriminantAnalysis(n_components=min(len(np.unique(train_y)) - 1, train_x.shape[1]))
			lda.fit(train_x, train_y)
			train_x = lda.transform(train_x)
			test_x = lda.transform(test_x)

		print(f'After: X_train: {train_x.shape}, y_train.shape: {train_y.shape}, X_test: {test_x.shape}')

		# # ###############################################################################################################
		# # # 5. build and evaluate models
		# neigh = KNeighborsClassifier(n_neighbors=n_comp)
		# neigh.fit(train_x, train_y)
		# pred_y = neigh.predict(test_x)
		# print('\nKNN')
		# # print(classification_report(test_y, pred_y))
		# cm = confusion_matrix(test_y, pred_y)
		# print(cm)
		# cm_file = make_confusion_matrix(cm, categories=label2idx.keys(),
		#                                 title=f'KNN-{idx2camera[data_type]}', out_dir='')
		# # print(f'confusion_matrix: {cm_file}')
		# train_acc = accuracy_score(train_y, neigh.predict(train_x))
		# acc = accuracy_score(test_y, pred_y)
		# print(acc, train_acc)
		# # missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files, model_name = 'RF', idx2label=idx2label)
		# # print(missclassified)
		# all_results['knn'] = (acc, train_acc, test_x.shape[1])

		# for v in range(1, 20):
		# 	print(f'min_samples_leaf: {v}')
		# 	rf = RandomForestClassifier(random_state=42, min_samples_leaf=v)
		# 	rf.fit(train_x, train_y)
		# 	pred_y = rf.predict(test_x)
		# 	# print(classification_report(test_y, pred_y))
		# 	print('\nRF')
		# 	cm = confusion_matrix(test_y, pred_y)
		# 	print(cm)
		# 	cm_file = make_confusion_matrix(cm, categories=label2idx.keys(), title=f'RF-{idx2camera[data_type]}', out_dir='')
		# 	print(f'confusion_matrix: {cm_file}')
		# 	acc = accuracy_score(test_y, pred_y)
		# 	print(acc, accuracy_score(train_y, rf.predict(train_x)))
		# 	missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files)
		# 	print(missclassified)
		# 	break

		print('\nOnevsrest(LR)')
		orc = OneVsRestClassifier(LogisticRegression(C=1, random_state=42, solver='liblinear'))
		orc.fit(train_x, train_y)
		pred_y = orc.predict(test_x)
		# print(classification_report(test_y, pred_y))
		cm = confusion_matrix(test_y, pred_y)
		print(cm)
		cm_file = make_confusion_matrix(cm, categories=label2idx.keys(), title=f'Onevsrest-{idx2camera[data_type]}',
		                                out_dir='')
		print(f'confusion_matrix: {cm_file}')
		train_acc = accuracy_score(train_y, orc.predict(train_x))
		acc = accuracy_score(test_y, pred_y)
		print(acc, train_acc)
		# missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files, model_name = 'RF', idx2label=idx2label)
		# print(missclassified)
		all_results['OVS(LR)'] = (acc, train_acc, test_x.shape[1])

		rf = RandomForestClassifier(n_estimators=100, random_state=42)
		rf.fit(train_x, train_y)
		pred_y = rf.predict(test_x)
		# print(classification_report(test_y, pred_y))
		print('\nRF')
		cm = confusion_matrix(test_y, pred_y)
		print(cm)
		cm_file = make_confusion_matrix(cm, categories=label2idx.keys(), title=f'RF-{idx2camera[data_type]}',
		                                out_dir='')
		print(f'confusion_matrix: {cm_file}')
		train_acc = accuracy_score(train_y, rf.predict(train_x))
		acc = accuracy_score(test_y, pred_y)
		print(acc, train_acc)
		missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files, model_name='RF', idx2label=idx2label)
		print(missclassified)
		# tmp = list(zip(rf.feature_importances_, features_names))
		# print(sorted(tmp, key = lambda x: x[0], reverse=True))
		# print(rf.feature_importances_, features_names)
		# feat_importance = rf.compute_feature_importances(normalize=False)
		# print("feat importance = " + str(feat_importance))
		all_results['RF'] = (acc, train_acc, test_x.shape[1])

		# dim = train_x.shape[1]
		# mlp = MLPClassifier(solver='adam', alpha=1e-3, learning_rate='adaptive', verbose=True, max_iter = 100,
		#                     early_stopping=True, activation='tanh', validation_fraction=0.1,
		#                     hidden_layer_sizes=(dim // 2, dim//4, 10, ), random_state=42)
		# mlp.fit(train_x, train_y)
		# pred_y = mlp.predict(test_x)
		# # print(classification_report(test_y, pred_y))
		# print('\nMLP')
		# cm = confusion_matrix(test_y, pred_y)
		# print(cm)
		# cm_file = make_confusion_matrix(cm, categories=label2idx.keys(), title=f'MLP-{idx2camera[data_type]}',
		#                                 out_dir='')
		# # print(f'confusion_matrix: {cm_file}')
		# train_acc = accuracy_score(train_y, mlp.predict(train_x))
		# acc = accuracy_score(test_y, pred_y)
		# print(acc, train_acc)
		# # missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files, model_name = 'RF', idx2label=idx2label)
		# # print(missclassified)
		# # tmp = list(zip(mlp.feature_importances_, features_names))
		# # print(sorted(tmp, key = lambda x: x[0], reverse=True))
		# # print(rf.feature_importances_, features_names)
		# # feat_importance = rf.compute_feature_importances(normalize=False)
		# # print("feat importance = " + str(feat_importance))
		# all_results['mlp'] = (acc, train_acc, test_x.shape[1])

		# dt = DecisionTreeClassifier(random_state=42)
		# dt.fit(train_x, train_y)
		# pred_y = dt.predict(test_x)
		# # print(classification_report(test_y, pred_y))
		# print('\nDT')
		# cm = confusion_matrix(test_y, pred_y)
		# print(cm)
		# cm_file = make_confusion_matrix(cm, categories=label2idx.keys(), title=f'DT-{idx2camera[data_type]}',
		#                                 out_dir='')
		# # print(f'confusion_matrix: {cm_file}')
		# train_acc = accuracy_score(train_y, dt.predict(train_x))
		# acc = accuracy_score(test_y, pred_y)
		# print(acc, train_acc)
		# # missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files, model_name = 'RF', idx2label=idx2label)
		# # print(missclassified)
		# all_results['dt'] = (acc, train_acc, test_x.shape[1])
		# feature_name = []
		# print(dt.feature_importances_, feature_name)
		# feat_importance = dt.tree_.compute_feature_importances(normalize=False)
		# print("feat importance = " + str(feat_importance))
		#
		# out = export_graphviz(dt, out_file='tree.dot')
		# os.system('dot -Tpng tree.dot -o tree.png')

		# print('\ngbc')
		# gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth = 1, random_state = 0)
		# gbc.fit(train_x, train_y)
		# pred_y = gbc.predict(test_x)
		# # print(classification_report(test_y, pred_y))
		# cm = confusion_matrix(test_y, pred_y)
		# print(cm)
		# cm_file = make_confusion_matrix(cm, categories=label2idx.keys(), title=f'gbc-{idx2camera[data_type]}',
		#                                 out_dir='')
		# print(f'confusion_matrix: {cm_file}')
		# train_acc = accuracy_score(train_y, gbc.predict(train_x))
		# acc = accuracy_score(test_y, pred_y)
		# print(acc, train_acc)
		# # missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files, model_name = 'RF', idx2label=idx2label)
		# # print(missclassified)
		# all_results['gbc'] = (acc, train_acc, test_x.shape[1])

		# import xgboost as xgb
		# # specify parameters via map
		# # param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
		# # num_round = 2
		# bst = xgb.XGBClassifier()
		# bst.fit(train_x, train_y)
		# # make prediction
		# pred_y = bst.predict(test_x)
		# # print(classification_report(test_y, pred_y))
		# cm = confusion_matrix(test_y, pred_y)
		# print(cm)
		# cm_file = make_confusion_matrix(cm, categories=label2idx.keys(), title=f'xgboost-{idx2camera[data_type]}',
		#                                 out_dir='')
		# print(f'confusion_matrix: {cm_file}')
		# train_acc = accuracy_score(train_y, bst.predict(train_x))
		# acc = accuracy_score(test_y, pred_y)
		# print(acc, train_acc)
		# # missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files, model_name = 'RF', idx2label=idx2label)
		# # print(missclassified)
		# all_results['bst'] = (acc, train_acc, test_x.shape[1])

		#
		# print('\nSVM(rbf)')
		# svm = SVC(kernel='rbf', random_state=42)
		# svm.fit(train_x, train_y)
		# pred_y = svm.predict(test_x)
		# # print(classification_report(test_y, pred_y))
		# cm = confusion_matrix(test_y, pred_y)
		# print(cm)
		# cm_file = make_confusion_matrix(cm, categories=label2idx.keys(), title=f'SVM-{idx2camera[data_type]}', out_dir='')
		# # print(f'confusion_matrix: {cm_file}')
		# train_acc = accuracy_score(train_y, svm.predict(train_x))
		# acc = accuracy_score(test_y, pred_y)
		# print(acc, train_acc)
		# # missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files, model_name = 'RF', idx2label=idx2label)
		# # print(missclassified)
		# all_results['svm'] = (acc, train_acc, test_x.shape[1])

		print('\nSVM(linear)')
		svm = SVC(kernel='linear', random_state=42)
		svm.fit(train_x, train_y)
		pred_y = svm.predict(test_x)
		# print(classification_report(test_y, pred_y))
		cm = confusion_matrix(test_y, pred_y)
		print(cm)
		cm_file = make_confusion_matrix(cm, categories=label2idx.keys(), title=f'SVM-linear-{idx2camera[data_type]}',
		                                out_dir='')
		print(f'confusion_matrix: {cm_file}')
		train_acc = accuracy_score(train_y, svm.predict(train_x))
		acc = accuracy_score(test_y, pred_y)
		print(acc, train_acc)
		# missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files, model_name = 'RF', idx2label=idx2label)
		# print(missclassified)
		all_results['SVM(linear)'] = (acc, train_acc, test_x.shape[1])
		if reduction_method not in ['pca', 'sir', 'lda']:
			return all_results
		res[n_comp] = copy.deepcopy(all_results)
		print('***', n_comp, all_results, '***\n\n')

	print(res)
	# methods = ['rf']
	# plot_data(res, methods, plot_type='test', x_label='n_comps', title='Test')
	return res


def plot_data(history, methods=['knn', 'rf', 'svm', 'svm-linear'], plot_type='test',
              x_label='Number of windows (m)', y_label='Accuracy', title=''):
	colors = ['r', 'b', 'g', 'm']
	for method, color in zip(methods, colors):
		X = []
		Y = []
		j = 0
		for key, vs in history.items():
			X.append(key)
			if plot_type == 'test':
				Y.append(vs[method][0])  # (test_acc, train_acc, dimension)
			elif plot_type == 'train':
				Y.append(vs[method][1])  # (test_acc, train_acc, dimension)
			else:  # dimension
				Y.append(vs[method][-1])  # (test_acc, train_acc, dimension)
		plt.plot(X, Y, '*-', label=method, color=color)

	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.legend(loc="upper right")
	out_file = f'{title}.png'
	plt.savefig(out_file, bbox_inches='tight', dpi=300)
	print(out_file)
	plt.show()


#
# @timer
# def main(random_state=10):
# 	history = {}
# 	ms = [1, 5, 10] + list(range(20, 501, 20))
# 	# ms = [1, 5, 10, 15, 20, 25, 30]
# 	# ms = [i for i in range(17)]
# 	ms = [40]
# 	print(ms)
# 	_out_file = f'{ROOT}/~history.dat'
# 	for i, m in enumerate(ms):
# 		try:
# 			tmp = _main(m, random_state)
# 		except Exception as e:
# 			print(f'Error: {m}, {e}')
# 			tmp = {'knn': (0, 0, 0), 'rf': (0, 0, 0), 'svm': (0, 0, 0), 'svm-linear': (0, 0, 0)}
# 		history[m] = tmp
# 		dump(history, _out_file)
# 		print(f'***{i}/{len(ms)}, m: {m}, {tmp}, {_out_file}***\n\n')
# 	# break
#
# 	# # # out_file = f'{ROOT}/~history-rf.dat'
# 	# # # dump(history, out_file)
# 	# # #
# 	# # # history = load(out_file)
# 	for key, value in history.items():
# 		print(key, value)
#
#
# # # methods = ['knn', 'rf', 'svm', 'svm-linear']
# # methods = ['knn', 'rf', 'svm']
# # methods = ['knn', 'rf']
# # plot_data(history, methods, plot_type='test', title='Test')
# # plot_data(history, methods, plot_type='train', title='Train')
# # plot_data(history, methods, plot_type='dimension', title='Dimension', y_label='Dimension')
# # #

if __name__ == '__main__':

	res = {}
	# repeats with different random states
	repeats = [42, 10, 100, 500, 1000]
	for i, random_state in enumerate(repeats):
		print(f'\n\nrepeat: {i + 1}/{len(repeats)}')
		tmp = _main(m=84, random_state=random_state)
		res[random_state] = tmp

	# get mean and std of each model's results
	for model_name in ['OVS(LR)', 'RF', 'SVM(linear)']:
		model_res = [vs[model_name][0] for rs, vs in res.items()]
		print(f'{model_name}: {np.mean(model_res):.2f} +/- {np.std(model_res):.2f}. Acc: {model_res}')
