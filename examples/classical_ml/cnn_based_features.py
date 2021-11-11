"""

run the below command under 'activity_recognition' folder:
    PYTHONPATH=. python3 examples/classical_ml/keypoints3d_based_features.py

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

from ar.features.feature import _get_fft, extract_feature_average
from ar.utils.utils import make_confusion_matrix, load, dump, check_path, timer
from examples.classical_ml.misc.fft_feature_inverse_regression_users import pca_plot2


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


def extract_feature_average(x):
	if len(x) == 0:
		x = np.sum(x, axis=0)
	else:
		x = np.sum(x, axis=0) / len(x)
	return x.reshape(1, -1)


def get_features(raw_data='', file='', feature_type='mean', dim=48):
	""" Get features

	Parameters
	---------
	Returns
	-------

	"""
	# print(file)
	if feature_type == 'mean':
		x = extract_feature_average(raw_data)
	elif feature_type == 'concatenated_frames':
		tmp = raw_data.flatten().tolist()
		thres = dim * 4096
		if len(tmp) > thres:
			x = tmp[:thres]
		else:
			x = tmp + [0] * (thres - len(tmp))
		x = np.asarray(x).reshape(1, -1)
	elif feature_type == 'concatenated_frames2':
		res = []
		window_size = len(raw_data) // dim
		if window_size == 0:
			window_size = 1
		cnt = 0
		for i in range(0, len(raw_data), window_size):
			if i == 0:
				res = np.mean(raw_data[i:i + window_size], axis=0)
			elif cnt == dim - 1:
				res = np.concatenate([res, np.mean(raw_data[i:], axis=0)])
				break
			else:
				res = np.concatenate([res, np.mean(raw_data[i:i + window_size], axis=0)])
			cnt += 1
		res = res.flatten().tolist()
		res = res + [0] * (dim * 4096 - len(res))
		x = np.asarray(res).reshape(1, -1)
	# if x.shape !=(1, 69632):
	# 	print(x.shape)
	return x.flatten()


@timer
def _get_data(root_dir='examples/out/cnn_based_features', camera_type='_1_vgg', classes=[], m=10):
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
			# elif camera_type == '_2_vgg':  # fps = 30
			# 	fps = 30
			# 	min_frames = 466  # number of frames
			# 	max_frames = 708  # number of frames
			# 	if raw_data.shape[0] < min_frames or raw_data.shape[0] > max_frames:
			# 		print(len(raw_data), file)
			# 		continue
			# elif camera_type == '_3_vgg':  # fps = 3
			# 	min_frames = 76  # number of frames
			# 	max_frames = 95  # number of frames
			# 	if raw_data.shape[0] < min_frames or raw_data.shape[0] > max_frames:
			# 		print(len(raw_data), file)
			# 		continue

			# for keypoint_idx, _ in enumerate(keypoints):
			# 	tmp = get_fft_features(raw_data, keypoint=keypoint_idx)
			# 	# data = special_keypoints(np.load(file))
			# 	data.extend(list(tmp))
			data = get_features(raw_data, file=file, dim=m)
			data = np.asarray(data)

			dataset.append(data)
			users.append(user)
			activities.append(classes.index(act))
			raw_files.append(file)
		print(act, len(files))
	qs = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 1]
	dur = np.quantile(duration, q=qs)
	print(f'***{camera_type}, durations: {list(zip(dur, qs))}\n')

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
	if data_type in ['_1.mp4', '_2_vgg', '_3_vgg']:
		dataset, users, activities, raw_files = history[data_type]
		# train_x, test_x, train_y, test_y = train_test_split(dataset, users, test_size=0.3, random_state=42)
		train_x, test_x, train_y, test_y, train_raw_files, test_raw_files = train_test_split(dataset, activities,
		                                                                                     raw_files, test_size=0.3,
		                                                                                     random_state=random_state)
	elif data_type in ['random']:
		# data_type == 'all':
		dataset1, users1, activities1, raw_files1 = history['_1.mp4']
		dataset2, users2, activities2, raw_files2 = history['_2_vgg']
		dataset3, users3, activities3, raw_files3 = history['_3_vgg']
		dataset = np.concatenate([dataset1, dataset2, dataset3], axis=0)
		users = np.concatenate([users1, users2, users3], axis=0)
		activities = np.concatenate([activities1, activities2, activities3], axis=0)
		raw_files = raw_files1 + raw_files2 + raw_files3

		train_x, test_x, train_y, test_y, train_raw_files, test_raw_files = train_test_split(dataset, activities,
		                                                                                     raw_files, test_size=0.3,
		                                                                                     random_state=random_state)
	else:
		# data_type == 'all':
		dataset1, users1, activities1, raw_files1 = history['_1_vgg']
		dataset2, users2, activities2, raw_files2 = history['_2_vgg']
		dataset3, users3, activities3, raw_files3 = history['_3_vgg']
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
				if f.replace('_1_vgg.npy', '_2_vgg.npy') == f2:
					train_.append(dataset2[i, :])
					activities_.append(activities2[i])
					users_.append(users2[i])
					raw_files_.append(raw_files2[i])
					y_.append(train_y[idx])
					break
			for j, f3 in enumerate(raw_files3):
				if f.replace('_1_vgg.npy', '_3_vgg.npy') == f3:
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
				if f.replace('_1_vgg.npy', '_2_vgg.npy') == f2:
					test_.append(dataset2[i, :])
					activities_.append(activities2[i])
					users_.append(users2[i])
					raw_files_.append(raw_files2[i])
					y_.append(test_y[idx])
					break

			for j, f3 in enumerate(raw_files3):
				if f.replace('_1_vgg.npy', '_3_vgg.npy') == f3:
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


@timer
def _main(m=10, random_state=10):
	ROOT = 'examples/classical_ml'
	###############################################################################################################
	# 1. get each camera data
	# root_dir: the location of 3d keypoints data
	root_dir = f'{ROOT}/out/cnn_based_features'
	classes = ['no_interaction', 'open_close_fridge', 'put_back_item', 'take_out_item', 'screen_interaction']
	label2idx = {v: i for i, v in enumerate(classes)}
	idx2label = {i: v for i, v in enumerate(classes)}
	print('label2idx: ', label2idx)
	idx2camera = {'_1_vgg': 'camera1', '_2_vgg': 'camera2', '_3_vgg': 'camera3', 'all': 'all', 'random': 'random'}
	# history_file: the location where I save the cnn_based_feature to
	history_file = f'{ROOT}/out/cnn_based_features/datasets.dat'
	if os.path.exists(history_file): os.remove(history_file)
	if not os.path.exists(history_file):
		history = get_data(root_dir, cameras=['_1_vgg', '_2_vgg', '_3_vgg'], classes=classes, m=m)
		# history = get_data(root_dir, cameras=['_1_vgg', '_2_vgg'], classes=classes, m=m)
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
	data_type = 'all'  # '_2_vgg', 'random', 'all'
	train_x, train_y, test_x, test_y, train_raw_files, test_raw_files = split_train_test(history, data_type,
	                                                                                     random_state)

	# # # add new features
	# # train_x, train_y = add_new_features(train_x, train_y, idx2label)
	# # test_x, test_y = add_new_features(test_x, test_y, idx2label)
	#
	# is_merge_label = False
	# if is_merge_label:
	# 	# classes = ['no_interaction', 'rest']
	# 	classes = ['no_interaction', 'open_put_take', 'screen_interaction']
	# 	label2idx = {v: i for i, v in enumerate(classes)}
	# 	train_y = merge_label(train_y)
	# 	test_y = merge_label(test_y)
	# 	is_plot = True
	# 	if is_plot:
	# 		pca_plot2(train_x, train_y, classes, title=f'')

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
	# n_comps = [1, 5, 10] + list(range(20, 301, 20)) + [train_x.shape[1]//2]
	dim = min(train_x.shape[0], train_x.shape[1])
	n_comps = list(range(1, dim, dim // 20))
	raw_train_x = copy.deepcopy(train_x)
	raw_test_x = copy.deepcopy(test_x)
	res = {}
	# n_comps = [v * 0.01 for v in range(80, 100 + 1, 5)]
	# the best n_comps are chosen when random state = 42. For the rest of repeats, we use the same best n_comps
	# n_comps = [60]
	print(n_comps)

	for n_comp in n_comps:
		all_results = {}
		train_x = copy.deepcopy(raw_train_x)
		test_x = copy.deepcopy(raw_test_x)
		reduction_method = 'sir1'
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

		ss = StandardScaler()
		# ss = RobustScaler()
		# ss = MinMaxScaler()
		ss.fit(train_x)
		train_x = ss.transform(train_x)
		test_x = ss.transform(test_x)
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
		# missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files, model_name='RF', idx2label=idx2label)
		# print(missclassified)
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

		import xgboost as xgb
		# specify parameters via map
		# param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
		# num_round = 2
		bst = xgb.XGBClassifier()
		bst.fit(train_x, train_y)
		# make prediction
		pred_y = bst.predict(test_x)
		# print(classification_report(test_y, pred_y))
		cm = confusion_matrix(test_y, pred_y)
		print(cm)
		cm_file = make_confusion_matrix(cm, categories=label2idx.keys(), title=f'xgboost-{idx2camera[data_type]}',
		                                out_dir='')
		print(f'confusion_matrix: {cm_file}')
		train_acc = accuracy_score(train_y, bst.predict(train_x))
		acc = accuracy_score(test_y, pred_y)
		print(acc, train_acc)
		# missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files, model_name = 'RF', idx2label=idx2label)
		# print(missclassified)
		all_results['bst'] = (acc, train_acc, test_x.shape[1])

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
	# print('***', n_comp, all_results, '***\n\n')

	print(f'res: ')
	for n_comp in res.keys():
		print(f'{n_comp}: {res[n_comp]}')
	# methods = ['rf']
	# plot_data(res, methods, plot_type='test', x_label='n_comps', title='Test')
	final = {}
	for model_name in all_results.keys():
		for i, n_comp in enumerate(res.keys()):
			if i == 0:
				final[model_name] = res[n_comp][model_name]
			elif final[model_name][0] < res[n_comp][model_name][0]:
				final[model_name] = res[n_comp][model_name]
	print(f'final: {final}')
	return final


if __name__ == '__main__':

	res = {}
	# repeats with different random states
	repeats = [42, 10, 100, 500, 1000]
	repeats = [42]
	for i, random_state in enumerate(repeats):
		print(f'\n\nrepeat: {i + 1}/{len(repeats)}')
		tmp = _main(m=10, random_state=random_state)
		res[random_state] = tmp

	# get mean and std of each model's results
	for model_name in ['OVS(LR)', 'RF', 'bst', 'SVM(linear)']:
		model_res = [vs[model_name][0] for rs, vs in res.items()]
		print(f'{model_name}: {np.mean(model_res):.2f} +/- {np.std(model_res):.2f}. Acc: {model_res}')
