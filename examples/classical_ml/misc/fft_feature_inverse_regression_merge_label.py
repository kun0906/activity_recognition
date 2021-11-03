"""

"""
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
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sliced import SlicedInverseRegression

from ar.features.feature import _get_fft
from ar.utils.utils import load, dump, check_path, timer
from examples.classical_ml.misc.fft_feature_inverse_regression_users import pca_plot2

keypoints = [
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


def get_fft_features(raw_data='', m=84, keypoint=7):
	"""

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

	n = raw_data.shape[0]
	raw_data = raw_data.reshape((n, 51))
	res = []
	# flg = 'fft2'
	# if flg == 'fft2': # the result is worse
	# 	fft_features = _get_fft2(raw_data)
	# 	res = heapq.nlargest(10, fft_features)
	# 	return np.asarray(res).reshape(-1, )

	for i in range(51):
		data = raw_data[:, i]
		flg = 'fft'
		if flg == 'fft':
			fft_features = _get_fft(data, fft_bin=m)
			fft_features = fft_features[0:1 + int(np.ceil((m - 1) / 2))]
		elif flg == 'std':
			fft_features = [np.mean(data), np.std(data)]
		# fft_features = list(np.quantile(data, q=[0, 0.25, 0.5, 0.75, 1]))
		# fft_features = list(np.quantile(data, q = [0, 0.25, 0.5, 0.75, 1])) + [np.mean(data), np.std(data)]
		elif flg == 'skew':
			# fft_features = [np.min(data), np.max(data)]
			# fft_features = [np.mean(data), np.std(data)]
			# fft_features = [skew(data)]
			# fft_features = [np.min(data), np.mean(data), np.std(data),  np.max(data)]
			fft_features = [np.mean(data), np.std(data), skew(data), kurtosis(data), np.min(data), np.max(data)]
		else:
			n = len(data)
			step = int(np.ceil(n / m))
			fft_features = []
			for i in range(0, len(data), step):
				vs = data[i:i + step]
				flg2 = 'stats'
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
	dataset = []
	users = []
	activities = []
	raw_files = []
	for act in classes:
		files = glob.glob(f'{root_dir}/data-clean/refrigerator/' + act + f'/*/*{camera_type}.npy')
		# files = [f'{root_dir}/data-clean/refrigerator/take_out_item/4/take_out_item_2_1616179391_1.mp4.npy',
		#          f'{root_dir}/data-clean/refrigerator/take_out_item/4/take_out_item_2_1616179391_1.mp4.npy']
		for file in files:
			# print('Processing file', file)
			user = int(file.split('/')[-2])

			data = []
			raw_data = np.load(file)
			# for keypoint_idx, _ in enumerate(keypoints):
			# 	tmp = get_fft_features(raw_data, keypoint=keypoint_idx)
			# 	# data = special_keypoints(np.load(file))
			# 	data.extend(list(tmp))
			data = get_fft_features(raw_data, m=m)
			data = np.asarray(data)

			dataset.append(data)
			users.append(user)
			activities.append(classes.index(act))
			raw_files.append(file)
		print(act, len(files))

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


def split_train_test(history, data_type=''):
	if data_type in ['_1.mp4', '_2.mkv', '_3.mp4']:
		dataset, users, activities, raw_files = history[data_type]
		# train_x, test_x, train_y, test_y = train_test_split(dataset, users, test_size=0.3, random_state=42)
		train_x, test_x, train_y, test_y, train_raw_files, test_raw_files = train_test_split(dataset, activities,
		                                                                                     raw_files, test_size=0.3,
		                                                                                     random_state=42)
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
		                                                                                     random_state=42)
	else:
		# data_type == 'all':
		dataset1, users1, activities1, raw_files1 = history['_1.mp4']
		dataset2, users2, activities2, raw_files2 = history['_2.mkv']
		dataset3, users3, activities3, raw_files3 = history['_3.mp4']
		# raw_files3 = []

		train_x, test_x, train_y, test_y, train_raw_files, test_raw_files = train_test_split(dataset1, activities1,
		                                                                                     raw_files1, test_size=0.3,
		                                                                                     random_state=42)

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


ROOT = 'examples/classical_ml'


def _main(m=10):
	all_results = {}
	###############################################################################################################
	# 1. get each camera data
	# root_dir: the location of 3d keypoints data
	root_dir = f'{ROOT}/out/keypoints3d-20210907/keypoints3d/data'
	classes = ['no_interaction', 'open_close_fridge', 'put_back_item', 'take_out_item', 'screen_interaction']
	label2idx = {v: i for i, v in enumerate(classes)}
	idx2label = {i: v for i, v in enumerate(classes)}
	print('label2idx: ', label2idx)
	idx2camera = {'_1.mp4': 'camera1', '_2.mkv': 'camera2', '_3.mp4': 'camera3', 'all': 'all', 'random': 'random'}
	# history_file: the location where I save the fft_feature to
	history_file = f'{ROOT}/out/fft_feature/datasets_fft2.dat'
	if os.path.exists(history_file): os.remove(history_file)
	if not os.path.exists(history_file):
		history = get_data(root_dir, cameras=['_1.mp4', '_2.mkv', '_3.mp4'], classes=classes, m=m)
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
	data_type = '_1.mp4'  # '_2.mkv', 'random', 'all'
	train_x, train_y, test_x, test_y, train_raw_files, test_raw_files = split_train_test(history, data_type)
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
	# ss = StandardScaler()
	ss = RobustScaler()
	ss.fit(train_x)
	train_x = ss.transform(train_x)
	test_x = ss.transform(test_x)

	###############################################################################################################
	# 4. Feature reduction
	print(f'before: X_train: {train_x.shape}, y_train.shape: {train_y.shape}, X_test: {test_x.shape}')
	reduction_method = 'pca1'
	if reduction_method == 'pca':
		pca = PCA(n_components=0.95)  # min(len(np.unique(train_y)) - 1, train_x.shape[1]))
		pca.fit(train_x)
		train_x = pca.transform(train_x)
		test_x = pca.transform(test_x)
	elif reduction_method == 'sir':  # sliced inverse regression
		# # Set the options for SIR
		sir = SlicedInverseRegression(n_directions='auto', n_slices=len(np.unique(train_y)))
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

	###############################################################################################################
	# # 5. build and evaluate models
	# neigh = KNeighborsClassifier(n_neighbors=1)
	# neigh.fit(train_x, train_y)
	# pred_y = neigh.predict(test_x)
	# print('\nKNN')
	# # print(classification_report(test_y, pred_y))
	# cm = confusion_matrix(test_y, pred_y)
	# print(cm)
	# # cm_file = make_confusion_matrix(cm, categories=label2idx.keys(),
	# #                                 title=f'KNN-{idx2camera[data_type]}', out_dir='')
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

	rf = RandomForestClassifier(random_state=42)
	rf.fit(train_x, train_y)
	pred_y = rf.predict(test_x)
	# print(classification_report(test_y, pred_y))
	print('\nRF')
	cm = confusion_matrix(test_y, pred_y)
	print(cm)
	# cm_file = make_confusion_matrix(cm, categories=label2idx.keys(), title=f'RF-{idx2camera[data_type]}', out_dir='')
	# print(f'confusion_matrix: {cm_file}')
	train_acc = accuracy_score(train_y, rf.predict(train_x))
	acc = accuracy_score(test_y, pred_y)
	print(acc, train_acc)
	missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files, model_name='RF', idx2label=idx2label)
	print(missclassified)
	all_results['rf'] = (acc, train_acc, test_x.shape[1])

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

	# print('\nOnevsrest')
	# orc = OneVsRestClassifier(LogisticRegression(C=1, random_state=42, solver='liblinear'))
	# orc.fit(train_x, train_y)
	# pred_y = orc.predict(test_x)
	# # print(classification_report(test_y, pred_y))
	# cm = confusion_matrix(test_y, pred_y)
	# print(cm)
	# cm_file = make_confusion_matrix(cm, categories=label2idx.keys(), title=f'Onevsrest-{idx2camera[data_type]}',
	#                                 out_dir='')
	# print(f'confusion_matrix: {cm_file}')
	# train_acc = accuracy_score(train_y, orc.predict(train_x))
	# acc = accuracy_score(test_y, pred_y)
	# print(acc, train_acc)
	# # missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files, model_name = 'RF', idx2label=idx2label)
	# # print(missclassified)
	# all_results['orc'] = (acc, train_acc, test_x.shape[1])

	# print('\nSVM(rbf)')
	# svm = SVC(kernel='rbf', random_state=42)
	# svm.fit(train_x, train_y)
	# pred_y = svm.predict(test_x)
	# # print(classification_report(test_y, pred_y))
	# cm = confusion_matrix(test_y, pred_y)
	# print(cm)
	# # cm_file = make_confusion_matrix(cm, categories=label2idx.keys(), title=f'SVM-{idx2camera[data_type]}', out_dir='')
	# # print(f'confusion_matrix: {cm_file}')
	# train_acc = accuracy_score(train_y, svm.predict(train_x))
	# acc = accuracy_score(test_y, pred_y)
	# print(acc, train_acc)
	# # missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files, model_name = 'RF', idx2label=idx2label)
	# # print(missclassified)
	# all_results['svm'] = (acc, train_acc, test_x.shape[1])

	# print('\nSVM(linear)')
	# svm = SVC(kernel = 'linear', random_state=42)
	# svm.fit(train_x, train_y)
	# pred_y = svm.predict(test_x)
	# # print(classification_report(test_y, pred_y))
	# print(confusion_matrix(test_y, pred_y))
	# train_acc = accuracy_score(train_y, svm.predict(train_x))
	# acc = accuracy_score(test_y, pred_y)
	# print(acc, train_acc)
	# # missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files, model_name = 'RF', idx2label=idx2label)
	# # print(missclassified)
	# all_results['svm-linear'] = (acc, train_acc, test_x.shape[1])

	return all_results


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


@timer
def main():
	history = {}
	ms = [1, 5, 10] + list(range(20, 101, 20))
	print(ms)
	_out_file = f'{ROOT}/~history.dat'
	for i, m in enumerate(ms):
		try:
			tmp = _main(m)
		except Exception as e:
			print(f'Error: {m}, {e}')
			tmp = {'knn': (0, 0, 0), 'rf': (0, 0, 0), 'svm': (0, 0, 0), 'svm-linear': (0, 0, 0)}
		history[m] = tmp
		dump(history, _out_file)
		print(f'***{i}/{len(ms)}, m: {m}, {tmp}, {_out_file}***\n\n')
		break

	out_file = f'{ROOT}/~history-rf.dat'
	dump(history, out_file)

	history = load(out_file)
	for key, value in history.items():
		print(key, value)

	# methods = ['knn', 'rf', 'svm', 'svm-linear']
	methods = ['knn', 'rf', 'svm']
	methods = ['rf']
	plot_data(history, methods, plot_type='test', title='Test')
	plot_data(history, methods, plot_type='train', title='Train')
	plot_data(history, methods, plot_type='dimension', title='Dimension', y_label='Dimension')


if __name__ == '__main__':
	main()
