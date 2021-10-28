"""

"""
import glob
import os
from collections import defaultdict, Counter
from copy import deepcopy
from itertools import combinations
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sliced import SlicedInverseRegression

from ar.features.feature import _get_fft
from ar.utils.utils import make_confusion_matrix, load, dump, check_path, timer

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
	res = defaultdict(list)
	# n, m = cm.shape
	for i, (y_, y_2) in enumerate(zip(test_y, pred_y)):
		if y_ != y_2:
			# print(y_, y_2, test_raw_files[i])
			in_file = os.path.relpath(test_raw_files[i], 'examples/classical_ml/out/keypoints3d-20210907/keypoints3d/')
			# tmp_file = os.path.relpath(in_file, f'data/data-clean/refrigerator/{idx2label[y_]}')
			out_file = f'examples/classical_ml/out/misclassified_users/{model_name}/{idx2label[y_]}->/{idx2label[y_2]}/{in_file}'[
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
	for i in range(51):
		data = raw_data[:, i]
		# data = data.reshape((-1,))
		flg = 'std1'
		if flg == 'fft':
			fft_features = _get_fft(data, fft_bin=m)
			fft_features = fft_features[0:1 + int(np.ceil((m - 1) / 2))]
		elif flg == 'std':
			# fft_features = [np.mean(data), np.std(data)]
			fft_features = [np.min(data), np.mean(data), np.std(data), np.max(data)]
		# fft_features = list(np.quantile(data, q=[0, 0.25, 0.5, 0.75, 1]))
		# fft_features = list(np.quantile(data, q = [0, 0.25, 0.5, 0.75, 1])) + [np.mean(data), np.std(data)]
		elif flg == 'skew':
			# fft_features = [np.min(data), np.max(data)]
			# fft_features = [np.mean(data), np.std(data)]
			# fft_features = [skew(data), kurtosis(data)]
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
					# tmp = [np.mean(vs), np.std(vs)]
					tmp = [np.mean(vs), np.std(vs), skew(vs), kurtosis(vs), np.min(vs), np.max(vs)]
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


def _get_data(root_dir='examples/out/keypoints3d-20210907/keypoints3d/data', camera_type='_1.mp4', classes=[]):
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
			data = get_fft_features(raw_data)
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


def pca_plot1(dataset, activities, classes, users, title):
	nrows = 3
	ncols = 3
	fig, axes = plt.subplots(nrows, ncols, figsize=(20, 15))  # w, h

	for i in sorted(np.unique(users)):
		q, r = np.divmod(i - 1, ncols)
		g = axes[q, r]
		from sklearn.decomposition import PCA
		pca = PCA(n_components=2)
		X_new = pca.fit_transform(dataset)
		X_new = X_new[users == i]
		act_new = activities[users == i]
		for act_idx in range(5):
			g.scatter(X_new[act_new == act_idx, 0], X_new[act_new == act_idx, 1])
		# g.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), ncol=2, borderaxespad=0,
		#          fancybox=False, shadow=False, fontsize=8, title='classes')
		handles, labels = g.get_legend_handles_labels()
		# fig.legend(handles, labels, loc='upper center')
		g.legend(handles, labels, loc='center left', bbox_to_anchor=(1.25, 1), ncol=1, borderaxespad=0.)
		g.set_title(f'user_{i}')
		g.set_ylim([0, 40])
	fig.suptitle(f'each user\'s data: {title}')
	plt.tight_layout()  # takes too much time
	# out_file = f'{coordinate}-{color}-{title}.png'
	# plt.savefig(out_file, bbox_inches='tight', dpi=300)
	# print(out_file)
	plt.show()  # takes too much time


def plot_3dkeypoints(X, y, random_state=42, title=''):
	N, D = X.shape
	y_label = [idx2label[v] for v in y]
	df = pd.DataFrame(np.concatenate([X, np.reshape(y, (-1, 1)), np.reshape(y_label, (-1, 1))], axis=1),
	                  columns=[f'x{i + 1}' for i in range(X.shape[1])] + ['y', 'y_label'])
	df = df.astype({"x1": float, "x2": float, 'y': int, 'y_label': str})

	for i_coord, (coordinate, color) in enumerate(zip(['x', 'y', 'z'], ['r', 'g', 'b'])):
		# only plot the red channel values for each point(R, G, B)
		nrows = 4
		# using the variable axs for multiple Axes
		ncols = 5  # D // (17 * 3)
		print(ncols)
		fig, axes = plt.subplots(nrows, ncols, figsize=(20, 15))  # w, h
		for i in range(0, 17):  # 17 3d_keypoints
			# the ith keypoint
			# df_point = df[:, [(j, j+1, j+2) for j in range(0, 255, 17*3)]] # (j, j+1, j+2): (R, G, B)
			df_point = df[df.columns[[j + (i * 3) + i_coord for j in range(0, D, 17 * 3)]]]  # R
			# df_point.shape (1000, 5)  # 1000 is the number of videos,  5 is the number of timestamps
			q, r = divmod(i, ncols)
			if q >= nrows: break
			g = axes[q, r]
			for i_video, lab_ in enumerate(y_label):
				# plt.plot(x, y)
				g.plot(range(0, D // (17 * 3)), [float(v) for v in df_point.iloc[i_video].values],
				       linestyle='', marker='*', color=label2color[lab_])
				# break
			if r == 0:
				g.set_ylabel(f'{coordinate} coordinate')
			if q == 4 - 1:
				g.set_xlabel('feature (aggregated on frames)')
			g.set_title(f'{i + 1}th keypoint')
			print(f'{i}, axes[{q}, {r}]: {g}')
		fig.suptitle(f'{title}: {coordinate} coordinate.')
		plt.tight_layout()  # takes too much time
		out_file = f'{coordinate}-{color}-{title}.png'
		plt.savefig(out_file, bbox_inches='tight', dpi=300)
		print(out_file)
		plt.show()  # takes too much time

	# ### FacetGrid
	# grid = sns.FacetGrid(df, col="y_label", hue="y_label", hue_order=list(sorted(set(y_label))), col_wrap=3)
	# grid.map(sns.scatterplot, "x1", "x2", s=100, alpha=0.3)
	# grid.add_legend()
	# plt.show()


def pca_plot2(X, y, classes, title):
	from sklearn.decomposition import PCA
	pca = PCA(n_components=2)
	X = pca.fit_transform(X)
	for i in range(len(classes)):
		plt.scatter(X[y == i, 0], X[y == i, 1])
		plt.title(title)
	plt.legend(classes)
	plt.show()


def split_train_test_users(history, data_type='', test_users_list=[]):
	# test_users_list = [1, 3]

	if data_type in ['_1.mp4', '_2.mkv', '_3.mp4']:
		dataset, users, activities, raw_files = history[data_type]
		# # train_x, test_x, train_y, test_y = train_test_split(dataset, users, test_size=0.3, random_state=42)
		# train_x, test_x, train_y, test_y, train_raw_files, test_raw_files = train_test_split(dataset, activities,
		#                                                                                      raw_files, test_size=0.3,
		#                                                                                      random_state=42)

		train_x = []
		train_y = []
		train_users = []
		train_raw_files = []
		test_x = []
		test_y = []
		test_users = []
		test_raw_files = []

		cameras = [[dataset, users, activities, raw_files]]
		for camera in cameras:
			for (d, u, a, r) in zip(*camera):
				if u in test_users_list:
					test_x.append(d)
					test_y.append(a)
					test_users.append(u)
					test_raw_files.append(r)
				else:
					train_x.append(d)
					train_y.append(a)
					train_users.append(u)
					train_raw_files.append(r)
		train_x = np.asarray(train_x)
		train_y = np.asarray(train_y)
		test_x = np.asarray(test_x)
		test_y = np.asarray(test_y)
		print(Counter(train_users), Counter(test_users))

	else:
		# data_type == 'all':
		dataset1, users1, activities1, raw_files1 = history['_1.mp4']
		dataset2, users2, activities2, raw_files2 = history['_2.mkv']
		dataset3, users3, activities3, raw_files3 = history['_3.mp4']

		train_x = []
		train_y = []
		train_users = []
		train_raw_files = []
		test_x = []
		test_y = []
		test_users = []
		test_raw_files = []

		cameras = [[dataset1, users1, activities1, raw_files1],
		           [dataset2, users2, activities2, raw_files2],
		           [dataset3, users3, activities3, raw_files3]
		           ]
		for camera in cameras:
			for (d, u, a, r) in zip(*camera):
				if u in test_users_list:
					test_x.append(d)
					test_y.append(a)
					test_users.append(u)
					test_raw_files.append(r)
				else:
					train_x.append(d)
					train_y.append(a)
					train_users.append(u)
					train_raw_files.append(r)
		train_x = np.asarray(train_x)
		train_y = np.asarray(train_y)
		test_x = np.asarray(test_x)
		test_y = np.asarray(test_y)
		print(Counter(train_users), Counter(test_users))

	return train_x, train_y, test_x, test_y, train_raw_files, test_raw_files


@timer
def get_data(root_dir, cameras=[], classes=[]):
	history = {}
	for i, camera in enumerate(cameras):
		dataset, users, activities, raw_files = _get_data(root_dir, camera_type=camera, classes=classes)
		history[camera] = (dataset, users, activities, raw_files)

	return history


def merge_label(train_y):
	res = []
	for v in train_y:
		if v == 0:
			res.append(v)
		elif v == 4:
			res.append(1)
		else:
			res.append(1)

	return np.asarray(res)


def main(test_users_list):
	###############################################################################################################
	# 1. get each camera data
	ROOT = 'examples/classical_ml'
	# root_dir: the location of 3d keypoints data
	root_dir = f'{ROOT}/out/keypoints3d-20210907/keypoints3d/data'
	classes = ['no_interaction', 'open_close_fridge', 'put_back_item', 'take_out_item', 'screen_interaction']
	label2idx = {v: i for i, v in enumerate(classes)}
	idx2label = {i: v for i, v in enumerate(classes)}
	print('label2idx: ', label2idx)
	idx2camera = {'_1.mp4': 'camera1', '_2.mkv': 'camera2', '_3.mp4': 'camera3', 'all': 'all', 'random': 'random',
	              'random12': 'random12'}
	# history_file: the location where I save the fft_feature to
	history_file = f'{ROOT}/out/fft_feature/datasets_users.dat'
	# if os.path.exists(history_file): os.remove(history_file)
	if not os.path.exists(history_file):
		history = get_data(root_dir, cameras=['_1.mp4', '_2.mkv', '_3.mp4'], classes=classes)
		check_path(history_file)
		dump(history, history_file)
	else:
		history = load(history_file)
	# # plot
	is_plot = False
	if is_plot:
		for camera, vs in history.items():
			dataset, users, activities, raw_files = vs
			# pca_plot(dataset, activities, classes, title=f'{camera}')
			pca_plot1(dataset, activities, classes, users, title=f'{camera}')

	###############################################################################################################
	# 2. combine all cameras' data
	data_type = 'all'  # '_2.mkv', 'random', 'all'
	# train_x, train_y, test_x, test_y, train_raw_files, test_raw_files = split_train_test(history, data_type)
	train_x, train_y, test_x, test_y, train_raw_files, test_raw_files = split_train_test_users(history, data_type,
	                                                                                           test_users_list)
	is_merge_label = False
	if is_merge_label:
		classes = ['no_interaction', 'rest']
		label2idx = {v: i for i, v in enumerate(classes)}
		train_y = merge_label(train_y)
		test_y = merge_label(test_y)

		is_plot = False
		if is_plot:
			pca_plot2(train_x, train_y, classes, title=f'')

		print(
			f'before sampling, train: {sorted(Counter(train_y).items(), key=lambda x: x[0], reverse=False)}, tot: {len(train_y)}')
		from imblearn.over_sampling import RandomOverSampler
		ros = RandomOverSampler(random_state=42)
		train_x, train_y = ros.fit_resample(train_x, train_y)
		print(sorted(Counter(train_y).items()))
	###############################################################################################################
	# 3. normalization
	print(f'train: {sorted(Counter(train_y).items(), key=lambda x: x[0], reverse=False)}, tot: {len(train_y)}\n'
	      f'test: {sorted(Counter(test_y).items(), key=lambda x: x[0], reverse=False)},  tot: {len(test_y)}')
	print(f'train_raw_files.shape: {len(train_raw_files)}, test_raw_files.shape: {len(test_raw_files)}')
	ss = StandardScaler()
	ss.fit(train_x)
	train_x = ss.transform(train_x)
	test_x = ss.transform(test_x)

	###############################################################################################################
	# 4. Feature reduction
	print(f'before feature reduction: X_train: {train_x.shape}, y_train.shape: {train_y.shape}, X_test: {test_x.shape}')
	reduction_method = 'sir'
	if reduction_method == 'pca':
		pca = PCA(n_components=0.98)  # min(len(np.unique(train_y)) - 1, train_x.shape[1]))
		pca.fit(train_x)
		train_x = pca.transform(train_x)
		test_x = pca.transform(test_x)
	elif reduction_method == 'sir':  # sliced inverse regression
		# # Set the options for SIR
		sir = SlicedInverseRegression(n_directions=6, n_slices=len(np.unique(train_y)))
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
	# 5. build and evaluate models
	# res = {}
	# neigh = KNeighborsClassifier(n_neighbors=1)
	# neigh.fit(train_x, train_y)
	# pred_y = neigh.predict(test_x)
	# print('\nKNN')
	# # print(classification_report(test_y, pred_y))
	# cm = confusion_matrix(test_y, pred_y)
	# print(cm)
	# cm_file = make_confusion_matrix(cm, categories=label2idx.keys(), title=f'KNN-{idx2camera[data_type]}-{test_users_list}', out_dir='')
	# print(f'confusion_matrix: {cm_file}')
	# # auc = roc_auc_score(test_y, pred_y)
	# acc = accuracy_score(test_y, pred_y)
	# print(acc, accuracy_score(train_y, neigh.predict(train_x)))
	# # accs.append((keypoint, acc))

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

	# rf = RandomForestClassifier(random_state=42)
	# rf.fit(train_x, train_y)
	# pred_y = rf.predict(test_x)
	# # print(classification_report(test_y, pred_y))
	# print('\nRF')
	# cm = confusion_matrix(test_y, pred_y)
	# print(cm)
	# cm_file = make_confusion_matrix(cm, categories=label2idx.keys(), title=f'RF-{idx2camera[data_type]}-{test_users_list}', out_dir='')
	# print(f'confusion_matrix: {cm_file}')
	# acc = accuracy_score(test_y, pred_y)
	# print(acc, accuracy_score(train_y, rf.predict(train_x)))
	# # missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files, model_name = 'RF', idx2label=idx2label)
	# # print(missclassified)

	# import xgboost as xgb
	# # specify parameters via map
	# param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
	# num_round = 2
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
	# acc = accuracy_score(test_y, pred_y)
	# print(acc, accuracy_score(train_y, bst.predict(train_x)))

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
	# acc = accuracy_score(test_y, pred_y)
	# print(acc, accuracy_score(train_y, orc.predict(train_x)))

	print('\nSVM(rbf)')
	svm = SVC(kernel='rbf', random_state=42)
	svm.fit(train_x, train_y)
	pred_y = svm.predict(test_x)
	# print(classification_report(test_y, pred_y))
	cm = confusion_matrix(test_y, pred_y)
	print(cm)
	cm_file = make_confusion_matrix(cm, categories=label2idx.keys(),
	                                title=f'SVM-{idx2camera[data_type]}-{test_users_list}', out_dir='')
	print(f'confusion_matrix: {cm_file}')
	acc = accuracy_score(test_y, pred_y)
	print(acc, accuracy_score(train_y, svm.predict(train_x)))
	missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files, model_name='svm', idx2label=idx2label)
	print(missclassified)


# svm = SVC(kernel = 'linear', random_state=42)
# svm.fit(train_x, train_y)
# pred_y = svm.predict(test_x)
# # print(classification_report(test_y, pred_y))
# print(confusion_matrix(test_y, pred_y))
# acc = accuracy_score(test_y, pred_y)
# print(acc, accuracy_score(train_y, svm.predict(train_x)))


if __name__ == '__main__':
	# lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	# for i in range(1, 4, 1):
	# 	for test_users_list in combinations(lst, i):
	# 		print(f'\ntest_users_list: {test_users_list}')
	# 		try:
	# 			main(test_users_list)
	# 		except Exception as e:
	# 			print(e, test_users_list)

	for i in range(1, 10, 1):
		print(f'\n\n******test_users_list=[{i}]')
		try:
			main(test_users_list=[i])
		except Exception as e:
			print(e)

# main(test_users_list=[3])
