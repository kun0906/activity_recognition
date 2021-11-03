import glob
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import glob
from collections import defaultdict, Counter

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import os


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          out_dir='.'):
	'''
	This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
	Arguments
	---------
	cf:            confusion matrix to be passed in
	group_names:   List of strings that represent the labels row by row to be shown in each square.
	categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
	count:         If True, show the raw number in the confusion matrix. Default is True.
	normalize:     If True, show the proportions for each category. Default is True.
	cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
				   Default is True.
	xyticks:       If True, show x and y ticks. Default is True.
	xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
	sum_stats:     If True, display summary statistics below the figure. Default is True.
	figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
	cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
				   See http://matplotlib.org/examples/color/colormaps_reference.html

	title:         Title for the heatmap. Default is None.
	'''

	# CODE TO GENERATE TEXT INSIDE EACH SQUARE
	blanks = ['' for i in range(cf.size)]

	if group_names and len(group_names) == cf.size:
		group_labels = ["{}\n".format(value) for value in group_names]
	else:
		group_labels = blanks

	if count:
		group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
	else:
		group_counts = blanks

	if percent:
		row_sum = np.sum(cf, axis=1)
		cf_row_sum = np.array([[value] * len(row_sum) for value in row_sum]).flatten()
		#         print(cf_row_sum)
		group_percentages = ["({0:.2%})".format(value) for value in cf.flatten() / cf_row_sum]
	else:
		group_percentages = blanks

	box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
	box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

	# CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
	if sum_stats:
		# Accuracy is sum of diagonal divided by total observations
		accuracy = np.trace(cf) / float(np.sum(cf))

		# if it is a binary confusion matrix, show some more stats
		if len(cf) == 2:
			# Metrics for Binary Confusion Matrices
			precision = cf[1, 1] / sum(cf[:, 1])
			recall = cf[1, 1] / sum(cf[1, :])
			f1_score = 2 * precision * recall / (precision + recall)
			stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
				accuracy, precision, recall, f1_score)
		else:
			stats_text = "\n\nAccuracy={:0.2f}".format(accuracy)
	else:
		stats_text = ""
	print(stats_text)
	# SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
	if figsize == None:
		# Get default figure size if not set
		figsize = plt.rcParams.get('figure.figsize')

	if xyticks == False:
		# Do not show categories if xyticks is False
		categories = False

	# MAKE THE HEATMAP VISUALIZATION
	plt.figure(figsize=figsize)
	sns.heatmap(cf / np.sum(cf, axis=1), annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories,
	            yticklabels=categories)
	plt.yticks(rotation=0)
	plt.xticks(rotation=90)
	if xyplotlabels:
		plt.ylabel('True label')
		plt.xlabel('Predicted label' + stats_text)
	else:
		plt.xlabel(stats_text)

	if title:
		plt.title(title)
	out_file = os.path.join(out_dir, f'{title}-confusion_matrix')
	# print(out_file)
	plt.savefig(out_file, bbox_inches='tight', dpi=300)
	plt.show()
	return out_file


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


def _get_fft(vs, fft_bin=None, fft_type='magnitude'):
	out = np.fft.fft(vs, n=fft_bin)
	if fft_type == 'phase':
		out = np.angle(out)  # phase
	else:
		# fft_type =='magnitude':
		out = np.abs(out)
	return list(out)


#
# def get_fft_features_backup(npy_file='', m=60, keypoint=7, coordinate=1):
# 	"""
#
# 	Parameters
# 	----------
# 	npy_file
# 	m:
# 	   without trimmming: m = 84
# 	   with trimming: m = 51
# 	keypoint
# 	coordinate
#
# 	Returns
# 	-------
#
# 	"""
# 	data = np.load(npy_file)
# 	data = data[:, keypoint, coordinate]
# 	data = data.reshape((-1,))
# 	n = len(data)
# 	step = int(np.ceil(n / m))
# 	fft_features = [np.mean(_get_fft(data[i:i + step])) for i in range(0, len(data), step)]
# 	fft_features = fft_features + [0] * (m - len(fft_features))
# 	return np.asarray(fft_features).reshape(-1, )

def get_missclassified(cm, test_y, pred_y, test_raw_files):
	res = defaultdict(list)
	n, m = cm.shape
	for i, (y_, y_2) in enumerate(zip(test_y, pred_y)):
		if y_ != y_2:
			res[y_].append(test_raw_files[i])

	return res


def get_fft_features(npy_file='', m=84, keypoint=7):
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
	raw_data = np.load(npy_file)
	res = []
	for coordinate in range(3):
		data = raw_data[:, keypoint, coordinate]
		data = data.reshape((-1,))
		n = len(data)
		step = int(np.ceil(n / m))
		fft_features = []
		for i in range(0, len(data), step):
			tmp = _get_fft(data[i:i + step])
			fft_features.extend([np.mean(tmp)])
			n_feat = 1
		# fft_features.extend([np.mean(tmp), np.std(tmp)])
		# n_feat = 2
		# fft_features.extend(np.quantile(tmp, q=[0.5]).tolist())
		# n_feat = 1
		fft_features = fft_features + [0] * (n_feat * m - len(fft_features))
		res.append(fft_features)
	return np.asarray(res).reshape(-1, )


def get_all_data(root_dir='examples/out/keypoints3d-20210907/keypoints3d/data'):
	classes = ['open_close_fridge', 'put_back_item', 'screen_interaction', 'take_out_item', 'no_interaction']

	dataset = []
	users = []
	activities = []
	raw_files = []
	for act in classes:
		files = glob.glob(f'{root_dir}/data-clean/refrigerator/' + act + '/*/*_2.mkv.npy')
		# files = [f'{root_dir}/data-clean/refrigerator/take_out_item/4/take_out_item_2_1616179391_1.mp4.npy',
		#          f'{root_dir}/data-clean/refrigerator/take_out_item/4/take_out_item_2_1616179391_1.mp4.npy']
		for file in files:
			# print('Processing file', file)
			user = int(file.split('/')[-2])

			data = []
			for keypoint_idx, _ in enumerate(keypoints):
				tmp = get_fft_features(file, keypoint=keypoint_idx)
				# data = special_keypoints(np.load(file))
				data.extend(list(tmp))
			data = np.asarray(data)

			dataset.append(data)
			users.append(user)
			activities.append(classes.index(act))
			raw_files.append(file)
		print(act, len(files))

	return np.array(dataset), np.array(users), np.array(activities), raw_files


accs = []
rf_accs = []
dataset, users, activities, raw_files = get_all_data()

from sklearn.decomposition import PCA

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

pca = PCA(n_components=2)
X = pca.fit_transform(dataset)

for i in range(5):
	plt.scatter(X[activities == i, 0], X[activities == i, 1])
	plt.title(f'all')

plt.legend(['open_close_fridge', 'put_back_item', 'screen_interaction', 'take_out_item', 'no_interaction'])
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#
# pca = PCA(n_components=100)
# dataset = pca.fit_transform(dataset)

# train_x, test_x, train_y, test_y = train_test_split(dataset, users, test_size=0.3, random_state=42)
train_x, test_x, train_y, test_y, train_raw_files, test_raw_files = train_test_split(dataset, activities,
                                                                                     raw_files, test_size=0.3,
                                                                                     random_state=42)
# train_y, train_raw_files = list(zip(*train_y))
# test_y, test_raw_files = list(zip(*test_y))
ss = StandardScaler()
ss.fit(train_x)
train_x = ss.transform(train_x)
test_x = ss.transform(test_x)

print(f'X_train: {train_x.shape}, X_test: {test_x.shape}')

print(f'train: {Counter(train_y)}, test: {Counter(test_y)}')

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_x, train_y)
pred_y = neigh.predict(test_x)
print(classification_report(test_y, pred_y))
print(confusion_matrix(test_y, pred_y))
# auc = roc_auc_score(test_y, pred_y)
acc = accuracy_score(test_y, pred_y)
print(acc, accuracy_score(train_y, neigh.predict(train_x)))
# accs.append((keypoint, acc))


label2idx = {'open_close_fridge': 0, 'put_back_item': 1, 'screen_interaction': 2,
             'take_out_item': 3, 'no_interaction': 4}

for v in range(1, 20):
	print(f'min_samples_leaf: {v}')
	rf = RandomForestClassifier(random_state=42, min_samples_leaf=v)
	rf.fit(train_x, train_y)
	pred_y = rf.predict(test_x)
	print(classification_report(test_y, pred_y))
	cm = confusion_matrix(test_y, pred_y)
	print(cm)
	cm_file = make_confusion_matrix(cm, categories=label2idx.keys(), title='RF', out_dir='')
	print(f'confusion_matrix: {cm_file}')
	acc = accuracy_score(test_y, pred_y)
	print(acc, accuracy_score(train_y, rf.predict(train_x)))
	missclassified = get_missclassified(cm, test_y, pred_y, test_raw_files)
	print(missclassified)

	break

#
# #
# #
# dt = DecisionTreeClassifier(random_state=42)
# dt.fit(train_x, train_y)
# pred_y = dt.predict(test_x)
# print(classification_report(test_y, pred_y))
# print(confusion_matrix(test_y, pred_y))
# acc = accuracy_score(test_y, pred_y)
# print(acc, accuracy_score(train_y, dt.predict(train_x)))
#


#
orc = OneVsRestClassifier(LogisticRegression(C=1, random_state=42, solver='liblinear'))
orc.fit(train_x, train_y)
pred_y = orc.predict(test_x)
print(classification_report(test_y, pred_y))
cm = confusion_matrix(test_y, pred_y)
print(cm)
cm_file = make_confusion_matrix(cm, categories=label2idx.keys(), title='onevsrest', out_dir='')
print(f'confusion_matrix: {cm_file}')
acc = accuracy_score(test_y, pred_y)
print(acc, accuracy_score(train_y, orc.predict(train_x)))

svm = SVC(kernel='rbf', random_state=42)
svm.fit(train_x, train_y)
pred_y = svm.predict(test_x)
print(classification_report(test_y, pred_y))
print(confusion_matrix(test_y, pred_y))
acc = accuracy_score(test_y, pred_y)
print(acc, accuracy_score(train_y, svm.predict(train_x)))

svm = SVC(kernel='linear', random_state=42)
svm.fit(train_x, train_y)
pred_y = svm.predict(test_x)
print(classification_report(test_y, pred_y))
print(confusion_matrix(test_y, pred_y))
acc = accuracy_score(test_y, pred_y)
print(acc, accuracy_score(train_y, svm.predict(train_x)))

# rf_accs.append((keypoint, acc))
#
# accs.sort(key=lambda x: x[1], reverse=True)
# print(f'\n\n{accs}')
# rf_accs.sort(key=lambda x: x[1], reverse=True)
# print(f'\n\n{rf_accs}')

# #
# rootdir0 = 'no_interaction'
# rootdir1 = 'open_close_fridge'
# rootdir2 = 'put_back_item'
# rootdir3 = 'screen_interaction'
# rootdir4 = 'take_out_item'
#
# rootdirs =[ rootdir0, rootdir1, rootdir2, rootdir3, rootdir4]
# #keypt = 10 # right wrist
#
# #keypt = 8 # right elbow
# keypt = 2 # "right_eye"
# keypt = 4 # "right_ear"
# keypt = 6 # right shoulder
# keypt = 12 # "right_hip"
# keypt = 14 # "right_knee"
# #keypt = 16 # "right_ankle"
#
# kypt_loc = 3*keypt
# kypt_y = kypt_loc+1
# kypt_z =kypt_loc+2
#
#
# def trim(file_path):
#    x = np.load(file_path)
#    n = x.shape[0]
#    x = x.reshape(n, -1)
#
#    is_trimming = True
#
#    if is_trimming:
#       thres = []
#       # find threshold
#       window = []
#       for i in range(1, n):
#          thres.append(sum(v ** 2 for v in x[i] - x[i - 1]))
#       thres = [float(f'{v:.2f}') for v in thres]
#       # print(file_path, thres)
#       # quant = np.quantile([float(v) for v in thres], q = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1])
#       # print(quant)
#       # left, right = quant[1], quant[7]      # [left, right]
#       # left, right = quant[1], quant[7]  # [left, right]
#       res = []
#       # only extract the frame (3d keyponts) that has human
#       entry = False
#       entry_idx = 0
#       exit_idx = n
#       for i in range(1, n - 1):
#          if not entry and thres[i] > sum(
#                  thres[:i]):  # find the entry frame (if current thre > sum of previous 5 thres)
#             res.append(x[i])
#             entry_idx = i
#             entry = True
#             # print(entry_idx, thres[i])
#          elif entry and thres[i] > sum(thres[
#                                        i + 1:]):  # thres[i] > sum(thres[i+1:i+5+1]) and thres[i+1] == 0.0: # find the entry frame (if current thre > sum of future 5 thres)
#             exit_idx = i
#             # print(exit_idx, thres[i])
#             break
#       if exit_idx == n:
#          # print(thres)
#          exit_idx = n
#       res = x[entry_idx: exit_idx]
#       print(f'{file_path} entry_idx: {entry_idx}, exit_idx: {exit_idx}, tot: {n}')
#    else:
#       res = x
#    return res
#
#
# def plot_average_person():
#    # sept 11: average over all persons
#    # set max frame # to 1000
#
#
#    persons = [1, 2, 3, 4, 5, 7, 8, 9]  # (0) no interaction & (3) screen interaction -1,
#
#    # camera 3 for rootdir1 is [2,5,7,8,9], rootdir2, rootdir3 is [2,5,7,8,9]
#    # camera 3 for rootdir0, rootdir4 is [2,5, 7,8,9]
#
#    x_tot = {}
#    y_tot = {}
#    z_tot = {}
#
#    x_std = {}
#    y_std = {}
#    z_std = {}
#
#    x_person_to_ave = []
#    y_person_to_ave = []
#    z_person_to_ave = []
#
#    x_std_all = []
#    y_std_all = []
#    z_std_all = []
#
#    ind = 4
#
#    rootdiri = rootdirs[ind]
#
#    smax_t = 1000
#    # same as above but save matrices after knowing the max # rows/frames
#    for p in persons:
#       # make map of matrix results
#       matmap = {}
#
#       rootdir = rootdiri + "/" + str(p)
#
#       nfiles = 0
#       for subdir, dirs, files in os.walk(rootdir):
#          for file in files:
#             npath = os.path.join(subdir, file)
#
#             if file.endswith(("2.mkv.npy")):
#                print(npath)
#                nfiles = nfiles + 1
#
#                trimmed_res = trim(npath)
#                # shape
#                s0 = np.shape(trimmed_res)[0]
#
#                sh = np.shape(trimmed_res)
#                print("shape is", sh)
#
#                if s0 > smax_t:
#                   trimmed_res = trimmed_res[:smax_t]
#
#                if s0 < smax_t:
#                   z51 = np.zeros(51)
#                   # append NA or -inf
#
#                   minus = smax_t - s0
#                   for i in range(minus):
#                      trimmed_res = np.vstack((trimmed_res, z51))
#
#                matmap[nfiles] = trimmed_res
#
#       xseries_tot = []
#       yseries_tot = []
#       zseries_tot = []
#
#       numvideos = nfiles
#
#       for i in range(1, numvideos):  # numvideos):
#
#          video_mat = matmap[i]
#
#          xseries = []
#          yseries = []
#          zseries = []
#          for row in video_mat:
#             # use indices to locate keypt x,y,z
#             rowx = row[kypt_loc]
#             rowy = row[kypt_y]
#             rowz = row[kypt_z]
#             xseries.append(rowx)
#             yseries.append(rowy)
#             zseries.append(rowz)
#
#          """if i==1:
#              xseries_tot = xseries
#              yseries_tot = yseries
#              zseries_tot = zseries
#              continue"""
#
#          # take average
#
#          # print("xseries prev", xseries_tot)
#          # print("xseries curr", xseries )
#
#          xseries_tot.append(xseries)
#          yseries_tot.append(yseries)
#          zseries_tot.append(zseries)
#
#       samp_data_x = np.array(xseries_tot)
#       xaved = np.average(samp_data_x, axis=0)
#
#       stdx = np.std(samp_data_x, axis=0)
#
#       samp_data_y = np.array(yseries_tot)
#       yaved = np.average(samp_data_y, axis=0)
#
#       stdy = np.std(samp_data_y, axis=0)
#
#       samp_data_z = np.array(zseries_tot)
#       zaved = np.average(samp_data_z, axis=0)
#
#       stdz = np.std(samp_data_z, axis=0)
#
#       x_tot[p] = xaved
#       y_tot[p] = yaved
#       z_tot[p] = zaved
#
#       x_std[p] = stdx
#       y_std[p] = stdy
#       z_std[p] = stdz
#
#       x_person_to_ave.append(xaved)
#       y_person_to_ave.append(yaved)
#       z_person_to_ave.append(zaved)
#
#       x_std_all.append(stdx)
#       y_std_all.append(stdy)
#       z_std_all.append(stdz)
#
#    # average over all persons
#    x_to_ave_samp = np.array(x_person_to_ave)
#    y_to_ave_samp = np.array(y_person_to_ave)
#    z_to_ave_samp = np.array(z_person_to_ave)
#
#    x_allperson_aved = np.average(x_to_ave_samp, axis=0)
#    y_allperson_aved = np.average(y_to_ave_samp, axis=0)
#    z_allperson_aved = np.average(z_to_ave_samp, axis=0)
#
#    x_to_ave_std = np.array(x_std_all)
#    y_to_ave_std = np.array(y_std_all)
#    z_to_ave_std = np.array(z_std_all)
#
#    x_std_aved = np.average(x_to_ave_std, axis=0)
#    y_std_aved = np.average(y_to_ave_std, axis=0)
#    z_std_aved = np.average(z_to_ave_std, axis=0)
#
#    std_x_aved = np.std(x_to_ave_samp, axis=0)
#    std_y_aved = np.std(y_to_ave_samp, axis=0)
#    std_z_aved = np.std(z_to_ave_samp, axis=0)
#
#    #ind = 3
#    x_allperson_averaged[ind] = x_allperson_aved
#    y_allperson_averaged[ind] =y_allperson_aved
#    z_allperson_averaged[ind] =z_allperson_aved
#    x_allperson_std[ind] = std_x_aved
#    y_allperson_std[ind] = std_y_aved
#    z_allperson_std[ind] =std_z_aved
#
#    # sept 13 plot
#
#    # combine into 1 errorbar plot
#
#    activities = [0, 1, 2, 3, 4]
#
#    fig, ax = plt.subplots(3, 1, figsize=(12, 6))
#
#    timevec = np.arange(smax_t)
#
#    # x
#    for a in activities:
#       print(a)
#       xseries_t = x_allperson_averaged[a]
#       xstd_t = x_allperson_std[a]
#       # uplims=True, lolims=True,
#       ax[0].errorbar(timevec, xseries_t, yerr=xstd_t, alpha=0.5,
#                      capsize=3)  # mark markersize=1) #uplims=True, lolims=True)
#       yseries_t = y_allperson_averaged[a]
#       ystd_t = y_allperson_std[a]
#       zseries_t = z_allperson_averaged[a]
#       zstd_t = z_allperson_std[a]
#       ax[1].errorbar(timevec, yseries_t, yerr=ystd_t, alpha=0.5, capsize=3)  # , marker=".", markersize=1)
#       ax[2].errorbar(timevec, zseries_t, yerr=zstd_t, alpha=0.5, capsize=3)  # , marker=".", markersize=1)
#
#       # plt.scatter(timevec, xseries_t)
#
#    ax[0].set_ylabel("x coordinate", fontsize=10)
#    ax[1].set_ylabel("y coordinate", fontsize=10)
#    ax[2].set_xlabel("frame number", fontsize=10)
#    ax[2].set_ylabel("z coordinate", fontsize=10)
#
#    ax[0].set_title("right knee, camera 2")
#
#    """ax[0].set_xlim((0,200))
#    ax[1].set_xlim((0,200))
#    ax[2].set_xlim((0,200))"""
#
#    # plt.xlim((0, 200))
#
#
#    labels = ["no interaction", "open/close fridge", "put back item", "screen interaction", "take out item"]
#
#    fig.legend(labels, loc='upper right')
#
#    plt.show()
#
