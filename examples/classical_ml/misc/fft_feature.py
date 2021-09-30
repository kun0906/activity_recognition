import glob

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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


def _get_fft(vs, fft_bin=None):
	return np.real(np.fft.fft(vs, n=fft_bin))


def get_fft_features_backup(npy_file='', m=60, keypoint=7, coordinate=1):
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
	data = np.load(npy_file)
	data = data[:, keypoint, coordinate]
	data = data.reshape((-1,))
	n = len(data)
	step = int(np.ceil(n / m))
	fft_features = [np.mean(_get_fft(data[i:i + step])) for i in range(0, len(data), step)]
	fft_features = fft_features + [0] * (m - len(fft_features))
	return np.asarray(fft_features).reshape(-1, )


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
			# fft_features.extend([np.mean(tmp)])
			# n_feat = 1
			fft_features.extend([np.mean(tmp), np.std(tmp)])
			n_feat = 2
		fft_features = fft_features + [0] * (n_feat * m - len(fft_features))
		res.append(fft_features)
	return np.asarray(res).reshape(-1, )


def get_all_data(root_dir='examples/out/keypoints3d-20210907/keypoints3d/data', keypoint_idx=10):
	classes = ['open_close_fridge', 'put_back_item', 'screen_interaction', 'take_out_item', 'no_interaction']

	dataset = []
	users = []
	activities = []

	for act in classes:
		files = glob.glob(f'{root_dir}/data-clean/refrigerator/' + act + '/*/*_2.mkv.npy')
		# files = [f'{root_dir}/data-clean/refrigerator/take_out_item/4/take_out_item_2_1616179391_1.mp4.npy',
		#          f'{root_dir}/data-clean/refrigerator/take_out_item/4/take_out_item_2_1616179391_1.mp4.npy']
		for file in files:
			# print('Processing file', file)
			user = int(file.split('/')[-2])
			data = get_fft_features(file, keypoint=keypoint_idx)
			# data = special_keypoints(np.load(file))

			dataset.append(data)
			users.append(user)
			activities.append(classes.index(act))
		print(act, len(files))

	return np.array(dataset), np.array(users), np.array(activities)


accs = []
rf_accs = []
for idx, keypoint in enumerate(keypoints):
	print(f'\n***keypoint_idx: {idx}, {keypoint}')
	dataset, users, activities = get_all_data(keypoint_idx=idx)

	from sklearn.decomposition import PCA

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')

	pca = PCA(n_components=2)
	X = pca.fit_transform(dataset)

	for i in range(5):
		plt.scatter(X[activities == i, 0], X[activities == i, 1])
		plt.title(f'{keypoint}')

	plt.legend(['open_close_fridge', 'put_back_item', 'screen_interaction', 'take_out_item', 'no_interaction'])
	plt.show()

	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

	pca = PCA(n_components=100)
	dataset = pca.fit_transform(dataset)
	# train_x, test_x, train_y, test_y = train_test_split(dataset, users, test_size=0.3, random_state=42)
	train_x, test_x, train_y, test_y = train_test_split(dataset, activities, test_size=0.3, random_state=42)
	ss = StandardScaler()
	ss.fit(train_x)
	train_x = ss.transform(train_x)
	test_x = ss.transform(test_x)
	print(f'X_train: {train_x.shape}, X_test: {test_x.shape}')
	neigh = KNeighborsClassifier(n_neighbors=1)
	neigh.fit(train_x, train_y)
	pred_y = neigh.predict(test_x)
	print(classification_report(test_y, pred_y))
	print(confusion_matrix(test_y, pred_y))
	# auc = roc_auc_score(test_y, pred_y)
	acc = accuracy_score(test_y, pred_y)
	print(acc)
	accs.append((keypoint, acc))

	rf = RandomForestClassifier(random_state=42)
	rf.fit(train_x, train_y)
	pred_y = rf.predict(test_x)
	print(classification_report(test_y, pred_y))
	print(confusion_matrix(test_y, pred_y))
	acc = accuracy_score(test_y, pred_y)
	print(acc)
	rf_accs.append((keypoint, acc))

accs.sort(key=lambda x: x[1], reverse=True)
print(f'\n\n{accs}')
rf_accs.sort(key=lambda x: x[1], reverse=True)
print(f'\n\n{rf_accs}')

#
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
