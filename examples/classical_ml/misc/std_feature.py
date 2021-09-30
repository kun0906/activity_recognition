import numpy as np
import matplotlib.pyplot as plt
import glob

# scales the vector to have n timesteps
from sklearn.ensemble import RandomForestClassifier


def special_keypoints(data, keypoint_name='left ear'):
	"""
	17 Keypoints detail
		"keypoints": [
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
		"Right_ankle"
		]

https://github.com/facebookresearch/detectron2/issues/1373


	Parameters
	----------
	data
	keypoint_name

	Returns
	-------

	"""
	N = data.shape[0]

	if keypoint_name == 'left ear':
		idx = 3
	# data = data[:, idx:idx+2, :]

	# normalize data
	# norm_max, norm_min = np.min(data, axis=0)
	# print(norm_max, norm_min)
	# data = data - norm_min#)/norm_max
	# print(norm_max)
	n = 50
	intervals = np.linspace(0, N, n)
	output = np.zeros((n, data.shape[1], data.shape[2]))

	for i, end in enumerate(intervals[1:]):
		boundary = int(end)

		if i > 0:
			start = int(intervals[i - 1])
		else:
			start = 0

		output[i, :, :] = np.std(data[start:boundary, :, :], axis=0)

	return output.reshape(-1)


# scales the vector to have n timesteps
def linscale(data, n=50):
	N = data.shape[0]

	# normalize data
	# norm_max, norm_min = np.min(data, axis=0)
	# print(norm_max, norm_min)
	# data = data - norm_min#)/norm_max
	# print(norm_max)

	intervals = np.linspace(0, N, n)
	output = np.zeros((n, data.shape[1], data.shape[2]))

	for i, end in enumerate(intervals[1:]):
		boundary = int(end)

		if i > 0:
			start = int(intervals[i - 1])
		else:
			start = 0

		output[i, :, :] = np.std(data[start:boundary, :, :], axis=0)

	return output.reshape(-1)


# gets all the data annotates with user and activity
def get_all_data(root_dir='examples/out/keypoints3d-20210907/keypoints3d/data'):
	classes = ['open_close_fridge', 'put_back_item', 'screen_interaction', 'take_out_item', 'no_interaction']

	dataset = []
	users = []
	activities = []

	for act in classes:
		files = glob.glob(f'{root_dir}/data-clean/refrigerator/' + act + '/*/*.mkv.npy')
		# files = [f'{root_dir}/data-clean/refrigerator/take_out_item/4/take_out_item_2_1616179391_1.mp4.npy',
		#          f'{root_dir}/data-clean/refrigerator/take_out_item/4/take_out_item_2_1616179391_1.mp4.npy']
		for file in files:
			# print('Processing file', file)
			user = int(file.split('/')[-2])
			data = linscale(np.load(file))
			# data = special_keypoints(np.load(file))

			dataset.append(data)
			users.append(user)
			activities.append(classes.index(act))
		print(act, len(files))

	return np.array(dataset), np.array(users), np.array(activities)


dataset, users, activities = get_all_data()

from sklearn.decomposition import PCA

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

pca = PCA(n_components=2)
X = pca.fit_transform(dataset)

for i in range(5):
	plt.scatter(X[activities == i, 0], X[activities == i, 1])

plt.legend(['open_close_fridge', 'put_back_item', 'screen_interaction', 'take_out_item', 'no_interaction'])
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

train_x, test_x, train_y, test_y = train_test_split(dataset, users, test_size=0.3, random_state=42)
# train_x, test_x, train_y, test_y = train_test_split(dataset, activities, test_size=0.3, random_state=42)
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_x, train_y)
pred_y = neigh.predict(test_x)
print(classification_report(test_y, pred_y))
print(confusion_matrix(test_y, pred_y))

#
# rf = RandomForestClassifier(random_state=42)
# rf.fit(train_x, train_y)
# pred_y = rf.predict(test_x)
# print(classification_report(test_y, pred_y))
# print(confusion_matrix(test_y, pred_y))
