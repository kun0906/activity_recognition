""" get keypoints2d_probs

	detectron2 output:
	pred_score is not probability
	https://detectron2.readthedocs.io/en/latest/tutorials/models.html?highlight=output#model-output-format
"""
import numpy as np


def main():
	in_file = 'examples/out-/data_2d_custom_2d_keypoints-with-probs.npz'
	raw_data = np.load(in_file, allow_pickle=True)
	print(list(raw_data))
	# print(raw_data['keypoints2d'])
	keypoints2d = raw_data['keypoints2d'].tolist()
	bb = []
	kp = []
	for k, vs in keypoints2d.items():
		vs = vs[0]
		# vs = {start_frame: 0 , end_frame: 100, bounding_boxes: [], bounding_boxes_probs: [],
		# keypoints: [2d] , keypoints_probs:[]}
		bb.extend(vs['bounding_boxes_probs'].tolist())

		tmp = []
		for vs_ in vs['keypoints_probs'].tolist():
			tmp.extend(vs_)
		kp.extend(tmp)
		line = [k, vs['bounding_boxes_probs'], vs['keypoints_probs']]
		print(line)

	# bb = [ 0 if str(v) == 'nan' else v for v in bb]
	# kp = [0 if str(v) == 'nan' else v for v in kp]
	bb = [v for v in bb if str(v) != 'nan']
	kp = [v for v in kp if str(v) != 'nan']
	qs = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]
	print([f'{v:.3}' for v in np.quantile(bb, q=qs)])
	# the score can be larger than 1: https://github.com/facebookresearch/detectron2/issues/319
	print([f'{v:.3}' for v in np.quantile(kp, q=qs)])


def single_main(file_lst):
	in_file = 'examples/out-/data_2d_custom_2d_keypoints-with-probs.npz'
	raw_data = np.load(in_file, allow_pickle=True)
	print(list(raw_data))
	# print(raw_data['keypoints2d'])
	keypoints2d = raw_data['keypoints2d'].tolist()
	bb = []
	kp = []
	for k, vs in keypoints2d.items():
		if k in file_lst:
			vs = vs[0]
			# vs = {start_frame: 0 , end_frame: 100, bounding_boxes: [], bounding_boxes_probs: [],
			# keypoints: [2d] , keypoints_probs:[]}
			bb.extend(vs['bounding_boxes_probs'].tolist())

			tmp = []
			for vs_ in vs['keypoints_probs'].tolist():
				tmp.extend(vs_)
			kp.extend(tmp)
			line = [k, vs['bounding_boxes_probs'], vs['keypoints_probs']]
			print(line)

	# bb = [ 0 if str(v) == 'nan' else v for v in bb]
	# kp = [0 if str(v) == 'nan' else v for v in kp]
	bb = [v for v in bb if str(v) != 'nan']
	kp = [v for v in kp if str(v) != 'nan']
	qs = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]
	print([f'{v:.3}' for v in np.quantile(bb, q=qs)])
	# the score can be larger than 1: https://github.com/facebookresearch/detectron2/issues/319
	print([f'{v:.3}' for v in np.quantile(kp, q=qs)])


if __name__ == '__main__':
	# main()
	# file_lst = ['no_interaction_4_1616182615_1.mp4']    'no_interaction -> predicted as open_close_fridge'
	file_lst = ['open_close_fridge_6_1625877626_1.mp4']  # 'open_close_fridge-> predicted as no_interaction'
	file_lst = ['no_interaction_6_1614039030_1.mp4']  # predict correctly
	single_main(file_lst)
