import os
from glob import glob


def get_3d_keypoints_demo():
	# from ar.features.video import trim
	# in_file = 'data/demo/ask_time_1_1614904536_1.mp4'
	# trim(in_file, start_time=9, end_time=13)

	# step 3: use 2d keypoints to obtain 3d_keypoints
	in_dir = 'examples/datasets/data-clean/refrigerator'  # mp4
	no_human = [f'{in_dir}/no_interaction/2/no_interaction_7_1625876038_3.mp4',
	            f'{in_dir}/no_interaction/2/no_interaction_9_1625876140_3.mp4']
	partial_human = [f'{in_dir}/no_interaction/2/no_interaction_1_1625875733_3.mp4',
	                 f'{in_dir}/no_interaction/2/no_interaction_9_1614039199_1.mp4']
	human = [f'{in_dir}/no_interaction/2/no_interaction_1_1625875733_3.mp4',
	         f'{in_dir}/no_interaction/2/no_interaction_3_1625875840_1.mp4',
	         f'{in_dir}/no_interaction/2/no_interaction_8_1625876088_1.mp4',
	         f'{in_dir}/no_interaction/2/no_interaction_7_1625876038_2.mkv',
	         f'{in_dir}/no_interaction/2/no_interaction_8_1625876088_2.mkv']
	for file_list in [no_human, partial_human, human]:
		for i, f in enumerate(sorted(file_list)):
			print(i, f)
			f_name = os.path.basename(f)
			# save the predicted 3d keypoints to out_file
			out_file = os.path.join('examples/classical_ml/out/strange', os.path.relpath(f, 'examples'))
			out_dir = os.path.dirname(out_file)
			if not os.path.exists(out_dir):
				os.makedirs(out_dir)
			viz_export = out_file  # npy
			viz_output = out_file[:-4] + '.mp4'  # pred.mp4
			cmd = f'python3.7 ar/features/videopose3d/2d_to_3d-single.py --viz-subject {f_name} --viz-video {f} --viz-export {viz_export} --viz-output {viz_output}'
			os.system(cmd)
		# return
	print('finished')


def get_3d_keypoints():
	# step 3: use 2d keypoints to obtain 3d_keypoints
	in_dir = 'examples/datasets/data-clean/refrigerator/'  # mp4
	camera1 = glob(in_dir + '/**/*1.mp4', recursive=True)
	camera2 = glob(in_dir + '/**/*2.mkv', recursive=True)
	camera3 = glob(in_dir + '/**/*3.mp4', recursive=True)
	camera32 = glob(in_dir + '/**/*3 2.mp4', recursive=True)
	print(f'camera1: {len(camera1)},camera2: {len(camera2)}, camera3: {len(camera3)}, and camera32: {len(camera32)}')
	for file_list in [camera1, camera2, camera3]:
		for i, f in enumerate(sorted(file_list)):
			print(i, f)
			f_name = os.path.basename(f)
			# save the predicted 3d keypoints to out_file
			out_file = os.path.join('examples/classical_ml/out/keypoints3d', os.path.relpath(f, 'examples'))
			out_dir = os.path.dirname(out_file)
			if not os.path.exists(out_dir):
				os.makedirs(out_dir)
			viz_export = out_file  # npy
			# viz_output = out_file[:-3]    # pred.mp4
			cmd = f'python3.7 ar/features/videopose3d/2d_to_3d-single.py --viz-subject {f_name} --viz-video {f} --viz-export {viz_export} --viz-output {viz_export}'
			os.system(cmd)
			return
	print('finished')


if __name__ == '__main__':
	# get_3d_keypoints()
	get_3d_keypoints_demo()
