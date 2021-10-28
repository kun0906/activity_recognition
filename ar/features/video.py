"""
    API for video process

"""

import cv2
import os

from moviepy.tools import subprocess_call
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def trim(in_file, start_time=0, end_time=0, out_dir=None):
    """

    Parameters
    ----------
    in_file: input video
    start_time:
    end_time
    out_dir

    Returns
    -------

    """

    f, ext = os.path.splitext(os.path.basename(in_file))
    out_file = os.path.join(out_dir, f'trim-{f}{ext}')
    if os.path.exists(out_file):
        return out_file

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #
    # Not that moviepy does not work in my MACOS (moviepy will call imageio_ffmpeg(has its own ffmpeg))
    # ffmpeg_extract_subclip(in_file, start_time, end_time, targetname=out_file)

    # directly call ffmpeg (installed with brew)
    t1 = start_time
    t2 = end_time
    cmd = ['ffmpeg', "-y",
           "-ss", "%0.2f" % t1,
           "-i", in_file,
           "-t", "%0.2f" % (t2 - t1),
           "-map", "0", "-vcodec", "copy", "-acodec", "copy", out_file]
    print(' '.join(cmd))
    subprocess_call(cmd)
    return out_file


import cv2
import os


def get_info(in_file):
	print(in_file)
	# capture video
	cap = cv2.VideoCapture(in_file)

	# # Get the Default resolutions
	# # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
	# fourcc = cv2.VideoWriter_fourcc(*'mp4v'.lower())
	# # fourcc = cv2.VideoWriter_fourcc(
	# #     *f"{fourcc & 255:c},{(fourcc >> 8) & 255:c}, {(fourcc >> 16) & 255:c}, {(fourcc >> 24) & 255:c}")
	# fourcc = cv2.VideoWriter_fourcc(*(chr(fourcc & 0xff) + chr((fourcc >> 8) & 0xff) + chr((fourcc >> 16) & 0xff)
	#                                   + chr((fourcc >> 24) & 0xff)))
	# print(fourcc)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	# Define the codec and filename.
	fps = cap.get(cv2.CAP_PROP_FPS)
	# get duration
	totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	if fps == 0:
		duration = 0
	else:
		duration = float(totalNoFrames) / float(fps)
	# out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
	# out = cv2.VideoWriter(out_file, fourcc, fps, (frame_width, frame_height), isColor=True)
	print(f'fps: {fps}, duration: {duration}, (height, width): ({frame_height}, {frame_width})')
	cap.release()
	cv2.destroyAllWindows()

	# ########
	# cmd = f'ffmpeg -i {in_file}'
	# os.system(cmd)
	video_info = {'fps': fps, 'duration': duration, 'tot': totalNoFrames}
	return video_info


#
# if __name__ == '__main__':
#     in_file = 'data/data-clean/refrigerator/take_out_item/1/take_out_item_1_1613085820_1.mp4'
#     get_info(in_file)
#     in_file = 'data/data-clean/refrigerator/take_out_item/1/take_out_item_1_1613085820_2.mkv'
#     get_info(in_file)
#     in_file = 'data/data-clean/refrigerator/open_close_fridge/7/open_close_fridge_1_1626047757_3.mp4'
#     get_info(in_file)


if __name__ == '__main__':
	in_file = 'data/demo/ask_time_1_1614904536_1.mp4'
	trim(in_file, start_time=9, end_time=13)
