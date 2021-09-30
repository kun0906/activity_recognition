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


if __name__ == '__main__':
    in_file = 'data/demo/ask_time_1_1614904536_1.mp4'
    trim(in_file, start_time=9, end_time=13)
