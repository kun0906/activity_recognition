"""
    API for video process

"""

import cv2
import os

def trim(in_file, start_time, end_time, out_file=None):
    """

    Parameters
    ----------
    in_file: input video
    start_time:
    end_time
    out_file

    Returns
    -------

    """
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    if not out_file:
        out_file = os.path.splitext(in_file)[0] + '-trim.mp4'
    ffmpeg_extract_subclip(in_file, start_time, end_time, targetname=out_file)



