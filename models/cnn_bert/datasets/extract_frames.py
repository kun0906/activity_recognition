'''

https://github.com/craston/MARS/blob/master/README.md



For HMDB51 and UCF101 datasets:
Code extracts frames from video at a rate of 25fps and scaling the
larger dimension of the frame is scaled to 256 pixels.
After extraction of all frames write a "done" file to signify proper completion
of frame extraction.
Usage:
  python extract_frames.py video_dir frame_dir 0 1

  video_dir => path of video files
  frame_dir => path of extracted jpg frames
'''

import sys, os, pdb
import numpy as np
import subprocess
from tqdm import tqdm
import patoolib
import shutil


def extract(vid_dir, frame_dir, start, end, redo=False):
    class_list = sorted(os.listdir(vid_dir))[start:end]

    print("Classes =", class_list)

    for ic, cls in enumerate(class_list):
        if cls == '.DS_Store': continue
        vlist = sorted(os.listdir(os.path.join(vid_dir, cls)))
        print("")
        print(ic + 1, len(class_list), cls, len(vlist))
        print("")
        for v in tqdm(vlist):
            outdir = os.path.join(frame_dir, cls, v[:-4])

            # Checking if frames already extracted
            if os.path.isfile(os.path.join(outdir, 'done')) and not redo: continue
            try:
                os.system('mkdir -p "%s"' % (outdir))
                # check if horizontal or vertical scaling factor
                o = subprocess.check_output(
                    'ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 "%s"' % (
                        os.path.join(vid_dir, cls, v)), shell=True).decode('utf-8')
                lines = o.splitlines()
                width = int(lines[0].split('=')[1])
                height = int(lines[1].split('=')[1])
                resize_str = '-1:256' if width > height else '256:-1'

                # extract frames
                os.system('ffmpeg -i "%s" -r 25 -q:v 2 -vf "scale=%s" "%s"  > /dev/null 2>&1' % (
                    os.path.join(vid_dir, cls, v), resize_str, os.path.join(outdir, '%05d.jpg')))
                nframes = len([fname for fname in os.listdir(outdir) if fname.endswith('.jpg') and len(fname) == 9])
                if nframes == 0: raise Exception

                os.system('touch "%s"' % (os.path.join(outdir, 'done')))
            except:
                print("ERROR", cls, v)


def extract_frames(vid_dir, frame_dir, start, end, redo=False):
    for ic, activity in enumerate(sorted(os.listdir(vid_dir))):
        activity_dir = os.path.join(vid_dir, activity)
        out_activity_dir = os.path.join(frame_dir, activity)
        if '.DS_Store' in activity_dir or not os.path.isdir(activity_dir): continue
        for part_id in sorted(os.listdir(activity_dir)):
            if '.DS_Store' in part_id: continue
            part_id_dir = os.path.join(activity_dir, part_id)
            out_part_id_dir = os.path.join(out_activity_dir, part_id)
            for file in sorted(os.listdir(part_id_dir)):
                try:
                    if '.DS_Store' in file: continue
                    if '_1.mp4' in file:
                        pass
                    elif 'mkv' in file:
                        continue
                    else:
                        continue
                    f = os.path.join(part_id_dir, file)
                    out_f = os.path.join(out_part_id_dir, file)
                    tmp = os.path.splitext(out_f)
                    out_f_dir = f'{tmp[0]}{tmp[1]}'
                    if not os.path.exists(out_f_dir):
                        os.makedirs(out_f_dir)

                    # # get duration
                    # o = subprocess.check_output(
                    #     'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "%s"' % (
                    #         f), shell=True).decode('utf-8')
                    # lines = o.splitlines()
                    # lines = lines[0].split('/')
                    # duration = int(np.ceil(int(lines[0]) / int(lines[1])))

                    # get height and width
                    # check if horizontal or vertical scaling factor
                    o = subprocess.check_output(
                        'ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 "%s"' % (
                            f), shell=True).decode('utf-8')
                    lines = o.splitlines()
                    width = int(lines[0].split('=')[1])
                    height = int(lines[1].split('=')[1])
                    resize_str = '-1:256' if width > height else '256:-1'
                    # get fps
                    o = subprocess.check_output(
                        'ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate "%s"' % (
                            f), shell=True).decode('utf-8')
                    lines = o.splitlines()
                    lines = lines[0].split('/')
                    fps = int(np.ceil(int(lines[0]) / int(lines[1])))
                    number_frame_per_second = fps  # in each second, only 1 frames is extracted from the video.
                    # extract frames
                    # cmd = 'ffmpeg -i "%s" -r 25 -q:v 2 -vf "scale=%s" "%s"  > /dev/null 2>&1' % (
                    cmd = 'ffmpeg -i "%s" -r %d -q:v 2 -vf "scale=%s" "%s"  > /dev/null 2>&1' % (
                        f, number_frame_per_second, resize_str, os.path.join(out_f_dir, 'img_%05d.jpg'))
                    print(cmd)
                    os.system(cmd)
                    # nframes = len([fname for fname in os.listdir(out_f_dir) if fname.endswith('.jpg') and len(fname) == 9])
                    # if nframes == 0: raise Exception
                    # os.system('touch "%s"' % (os.path.join(out_f_dir, 'done')))
                except:
                    print("ERROR", activity_dir, f)


def extract_frames_main(in_dir, out_dir, overwrite=True):
    # if not os.path.exists(in_dir):
    #     os.makedirs(in_dir)
    #     for f in sorted(os.listdir(in_dir)):
    #         if '.DS_Store' in f: continue
    #         src = os.path.join(in_dir + '_zip', f)
    #         dst = in_dir
    #         # brew install rar
    #         # cmd = f'unrar x \'{src}\' \'{dst}\''
    #         cmd = f'/usr/local/bin/unrar x {src} {dst} -inul -y'
    #         print(cmd)
    #         # os.system(cmd)    # not work
    #         subprocess.call(cmd.split())

    start = 0
    end = -1
    # extract(in_dir, out_dir, start, end, redo=True)
    if overwrite:
        extract_frames(in_dir, out_dir, start, end, redo=True)

    # # move all subfolders to parent folders
    # for f in sorted(os.listdir(out_dir+'_tmp/')):
    #     if '.DS_Store' in f: continue
    #     source = os.path.splitext(os.path.join(out_dir +'_tmp', f))[0]
    #     destination = out_dir
    #     if not os.path.exists(destination):
    #         os.makedirs(destination)
    #     print(f, source, destination)
    #     files_list = os.listdir(source)
    #     for files in files_list:
    #         shutil.move(os.path.join(source, files), os.path.join(destination, files))


if __name__ == '__main__':
    extract_frames_main()
