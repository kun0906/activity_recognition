import numpy as np
import cv2

in_file = 'data/demo/ask_time_1_1614904536_1.mp4.mp4'
# capture video
cap = cv2.VideoCapture(in_file)
# out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
out_file = in_file + '.mp4'
# out = cv2.VideoWriter(out_file, -1, 20.0,(640,480))
# Get the Default resolutions
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
# fourcc_str = format("%c%c%c%c", fourcc & 255, (fourcc >> 8) & 255, (fourcc >> 16) & 255, (fourcc >> 24) & 255)
# fourcc_str = cv2.VideoWriter_fourcc(*f"{fourcc & 255:c},{(fourcc >> 8) & 255:c}, {(fourcc >> 16) & 255:c}, {(fourcc >> 24) & 255:c}")
h = int(fourcc)
fourcc_str = chr(h & 0xff) + chr((h >> 8) & 0xff) + chr((h >> 16) & 0xff) + chr((h >> 24) & 0xff)
print(fourcc_str)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
print(fourcc, fps, frame_width, frame_height)
# Define the codec and filename.
# out = cv2.VideoWriter(out_file,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
out = cv2.VideoWriter(out_file, fourcc, fps, (frame_width, frame_height), isColor=True)
# out = cv2.VideoWriter(out_file,cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width,frame_height), isColor=True)


# descripe a loop
# read video frame by frame
while True:
    ret, img = cap.read()
    # print(ret)
    if ret:
        # cv2.imshow('Original Video',img)
        # flip for truning(fliping) frames of video
        img2 = cv2.flip(img, 1)  # Horizontal
        # cv2.imshow('Flipped video',img2)
        out.write(img2)
        # k=cv2.waitKey(30) & 0xff
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
print(out_file)
