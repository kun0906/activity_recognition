import cv2
import os


def get_info(in_file):
    print(in_file)
    # capture video
    cap = cv2.VideoCapture(in_file)

    # Get the Default resolutions
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(
    #     *f"{fourcc & 255:c},{(fourcc >> 8) & 255:c}, {(fourcc >> 16) & 255:c}, {(fourcc >> 24) & 255:c}")
    fourcc = cv2.VideoWriter_fourcc(*(chr(fourcc & 0xff) + chr((fourcc >> 8) & 0xff) + chr((fourcc >> 16) & 0xff)
                                      + chr((fourcc >> 24) & 0xff)))
    # print(fourcc)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # Define the codec and filename.
    fps = cap.get(cv2.CAP_PROP_FPS)
    # get duration
    totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = float(totalNoFrames) / float(fps)
    # out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
    # out = cv2.VideoWriter(out_file, fourcc, fps, (frame_width, frame_height), isColor=True)
    print(f'fps: {fps}, duration: {duration}, (height, width): ({frame_height}, {frame_width})')
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    in_file = 'data/data-clean/refrigerator/take_out_item/1/take_out_item_1_1613085820_1.mp4'
    get_info(in_file)
    in_file = 'data/data-clean/refrigerator/take_out_item/1/take_out_item_1_1613085820_2.mkv'
    get_info(in_file)
