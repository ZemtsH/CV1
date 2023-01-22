import numpy as np
import cv2
import argparse
from numba import njit
import time

@njit
def median_filter_speed(img, size=5):
    extreme = size // 2
    width, height = img.shape[:2]
    img_blur = np.zeros((width, height), dtype=np.uint8)
    for x in range(extreme, width - extreme):
        for y in range(extreme, height - extreme):
            window = np.zeros(size * size, dtype=np.uint8)
            i = 0
            for w_x in range(size):
                for w_y in range(size):
                    window[i] = img[x + w_x - extreme, y + w_y - extreme]
                    i += 1
            window = np.sort(window)

            img_blur[x, y] = window[size * size // 2]
    return img_blur.astype(np.uint8)

def median_filter(img, size=5):
    extreme = size // 2
    width, height = img.shape[:2]
    img_blur = np.zeros((width, height), dtype=np.uint8)
    for x in range(extreme, width - extreme):
        for y in range(extreme, height - extreme):
            window = np.zeros(size * size, dtype=np.uint8)
            i = 0
            for w_x in range(size):
                for w_y in range(size):
                    window[i] = img[x + w_x - extreme, y + w_y - extreme]
                    i += 1
            window = np.sort(window)

            img_blur[x, y] = window[size * size // 2]
    return img_blur.astype(np.uint8)

def action(path, func, size=5, show=True):
    cap = cv2.VideoCapture(path)
    success, img = cap.read()
    show_orig = True
    frame_sum = 0
    start_time = time.time()
    while success:
        frame_sum += 1
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        action_frame = func(frame, size)
        if show:
            if cv2.waitKey(10) == ord('t'):
                show_orig = not show_orig
            frame_to_show = frame if show_orig else action_frame
            cv2.imshow('frame', frame_to_show)
        success, img = cap.read()
    end_time = time.time()
    cap.release()
    cv2.destroyAllWindows()
    all_time = end_time - start_time
    return all_time, all_time / frame_sum

if __name__ == "__main__":
    video_time, frame_time = action("video.mp4", cv2.medianBlur, size=7, show=False)
    print("OpenCV: время обработки кадра:", frame_time)
    print("OpenCV: время обработки видео:", video_time)

    video_time, frame_time = action("video.mp4", median_filter, size=7,
                                                     show=False)
    print("Нативно: время обработки кадра:", frame_time)
    print("Нативно: время обработки видео:", video_time)

    video_time, frame_time = action("video.mp4", median_filter_speed, size=7,
                                                     show=False)
    print("numba: время обработки кадра:", frame_time)
    print("numba: время обработки видео:", video_time)

