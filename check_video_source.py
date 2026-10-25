import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2

source = "rtsp://127.0.0.1:8554/cows"
capture = cv2.VideoCapture(source)

if not capture.isOpened():
    raise RuntimeError(f"Не удалось подключиться к {source}")

ok, frame = capture.read()
capture.release()

if not ok:
    raise RuntimeError("Подключение есть, но кадр не получен")

print(f"Кадр получен, форма массива: {frame.shape}")