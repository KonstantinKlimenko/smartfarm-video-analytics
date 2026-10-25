"""Извлечь кадры ролика 0004 для ручной разметки."""

from pathlib import Path
import cv2


source = Path(
    "data/cvb_preview/"
    "0002_arm01_gopro1_20200322_222554_beh7_ani1_ins1_cut_1.mp4"
)
output = Path("data/cvb_0002_check")
frame_numbers = (0, 120, 240, 360, 420)

output.mkdir(parents=True, exist_ok=True)
capture = cv2.VideoCapture(str(source))
if not capture.isOpened():
    raise RuntimeError(f"Не удалось открыть {source}")

for frame_number in frame_numbers:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Не удалось прочитать кадр {frame_number}")

    image_path = output / f"frame_{frame_number:04d}.jpg"
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"Не удалось сохранить {image_path}")

capture.release()
print(f"Сохранено {len(frame_numbers)} кадров в {output.resolve()}")