"""Собрать отдельный MP4 из каждой папки JPG внутри архива CSIRO CVB.

Запуск: python archive/lessons/frames_zip_to_video.py "путь/к/архиву.zip"
"""

import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import cv2
import numpy as np


FPS = 30
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "cvb_preview"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Укажи путь к ZIP: python frames_zip_to_video.py архив.zip")

    archive_path = Path(sys.argv[1])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with ZipFile(archive_path) as archive:
        folders = defaultdict(list)
        for name in archive.namelist():
            if name.startswith("data/raw_frames/") and name.lower().endswith(".jpg"):
                folders[PurePosixPath(name).parent.name].append(name)
        if not folders:
            raise ValueError("В архиве нет папок data/raw_frames с JPG")

        for folder, names in sorted(folders.items()):
            names.sort()  # img_00001.jpg, img_00002.jpg, ...
            output_path = OUTPUT_DIR / f"{folder}.mp4"
            writer = None
            try:
                for name in names:
                    jpg_bytes = archive.read(name)
                    frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError(f"Не удалось прочитать {name}")

                    if writer is None:
                        height, width = frame.shape[:2]
                        codec = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(output_path), codec, FPS, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError(f"Не удалось создать {output_path}")
                    elif frame.shape[:2] != (height, width):
                        raise ValueError(f"Размер кадра изменился: {name}")

                    writer.write(frame)
            finally:
                if writer is not None:
                    writer.release()

            print(f"{folder}: {len(names)} frames: {output_path}")


if __name__ == "__main__":
    main()
