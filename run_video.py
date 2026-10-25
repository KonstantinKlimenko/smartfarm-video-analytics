from pathlib import Path

import cv2
from ultralytics import YOLO
import csv
import os
import psycopg
from dotenv import load_dotenv
import time
import argparse


project_dir = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Анализ коров на видео")
parser.add_argument(
    "--source",
    default=str(
        project_dir / "data" / "mixkit-cows-walking-trough-a-road-7429-hd-ready.mp4"
    )
)
parser.add_argument("--name", default="mixkit_base_demo")
parser.add_argument("--max-frames", type=int, default=600)
parser.add_argument(
    "--weights",
    choices=("base", "v2", "v3"),
    default="base",
)

args = parser.parse_args()
source = args.source
source_name = args.name
max_frames = args.max_frames
is_rtsp = source.startswith(("rtsp://", "rtsps://"))

if args.weights == "base":
    model_path = project_dir / "yolo11n.pt"
    cow_class = 19
elif args.weights == "v2":
    model_path = (
        project_dir / "outputs" / "training"
        / "curated_v2_40epochs" / "weights" / "best.pt"
    )
    cow_class = 0
else:
    model_path = (
            project_dir / "outputs" / "training"
            / "curated_v3_40epochs" / "weights" / "best.pt"
        )
    cow_class = 0

output_dir = project_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

model = YOLO(str(model_path))
if is_rtsp:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    capture = cv2.VideoCapture(
        source,
        cv2.CAP_FFMPEG,
        [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
        ],
    )
else:
    capture = cv2.VideoCapture(source)

if not capture.isOpened():
    raise RuntimeError(f"Не удалось открыть поток: {source}")

fps = capture.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_x = width // 2
last_centr_x = {}
counted_ids = set()

output_path = output_dir / f"{source_name}_tracking.mp4"

scale = min(1.0, 1920 / width, 1080 / height)
output_size = (
    max(2, (int(width * scale) // 2) * 2),
    max(2, (int(height * scale) // 2) * 2),
)

codec = cv2.VideoWriter_fourcc(*"avc1")
backend = cv2.CAP_MSMF if os.name == "nt" else cv2.CAP_FFMPEG
writer = cv2.VideoWriter(
    str(output_path), backend, codec, fps, output_size
)
if not writer.isOpened():
    raise RuntimeError(f"Не удалось создать H.264-видео: {output_path}")

load_dotenv(project_dir / ".env")

connection = psycopg.connect(
    host="127.0.0.1",
    port=5433,
    dbname="smartfarm",
    user="smartfarm",
    password=os.environ["SMARTFARM_DB_PASSWORD"],
    autocommit=True,
)

connection.execute(
    "DELETE FROM crossing_events WHERE video_name = %s",
    (source_name,),
)

events = []
track_rows = []
started_at = time.monotonic()
batch_started_at = time.monotonic()
frame_number = 0
while True:
    ok, frame = capture.read()
    if not ok:
        if is_rtsp:
            print(f"Не удалось получить кадр после {frame_number} обработанных кадров")
        else:
            print(f"Видео закончилось: обработано {frame_number} кадров")
        break

    result = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[cow_class],
        conf=0.25,
        imgsz=640,
        verbose=False,
    )[0]

    if result.boxes.id is not None:
        ids = result.boxes.id.int().cpu().tolist()
        coordinates = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()

        for track_id, box, confidence in zip(ids, coordinates, confidences):
            x1, y1, x2, y2 = box

            track_rows.append(
                [frame_number, track_id, x1, y1, x2, y2, confidence]
            )
            center_x = (x1 + x2) / 2
            previous_x = last_centr_x.get(track_id)

            if (
                previous_x is not None
                and previous_x > line_x >= center_x
                and track_id not in counted_ids
            ):
                counted_ids.add(track_id)
                
                if is_rtsp:
                    time_seconds = time.monotonic() - started_at
                else:
                    time_seconds = frame_number / fps
                
                event = (
                        source_name,
                        track_id,
                        frame_number,
                        time_seconds,
                        "right_to_left",
                    )
                events.append(event)

                connection.execute(
                    """
                    INSERT INTO crossing_events
                        (video_name, track_id, frame_number, time_seconds, direction)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    event,
                )
                
                print(f"Кадр {frame_number}: корова ID {track_id} пересекла линию")
            last_centr_x[track_id] = center_x

    annotated_frame = result.plot(conf=False, font_size=14, line_width=2)
    cv2.line(annotated_frame, (line_x, 0), (line_x, height), (0, 0, 255), 2)
    cv2.putText(
        annotated_frame,
        f"Crossed: {len(counted_ids)}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3,
    )
    if output_size != (width, height):
        annotated_frame = cv2.resize(annotated_frame, output_size)
    writer.write(annotated_frame)
    
    if (frame_number + 1) % 30 == 0:
        elapsed = time.monotonic() - batch_started_at
        processing_fps = 30 / elapsed

        ids = result.boxes.id
        track_ids = [] if ids is None else ids.int().cpu().tolist()
        print(
            f"Кадр {frame_number}: {processing_fps:.1f} FPS, "
            f"активные ID: {track_ids}"
        )

        batch_started_at = time.monotonic()

    frame_number += 1

    if frame_number >= max_frames:
        break

capture.release()
writer.release()
connection.close()
csv_path = output_dir / f"{source_name}_tracks.csv"

with csv_path.open("w", newline="", encoding="utf-8") as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(["frame", "id", "x1", "y1", "x2", "y2", "confidence"])
    csv_writer.writerows(track_rows)

print(f"Треки: {csv_path}")
print(f"Готово: {output_path}")
print(f"Всего пересекли линию: {len(counted_ids)}")
print(f"Записано событий в PostgreSQL: {len(events)}")