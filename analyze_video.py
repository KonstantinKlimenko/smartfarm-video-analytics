from pathlib import Path

import cv2
from ultralytics import YOLO
import csv
import os
import psycopg
from dotenv import load_dotenv


project_dir = Path(__file__).resolve().parent
video_path = project_dir / "data" / "mixkit-cows-walking-trough-a-road-7429-hd-ready.mp4"
model_path = (
    project_dir / "outputs" / "training"
    / "curated_v2_40epochs" / "weights" / "best.pt"
)
output_dir = project_dir / "outputs" / "cvb_baseline"
output_dir.mkdir(parents=True, exist_ok=True)

model = YOLO(str(model_path))
capture = cv2.VideoCapture(str(video_path))
if not capture.isOpened():
    raise RuntimeError(f"Не удалось открыть видео: {video_path}")

fps = capture.get(cv2.CAP_PROP_FPS)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_x = width // 2
last_centr_x = {}
counted_ids = set()

output_path = project_dir / "outputs" / "7429_tracking_v2.mp4"
codec = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(output_path), codec, fps, (width, height))
if not writer.isOpened():
    raise RuntimeError(f"Не удалось создать видео: {output_path}")

events = []
track_rows = []
frame_number = 0
while True:
    ok, frame = capture.read()
    if not ok:
        break

    result = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
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
                events.append(
                    (
                        video_path.name,
                        track_id,
                        frame_number,
                        frame_number / fps,
                        "right_to_left",
                    )
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
    writer.write(annotated_frame)

    if frame_number % 30 == 0:
        ids = result.boxes.id
        track_ids = [] if ids is None else ids.int().cpu().tolist()
        print(f"Frame {frame_number}: Ids {track_ids}")

    frame_number += 1

capture.release()
writer.release()
csv_path = project_dir / "outputs" / "7429_tracks_v2.csv"

with csv_path.open("w", newline="", encoding="utf-8") as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(["frame", "id", "x1", "y1", "x2", "y2", "confidence"])
    csv_writer.writerows(track_rows)

load_dotenv(project_dir / ".env")

with psycopg.connect(
    host="127.0.0.1",
    port=5433,
    dbname="smartfarm",
    user="smartfarm",
    password=os.environ["SMARTFARM_DB_PASSWORD"],
) as connection:
    with connection.cursor() as cursor:
        cursor.execute(
            "DELETE FROM crossing_events WHERE video_name = %s",
            (video_path.name,),
        )
        cursor.executemany(
            """
            INSERT INTO crossing_events
                (video_name, track_id, frame_number, time_seconds, direction)
            VALUES (%s, %s, %s, %s, %s)
            """,
            events,
        )

print(f"Треки: {csv_path}")
print(f"Готово: {output_path}")
print(f"Всего пересекли линию: {len(counted_ids)}")
print(f"Записано событий в PostgreSQL: {len(events)}")