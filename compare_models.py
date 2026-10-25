from pathlib import Path

import cv2
from ultralytics import YOLO


project_dir = Path(__file__).resolve().parent
video_path = project_dir / "data" / "10306945-uhd_3840_2160_30fps.mp4"
output_dir = project_dir / "outputs" / "model_comparison" / "feeder"
output_dir.mkdir(parents=True, exist_ok=True)

base_model = YOLO(str(project_dir.parent / "Project_for_CV" / "yolo11n.pt"))
v3_model = YOLO(
    str(project_dir / "outputs" / "training"
        / "curated_v3_40epochs" / "weights" / "best.pt")
)

capture = cv2.VideoCapture(str(video_path))
if not capture.isOpened():
    raise OSError(f"Не удалось открыть видео: {video_path}")

fps = capture.get(cv2.CAP_PROP_FPS)

for seconds in (0, 20, 40, 60, 80):
    frame_number = int(seconds * fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    if not ok:
        print(f"Не удалось прочитать кадр около {seconds}-й секунды")
        continue

    frame = cv2.resize(frame, (1280, 720))

    base_result = base_model.predict(
        frame, classes=[19], conf=0.25, imgsz=640, verbose=False
    )[0]
    v3_result = v3_model.predict(
        frame, classes=[0], conf=0.25, imgsz=640, verbose=False
    )[0]

    comparison = cv2.hconcat([
        base_result.plot(labels=False, conf=False),
        v3_result.plot(labels=False, conf=False),
    ])

    output_path = output_dir / f"second_{seconds:02d}.jpg"
    if not cv2.imwrite(str(output_path), comparison):
        raise OSError(f"Не удалось сохранить: {output_path}")

    print(
        f"{seconds} с: базовая — {len(base_result.boxes)} рамок, "
        f"v3 — {len(v3_result.boxes)} рамок; {output_path}"
    )

capture.release()