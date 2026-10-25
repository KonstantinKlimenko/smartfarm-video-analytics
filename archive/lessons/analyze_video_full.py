"""Track cattle in a video and record crossings of a configurable line."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


COW_CLASS_ID = 19  # COCO's cow class


def parse_line(value: str) -> tuple[float, float, float, float]:
    try:
        points = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("line must be x1,y1,x2,y2") from exc
    if len(points) != 4 or any(not 0 <= point <= 1 for point in points):
        raise argparse.ArgumentTypeError("line needs four coordinates between 0 and 1")
    if points[:2] == points[2:]:
        raise argparse.ArgumentTypeError("line endpoints must differ")
    return points


def side_of_line(x: float, y: float, line: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = line
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="path to a local video file")
    parser.add_argument("--model", default="yolo11n.pt", help="COCO model or cattle model path")
    parser.add_argument("--line", type=parse_line, default=parse_line("0.5,0,0.5,1"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="e.g. cpu or 0 for the first CUDA GPU")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if not Path(args.source).is_file():
        parser.error(f"video not found: {args.source}")
    if not 0 < args.confidence <= 1:
        parser.error("confidence must be in (0, 1]")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("max-frames must be positive")

    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 1 or height < 1:
        raise RuntimeError("Video reports invalid frame dimensions")

    args.output.mkdir(parents=True, exist_ok=True)
    video_path = args.output / "annotated.mp4"
    events_path = args.output / "events.jsonl"
    summary_path = args.output / "summary.json"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV cannot write {video_path}")

    model = YOLO(args.model)
    lx1, ly1, lx2, ly2 = args.line
    line_px = (int(lx1 * width), int(ly1 * height), int(lx2 * width), int(ly2 * height))
    # The band prevents tiny box movements along the line from creating false crossings.
    deadband = max(width, height) * 0.01
    line_length = ((line_px[2] - line_px[0]) ** 2 + (line_px[3] - line_px[1]) ** 2) ** 0.5
    previous_side: dict[int, int] = {}
    crossings = 0
    frames = 0
    detected_frames = 0
    processing_seconds = 0.0

    try:
        with events_path.open("w", encoding="utf-8") as events:
            while args.max_frames is None or frames < args.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_time = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 or frames / fps
                start = time.perf_counter()
                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[COW_CLASS_ID],
                    conf=args.confidence,
                    device=args.device,
                    verbose=False,
                )[0]
                processing_seconds += time.perf_counter() - start
                annotated = result.plot()
                cv2.line(annotated, line_px[:2], line_px[2:], (0, 255, 255), 2)
                boxes = result.boxes
                count = len(boxes) if boxes is not None else 0
                detected_frames += count > 0
                if boxes is not None and boxes.id is not None:
                    for track_id, box in zip(boxes.id.int().cpu().tolist(), boxes.xyxy.cpu().tolist()):
                        x1, y1, x2, y2 = box
                        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                        signed_distance = side_of_line(center_x, center_y, line_px) / line_length
                        if abs(signed_distance) <= deadband:
                            continue
                        current_side = 1 if signed_distance > 0 else -1
                        old_side = previous_side.get(track_id)
                        if old_side is not None and old_side != current_side:
                            crossings += 1
                            event = {
                                "type": "line_crossing",
                                "track_id": track_id,
                                "video_time_seconds": round(frame_time, 3),
                                "direction": "positive_to_negative" if old_side > 0 else "negative_to_positive",
                                "center": [round(center_x), round(center_y)],
                            }
                            events.write(json.dumps(event, ensure_ascii=False) + "\n")
                        previous_side[track_id] = current_side
                cv2.putText(
                    annotated,
                    f"Cows: {count}  Crossings: {crossings}",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )
                writer.write(annotated)
                frames += 1
    finally:
        capture.release()
        writer.release()

    summary = {
        "source": args.source,
        "model": args.model,
        "frames_processed": frames,
        "video_seconds": round(frames / fps, 2),
        "frames_with_cows": detected_frames,
        "line_crossings": crossings,
        "processing_fps": round(frames / processing_seconds, 2) if processing_seconds else 0,
        "output_video": str(video_path),
        "events": str(events_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
