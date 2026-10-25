from pathlib import Path

import cv2
from ultralytics import YOLO

project_dir = Path(__file__).resolve().parents[2]
image_dir = project_dir / "data" / "annotation-practice"
output_dir = project_dir / "outputs" / "baseline_conf025"
output_dir.mkdir(parents=True, exist_ok=True)

model_path = project_dir.parent / "Project_for_CV" / "yolo11n.pt"
model = YOLO(str(model_path))

frame_names = (
    "frame_0030",
    "frame_0150",
    "frame_0210",
    "frame_0270",
    "frame_0330",
)

for frame_name in frame_names:
    image_path = image_dir / f"{frame_name}.jpg"
    result = model.predict(
        source=str(image_path),
        classes=[19],
        conf=0.25,
        verbose=False,
    )[0]

    output_path = output_dir / f"{frame_name}_pred.jpg"
    cv2.imwrite(str(output_path), result.plot())
    print(f"{frame_name}: найдено {len(result.boxes)} коров")
