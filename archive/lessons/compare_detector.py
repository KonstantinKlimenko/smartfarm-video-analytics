from pathlib import Path
import cv2
from ultralytics import YOLO

project_dir = Path(__file__).resolve().parents[2]
model_path = project_dir.parent / "Project_for_CV" / "yolo11n.pt"
image_path = project_dir / "data" / "annotation-practice" / "frame_0090.jpg"
output_path = project_dir / "outputs" / "pred_frame_0090_conf010.jpg"

model = YOLO(str(model_path))
result = model.predict(
    source=str(image_path),
    classes=[19],
    conf=0.10,
    verbose=False,
)[0]

image_with_predictions = result.plot()
cv2.imwrite(str(output_path), image_with_predictions)

print(f"Модель нашла коров: {len(result.boxes)}")
print(f"Результат: {output_path}")
