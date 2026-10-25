from pathlib import Path
from ultralytics import YOLO

project_dir = Path(__file__).resolve().parent
model_path = "yolo11n.pt"
data_path = project_dir / "data" / "dataset_curated_v3" / "data.yaml"

model = YOLO(str(model_path))

if __name__ == "__main__":
    model.train(
        data=str(data_path),
        epochs=40,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        amp=False,
        project=str(project_dir / "outputs" / "training"),
        name="curated_v3_40epochs",
    )