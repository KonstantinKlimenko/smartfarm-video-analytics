from pathlib import Path
from zipfile import ZipFile
import cv2

project_dir = Path(__file__).resolve().parents[2]
archive_path = project_dir / "data" / "1st_batch.zip"
image_dir = project_dir / "data" / "annotation-practice"
output_dir = project_dir / "outputs" / "batch_review"
output_dir.mkdir(parents=True, exist_ok=True)

with ZipFile(archive_path) as archive:
    for name in sorted(archive.namelist()):
        if not (name.startswith("labels/") and name.endswith(".txt")):
            continue

        stem = Path(name).stem
        image_path = image_dir / f"{stem}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)

        image_height, image_width = image.shape[:2]
        text = archive.read(name).decode("utf-8")
        lines = [line for line in text.splitlines() if line.strip()]

        for line in lines:
            parts = line.split()
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            x1 = round((x_center - box_width / 2) * image_width)
            y1 = round((y_center - box_height / 2) * image_height)
            x2 = round((x_center + box_width / 2) * image_width)
            y2 = round((y_center + box_height / 2) * image_height)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 2)

        output_path = output_dir / f"{stem}_review.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"{stem}: нарисовано {len(lines)} рамок")
