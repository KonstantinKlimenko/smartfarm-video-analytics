from pathlib import Path
from zipfile import ZipFile
import cv2

project_dir = Path(__file__).resolve().parents[2]
archive_path = project_dir / "data" / "1st_image.zip"

with ZipFile(archive_path) as archive:
    text = archive.read("labels/train/frame_0090.txt").decode("utf-8")

lines = [line for line in text.splitlines() if line.strip()]

print("Количество рамок:", len(lines))
print(f"Первая строка: {lines[0]}")

image_path = project_dir / "data" / "annotation-practice" / "frame_0090.jpg"
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)

image_height, image_width = image.shape[:2]

for line in lines:
    parts = line.split()

    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    box_width = float(parts[3])
    box_height = float(parts[4])

    x1 = round((x_center - box_width / 2) * image_width)
    y1 = round((y_center - box_height / 2) * image_height)
    x2 = round((x_center + box_width / 2) * image_width)
    y2 = round((y_center + box_height / 2) * image_height)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 2)

output_path = project_dir / "outputs" / "all_boxes.jpg"
cv2.imwrite(str(output_path), image)
print(f"Нарисовано рамок: {len(lines)}")
print(f"Результат: {output_path}")
