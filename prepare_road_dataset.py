from pathlib import Path
from zipfile import ZipFile
import shutil

root = Path(__file__).resolve().parent
data = root / "data"

sources = [
    ("colors", data / "12_photos.zip",
     data / "cvat_practice_batch_01" / "images",
     {"06_black", "10_herd"}),

    ("road", data / "36_photo.zip",
     data / "cvat_road_batch_02" / "images",
     {"14_road", "15_road", "16_road",
      "22_road", "23_road", "26_road"}),
    
    ("cvb_0004", data / "9_photo.zip",
     data / "cvb_0004_annotation",
     set()),

    ("new_cows", data / "57_photo.zip",
     data / "for_cvat",
     set()),

    ("new_cows_val", data / "7.zip",
     data / "for_cvat" / "val",
     set()),
]

val_stems = {
    "02_brown", "08_black",
    "04_road", "09_road", "17_road",
    "24_road", "27_road", "28_road",
    "cow_0066_angle1", "cow_0067_angle1", "cow_0068_angle1",
    "cow_0069_angle1", "cow_0070_angle1", "cow_0071_angle1",
    "cow_0072_angle1",
}

split_counts = {"train": 0, "val": 0}

dataset_dir = data / "dataset_curated_v3"

for split in ("train", "val"):
    (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

total = 0

for group, zip_path, image_dir, excluded in sources:
    kept = 0

    with ZipFile(zip_path) as archive:
        label_files = sorted(
            name for name in archive.namelist()
            if name.startswith("labels/train/") and name.endswith(".txt")
        )

        for label_name in label_files:
            stem = Path(label_name).stem

            if stem in excluded:
                print(f"Пропускаем {stem}")
                continue

            image_path = None
            for extension in (".jpg", ".jpeg", ".png"):
                candidate = image_dir / f"{stem}{extension}"
                if candidate.is_file():
                    image_path = candidate
                    break

            if image_path is None:
                raise FileNotFoundError(f"Не найдено изображение для {stem} в {image_dir}")

            split = "val" if stem in val_stems else "train"
            split_counts[split] += 1
            image_target = dataset_dir / "images" / split / image_path.name
            label_target = dataset_dir / "labels" / split / Path(label_name).name
            shutil.copy2(image_path, image_target)
            label_target.write_bytes(archive.read(label_name))
            kept += 1
            print(f"{image_path.name} → {split}")

    total += kept
    print(f"{group}: осталось {kept}\n")

print(f"Всего для датасета: {total}")
print(f"Разделение: {split_counts}")

yaml_path = dataset_dir / "data.yaml"

yaml_text = (
    f'path: "{dataset_dir.as_posix()}"\n'
    "train: images/train\n"
    "val: images/val\n"
    "names:\n"
    "  0: cow\n"
)

yaml_path.write_text(yaml_text, encoding="utf-8")
print(f"Конфигурация YOLO: {yaml_path}")