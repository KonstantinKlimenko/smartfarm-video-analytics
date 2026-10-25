from pathlib import Path
import shutil
import tarfile

project_dir = Path(__file__).resolve().parents[2]
archive_path = project_dir / "data" / "external" / "8-calves" / "data" / "pmfeed.tar"
dataset_dir = project_dir / "data" / "dataset_8calves"

counts = {"train": 0, "val": 0}

with tarfile.open(archive_path) as archive:
    for member in archive:
        if not member.isfile():
            continue

        filename = Path(member.name).name
        if filename != member.name or Path(filename).suffix not in {".png", ".txt"}:
            raise ValueError(f"Неожиданный финал в архиве: {member.name}")

        group = filename.split("_")[1]
        split  = "val" if group == "4" else "train"
        kind = "images" if filename.endswith(".png") else "labels"

        target_dir = dataset_dir / kind / split
        target_dir.mkdir(parents=True,  exist_ok=True)

        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"Не удалось прочитать {filename}")

        with source, (target_dir / filename).open("wb") as destination:
            shutil.copyfileobj(source, destination)

        if kind == "images":
            counts[split] += 1

for split in ("train", "val"):
    images = {path.stem for path in (dataset_dir / "images" / split).glob("*.png")}
    labels = {path.stem for path in (dataset_dir / "labels" / split).glob("*.txt")}
    if labels != images:
        raise ValueError(f"В {split} изображения и разметка не совпадают")
    print(f"{split}: {counts[split]} изображений, {len(labels)} файлов разметки")

yaml_path = dataset_dir / "data.yaml"
yaml_path.write_text(
    f"path: {dataset_dir.as_posix()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n"
    "  0: cow\n",
    encoding="utf-8",
)
print(f"Конфигурация: {yaml_path}")
