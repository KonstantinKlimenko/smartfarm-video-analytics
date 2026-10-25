"""Fetch a small, credited Wikimedia Commons image batch for CVAT practice."""

import argparse
import csv
import html
import re
import sys
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import requests


SOURCES = (
    ("brown", "Brown calves.jpg"),
    ("brown", "Cattle walking.jpg"),
    ("brown", "Lying cattle.jpg"),
    ("brown", "Cattle in Armenia.jpg"),
    ("black", "Deutsch-angus kühe.jpg"),
    ("black", "Black Cows at brisbane EKKA-1 (36175500200).jpg"),
    ("black", "Cows at Outlane - geograph.org.uk - 3552674.jpg"),
    ("black", "Cow near Buck Mountain Road.JPG"),
    ("herd", "Cattle herd, Yazılıkaya 01.jpg"),
    ("herd", "A herd of cows in Punjab.jpg"),
    ("herd", "Cattle herding on the pastures of the southern slopes of Eğribel Pass 01.jpg"),
    ("herd", "Cow Crossing (3847013038).jpg"),
)

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "SmartFarmVideoAnalytics/0.1 (educational annotation project)"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "cvat_practice_batch_01"


def plain_text(value):
    return html.unescape(re.sub(r"<[^>]+>", "", value or "")).strip()


def main(download):
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    titles = [f"File:{title}" for _, title in SOURCES]
    response = session.get(
        API_URL,
        params={
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "titles": "|".join(titles),
            "iiprop": "url|extmetadata",
            "iiurlwidth": 1600,
        },
        timeout=30,
    )
    response.raise_for_status()
    pages = {page["title"]: page for page in response.json()["query"]["pages"].values()}

    rows = []
    for index, (group, title) in enumerate(SOURCES, start=1):
        page = pages.get(f"File:{title}")
        if not page or "imageinfo" not in page:
            raise ValueError(f"No Wikimedia Commons file found: {title}")
        info = page["imageinfo"][0]
        metadata = info.get("extmetadata", {})
        license_name = plain_text(metadata.get("LicenseShortName", {}).get("value"))
        license_url = metadata.get("LicenseUrl", {}).get("value", "")
        author = plain_text(metadata.get("Artist", {}).get("value"))
        if not license_name or not (license_name.startswith("CC ") or license_name == "Public domain"):
            raise ValueError(f"Check license before using {title}: {license_name!r}")

        filename = f"{index:02d}_{group}.jpg"
        row = {
            "filename": filename,
            "commons_title": title,
            "source_page": info["descriptionurl"],
            "author": author,
            "license": license_name,
            "license_url": license_url,
            "image_url": info["thumburl"],
            "width": info["thumbwidth"],
            "height": info["thumbheight"],
        }
        rows.append(row)
        print(f"{filename}: {info['thumbwidth']}x{info['thumbheight']} | {license_name} | {title}")

    if not download:
        print("Metadata only; no images downloaded.")
        return

    image_dir = OUTPUT_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        image_path = image_dir / row["filename"]
        if image_path.exists():
            print(f"Already present: {image_path.name}")
            continue
        image_response = session.get(row["image_url"], timeout=60)
        image_response.raise_for_status()
        if not image_response.headers.get("Content-Type", "").startswith("image/"):
            raise ValueError(f"Unexpected content type for {row['filename']}")
        image_path.write_bytes(image_response.content)
        print(f"Downloaded: {image_path.name} ({image_path.stat().st_size:,} bytes)")

    with (OUTPUT_DIR / "sources.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Source manifest: {OUTPUT_DIR / 'sources.csv'}")

    zip_path = OUTPUT_DIR / "images_for_cvat.zip"
    with ZipFile(zip_path, "w", compression=ZIP_STORED) as archive:
        for row in rows:
            image_path = image_dir / row["filename"]
            archive.write(image_path, arcname=image_path.name)
    print(f"CVAT upload archive: {zip_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    main(args.download)
