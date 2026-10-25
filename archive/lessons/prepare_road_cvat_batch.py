"""Curate road-cattle photos from Wikimedia Commons for a CVAT practice batch.

Run ``--preview`` first. It downloads small previews and writes contact sheets.
After reviewing those sheets, put chosen preview numbers in SELECTED_NUMBERS
and run ``--download`` to fetch 1600px copies with attribution metadata.
"""

import argparse
import csv
import html
import io
import json
import re
import sys
import time
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import requests
from PIL import Image, ImageDraw


API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "SmartFarmVideoAnalytics/0.1 (educational annotation project)"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "cvat_road_batch_02"
CATEGORY = "Category:Cattle on roads"
SELECTED_NUMBERS = (
    1, 3, 8, 9, 23, 24, 25, 26, 27, 28, 29, 31, 32, 38, 39, 43, 46,
    47, 50, 52, 55, 62, 64, 65, 66, 67, 68, 70, 72, 73, 74, 75, 76,
    78, 79, 80, 82, 84, 86, 88, 89, 100, 101, 102, 104, 105, 107,
    108, 109, 110,
)


def plain_text(value):
    return html.unescape(re.sub(r"<[^>]+>", "", value or "")).strip()


def api_get(session, **params):
    for attempt in range(6):
        response = session.get(API_URL, params={"action": "query", "format": "json", **params}, timeout=60)
        if response.status_code == 429:
            time.sleep(min(int(response.headers.get("Retry-After", "10")), 60) + attempt * 5)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("Wikimedia API rate limit persisted after retries")


def image_get(session, url):
    for attempt in range(8):
        response = session.get(url, timeout=90)
        if response.status_code in (429, 503):
            time.sleep(min(int(response.headers.get("Retry-After", "15")), 90) + attempt * 10)
            continue
        response.raise_for_status()
        return response.content
    raise RuntimeError("Wikimedia image rate limit persisted after retries")


def get_candidates(session):
    result = api_get(
        session,
        list="categorymembers",
        cmtitle=CATEGORY,
        cmtype="file",
        cmlimit="max",
    )
    titles = [item["title"] for item in result["query"]["categorymembers"]]
    rows = []
    for start in range(0, len(titles), 40):
        data = api_get(
            session,
            prop="imageinfo",
            titles="|".join(titles[start : start + 40]),
            iiprop="url|size|mime|extmetadata",
            iiurlwidth=400,
        )
        for page in data["query"]["pages"].values():
            info = page.get("imageinfo", [{}])[0]
            meta = info.get("extmetadata", {})
            license_name = plain_text(meta.get("LicenseShortName", {}).get("value"))
            if not (license_name.startswith("CC ") or license_name == "Public domain"):
                continue
            if not info.get("mime", "").startswith("image/") or "thumburl" not in info:
                continue
            if min(info.get("width", 0), info.get("height", 0)) < 500:
                continue
            rows.append(
                {
                    "title": page["title"],
                    "source_page": info["descriptionurl"],
                    "author": plain_text(meta.get("Artist", {}).get("value")),
                    "license": license_name,
                    "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                    "width": info["width"],
                    "height": info["height"],
                    "preview_url": info["thumburl"],
                }
            )
    return sorted(rows, key=lambda item: item["title"])


def preview(session):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = get_candidates(session)
    (OUTPUT_DIR / "candidates.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    preview_dir = OUTPUT_DIR / "previews"
    preview_dir.mkdir(exist_ok=True)
    for index, row in enumerate(rows, 1):
        path = preview_dir / f"{index:03d}.jpg"
        if path.exists():
            continue
        try:
            response = session.get(row["preview_url"], timeout=60)
            response.raise_for_status()
            with Image.open(io.BytesIO(response.content)) as raw:
                raw.convert("RGB").save(path, quality=85)
        except (requests.RequestException, OSError) as error:
            print(f"Preview skipped {index}: {error}")

    for sheet_start in range(0, len(rows), 24):
        sheet = Image.new("RGB", (1600, 1500), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(rows[sheet_start : sheet_start + 24]):
            number = sheet_start + offset + 1
            path = preview_dir / f"{number:03d}.jpg"
            if not path.exists():
                continue
            with Image.open(path) as raw:
                image = raw.copy()
            image.thumbnail((380, 215))
            x = (offset % 4) * 400 + 10
            y = (offset // 4) * 250 + 25
            sheet.paste(image, (x, y))
            draw.text((x, y - 18), f"{number:03d} {row['title'][5:48]}", fill="black")
        target = OUTPUT_DIR / f"candidates_{sheet_start // 24 + 1:02d}.jpg"
        sheet.save(target, quality=90)
        print(target)
    print(f"Candidates: {len(rows)}")


def download(session):
    if not SELECTED_NUMBERS:
        raise ValueError("Review candidate sheets and fill SELECTED_NUMBERS first")
    rows = json.loads((OUTPUT_DIR / "candidates.json").read_text(encoding="utf-8"))
    chosen = [rows[number - 1] for number in SELECTED_NUMBERS]
    image_urls = {}
    for start in range(0, len(chosen), 40):
        data = api_get(
            session,
            prop="imageinfo",
            titles="|".join(row["title"] for row in chosen[start : start + 40]),
            iiprop="url",
            iiurlwidth=1600,
        )
        for page in data["query"]["pages"].values():
            image_urls[page["title"]] = page["imageinfo"][0]["thumburl"]
    image_dir = OUTPUT_DIR / "images"
    image_dir.mkdir(exist_ok=True)
    selected = []
    for index, original_row in enumerate(chosen, 1):
        row = original_row.copy()
        title = row["title"]
        image_url = image_urls[title]
        row["filename"] = f"{index:02d}_road.jpg"
        row["image_url"] = image_url
        target = image_dir / row["filename"]
        if not target.exists():
            with Image.open(io.BytesIO(image_get(session, image_url))) as raw:
                raw.convert("RGB").save(target, quality=92)
            time.sleep(1)
        selected.append(row)
        print(f"{index}/{len(SELECTED_NUMBERS)} {target.name}: {title}")
    with (OUTPUT_DIR / "sources.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=selected[0].keys())
        writer.writeheader()
        writer.writerows(selected)
    with ZipFile(OUTPUT_DIR / "images_for_cvat.zip", "w", compression=ZIP_STORED) as archive:
        for row in selected:
            archive.write(image_dir / row["filename"], arcname=row["filename"])
    print(OUTPUT_DIR / "images_for_cvat.zip")


def package_existing():
    """Create a usable partial batch when Wikimedia throttles large downloads."""
    rows = json.loads((OUTPUT_DIR / "candidates.json").read_text(encoding="utf-8"))
    image_dir = OUTPUT_DIR / "images"
    selected = []
    for index, number in enumerate(SELECTED_NUMBERS, 1):
        target = image_dir / f"{index:02d}_road.jpg"
        if not target.exists():
            break
        row = rows[number - 1].copy()
        row["filename"] = target.name
        selected.append(row)
    if not selected:
        raise ValueError("No complete images available")
    with (OUTPUT_DIR / "sources.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=selected[0].keys())
        writer.writeheader()
        writer.writerows(selected)
    with ZipFile(OUTPUT_DIR / "images_for_cvat.zip", "w", compression=ZIP_STORED) as archive:
        for row in selected:
            archive.write(image_dir / row["filename"], arcname=row["filename"])
    print(f"Packaged {len(selected)} images: {OUTPUT_DIR / 'images_for_cvat.zip'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--package-existing", action="store_true")
    arguments = parser.parse_args()
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    if arguments.preview:
        preview(session)
    if arguments.download:
        download(session)
    if arguments.package_existing:
        package_existing()
