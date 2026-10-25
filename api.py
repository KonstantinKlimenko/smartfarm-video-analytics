import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from fastapi import FastAPI
from html import escape
from fastapi.responses import HTMLResponse, FileResponse
from urllib.parse import quote
import subprocess
import sys
from typing import Literal
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

project_dir = Path(__file__).resolve().parent
load_dotenv(project_dir / ".env")

app = FastAPI(title="SmartFarm Video Analytics")

def open_db():
    return psycopg.connect(
        host="127.0.0.1",
        port=5433,
        dbname="smartfarm",
        user="smartfarm",
        password=os.environ["SMARTFARM_DB_PASSWORD"],
        row_factory=dict_row,
    )

@app.get("/events")
def get_events():
    with open_db() as connection:
        events = connection.execute(
            """
            SELECT id, video_name, track_id, frame_number, time_seconds, direction
            FROM crossing_events
            ORDER BY id      
            """
        ).fetchall()

    for event in events:
        event["time_seconds"] = float(event["time_seconds"])

    return events

@app.get("/summary")
def get_summary():
    with open_db() as connection:
        rows = connection.execute(
            """
            SELECT video_name, COUNT(*) AS total_crossings
            FROM crossing_events
            GROUP BY video_name
            ORDER BY video_name
            """
        ).fetchall()

    return {"runs": rows}

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (project_dir / "dashboard.html").read_text(encoding="utf-8")

class RunRequest(BaseModel):
    filename:str
    weights: Literal["base", "v2", "v3"] = "base"
    max_frames: int = Field(default=302, ge=1, le=10000)

def run_analysis(run_id: UUID, source_path: Path, run_name: str,
                 weights: str, max_frames:int):
    try:
        with open_db() as connection:
            connection.execute(
                "UPDATE analysis_runs SET status = 'running', "
                "started_at = NOW() WHERE id = %s",
                (run_id,),
            )

        command = [
            sys.executable, str(project_dir / "run_video.py"),
            "--source", str(source_path),
            "--name", run_name,
            "--weights", weights,
            "--max-frames", str(max_frames),
        ]
        result = subprocess.run(
            command, cwd=project_dir, capture_output=True, text=True
        )

        with open_db() as connection:
            if result.returncode == 0:
                connection.execute(
                    "UPDATE crossing_events SET run_id = %s "
                    "WHERE video_name = %s",
                    (run_id, run_name),
                )
                connection.execute(
                    "UPDATE analysis_runs SET status = 'completed', "
                    "finished_at = NOW(), output_video = %s WHERE id = %s",
                    (f"outputs/{run_name}_tracking.mp4", run_id),
                )
            else:
                connection.execute(
                    "UPDATE analysis_runs SET status ='failed', "
                    "finished_at = NOW(), error_message = %s WHERE id = %s",
                    ((result.stderr or result.stdout)[-2000:], run_id),
                )
    except Exception as exc:
        with open_db() as connection:
            connection.execute(
                "UPDATE analysis_runs SET status = 'failed', "
                "finished_at = NOW(), error_message = %s WHERE id = %s",
                (str(exc), run_id),
            )

@app.post("/runs")
def create_run(request: RunRequest, background_tasks: BackgroundTasks):
    if Path(request.filename).name != request.filename:
        raise HTTPException(400, "Укажи только имя файла из папки data")

    source_path = project_dir / "data" / request.filename
    if not source_path.is_file() or source_path.suffix.lower() not in {".mp4", ".webm"}:
        raise HTTPException(404, "Видео не найдено в папке data")
    run_id = uuid4()
    run_name = f"run_{run_id.hex}"

    with open_db() as connection:
        connection.execute(
            "INSERT INTO analysis_runs (id, source_name, weights, status) "
            "VALUES (%s, %s, %s, 'queued')",
            (run_id, request.filename, request.weights),
        )

    background_tasks.add_task(
        run_analysis, run_id, source_path, run_name,
        request.weights, request.max_frames,
    )
    return {"id": run_id, "status": "queued"}

@app.get("/runs/{run_id}")
def get_run(run_id: UUID):
    with open_db() as connection:
        run = connection.execute(
            "SELECT * FROM analysis_runs WHERE id = %s", (run_id,)
        ).fetchone()

    if run is None:
        raise HTTPException(404, "Запуск не найден")
    return run

@app.get("/runs/{run_id}/events")
def get_run_events(run_id: UUID):
    get_run(run_id)  # Проверяем, что такой запуск существует

    with open_db() as connection:
        events = connection.execute(
            """
            SELECT id, track_id, frame_number, time_seconds, direction
            FROM crossing_events
            WHERE run_id = %s
            ORDER BY time_seconds, id
            """,
            (run_id,),
        ).fetchall()

    for event in events:
        event["time_seconds"] = float(event["time_seconds"])

    return events


@app.get("/runs/{run_id}/video")
def get_run_video(run_id: UUID):
    run = get_run(run_id)

    if run["status"] != "completed" or not run["output_video"]:
        raise HTTPException(409, "Обработанное видео ещё не готово")

    outputs_dir = (project_dir / "outputs").resolve()
    video_path = (project_dir / run["output_video"]).resolve()

    if not video_path.is_relative_to(outputs_dir):
        raise HTTPException(500, "Некорректный путь к результату")
    if not video_path.is_file():
        raise HTTPException(404, "Файл результата не найден")

    return FileResponse(video_path, media_type="video/mp4")

@app.get("/runs")
def list_runs():
    with open_db() as connection:
        runs = connection.execute(
            """
            SELECT
                r.id, r.source_name, r.weights, r.status,
                r.created_at, r.finished_at, r.output_video,
                COUNT(e.id) AS total_crossings
            FROM analysis_runs AS r
            LEFT JOIN crossing_events AS e ON e.run_id = r.id
            GROUP BY r.id
            ORDER BY r.created_at DESC
            LIMIT 50
            """
        ).fetchall()

    return runs

@app.get("/sources")
def list_sources():
    data_dir = project_dir / "data"
    files = sorted(
        path.name
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".webm"}
    )
    return {"files": files}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)