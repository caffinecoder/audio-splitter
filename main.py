"""
Audio Segment Tool — FastAPI Backend
Matches the API contract expected by the frontend:
  POST   /upload
  POST   /segment
  GET    /segment/{job_id}/status
  GET    /segment/{job_id}/download-all
  DELETE /segment/{job_id}
"""

import os
import uuid
import shutil
import asyncio
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_DIR   = Path("jobs")          # all job data lives here
BASE_DIR.mkdir(exist_ok=True)

SAMPLE_RATE     = 22050
MIN_DURATION    = 5.0              # seconds — segments shorter than this are skipped
MIN_RMS         = 0.01             # energy floor

ALLOWED_MIME_PREFIXES = ("audio/",)

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Audio Segment API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated segment files at /files/<job_id>/segments/<filename>
app.mount("/files", StaticFiles(directory=str(BASE_DIR)), name="files")

# ─── In-memory job store ──────────────────────────────────────────────────────
# For production, swap this for Redis / a database.

jobs: dict[str, dict] = {}
# Shape of each job entry:
# {
#   "status":        "pending" | "processing" | "done" | "error",
#   "progress":      int (0-100),
#   "message":       str,
#   "audio_path":    Path,
#   "segments_dir":  Path,
#   "segments":      list[dict],   # populated when done
#   "zip_path":      Path | None,
#   "filename":      str,
#   "size":          int,
# }

# ─── Pydantic models ──────────────────────────────────────────────────────────

class SegmentRequest(BaseModel):
    jobId: str
    segmentMinutes: int = 5

# ─── Audio processing (runs in a thread pool) ─────────────────────────────────

def _process_audio(job_id: str, segment_minutes: int):
    """
    Blocking function — called via run_in_executor so it doesn't block the
    event loop. Updates jobs[job_id] in-place as it progresses.
    """
    job = jobs[job_id]
    job["status"]   = "processing"
    job["progress"] = 0

    try:
        audio_path   = job["audio_path"]
        segments_dir = job["segments_dir"]
        segments_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Load ───────────────────────────────────────────────────────────
        job["progress"] = 10
        audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

        # ── 2. Noise reduction ────────────────────────────────────────────────
        job["progress"] = 25
        reduced = nr.reduce_noise(y=audio, sr=sr)

        # ── 3. Split into requested-length segments ────────────────────────────
        job["progress"] = 40
        seg_len = sr * 60 * segment_minutes
        raw_segments = [
            reduced[i : i + seg_len]
            for i in range(0, len(reduced), seg_len)
        ]

        # ── 4. Filter & save ──────────────────────────────────────────────────
        job["progress"] = 60
        saved = []
        for idx, seg in enumerate(raw_segments):
            duration = librosa.get_duration(y=seg, sr=sr)
            rms = float(np.sqrt(np.mean(seg ** 2)))

            if duration < MIN_DURATION or rms < MIN_RMS:
                continue  # skip near-silent / too-short segments

            filename = f"segment_{idx:03d}.wav"
            out_path = segments_dir / filename
            sf.write(str(out_path), seg, sr)

            start_sec = idx * 60 * segment_minutes
            end_sec   = start_sec + duration

            saved.append({
                "id":          f"{job_id}-{idx}",
                "label":       f"Segment {idx + 1}",
                "startSec":    round(start_sec, 2),
                "endSec":      round(end_sec, 2),
                "downloadUrl": f"/files/{job_id}/segments/{filename}",
            })

        # ── 5. Build zip ──────────────────────────────────────────────────────
        job["progress"] = 85
        zip_path = BASE_DIR / job_id / "segments.zip"
        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for entry in saved:
                seg_file = segments_dir / Path(entry["downloadUrl"]).name
                zf.write(str(seg_file), Path(entry["downloadUrl"]).name)

        # ── Done ──────────────────────────────────────────────────────────────
        job["progress"] = 100
        job["segments"] = saved
        job["zip_path"] = zip_path
        job["status"]   = "done"

    except Exception as exc:
        job["status"]  = "error"
        job["message"] = str(exc)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...)):
    """
    Accepts a multipart audio file.
    Returns { jobId, filename, size, duration }
    """
    if not (audio.content_type or "").startswith(ALLOWED_MIME_PREFIXES):
        raise HTTPException(400, "File must be an audio type.")

    job_id   = str(uuid.uuid4())
    job_dir  = BASE_DIR / job_id
    job_dir.mkdir(parents=True)

    # Save the original upload
    suffix    = Path(audio.filename or "audio").suffix or ".wav"
    save_path = job_dir / f"original{suffix}"
    content   = await audio.read()

    with open(save_path, "wb") as f:
        f.write(content)

    # Peek at duration without loading the whole array into RAM
    try:
        duration = librosa.get_duration(path=str(save_path))
    except Exception:
        duration = 0.0

    jobs[job_id] = {
        "status":       "pending",
        "progress":     0,
        "message":      "",
        "audio_path":   save_path,
        "segments_dir": job_dir / "segments",
        "segments":     [],
        "zip_path":     None,
        "filename":     audio.filename,
        "size":         len(content),
        "duration":     round(duration, 2),
    }

    return {
        "jobId":    job_id,
        "filename": audio.filename,
        "size":     len(content),
        "duration": round(duration, 2),
    }


@app.post("/segment")
async def start_segmentation(req: SegmentRequest, background_tasks: BackgroundTasks):
    """
    Kicks off segmentation as a background task.
    Returns { jobId, segmentMinutes }
    """
    if req.jobId not in jobs:
        raise HTTPException(404, "Job not found. Upload audio first.")

    job = jobs[req.jobId]
    if job["status"] == "processing":
        raise HTTPException(409, "Job is already processing.")

    # Run the blocking work in a thread so we don't stall the event loop
    background_tasks.add_task(
        asyncio.get_event_loop().run_in_executor,
        None,
        _process_audio,
        req.jobId,
        max(1, min(req.segmentMinutes, 120)),
    )

    return {
        "jobId":          req.jobId,
        "segmentMinutes": req.segmentMinutes,
    }


@app.get("/segment/{job_id}/status")
async def get_job_status(job_id: str):
    """
    Polls segmentation progress.
    Returns { status, progress, segments? }
    """
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")

    job = jobs[job_id]
    response = {
        "status":   job["status"],
        "progress": job["progress"],
    }

    if job["status"] == "done":
        response["segments"] = job["segments"]
    elif job["status"] == "error":
        response["message"] = job["message"]

    return response


@app.get("/segment/{job_id}/download-all")
async def download_all(job_id: str):
    """
    Returns the zip of all segments as a file download.
    """
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")

    job = jobs[job_id]

    if job["status"] != "done":
        raise HTTPException(409, "Segments not ready yet.")

    zip_path = job.get("zip_path")
    if not zip_path or not Path(zip_path).exists():
        raise HTTPException(500, "Zip file missing. Re-process the audio.")

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"segments_{job_id[:8]}.zip",
    )


@app.delete("/segment/{job_id}")
async def delete_job(job_id: str):
    """
    Cleans up all server-side files and removes the job from memory.
    """
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")

    job_dir = BASE_DIR / job_id
    try:
        shutil.rmtree(str(job_dir), ignore_errors=True)
    except Exception:
        pass

    jobs.pop(job_id, None)
    return {"deleted": True, "jobId": job_id}


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}