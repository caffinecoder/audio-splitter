"""
Audio Segment Tool — FastAPI Backend
- Noise reduction is optional (skipped by default for speed)
- Job state persisted to disk
- Interrupted jobs auto-resume on startup
"""

import uuid
import shutil
import zipfile
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import librosa
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_DIR = Path("jobs")
BASE_DIR.mkdir(exist_ok=True)

SAMPLE_RATE  = 22050
MIN_DURATION = 5.0
MIN_RMS      = 0.01

executor = ThreadPoolExecutor(max_workers=2)

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Audio Segment API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=str(BASE_DIR)), name="files")

# ─── Job state: persisted to disk ─────────────────────────────────────────────

def job_file(job_id: str) -> Path:
    return BASE_DIR / job_id / "job.json"

def load_job(job_id: str) -> dict | None:
    f = job_file(job_id)
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None

def save_job(job_id: str, data: dict):
    f = job_file(job_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: str(v) if isinstance(v, Path) else v for k, v in data.items()}
    f.write_text(json.dumps(serializable))

def update_job(job_id: str, **kwargs):
    job = load_job(job_id) or {}
    job.update(kwargs)
    save_job(job_id, job)

# ─── Models ───────────────────────────────────────────────────────────────────

class SegmentRequest(BaseModel):
    jobId: str
    segmentMinutes: int = 5
    denoise: bool = False  # off by default — too slow on free tier

# ─── Startup: resume interrupted jobs ─────────────────────────────────────────

@app.on_event("startup")
async def resume_interrupted_jobs():
    if not BASE_DIR.exists():
        return
    for job_dir in BASE_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        job_id = job_dir.name
        job = load_job(job_id)
        if job and job.get("status") == "processing":
            update_job(job_id, progress=5, message="Resuming after server restart…")
            executor.submit(
                _process_audio,
                job_id,
                job.get("segmentMinutes", 5),
                job.get("denoise", False),
            )

# ─── Blocking audio work ──────────────────────────────────────────────────────

def _process_audio(job_id: str, segment_minutes: int, denoise: bool = False):
    job = load_job(job_id)
    if not job:
        return

    update_job(job_id, status="processing", progress=0)

    try:
        audio_path   = Path(job["audio_path"])
        segments_dir = BASE_DIR / job_id / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load
        update_job(job_id, progress=10)
        audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

        # 2. Noise reduction (optional — slow on free tier)
        update_job(job_id, progress=20)
        if denoise:
            try:
                import noisereduce as nr
                audio = nr.reduce_noise(y=audio, sr=sr)
            except Exception:
                pass  # skip silently if it fails

        # 3. Split
        update_job(job_id, progress=40)
        seg_len      = sr * 60 * segment_minutes
        raw_segments = [
            audio[i: i + seg_len]
            for i in range(0, len(audio), seg_len)
        ]

        # 4. Filter & save
        update_job(job_id, progress=60)
        saved = []
        for idx, seg in enumerate(raw_segments):
            duration = librosa.get_duration(y=seg, sr=sr)
            rms      = float(np.sqrt(np.mean(seg ** 2)))
            if duration < MIN_DURATION or rms < MIN_RMS:
                continue
            filename = f"segment_{idx:03d}.wav"
            out_path = segments_dir / filename
            sf.write(str(out_path), seg, sr)
            start_sec = idx * 60 * segment_minutes
            saved.append({
                "id":          f"{job_id}-{idx}",
                "label":       f"Segment {idx + 1}",
                "startSec":    round(start_sec, 2),
                "endSec":      round(start_sec + duration, 2),
                "downloadUrl": f"/files/{job_id}/segments/{filename}",
            })

        # 5. Zip
        update_job(job_id, progress=85)
        zip_path = BASE_DIR / job_id / "segments.zip"
        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for entry in saved:
                seg_file = segments_dir / Path(entry["downloadUrl"]).name
                zf.write(str(seg_file), Path(entry["downloadUrl"]).name)

        update_job(job_id,
                   progress=100,
                   status="done",
                   segments=saved,
                   zip_path=str(zip_path))

    except Exception as exc:
        update_job(job_id, status="error", message=str(exc))

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...)):
    if not (audio.content_type or "").startswith("audio/"):
        raise HTTPException(400, "File must be an audio type.")

    job_id  = str(uuid.uuid4())
    job_dir = BASE_DIR / job_id
    job_dir.mkdir(parents=True)

    suffix    = Path(audio.filename or "audio").suffix or ".wav"
    save_path = job_dir / f"original{suffix}"
    content   = await audio.read()

    with open(save_path, "wb") as f:
        f.write(content)

    try:
        duration = librosa.get_duration(path=str(save_path))
    except Exception:
        duration = 0.0

    save_job(job_id, {
        "status":         "pending",
        "progress":       0,
        "message":        "",
        "audio_path":     str(save_path),
        "segments":       [],
        "zip_path":       None,
        "filename":       audio.filename,
        "size":           len(content),
        "duration":       round(duration, 2),
        "segmentMinutes": 5,
        "denoise":        False,
    })

    return {
        "jobId":    job_id,
        "filename": audio.filename,
        "size":     len(content),
        "duration": round(duration, 2),
    }


@app.post("/segment")
async def start_segmentation(req: SegmentRequest):
    job = load_job(req.jobId)
    if not job:
        raise HTTPException(404, "Job not found. Upload audio first.")
    if job["status"] == "processing":
        raise HTTPException(409, "Job is already processing.")

    seg_min = max(1, min(req.segmentMinutes, 120))
    update_job(req.jobId, segmentMinutes=seg_min, denoise=req.denoise)
    executor.submit(_process_audio, req.jobId, seg_min, req.denoise)
    return {"jobId": req.jobId, "segmentMinutes": seg_min, "denoise": req.denoise}


@app.get("/segment/{job_id}/status")
async def get_job_status(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    response = {"status": job["status"], "progress": job["progress"]}
    if job["status"] == "done":
        response["segments"] = job["segments"]
    elif job["status"] == "error":
        response["message"] = job.get("message", "Unknown error")
    return response


@app.get("/segment/{job_id}/download-all")
async def download_all(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job["status"] != "done":
        raise HTTPException(409, "Segments not ready yet.")
    zip_path = job.get("zip_path")
    if not zip_path or not Path(zip_path).exists():
        raise HTTPException(500, "Zip file missing.")
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"segments_{job_id[:8]}.zip",
    )


@app.delete("/segment/{job_id}")
async def delete_job(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    shutil.rmtree(str(BASE_DIR / job_id), ignore_errors=True)
    return {"deleted": True, "jobId": job_id}