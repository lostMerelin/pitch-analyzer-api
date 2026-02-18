from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.audio_decode import AudioDecodeError, decode_to_mono_float32
from app.pitch_tracker import PitchOptions, downsample_pitch_track, run_pitch_tracking
from app.root_note import estimate_root_note, nearest_note_and_deviation

app = FastAPI(
    title="Pitch Analyzer API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pitch-analyzer.netlify.app",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    payload = await file.read()
    try:
        samples, sr = decode_to_mono_float32(payload, file.filename or "upload")
    except AudioDecodeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:  # pragma: no cover - safety net.
        raise HTTPException(status_code=500, detail="Internal audio decode error.") from exc

    try:
        frames = run_pitch_tracking(samples, sr=sr, opts=PitchOptions(sr=sr))
        root_note, f_star = estimate_root_note(frames)
        nearest_note = None
        deviation_hz = None
        deviation_cents = None
        if f_star is not None:
            nearest_note, deviation_hz, deviation_cents = nearest_note_and_deviation(f_star)
            if deviation_hz is not None:
                deviation_hz = round(float(deviation_hz), 2)
        pitch_track = downsample_pitch_track(frames, step_sec=0.05)
    except Exception as exc:  # pragma: no cover - safety net.
        raise HTTPException(status_code=500, detail="Audio analysis failed.") from exc

    return {
        "filename": file.filename,
        "root_note": root_note,
        "nearest_note": nearest_note,
        "deviation_hz": deviation_hz,
        "deviation_cents": deviation_cents,
        "pitch_track": pitch_track,
    }
