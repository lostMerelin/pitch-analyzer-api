from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Pitch Analyzer API", version="0.1.0")

# CORS: разрешаем фронтенд Netlify + локальную разработку
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pitch-analyzer.netlify.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Заглушка: позже добавим реальный анализ
    return {
        "filename": file.filename,
        "root_note": None,
        "nearest_note": None,
        "deviation_hz": None,
        "deviation_cents": None,
        "pitch_track": [],
    }
