import os
import uuid
import logging
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sathi")

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

API_KEY = os.getenv("GEMINI_API_KEY")
logger.info(f"API key loaded: {'YES' if API_KEY else 'NO'}")

# Configure the client with robust retry logic
# This handles 503 errors automatically with exponential backoff
retry_config = types.HttpRetryOptions(
    attempts=5,          # Try up to 5 times
    initial_delay=1.0,   # Start with 1 second
    max_delay=16.0,      # Max wait of 16 seconds
    exp_base=2.0,        # Exponential multiplier (1s, 2s, 4s, 8s, 16s)
    jitter=1.0,          # Add randomness to avoid thundering herd
)

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(
        api_version="v1",  # Use the stable v1 endpoint
        retry_options=retry_config
    )
)

# ✅ Primary model: Google's stable alias
MODEL = "gemini-flash-latest"

# ✅ Fallback models: tried in order if the primary model returns 503 or 404
# Updated based on current model availability (2026)
FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

app = FastAPI(title="Sathi AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],
)

sessions = {}

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    logger.info(f"[{session_id}] User: {req.message[:80]}")

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]
    history.append(
        types.Content(role="user", parts=[types.Part(text=req.message)])
    )

    def stream():
        full_reply = ""
        last_error = None

        # Try the primary model, then each fallback
        models_to_try = [MODEL] + FALLBACK_MODELS

        for model_name in models_to_try:
            try:
                logger.info(f"[{session_id}] Trying model: {model_name}")

                response = client.models.generate_content_stream(
                    model=model_name,
                    contents=history,
                )

                for chunk in response:
                    if chunk.text:
                        full_reply += chunk.text
                        yield chunk.text

                logger.info(f"[{session_id}] Model {model_name} succeeded")
                break

            except Exception as e:
                last_error = e
                error_str = str(e)
                logger.warning(f"[{session_id}] Model {model_name} failed: {error_str[:120]}")

                # If the model is unavailable (503) or not found (404), try the next one
                if "503" in error_str or "UNAVAILABLE" in error_str or "404" in error_str or "NOT_FOUND" in error_str:
                    continue
                else:
                    break

        if full_reply:
            history.append(
                types.Content(role="model", parts=[types.Part(text=full_reply)])
            )
            logger.info(f"[{session_id}] Replied: {len(full_reply)} chars")
        else:
            error_msg = str(last_error) if last_error else "All models unavailable"
            logger.error(f"[{session_id}] All models failed. Last error: {error_msg}")
            yield f"\n\n⚠️ All AI models are currently busy. Please try again in a moment.\n\n(Details: {error_msg[:200]})"

    return StreamingResponse(
        stream(),
        media_type="text/plain",
        headers={"X-Session-Id": session_id},
    )

@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
