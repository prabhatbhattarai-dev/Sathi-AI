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

# Setup logging so we can see errors in Render logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sathi")

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

API_KEY = os.getenv("GEMINI_API_KEY")
logger.info(f"API key loaded: {'YES' if API_KEY else 'NO'} (length: {len(API_KEY) if API_KEY else 0})")

client = genai.Client(api_key=API_KEY)

# ✅ Primary model: Google's stable alias that always points to the latest Flash model
MODEL = "gemini-flash-latest"

# ✅ Fallback models: tried in order if the primary model returns 503 or 404
FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.8-flash",
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

    # Get or create history for this session
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    # Append user message
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

                # If we got here, the model worked
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
                    # Some other error (bad API key, etc.) — don't retry
                    break

        if full_reply:
            # Save assistant reply to history
            history.append(
                types.Content(role="model", parts=[types.Part(text=full_reply)])
            )
            logger.info(f"[{session_id}] Replied: {len(full_reply)} chars")
        else:
            # All models failed
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
