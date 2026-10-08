import os
import uuid
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sathi")

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

API_KEY = os.getenv("GEMINI_API_KEY")
logger.info(f"API key loaded: {'YES' if API_KEY else 'NO'}")

client = genai.Client(api_key=API_KEY)

# Primary model + fallbacks (tried in order if one is busy or retired)
MODEL = "gemini-2.5-flash"
FALLBACK_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
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


def try_stream(model_name, history):
    """Generator that yields text chunks. Raises on failure."""
    response = client.models.generate_content_stream(
        model=model_name,
        contents=history,
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    logger.info(f"[{session_id}] User: {req.message[:80]}")

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]
    history.append(types.Content(role="user", parts=[types.Part(text=req.message)]))

    def stream():
        full_reply = ""
        last_error = None
        models_to_try = [MODEL] + FALLBACK_MODELS

        for model_name in models_to_try:
            # Retry same model up to 2 times on transient errors
            for attempt in range(2):
                try:
                    logger.info(f"[{session_id}] Trying model={model_name} attempt={attempt+1}")
                    got_any = False
                    for chunk in try_stream(model_name, history):
                        got_any = True
                        full_reply += chunk
                        yield chunk
                    if got_any:
                        logger.info(f"[{session_id}] Model {model_name} succeeded")
                        history.append(
                            types.Content(role="model", parts=[types.Part(text=full_reply)])
                        )
                        return
                except Exception as e:
                    last_error = e
                    err = str(e)
                    logger.warning(f"[{session_id}] {model_name} attempt {attempt+1} failed: {err[:150]}")
                    is_transient = any(k in err for k in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded", "high demand"])
                    is_retired = any(k in err for k in ["404", "NOT_FOUND", "no longer available"])
                    if is_retired:
                        break
                    if is_transient and attempt == 0:
                        time.sleep(2)
                        continue
                    break

        err_msg = str(last_error) if last_error else "All models busy"
        logger.error(f"[{session_id}] All attempts failed. Last: {err_msg[:200]}")
        yield f"\n\n⚠️ AI is temporarily busy. Please try again in a few seconds.\n\n(Details: {err_msg[:180]})"

    return StreamingResponse(
        stream(),
        media_type="text/plain",
        headers={"X-Session-Id": session_id},
    )


@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
