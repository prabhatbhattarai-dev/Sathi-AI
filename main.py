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

# Models tried in order. If one fails with 503/404, next one is tried.
MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
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
    history.append(types.Content(role="user", parts=[types.Part(text=req.message)]))

    def stream():
        full_reply = ""
        last_error = None

        for model_name in MODELS:
            # Try each model up to 3 times with increasing waits
            for attempt in range(3):
                try:
                    logger.info(f"[{session_id}] Trying {model_name} (attempt {attempt+1})")
                    got_any = False

                    response = client.models.generate_content_stream(
                        model=model_name,
                        contents=history,
                    )
                    for chunk in response:
                        if chunk.text:
                            got_any = True
                            full_reply += chunk.text
                            yield chunk.text

                    if got_any:
                        logger.info(f"[{session_id}] ✅ Success with {model_name}")
                        history.append(
                            types.Content(role="model", parts=[types.Part(text=full_reply)])
                        )
                        return

                except Exception as e:
                    last_error = e
                    err = str(e)
                    logger.warning(f"[{session_id}] {model_name} attempt {attempt+1} failed: {err[:150]}")

                    is_retired = any(k in err for k in ["404", "NOT_FOUND", "no longer available"])
                    is_busy = any(k in err for k in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "high demand", "overloaded"])

                    if is_retired:
                        break  # move to next model immediately
                    if is_busy and attempt < 2:
                        wait = 3 * (attempt + 1)  # 3s, 6s
                        logger.info(f"[{session_id}] Waiting {wait}s before retry...")
                        time.sleep(wait)
                        continue
                    break  # move to next model

        err_msg = str(last_error) if last_error else "All models busy"
        logger.error(f"[{session_id}] All failed: {err_msg[:200]}")
        yield f"\n\n⚠️ Sathi is a bit overloaded right now. Please try again in a few seconds.\n\n_(Technical: {err_msg[:150]})_"

    return StreamingResponse(
        stream(),
        media_type="text/plain",
        headers={"X-Session-Id": session_id},
    )


@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
