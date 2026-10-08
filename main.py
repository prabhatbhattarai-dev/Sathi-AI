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

# ✅ FIXED: Google retired gemini-2.0-flash. Use the current model.
MODEL = "gemini-3.8-flash"

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
        try:
            response = client.models.generate_content_stream(
                model=MODEL,
                contents=history,
            )
            for chunk in response:
                if chunk.text:
                    full_reply += chunk.text
                    yield chunk.text

            # Save assistant reply to history
            history.append(
                types.Content(role="model", parts=[types.Part(text=full_reply)])
            )
            logger.info(f"[{session_id}] Replied: {len(full_reply)} chars")

        except Exception as e:
            logger.exception(f"[{session_id}] Gemini error: {e}")
            yield f"\n\n⚠️ Error: {str(e)}"

    return StreamingResponse(
        stream(),
        media_type="text/plain",
        headers={"X-Session-Id": session_id},
    )


@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
