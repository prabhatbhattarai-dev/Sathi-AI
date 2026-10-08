import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

load_dotenv()

# Absolute paths — prevents "file not found" errors when deployed
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

# Create FastAPI app
app = FastAPI(title="Sathi AI")

# Allow the frontend (Vercel) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],
)

# In-memory chat sessions (resets when server restarts — fine for MVP)
sessions = {}


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = client.chats.create(model=MODEL)

    chat_session = sessions[session_id]

    def stream():
        for chunk in chat_session.send_message_stream(req.message):
            if chunk.text:
                yield chunk.text

    return StreamingResponse(
        stream(),
        media_type="text/plain",
        headers={"X-Session-Id": session_id},
    )


@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


# Serve any other static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
