import os
import uuid
import logging
import time
import sqlite3
import base64
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sathi")

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = BASE_DIR / "chat_history.db"

UPLOAD_DIR.mkdir(exist_ok=True)

# ---------- Database Setup ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_conversation(conv_id, title=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
              (conv_id, title or "New Chat", datetime.now().isoformat()))
    conn.commit()
    conn.close()

def save_message(conv_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
              (conv_id, role, content, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_conversations():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]

def get_messages(conv_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id", (conv_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]

# ---------- Gemini Client ----------
API_KEY = os.getenv("GEMINI_API_KEY")
logger.info(f"API key loaded: {'YES' if API_KEY else 'NO'}")

client = genai.Client(api_key=API_KEY)

MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-flash-latest",
    "gemini-3.8-flash",
]

# System instruction shapes Sathi's personality[reference:0]
SYSTEM_INSTRUCTION = """You are Sathi AI, a friendly and helpful AI assistant from Nepal.
- Be warm, respectful, and use Nepali cultural context when relevant.
- If you don't know something, say so honestly.
- Use markdown formatting for code, lists, and emphasis.
- Keep responses concise unless the user asks for detail.
- When the user asks about Nepal, provide accurate, up-to-date information.
- You are made with ❤️ in Nepal, powered by Gemini."""

# Tools: Google Search grounding + code execution[reference:1]
tools = [
    types.Tool(google_search=types.GoogleSearch()),
    types.Tool(code_execution=types.ToolCodeExecution()),
]

app = FastAPI(title="Sathi AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],
)

# In-memory session cache (also persisted to SQLite)
sessions = {}

# ---------- Models ----------
class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

class NewChatRequest(BaseModel):
    title: str | None = None

# ---------- Routes ----------
@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    logger.info(f"[{session_id}] User: {req.message[:80]}")

    # Load history from DB
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    # Save user message
    save_conversation(session_id, req.message[:40])
    save_message(session_id, "user", req.message)
    history.append(types.Content(role="user", parts=[types.Part(text=req.message)]))

    def stream():
        full_reply = ""
        last_error = None

        for model_name in MODELS:
            for attempt in range(2):
                try:
                    logger.info(f"[{session_id}] Trying {model_name} (attempt {attempt+1})")

                    response = client.models.generate_content_stream(
                        model=model_name,
                        contents=history,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_INSTRUCTION,
                            tools=tools,
                        ),
                    )
                    for chunk in response:
                        if chunk.text:
                            full_reply += chunk.text
                            yield chunk.text

                    if full_reply:
                        logger.info(f"[{session_id}] ✅ Success with {model_name}")
                        save_message(session_id, "model", full_reply)
                        history.append(
                            types.Content(role="model", parts=[types.Part(text=full_reply)])
                        )
                        return

                except Exception as e:
                    last_error = e
                    err = str(e)
                    logger.warning(f"[{session_id}] {model_name} failed: {err[:150]}")
                    if any(k in err for k in ["404", "NOT_FOUND", "no longer available"]):
                        break
                    if any(k in err for k in ["503", "UNAVAILABLE", "429"]):
                        time.sleep(2)
                        continue
                    break

        err_msg = str(last_error) if last_error else "All models busy"
        yield f"\n\n⚠️ Sathi is temporarily busy. Please try again.\n\n(Details: {err_msg[:150]})"

    return StreamingResponse(
        stream(),
        media_type="text/plain",
        headers={"X-Session-Id": session_id},
    )

@app.post("/new-chat")
async def new_chat(req: NewChatRequest):
    conv_id = str(uuid.uuid4())
    save_conversation(conv_id, req.title or "New Chat")
    return {"id": conv_id}

@app.get("/conversations")
async def conversations():
    return get_conversations()

@app.get("/conversations/{conv_id}")
async def conversation_detail(conv_id: str):
    return {"id": conv_id, "messages": get_messages(conv_id)}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload image or document for multimodal chat."""
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".bin"
    dest = UPLOAD_DIR / f"{file_id}{ext}"
    content = await file.read()
    dest.write_bytes(content)

    # Encode for Gemini
    b64 = base64.b64encode(content).decode("utf-8")
    mime = file.content_type or "application/octet-stream"

    return {
        "file_id": file_id,
        "filename": file.filename,
        "mime_type": mime,
        "base64": b64[:100] + "...",  # truncated for preview
        "path": str(dest),
    }

@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
