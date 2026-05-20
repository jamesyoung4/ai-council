from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

load_dotenv()

import attachments  # noqa: E402
import db  # noqa: E402
from council import run_council  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/api/conversations")
async def list_conversations() -> JSONResponse:
    return JSONResponse(db.list_conversations())


@app.post("/api/conversations")
async def create_conversation() -> JSONResponse:
    conv_id = db.create_conversation()
    return JSONResponse({"id": conv_id, "title": "New conversation"})


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: int) -> JSONResponse:
    conv = db.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(404, detail="Conversation not found")
    return JSONResponse(conv)


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: int) -> JSONResponse:
    db.delete_conversation(conv_id)
    return JSONResponse({"ok": True})


@app.post("/api/conversations/{conv_id}/ask")
async def ask(
    conv_id: int,
    question: str = Form(...),
    files: list[UploadFile] = File(default=[]),
) -> StreamingResponse:
    if db.get_conversation(conv_id) is None:
        raise HTTPException(404, detail="Conversation not found")

    extracted: list[dict] = []
    for f in files:
        if not f.filename:
            continue
        try:
            data = await f.read()
            text = attachments.extract_text(f.filename, data)
        except attachments.AttachmentError as e:
            raise HTTPException(400, detail=str(e))
        extracted.append({"filename": f.filename, "content": text})

    return StreamingResponse(
        run_council(conv_id, question, extracted),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
