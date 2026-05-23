from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

import attachments  # noqa: E402
import db  # noqa: E402
from council import DEFAULT_PERSONAS, resolve_personas, run_council  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(lifespan=lifespan)


class Persona(BaseModel):
    name: str
    system: str


class CreateConversation(BaseModel):
    personas: list[Persona] | None = None


class UpdatePersonas(BaseModel):
    personas: list[Persona]


def _validate_personas(personas: list[Persona] | None) -> list[dict]:
    if not personas:
        return []
    if len(personas) != 3:
        raise HTTPException(400, detail="must provide exactly 3 personas (or none)")
    out = []
    for p in personas:
        name = (p.name or "").strip()
        system = (p.system or "").strip()
        if not name:
            raise HTTPException(400, detail="each persona needs a non-empty name")
        if not system:
            raise HTTPException(400, detail=f"persona '{name}' needs a non-empty system prompt")
        if len(name) > 60:
            raise HTTPException(400, detail=f"persona name too long: '{name}'")
        if len(system) > 4000:
            raise HTTPException(400, detail=f"persona '{name}' system prompt too long")
        out.append({"name": name, "system": system})
    names = [p["name"] for p in out]
    if len(set(names)) != 3:
        raise HTTPException(400, detail="persona names must be unique")
    return out


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/api/personas/default")
async def default_personas() -> JSONResponse:
    return JSONResponse(DEFAULT_PERSONAS)


@app.get("/api/conversations")
async def list_conversations() -> JSONResponse:
    return JSONResponse(db.list_conversations())


@app.post("/api/conversations")
async def create_conversation(body: CreateConversation | None = None) -> JSONResponse:
    personas = _validate_personas(body.personas if body else None)
    conv_id = db.create_conversation(personas=personas)
    return JSONResponse(
        {
            "id": conv_id,
            "title": "New conversation",
            "personas": resolve_personas(personas),
        }
    )


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: int) -> JSONResponse:
    conv = db.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(404, detail="Conversation not found")
    conv["personas"] = resolve_personas(conv["personas"])
    return JSONResponse(conv)


@app.patch("/api/conversations/{conv_id}/personas")
async def update_personas(conv_id: int, body: UpdatePersonas) -> JSONResponse:
    if db.get_conversation(conv_id) is None:
        raise HTTPException(404, detail="Conversation not found")
    if db.turn_count(conv_id) > 0:
        raise HTTPException(409, detail="Cannot change personas after the council has spoken")
    personas = _validate_personas(body.personas)
    db.update_personas(conv_id, personas)
    return JSONResponse({"personas": resolve_personas(personas)})


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: int) -> JSONResponse:
    db.delete_conversation(conv_id)
    return JSONResponse({"ok": True})


@app.post("/api/conversations/{conv_id}/ask")
async def ask(
    conv_id: int,
    question: str = Form(...),
    use_web: bool = Form(False),
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
        run_council(conv_id, question, extracted, use_web=use_web),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
