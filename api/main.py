import os
import sys
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import magic
from openai import OpenAI
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.rate_limiter import limiter
from src.auth.dependencies import require_authenticated_user, require_admin
from src.auth.models import TokenData
from src.auth.routes import router as auth_router
from src.utils.observability import get_langfuse, trace_context
from src.query.cache import get_cached, make_key, set_cached
from src.query.retriever import retrieve
from src.query.reranker import rerank
from src.query.augmenter import build_prompt
from src.query.generator import generate
from src.ingestion.loader import load_documents, save_ingested_hashes, HASH_STORE
from src.ingestion.parser import parse_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_documents

logger = get_logger("api")
config = get_config()

app = FastAPI(title="RAG API", description="Production-ready RAG service", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

_raw_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000")
CORS_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


@app.on_event("shutdown")
def shutdown() -> None:
    get_langfuse().flush()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    latencies: dict
    chunks_used: int
    cached: bool = False


class HealthResponse(BaseModel):
    status: str
    version: str
    checks: dict


class StatsResponse(BaseModel):
    document_count: int
    chunk_count: int
    last_ingestion: str | None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Deep health check : vérifie ChromaDB et l'accès OpenAI."""
    checks: dict = {}
    overall = "ok"

    try:
        client = chromadb.PersistentClient(path=config["paths"]["chroma_db"])
        collection = client.get_or_create_collection(config["embedding"]["collection_name"])
        checks["chromadb"] = {"status": "ok", "chunks": collection.count()}
    except Exception as e:
        checks["chromadb"] = {"status": "error", "detail": str(e)}
        overall = "degraded"
        logger.error(f"❌ Health ChromaDB : {e}")

    try:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        openai_client.models.list()
        checks["openai"] = {"status": "ok"}
    except Exception as e:
        checks["openai"] = {"status": "error", "detail": str(e)}
        overall = "degraded"
        logger.error(f"❌ Health OpenAI : {e}")

    logger.info(f"🏥 Health check — status={overall} checks={checks}")
    return HealthResponse(status=overall, version="1.0.0", checks=checks)


# ---------------------------------------------------------------------------
# Stats (tout utilisateur authentifié)
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".txt"}


@app.get("/api/stats", response_model=StatsResponse)
def stats(token_data: TokenData = Depends(require_authenticated_user)) -> StatsResponse:
    """Statistiques de l'index documentaire."""
    logger.info(f"📊 /api/stats — user: {token_data.username}")

    # Chunk count via ChromaDB
    try:
        chroma_client = chromadb.PersistentClient(path=config["paths"]["chroma_db"])
        collection = chroma_client.get_or_create_collection(config["embedding"]["collection_name"])
        chunk_count = collection.count()
    except Exception as e:
        logger.error(f"❌ Stats ChromaDB : {e}")
        chunk_count = -1

    # Document count via data/raw/
    raw_dir = Path(config["paths"]["raw_data"])
    document_count = sum(
        1 for f in raw_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ) if raw_dir.exists() else 0

    # Last ingestion via mtime de ingested_files.json
    last_ingestion: str | None = None
    if HASH_STORE.exists():
        mtime = HASH_STORE.stat().st_mtime
        last_ingestion = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

    return StatsResponse(
        document_count=document_count,
        chunk_count=chunk_count,
        last_ingestion=last_ingestion,
    )


# ---------------------------------------------------------------------------
# Query (admin + user)
# ---------------------------------------------------------------------------


@app.post("/api/query", response_model=QueryResponse)
@limiter.limit("30/minute")
def query(
    request: Request,
    body: QueryRequest,
    token_data: TokenData = Depends(require_authenticated_user),
) -> QueryResponse:
    logger.info(f"❓ /api/query — '{body.question}' — user: {token_data.username} (role={token_data.role})")

    latencies: dict = {}

    try:
        with trace_context(
            "rag-query",
            input={"question": body.question, "top_k": body.top_k},
            user_id=token_data.username,
        ) as trace:
            t0 = time.time()
            documents, chunk_ids = retrieve(body.question, k=20)
            latencies["retrieval"] = round(time.time() - t0, 3)

            if not documents:
                raise HTTPException(status_code=404, detail="Aucun document pertinent trouvé.")

            cache_key = make_key(body.question, body.top_k, chunk_ids)
            cached_response = get_cached(cache_key)
            if cached_response is not None:
                return cached_response.model_copy(update={"cached": True})

            t0 = time.time()
            documents = rerank(body.question, documents, top_k=body.top_k)
            latencies["rerank"] = round(time.time() - t0, 3)

            sources = list(dict.fromkeys(doc.metadata.get("source_file", "?") for doc in documents))

            t0 = time.time()
            prompt = build_prompt(body.question, documents)
            answer = generate(prompt)
            latencies["llm"] = round(time.time() - t0, 3)

            latencies["total"] = round(sum(latencies.values()), 3)
            trace.update(output={"answer": answer, "latencies": latencies})

            logger.info(f"✅ Réponse générée en {latencies['total']}s")

            response = QueryResponse(
                answer=answer,
                sources=sources,
                latencies=latencies,
                chunks_used=len(documents),
            )
            set_cached(cache_key, response)
            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur /api/query : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Upload (admin uniquement)
# ---------------------------------------------------------------------------

ALLOWED_MIME_TYPES = {
    ".pdf": {"application/pdf"},
    ".html": {"text/html"},
    ".txt": {"text/plain"},
}


@app.post("/api/upload")
@limiter.limit("5/minute")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    token_data: TokenData = Depends(require_admin),
) -> dict:
    logger.info(f"📤 /api/upload — '{file.filename}' — user: {token_data.username}")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Format non supporté : {ext}")

    header = await file.read(4096)
    await file.seek(0)
    detected_mime = magic.from_buffer(header, mime=True).split(";")[0].strip()
    if detected_mime not in ALLOWED_MIME_TYPES.get(ext, set()):
        raise HTTPException(
            status_code=400,
            detail=f"Contenu incompatible avec l'extension {ext} (détecté : {detected_mime}).",
        )

    safe_name = Path(file.filename or "").name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Nom de fichier invalide.")
    dest = Path("data/raw") / safe_name
    if not str(dest.resolve()).startswith(str(Path("data/raw").resolve()) + os.sep):
        raise HTTPException(status_code=400, detail="Nom de fichier non autorisé.")

    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        docs, ingested_hashes = load_documents("data/raw")
        docs = [d for d in docs if d.metadata.get("source_file") == safe_name]
        docs = parse_documents(docs)
        chunks = chunk_documents(docs)
        added = embed_documents(chunks)
        save_ingested_hashes(ingested_hashes)

        return {"filename": safe_name, "chunks_indexed": added, "status": "success"}

    except Exception as e:
        logger.error(f"❌ Erreur /api/upload : {e}")
        raise HTTPException(status_code=500, detail=str(e))
