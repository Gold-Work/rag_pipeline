import os
import sys
import time
import shutil
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from langfuse import Langfuse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.utils.logger import get_logger
from src.utils.auth import create_access_token, verify_token
from src.query.retriever import retrieve
from src.query.reranker import rerank
from src.query.augmenter import build_prompt
from src.query.generator import generate
from src.ingestion.loader import load_documents
from src.ingestion.parser import parse_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_documents

logger = get_logger("api")

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="RAG API", description="Production-ready RAG service", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
def shutdown():
    langfuse.flush()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    latencies: dict
    chunks_used: int

class HealthResponse(BaseModel):
    status: str
    version: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str


@app.post("/api/auth/token", response_model=TokenResponse)
@limiter.limit("10/minute")
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    username = os.getenv("API_USERNAME", "admin")
    password = os.getenv("API_PASSWORD", "changeme123")

    if form_data.username != username or form_data.password != password:
        raise HTTPException(status_code=401, detail="Identifiants invalides")

    token = create_access_token(data={"sub": form_data.username})
    logger.info(f"✅ Login réussi : {form_data.username}")
    return TokenResponse(access_token=token, token_type="bearer")


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/api/query", response_model=QueryResponse)
@limiter.limit("30/minute")
def query(request: Request, body: QueryRequest, username: str = Depends(verify_token)):
    logger.info(f"❓ /api/query — '{body.question}' — user: {username}")
    latencies = {}

    trace = langfuse.trace(
        name="rag-query",
        input={"question": body.question},
        user_id=username
    )

    try:
        t0 = time.time()
        span_retrieval = trace.span(name="retrieval", input={"query": body.question})
        documents = retrieve(body.question, k=20)
        latencies["retrieval"] = round(time.time() - t0, 3)
        span_retrieval.end(output={"chunks_found": len(documents)})

        if not documents:
            raise HTTPException(status_code=404, detail="Aucun document pertinent trouvé.")

        t0 = time.time()
        span_rerank = trace.span(name="reranking", input={"chunks_in": len(documents)})
        documents = rerank(body.question, documents, top_k=body.top_k)
        latencies["rerank"] = round(time.time() - t0, 3)
        span_rerank.end(output={"chunks_out": len(documents)})

        sources = list(dict.fromkeys([
            doc.metadata.get("source_file", "?") for doc in documents
        ]))

        t0 = time.time()
        prompt = build_prompt(body.question, documents)
        span_llm = trace.generation(name="generation", input=prompt)
        answer = generate(prompt)
        latencies["llm"] = round(time.time() - t0, 3)
        span_llm.end(output=answer)

        latencies["total"] = round(sum(latencies.values()), 3)

        trace.update(output={"answer": answer, "latencies": latencies})
        langfuse.flush()

        logger.info(f"✅ Réponse générée en {latencies['total']}s")

        return QueryResponse(
            answer=answer,
            sources=sources,
            latencies=latencies,
            chunks_used=len(documents)
        )

    except HTTPException:
        raise
    except Exception as e:
        trace.update(output={"error": str(e)})
        langfuse.flush()
        logger.error(f"❌ Erreur /api/query : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
@limiter.limit("5/minute")
async def upload(request: Request, file: UploadFile = File(...), username: str = Depends(verify_token)):
    logger.info(f"📤 /api/upload — '{file.filename}' — user: {username}")

    allowed = {".pdf", ".html", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Format non supporté : {ext}")

    try:
        dest = Path("data/raw") / file.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        docs = load_documents("data/raw")
        docs = [d for d in docs if d.metadata.get("source_file") == file.filename]
        docs = parse_documents(docs)
        chunks = chunk_documents(docs)
        added = embed_documents(chunks)

        return {"filename": file.filename, "chunks_indexed": added, "status": "success"}

    except Exception as e:
        logger.error(f"❌ Erreur /api/upload : {e}")
        raise HTTPException(status_code=500, detail=str(e))
