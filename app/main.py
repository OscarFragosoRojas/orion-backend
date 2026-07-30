from app.indexer import build_vectorstore_from_base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Proof Orion Backend",
    description="A simple backend API",
    version="0.1.0"
)

# Set up CORS if you are going to communicate with a frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Proof Orion Backend API!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

class PDFPayload(BaseModel):
    pdf_base64: str
    filename: Optional[str] = "documento.pdf"


from app.rag import RagService
from app.schemas import AskRequest, AskResponse

# Instanciamos el servicio RAG globalmente.
# Al iniciar, cargará el índice si existe.
rag_service = RagService()

@app.post("/ingest-pdf")
async def ingest_pdf(payload: PDFPayload):
    global rag_service
    try:
        vs = build_vectorstore_from_base64(
            pdf_base64=payload.pdf_base64,
            save_to_disk=True
        )
        # Recargar el servicio RAG para que tome el nuevo índice actualizado
        rag_service = RagService()
        
        # Pedirle automáticamente el resumen al RAG
        summary_answer, _ = rag_service.answer("Por favor, haz un resumen general de los temas principales y puntos clave de este documento.")
        
        return {
            "status": "success",
            "message": "PDF procesado correctamente",
            "total_chunks": vs.index.ntotal,
            "summary": summary_answer
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    try:
        answer, sources = rag_service.answer(req.question)
        return AskResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------
# 
#           Para crear el entorno virtual
# 
# ------------------------------------------------------------------------------

#python3 -m venv .venv
#source .venv/bin/activate

# ------------------------------------------------------------------------------
# 
#           Para instalar las dependencias
# 
# ------------------------------------------------------------------------------

#pip install fastapi uvicorn