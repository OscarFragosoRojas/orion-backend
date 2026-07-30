from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List

from . import models, schemas
from .database import engine, get_db
from .indexer import build_vectorstore_from_base64
from .rag import RagService

# Crear las tablas de BD
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Proof Orion Backend",
    description="A simple backend API",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG global
rag_service = RagService()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Proof Orion Backend API!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/projects", response_model=schemas.ProjectResponse)
def create_project(project: schemas.ProjectCreate, db: Session = Depends(get_db)):
    db_project = models.Project(name=project.name)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

@app.get("/projects", response_model=List[schemas.ProjectResponse])
def get_projects(db: Session = Depends(get_db)):
    return db.query(models.Project).all()

@app.post("/ingest-pdf")
async def ingest_pdf(payload: schemas.PDFPayload, db: Session = Depends(get_db)):
    # 1. Verificar proyecto
    project = db.query(models.Project).filter(models.Project.id == payload.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # 2. Crear documento en BD (procesando)
    db_doc = models.Document(project_id=payload.project_id, filename=payload.filename)
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    try:
        # 3. Construir vector store
        vs = build_vectorstore_from_base64(
            pdf_base64=payload.pdf_base64,
            project_id=payload.project_id,
            document_id=db_doc.id,
            clear_previous=payload.clear_previous
        )
        
        # 4. Obtener resumen de este documento en específico
        # Para esto, filtramos temporalmente por este document_id en el RagService (o project_id si cleared)
        summary_answer, _ = rag_service.answer(
            "Por favor, haz un resumen general de los temas principales y puntos clave de este documento.",
            project_id=payload.project_id
        )

        # 5. Guardar resumen
        db_doc.summary = summary_answer
        db.commit()
        
        return {
            "status": "success",
            "message": "PDF procesado correctamente",
            "document_id": db_doc.id,
            "summary": summary_answer
        }
    except Exception as e:
        db.delete(db_doc)
        db.commit()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ask", response_model=schemas.AskResponse)
async def ask_question(req: schemas.AskRequest):
    try:
        answer, sources = rag_service.answer(req.question, project_id=req.project_id)
        return schemas.AskResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))