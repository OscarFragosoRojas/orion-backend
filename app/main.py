from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List

from . import models, schemas
from .database import engine, get_db
from .indexer import build_vectorstore_from_base64
from .rag import RagService
from .storage import upload_pdf_from_base64
from .models import DUMMY_TEAM_MEMBERS

# ---------------------------------------------------------------------------
# Recrear tablas y hacer seed de datos dummy
# ---------------------------------------------------------------------------
models.Base.metadata.drop_all(bind=engine)
models.Base.metadata.create_all(bind=engine)

def seed_team_members(db: Session) -> None:
    """Inserta miembros dummy si la tabla está vacía."""
    if db.query(models.TeamMember).count() == 0:
        db.add_all([
            models.TeamMember(name=m["name"], role=m["role"], project_id=None)
            for m in DUMMY_TEAM_MEMBERS
        ])
        db.commit()

# Seed al arrancar
_db_seed = next(get_db())
try:
    seed_team_members(_db_seed)
except Exception:
    _db_seed.rollback()
finally:
    _db_seed.close()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "Welcome to the Proof Orion Backend API!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# Team Members
# ---------------------------------------------------------------------------

@app.get("/team-members", response_model=List[schemas.TeamMemberResponse])
def get_team_members(db: Session = Depends(get_db)):
    """Lista todos los miembros disponibles (con y sin proyecto asignado)."""
    return db.query(models.TeamMember).all()

# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

@app.post("/projects", response_model=schemas.ProjectResponse)
def create_project(project: schemas.ProjectCreate, db: Session = Depends(get_db)):
    db_project = models.Project(
        name=project.name,
        client=project.client,
        description=project.description,
        project_type=project.project_type,
        start_date=project.start_date,
        end_date=project.end_date,
    )
    db.add(db_project)
    db.flush()  # obtener el ID antes de crear los miembros

    for member in project.team_members:
        db_member = models.TeamMember(
            project_id=db_project.id,
            name=member.name,
            role=member.role,
        )
        db.add(db_member)

    db.commit()
    db.refresh(db_project)
    return db_project

@app.get("/projects", response_model=List[schemas.ProjectResponse])
def get_projects(db: Session = Depends(get_db)):
    return db.query(models.Project).all()

# ---------------------------------------------------------------------------
# Documents / Ingest PDF
# ---------------------------------------------------------------------------

@app.post("/ingest-pdf")
async def ingest_pdf(payload: schemas.PDFPayload, db: Session = Depends(get_db)):
    # 1. Verificar proyecto
    project = db.query(models.Project).filter(models.Project.id == payload.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # 2. Subir PDF a Cloudfare R2
    try:
        gcs_uri, public_url = upload_pdf_from_base64(
            pdf_base64=payload.pdf_base64,
            project_id=payload.project_id,
            filename=payload.filename or "documento.pdf",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al subir a Archivo: {e}")

    # 3. Crear documento en BD con la ruta del archivo
    db_doc = models.Document(
        project_id=payload.project_id,
        filename=payload.filename,
        file_path=gcs_uri,
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    try:
        # 4. Construir vector store
        vs = build_vectorstore_from_base64(
            pdf_base64=payload.pdf_base64,
            project_id=payload.project_id,
            document_id=db_doc.id,
            clear_previous=payload.clear_previous
        )

        # 5. Obtener resumen del documento
        summary_answer, _ = rag_service.answer(
            "Por favor, haz un resumen general de los temas principales y puntos clave de este documento.",
            project_id=payload.project_id,
            document_id=db_doc.id
        )

        # 6. Guardar resumen
        db_doc.summary = summary_answer
        db.commit()

        return {
            "status": "success",
            "message": "PDF procesado correctamente",
            "document_id": db_doc.id,
            "summary": summary_answer,
            "gcs_uri": gcs_uri,
            "public_url": public_url,
        }
    except Exception as e:
        db.delete(db_doc)
        db.commit()
        raise HTTPException(status_code=400, detail=str(e))

# ---------------------------------------------------------------------------
# RAG / Ask
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=schemas.AskResponse)
async def ask_question(req: schemas.AskRequest):
    try:
        answer, sources = rag_service.answer(req.question, project_id=req.project_id)
        return schemas.AskResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Ejecutar Entorno Virtual: source .venv/bin/activate
# Ejecutar desde terminal: uvicorn main:app --reload