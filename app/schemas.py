from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# ---------------------------------------------------------------------------
# Team Members
# ---------------------------------------------------------------------------

class TeamMemberCreate(BaseModel):
    name: str
    role: str  # developer, pm, designer, qa, devops

class TeamMemberResponse(BaseModel):
    id: int
    name: str
    role: str

    class Config:
        from_attributes = True

# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

class ProjectCreate(BaseModel):
    name: str
    client: str
    description: str
    project_type: str
    start_date: datetime
    end_date: datetime
    team_members: List[TeamMemberCreate] = []

class ProjectResponse(BaseModel):
    id: int
    name: str
    client: str
    description: str
    project_type: str
    start_date: datetime
    end_date: datetime
    progress: int  # 0-100
    created_at: datetime
    team_members: List[TeamMemberResponse] = []

    class Config:
        from_attributes = True

# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentResponse(BaseModel):
    id: int
    project_id: int
    filename: str
    summary: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """
    Petición enviada desde el frontend.

    question:
      - Pregunta del usuario.
      - Puede ser también una petición de resumen ("resume el capítulo 3").
    """
    project_id: int
    question: str = Field(..., min_length=1, max_length=4000)
    document_id: Optional[int] = None

class SourceChunk(BaseModel):
    """
    Fragmento de fuente utilizado para responder.

    Se devuelve de manera explícita para que la app no sea una "caja negra".
    """
    page: Optional[int] = None
    snippet: str

class AskResponse(BaseModel):
    """
    Respuesta final del sistema.
    """
    answer: str
    sources: List[SourceChunk] = []

class PDFPayload(BaseModel):
    project_id: int
    pdf_base64: str
    filename: Optional[str] = "documento.pdf"
    clear_previous: Optional[bool] = False