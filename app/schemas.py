from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ProjectCreate(BaseModel):
    name: str

class ProjectResponse(BaseModel):
    id: int
    name: str
    created_at: datetime

    class Config:
        from_attributes = True

class DocumentResponse(BaseModel):
    id: int
    project_id: int
    filename: str
    summary: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class AskRequest(BaseModel):
    """
    Petición enviada desde el frontend.

    question:
      - Pregunta del usuario.
      - Puede ser también una petición de resumen ("resume el capítulo 3").
    """
    project_id: int
    question: str = Field(..., min_length=1, max_length=4000)

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