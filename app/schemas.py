from pydantic import BaseModel, Field
from typing import List, Optional


class AskRequest(BaseModel):
    """
    Petición enviada desde el frontend.

    question:
      - Pregunta del usuario.
      - Puede ser también una petición de resumen ("resume el capítulo 3").
    """
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