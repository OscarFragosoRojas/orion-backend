from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

from .database import SQLALCHEMY_DATABASE_URL

def build_vectorstore_from_base64(
    pdf_base64: str,
    project_id: int,
    document_id: int,
    *,
    embedding_model: str = "mxbai-embed-large",
    clear_previous: bool = False,
) -> PGVector:
    """
    Recibe un string en Base64 que representa un PDF, lo procesa y genera el vectorstore en Postgres.
    """
    # 1. Limpiar el prefijo 'data:application/pdf;base64,' si el frontend lo envía
    if "," in pdf_base64:
        pdf_base64 = pdf_base64.split(",")[1]

    # 2. Decodificar la cadena Base64 a bytes de PDF
    try:
        pdf_bytes = base64.b64decode(pdf_base64)
    except Exception as e:
        raise ValueError(f"Error al decodificar el string Base64: {e}")

    embeddings = OllamaEmbeddings(model=embedding_model)

    # 3. Guardar en un archivo temporal para que PyPDFLoader pueda leerlo
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_file:
        temp_file.write(pdf_bytes)
        temp_file.flush()

        # Cargar los documentos desde el archivo temporal
        loader = PyPDFLoader(temp_file.name)
        docs = loader.load()

    # 4. Dividir en chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Inyectar project_id y document_id como metadata
    for chunk in chunks:
        chunk.metadata["project_id"] = project_id
        chunk.metadata["document_id"] = document_id

    # 5. Conectar a PGVector
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="orion_docs",
        connection=SQLALCHEMY_DATABASE_URL,
        use_jsonb=True,
    )

    if clear_previous:
        # Aquí idealmente borraríamos solo los del project_id, 
        # pero la API de PGVector no soporta delete por metadata fácilmente desde aquí.
        # Por simplicidad en este MVP, borramos la colección entera si piden clear_previous
        vectorstore.drop_tables()
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name="orion_docs",
            connection=SQLALCHEMY_DATABASE_URL,
            use_jsonb=True,
        )

    # Añadir los chunks a postgres
    vectorstore.add_documents(chunks)

    return vectorstore

def get_vectorstore(embedding_model: str = "mxbai-embed-large") -> PGVector:
    """Retorna la instancia del vectorstore de Postgres."""
    embeddings = OllamaEmbeddings(model=embedding_model)
    return PGVector(
        embeddings=embeddings,
        collection_name="orion_docs",
        connection=SQLALCHEMY_DATABASE_URL,
        use_jsonb=True,
    )