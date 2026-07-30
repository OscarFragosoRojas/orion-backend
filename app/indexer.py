from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


INDEX_DIR = Path(__file__).resolve().parents[1] / "storage" / "faiss_index"


def build_vectorstore_from_base64(
    pdf_base64: str,
    *,
    embedding_model: str = "mxbai-embed-large",
    save_to_disk: bool = True,
) -> FAISS:
    """
    Recibe un string en Base64 que representa un PDF, lo procesa y genera el vectorstore.

    pdf_base64: Cadena en Base64 (puede o no incluir el prefijo data:application/pdf;base64,)
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
        temp_file.flush()  # Asegurar que todos los bytes se escriban en disco

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

    # 5. Crear o actualizar el VectorStore
    if save_to_disk and INDEX_DIR.exists() and (INDEX_DIR / "index.faiss").exists():
        # Si ya existe, lo cargamos y añadimos los nuevos documentos
        vectorstore = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(chunks)
    else:
        # Si no existe, creamos uno nuevo desde cero
        vectorstore = FAISS.from_documents(chunks, embeddings)

    # 6. Opcional: Persistir el índice si se requiere
    if save_to_disk:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(INDEX_DIR))

    return vectorstore

def get_vectorstore(embedding_model: str = "mxbai-embed-large") -> Optional[FAISS]:
    """Carga el vectorstore desde disco si existe."""
    if INDEX_DIR.exists() and (INDEX_DIR / "index.faiss").exists():
        embeddings = OllamaEmbeddings(model=embedding_model)
        return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    return None