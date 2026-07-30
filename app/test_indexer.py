from app.indexer import get_vectorstore

if __name__ == "__main__":
    vs = get_vectorstore()

    if vs is None:
        print("No hay ningún índice creado todavía. Por favor corre el servidor y sube un PDF primero.")
    else:
        query = "¿Cómo se crean equipos de alto rendimiento?"
        docs = vs.similarity_search(query, k=5)

        print(f"Consulta: {query}\n")
        for i, d in enumerate(docs, 1):
            page = d.metadata.get("page", "?")
            snippet = d.page_content[:300].replace("\n", " ")
            print(f"[{i}] pág. {page} -> {snippet}...")