"""
EcoSort AI — RAG Engine Module
Retrieval-Augmented Generation engine using ChromaDB and sentence-transformers
to provide context-aware disposal recommendations from the knowledge base.
"""

import os
import hashlib
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"
CHROMA_PERSIST_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "ecosort_waste_guidance"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Map filenames (without extension) → canonical category names
_FILENAME_TO_CATEGORY = {
    "plastic": "Plastic",
    "paper": "Paper",
    "glass": "Glass",
    "metal": "Metal",
    "organic": "Organic",
    "ewaste": "E-Waste",
}


# ---------------------------------------------------------------------------
# RAGEngine
# ---------------------------------------------------------------------------
class RAGEngine:
    """Local RAG engine backed by ChromaDB + sentence-transformers.

    Usage::

        rag = RAGEngine()
        rag.index_knowledge_base()           # safe to call every startup
        results = rag.retrieve("Plastic")    # returns list of docs
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        knowledge_dir: str | Path | None = None,
    ):
        self._persist_dir = Path(persist_dir) if persist_dir else CHROMA_PERSIST_DIR
        self._knowledge_dir = Path(knowledge_dir) if knowledge_dir else KNOWLEDGE_BASE_DIR

        # Embedding function (runs locally, free)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
        )

        # Initialise ChromaDB with persistent storage
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_knowledge_base(self) -> dict:
        """Read all markdown files from the knowledge base and index them.

        Uses a content hash as the document ID so that:
        - Unchanged documents are NOT re-inserted (dedup on restart).
        - Updated documents get a new ID and are added.

        Returns a summary dict: {added: int, skipped: int, total: int}.
        """
        md_files = sorted(self._knowledge_dir.glob("*.md"))
        if not md_files:
            raise FileNotFoundError(
                f"No .md files found in {self._knowledge_dir}"
            )

        added = 0
        skipped = 0

        for md_path in md_files:
            content = md_path.read_text(encoding="utf-8")
            stem = md_path.stem.lower()
            category = _FILENAME_TO_CATEGORY.get(stem, stem.title())

            # Deterministic ID based on content hash — avoids duplicates
            doc_id = f"{stem}_{hashlib.sha256(content.encode()).hexdigest()[:12]}"

            # Check if this exact document is already indexed
            existing = self._collection.get(ids=[doc_id])
            if existing and existing["ids"]:
                skipped += 1
                continue

            self._collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[{
                    "category": category,
                    "source_file": md_path.name,
                }],
            )
            added += 1

        return {"added": added, "skipped": skipped, "total": len(md_files)}

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        n_results: int = 1,
        category_filter: str | None = None,
    ) -> list[dict]:
        """Retrieve the most relevant knowledge-base documents.

        Args:
            query:           Free-text query (e.g. a waste category name,
                             an item description, or a disposal question).
            n_results:       Number of results to return.
            category_filter: If set, restrict results to this category.

        Returns:
            A list of dicts, each with keys:
            ``document``, ``category``, ``source_file``, ``distance``.
        """
        kwargs: dict = {
            "query_texts": [query],
            "n_results": min(n_results, self._collection.count() or 1),
        }

        if category_filter:
            kwargs["where"] = {"category": category_filter}

        results = self._collection.query(**kwargs)

        out: list[dict] = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            out.append({
                "document": results["documents"][0][i],
                "category": meta.get("category", "Unknown"),
                "source_file": meta.get("source_file", ""),
                "distance": results["distances"][0][i] if results["distances"] else None,
            })

        return out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def get_disposal_guidance(self, category: str) -> str | None:
        """Given a waste category, return the most relevant guidance text.

        This is the main integration point for the waste analyzer.
        Returns the full document text, or None if nothing is found.
        """
        results = self.retrieve(
            query=f"{category} waste disposal recycling guidance",
            n_results=1,
            category_filter=category,
        )
        return results[0]["document"] if results else None

    @property
    def document_count(self) -> int:
        """Number of documents currently in the collection."""
        return self._collection.count()
