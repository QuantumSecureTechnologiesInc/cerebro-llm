"""RAG (Retrieval-Augmented Generation) pipeline for Cerebro.

Provides document-grounded generation with:
- Document ingestion and chunking
- Embedding-based retrieval
- Context injection into prompts
- Source attribution in responses
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    """A document in the knowledge base."""
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    source: str = ""
    chunks: list[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    """A retrieved chunk with relevance score."""
    content: str
    document_id: str
    score: float
    source: str = ""
    chunk_index: int = 0


class DocumentChunker:
    """Splits documents into retrieval-sized chunks.

    Supports:
    - Fixed-size chunking with overlap
    - Sentence-boundary aware chunking
    - Markdown heading-aware chunking

    Args:
        chunk_size: Target chunk size in characters.
        overlap: Overlap between chunks in characters.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        Uses sentence boundaries when possible.

        Args:
            text: Input text to chunk.

        Returns:
            List of text chunks.
        """
        if len(text) <= self.chunk_size:
            return [text]

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Keep overlap from end of current chunk
                    current_chunk = current_chunk[-self.overlap:] + " " + sentence
                else:
                    chunks.append(sentence[:self.chunk_size])
                    current_chunk = sentence[self.chunk_size - self.overlap:]
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_document(self, doc: Document) -> Document:
        """Chunk a document and store chunks.

        Args:
            doc: Document to chunk.

        Returns:
            Document with chunks populated.
        """
        doc.chunks = self.chunk_text(doc.content)
        return doc


class VectorStore:
    """Simple in-memory vector store for retrieval.

    Uses TF-IDF-like similarity for text matching.
    For production, use EmbeddingVectorStore with FAISS/ChromaDB.

    Args:
        similarity_threshold: Minimum similarity score.
    """

    def __init__(self, similarity_threshold: float = 0.1) -> None:
        self.similarity_threshold = similarity_threshold
        self._chunks: list[RetrievedChunk] = []
        self._documents: dict[str, Document] = {}

    def add_document(self, doc: Document, chunker: DocumentChunker | None = None) -> None:
        """Add a document to the store.

        Args:
            doc: Document to add.
            chunker: Optional chunker (uses default if None).
        """
        if chunker is None:
            chunker = DocumentChunker()

        doc = chunker.chunk_document(doc)
        self._documents[doc.id] = doc

        for i, chunk_text in enumerate(doc.chunks):
            self._chunks.append(RetrievedChunk(
                content=chunk_text,
                document_id=doc.id,
                score=0.0,
                source=doc.source,
                chunk_index=i,
            ))

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Search for relevant chunks.

        Uses TF-IDF-like cosine similarity.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            List of RetrievedChunk sorted by relevance.
        """
        query_terms = set(query.lower().split())
        scored = []

        for chunk in self._chunks:
            chunk_terms = set(chunk.content.lower().split())
            overlap = query_terms & chunk_terms
            if overlap:
                score = len(overlap) / max(len(query_terms), 1)
                if score >= self.similarity_threshold:
                    scored.append(RetrievedChunk(
                        content=chunk.content,
                        document_id=chunk.document_id,
                        score=score,
                        source=chunk.source,
                        chunk_index=chunk.chunk_index,
                    ))

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        self._chunks.clear()
        self._documents.clear()

    @property
    def num_documents(self) -> int:
        return len(self._documents)

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)


class EmbeddingVectorStore:
    """Vector store using real embeddings (FAISS, ChromaDB, or numpy).

    Supports multiple backends:
    - numpy: Pure numpy cosine similarity (no deps, slow for large collections)
    - faiss: Facebook AI Similarity Search (fast, GPU support)
    - chromadb: ChromaDB (persistent, metadata filtering)

    Args:
        backend: "auto", "faiss", "chromadb", or "numpy".
        embedding_model: HuggingFace model name for embeddings.
        persist_dir: Directory for persistent storage (ChromaDB).
        device: Device for embedding computation.
    """

    def __init__(
        self,
        backend: str = "auto",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_dir: str | None = None,
        device: str = "auto",
    ) -> None:
        self.backend = backend
        self.embedding_model_name = embedding_model
        self.persist_dir = persist_dir
        self.device = device

        self._chunks: list[RetrievedChunk] = []
        self._documents: dict[str, Document] = {}
        self._embeddings = None
        self._index = None
        self._embedder = None
        self._chroma_collection = None
        self._initialized = False

    def _init_embedder(self) -> None:
        """Lazy-initialize the embedding model."""
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
            if self.device != "auto":
                self._embedder = self._embedder.to(self.device)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embedding-based retrieval. "
                "Install: pip install sentence-transformers"
            )

    def _init_backend(self) -> None:
        """Detect and initialize the best available backend."""
        if self._initialized:
            return

        if self.backend == "auto":
            # Try ChromaDB first (persistent), then FAISS (fast), then numpy
            try:
                import chromadb
                self.backend = "chromadb"
            except ImportError:
                try:
                    import faiss
                    self.backend = "faiss"
                except ImportError:
                    self.backend = "numpy"

        if self.backend == "chromadb":
            self._init_chromadb()
        elif self.backend == "faiss":
            self._init_faiss()

        self._initialized = True

    def _init_chromadb(self) -> None:
        """Initialize ChromaDB backend."""
        import chromadb
        from chromadb.config import Settings

        if self.persist_dir:
            client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            client = chromadb.Client(Settings(anonymized_telemetry=False))

        self._chroma_collection = client.get_or_create_collection(
            name="cerebro_rag",
            metadata={"hnsw:space": "cosine"},
        )

    def _init_faiss(self) -> None:
        """Pre-allocate FAISS index (built lazily on first search)."""
        self._index = None
        self._embeddings = None

    def _compute_embeddings(self, texts: list[str]) -> "np.ndarray":
        """Compute embeddings for a list of texts."""
        self._init_embedder()
        import numpy as np
        embeddings = self._embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def add_document(self, doc: Document, chunker: DocumentChunker | None = None) -> None:
        """Add a document with embedding-based indexing.

        Args:
            doc: Document to add.
            chunker: Optional chunker.
        """
        if chunker is None:
            chunker = DocumentChunker()

        doc = chunker.chunk_document(doc)
        self._documents[doc.id] = doc

        for i, chunk_text in enumerate(doc.chunks):
            chunk = RetrievedChunk(
                content=chunk_text,
                document_id=doc.id,
                score=0.0,
                source=doc.source,
                chunk_index=i,
            )
            self._chunks.append(chunk)

        # Compute embeddings for new chunks
        self._index_chunks(doc.chunks)

    def _index_chunks(self, texts: list[str]) -> None:
        """Index new chunks in the backend."""
        self._init_backend()

        if not texts:
            return

        embeddings = self._compute_embeddings(texts)

        if self.backend == "chromadb" and self._chroma_collection:
            start_idx = len(self._chunks) - len(texts)
            ids = [f"chunk_{start_idx + j}" for j in range(len(texts))]
            metadatas = [
                {
                    "doc_id": self._chunks[start_idx + j].document_id,
                    "source": self._chunks[start_idx + j].source or "",
                }
                for j in range(len(texts))
            ]
            self._chroma_collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
            )
        else:
            # FAISS or numpy: accumulate embeddings
            import numpy as np
            if self._embeddings is None:
                self._embeddings = embeddings
            else:
                self._embeddings = np.vstack([self._embeddings, embeddings])
            self._index = None  # Rebuild FAISS index on next search

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Search using semantic embeddings.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            List of RetrievedChunk sorted by relevance.
        """
        self._init_backend()

        if not self._chunks:
            return []

        query_embedding = self._compute_embeddings([query])

        if self.backend == "chromadb" and self._chroma_collection:
            results = self._chroma_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(top_k, len(self._chunks)),
            )
            scored = []
            for i, doc_id in enumerate(results["ids"][0]):
                chunk_idx = int(doc_id.split("_")[1])
                if chunk_idx < len(self._chunks):
                    chunk = self._chunks[chunk_idx]
                    scored.append(RetrievedChunk(
                        content=chunk.content,
                        document_id=chunk.document_id,
                        score=1.0 - results["distances"][0][i],  # convert distance to similarity
                        source=chunk.source,
                        chunk_index=chunk.chunk_index,
                    ))
            return scored

        # FAISS or numpy backend
        import numpy as np

        if self._embeddings is None:
            return []

        if self.backend == "faiss":
            try:
                import faiss
                if self._index is None:
                    dim = self._embeddings.shape[1]
                    self._index = faiss.IndexFlatIP(dim)  # inner product = cosine for normalized
                    faiss.normalize_L2(self._embeddings)
                    self._index.add(self._embeddings)

                query_norm = query_embedding.copy()
                faiss.normalize_L2(query_norm)
                scores, indices = self._index.search(query_norm, min(top_k, len(self._chunks)))

                scored = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or idx >= len(self._chunks):
                        continue
                    chunk = self._chunks[idx]
                    scored.append(RetrievedChunk(
                        content=chunk.content,
                        document_id=chunk.document_id,
                        score=float(score),
                        source=chunk.source,
                        chunk_index=chunk.chunk_index,
                    ))
                return scored
            except ImportError:
                pass

        # Numpy fallback: cosine similarity
        emb_norm = self._embeddings / (np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8)
        query_norm = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8)
        scores = np.dot(query_norm, emb_norm.T)[0]

        top_indices = np.argsort(scores)[::-1][:min(top_k, len(scores))]

        scored = []
        for idx in top_indices:
            chunk = self._chunks[idx]
            scored.append(RetrievedChunk(
                content=chunk.content,
                document_id=chunk.document_id,
                score=float(scores[idx]),
                source=chunk.source,
                chunk_index=chunk.chunk_index,
            ))
        return scored

    def clear(self) -> None:
        self._chunks.clear()
        self._documents.clear()
        self._embeddings = None
        self._index = None
        if self._chroma_collection:
            try:
                self._chroma_collection.delete(where={})
            except (RuntimeError, ValueError, AttributeError):
                import logging
                logging.getLogger("cerebro.rag").debug("Chroma collection clear failed", exc_info=True)

    @property
    def num_documents(self) -> int:
        return len(self._documents)

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline.

    Combines document retrieval with model generation:
    1. User query → retrieve relevant chunks
    2. Inject chunks as context into prompt
    3. Generate grounded response
    4. Include source citations

    Args:
        vector_store: Vector store for retrieval.
        engine: Inference engine for generation.
        tokenizer: Cerebro tokenizer.
        top_k: Number of chunks to retrieve.
    """

    def __init__(
        self,
        vector_store: VectorStore | EmbeddingVectorStore | None = None,
        engine=None,
        tokenizer=None,
        top_k: int = 5,
    ) -> None:
        self.store = vector_store or VectorStore()
        self.engine = engine
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.chunker = DocumentChunker()

    def add_document(self, content: str, source: str = "", metadata: dict | None = None) -> str:
        """Add a document to the knowledge base.

        Args:
            content: Document text content.
            source: Source identifier (filename, URL, etc).
            metadata: Optional metadata dict.

        Returns:
            Document ID.
        """
        doc_id = hashlib.md5(content[:1000].encode()).hexdigest()[:12]
        doc = Document(
            id=doc_id,
            content=content,
            metadata=metadata or {},
            source=source,
        )
        self.store.add_document(doc, self.chunker)
        return doc_id

    def add_file(self, file_path: str) -> str:
        """Add a file to the knowledge base.

        Args:
            file_path: Path to text file.

        Returns:
            Document ID.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.add_document(content, source=os.path.basename(file_path))

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Search query.

        Returns:
            List of retrieved chunks.
        """
        return self.store.search(query, top_k=self.top_k)

    def build_context_prompt(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """Build a RAG-augmented prompt.

        Args:
            query: User query.
            chunks: Retrieved chunks.

        Returns:
            Formatted prompt with context.
        """
        if not chunks:
            return query

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = f" (Source: {chunk.source})" if chunk.source else ""
            context_parts.append(f"[Context {i}]{source}\n{chunk.content}")

        context = "\n\n".join(context_parts)
        return (
            f"Use the following context to answer the question.\n"
            f"Cite sources using [1], [2], etc.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

    def generate_answer(self, query: str) -> dict:
        """Generate a grounded answer with citations.

        Args:
            query: User query.

        Returns:
            Dict with answer, sources, and retrieved chunks.
        """
        chunks = self.retrieve(query)
        prompt = self.build_context_prompt(query, chunks)

        if self.engine and self.tokenizer:
            import torch
            tokens = self.tokenizer.encode(prompt, add_bos=True)
            input_ids = torch.tensor([tokens], dtype=torch.long)
            generated = self.engine.generate(input_ids, max_new_tokens=512)
            output_tokens = generated[0].tolist()
            answer = self.tokenizer.decode(output_tokens[len(tokens):], skip_special=True)
        else:
            answer = f"[RAG response based on {len(chunks)} retrieved chunks]"

        sources = list(set(c.source for c in chunks if c.source))
        return {
            "answer": answer,
            "sources": sources,
            "num_chunks": len(chunks),
            "query": query,
        }
