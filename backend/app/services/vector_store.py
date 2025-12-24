"""
Vector store service using pgvector.

TODO: Implement this service to:
1. Generate embeddings for text chunks
2. Store embeddings in PostgreSQL with pgvector
3. Perform similarity search
4. Link related images and tables
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.document import DocumentChunk, DocumentImage, DocumentTable
from app.core.config import settings


class VectorStore:
    """
    Vector store for document embeddings and similarity search.
    
    This is a SKELETON implementation. You need to implement the core logic.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.embeddings_model = None  # TODO: Initialize embedding model
        self._ensure_extension()
    
    def _ensure_extension(self):
        """
        Ensure pgvector extension is enabled.
        
        This is implemented as an example.
        """
        try:
            self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            self.db.commit()
        except Exception as e:
            print(f"pgvector extension already exists or error: {e}")
            self.db.rollback()
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using sentence-transformers.
        """
        from sentence_transformers import SentenceTransformer
        
        if not hasattr(self, '_model'):
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
        
        return self._model.encode(text, convert_to_numpy=True)
    
    async def store_chunk(
        self, 
        content: str, 
        document_id: int,
        page_number: int,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentChunk:
        """
        Store a text chunk with its embedding.
        """
        from datetime import datetime
        
        # Generate embedding
        embedding = await self.generate_embedding(content)
        
        # Create chunk record
        chunk = DocumentChunk(
            document_id=document_id,
            content=content,
            embedding=embedding.tolist(),
            page_number=page_number,
            chunk_index=chunk_index,
            metadata=metadata or {},
            created_at=datetime.now()
        )
        
        self.db.add(chunk)
        self.db.commit()
        self.db.refresh(chunk)
        
        return chunk
    
    async def similarity_search(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        """
        # Generate query embedding
        query_embedding = await self.generate_embedding(query)
        query_str ="[" + ",".join(map(str, query_embedding.tolist())) + "]"
        # Build query
        base_query = """
        SELECT 
            id,
            content,
            page_number,
            chunk_metadata,
            chunk_index,
            document_id,
            1 - (embedding <=> :query_embedding ::vector) as similarity
        FROM document_chunks
        """
        
        params = {
            "query_embedding": query_str,
            "k": k
}
        
        if document_id is not None:
            base_query += " WHERE document_id = :document_id"
            params["document_id"] = document_id
        
        base_query += " ORDER BY embedding <=> :query_embedding ::vector LIMIT :k"
        #params["k"] = k
        
        # Execute query
        result = self.db.execute(text(base_query), params)
        rows = result.fetchall()
        
        # Format results
        results = []
        for row in rows:
            chunk_data = {
                "id": row.id,
                "content": row.content,
                "score": float(row.similarity) if row.similarity is not None else None,
                "page_number": row.page_number,
                "chunk_index": row.chunk_index,
                "document_id": row.document_id,
                "chunk_metadata": row.chunk_metadata or {}
            }
            results.append(chunk_data)
        
        return results
    
    async def get_related_content(
        self,
        chunk_ids: List[int]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get related images and tables for given chunks.
        """
        if not chunk_ids:
            return {"images": [], "tables": []}
        
        # Get chunks to find document_id and page numbers
        chunks_query = self.db.query(DocumentChunk).filter(DocumentChunk.id.in_(chunk_ids))
        chunks = chunks_query.all()
        
        if not chunks:
            return {"images": [], "tables": []}
        
        # Get document IDs and page numbers
        document_ids = list(set(chunk.document_id for chunk in chunks))
        page_numbers = list(set(chunk.page_number for chunk in chunks if chunk.page_number))
        print(f"Document IDs: {document_ids}, Page Numbers: {page_numbers}")
        # Query related images
        images_query = self.db.query(DocumentImage).filter(
            DocumentImage.document_id.in_(document_ids)
        )
        if page_numbers:
            images_query = images_query.filter(DocumentImage.page_number.in_(page_numbers))
        
        images = images_query.all()
        images_data = [
            {
                "id": img.id,
                "url": img.file_path,
                "page_number": img.page_number,
                "caption": img.caption,
                "width": img.width,
                "height": img.height,
                "metadata": img.image_metadata
            }
            for img in images
        ]
        
        # Query related tables
        tables_query = self.db.query(DocumentTable).filter(
            DocumentTable.document_id.in_(document_ids)
        )
        if page_numbers:
            tables_query = tables_query.filter(DocumentTable.page_number.in_(page_numbers))
        
        tables = tables_query.all()
        tables_data = [
            {
                "id": tbl.id,
                "url": tbl.image_path,
                "page_number": tbl.page_number,
                "caption": tbl.caption,
                "rows": tbl.rows,
                "columns": tbl.columns,
                "data": tbl.data,
                "metadata": tbl.table_metadata
            }
            for tbl in tables
        ]
        
        return {
            "images": images_data,
            "tables": tables_data
        }
