"""
Document processing service using Docling

TODO: Implement this service to:
1. Parse PDF documents using Docling
2. Extract text, images, and tables
3. Store extracted content in database
4. Generate embeddings for text chunks
"""
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from app.models.document import Document, DocumentChunk, DocumentImage, DocumentTable
from app.services.vector_store import VectorStore
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker   
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from datetime import datetime
from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import ImageRefMode, PictureItem, TableItem, TextItem, DoclingDocument 
from app.core.config import settings
from pathlib import Path
import os
import hashlib

load_dotenv()  # Load environment variables from .env file
class DocumentProcessor:
    """
    Process PDF documents and extract multimodal content.
    
    This is a SKELETON implementation. You need to implement the core logic.
    """
    
    MAX_TOKENS = 300
    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    #model = SentenceTransformer(model_id)
    #embedding_dim = model.get_sentence_embedding_dimension()  # 384 for MiniLM
    #upload_dir = Path(settings.UPLOAD_DIR)


    def __init__(self, db: Session):
        self.db = db
        # Embedding model
        self.model = SentenceTransformer(self.MODEL_ID)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()  # 384

        # Paths
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.vector_store = VectorStore(db)
    
    async def process_document(self, file_path: str, document_id: int) -> Dict[str, Any]:
        """
        Process a PDF document using Docling.
        
        Implementation steps:
        1. Update document status to 'processing'
        2. Use Docling to parse the PDF
        3. Extract and save text chunks
        4. Extract and save images
        5. Extract and save tables
        6. Generate embeddings for text chunks
        7. Update document status to 'completed'
        8. Handle errors appropriately
        
        Args:
            file_path: Path to the uploaded PDF file
            document_id: Database ID of the document
            
        Returns:
            {
                "status": "success" or "error",
                "text_chunks": <count>,
                "images": <count>,
                "tables": <count>,
                "processing_time": <seconds>
            }
        """
        print("Starting document processing...", document_id, file_path)
        import time
        start_time = time.time()
        
        try:
            print("update status to processing")
            # Update status to processing
            await self._update_document_status(document_id, "processing")
            
            # Convert document with Docling
            print(f"Converting document: {file_path}")
            try:
                pdf_options = PdfPipelineOptions(
                extract_text=True,
                extract_images=True,
                extract_tables=True,
                generate_picture_images=True,
                ocr=False,              # skip OCR for digital PDFs
                table_method="lattice", # better for structured tables
                dpi=300,
                image_dpi=300,
                max_pages=None,
                parallel=True
            )
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
                }
                )       
                result = converter.convert(file_path)
               # print(f"Document converted successfully. Pages: {result.document.export_to_markdown()}")
            except Exception as convert_error:
                print(f"Error during document conversion: {convert_error}")
                raise convert_error
            total_text_chunks = 0
            total_images = 0
            total_tables = 0

            # Extract text chunks
            try:
                chunks = []
                
                #chunks = self.chunk_document_for_rag(result, document_id)
                #print("Built chunks from blocks.")
    
                # 🔹 Use Docling document object directly for chunking
                chunks = self._chunk_text(result.document, document_id, 1)
                """
                for page_idx, page in result.document.pages.items():
                    # Chunk each page using the Docling document
                    page_idx = page.page_no
                    
                    page_chunks = self._chunk_text(result.document, document_id, page_idx)
                    chunks.extend(page_chunks)
                """   

                if not chunks:
                    raise ValueError("No extractable text found in document")
                
                total_text_chunks += len(chunks)
                print(f"Total document-level chunks created: {len(chunks)}")
                
                await self._save_text_chunks(chunks, document_id)

                

            except Exception as exc:
                print(f"Error during document text extraction or chunking: {exc}")
                raise

            print (f"Extracted {total_text_chunks} text chunks.")
            
            # Extract images
            for items in result.document.iterate_items():
              
                item = items[0]
                prov = item.prov[0] if item.prov else None
                
                if isinstance(item, PictureItem):
                    try:
                        
                    
                        total_images += 1
                        dir_path = self.upload_dir / "images"
                        dir_path.mkdir(parents=True, exist_ok=True)
                        element_image_filename = dir_path / f"doc{document_id}_page{prov.page_no}_img{total_images}.png"
                        image =   item.get_image(result.document)
                        with element_image_filename.open("wb") as fp:
                            image.save(fp, "PNG")
                        image_metadata = {
                                "type": "image",
                                "document_id": document_id,
                                "page": prov.page_no if prov else None,
                                "bbox": self.bbox_to_list(prov.bbox) if prov else None,
                                #"image_bytes": item.image , # raw image bytes,
                                "file_path": str(element_image_filename),
                                "width": image.width if image else None,
                                "height": image.height if image else None,
                                "ref_id": self.make_element_id(document_id, prov.page_no, self.bbox_to_list(prov.bbox)) if prov else None
                        }
                        await self._save_image(item, document_id, prov.page_no, image_metadata)
                        
                        
                    except Exception as image_error:
                        print(f"Error saving image item: {image_error}")
                        continue
            
            # Extract tables

                if isinstance(item, TableItem):
                    try:
                        print("saving table item..." )
                        caption = item.caption if hasattr(item, "caption") else None
                        table_metadata = {
                                "type": "table",
                                "document_id": document_id,
                                "caption": caption,
                                "page": prov.page_no if prov else None,                                
                                "bbox": self.bbox_to_list(prov.bbox) if prov else None,
                                "ref_id": self.make_element_id(document_id, prov.page_no, self.bbox_to_list(prov.bbox)) if prov else None
                        }
                        await self._save_table(item, document_id, prov.page_no, table_metadata)
                        total_tables += 1
                    except Exception as table_error:
                        print(f"Error saving table item: {table_error}")
                        continue
            print(f"Extracted {total_tables} tables.")
            print(f"Extracted {total_images} images.")
            
            # Update document with counts
            document = self.db.query(Document).filter(Document.id == document_id).first()
            print("updating document counts...", document_id )
            if document:
                document.total_pages = len(result.pages)
                document.text_chunks_count = total_text_chunks
                document.images_count = total_images
                document.tables_count = total_tables
            print("updating document counts...", document.total_pages, document.text_chunks_count, document.images_count, document.tables_count )
            try:
                # Update status to completed
                await self._update_document_status(document_id, "completed")
            except Exception as status_error:
                print(f"Error updating document status to completed: {status_error}")
                raise status_error
            
            processing_time = time.time() - start_time
            print("Document processing completed.", processing_time)
            return {
                "status": "success",
                "text_chunks": total_text_chunks,
                "images": total_images,
                "tables": total_tables,
                "processing_time": round(processing_time, 2)
            }
            
        except Exception as e:
            await self._update_document_status(document_id, "failed", str(e))
            processing_time = time.time() - start_time
            return {
                "status": "error",
                "text_chunks": 0,
                "images": 0,
                "tables": 0,
                "processing_time": round(processing_time, 2),
                "error": str(e)
            }
       # raise NotImplementedError("Document processing not implemented yet")
    
    def _chunk_text(self, text: str, document_id: int, page_number: int) -> List[Dict[str, Any]]:
        """
        Split text into chunks for vector storage.
        
        TODO: Implement text chunking strategy
        - Split by sentences or paragraphs
        - Maintain context with overlap
        - Keep metadata (page number, position, etc.)
        
        Returns:
            List of chunk dictionaries with content and metadata
        """
        print("Chunking text...")
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        
        chunker = HybridChunker(
            tokenizer=tokenizer,
            model=self.model,
            max_tokens=self.MAX_TOKENS,
            merge_peers=True,
            chunk_overlap=50
        )
        
        chunk_iter = chunker.chunk(text)
        raw_chunks = list(chunk_iter)
        chunked_docs: List[Dict[str, Any]] = []
        for chunk in raw_chunks:

            prov= chunk.meta.doc_items[0].prov[0] if chunk.meta.doc_items and chunk.meta.doc_items[0].prov else None
            print(prov.page_no if prov else "no prov")
            content_type = getattr(chunk.meta, "content_type", "text")
            chunked_docs.append({
                "content": chunk.text,
                "metadata": {
                    "document_id": document_id,
                    "page_numbers": (
                        [
                            page_no
                            for page_no in sorted(
                                {
                                    prov.page_no
                                    for item in chunk.meta.doc_items
                                    for prov in item.prov
                                }
                            )
                        ]
                        or None
                    ),
                    "title": chunk.meta.headings[0] if chunk.meta.headings else None,
                    "token_count": len(
                        tokenizer.encode(chunk.text, add_special_tokens=False)
                    ),
                    "content_type": content_type,
                    "bounding_box": getattr(chunk.meta, "bbox", None), 
                    "table_info": getattr(chunk.meta, "table_info", None), # rows, cols, headers 
                    "image_info": getattr(chunk.meta, "image_info", None), # caption, OCR text 
                    "ref_id": self.make_element_id(document_id, prov.page_no, self.bbox_to_list(prov.bbox)) if prov else None
                }
            })

        return chunked_docs

    async def _save_text_chunks(self, chunks: List[Dict[str, Any]], document_id: int):
        """
        Save text chunks to database with embeddings.
        
        TODO: Implement chunk storage
        - Generate embeddings
        - Store in database
        - Link related images/tables in metadata
        """
        print("Saving text chunks with embeddings...")
        if not chunks:
            return
        texts = [chunk["content"] for chunk in chunks]
        vectorStore = VectorStore(self.db)
        embeddings = await vectorStore.generate_embedding(texts)
        #embeddings = self.model.encode(texts, convert_to_numpy=True)
        db_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            db_chunk = DocumentChunk(
                document_id=document_id,
                content=chunk["content"],
                embedding=embedding.tolist(),  # pgvector / JSON compatible
                page_number=chunk.get("metadata", {}).get("page_number"),
                chunk_index=i,
                chunk_metadata={
                    **chunk.get("metadata", {})
                    
                },
                created_at=datetime.utcnow()
            )
            db_chunks.append(db_chunk)
        self.db.bulk_save_objects(db_chunks)
        self.db.commit()
    
    async def _save_image(
        self, 
        image_item: bytes, 
        document_id: int, 
        page_number: int,
        metadata: Dict[str, Any]
    ) -> DocumentImage:
        """
        Save an extracted image.
        
        TODO: Implement image saving
        - Save image file to disk
        - Create DocumentImage record
        - Extract caption if available
        """
      
        
        
        # Create database record
        db_image = DocumentImage(
            document_id=document_id,
            file_path=metadata.get("file_path", ""),
            page_number=page_number,
            caption=metadata.get("caption", ""),
            width=metadata.get("width"),
            height=metadata.get("height"),
            image_metadata=metadata,
            created_at=datetime.now()
        )
        #print("db image:", db_image)
        self.db.add(db_image)
        self.db.commit()
        self.db.refresh(db_image)

       
        # Build chunk for image
        chunk = {
            "content": metadata.get("caption", "") or "",  # use caption/OCR text
            "metadata": {
                "document_id": document_id,
                "page_number": page_number,
                "content_type": "image",
                "file_path": metadata.get("file_path"),
                "width": metadata.get("width"),
                "height": metadata.get("height"),
                "bounding_box": metadata.get("bbox"),
                "provenance_id": metadata.get("prov_id")
            }
        }

        # Optional: generate embedding from caption text
        embedding = None
        if chunk["content"]:
            embedding = self.model.encode(chunk["content"], convert_to_numpy=True).tolist()

        db_chunk = DocumentChunk(
            document_id=document_id,
            content=chunk["content"],
            embedding=embedding,
            page_number=page_number,
            metadata=chunk["metadata"],
            created_at=datetime.utcnow()
        )
        self.db.add(db_chunk)
        self.db.commit()
        self.db.refresh(db_chunk)
        return db_chunk

        
        #return db_image
    
    async def _save_table(
        self,
        table_item: Any,
        document_id: int,
        page_number: int,
        metadata: Dict[str, Any]
    ) -> DocumentTable:
        """
        Save an extracted table.
        
        - Render table as image
        - Store structured data as JSON
        - Create DocumentTable record
        - Extract caption if available
        """
        import os
        import uuid
        from PIL import Image, ImageDraw, ImageFont
        import pandas as pd
        
        print("Saving table...")
        # Try to get structured data
        table_data = None
        rows = 0
        columns = 0
        
        try:
            # Docling tables might have export methods
            if hasattr(table_item, 'export_to_dataframe'):
                df = table_item.export_to_dataframe()
                table_data = df.to_dict('records')
                rows, columns = df.shape
            elif hasattr(table_item, 'data'):
                table_data = table_item.data
                if isinstance(table_data, list) and table_data:
                    rows = len(table_data)
                    columns = len(table_data[0]) if isinstance(table_data[0], (list, dict)) else 1
        except Exception:
            pass
        
        # Render table as image
        table_id = str(uuid.uuid4())
        table_filename = f"{table_id}.png"
        table_dir = os.path.join(settings.UPLOAD_DIR, "tables")
        os.makedirs(table_dir, exist_ok=True)
        table_path = os.path.join(table_dir, table_filename)
        
        # Simple table rendering (you might want to use a proper table rendering library)
        try:
            #self.render_table_image(table_data, table_path)
            
            if table_data and isinstance(table_data, list):
                # Create a simple text-based image
                img = Image.new('RGB', (800, 400), color='white')
                draw = ImageDraw.Draw(img)
                y_offset = 10
                for row in table_data[:10]:  # Limit to first 10 rows
                    if isinstance(row, dict):
                        text = " | ".join(str(v) for v in row.values())
                    else:
                        text = str(row)
                    draw.text((10, y_offset), text[:100], fill='black')  # Truncate long text
                    y_offset += 20
                img.save(table_path)
            else:
                # Fallback: create placeholder image
                img = Image.new('RGB', (400, 200), color='lightgray')
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), "Table Preview", fill='black')
                img.save(table_path)
                
        except Exception:
            # If rendering fails, create a minimal placeholder
            print("Rendering table failed, creating placeholder1")
            img = Image.new('RGB', (200, 100), color='white')
            img.save(table_path)
        
        # Create database record
        db_table = DocumentTable(
            document_id=document_id,
            image_path=table_path,
            data=table_data,
            page_number=page_number,
            caption=metadata.get("caption", ""),
            rows=rows,
            columns=columns,
            table_metadata=metadata,
            created_at=datetime.now(),
        )
        
        self.db.add(db_table)
        self.db.commit()
        self.db.refresh(db_table)
        
        return db_table
    
    async def _update_document_status(
        self, 
        document_id: int, 
        status: str, 
        error_message: str = None
    ):
        """
        Update document processing status.
        
        This is implemented as an example.
        """
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.processing_status = status
            if error_message:
                document.error_message = error_message
            self.db.commit()
    

    @staticmethod
    def bbox_to_list(bbox):
        return [bbox.l, bbox.t, bbox.r, bbox.b] if bbox else None

    def make_element_id(self, document_id: int, page_no: int, bbox: List[float]) -> str:
        key = f"{document_id}-{page_no}-{bbox[0]}-{bbox[1]}-{bbox[2]}-{bbox[3]}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]  # short stable ID 



    
   

