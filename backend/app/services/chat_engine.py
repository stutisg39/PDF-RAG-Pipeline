"""
Chat engine service for multimodal RAG.

TODO: Implement this service to:
1. Process user messages
2. Search for relevant context using vector store
3. Find related images and tables
4. Generate responses using LLM
5. Support multi-turn conversations
"""
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.models.conversation import Conversation, Message
from app.services.vector_store import VectorStore
from app.core.config import settings
from openai import OpenAI
from dotenv import load_dotenv
from app.models.document import Document, DocumentChunk, DocumentImage, DocumentTable
from pathlib import Path
import time


class ChatEngine:
    """
    Multimodal chat engine with RAG.
    
    This is a SKELETON implementation. You need to implement the core logic.
    """
    load_dotenv()  # Load environment variables from .env file
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    def __init__(self, db: Session):
        self.db = db
        self.vectorStore = VectorStore(db)
        self.llm = None  # TODO: Initialize LLM (OpenAI, Ollama, etc.)
        
        
    
    async def process_message(
        self,
        conversation_id: int,
        message: str,
        document_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message and generate multimodal response.
        
        Implementation steps:
        1. Load conversation history (for multi-turn support)
        2. Search vector store for relevant context
        3. Find related images and tables
        4. Build prompt with context and history
        5. Generate response using LLM
        6. Format response with sources (text, images, tables)
        
        Args:
            conversation_id: Conversation ID
            message: User message
            document_id: Optional document ID to scope search
            
        Returns:
            {
            "answer": "...",
            "sources": [
                {
                "type": "text",
                "content": "...",
                "page": 3,
                "score": 0.95
                },
                {
                "type": "image",
                "url": "/uploads/images/xxx.png",
                "caption": "Figure 1: ...",
                "page": 3
                },
                {
                "type": "table",
                "url": "/uploads/tables/yyy.png",
                "caption": "Table 1: ...",
                "page": 5,
                "data": {...}  # structured table data
                }
            ],
            "processing_time": 2.5
            }
        """
        start_time = time.time()
        
        try:
            print
            # 1. Load conversation history
            history = await self._load_conversation_history(conversation_id)
            print(f"Loaded {len(history)} messages from conversation {conversation_id}")
            # 2. Search vector store for relevant context
            context = await self._search_context(message, document_id)
            print(f"Found {len(context)} relevant context chunks")
            # 3. Find related images and tables
            print  ( "Finding related media...")
            media = await self._find_related_media(context)
            print(f"Found {len(media.get('images', []))} related images and {len(media.get('tables', []))} related tables")

            #media = {"images": [], "tables": []}  # TODO: Implement media finding   
            # 4. Generate response using LLM
            answer =  self._generate_response(message, context, history, media)
            print("Generated answer from LLM")
            # 5. Save message to conversation
            user_msg = Message(conversation_id=conversation_id, role="user", content=message)
            assistant_msg = Message(conversation_id=conversation_id, role="assistant", content=answer)
            self.db.add(user_msg)
            self.db.add(assistant_msg)
            self.db.commit()
            
            # 6. Format response with sources
            sources = self._format_sources(context, media)
            
            processing_time = time.time() - start_time
            
            return {
            "answer": answer,
            "sources": sources,
            "processing_time": round(processing_time, 2)
            }
            
        except Exception as e:
            self.db.rollback()
            raise RuntimeError(f"Error processing message: {str(e)}")
        # TODO: Implement message processing
        # 
        #  Example LLM usage with OpenAI:
        # from openai import OpenAI
        # client = OpenAI(api_key=settings.OPENAI_API_KEY)
        # 
        # response = client.chat.completions.create(
        #     model=settings.OPENAI_MODEL,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt}
        #     ]
        # )
        # 
        # Example with LangChain:
        # from langchain_openai import ChatOpenAI
        # from langchain.prompts import ChatPromptTemplate
        # 
        # llm = ChatOpenAI(model=settings.OPENAI_MODEL)
        # prompt = ChatPromptTemplate.from_messages([...])
        # chain = prompt | llm
        # response = chain.invoke({...})
        
    
    async def _load_conversation_history(
        self,
        conversation_id: int,
        limit: int = 5
    ) -> List[Dict[str, str]]:
        """
        Load recent conversation history.
        
        TODO: Implement conversation history loading
        - Load last N messages from conversation
        - Format for LLM context
        - Include both user and assistant messages
        
        Returns:
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        """

        print(f"Loading last {limit} messages for conversation {conversation_id}")
        # 1. Fetch last N messages (most recent first or last depending on repo)
        messages = await self.fetch_last_messages(
            conversation_id=conversation_id,
            limit=limit
        )

        if not messages:
            return []

        # 2. Ensure chronological order (oldest → newest)
        messages = sorted(messages, key=lambda m: m.created_at)

        # 3. Format for LLM context
        history: List[Dict[str, str]] = []
        for message in messages:
            if message.role not in ("user", "assistant"):
                continue  # skip system/tool messages if stored

            history.append({
                "role": message.role,
                "content": message.content
            })
        print(f"Loaded {len(history)} messages from conversation {conversation_id}")
        return history
    
    async def _search_context(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant context using vector store.
        
        TODO: Implement context search
        - Use vector store similarity search
        - Filter by document if specified
        - Return relevant chunks with metadata
        """
        
        results = await self.vectorStore.similarity_search(
        query=query,
        document_id=document_id,
        k=k
    )

        if not results:
            return []
        return results
    
    async def _find_related_media(
        self,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find related images and tables from context chunks.
        
        TODO: Implement related media finding
        - Extract image/table references from chunk metadata
        - Query database for actual image/table records
        - Return with URLs for frontend display
        
        Returns:
            {
                "images": [
                    {
                        "url": "/uploads/images/xxx.png",
                        "caption": "...",
                        "page": 3
                    }
                ],
                "tables": [
                    {
                        "url": "/uploads/tables/yyy.png",
                        "caption": "...",
                        "page": 5,
                        "data": {...}
                    }
                ]
            }
        """
        related_content = await self.vectorStore.get_related_content(chunk_ids=[chunk["id"] for chunk in context_chunks])
        related_images: List[Dict[str, Any]] = related_content.get("images", [])
        related_tables: List[Dict[str, Any]] = related_content.get("tables", [])
        """
        # Collect all provenance IDs from the chunks
        ids = []
        for chunk in context_chunks:
            print(chunk.)
            metadata = chunk.get("chunk_metadata", {})
            print(metadata)  # debug print
            ref_id = metadata.get("ref_id")
            if ref_id:
                ids.append(ref_id)

        print(f"Extracted {len(ids)} provenance IDs from context chunks")
        print
        if not ids:
            return {"images": [], "tables": []}

        # Query images linked by prov_id
        db_images = (
            self.db.query(DocumentImage)
            .filter(DocumentImage.image_metadata["ref_id"].astext.in_(ids))
            .all()
        )
        print(f"Found {len(db_images)} related images in DB")
        for img in db_images:
            related_images.append({
                "url": f"/uploads/images/{Path(img.file_path).name}",
                "caption": img.caption,
                "page": img.page_number,
                "width": img.width,
                "height": img.height
            })

        # Query tables linked by prov_id
        db_tables = (
            self.db.query(DocumentTable)
            .filter(DocumentTable.table_metadata["ref_id"].astext.in_(ids))
            .all()
        )
        print(f"Found {len(db_tables)} related tables in DB")
        for tbl in db_tables:
            related_tables.append({
                "url": f"/uploads/tables/{Path(tbl.image_path).name}",
                "caption": tbl.caption,
                "page": tbl.page_number,
                "rows": tbl.rows,
                "columns": tbl.columns,
                "data": tbl.data
            })
"""
        return {
            "images": related_images,
            "tables": related_tables
        }

          
    


    
    def _generate_response(
        self,
        message: str,
        context: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        media: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Generate response using LLM.
        
        TODO: Implement LLM response generation
        - Build comprehensive prompt with:
          - System instructions
          - Conversation history
          - Retrieved context
          - Available images/tables
        - Call LLM API
        - Return generated answer
        
        Prompt engineering tips:
        - Instruct LLM to reference images/tables when relevant
        - Include context from previous messages
        - Ask LLM to cite sources
        - Format for good UX (bullet points, etc.)
        """
        system_prompt = (
            "You are a helpful assistant answering user questions using the provided context.\n"
            "Rules:\n"
            "- Use the retrieved context when relevant.\n"
            "- If an explicit answer is not present, infer a reasonable answer from the context and state that it is an inference.\n"
            "- Reference images or tables if they are relevant.\n"
            "- Be clear, concise, and well-structured.\n"
            "- Use bullet points when helpful.\n"
            "- Cite sources using [source_id] when possible.\n")
        # 4. Assemble messages for LLM
        messages: List[Dict[str, str]] = []

        messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": message})
        #messages.append({"role": "system", "content": self._format_sources(context, [])})
        #print(f"LLM prompt messages: {messages}")
        # 5. Call LLM API
        response =  self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            temperature=0.2
         )

        # 6. Extract answer text
        answer = response.choices[0].message.content
        #print(f"LLM response: {answer}")

        return answer
    
    def _format_sources(
        self,
        context: List[Dict[str, Any]],
        media: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Format sources for response.
        
        This is implemented as an example.
        """
        sources = []
        
        # Add text sources
        for chunk in context[:3]:  # Top 3 text chunks
            sources.append({
                "type": "text",
                "content": chunk["content"],
                "page": chunk.get("page_number"),
                "score": chunk.get("score", 0.0)
            })
        
        # Add image sources
        for image in media.get("images", []):
            sources.append({
                "type": "image",
                "url": image["url"],
                "caption": image.get("caption"),
                "page": image.get("page")
            })
        
        # Add table sources
        for table in media.get("tables", []):
            sources.append({
                "type": "table",
                "url": table["url"],
                "caption": table.get("caption"),
                "page": table.get("page"),
                "data": table.get("data")
            })
        
        return sources
    
    async def fetch_last_messages(self, conversation_id: int, limit: int = 5):
        print
        messages = (
            self.db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())  # or Message.id.desc()
            .limit(limit)
            .all()
        )

        # Reverse so messages are chronological
        messages.reverse()

        return messages

