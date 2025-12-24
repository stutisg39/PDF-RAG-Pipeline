# Multimodal Document Chat System 

## Project Overview

Build a system that allows users to upload PDF documents, extract text, images, and tables, and engage in multimodal chat based on the extracted content.

### Core Features
1. **Document Processing**: PDF parsing using Docling (extract text, images, tables)
2. **Vector Store**: Store extracted content in vector database
3. **Multimodal Chat**: Provide answers with related images/tables for text questions
4. **Multi-turn Conversation**: Maintain conversation context for continuous questioning

---

### Infrastructure Setup
- Docker Compose configuration (PostgreSQL+pgvector, Redis, Backend, Frontend)
- Database schema and models (SQLAlchemy)
- API base structure (FastAPI)
- Frontend base structure (Next.js + TailwindCSS)

### Database Models
- `Document` - Uploaded document information
- `DocumentChunk` - Text chunks (with vector embeddings)
- `DocumentImage` - Extracted images
- `DocumentTable` - Extracted tables
- `Conversation` - Chat sessions
- `Message` - Chat messages

### API Endpoints (Skeleton provided)
- `POST /api/documents/upload` - Upload document
- `GET /api/documents` - List documents
- `GET /api/documents/{id}` - Document details
- `DELETE /api/documents/{id}` - Delete document
- `POST /api/chat` - Send chat message
- `GET /api/conversations` - List conversations
- `GET /api/conversations/{id}` - Get conversation history

### Frontend Pages (Layout only)
- `/` - Home (document list)
- `/upload` - Document upload
- `/chat` - Chat interface
- `/documents/[id]` - Document details

### Development Tools
- FastAPI Swagger UI (`http://localhost:8000/docs`)
- Hot reload (Backend & Frontend)
- Environment configuration

---

### 1. Document Processing Pipeline (Critical)

**Location**: `backend/app/services/document_processor.py`


## System Architecture

```
┌─────────────┐
│   Frontend  │ (Next.js)
│  Chat UI    │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────┐
│   Backend   │ (FastAPI)
│  API Server │
└──────┬──────┘
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌─────────────┐   ┌─────────────┐
│  Document   │   │    Chat     │
│  Processor  │   │   Engine    │
│  (Docling)  │   │   (RAG)     │
└──────┬──────┘   └──────┬──────┘
       │                 │
       ▼                 ▼
┌─────────────────────────────┐
│      Vector Store           │
│    (PostgreSQL+pgvector)    │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│    File Storage             │
│  (Images, Tables, PDFs)     │
└─────────────────────────────┘
```

---

## Data Models

### Document
```python
class Document:
    id: int
    filename: str
    file_path: str
    upload_date: datetime
    processing_status: str  # 'pending', 'processing', 'completed', 'error'
    total_pages: int
    text_chunks_count: int
    images_count: int
    tables_count: int
```

### DocumentChunk
```python
class DocumentChunk:
    id: int
    document_id: int
    content: str
    embedding: Vector(1536)  # pgvector
    page_number: int
    chunk_index: int
    metadata: JSON  # {related_images: [...], related_tables: [...]}
```

### DocumentImage
```python
class DocumentImage:
    id: int
    document_id: int
    file_path: str
    page_number: int
    caption: str
    width: int
    height: int
```

### DocumentTable
```python
class DocumentTable:
    id: int
    document_id: int
    image_path: str  # Rendered table as image
    data: JSON  # Structured table data
    page_number: int
    caption: str
```

### Conversation & Message
```python
class Conversation:
    id: int
    title: str
    created_at: datetime
    document_id: Optional[int]  # Conversation about specific document

class Message:
    id: int
    conversation_id: int
    role: str  # 'user', 'assistant'
    content: str
    sources: JSON  # Sources used in answer (text, images, tables)
    created_at: datetime
```

---

## Tech Stack

### Backend
- **Framework**: FastAPI
- **PDF Processing**: Docling
- **Vector DB**: PostgreSQL + pgvector
- **Embeddings**: OpenAI API or HuggingFace
- **LLM**: OpenAI GPT-4o-mini or Ollama
- **Task Queue**: Celery + Redis (optional)

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Styling**: TailwindCSS
- **UI Components**: shadcn/ui
- **State Management**: React Hooks
- **API Client**: fetch/axios

### Infrastructure
- **Database**: PostgreSQL 15 + pgvector
- **Cache**: Redis
- **Container**: Docker + Docker Compose

---

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Node.js 18+
- Python 3.11+
- OpenAI API Key (or Ollama for local LLM)

### Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd coding-test-4th

# 2. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Start all services
docker-compose up -d

# 4. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---
### Limitations
1. Agent currently uses free version of OPENAI model which supports only text based searches. 

### Future Improvements
1. Use any other multimodal like Gemini model for searches 
2. Show tables in a better way
3. Currently showing images and tables based on document id and page number in chunk ids. For future need to improve this  and make it more contextual
4. The model currently supports top 5 chunks. For future need to include searches/ answers based on full docuement context. Example summarize the document.

### Screenshots :
  - Document upload screen
    <img width="975" height="301" alt="image" src="https://github.com/user-attachments/assets/6f6372a1-e6dd-4057-b3ea-e0796cd0c068" />

  - Document processing completion screen
  - <img width="975" height="601" alt="image" src="https://github.com/user-attachments/assets/dbb39f92-8431-4823-81c9-f154ccb936df" />
  <img width="975" height="561" alt="image" src="https://github.com/user-attachments/assets/55d20283-ee3c-4f1d-8c59-244aed3073d6" />


  - Chat interface
    <img width="975" height="640" alt="image" src="https://github.com/user-attachments/assets/65d4ffcc-c3a2-45ef-ab1f-8afe0a70dda2" />

  - Answer example with images
    <img width="970" height="733" alt="image" src="https://github.com/user-attachments/assets/b5c15c96-6dcb-4f4b-b9a1-5ea7c5340224" />
    <img width="975" height="638" alt="image" src="https://github.com/user-attachments/assets/257fe3da-ba46-4f7f-bb98-583567515c11" />


  - Answer example with tables
  - <img width="975" height="440" alt="image" src="https://github.com/user-attachments/assets/2b5098d2-ad45-40dc-82b4-413639f71b08" />
  <img width="975" height="539" alt="image" src="https://github.com/user-attachments/assets/c5940962-1ab1-4d3c-9667-f7d5297b81af" />

  - Multi turn conversation
    <img width="975" height="644" alt="image" src="https://github.com/user-attachments/assets/ab2a2250-b9dd-4fbc-a9fb-57aa7e3462c0" />
    <img width="975" height="631" alt="image" src="https://github.com/user-attachments/assets/45582d2f-d8d6-42ab-bfe1-8160b2abd5f8" />






---

## Sample PDF

A sample PDF file is provided: `1706.03762v7.pdf`

This is a technical paper ("Attention Is All You Need") with:
- Multiple pages with text content
- Diagrams and architecture figures
- Tables with experimental results
- Complex layouts for testing

You should use this PDF to test your implementation.

---



## FAQ

**Q: Docling won't install.**
A: Try `pip install docling` or use the Docker image.

**Q: I don't have an OpenAI API key.**
A: You can install Ollama locally and use a free LLM (see Free LLM Options section).

**Q: Where should I save images?**
A: Save to `backend/uploads/images/` directory and store only the path in DB.

**Q: How should I display tables?**
A: Render tables as images or display JSON data as HTML tables in frontend.

**Q: How do I test the system locally?**
A: Follow the Getting Started section and use the provided sample PDF (1706.03762v7.pdf).


---

## Tips for Success

1. **Start Simple**: Get core features working before adding advanced features
2. **Test Early**: Test document processing with sample PDF immediately
3. **Use Tools**: Leverage Docling, LangChain to save time
4. **Focus on Core**: Perfect the RAG pipeline first
5. **Document Well**: Clear README helps evaluators understand your work
6. **Handle Errors**: Graceful error handling shows maturity
7. **Ask Questions**: If requirements are unclear, document your assumptions

---

## Support

For questions about this coding challenge:
- Open an issue in this repository
- Email: recruitment@interopera.co

---

**Version**: 1.0  
**Last Updated**: 2025-11-03  
**Author**: InterOpera-Apps Hiring Team
