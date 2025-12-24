# Multimodal Document Chat System - Coding Test

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



### README Must Include
- Project overview
- Tech stack
- Setup instructions (Docker)
- Environment variables (.env.example)
- API testing examples
- Features implemented
- Known limitations
- Future improvements
- Screenshots (minimum 5):
  - Document upload screen
    <img width="975" height="301" alt="image" src="https://github.com/user-attachments/assets/6f6372a1-e6dd-4057-b3ea-e0796cd0c068" />

  - Document processing completion screen
  - Chat interface
  - Answer example with images
  - Answer example with tables

### How to Submit
1. Push code to GitHub
2. Test that `docker-compose up` works
3. Send repository URL via email
4. Include any special instructions

---

## Test Scenarios

### Scenario 1: Basic Document Processing
1. Upload a technical paper PDF
2. Verify text, images, and tables extraction
3. Check extracted content on document detail page

### Scenario 2: Text-based Question
1. Ask "What is the main conclusion of this paper?"
2. Verify answer is generated with relevant text context

### Scenario 3: Image-related Question
1. Ask "Show me the architecture diagram"
2. Verify related images are displayed in chat

### Scenario 4: Table-related Question
1. Ask "What are the experimental results?"
2. Verify related tables are displayed in chat

### Scenario 5: Multi-turn Conversation
1. First question: "What is the dataset used?"
2. Follow-up: "How many samples does it contain?"
3. Verify previous conversation context is maintained

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

## Implementation Guidelines

Refer to the service skeleton files for detailed implementation guidance:
- `backend/app/services/document_processor.py` - Document processing guidelines
- `backend/app/services/vector_store.py` - Vector store implementation tips
- `backend/app/services/chat_engine.py` - Chat engine implementation tips

Each file contains detailed TODO comments with implementation hints and examples.

---

## Troubleshooting

### Document Processing Issues
**Problem**: Docling can't extract tables
**Solution**: 
- Check PDF format (ensure it's not scanned image)
- Add fallback parsing logic
- Manually define table structure patterns

### LLM API Costs
**Problem**: OpenAI API is expensive
**Solution**: Use free alternatives
- Use caching for repeated queries
- Use cheaper models (gpt-3.5-turbo)
- Use local LLM (Ollama) for development

### Vector Search Issues
**Problem**: Search results are not relevant
**Solution**:
- Verify embedding model is working
- Check chunk size and overlap settings
- Ensure pgvector extension is installed
- Test with simple queries first

### CORS Issues
**Problem**: Frontend can't call backend API
**Solution**:
- Add CORS middleware in FastAPI
- Allow origin: http://localhost:3000
- Check network configuration in Docker

---

## Free LLM Options

You don't need to pay for OpenAI API! Here are free alternatives:

### Option 1: Ollama (Recommended for Development)

**Completely free, runs locally on your machine**

1. **Install Ollama**
```bash
# Mac
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

2. **Download a model**
```bash
# Llama 3.2 (3B - fast, good for development)
ollama pull llama3.2

# Or Llama 3.1 (8B - better quality)
ollama pull llama3.1

# Or Mistral (7B - good balance)
ollama pull mistral
```

3. **Update your .env**
```bash
# Use Ollama instead of OpenAI
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

**Pros**: Free, private, no API limits, works offline
**Cons**: Requires decent hardware (8GB+ RAM), slower than cloud APIs

---

### Option 2: Google Gemini (Free Tier)

**Free tier: 60 requests per minute**

1. **Get free API key**
   - Go to https://makersuite.google.com/app/apikey
   - Click "Create API Key"
   - Copy your key

2. **Update .env**
```bash
GOOGLE_API_KEY=your-gemini-api-key
LLM_PROVIDER=gemini
```

**Pros**: Free, fast, good quality
**Cons**: Rate limits, requires internet

---

### Option 3: Groq (Free Tier)

**Free tier: Very fast inference, generous limits**

1. **Get free API key**
   - Go to https://console.groq.com
   - Sign up and get API key

2. **Update .env**
```bash
GROQ_API_KEY=your-groq-api-key
LLM_PROVIDER=groq
```

**Pros**: Free, extremely fast, good quality
**Cons**: Rate limits, requires internet

---

### Comparison Table

| Provider | Cost | Speed | Quality | Setup |
|----------|------|-------|---------|-------|
| **Ollama** | Free | Medium | Good | Easy |
| **Gemini** | Free | Fast | Very Good | Very Easy |
| **Groq** | Free | Very Fast | Good | Very Easy |
| OpenAI | Paid | Fast | Excellent | Very Easy |

**Recommended**: Use **Ollama** for development (free, no limits)

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

## Questions?

If you have any questions, please create an issue or contact us via email.

Good luck!

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
