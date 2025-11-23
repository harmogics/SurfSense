# Поток обработки документов (Document Processing Flow)

## Введение

Обработка документов в SurfSense представляет собой сложный pipeline от загрузки файла до индексации embeddings в векторной базе данных. Этот документ описывает complete flow от пользовательского действия до готовности документа для поиска.

## Архитектура обработки

```
┌──────────────────────────────────────────────────────────────┐
│            FRONTEND: File Upload UI                          │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     │ 1. Upload file (multipart/form-data)
                     │
┌────────────────────▼─────────────────────────────────────────┐
│        BACKEND: Document Creation Endpoint                   │
│        POST /api/v1/documents                                │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     │ 2. Store file & create Document record
                     │
┌────────────────────▼─────────────────────────────────────────┐
│           DATABASE: Documents Table                          │
│           Status: "processing"                               │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     │ 3. Trigger async processing (Celery task)
                     │
┌────────────────────▼─────────────────────────────────────────┐
│         CELERY: process_file_in_background                   │
│         - Extract text & metadata (ETL)                      │
│         - Chunk text                                         │
│         - Generate embeddings                                │
│         - Store chunks                                       │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     │ 4. Update document status
                     │
┌────────────────────▼─────────────────────────────────────────┐
│           DATABASE: Document & Chunks Tables                 │
│           Status: "completed" или "failed"                   │
│           + Embeddings in pgvector                           │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     │ 5. Available for search
                     │
┌────────────────────▼─────────────────────────────────────────┐
│              RAG Pipeline Integration                        │
│              (Hybrid Search + Reranking)                     │
└──────────────────────────────────────────────────────────────┘
```

## Детальный поток

### Фаза 1: Frontend Upload

**Component**: `surfsense_web/app/dashboard/[search_space_id]/documents/page.tsx`

```typescript
const DocumentsPage = () => {
  const [uploadingFiles, setUploadingFiles] = useState<File[]>([]);
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});

  const handleFileUpload = async (files: FileList) => {
    const filesArray = Array.from(files);

    // Set uploading state
    setUploadingFiles(filesArray);

    for (const file of filesArray) {
      await uploadSingleFile(file);
    }

    // Refresh document list
    refetch();
  };

  const uploadSingleFile = async (file: File) => {
    // Create FormData
    const formData = new FormData();
    formData.append("file", file);
    formData.append("search_space_id", searchSpaceId);

    try {
      // Upload with progress tracking
      const response = await fetch(
        `${BACKEND_URL}/api/v1/documents`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
          },
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const newDocument = await response.json();

      toast.success(`${file.name} uploaded successfully`);

      // Poll for processing status
      pollDocumentStatus(newDocument.id);

    } catch (error) {
      console.error("Upload error:", error);
      toast.error(`Failed to upload ${file.name}`);
    } finally {
      setUploadingFiles((prev) =>
        prev.filter((f) => f.name !== file.name)
      );
    }
  };

  // Poll document status until processing completes
  const pollDocumentStatus = async (documentId: number) => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(
          `${BACKEND_URL}/api/v1/documents/${documentId}`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );

        const document = await response.json();

        if (document.status === "completed") {
          clearInterval(interval);
          toast.success("Document processed successfully");
          refetch(); // Refresh list
        } else if (document.status === "failed") {
          clearInterval(interval);
          toast.error("Document processing failed");
        }
      } catch (error) {
        console.error("Poll error:", error);
        clearInterval(interval);
      }
    }, 2000); // Poll every 2 seconds
  };

  return (
    <div>
      <DropzoneUI
        onDrop={handleFileUpload}
        accept={{
          "application/pdf": [".pdf"],
          "text/plain": [".txt"],
          "application/msword": [".doc"],
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
            ".docx",
          ],
          // ... more MIME types
        }}
      />

      <DocumentList
        documents={documents}
        uploadingFiles={uploadingFiles}
        uploadProgress={uploadProgress}
      />
    </div>
  );
};
```

### Фаза 2: Backend Document Creation

**File**: `surfsense_backend/app/routes/documents_routes.py`

```python
@router.post("", response_model=DocumentResponse)
async def create_document(
    file: UploadFile,
    search_space_id: int = Form(...),
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    """
    Upload and create document.
    Triggers async processing in background.
    """

    # 1. Validate search space ownership
    search_space = await session.execute(
        select(SearchSpace).filter(
            SearchSpace.id == search_space_id,
            SearchSpace.user_id == user.id,
        )
    )
    if not search_space.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Access denied")

    # 2. Validate file type
    allowed_types = [
        "application/pdf",
        "text/plain",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        # ... more types
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file.content_type} not supported"
        )

    # 3. Read file content
    file_content = await file.read()

    # 4. Calculate content hash (for deduplication)
    content_hash = hashlib.sha256(file_content).hexdigest()

    # 5. Check for duplicate
    existing = await session.execute(
        select(Document).filter(
            Document.content_hash == content_hash,
            Document.user_id == user.id,
            Document.search_space_id == search_space_id,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=409,
            detail="Document already exists"
        )

    # 6. Store file to disk/S3
    file_path = await store_file(
        file_content,
        user_id=str(user.id),
        filename=file.filename,
    )

    # 7. Create Document record
    document = Document(
        name=file.filename,
        file_path=file_path,
        content_hash=content_hash,
        file_type=file.content_type,
        file_size=len(file_content),
        status="processing",  # Initial status
        user_id=user.id,
        search_space_id=search_space_id,
        created_at=datetime.utcnow(),
    )

    session.add(document)
    await session.commit()
    await session.refresh(document)

    # 8. Trigger background processing
    task = process_file_in_background.delay(
        file_id=document.id,
        user_id=str(user.id),
        search_space_id=search_space_id,
        etl_service="UNSTRUCTURED",  # Default ETL service
    )

    # 9. Store task ID for tracking
    document.celery_task_id = task.id
    await session.commit()

    logger.info(
        f"Document created: {document.id} | "
        f"User: {user.id} | "
        f"Task: {task.id}"
    )

    return document
```

### Фаза 3: Celery Background Processing

**File**: `surfsense_backend/app/celery_tasks/document_processing.py`

```python
@celery_app.task(bind=True, max_retries=3)
def process_file_in_background(
    self,
    file_id: int,
    user_id: str,
    search_space_id: int,
    etl_service: str = "UNSTRUCTURED",
) -> tuple[int, str | None]:
    """
    Background task for document processing.

    Steps:
    1. Extract text & metadata (ETL)
    2. Chunk text
    3. Generate embeddings
    4. Store chunks with embeddings
    5. Update document status

    Args:
        file_id: Document ID
        user_id: User ID
        search_space_id: Search space ID
        etl_service: ETL service to use (UNSTRUCTURED, LLAMACLOUD, DOCLING)

    Returns:
        (file_id, error_message | None)
    """

    logger.info(f"Processing document {file_id} with {etl_service}")

    try:
        # Get database session
        async with get_async_session_context() as session:

            # === STEP 1: Load Document ===
            result = await session.execute(
                select(Document).filter(Document.id == file_id)
            )
            document = result.scalar_one_or_none()

            if not document:
                raise ValueError(f"Document {file_id} not found")

            # === STEP 2: Extract text using ETL service ===
            logger.info(f"Extracting text from {document.name}")

            if etl_service == "UNSTRUCTURED":
                extracted_data = await UnstructuredService.extract_text(
                    document.file_path
                )
            elif etl_service == "LLAMACLOUD":
                extracted_data = await LlamaCloudService.extract_text(
                    document.file_path
                )
            elif etl_service == "DOCLING":
                extracted_data = await DoclingService.extract_text(
                    document.file_path
                )
            else:
                raise ValueError(f"Unknown ETL service: {etl_service}")

            # extracted_data format:
            # {
            #     "text": "Full document text",
            #     "metadata": {
            #         "author": "...",
            #         "created_date": "...",
            #         "page_count": 10,
            #     },
            #     "elements": [...]  # Optional structured elements
            # }

            full_text = extracted_data["text"]
            metadata = extracted_data.get("metadata", {})

            # === STEP 3: Update Document with extracted metadata ===
            document.metadata = metadata
            document.content_preview = full_text[:500]  # First 500 chars

            # === STEP 4: Chunk text ===
            logger.info(f"Chunking text for document {file_id}")

            chunker = get_chunker(document.file_type)  # RecursiveChunker or CodeChunker

            chunks = chunker.chunk(
                text=full_text,
                chunk_size=512,  # Tokens
                chunk_overlap=50,
            )

            logger.info(f"Created {len(chunks)} chunks for document {file_id}")

            # === STEP 5: Generate embeddings ===
            logger.info(f"Generating embeddings for {len(chunks)} chunks")

            # Get user's embedding model preference
            embedding_service = await get_embedding_service(
                session,
                user_id,
                search_space_id,
            )

            # Batch embed chunks (for efficiency)
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                batch_texts = [chunk.text for chunk in batch]

                embeddings = await embedding_service.embed_documents(batch_texts)
                all_embeddings.extend(embeddings)

            logger.info(f"Generated {len(all_embeddings)} embeddings")

            # === STEP 6: Store chunks with embeddings ===
            logger.info(f"Storing chunks for document {file_id}")

            chunk_objects = []

            for idx, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                chunk_obj = Chunk(
                    document_id=document.id,
                    chunk_index=idx,
                    text=chunk.text,
                    embedding=embedding,  # pgvector column
                    token_count=chunk.token_count,
                    metadata={
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        **metadata,  # Include document metadata
                    },
                    user_id=user_id,
                    search_space_id=search_space_id,
                )
                chunk_objects.append(chunk_obj)

            # Bulk insert
            session.add_all(chunk_objects)

            # === STEP 7: Update document status ===
            document.status = "completed"
            document.chunk_count = len(chunks)
            document.processed_at = datetime.utcnow()

            await session.commit()

            logger.info(
                f"Document {file_id} processed successfully. "
                f"{len(chunks)} chunks stored."
            )

            return (file_id, None)

    except Exception as e:
        logger.exception(f"Error processing document {file_id}: {e}")

        # Update document status to failed
        try:
            async with get_async_session_context() as session:
                result = await session.execute(
                    select(Document).filter(Document.id == file_id)
                )
                document = result.scalar_one_or_none()

                if document:
                    document.status = "failed"
                    document.error_message = str(e)
                    await session.commit()

        except Exception as db_error:
            logger.exception(f"Failed to update document status: {db_error}")

        # Retry task
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))

        return (file_id, str(e))
```

### Фаза 4: ETL Service Implementation

**File**: `surfsense_backend/app/services/docling_service.py`

```python
class DoclingService:
    """
    ETL service using IBM Docling library.
    Supports PDF, DOCX, HTML, images, and more.
    """

    @staticmethod
    async def extract_text(file_path: str) -> dict:
        """
        Extract text and metadata from document.

        Args:
            file_path: Path to document file

        Returns:
            {
                "text": "Full document text",
                "metadata": {...},
                "elements": [...]
            }
        """

        try:
            # Initialize DocumentConverter
            converter = DocumentConverter()

            # Convert document
            result = converter.convert(file_path)

            # Extract text
            full_text = result.document.export_to_markdown()

            # Extract metadata
            metadata = {
                "page_count": len(result.document.pages),
                "author": result.document.metadata.get("author"),
                "title": result.document.metadata.get("title"),
                "creation_date": result.document.metadata.get("creation_date"),
            }

            # Extract structured elements (tables, figures, etc.)
            elements = []
            for page in result.document.pages:
                for element in page.elements:
                    elements.append({
                        "type": element.type,
                        "text": element.text,
                        "bbox": element.bbox,
                        "page": page.page_num,
                    })

            return {
                "text": full_text,
                "metadata": metadata,
                "elements": elements,
            }

        except Exception as e:
            logger.error(f"Docling extraction failed for {file_path}: {e}")
            raise
```

### Фаза 5: Chunking Strategy

**File**: `surfsense_backend/app/services/chunking_service.py`

```python
from chonkie import RecursiveChunker, CodeChunker, Tokenizer


def get_chunker(file_type: str):
    """
    Get appropriate chunker based on file type.

    Args:
        file_type: MIME type or file extension

    Returns:
        Chunker instance
    """

    # Tokenizer (for token counting)
    tokenizer = Tokenizer.from_pretrained("gpt-3.5-turbo")

    # Code files
    if file_type in [
        "text/x-python",
        "application/javascript",
        "text/x-java",
        "text/x-c",
    ]:
        return CodeChunker(
            tokenizer=tokenizer,
            chunk_size=512,
            chunk_overlap=50,
        )

    # Regular text/documents
    else:
        return RecursiveChunker(
            tokenizer=tokenizer,
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )


class ChunkingService:
    """Service for text chunking"""

    @staticmethod
    def chunk_text(
        text: str,
        file_type: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[Chunk]:
        """
        Chunk text into smaller pieces.

        Args:
            text: Full text to chunk
            file_type: File type for chunker selection
            chunk_size: Max tokens per chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of Chunk objects
        """

        chunker = get_chunker(file_type)

        chunks = chunker.chunk(text)

        # Convert to Chunk objects with metadata
        chunk_objects = []

        for idx, chunk in enumerate(chunks):
            chunk_objects.append(
                Chunk(
                    text=chunk.text,
                    token_count=chunk.token_count,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    chunk_index=idx,
                )
            )

        return chunk_objects
```

### Фаза 6: Embedding Generation

**File**: `surfsense_backend/app/services/embedding_service.py`

```python
class EmbeddingService:
    """
    Service for generating embeddings.
    Supports multiple embedding models.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.dimension = self._get_dimension()

    def _get_dimension(self) -> int:
        """Get embedding dimension for model"""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.model_name, 1536)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """

        try:
            # Use OpenAI embeddings API
            client = AsyncOpenAI()

            response = await client.embeddings.create(
                model=self.model_name,
                input=texts,
            )

            embeddings = [item.embedding for item in response.data]

            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for single query"""
        embeddings = await self.embed_documents([text])
        return embeddings[0]


async def get_embedding_service(
    session: AsyncSession,
    user_id: str,
    search_space_id: int,
) -> EmbeddingService:
    """
    Get user's preferred embedding service.

    Args:
        session: Database session
        user_id: User ID
        search_space_id: Search space ID

    Returns:
        Configured EmbeddingService
    """

    # Fetch user preference
    result = await session.execute(
        select(SearchSpace).filter(
            SearchSpace.id == search_space_id,
            SearchSpace.user_id == user_id,
        )
    )
    search_space = result.scalar_one_or_none()

    # Get embedding model from config (or default)
    embedding_model = search_space.embedding_model or "text-embedding-3-small"

    return EmbeddingService(model_name=embedding_model)
```

### Фаза 7: Storage в PostgreSQL

**Database Schema**:

```sql
-- Documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_type VARCHAR(100),
    file_size BIGINT,
    content_hash VARCHAR(64) UNIQUE,
    content_preview TEXT,
    metadata JSONB,

    status VARCHAR(50) DEFAULT 'processing',  -- processing, completed, failed
    error_message TEXT,

    chunk_count INT DEFAULT 0,
    processed_at TIMESTAMP,

    celery_task_id VARCHAR(255),

    user_id UUID NOT NULL REFERENCES "user"(id),
    search_space_id INT NOT NULL REFERENCES search_spaces(id),

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(content_hash, user_id, search_space_id)
);

-- Chunks table
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    chunk_index INT NOT NULL,
    text TEXT NOT NULL,
    embedding VECTOR(1536),  -- pgvector type

    token_count INT,
    metadata JSONB,

    user_id UUID NOT NULL REFERENCES "user"(id),
    search_space_id INT NOT NULL REFERENCES search_spaces(id),

    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(document_id, chunk_index)
);

-- Indices
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_user_id ON chunks(user_id);
CREATE INDEX idx_chunks_search_space_id ON chunks(search_space_id);

-- HNSW index for vector similarity search
CREATE INDEX idx_chunks_embedding_hnsw ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search
CREATE INDEX idx_chunks_text_gin ON chunks
USING gin(to_tsvector('english', text));
```

## Status Tracking & Polling

### Frontend Status Display

```typescript
const DocumentStatus = ({ document }: { document: Document }) => {
  const getStatusDisplay = () => {
    switch (document.status) {
      case "processing":
        return (
          <div className="status-processing">
            <Spinner />
            <span>Processing...</span>
          </div>
        );

      case "completed":
        return (
          <div className="status-completed">
            <CheckIcon />
            <span>{document.chunk_count} chunks</span>
          </div>
        );

      case "failed":
        return (
          <div className="status-failed">
            <ErrorIcon />
            <span>Failed: {document.error_message}</span>
          </div>
        );

      default:
        return null;
    }
  };

  return <div className="document-status">{getStatusDisplay()}</div>;
};
```

### Backend Status Endpoint

```python
@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    """Get document details including processing status"""

    result = await session.execute(
        select(Document).filter(
            Document.id == document_id,
            Document.user_id == user.id,
        )
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return document
```

## Error Handling & Retry Logic

### Celery Retry Strategy

```python
@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,  # 1 minute
)
def process_file_in_background(self, ...):
    try:
        # ... processing logic ...

    except TemporaryError as e:
        # Retry для temporary errors (network, API rate limit)
        if self.request.retries < self.max_retries:
            countdown = 60 * (2 ** self.request.retries)  # Exponential backoff
            raise self.retry(exc=e, countdown=countdown)

    except PermanentError as e:
        # Не retry для permanent errors (invalid file format)
        logger.error(f"Permanent error: {e}")
        # Update document status to failed
```

### Frontend Error Display

```typescript
{
  document.status === "failed" && (
    <Alert variant="destructive">
      <AlertTitle>Processing Failed</AlertTitle>
      <AlertDescription>
        {document.error_message || "An unknown error occurred"}
        <Button onClick={() => retryProcessing(document.id)}>Retry</Button>
      </AlertDescription>
    </Alert>
  );
}
```

## Performance Optimization

### 1. Batch Processing

```python
# Process multiple documents in parallel
@celery_app.task
def process_multiple_documents(document_ids: list[int]):
    # Create subtasks
    tasks = [
        process_file_in_background.s(doc_id)
        for doc_id in document_ids
    ]

    # Execute in parallel
    job = group(tasks)
    result = job.apply_async()

    return result.id
```

### 2. Embedding Caching

```python
# Cache embeddings для duplicate chunks
class EmbeddingService:
    _cache: dict[str, list[float]] = {}

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []

        for text in texts:
            # Check cache
            cache_key = hashlib.md5(text.encode()).hexdigest()

            if cache_key in self._cache:
                embeddings.append(self._cache[cache_key])
            else:
                embedding = await self._generate_embedding(text)
                self._cache[cache_key] = embedding
                embeddings.append(embedding)

        return embeddings
```

### 3. Chunking Optimization

```python
# Use chunking library (chonkie) для efficient chunking
# Supports token-aware chunking без overhead LLM calls
```

## Integration с RAG Pipeline

После обработки документ сразу доступен в поиске:

```python
# Hybrid search automatically includes new chunks
results = await ChunksHybridSearchRetriever.hybrid_search(
    query_text="user query",
    top_k=5,
    user_id=user_id,
    search_space_id=search_space_id,
    document_type="FILE",  # Includes uploaded documents
)
```

## Итоговая диаграмма

```
User uploads file.pdf
    ↓
Frontend: POST /api/v1/documents (multipart/form-data)
    ↓
Backend: Create Document record (status="processing")
    ↓
Backend: Trigger Celery task
    ↓
Celery Worker:
    ├─ ETL: Extract text (Docling/Unstructured/LlamaCloud)
    ├─ Chunk: Split into 512-token chunks (chonkie)
    ├─ Embed: Generate vectors (OpenAI embeddings)
    └─ Store: Save chunks + embeddings (PostgreSQL + pgvector)
    ↓
Database: Document status="completed", chunks stored
    ↓
Frontend: Poll status → Display "Ready"
    ↓
User can now search → Document chunks included in results
```

Этот pipeline обеспечивает:
- **Async processing** - не блокирует UI
- **Reliable** - retry logic для temporary failures
- **Scalable** - Celery distributed processing
- **Fast search** - HNSW index для vector search
- **Accurate** - Hybrid search (vector + full-text)
