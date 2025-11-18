# Векторизация, Chunking, Поиск и Интеграция Источников

## Введение

Этот документ объединяет описание трех взаимосвязанных компонентов:
1. **Embeddings & Vector Search** - векторные представления и семантический поиск
2. **Chunking & Indexing** - разбиение контента и индексация
3. **Connectors** - интеграция внешних источников данных

## ЧАСТЬ 1: Embeddings и Векторный Поиск

### 1.1 Конфигурация Embedding моделей

**Местоположение**: `surfsense_backend/app/config/__init__.py:161-183`

```python
from app.embeddings.auto_embeddings import AutoEmbeddings

class Config:
    # Embedding model configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Auto-initialization
    embedding_model_instance = AutoEmbeddings.get_embeddings(
        EMBEDDING_MODEL,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Dimension varies by model
    # text-embedding-3-small: 1536 dimensions
    # text-embedding-3-large: 3072 dimensions
    # text-embedding-ada-002: 1536 dimensions
```

### 1.2 Поддерживаемые Embedding модели

| Модель | Провайдер | Размерность | Использование |
|--------|-----------|-------------|---------------|
| **text-embedding-3-small** | OpenAI | 1536 | Оптимальное соотношение скорость/качество |
| **text-embedding-3-large** | OpenAI | 3072 | Максимальное качество |
| **text-embedding-ada-002** | OpenAI | 1536 | Legacy, стабильная модель |
| **Azure OpenAI embeddings** | Azure | 1536/3072 | Для enterprise окружений |
| **Custom models** | HuggingFace/Local | Variable | Domain-specific embeddings |

### 1.3 Генерация Embeddings

#### Document-level embedding (summary)

```python
# В generate_document_summary()
from app.config import config

# 1. Генерация conceptual summary через LLM
summary_content = await llm.ainvoke(SUMMARY_PROMPT_TEMPLATE.format(...))

# 2. Обогащение метаданными
enhanced_summary = f"{metadata_markdown}\n{summary_content}"

# 3. Генерация embedding
embedding = config.embedding_model_instance.embed(enhanced_summary)
# Output: list[float] с длиной 1536 или 3072

# 4. Сохранение в Document
document.embedding = embedding
```

#### Chunk-level embeddings

```python
# В create_document_chunks()
chunks = config.chunker_instance.chunk(content)

for chunk in chunks:
    # Embedding для каждого chunk
    chunk_embedding = config.embedding_model_instance.embed(chunk.text)

    chunk_obj = Chunk(
        content=chunk.text,
        embedding=chunk_embedding,
        document_id=document.id
    )
    session.add(chunk_obj)
```

### 1.4 Хранение в PostgreSQL + pgvector

**Database Schema**:
```python
from pgvector.sqlalchemy import Vector

class Document(Base):
    embedding = Column(Vector(1536), nullable=True)  # Dimension = model dimension

class Chunk(Base):
    embedding = Column(Vector(1536), nullable=True)
```

**Создание индекса для быстрого поиска**:
```sql
-- IVFFlat index для approximate nearest neighbor search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### 1.5 Семантический поиск

#### Vector Search

**Функция**: `vector_search()`
**Местоположение**: `surfsense_backend/app/retriver/chunks_hybrid_search.py:11-61`

```python
async def vector_search(
    session: AsyncSession,
    query_text: str,
    top_k: int,
    user_id: str,
    search_space_id: int
) -> list[Chunk]:
    """
    Векторный поиск по cosine similarity.

    Process:
    1. Генерация query embedding
    2. Поиск ближайших векторов в pgvector
    3. Фильтрация по user_id и search_space_id
    4. Возврат top_k результатов

    Distance Metric: Cosine distance (<=> operator)
    - distance = 1 - cosine_similarity
    - Lower distance = higher similarity
    """
    from app.config import config

    # 1. Генерация embedding для запроса
    query_embedding = config.embedding_model_instance.embed(query_text)

    # 2. SQL query с pgvector
    result = await session.execute(
        select(Chunk)
        .join(Document)
        .where(
            Document.user_id == user_id,
            Document.search_space_id == search_space_id
        )
        .order_by(Chunk.embedding.op("<=>")(query_embedding))  # Cosine distance
        .limit(top_k)
    )

    return result.scalars().all()
```

**Cosine distance operator** (<=>):
```
distance = 1 - (A · B) / (||A|| * ||B||)

Range: [0, 2]
- 0: identical vectors (perfect match)
- 1: orthogonal (no similarity)
- 2: opposite vectors
```

### 1.6 Full-Text Search

**Функция**: `full_text_search()`
**Местоположение**: `surfsense_backend/app/retriver/chunks_hybrid_search.py:63-113`

```python
async def full_text_search(
    session: AsyncSession,
    query_text: str,
    top_k: int,
    user_id: str,
    search_space_id: int
) -> list[Chunk]:
    """
    Полнотекстовый поиск через PostgreSQL FTS.

    Process:
    1. Конвертация query в tsquery
    2. Поиск по tsvector индексу
    3. Ранжирование по ts_rank_cd
    4. Возврат top_k результатов

    Advantages:
    - Keyword matching (exact terms)
    - Boolean operators support
    - Language-aware stemming
    """
    from sqlalchemy import func

    # 1. Создание tsvector и tsquery
    tsvector = func.to_tsvector('english', Chunk.content)
    tsquery = func.plainto_tsquery('english', query_text)

    # 2. FTS query
    result = await session.execute(
        select(Chunk)
        .join(Document)
        .where(
            Document.user_id == user_id,
            Document.search_space_id == search_space_id,
            tsvector.op('@@')(tsquery)  # Match operator
        )
        .order_by(func.ts_rank_cd(tsvector, tsquery).desc())  # Relevance ranking
        .limit(top_k)
    )

    return result.scalars().all()
```

### 1.7 Hybrid Search с RRF

**Функция**: `hybrid_search()`
**Местоположение**: `surfsense_backend/app/retriver/chunks_hybrid_search.py:115-266`

**Концепция**: Объединение vector search и full-text search через Reciprocal Rank Fusion.

```python
async def hybrid_search(
    session: AsyncSession,
    query_text: str,
    top_k: int,
    user_id: str,
    search_space_id: int,
    alpha: float = 0.5  # Balance between semantic and keyword
) -> list[Chunk]:
    """
    Гибридный поиск с RRF (Reciprocal Rank Fusion).

    Algorithm:
    1. Выполнить semantic search (vector) → ranks
    2. Выполнить keyword search (FTS) → ranks
    3. Compute RRF score for each result
    4. Merge and re-rank by RRF score
    5. Return top_k

    RRF Formula:
    score(chunk) = α * (1/(k + rank_semantic)) + (1-α) * (1/(k + rank_keyword))
    where k = 60 (constant)

    Benefits:
    - Combines semantic understanding with keyword precision
    - Robust to outliers in individual rankings
    - No parameter tuning required (besides α)
    """
    from sqlalchemy import literal, func

    # Constants
    K = 60  # RRF constant

    # Генерация query embedding
    query_embedding = config.embedding_model_instance.embed(query_text)

    # CTE #1: Semantic Search
    semantic_cte = (
        select(
            Chunk.id.label('chunk_id'),
            func.row_number().over(
                order_by=Chunk.embedding.op("<=>")(query_embedding)
            ).label('semantic_rank')
        )
        .join(Document)
        .where(
            Document.user_id == user_id,
            Document.search_space_id == search_space_id
        )
        .limit(top_k * 2)  # Get more results for better fusion
        .cte('semantic_results')
    )

    # CTE #2: Keyword Search
    tsvector = func.to_tsvector('english', Chunk.content)
    tsquery = func.plainto_tsquery('english', query_text)

    keyword_cte = (
        select(
            Chunk.id.label('chunk_id'),
            func.row_number().over(
                order_by=func.ts_rank_cd(tsvector, tsquery).desc()
            ).label('keyword_rank')
        )
        .join(Document)
        .where(
            Document.user_id == user_id,
            Document.search_space_id == search_space_id,
            tsvector.op('@@')(tsquery)
        )
        .limit(top_k * 2)
        .cte('keyword_results')
    )

    # FULL OUTER JOIN + RRF scoring
    rrf_query = (
        select(
            func.coalesce(semantic_cte.c.chunk_id, keyword_cte.c.chunk_id).label('chunk_id'),
            (
                alpha * (1.0 / (K + func.coalesce(semantic_cte.c.semantic_rank, 1000))) +
                (1 - alpha) * (1.0 / (K + func.coalesce(keyword_cte.c.keyword_rank, 1000)))
            ).label('rrf_score')
        )
        .select_from(
            semantic_cte.outerjoin(
                keyword_cte,
                semantic_cte.c.chunk_id == keyword_cte.c.chunk_id,
                full=True
            )
        )
        .order_by(literal_column('rrf_score').desc())
        .limit(top_k)
    )

    # Execute and fetch chunks
    result = await session.execute(rrf_query)
    chunk_ids = [row.chunk_id for row in result]

    # Fetch full chunk objects
    chunks = await session.execute(
        select(Chunk).where(Chunk.id.in_(chunk_ids))
    )

    # Preserve RRF ordering
    chunks_dict = {chunk.id: chunk for chunk in chunks.scalars()}
    ordered_chunks = [chunks_dict[cid] for cid in chunk_ids if cid in chunks_dict]

    return ordered_chunks
```

**Пример RRF вычисления**:
```
Query: "database performance optimization"

Semantic Search Results:        Keyword Search Results:
1. chunk_42 (rank 1)             1. chunk_108 (rank 1)
2. chunk_108 (rank 2)            2. chunk_42 (rank 2)
3. chunk_205 (rank 3)            3. chunk_315 (rank 3)

RRF Scores (k=60, α=0.5):
chunk_42:  0.5*(1/61) + 0.5*(1/62) = 0.0164
chunk_108: 0.5*(1/62) + 0.5*(1/61) = 0.0164
chunk_205: 0.5*(1/63) + 0.5*(1/1060) = 0.0079
chunk_315: 0.5*(1/1060) + 0.5*(1/63) = 0.0079

Merged Ranking:
1. chunk_42 (tie)
2. chunk_108 (tie)
3. chunk_205
4. chunk_315
```

---

## ЧАСТЬ 2: Chunking и Индексация

### 2.1 Chunking конфигурация

**Местоположение**: `surfsense_backend/app/config/__init__.py:178-183`

```python
from chonkie import RecursiveChunker, CodeChunker

# Text chunker
chunker_instance = RecursiveChunker(
    chunk_size=getattr(embedding_model_instance, "max_seq_length", 512)
)

# Code chunker (для программного кода)
code_chunker_instance = CodeChunker(
    chunk_size=getattr(embedding_model_instance, "max_seq_length", 512)
)
```

**Adaptive Chunk Size**:
- Размер chunk адаптируется под embedding модель
- text-embedding-3-small: max_seq_length ≈ 512 tokens
- text-embedding-3-large: max_seq_length ≈ 512 tokens
- Для длинных контекстов можно увеличить

### 2.2 Процесс Chunking

**Функция**: `create_document_chunks()`
**Местоположение**: `surfsense_backend/app/utils/document_converters.py:148-164`

```python
async def create_document_chunks(
    content: str,
    document_id: int | None = None
) -> list[Chunk]:
    """
    Создает chunks из контента документа с embeddings.

    Strategy:
    1. Recursive chunking с сохранением структуры
    2. Overlap между chunks для context continuity
    3. Adaptive sizing по embedding model capacity
    4. Embedding generation для каждого chunk

    Процесс:
    1. Chunking контента
    2. Генерация embedding для каждого chunk
    3. Создание Chunk объектов
    4. Связывание с parent Document
    """
    from app.config import config

    # 1. Chunking
    chunks = config.chunker_instance.chunk(content)

    # 2. Создание Chunk объектов с embeddings
    chunk_objects = []

    for chunk in chunks:
        # Генерация embedding
        chunk_embedding = config.embedding_model_instance.embed(chunk.text)

        # Создание объекта
        chunk_obj = Chunk(
            content=chunk.text,
            embedding=chunk_embedding,
            document_id=document_id
        )

        chunk_objects.append(chunk_obj)

    return chunk_objects
```

### 2.3 RecursiveChunker (стратегия разбиения)

**Алгоритм**:
```python
def recursive_chunk(text: str, chunk_size: int) -> list[str]:
    """
    Рекурсивное разбиение текста с сохранением структуры.

    Приоритет разделителей (от высшего к низшему):
    1. "\n\n" (параграфы)
    2. "\n" (строки)
    3. ". " (предложения)
    4. ", " (запятые)
    5. " " (слова)

    Преимущества:
    - Сохраняет смысловые границы
    - Минимизирует разрывы концепций
    - Адаптивный размер с учетом контекста
    """
    separators = ["\n\n", "\n", ". ", ", ", " "]

    for separator in separators:
        if separator in text:
            parts = text.split(separator)
            chunks = []
            current_chunk = ""

            for part in parts:
                if len(current_chunk) + len(part) < chunk_size:
                    current_chunk += part + separator
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = part + separator

            if current_chunk:
                chunks.append(current_chunk)

            return chunks

    # Fallback: character-level splitting
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
```

### 2.4 CodeChunker (для программного кода)

**Специализация для кода**:
```python
class CodeChunker:
    """
    Chunker оптимизированный для программного кода.

    Features:
    - Function/class boundary awareness
    - Сохранение синтаксической структуры
    - Comment preservation
    - Import/dependency tracking
    """

    def chunk(self, code: str, language: str = "python") -> list[str]:
        # Парсинг AST для определения границ функций/классов
        # Разбиение по логическим единицам (functions, classes, modules)
        # Сохранение imports и docstrings
        ...
```

### 2.5 Индексация в БД

**Process**:
```python
# После обработки документа и chunking
async with session.begin():
    # 1. Сохранение Document
    document = Document(
        title=title,
        content=content,
        embedding=summary_embedding,
        document_type=DocumentType.FILE,
        user_id=user_id,
        search_space_id=search_space_id
    )
    session.add(document)
    await session.flush()  # Получаем document.id

    # 2. Создание и сохранение chunks
    chunks = await create_document_chunks(content, document.id)
    for chunk in chunks:
        session.add(chunk)

    await session.commit()
```

---

## ЧАСТЬ 3: Connector Integration (Интеграция источников)

### 3.1 Поддерживаемые Connectors

**Директория**: `surfsense_backend/app/tasks/connector_indexers/`

| Connector | Тип данных | Метаданные |
|-----------|------------|------------|
| **Slack** | Сообщения, треды | channel, author, timestamp, reactions |
| **GitHub** | Issues, PRs, код | repo, author, labels, comments |
| **Notion** | Страницы, базы данных | workspace, author, last_edited |
| **Jira** | Tasks, issues | project, assignee, status, priority |
| **Confluence** | Wiki страницы | space, author, version |
| **Discord** | Сообщения, каналы | server, channel, author |
| **Google Calendar** | События | organizer, attendees, time |
| **Gmail** | Emails | sender, recipients, subject, date |
| **Linear** | Issues, projects | team, assignee, status |
| **ClickUp** | Tasks, docs | space, assignee, due_date |
| **Airtable** | Records, tables | base, table, fields |

### 3.2 Base Indexer функции

**Файл**: `surfsense_backend/app/tasks/connector_indexers/base.py`

```python
async def check_duplicate_document_by_hash(
    session: AsyncSession,
    content_hash: str,
    user_id: str,
    search_space_id: int
) -> Document | None:
    """Проверяет дубликаты по content_hash"""
    ...

async def check_document_by_unique_identifier(
    session: AsyncSession,
    unique_identifier_hash: str,
    user_id: str,
    search_space_id: int
) -> Document | None:
    """Проверяет по источник-специфичному ID"""
    ...

async def update_connector_last_indexed(
    session: AsyncSession,
    connector: Connector
) -> None:
    """Обновляет метку последней индексации"""
    ...
```

### 3.3 Пример: Slack Indexer

**Файл**: `surfsense_backend/app/tasks/connector_indexers/slack_indexer.py`

```python
async def index_slack_messages(
    session: AsyncSession,
    connector_id: int,
    search_space_id: int,
    user_id: str,
    start_date: str | None = None,
    end_date: str | None = None
) -> tuple[int, str | None]:
    """
    Индексирует Slack сообщения.

    Process:
    1. Получение Slack connector config
    2. Инициализация Slack client
    3. Получение списка каналов
    4. Для каждого канала:
       - Fetch messages в диапазоне дат
       - Проверка дубликатов (по message_id)
       - Форматирование в Markdown
       - Генерация summary + embedding
       - Chunking + chunk embeddings
       - Сохранение в БД
    5. Обновление last_indexed_at

    Metadata:
    {
        "channel": "engineering",
        "channel_id": "C123456",
        "author": "user@example.com",
        "author_id": "U789012",
        "timestamp": "2024-03-15T10:30:00Z",
        "message_url": "https://workspace.slack.com/archives/C123456/p1234567890",
        "has_reactions": true,
        "reactions": ["👍": 5, "🎉": 2],
        "thread_ts": null  # or timestamp if part of thread
    }
    """
    from slack_sdk import WebClient
    import hashlib

    # 1. Get connector
    connector = await get_connector_by_id(session, connector_id, ConnectorType.SLACK)
    slack_token = connector.credentials['access_token']
    slack_client = WebClient(token=slack_token)

    # 2. Calculate date range
    start_ts, end_ts = calculate_date_range(connector, start_date, end_date)

    # 3. Get channels
    channels = slack_client.conversations_list()['channels']

    indexed_count = 0

    for channel in channels:
        channel_id = channel['id']
        channel_name = channel['name']

        # 4. Fetch messages
        messages = slack_client.conversations_history(
            channel=channel_id,
            oldest=start_ts,
            latest=end_ts
        )['messages']

        for message in messages:
            # Unique identifier для дедупликации
            unique_id = f"slack:{channel_id}:{message['ts']}"
            unique_id_hash = hashlib.sha256(unique_id.encode()).hexdigest()

            # Проверка дубликатов
            existing = await check_document_by_unique_identifier(
                session, unique_id_hash, user_id, search_space_id
            )
            if existing:
                continue  # Уже индексировано

            # Форматирование message
            content = format_slack_message_to_markdown(message, channel_name)

            # Metadata
            metadata = {
                "channel": channel_name,
                "channel_id": channel_id,
                "author": message.get('user'),
                "timestamp": message['ts'],
                "message_url": f"https://{connector.workspace}.slack.com/archives/{channel_id}/p{message['ts'].replace('.', '')}",
                "reactions": extract_reactions(message)
            }

            # Content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Генерация summary + embedding
            summary, embedding = await generate_document_summary(
                content=content,
                user_llm=get_user_llm(session, user_id),
                document_metadata=metadata
            )

            # Создание Document
            document = Document(
                title=f"Slack: {channel_name} - {message.get('user')}",
                content=content,
                document_type=DocumentType.SLACK,
                document_metadata=metadata,
                content_hash=content_hash,
                unique_identifier_hash=unique_id_hash,
                embedding=embedding,
                user_id=user_id,
                search_space_id=search_space_id,
                connector_id=connector_id
            )
            session.add(document)
            await session.flush()

            # Chunking
            chunks = await create_document_chunks(content, document.id)
            for chunk in chunks:
                session.add(chunk)

            indexed_count += 1

    # 5. Update last_indexed
    await update_connector_last_indexed(session, connector)
    await session.commit()

    return indexed_count, None
```

### 3.4 Connector Service (поиск по источникам)

**Файл**: `surfsense_backend/app/services/connector_service.py`

```python
class ConnectorService:
    async def search_all_sources(
        self,
        user_query: str,
        user_id: str,
        search_space_id: int,
        top_k: int = 20,
        search_mode: SearchMode = SearchMode.CHUNKS
    ) -> SearchResults:
        """
        Поиск по всем источникам (files, urls, connectors).

        Returns:
            SearchResults with documents/chunks and metadata
        """
        # Hybrid search
        results = await hybrid_search(
            session=self.session,
            query_text=user_query,
            top_k=top_k,
            user_id=user_id,
            search_space_id=search_space_id
        )

        # Group by source type
        grouped_results = self.group_by_source_type(results)

        return SearchResults(
            documents=results,
            sources=grouped_results,
            total_count=len(results)
        )
```

---

## Резюме: Полный Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│               COMPLETE INDEXING & SEARCH PIPELINE           │
└─────────────────────────────────────────────────────────────┘

SOURCE INGESTION
────────────────
  Files / URLs / Connectors → ETL Processing → Markdown

           ↓

CONCEPTUAL EXTRACTION
─────────────────────
  LLM Summarization → Conceptual Summary + Metadata

           ↓

EMBEDDING GENERATION
────────────────────
  Document Embedding (summary)
  + Chunk Embeddings (content)

           ↓

CHUNKING
────────
  RecursiveChunker / CodeChunker → Semantic chunks

           ↓

DATABASE STORAGE
────────────────
  PostgreSQL + pgvector
  - Document table (with embedding)
  - Chunk table (with embeddings)
  - Indexes for fast retrieval

           ↓

SEARCH & RETRIEVAL
──────────────────
  Query → Embedding → Hybrid Search (Vector + FTS + RRF)
                   ↓
              Reranking (Cohere, Pinecone)
                   ↓
              Top-K Results

           ↓

AI AGENT PROCESSING
───────────────────
  Results → Q&A Agent → Answer with Citations
```

## Ключевые компоненты и файлы

| Компонент | Файл | Ключевые функции |
|-----------|------|------------------|
| **Embeddings** | `config/__init__.py` | `embedding_model_instance` |
| **Vector Search** | `retriver/chunks_hybrid_search.py` | `vector_search()`, `hybrid_search()` |
| **Chunking** | `utils/document_converters.py` | `create_document_chunks()` |
| **Indexers** | `tasks/connector_indexers/*` | Connector-specific indexing |
| **Search Service** | `services/connector_service.py` | `search_all_sources()` |

---

## Оптимизация производительности

### Индексирование
- **Batch processing**: Индексация документов батчами для снижения overhead
- **Async operations**: Использование async/await для параллельной обработки
- **Caching**: Кеширование embeddings для повторяющихся chunks

### Поиск
- **pgvector IVFFlat**: Approximate nearest neighbor для O(log N) вместо O(N)
- **Hybrid RRF**: Оптимальный баланс precision/recall
- **Reranking**: Улучшение качества результатов на 15-30%
