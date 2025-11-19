# Семантические архитектурные паттерны

## Введение

Семантические паттерны в SurfSense обеспечивают интеллектуальный поиск и обработку информации на основе семантического значения, а не только ключевых слов. Эти паттерны используют embeddings, векторный поиск и AI модели для понимания смысла запросов и документов.

## 1. RAG Pattern (Retrieval-Augmented Generation)

### Описание

RAG (Retrieval-Augmented Generation) - это паттерн, который комбинирует поиск информации (Retrieval) с генерацией текста (Generation), используя найденную информацию как контекст для LLM.

### Архитектура RAG

```
┌──────────────────────────────────────────────────────┐
│                  RAG PIPELINE                        │
└──────────────────────────────────────────────────────┘

   User Query
       │
       ▼
┌─────────────────┐
│  1. RETRIEVAL   │  ← Hybrid Search (Vector + FTS + RRF)
└────────┬────────┘
         │ Retrieved Documents
         ▼
┌─────────────────┐
│ 2. AUGMENTATION │  ← Context preparation + Token optimization
└────────┬────────┘
         │ Augmented Context
         ▼
┌─────────────────┐
│  3. GENERATION  │  ← LLM with context → Answer
└────────┬────────┘
         │ Generated Answer
         ▼
    Response with
     Citations
```

### Реализация в SurfSense

**Файл**: `surfsense_backend/app/agents/researcher/nodes.py`

#### Phase 1: Retrieval

```python
async def handle_qna_workflow(
    state: State,
    config: RunnableConfig,
    writer: StreamWriter
) -> dict:
    """
    RAG Pattern implementation в Q&A workflow.
    """
    configuration = Configuration.from_runnable_config(config)
    streaming_service = state.streaming_service

    # ═══════════════════════════════════════════════
    # PHASE 1: RETRIEVAL
    # ═══════════════════════════════════════════════

    # 1.1. Переформулированный запрос из предыдущего узла
    user_query = state.reformulated_query

    # 1.2. Fetch relevant documents using hybrid search
    writer({
        "yield_value": streaming_service.format_terminal_info_delta(
            "🔎 Searching for relevant documents..."
        )
    })

    # Create connector service для поиска
    connector_service = ConnectorService(
        state.db_session,
        user_id=configuration.user_id
    )
    await connector_service.initialize_counter()

    # Выполнить поиск по выбранным источникам
    relevant_documents = await fetch_relevant_documents(
        research_questions=[user_query],
        user_id=configuration.user_id,
        search_space_id=configuration.search_space_id,
        db_session=state.db_session,
        connectors_to_search=configuration.connectors_to_search,
        writer=writer,
        state=state,
        top_k=20,  # Retrieve top 20 initially
        connector_service=connector_service,
        search_mode=SearchMode.CHUNKS
    )

    writer({
        "yield_value": streaming_service.format_terminal_info_delta(
            f"✅ Found {len(relevant_documents)} relevant documents",
            message_type="success"
        )
    })

    # 1.3. Combine with user-selected documents (if any)
    user_selected_documents = configuration.user_selected_documents or []
    all_documents = user_selected_documents + relevant_documents

    # ═══════════════════════════════════════════════
    # PHASE 2: AUGMENTATION & GENERATION
    # ═══════════════════════════════════════════════

    # 2.1. Pass to Q&A SubAgent for augmentation and generation
    qna_agent_graph = build_qna_graph()

    qna_state = {
        "user_query": user_query,
        "relevant_documents": all_documents,  # AUGMENTATION: provide context
        "user_id": configuration.user_id,
        "search_space_id": configuration.search_space_id,
        "db_session": state.db_session,
        "streaming_service": streaming_service
    }

    qna_config = {
        "configurable": {
            "user_query": user_query,
            "relevant_documents": all_documents,
            "user_id": configuration.user_id,
            "search_space_id": configuration.search_space_id
        }
    }

    # 2.2. Execute Q&A SubAgent (handles augmentation and generation)
    complete_content = ""
    sources = []

    async for chunk_type, chunk in qna_agent_graph.astream(qna_state, qna_config):
        if chunk_type == "rerank_documents":
            # Reranked documents ready
            writer({
                "yield_value": streaming_service.format_terminal_info_delta(
                    "📊 Reranked documents by relevance"
                )
            })

        elif chunk_type == "answer_question":
            if "delta" in chunk:
                # Stream generated answer chunks
                writer({
                    "yield_value": streaming_service.format_text_chunk(
                        chunk["delta"]
                    )
                })
                complete_content += chunk["delta"]

            if "final_answer" in chunk:
                complete_content = chunk["final_answer"]

            if "sources" in chunk:
                sources = chunk["sources"]

    # Stream sources
    if sources:
        writer({
            "yield_value": streaming_service.format_sources_delta(sources)
        })

    return {"final_written_report": complete_content}
```

#### Phase 2: Augmentation (в Q&A SubAgent)

**Файл**: `surfsense_backend/app/agents/researcher/qna_agent/nodes.py`

```python
async def answer_question(
    state: QnAState,
    config: RunnableConfig,
    writer: StreamWriter
) -> dict:
    """
    RAG: Augmentation + Generation.
    """
    configuration = Configuration.from_runnable_config(config)

    # ═══════════════════════════════════════════════
    # PHASE 2: AUGMENTATION
    # ═══════════════════════════════════════════════

    # 2.1. Get reranked documents from previous node
    reranked_documents = state.reranked_documents  # Top 10 after reranking

    # 2.2. Optimize documents for token limit
    fast_llm = await get_user_fast_llm(
        state.db_session,
        configuration.user_id,
        configuration.search_space_id
    )

    # Get model's context window
    model_info = litellm.get_model_info(fast_llm.model)
    max_context_tokens = model_info.get("max_input_tokens", 8192)

    # Reserve tokens for prompt and output
    PROMPT_TOKENS = 500
    OUTPUT_TOKENS = 2000
    available_tokens = max_context_tokens - PROMPT_TOKENS - OUTPUT_TOKENS

    # 2.3. Format documents with token optimization
    formatted_context = ""
    sources_metadata = []
    current_tokens = 0

    for idx, doc in enumerate(reranked_documents, 1):
        doc_text = f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        doc_text += f"[{idx}] {doc['document']['title']}\n"
        doc_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        doc_text += doc["content"]

        # Count tokens
        doc_tokens = litellm.token_counter(
            model=fast_llm.model,
            text=doc_text
        )

        if current_tokens + doc_tokens > available_tokens:
            # Truncate document to fit
            remaining_tokens = available_tokens - current_tokens
            if remaining_tokens > 100:  # Minimum viable chunk
                truncated = truncate_to_tokens(doc_text, remaining_tokens)
                formatted_context += truncated
            break

        formatted_context += doc_text
        current_tokens += doc_tokens

        # Track source for citations
        sources_metadata.append({
            "citation_number": idx,
            "document_id": doc["document"]["id"],
            "chunk_id": doc["chunk_id"],
            "title": doc["document"]["title"],
            "url": doc["document"].get("url"),
            "type": doc["document"]["document_type"]
        })

    # ═══════════════════════════════════════════════
    # PHASE 3: GENERATION
    # ═══════════════════════════════════════════════

    # 3.1. Build QA prompt with augmented context
    from app.prompts import QA_PROMPT_TEMPLATE

    qa_prompt = QA_PROMPT_TEMPLATE.format(
        user_query=state.user_query,
        context=formatted_context  # AUGMENTED CONTEXT
    )

    # 3.2. Generate answer with LLM
    complete_answer = ""

    async for chunk in fast_llm.astream(qa_prompt, temperature=0.3):
        delta = chunk.content

        # Stream to user
        writer({
            "yield_value": state.streaming_service.format_text_chunk(delta)
        })

        complete_answer += delta

    # 3.3. Extract citations from answer
    cited_sources = extract_citations_from_answer(
        complete_answer,
        sources_metadata
    )

    return {
        "final_answer": complete_answer,
        "sources": cited_sources
    }
```

### QA Prompt Template

**Файл**: `surfsense_backend/app/prompts/__init__.py`

```python
QA_PROMPT_TEMPLATE = """
You are a knowledgeable research assistant. Answer the user's question based on the provided context documents.

## Context Documents
{context}

## User's Question
{user_query}

## Instructions
1. **Answer Accuracy**: Base your answer strictly on the provided context
2. **Citations**: Cite your sources using [1], [2], [3] format after relevant statements
3. **Comprehensiveness**: Provide a detailed and complete answer
4. **Structure**: Use markdown formatting (headers, lists, code blocks) for clarity
5. **Objectivity**: Present information objectively without personal opinions
6. **Limitations**: If the context doesn't contain sufficient information, acknowledge it

## Output Format
Provide a well-structured answer with:
- Clear introduction
- Main content with appropriate citations [N]
- Conclusion or summary if applicable
- Code examples if relevant (with proper syntax highlighting)

Answer:
"""
```

### Преимущества RAG Pattern

1. **Актуальность**: Использует актуальную информацию из базы знаний
2. **Точность**: Ответы основаны на фактах из документов
3. **Прозрачность**: Цитаты позволяют проверить источники
4. **Контекстуальность**: LLM получает специфичный контекст для запроса
5. **Масштабируемость**: Не требует переобучения LLM при добавлении новых данных

---

## 2. Embedding Pipeline Pattern

### Описание

Embedding Pipeline Pattern описывает процесс конвертации текста в векторное представление (embeddings) для семантического поиска.

### Архитектура Pipeline

```
┌─────────────────────────────────────────────────────┐
│              EMBEDDING PIPELINE                     │
└─────────────────────────────────────────────────────┘

   Raw Text
      │
      ▼
┌──────────────┐
│ Preprocessing│  ← Normalization, cleaning
└──────┬───────┘
       │ Normalized Text
       ▼
┌──────────────┐
│ Tokenization │  ← Split into tokens
└──────┬───────┘
       │ Tokens
       ▼
┌──────────────┐
│ Embedding    │  ← text-embedding-3-small/large
│ Model        │
└──────┬───────┘
       │ Vector[1536/3072]
       ▼
┌──────────────┐
│ Storage      │  ← PostgreSQL + pgvector
└──────────────┘
```

### Реализация в SurfSense

**Файл**: `surfsense_backend/app/retriver/chunks_hybrid_search.py`

#### Vector Search с Embedding Pipeline

```python
from app.config import config

async def vector_search(
    self,
    query_text: str,
    top_k: int,
    user_id: str,
    search_space_id: int
) -> list:
    """
    Embedding Pipeline для векторного поиска.

    Pipeline:
    1. Query text → Embedding model → Query vector
    2. Query vector → PostgreSQL pgvector → Similar vectors
    3. Similar vectors → Documents/Chunks
    """

    # ═══════════════════════════════════════════════
    # STEP 1: EMBEDDING GENERATION
    # ═══════════════════════════════════════════════

    # Get embedding model instance from global config
    embedding_model = config.embedding_model_instance

    # Convert query text to vector
    query_embedding = embedding_model.embed(query_text)
    # Output: list[float] with length 1536 or 3072

    # ═══════════════════════════════════════════════
    # STEP 2: VECTOR SIMILARITY SEARCH
    # ═══════════════════════════════════════════════

    # Build SQL query with pgvector operators
    from sqlalchemy import select
    from app.db import Chunk, Document, SearchSpace

    query = (
        select(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .join(SearchSpace, Document.search_space_id == SearchSpace.id)
        .where(
            SearchSpace.user_id == user_id,
            SearchSpace.id == search_space_id
        )
        # pgvector: cosine distance operator (<=>)
        .order_by(Chunk.embedding.op("<=>")(query_embedding))
        .limit(top_k)
    )

    # Execute query
    result = await self.db_session.execute(query)
    chunks = result.scalars().all()

    return chunks
```

#### Embedding Model Configuration

**Файл**: `surfsense_backend/app/config/__init__.py`

```python
from app.embeddings.auto_embeddings import AutoEmbeddings
import os

class Config:
    """Global configuration including embedding model"""

    # Embedding model selection
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Azure OpenAI configuration (if using Azure)
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

    # OpenAI configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Initialize embedding model instance
    embedding_model_instance = AutoEmbeddings.get_embeddings(
        EMBEDDING_MODEL,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_api_key=AZURE_API_KEY,
        openai_api_key=OPENAI_API_KEY
    )

    # Get embedding dimension from model
    EMBEDDING_DIMENSION = getattr(
        embedding_model_instance,
        "dimension",
        1536  # Default for text-embedding-3-small
    )
```

#### Document Embedding (при индексации)

**Файл**: `surfsense_backend/app/utils/document_converters.py`

```python
async def generate_document_summary(
    content: str,
    user_llm: Any,
    document_metadata: dict | None = None
) -> tuple[str, list[float]]:
    """
    Generate summary and embedding for document.

    Embedding Pipeline:
    1. Content → LLM → Summary
    2. Summary + Metadata → Enhanced summary
    3. Enhanced summary → Embedding model → Vector
    """
    from app.config import config

    # Step 1: Generate summary via LLM
    optimized_content = optimize_content_for_context_window(
        content,
        document_metadata,
        user_llm.model_name
    )

    from app.prompts import SUMMARY_PROMPT_TEMPLATE
    summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
        metadata=format_metadata(document_metadata),
        content=optimized_content
    )

    summary_response = await user_llm.ainvoke(summary_prompt)
    summary_content = summary_response.content

    # Step 2: Enhance summary with metadata
    metadata_markdown = format_metadata_as_markdown(document_metadata)
    enhanced_summary = f"{metadata_markdown}\n\n{summary_content}"

    # Step 3: Generate embedding
    # EMBEDDING PIPELINE: Text → Vector
    embedding_model = config.embedding_model_instance
    summary_embedding = embedding_model.embed(enhanced_summary)

    return summary_content, summary_embedding
```

#### Chunk Embedding

**Файл**: `surfsense_backend/app/utils/document_converters.py`

```python
async def create_document_chunks(
    content: str,
    document_id: int | None = None
) -> list:
    """
    Create chunks with embeddings.

    Embedding Pipeline for each chunk:
    1. Content → Chunker → Chunks
    2. Each chunk → Embedding model → Chunk vector
    """
    from app.config import config

    # Step 1: Chunking
    chunker = config.chunker_instance
    chunks = chunker.chunk(content)

    # Step 2: Embed each chunk
    chunk_objects = []

    for chunk in chunks:
        # EMBEDDING PIPELINE: Chunk text → Vector
        chunk_embedding = config.embedding_model_instance.embed(chunk.text)

        chunk_obj = Chunk(
            content=chunk.text,
            embedding=chunk_embedding,  # Vector[1536/3072]
            document_id=document_id
        )

        chunk_objects.append(chunk_obj)

    return chunk_objects
```

### Поддерживаемые Embedding модели

| Модель | Размерность | Провайдер | Использование |
|--------|-------------|-----------|---------------|
| **text-embedding-3-small** | 1536 | OpenAI | Баланс скорость/качество |
| **text-embedding-3-large** | 3072 | OpenAI | Максимальное качество |
| **text-embedding-ada-002** | 1536 | OpenAI | Legacy, стабильная |
| **Azure embeddings** | 1536/3072 | Azure OpenAI | Enterprise окружения |

### Преимущества Embedding Pipeline

1. **Семантическое понимание**: Векторы кодируют смысл, не только слова
2. **Многоязычность**: Работает с разными языками
3. **Масштабируемость**: Один раз генерируем, многократно используем
4. **Эффективность**: pgvector IVFFlat индексы для быстрого поиска
5. **Гибкость**: Легко сменить модель embeddings

---

## 3. Hybrid Search Pattern

### Описание

Hybrid Search Pattern комбинирует несколько методов поиска для достижения лучших результатов:
- **Vector Search**: семантическое понимание
- **Full-Text Search**: точное совпадение ключевых слов
- **RRF (Reciprocal Rank Fusion)**: объединение результатов

### Архитектура Hybrid Search

```
┌──────────────────────────────────────────────────────┐
│              HYBRID SEARCH PATTERN                   │
└──────────────────────────────────────────────────────┘

        Query Text
            │
            ├──────────────────┬──────────────────┐
            ▼                  ▼                  ▼
    ┌──────────────┐   ┌──────────────┐   Optional:
    │ Vector Search│   │ Full-Text    │   Filters
    │ (Semantic)   │   │ Search (FTS) │   (metadata,
    │              │   │              │    date, etc.)
    │ embed(query) │   │ to_tsquery   │
    │ cosine <=>   │   │ ts_rank_cd   │
    └──────┬───────┘   └──────┬───────┘
           │                  │
           │ Results (ranked) │ Results (ranked)
           │                  │
           └──────────┬───────┘
                      ▼
            ┌─────────────────┐
            │ Reciprocal Rank │
            │ Fusion (RRF)    │
            │                 │
            │ score = 1/(k+r₁)│
            │       + 1/(k+r₂)│
            └────────┬────────┘
                     │
                     ▼
            Merged & Ranked
             Results (top_k)
```

### Реализация в SurfSense

**Файл**: `surfsense_backend/app/retriver/chunks_hybrid_search.py`

#### Hybrid Search с RRF

```python
async def hybrid_search(
    self,
    query_text: str,
    top_k: int,
    user_id: str,
    search_space_id: int,
    document_type: str | None = None
) -> list:
    """
    Hybrid Search: Vector + Full-Text + RRF.

    Algorithm:
    1. Perform semantic search (vector) → ranked results
    2. Perform keyword search (FTS) → ranked results
    3. Apply RRF (Reciprocal Rank Fusion) → merged ranking
    4. Return top_k results
    """
    from sqlalchemy import select, func, literal, text
    from app.config import config

    # Configuration
    K = 60  # RRF constant
    n_results = top_k * 2  # Get more for better fusion

    # Get query embedding for semantic search
    embedding_model = config.embedding_model_instance
    query_embedding = embedding_model.embed(query_text)

    # ═══════════════════════════════════════════════
    # CTE 1: SEMANTIC SEARCH (Vector Similarity)
    # ═══════════════════════════════════════════════

    semantic_search_cte = (
        select(
            Chunk.id.label('chunk_id'),
            # Assign rank based on similarity
            func.row_number()
            .over(order_by=Chunk.embedding.op("<=>")(query_embedding))
            .label('semantic_rank')
        )
        .join(Document, Chunk.document_id == Document.id)
        .join(SearchSpace, Document.search_space_id == SearchSpace.id)
        .where(
            SearchSpace.user_id == user_id,
            SearchSpace.id == search_space_id
        )
        .order_by(Chunk.embedding.op("<=>")(query_embedding))
        .limit(n_results)
        .cte('semantic_search')
    )

    # ═══════════════════════════════════════════════
    # CTE 2: KEYWORD SEARCH (Full-Text Search)
    # ═══════════════════════════════════════════════

    # Build tsvector and tsquery
    tsvector = func.to_tsvector('english', Chunk.content)
    tsquery = func.plainto_tsquery('english', query_text)

    keyword_search_cte = (
        select(
            Chunk.id.label('chunk_id'),
            # Assign rank based on FTS relevance
            func.row_number()
            .over(order_by=func.ts_rank_cd(tsvector, tsquery).desc())
            .label('keyword_rank')
        )
        .join(Document, Chunk.document_id == Document.id)
        .join(SearchSpace, Document.search_space_id == SearchSpace.id)
        .where(
            SearchSpace.user_id == user_id,
            SearchSpace.id == search_space_id,
            tsvector.op('@@')(tsquery)  # Full-text match operator
        )
        .order_by(func.ts_rank_cd(tsvector, tsquery).desc())
        .limit(n_results)
        .cte('keyword_search')
    )

    # ═══════════════════════════════════════════════
    # RRF: RECIPROCAL RANK FUSION
    # ═══════════════════════════════════════════════

    # FULL OUTER JOIN + RRF score calculation
    rrf_query = (
        select(
            # Get chunk_id from either CTE
            func.coalesce(
                semantic_search_cte.c.chunk_id,
                keyword_search_cte.c.chunk_id
            ).label('chunk_id'),

            # RRF Score Formula:
            # score = 1/(k + rank_semantic) + 1/(k + rank_keyword)
            # If a result appears in only one method, the missing rank = 1000 (low score)
            (
                1.0 / (K + func.coalesce(semantic_search_cte.c.semantic_rank, 1000)) +
                1.0 / (K + func.coalesce(keyword_search_cte.c.keyword_rank, 1000))
            ).label('rrf_score')
        )
        .select_from(
            semantic_search_cte.outerjoin(
                keyword_search_cte,
                semantic_search_cte.c.chunk_id == keyword_search_cte.c.chunk_id,
                full=True  # FULL OUTER JOIN
            )
        )
        .order_by(text('rrf_score DESC'))
        .limit(top_k)
    )

    # Execute RRF query
    result = await self.db_session.execute(rrf_query)
    rrf_results = result.fetchall()

    # ═══════════════════════════════════════════════
    # FETCH FULL CHUNK OBJECTS
    # ═══════════════════════════════════════════════

    chunk_ids = [row.chunk_id for row in rrf_results]
    rrf_scores = {row.chunk_id: row.rrf_score for row in rrf_results}

    # Fetch full chunk data
    chunks_query = (
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.id.in_(chunk_ids))
    )

    chunks_result = await self.db_session.execute(chunks_query)
    chunks = chunks_result.all()

    # ═══════════════════════════════════════════════
    # FORMAT RESULTS WITH SCORES
    # ═══════════════════════════════════════════════

    serialized_results = []

    for chunk, document in chunks:
        serialized_results.append({
            "chunk_id": chunk.id,
            "content": chunk.content,
            "score": rrf_scores.get(chunk.id, 0.0),  # RRF score
            "document": {
                "id": document.id,
                "title": document.title,
                "document_type": document.document_type.value,
                "url": document.document_metadata.get("url"),
                # ... other metadata
            }
        })

    # Sort by RRF score (descending) and preserve order
    serialized_results.sort(key=lambda x: x["score"], reverse=True)

    return serialized_results[:top_k]
```

### RRF (Reciprocal Rank Fusion) Formula

```
For each document d:

score(d) = 1/(k + rank_semantic(d)) + 1/(k + rank_keyword(d))

where:
- k = 60 (constant, reduces impact of high ranks)
- rank_semantic(d) = rank in vector search results (1-indexed)
- rank_keyword(d) = rank in full-text search results (1-indexed)
- If d not in one of the methods: rank = 1000 (very low contribution)
```

**Пример**:
```
Query: "database performance optimization"

Semantic Search Results:       Keyword Search Results:
1. chunk_42 (rank 1)            1. chunk_108 (rank 1)
2. chunk_108 (rank 2)           2. chunk_42 (rank 2)
3. chunk_205 (rank 3)           3. chunk_315 (rank 3)
4. chunk_99 (rank 4)            [chunk_205 not in results]

RRF Scores (k=60):
chunk_42:  1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
chunk_108: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
chunk_205: 1/(60+3) + 1/(60+1000) = 0.0159 + 0.0009 = 0.0168
chunk_315: 1/(60+1000) + 1/(60+3) = 0.0009 + 0.0159 = 0.0168
chunk_99:  1/(60+4) + 1/(60+1000) = 0.0156 + 0.0009 = 0.0165

Final Ranking:
1. chunk_42 (0.0325) - appears high in both
2. chunk_108 (0.0325) - appears high in both
3. chunk_205 (0.0168) - strong in semantic only
4. chunk_315 (0.0168) - strong in keyword only
5. chunk_99 (0.0165) - moderate in semantic
```

### Преимущества Hybrid Search

1. **Best of both worlds**: Семантика + точное совпадение
2. **Robustness**: Компенсирует слабости каждого метода
3. **No parameter tuning**: RRF не требует настройки весов
4. **Domain agnostic**: Работает для разных типов запросов
5. **Proven effectiveness**: RRF показывает лучшие результаты в академических исследованиях

---

## 4. Reranking Pattern

### Описание

Reranking Pattern улучшает результаты поиска, используя более сложную модель для переранжирования первичных результатов.

### Архитектура

```
┌──────────────────────────────────────────────────────┐
│              RERANKING PATTERN                       │
└──────────────────────────────────────────────────────┘

    Query + Initial Search Results (top 20)
                │
                ▼
        ┌───────────────┐
        │ First-stage   │  ← Hybrid Search (fast)
        │ Retrieval     │
        └───────┬───────┘
                │ Top 20 candidates
                ▼
        ┌───────────────┐
        │ Reranking     │  ← Cross-encoder model (slow but accurate)
        │ Model         │     - Cohere rerank
        │               │     - Pinecone rerank
        │               │     - Custom models
        └───────┬───────┘
                │ Reranked top 10
                ▼
        Better ranked results
```

### Реализация в SurfSense

**Файл**: `surfsense_backend/app/services/reranker_service.py`

#### RerankerService

```python
from rerankers import Reranker
from typing import Optional

class RerankerService:
    """
    Service for reranking search results using cross-encoder models.
    """

    def __init__(self, reranker_instance=None):
        self.reranker_instance = reranker_instance

    def rerank_documents(
        self,
        query_text: str,
        documents: list[dict]
    ) -> list[dict]:
        """
        Rerank documents using cross-encoder model.

        Process:
        1. Convert documents to reranker format
        2. Call reranker model with query and documents
        3. Update scores with reranker scores
        4. Return reranked results

        Args:
            query_text: User query
            documents: List of documents with initial scores (from hybrid search)

        Returns:
            Reranked documents with updated scores
        """
        if not self.reranker_instance or not documents:
            return documents  # Fallback to original

        try:
            # ═══════════════════════════════════════════════
            # STEP 1: CONVERT TO RERANKER FORMAT
            # ═══════════════════════════════════════════════

            reranker_docs = []

            for doc in documents:
                from rerankers.documents import RerankerDocument

                reranker_docs.append(
                    RerankerDocument(
                        text=doc.get("content", ""),
                        doc_id=doc.get("chunk_id"),
                        metadata={
                            "document_id": doc.get("document", {}).get("id"),
                            "original_score": doc.get("score", 0.0),  # RRF score
                            "title": doc.get("document", {}).get("title")
                        }
                    )
                )

            # ═══════════════════════════════════════════════
            # STEP 2: CALL RERANKER MODEL
            # ═══════════════════════════════════════════════

            reranking_results = self.reranker_instance.rank(
                query=query_text,
                docs=reranker_docs
            )

            # ═══════════════════════════════════════════════
            # STEP 3: UPDATE SCORES
            # ═══════════════════════════════════════════════

            serialized_results = []

            for result in reranking_results.results:
                # Find original document
                original_doc = next(
                    (d for d in documents if d["chunk_id"] == result.document.doc_id),
                    None
                )

                if original_doc:
                    # Create new document with reranker score
                    reranked_doc = original_doc.copy()

                    # UPDATE: Replace hybrid search score with reranker score
                    reranked_doc["score"] = float(result.score)
                    reranked_doc["rank"] = result.rank
                    reranked_doc["original_score"] = original_doc.get("score", 0.0)

                    serialized_results.append(reranked_doc)

            return serialized_results

        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            return documents  # Fallback to original on error

    @staticmethod
    def get_reranker_instance() -> Optional["RerankerService"]:
        """
        Get reranker service instance from global config.

        Reranker models:
        - Cohere: rerank-english-v3.0, rerank-multilingual-v3.0
        - Pinecone: bge-reranker-v2-m3
        - Custom: any cross-encoder model via HuggingFace
        """
        from app.config import config

        if hasattr(config, "reranker_instance") and config.reranker_instance:
            return RerankerService(config.reranker_instance)

        return None
```

#### Configuration

**Файл**: `surfsense_backend/app/config/__init__.py`

```python
class Config:
    # Reranker configuration
    RERANKERS_ENABLED = os.getenv("RERANKERS_ENABLED", "false").lower() == "true"
    RERANKERS_MODEL_NAME = os.getenv("RERANKERS_MODEL_NAME", "")
    RERANKERS_MODEL_TYPE = os.getenv("RERANKERS_MODEL_TYPE", "")  # cohere, pinecone, etc.

    # Initialize reranker instance if enabled
    reranker_instance = None
    if RERANKERS_ENABLED and RERANKERS_MODEL_NAME:
        from rerankers import Reranker

        if RERANKERS_MODEL_TYPE == "cohere":
            reranker_instance = Reranker(
                RERANKERS_MODEL_NAME,
                api_key=os.getenv("COHERE_API_KEY")
            )
        elif RERANKERS_MODEL_TYPE == "pinecone":
            reranker_instance = Reranker(RERANKERS_MODEL_NAME)
        else:
            # Default: load from HuggingFace
            reranker_instance = Reranker(RERANKERS_MODEL_NAME)
```

#### Usage in Q&A Agent

**Файл**: `surfsense_backend/app/agents/researcher/qna_agent/nodes.py`

```python
async def rerank_documents(
    state: QnAState,
    config: RunnableConfig,
    writer: StreamWriter
) -> dict:
    """
    Rerank documents using RerankerService.

    Input: state.relevant_documents (top 20 from hybrid search)
    Output: state.reranked_documents (top 10 after reranking)
    """
    from app.services.reranker_service import RerankerService

    # Check if reranking is enabled
    reranker_service = RerankerService.get_reranker_instance()

    if not reranker_service:
        # Reranking disabled, return top 10 from hybrid search
        return {"reranked_documents": state.relevant_documents[:10]}

    # Rerank top 20 results
    reranked_results = reranker_service.rerank_documents(
        query_text=state.user_query,
        documents=state.relevant_documents  # Top 20 from hybrid search
    )

    # Take top 10 after reranking
    top_reranked = reranked_results[:10]

    return {"reranked_documents": top_reranked}
```

### Reranking Models

| Модель | Тип | Языки | Качество |
|--------|-----|-------|----------|
| **Cohere rerank-english-v3.0** | Cross-encoder | English | Очень высокое |
| **Cohere rerank-multilingual-v3.0** | Cross-encoder | 100+ languages | Высокое |
| **Pinecone bge-reranker-v2-m3** | Cross-encoder | Multilingual | Высокое |
| **Custom HF models** | Cross-encoder | Varies | Varies |

### Преимущества Reranking Pattern

1. **Improved precision**: +15-30% improvement over hybrid search alone
2. **Better relevance**: Cross-encoders understand query-document relationship
3. **Trade-off**: Slower but more accurate (applied to small candidate set)
4. **Plug-and-play**: Easy to enable/disable via configuration
5. **Fallback safety**: Automatically falls back to original scores on error

---

## Резюме: Семантические паттерны

| Паттерн | Файл | Назначение | Ключевые компоненты |
|---------|------|------------|---------------------|
| **RAG** | `agents/researcher/nodes.py` | Context-aware generation | Retrieval → Augmentation → Generation |
| **Embedding Pipeline** | `retriver/chunks_hybrid_search.py` | Text vectorization | Text → Model → Vector → Storage |
| **Hybrid Search** | `retriver/chunks_hybrid_search.py` | Best-of-both search | Vector + FTS + RRF |
| **Reranking** | `services/reranker_service.py` | Improve precision | Initial search → Cross-encoder → Top results |

Эти паттерны обеспечивают интеллектуальный, точный и контекстуально-осведомленный поиск и генерацию ответов на основе семантического понимания.
