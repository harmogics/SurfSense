# Классические архитектурные паттерны

## Введение

SurfSense использует проверенные временем архитектурные паттерны для организации кода, обеспечивая поддерживаемость, тестируемость и масштабируемость системы. Эти паттерны создают четкое разделение ответственности между компонентами.

## 1. Service Layer Pattern

### Описание

Service Layer Pattern инкапсулирует бизнес-логику в сервисные классы, отделяя её от контроллеров и data access layer. Сервисы предоставляют высокоуровневый API для выполнения операций.

### Реализация в SurfSense

#### LLMService

**Файл**: `surfsense_backend/app/services/llm_service.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db import LLMConfig, UserSearchSpacePreference
import litellm

class LLMRole:
    """Роли LLM в системе"""
    LONG_CONTEXT = "long_context"  # Для больших документов
    FAST = "fast"                  # Для быстрых ответов
    STRATEGIC = "strategic"        # Для reasoning задач


async def get_user_llm_instance(
    session: AsyncSession,
    user_id: str,
    search_space_id: int,
    role: str
) -> ChatLiteLLM | None:
    """
    Service method: получение LLM instance для пользователя.

    Business Logic:
    1. Get user preferences for search space
    2. Determine which LLM config to use based on role
    3. Check if global or user-specific config
    4. Build and return LLM instance

    Args:
        session: Database session
        user_id: User identifier
        search_space_id: Search space identifier
        role: LLM role (LONG_CONTEXT, FAST, STRATEGIC)

    Returns:
        Configured LLM instance or None
    """
    # ═══════════════════════════════════════════════
    # STEP 1: GET USER PREFERENCES
    # ═══════════════════════════════════════════════

    result = await session.execute(
        select(UserSearchSpacePreference).where(
            UserSearchSpacePreference.user_id == user_id,
            UserSearchSpacePreference.search_space_id == search_space_id,
        )
    )
    preference = result.scalars().first()

    if not preference:
        return None

    # ═══════════════════════════════════════════════
    # STEP 2: SELECT LLM CONFIG ID BASED ON ROLE
    # ═══════════════════════════════════════════════

    llm_config_id = None

    if role == LLMRole.LONG_CONTEXT:
        llm_config_id = preference.long_context_llm_id
    elif role == LLMRole.FAST:
        llm_config_id = preference.fast_llm_id
    elif role == LLMRole.STRATEGIC:
        llm_config_id = preference.strategic_llm_id

    if llm_config_id is None:
        return None

    # ═══════════════════════════════════════════════
    # STEP 3: LOAD CONFIGURATION
    # ═══════════════════════════════════════════════

    # Negative IDs indicate global configs
    if llm_config_id < 0:
        global_config = get_global_llm_config(llm_config_id)
        provider = global_config["provider"]
        model_name = global_config["model"]
        api_key = global_config.get("api_key")
    else:
        # User-specific config
        llm_config_result = await session.execute(
            select(LLMConfig).where(LLMConfig.id == llm_config_id)
        )
        llm_config = llm_config_result.scalars().first()

        if not llm_config:
            return None

        provider = llm_config.provider.value
        model_name = llm_config.model_name
        api_key = decrypt_api_key(llm_config.api_key)

    # ═══════════════════════════════════════════════
    # STEP 4: BUILD LLM INSTANCE
    # ═══════════════════════════════════════════════

    # Map provider to LiteLLM format
    provider_map = {
        "OPENAI": "openai",
        "ANTHROPIC": "anthropic",
        "GOOGLE": "gemini",
        "GROQ": "groq",
        "OLLAMA": "ollama",
        "AZURE_OPENAI": "azure",
        # ... 20+ providers
    }

    provider_prefix = provider_map.get(provider, provider.lower())
    model_string = f"{provider_prefix}/{model_name}"

    # Create LiteLLM instance
    from langchain_community.chat_models import ChatLiteLLM

    litellm_kwargs = {
        "model": model_string,
        "api_key": api_key,
        "temperature": 0.7,
        "max_tokens": 2000,
    }

    # Add provider-specific parameters
    if provider == "AZURE_OPENAI":
        litellm_kwargs["api_base"] = llm_config.api_base
        litellm_kwargs["api_version"] = "2024-02-15-preview"

    return ChatLiteLLM(**litellm_kwargs)


# ═══════════════════════════════════════════════
# CONVENIENCE SERVICE METHODS
# ═══════════════════════════════════════════════

async def get_user_long_context_llm(
    session: AsyncSession,
    user_id: str,
    search_space_id: int
):
    """Service method: get Long Context LLM"""
    return await get_user_llm_instance(
        session, user_id, search_space_id, LLMRole.LONG_CONTEXT
    )


async def get_user_fast_llm(
    session: AsyncSession,
    user_id: str,
    search_space_id: int
):
    """Service method: get Fast LLM"""
    return await get_user_llm_instance(
        session, user_id, search_space_id, LLMRole.FAST
    )


async def get_user_strategic_llm(
    session: AsyncSession,
    user_id: str,
    search_space_id: int
):
    """Service method: get Strategic LLM"""
    return await get_user_llm_instance(
        session, user_id, search_space_id, LLMRole.STRATEGIC
    )
```

#### QueryService

**Файл**: `surfsense_backend/app/services/query_service.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession

class QueryService:
    """
    Service for query-related business logic.

    Responsibilities:
    - Query reformulation
    - Chat history formatting
    - Query optimization
    """

    @staticmethod
    async def reformulate_query_with_chat_history(
        user_query: str,
        session: AsyncSession,
        user_id: str,
        search_space_id: int,
        chat_history_str: str | None = None,
    ) -> str:
        """
        Service method: reformulate query using chat history.

        Business Logic:
        1. Get user's strategic LLM
        2. Format chat history
        3. Build reformulation prompt
        4. Call LLM to reformulate
        5. Return improved query
        """
        from app.prompts import REFORMULATE_QUERY_PROMPT
        from app.services.llm_service import get_user_strategic_llm

        # Get Strategic LLM for reasoning
        strategic_llm = await get_user_strategic_llm(
            session, user_id, search_space_id
        )

        if not strategic_llm:
            return user_query  # Fallback to original

        # Build prompt with chat history
        prompt = REFORMULATE_QUERY_PROMPT.format(
            chat_history=chat_history_str or "No previous context",
            user_query=user_query
        )

        # Call LLM
        response = await strategic_llm.ainvoke(
            prompt,
            temperature=0.3  # Low temperature for consistency
        )

        reformulated = response.content.strip()

        return reformulated if reformulated else user_query

    @staticmethod
    async def langchain_chat_history_to_str(chat_history: list) -> str:
        """
        Service utility: convert chat history to string format.

        Business Logic:
        - Format messages with roles
        - Preserve conversation flow
        - Handle different message types
        """
        if not chat_history:
            return ""

        formatted_messages = []

        for message in chat_history:
            role = message.type  # "human" or "ai"
            content = message.content

            if role == "human":
                formatted_messages.append(f"User: {content}")
            elif role == "ai":
                formatted_messages.append(f"Assistant: {content}")

        return "\n".join(formatted_messages)
```

### Преимущества Service Layer

1. **Разделение ответственности**: Бизнес-логика отделена от presentation и data access
2. **Переиспользование**: Сервисные методы используются в разных контроллерах/агентах
3. **Тестируемость**: Легко тестировать бизнес-логику изолированно
4. **Поддерживаемость**: Изменения в бизнес-логике локализованы
5. **Композиция**: Сервисы могут вызывать друг друга

---

## 2. Repository Pattern

### Описание

Repository Pattern абстрагирует data access layer, предоставляя коллекционно-подобный интерфейс для работы с данными. Изолирует бизнес-логику от деталей работы с БД.

### Реализация в SurfSense

**Файл**: `surfsense_backend/app/retriver/chunks_hybrid_search.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from app.db import Chunk, Document, SearchSpace

class ChucksHybridSearchRetriever:
    """
    Repository for chunk-level search operations.

    Provides collection-like interface for searching chunks
    without exposing SQL details to business logic.
    """

    def __init__(self, db_session: AsyncSession):
        """Initialize repository with database session"""
        self.db_session = db_session

    # ═══════════════════════════════════════════════
    # QUERY METHODS (Repository Interface)
    # ═══════════════════════════════════════════════

    async def vector_search(
        self,
        query_text: str,
        top_k: int,
        user_id: str,
        search_space_id: int,
        document_type: str | None = None
    ) -> list:
        """
        Repository method: vector similarity search.

        Abstracts:
        - Query embedding generation
        - pgvector cosine similarity
        - User/space filtering
        - Document type filtering
        """
        from app.config import config

        # Generate query embedding
        embedding_model = config.embedding_model_instance
        query_embedding = embedding_model.embed(query_text)

        # Build query
        query = (
            select(Chunk)
            .join(Document, Chunk.document_id == Document.id)
            .join(SearchSpace, Document.search_space_id == SearchSpace.id)
            .where(
                SearchSpace.user_id == user_id,
                SearchSpace.id == search_space_id
            )
        )

        # Optional: filter by document type
        if document_type:
            query = query.where(Document.document_type == document_type)

        # Order by cosine similarity
        query = query.order_by(
            Chunk.embedding.op("<=>")(query_embedding)
        ).limit(top_k)

        # Execute
        result = await self.db_session.execute(query)
        return result.scalars().all()

    async def full_text_search(
        self,
        query_text: str,
        top_k: int,
        user_id: str,
        search_space_id: int,
        document_type: str | None = None
    ) -> list:
        """
        Repository method: full-text search.

        Abstracts:
        - PostgreSQL FTS operators
        - Ranking by relevance
        - User/space filtering
        """
        # Build tsvector and tsquery
        tsvector = func.to_tsvector('english', Chunk.content)
        tsquery = func.plainto_tsquery('english', query_text)

        query = (
            select(Chunk)
            .join(Document, Chunk.document_id == Document.id)
            .join(SearchSpace, Document.search_space_id == SearchSpace.id)
            .where(
                SearchSpace.user_id == user_id,
                SearchSpace.id == search_space_id,
                tsvector.op('@@')(tsquery)  # Match operator
            )
        )

        if document_type:
            query = query.where(Document.document_type == document_type)

        # Order by relevance
        query = query.order_by(
            func.ts_rank_cd(tsvector, tsquery).desc()
        ).limit(top_k)

        result = await self.db_session.execute(query)
        return result.scalars().all()

    async def hybrid_search(
        self,
        query_text: str,
        top_k: int,
        user_id: str,
        search_space_id: int,
        document_type: str | None = None
    ) -> list:
        """
        Repository method: hybrid search with RRF.

        Abstracts:
        - Complex SQL with CTEs
        - RRF algorithm
        - Result formatting
        """
        # [Complex SQL implementation - см. 02-semantic-patterns.md]
        # ...

        return serialized_results

    # ═══════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════

    async def get_by_id(self, chunk_id: int) -> Chunk | None:
        """Repository method: get chunk by ID"""
        result = await self.db_session.execute(
            select(Chunk).where(Chunk.id == chunk_id)
        )
        return result.scalars().first()

    async def get_by_document(self, document_id: int) -> list[Chunk]:
        """Repository method: get all chunks for document"""
        result = await self.db_session.execute(
            select(Chunk).where(Chunk.document_id == document_id)
        )
        return result.scalars().all()
```

**Document Repository**:

**Файл**: `surfsense_backend/app/retriver/documents_hybrid_search.py`

```python
class DocumentHybridSearchRetriever:
    """
    Repository for document-level search operations.

    Similar interface to ChucksHybridSearchRetriever
    but operates on full documents instead of chunks.
    """

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def vector_search(self, query_text: str, top_k: int, ...) -> list:
        """Search documents by summary embedding"""
        # Similar to chunks but uses Document.embedding
        ...

    async def full_text_search(self, query_text: str, top_k: int, ...) -> list:
        """Search documents by content"""
        # FTS on Document.content
        ...

    async def hybrid_search(self, query_text: str, top_k: int, ...) -> list:
        """Hybrid search on documents"""
        # RRF combining vector and FTS
        ...
```

### Использование Repository

```python
# In service or agent
async def search_for_answer(user_query: str, user_id: str, search_space_id: int):
    # Create repository instance
    chunk_repo = ChucksHybridSearchRetriever(session)

    # Use repository methods (abstracts SQL)
    chunks = await chunk_repo.hybrid_search(
        query_text=user_query,
        top_k=20,
        user_id=user_id,
        search_space_id=search_space_id,
        document_type="SLACK_CONNECTOR"
    )

    # Business logic works with chunks, not SQL
    for chunk in chunks:
        process_chunk(chunk)
```

### Преимущества Repository Pattern

1. **Абстракция данных**: Скрывает детали SQL от бизнес-логики
2. **Централизация запросов**: Все запросы к chunks в одном месте
3. **Тестируемость**: Легко mock'ать repository для тестов
4. **Переиспользование**: Единые методы поиска для всей системы
5. **Изменяемость**: Легко сменить БД или структуру запросов

---

## 3. Factory Pattern

### Описание

Factory Pattern предоставляет интерфейс для создания объектов, позволяя подклассам изменять тип создаваемых объектов. Инкапсулирует логику создания сложных объектов.

### Реализация в SurfSense

**Файл**: `surfsense_backend/app/services/llm_service.py`

#### LLM Factory

```python
async def get_user_llm_instance(
    session: AsyncSession,
    user_id: str,
    search_space_id: int,
    role: str
) -> ChatLiteLLM | None:
    """
    FACTORY METHOD: Create LLM instance based on user config and role.

    Factory parameters:
    - user_id, search_space_id: determine user config
    - role: LONG_CONTEXT, FAST, STRATEGIC

    Returns:
    - Configured LLM instance for specific role
    """
    # [Implementation shown above in Service Layer section]
    ...


# ═══════════════════════════════════════════════
# FACTORY VARIANTS (Convenience Factories)
# ═══════════════════════════════════════════════

async def get_user_long_context_llm(session, user_id, search_space_id):
    """Factory: Create Long Context LLM"""
    return await get_user_llm_instance(
        session, user_id, search_space_id, LLMRole.LONG_CONTEXT
    )


async def get_user_fast_llm(session, user_id, search_space_id):
    """Factory: Create Fast LLM"""
    return await get_user_llm_instance(
        session, user_id, search_space_id, LLMRole.FAST
    )


async def get_user_strategic_llm(session, user_id, search_space_id):
    """Factory: Create Strategic LLM"""
    return await get_user_llm_instance(
        session, user_id, search_space_id, LLMRole.STRATEGIC
    )
```

#### Provider Factory (Abstract Factory Pattern)

```python
def create_llm_for_provider(
    provider: str,
    model_name: str,
    api_key: str,
    **kwargs
) -> ChatLiteLLM:
    """
    Abstract Factory: Create LLM instance for specific provider.

    Supports 20+ providers with different initialization requirements.
    """
    # Provider mapping
    provider_map = {
        "OPENAI": "openai",
        "ANTHROPIC": "anthropic",
        "GOOGLE": "gemini",
        "GROQ": "groq",
        "OLLAMA": "ollama",
        "AZURE_OPENAI": "azure",
        "AWS_BEDROCK": "bedrock",
        "VERTEX_AI": "vertex_ai",
        "COHERE": "cohere",
        "MISTRAL": "mistral",
        "DEEPSEEK": "openai",  # Uses OpenAI-compatible API
        "XAI": "openai",       # Grok via OpenAI API
        # ... 15+ more
    }

    provider_prefix = provider_map.get(provider, provider.lower())
    model_string = f"{provider_prefix}/{model_name}"

    # Base parameters
    litellm_kwargs = {
        "model": model_string,
        "api_key": api_key,
        "temperature": kwargs.get("temperature", 0.7),
        "max_tokens": kwargs.get("max_tokens", 2000),
    }

    # Provider-specific configuration
    if provider == "AZURE_OPENAI":
        litellm_kwargs.update({
            "api_base": kwargs.get("api_base"),
            "api_version": kwargs.get("api_version", "2024-02-15-preview"),
            "deployment_name": model_name
        })

    elif provider == "OLLAMA":
        litellm_kwargs.update({
            "api_base": kwargs.get("api_base", "http://localhost:11434")
        })

    elif provider == "AWS_BEDROCK":
        litellm_kwargs.update({
            "aws_access_key_id": kwargs.get("aws_access_key_id"),
            "aws_secret_access_key": kwargs.get("aws_secret_access_key"),
            "aws_region_name": kwargs.get("aws_region_name", "us-east-1")
        })

    # Create instance
    from langchain_community.chat_models import ChatLiteLLM
    return ChatLiteLLM(**litellm_kwargs)
```

#### Global LLM Config Factory

```python
def get_global_llm_config(llm_config_id: int) -> dict:
    """
    Factory: Get global (predefined) LLM configuration.

    Global configs have negative IDs:
    -1: GPT-4o (OPENAI)
    -2: Claude-3.5 Sonnet (ANTHROPIC)
    -3: Gemini-1.5 Pro (GOOGLE)
    -4: Llama-3.1-70B (GROQ)
    """
    global_configs = {
        -1: {
            "provider": "OPENAI",
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        -2: {
            "provider": "ANTHROPIC",
            "model": "claude-3-5-sonnet-20241022",
            "api_key": os.getenv("ANTHROPIC_API_KEY")
        },
        -3: {
            "provider": "GOOGLE",
            "model": "gemini-1.5-pro",
            "api_key": os.getenv("GOOGLE_API_KEY")
        },
        -4: {
            "provider": "GROQ",
            "model": "llama-3.1-70b-versatile",
            "api_key": os.getenv("GROQ_API_KEY")
        },
        # ... more predefined configs
    }

    return global_configs.get(llm_config_id, global_configs[-1])  # Default to GPT-4o
```

### Embedding Factory

**Файл**: `surfsense_backend/app/embeddings/auto_embeddings.py`

```python
class AutoEmbeddings:
    """
    Factory for creating embedding model instances.

    Supports:
    - OpenAI embeddings
    - Azure OpenAI embeddings
    - Custom HuggingFace models
    """

    @staticmethod
    def get_embeddings(
        model_name: str,
        azure_endpoint: str | None = None,
        azure_api_key: str | None = None,
        openai_api_key: str | None = None
    ):
        """
        Factory method: Create embedding model instance.

        Args:
            model_name: Model identifier (e.g., "text-embedding-3-small")
            azure_endpoint: Azure OpenAI endpoint (if using Azure)
            azure_api_key: Azure API key
            openai_api_key: OpenAI API key

        Returns:
            Embedding model instance with .embed() method
        """
        # Azure OpenAI embeddings
        if azure_endpoint and azure_api_key:
            from langchain_openai import AzureOpenAIEmbeddings

            return AzureOpenAIEmbeddings(
                model=model_name,
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version="2024-02-15-preview"
            )

        # OpenAI embeddings
        elif openai_api_key:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model=model_name,
                api_key=openai_api_key
            )

        # HuggingFace embeddings (local)
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name=model_name
            )
```

### Преимущества Factory Pattern

1. **Инкапсуляция создания**: Сложная логика создания скрыта
2. **Гибкость**: Легко добавлять новые типы объектов (новые LLM провайдеры)
3. **Конфигурируемость**: Создание на основе конфигурации
4. **Переиспользование**: Единая точка для создания LLM/embeddings
5. **Тестируемость**: Легко подменять factories в тестах

---

## 4. Strategy Pattern

### Описание

Strategy Pattern определяет семейство алгоритмов, инкапсулирует каждый и делает их взаимозаменяемыми. Позволяет выбирать алгоритм во время выполнения.

### Реализация в SurfSense

**Файл**: `surfsense_backend/app/tasks/document_processors/file_processors.py`

#### ETL Strategy

```python
# ═══════════════════════════════════════════════
# STRATEGY 1: Unstructured ETL
# ═══════════════════════════════════════════════

async def add_received_file_document_using_unstructured(
    session: AsyncSession,
    file_name: str,
    unstructured_processed_elements: list,
    search_space_id: int,
    user_id: str,
    original_file_content: bytes
) -> Document | None:
    """
    Strategy: Process document using Unstructured service.

    Features:
    - Advanced element detection (tables, figures, formulas)
    - OCR for images
    - Structure preservation
    """
    # Convert Unstructured elements to Markdown
    from app.utils.document_converters import convert_document_to_markdown

    file_in_markdown = await convert_document_to_markdown(
        unstructured_processed_elements
    )

    # Generate summary and embedding
    summary, embedding = await generate_document_summary(
        content=file_in_markdown,
        user_llm=user_llm,
        document_metadata={"FILE_NAME": file_name, "ETL_SERVICE": "UNSTRUCTURED"}
    )

    # Create chunks
    chunks = await create_document_chunks(file_in_markdown)

    # Create Document
    document = Document(
        title=extract_title(file_in_markdown) or file_name,
        content=file_in_markdown,
        embedding=embedding,
        chunks=chunks,
        document_type=DocumentType.FILE,
        document_metadata={"ETL_SERVICE": "UNSTRUCTURED"},
        user_id=user_id,
        search_space_id=search_space_id
    )

    session.add(document)
    await session.commit()

    return document


# ═══════════════════════════════════════════════
# STRATEGY 2: LlamaCloud ETL
# ═══════════════════════════════════════════════

async def add_received_file_document_using_llamacloud(
    session: AsyncSession,
    file_name: str,
    llamacloud_markdown_document: str,  # Already processed
    search_space_id: int,
    user_id: str,
    original_file_content: bytes
) -> Document | None:
    """
    Strategy: Process document using LlamaCloud service.

    Features:
    - Managed ETL pipeline
    - Direct Markdown output
    - Optimized for LlamaIndex ecosystem
    """
    file_in_markdown = llamacloud_markdown_document

    # Generate summary and embedding
    summary, embedding = await generate_document_summary(
        content=file_in_markdown,
        user_llm=user_llm,
        document_metadata={"FILE_NAME": file_name, "ETL_SERVICE": "LLAMACLOUD"}
    )

    # Create chunks
    chunks = await create_document_chunks(file_in_markdown)

    # Create Document
    document = Document(
        title=extract_title(file_in_markdown) or file_name,
        content=file_in_markdown,
        embedding=embedding,
        chunks=chunks,
        document_type=DocumentType.FILE,
        document_metadata={"ETL_SERVICE": "LLAMACLOUD"},
        user_id=user_id,
        search_space_id=search_space_id
    )

    session.add(document)
    await session.commit()

    return document


# ═══════════════════════════════════════════════
# STRATEGY 3: Docling ETL
# ═══════════════════════════════════════════════

async def add_received_file_document_using_docling(
    session: AsyncSession,
    file_name: str,
    docling_markdown_document: str,
    search_space_id: int,
    user_id: str,
    original_file_content: bytes
) -> Document | None:
    """
    Strategy: Process document using Docling service.

    Features:
    - Local or cloud processing
    - Optimized for large documents
    - Batch processing support
    """
    file_in_markdown = docling_markdown_document

    # Special: Use Docling service for large document summary
    from app.services.docling_service import docling_service

    if len(file_in_markdown) > 50000:  # Large document
        summary = await docling_service.process_large_document_summary(
            content=file_in_markdown,
            metadata={"FILE_NAME": file_name}
        )
        # Generate embedding for summary
        embedding = config.embedding_model_instance.embed(summary)
    else:
        # Standard summary generation
        summary, embedding = await generate_document_summary(
            content=file_in_markdown,
            user_llm=user_llm,
            document_metadata={"FILE_NAME": file_name, "ETL_SERVICE": "DOCLING"}
        )

    # Create chunks
    chunks = await create_document_chunks(file_in_markdown)

    # Create Document
    document = Document(
        title=extract_title(file_in_markdown) or file_name,
        content=file_in_markdown,
        embedding=embedding,
        chunks=chunks,
        document_type=DocumentType.FILE,
        document_metadata={"ETL_SERVICE": "DOCLING"},
        user_id=user_id,
        search_space_id=search_space_id
    )

    session.add(document)
    await session.commit()

    return document


# ═══════════════════════════════════════════════
# STRATEGY SELECTOR (Context)
# ═══════════════════════════════════════════════

async def process_file_in_background(
    file_path: str,
    filename: str,
    user_id: str,
    search_space_id: int
) -> tuple[int, str | None]:
    """
    Context: Selects and executes ETL strategy based on configuration.

    Strategy selection:
    - config.ETL_SERVICE determines which strategy to use
    - Each strategy implements the same interface (returns Document)
    """
    from app.config import config as app_config

    # Read file
    with open(file_path, 'rb') as f:
        file_content = f.read()

    # STRATEGY SELECTION
    etl_service = app_config.ETL_SERVICE  # From environment

    if etl_service == "UNSTRUCTURED":
        # Execute Strategy 1
        from app.loaders.unstructured_loader import UnstructuredLoader

        loader = UnstructuredLoader(file_path)
        elements = await loader.aload()

        document = await add_received_file_document_using_unstructured(
            session=session,
            file_name=filename,
            unstructured_processed_elements=elements,
            search_space_id=search_space_id,
            user_id=user_id,
            original_file_content=file_content
        )

    elif etl_service == "LLAMACLOUD":
        # Execute Strategy 2
        from llama_parse import LlamaParse

        parser = LlamaParse(
            api_key=app_config.LLAMA_CLOUD_API_KEY,
            result_type="markdown"
        )
        result = await parser.aparse(file_path)
        markdown_doc = result[0].text

        document = await add_received_file_document_using_llamacloud(
            session=session,
            file_name=filename,
            llamacloud_markdown_document=markdown_doc,
            search_space_id=search_space_id,
            user_id=user_id,
            original_file_content=file_content
        )

    elif etl_service == "DOCLING":
        # Execute Strategy 3
        from app.services.docling_service import create_docling_service

        docling_service = create_docling_service()
        markdown_doc = await docling_service.process_document(
            file_path, filename
        )

        document = await add_received_file_document_using_docling(
            session=session,
            file_name=filename,
            docling_markdown_document=markdown_doc,
            search_space_id=search_space_id,
            user_id=user_id,
            original_file_content=file_content
        )

    else:
        raise ValueError(f"Unknown ETL service: {etl_service}")

    return document.id, None
```

### Преимущества Strategy Pattern

1. **Гибкость**: Легко переключаться между ETL сервисами
2. **Расширяемость**: Легко добавлять новые стратегии (ETL сервисы)
3. **Изоляция**: Каждая стратегия независима
4. **Конфигурируемость**: Выбор стратегии через конфигурацию
5. **Единый интерфейс**: Все стратегии возвращают Document

---

## 5. Adapter Pattern

### Описание

Adapter Pattern преобразует интерфейс класса в другой интерфейс, ожидаемый клиентами. Позволяет классам работать вместе, несмотря на несовместимые интерфейсы.

### Реализация в SurfSense

**Файл**: `surfsense_backend/app/services/connector_service.py`

#### ConnectorService (Universal Adapter)

```python
class ConnectorService:
    """
    Universal Adapter for different data source connectors.

    Adapts 15+ different APIs to unified interface:
    - Slack API → Unified document format
    - GitHub API → Unified document format
    - Notion API → Unified document format
    - etc.

    All methods return: tuple(source_object: dict, documents: list)
    """

    def __init__(self, session: AsyncSession, user_id: str | None = None):
        self.session = session
        self.chunk_retriever = ChucksHybridSearchRetriever(session)
        self.document_retriever = DocumentHybridSearchRetriever(session)
        self.user_id = user_id
        self.source_id_counter = 100000

    # ═══════════════════════════════════════════════
    # ADAPTER 1: Slack
    # ═══════════════════════════════════════════════

    async def search_slack(
        self,
        user_query: str,
        user_id: str,
        search_space_id: int,
        top_k: int = 20,
        search_mode: SearchMode = SearchMode.CHUNKS
    ) -> tuple[dict, list]:
        """
        Adapter: Slack API → Unified format

        Input: Slack-specific search parameters
        Output: Standardized (source_object, documents) tuple
        """
        # Search using repository
        slack_chunks = await self.chunk_retriever.hybrid_search(
            query_text=user_query,
            top_k=top_k,
            user_id=user_id,
            search_space_id=search_space_id,
            document_type="SLACK_CONNECTOR"  # Slack-specific filter
        )

        # ADAPT: Transform to unified format
        source_object = {
            "id": "slack_source",
            "title": "Slack Messages",
            "description": f"Found {len(slack_chunks)} relevant Slack messages",
            "type": "SLACK_CONNECTOR",
            "icon": "slack"
        }

        # Transform chunks to unified document format
        unified_documents = []

        for chunk in slack_chunks:
            metadata = chunk.document.document_metadata

            unified_doc = {
                "chunk_id": chunk.id,
                "content": chunk.content,
                "score": chunk.score,  # From hybrid search
                "document": {
                    "id": chunk.document.id,
                    "title": f"Slack: {metadata.get('channel_name', 'Unknown')}",
                    "document_type": "SLACK_CONNECTOR",
                    "url": f"https://slack.com/app_redirect?channel={metadata.get('channel_id')}",
                    "metadata": {
                        "channel": metadata.get("channel_name"),
                        "author": metadata.get("author"),
                        "timestamp": metadata.get("timestamp")
                    }
                }
            }

            unified_documents.append(unified_doc)

        return source_object, unified_documents

    # ═══════════════════════════════════════════════
    # ADAPTER 2: Notion
    # ═══════════════════════════════════════════════

    async def search_notion(
        self,
        user_query: str,
        user_id: str,
        search_space_id: int,
        top_k: int = 20
    ) -> tuple[dict, list]:
        """
        Adapter: Notion API → Unified format
        """
        notion_chunks = await self.chunk_retriever.hybrid_search(
            query_text=user_query,
            top_k=top_k,
            user_id=user_id,
            search_space_id=search_space_id,
            document_type="NOTION_CONNECTOR"
        )

        source_object = {
            "id": "notion_source",
            "title": "Notion Pages",
            "description": f"Found {len(notion_chunks)} relevant Notion pages",
            "type": "NOTION_CONNECTOR",
            "icon": "notion"
        }

        # ADAPT: Notion-specific transformation
        unified_documents = []

        for chunk in notion_chunks:
            metadata = chunk.document.document_metadata

            unified_doc = {
                "chunk_id": chunk.id,
                "content": chunk.content,
                "score": chunk.score,
                "document": {
                    "id": chunk.document.id,
                    "title": chunk.document.title,
                    "document_type": "NOTION_CONNECTOR",
                    "url": metadata.get("page_url"),
                    "metadata": {
                        "workspace": metadata.get("workspace"),
                        "last_edited": metadata.get("last_edited_time"),
                        "author": metadata.get("created_by")
                    }
                }
            }

            unified_documents.append(unified_doc)

        return source_object, unified_documents

    # ═══════════════════════════════════════════════
    # ADAPTER 3: GitHub
    # ═══════════════════════════════════════════════

    async def search_github(
        self,
        user_query: str,
        user_id: str,
        search_space_id: int,
        top_k: int = 20
    ) -> tuple[dict, list]:
        """
        Adapter: GitHub API → Unified format
        """
        github_chunks = await self.chunk_retriever.hybrid_search(
            query_text=user_query,
            top_k=top_k,
            user_id=user_id,
            search_space_id=search_space_id,
            document_type="GITHUB_CONNECTOR"
        )

        source_object = {
            "id": "github_source",
            "title": "GitHub Issues & PRs",
            "description": f"Found {len(github_chunks)} relevant GitHub items",
            "type": "GITHUB_CONNECTOR",
            "icon": "github"
        }

        # ADAPT: GitHub-specific transformation
        unified_documents = []

        for chunk in github_chunks:
            metadata = chunk.document.document_metadata

            unified_doc = {
                "chunk_id": chunk.id,
                "content": chunk.content,
                "score": chunk.score,
                "document": {
                    "id": chunk.document.id,
                    "title": f"#{metadata.get('number')} {chunk.document.title}",
                    "document_type": "GITHUB_CONNECTOR",
                    "url": metadata.get("html_url"),
                    "metadata": {
                        "repository": metadata.get("repository"),
                        "state": metadata.get("state"),  # open/closed
                        "author": metadata.get("author"),
                        "labels": metadata.get("labels", [])
                    }
                }
            }

            unified_documents.append(unified_doc)

        return source_object, unified_documents

    # ═══════════════════════════════════════════════
    # ADAPTER 4: Tavily API (External Web Search)
    # ═══════════════════════════════════════════════

    async def search_tavily(
        self,
        user_query: str,
        user_id: str,
        search_space_id: int,
        top_k: int = 5
    ) -> tuple[dict, list]:
        """
        Adapter: Tavily external API → Unified format

        Special: External API call, not database search
        """
        from tavily import TavilyClient

        # Get connector config
        tavily_connector = await self.get_connector_by_type(
            user_id, search_space_id, "TAVILY_API"
        )

        if not tavily_connector:
            return {}, []

        # Call external API
        tavily_client = TavilyClient(
            api_key=decrypt_api_key(tavily_connector.credentials["api_key"])
        )

        response = tavily_client.search(
            query=user_query,
            max_results=top_k
        )

        # ADAPT: External API format → Unified format
        source_object = {
            "id": "tavily_source",
            "title": "Web Search (Tavily)",
            "description": f"Found {len(response.get('results', []))} web results",
            "type": "TAVILY_API",
            "icon": "web"
        }

        unified_documents = []

        for idx, result in enumerate(response.get("results", [])):
            # External API → Internal format
            unified_doc = {
                "chunk_id": self.source_id_counter + idx,
                "content": result.get("content", ""),
                "score": result.get("score", 0.0),  # Tavily relevance score
                "document": {
                    "id": f"tavily_{idx}",
                    "title": result.get("title", ""),
                    "document_type": "WEB_SEARCH",
                    "url": result.get("url", ""),
                    "metadata": {
                        "source": "Tavily API",
                        "published_date": result.get("published_date")
                    }
                }
            }

            unified_documents.append(unified_doc)

        self.source_id_counter += len(unified_documents)

        return source_object, unified_documents

    # ... 10+ more adapters (Linear, Jira, Discord, Gmail, etc.)
```

### Использование Adapter

```python
# Agent node uses unified interface
async def fetch_relevant_documents(
    user_query: str,
    connectors_to_search: list[str],
    connector_service: ConnectorService
):
    all_documents = []

    for connector in connectors_to_search:
        # Unified interface for all connectors
        if connector == "SLACK_CONNECTOR":
            source, docs = await connector_service.search_slack(...)
        elif connector == "NOTION_CONNECTOR":
            source, docs = await connector_service.search_notion(...)
        elif connector == "GITHUB_CONNECTOR":
            source, docs = await connector_service.search_github(...)
        elif connector == "TAVILY_API":
            source, docs = await connector_service.search_tavily(...)

        # All return same format: (source_object, documents)
        all_documents.extend(docs)

    return all_documents
```

### Преимущества Adapter Pattern

1. **Единый интерфейс**: Все источники имеют одинаковый API
2. **Масштабируемость**: Легко добавлять новые connectors
3. **Изоляция**: Изменения в external API не влияют на бизнес-логику
4. **Переиспользование**: Unified format используется во всей системе
5. **Тестируемость**: Легко mock'ать адаптеры

---

## 6. Dependency Injection

### Описание

Dependency Injection - паттерн, при котором зависимости передаются в объект извне, а не создаются внутри него. Упрощает тестирование и повышает гибкость.

### Реализация в SurfSense

#### Constructor Injection (Services)

```python
# services/connector_service.py
class ConnectorService:
    def __init__(
        self,
        session: AsyncSession,  # INJECTED
        user_id: str | None = None
    ):
        """
        Constructor Injection: Dependencies passed as parameters.
        """
        self.session = session  # Injected DB session
        # Create repositories with injected session
        self.chunk_retriever = ChucksHybridSearchRetriever(session)  # INJECTION
        self.document_retriever = DocumentHybridSearchRetriever(session)  # INJECTION
        self.user_id = user_id

# Usage
connector_service = ConnectorService(
    session=db_session,  # Inject session
    user_id="user123"
)
```

#### State-based Injection (Agents)

```python
# agents/researcher/state.py
@dataclass
class State:
    """
    Agent State with injected dependencies.
    """
    # INJECTED DEPENDENCIES
    db_session: AsyncSession          # Database session injection
    streaming_service: StreamingService  # Streaming service injection

    # State data
    chat_history: list[Any] | None = field(default_factory=list)
    reformulated_query: str | None = field(default=None)
    # ...

# Usage in agent node
async def reformulate_user_query(state: State, config, writer):
    # Use injected dependencies
    db_session = state.db_session  # INJECTED
    streaming_service = state.streaming_service  # INJECTED

    # No need to create these dependencies inside the function
    llm = await get_user_strategic_llm(db_session, user_id, search_space_id)
```

#### Configuration-based Injection

```python
# config/__init__.py
class Config:
    """
    Global configuration with injected instances.
    """
    # INJECTED: Embedding model instance
    embedding_model_instance = AutoEmbeddings.get_embeddings(
        EMBEDDING_MODEL,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_api_key=AZURE_API_KEY,
        openai_api_key=OPENAI_API_KEY
    )

    # INJECTED: Chunker instance
    chunker_instance = RecursiveChunker(
        chunk_size=getattr(embedding_model_instance, "max_seq_length", 512)
    )

    # INJECTED: Reranker instance
    reranker_instance = None
    if RERANKERS_ENABLED:
        reranker_instance = Reranker(RERANKERS_MODEL_NAME, ...)

# Usage
from app.config import config

# Access injected instances
embedding = config.embedding_model_instance.embed(text)  # INJECTED
chunks = config.chunker_instance.chunk(content)  # INJECTED
```

#### Method Injection

```python
async def generate_document_summary(
    content: str,
    user_llm: Any,  # INJECTED LLM
    document_metadata: dict | None = None
) -> tuple[str, list[float]]:
    """
    Method Injection: LLM passed as parameter.

    Advantages:
    - Different LLM for different use cases
    - Easy to test with mock LLM
    - No tight coupling to specific LLM
    """
    # Use injected LLM
    summary_response = await user_llm.ainvoke(prompt)
    # ...

# Usage with different LLMs
summary1, emb1 = await generate_document_summary(
    content, long_context_llm  # Inject Long Context LLM
)

summary2, emb2 = await generate_document_summary(
    content, fast_llm  # Inject Fast LLM
)
```

### Преимущества Dependency Injection

1. **Тестируемость**: Легко подменять зависимости mock'ами
2. **Гибкость**: Можно использовать разные реализации
3. **Декаплинг**: Компоненты не зависят от конкретных реализаций
4. **Композиция**: Легко комбинировать зависимости
5. **Конфигурируемость**: Зависимости настраиваются в одном месте

---

## Резюме: Классические паттерны

| Паттерн | Файл | Назначение | Ключевые компоненты |
|---------|------|------------|---------------------|
| **Service Layer** | `services/*.py` | Бизнес-логика | LLMService, QueryService, RerankerService |
| **Repository** | `retriver/*.py` | Data access | ChucksHybridSearchRetriever, DocumentRetriever |
| **Factory** | `services/llm_service.py` | Создание объектов | get_user_llm_instance, AutoEmbeddings |
| **Strategy** | `tasks/document_processors/file_processors.py` | Взаимозаменяемые алгоритмы | Unstructured, LlamaCloud, Docling |
| **Adapter** | `services/connector_service.py` | Унификация интерфейсов | 15+ connector adapters |
| **Dependency Injection** | Везде | Инверсия зависимостей | Constructor, State, Config, Method injection |

Эти паттерны создают чистую, модульную и поддерживаемую архитектуру, следующую принципам SOLID и best practices разработки.
