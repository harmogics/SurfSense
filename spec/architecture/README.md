# Архитектурные паттерны и шаблоны проектирования в SurfSense

## Введение

Данная документация описывает архитектурные паттерны и шаблоны проектирования, используемые в SurfSense на уровне исходного кода. Система демонстрирует профессиональное применение как классических паттернов (Gang of Four), так и современных паттернов для AI/ML систем.

## Структура документации

### [01-agent-patterns.md](./01-agent-patterns.md) - Агентские паттерны

Паттерны для организации AI агентов и workflow orchestration:

1. **State Machine Pattern (LangGraph StateGraph)**
   - **Файл**: `agents/researcher/graph.py`
   - **Назначение**: Организация workflow как конечного автомата
   - **Компоненты**: States, Nodes, Edges, Compiled Graph
   - **Пример**: Researcher Agent (reformulate → qna → questions)

2. **Chain of Responsibility**
   - **Файл**: `agents/researcher/nodes.py`
   - **Назначение**: Последовательная обработка запроса через цепочку узлов
   - **Компоненты**: Handler nodes, State propagation
   - **Пример**: reformulate_user_query → handle_qna_workflow → generate_further_questions

3. **Observer Pattern (Pub/Sub)**
   - **Файл**: `services/streaming_service.py`
   - **Назначение**: Real-time streaming updates к клиенту
   - **Компоненты**: StreamingService (Publisher), Writer, Events
   - **Типы событий**: TERMINAL_INFO, SOURCES, ANSWER, FURTHER_QUESTIONS

4. **Command Pattern**
   - **Файл**: `agents/researcher/nodes.py`
   - **Назначение**: Инкапсуляция команд поиска по разным источникам
   - **Компоненты**: fetch_relevant_documents, connector commands
   - **Источники**: Slack, Notion, GitHub, Linear, Jira, Tavily, +10

### [02-semantic-patterns.md](./02-semantic-patterns.md) - Семантические паттерны

Паттерны для семантической обработки и интеллектуального поиска:

1. **RAG Pattern (Retrieval-Augmented Generation)**
   - **Файл**: `agents/researcher/nodes.py`
   - **Назначение**: Context-aware генерация ответов
   - **Фазы**: Retrieval (hybrid search) → Augmentation (context prep) → Generation (LLM)
   - **Ключевые компоненты**: fetch_relevant_documents, QA agent, prompt templates

2. **Embedding Pipeline Pattern**
   - **Файл**: `retriver/chunks_hybrid_search.py`
   - **Назначение**: Конвертация текста в векторные представления
   - **Pipeline**: Text → Preprocessing → Tokenization → Embedding Model → Vector Storage
   - **Модели**: text-embedding-3-small/large (1536/3072 dims)

3. **Hybrid Search Pattern**
   - **Файл**: `retriver/chunks_hybrid_search.py`
   - **Назначение**: Комбинация semantic и keyword search
   - **Компоненты**: Vector Search (cosine) + Full-Text Search (FTS) + RRF
   - **Formula**: `score = 1/(k+rank_semantic) + 1/(k+rank_keyword)`, k=60

4. **Reranking Pattern**
   - **Файл**: `services/reranker_service.py`
   - **Назначение**: Улучшение precision результатов поиска
   - **Pipeline**: Initial Search (top 20) → Cross-encoder (Cohere/Pinecone) → Top 10
   - **Улучшение**: +15-30% precision

### [03-architectural-patterns.md](./03-architectural-patterns.md) - Классические паттерны

Проверенные временем архитектурные паттерны:

1. **Service Layer Pattern**
   - **Файлы**: `services/llm_service.py`, `services/query_service.py`
   - **Назначение**: Инкапсуляция бизнес-логики
   - **Сервисы**: LLMService, QueryService, RerankerService, StreamingService
   - **Методы**: get_user_llm_instance, reformulate_query_with_chat_history

2. **Repository Pattern**
   - **Файлы**: `retriver/chunks_hybrid_search.py`, `retriver/documents_hybrid_search.py`
   - **Назначение**: Абстракция data access layer
   - **Репозитории**: ChucksHybridSearchRetriever, DocumentHybridSearchRetriever
   - **Методы**: vector_search, full_text_search, hybrid_search

3. **Factory Pattern**
   - **Файл**: `services/llm_service.py`, `embeddings/auto_embeddings.py`
   - **Назначение**: Создание сложных объектов
   - **Factories**: get_user_llm_instance (LLM по роли), AutoEmbeddings.get_embeddings
   - **Варианты**: Long Context LLM, Fast LLM, Strategic LLM

4. **Strategy Pattern**
   - **Файл**: `tasks/document_processors/file_processors.py`
   - **Назначение**: Взаимозаменяемые алгоритмы обработки
   - **Стратегии**: Unstructured ETL, LlamaCloud ETL, Docling ETL
   - **Выбор**: Через конфигурацию `config.ETL_SERVICE`

5. **Adapter Pattern**
   - **Файл**: `services/connector_service.py`
   - **Назначение**: Унификация интерфейсов внешних API
   - **Адаптеры**: 15+ connectors (Slack, Notion, GitHub, Jira, Linear, etc.)
   - **Формат**: Все возвращают `tuple(source_object: dict, documents: list)`

6. **Dependency Injection**
   - **Файлы**: Везде в системе
   - **Типы**: Constructor injection, State injection, Config injection, Method injection
   - **Примеры**: ConnectorService(session), State(db_session, streaming_service)

## Сводная таблица всех паттернов

| Категория | Паттерн | Файл | Ключевые компоненты |
|-----------|---------|------|---------------------|
| **Агентские** | State Machine | `agents/researcher/graph.py` | StateGraph, nodes, edges |
| | Chain of Responsibility | `agents/researcher/nodes.py` | Handler nodes, State flow |
| | Observer | `services/streaming_service.py` | StreamingService, events, writer |
| | Command | `agents/researcher/nodes.py` | Connector commands, fetch_relevant_documents |
| **Семантические** | RAG | `agents/researcher/nodes.py` | Retrieval → Augmentation → Generation |
| | Embedding Pipeline | `retriver/chunks_hybrid_search.py` | Text → Model → Vector → Storage |
| | Hybrid Search | `retriver/chunks_hybrid_search.py` | Vector + FTS + RRF |
| | Reranking | `services/reranker_service.py` | Cross-encoder, score update |
| **Классические** | Service Layer | `services/*.py` | LLMService, QueryService, etc. |
| | Repository | `retriver/*.py` | ChucksRetriever, DocumentRetriever |
| | Factory | `services/llm_service.py` | get_user_llm_instance, AutoEmbeddings |
| | Strategy | `tasks/document_processors/file_processors.py` | ETL strategies (3 services) |
| | Adapter | `services/connector_service.py` | 15+ connector adapters |
| | Dependency Injection | Везде | Constructor, State, Config, Method |

## Ключевые архитектурные принципы

### 1. Разделение ответственности (Separation of Concerns)

```
┌─────────────────────────────────────────────┐
│           LAYERED ARCHITECTURE              │
└─────────────────────────────────────────────┘

┌─ PRESENTATION LAYER ──────────────────────┐
│ • Agent Nodes (workflow)                  │
│ • Streaming (real-time updates)           │
└───────────────┬───────────────────────────┘
                │
┌─ SERVICE LAYER ───────────────────────────┐
│ • LLMService (business logic)             │
│ • QueryService (query operations)         │
│ • RerankerService (reranking logic)       │
│ • ConnectorService (adapter layer)        │
└───────────────┬───────────────────────────┘
                │
┌─ REPOSITORY LAYER ────────────────────────┐
│ • ChucksHybridSearchRetriever             │
│ • DocumentHybridSearchRetriever           │
│ (Data access abstraction)                 │
└───────────────┬───────────────────────────┘
                │
┌─ DATA LAYER ──────────────────────────────┐
│ • PostgreSQL + pgvector                   │
│ • SQLAlchemy ORM                          │
│ • Database models                         │
└───────────────────────────────────────────┘
```

### 2. Dependency Inversion Principle (SOLID)

- **Высокоуровневые модули** (агенты) не зависят от низкоуровневых (БД)
- Оба зависят от **абстракций** (репозитории, сервисы)
- Зависимости **инжектятся** через конструкторы и state

### 3. Open/Closed Principle

- **Open for extension**: Легко добавлять новые connectors, ETL сервисы, LLM провайдеры
- **Closed for modification**: Существующий код не меняется при добавлении нового

### 4. Single Responsibility Principle

- Каждый класс/модуль имеет **одну ответственность**:
  - `LLMService` - управление LLM
  - `QueryService` - операции с запросами
  - `RerankerService` - reranking логика
  - `StreamingService` - streaming events

### 5. Liskov Substitution Principle

- **Connectors**: все адаптеры взаимозаменяемы (одинаковый интерфейс)
- **ETL strategies**: любая стратегия может заменить другую
- **LLM instances**: любой LLM provider может быть использован

## Диаграмма взаимодействия паттернов

```
┌──────────────────────────────────────────────────────────────┐
│              PATTERN INTERACTION DIAGRAM                     │
└──────────────────────────────────────────────────────────────┘

User Query
    │
    ▼
┌─────────────────────┐
│ State Machine       │  ← Agent workflow orchestration
│ (LangGraph)         │
└─────┬───────────────┘
      │
      ├─→ Chain of Responsibility (nodes)
      │   ├─→ reformulate_query
      │   ├─→ handle_qna (RAG Pattern)
      │   │   ├─→ Service Layer (ConnectorService)
      │   │   │   ├─→ Adapter Pattern (15+ connectors)
      │   │   │   └─→ Repository Pattern (search)
      │   │   │       ├─→ Embedding Pipeline
      │   │   │       ├─→ Hybrid Search (Vector + FTS + RRF)
      │   │   │       └─→ Reranking Pattern
      │   │   └─→ Factory Pattern (get LLM by role)
      │   │       └─→ Strategy Pattern (LLM provider)
      │   └─→ generate_questions
      │
      └─→ Observer Pattern (streaming)
          └─→ Real-time updates to client

All connected via Dependency Injection
```

## Преимущества архитектуры

1. **Масштабируемость**:
   - Легко добавлять новые connectors (Adapter)
   - Легко добавлять новые ETL сервисы (Strategy)
   - Легко добавлять новые LLM провайдеры (Factory)

2. **Поддерживаемость**:
   - Четкое разделение ответственности (Service Layer)
   - Изолированная бизнес-логика
   - Модульная структура

3. **Тестируемость**:
   - Dependency Injection упрощает mock'и
   - Repository абстрагирует БД
   - Service Layer изолирует бизнес-логику

4. **Гибкость**:
   - Strategy Pattern для взаимозаменяемых алгоритмов
   - Adapter Pattern для унификации внешних API
   - Factory Pattern для конфигурируемого создания объектов

5. **Расширяемость**:
   - Open/Closed Principle
   - State Machine легко добавлять новые узлы
   - Chain of Responsibility легко расширять цепочку

## Примеры использования паттернов

### Добавление нового Connector (Adapter)

```python
# В ConnectorService
async def search_new_connector(
    self,
    user_query: str,
    user_id: str,
    search_space_id: int,
    top_k: int = 20
) -> tuple[dict, list]:
    """Новый адаптер следует тому же паттерну"""
    chunks = await self.chunk_retriever.hybrid_search(
        query_text=user_query,
        top_k=top_k,
        user_id=user_id,
        search_space_id=search_space_id,
        document_type="NEW_CONNECTOR"
    )

    source_object = {
        "id": "new_source",
        "title": "New Data Source",
        "type": "NEW_CONNECTOR"
    }

    # Adapt to unified format
    unified_documents = [...]

    return source_object, unified_documents
```

### Добавление нового ETL сервиса (Strategy)

```python
# В file_processors.py
async def add_received_file_document_using_new_etl(
    session, file_name, processed_content, ...
) -> Document:
    """Новая стратегия следует тому же интерфейсу"""
    # Process with new ETL
    file_in_markdown = processed_content

    # Same downstream processing
    summary, embedding = await generate_document_summary(...)
    chunks = await create_document_chunks(...)

    # Create Document (same structure)
    document = Document(...)
    return document

# В process_file_in_background():
if etl_service == "NEW_ETL":
    document = await add_received_file_document_using_new_etl(...)
```

### Добавление нового Node в граф (State Machine)

```python
# В graph.py
workflow.add_node("new_processing_step", new_node_function)

# Добавить в цепочку
workflow.add_edge("handle_qna_workflow", "new_processing_step")
workflow.add_edge("new_processing_step", "generate_further_questions")

# В nodes.py
async def new_node_function(state: State, config, writer) -> dict:
    """Новый узел следует паттерну Chain of Responsibility"""
    # Process state
    result = process(state)

    # Return state update
    return {"new_field": result}
```

## Как читать документацию

**Для понимания агентской архитектуры**: Начните с `01-agent-patterns.md`

**Для понимания поиска и семантики**: Читайте `02-semantic-patterns.md`

**Для понимания общей структуры**: Читайте `03-architectural-patterns.md`

**Для добавления новых features**:
1. Определите, какой паттерн применим
2. Найдите примеры в соответствующем документе
3. Следуйте тому же паттерну для consistency

## Связь с основной документацией

Эта документация дополняет основную спецификацию из `/spec`:

- `spec/00-overview.md` - Общий обзор архитектуры
- `spec/01-document-ingestion.md` - ETL процесс (использует Strategy Pattern)
- `spec/02-graph-architecture.md` - Граф знаний (использует State Machine)
- `spec/03-llm-agents.md` - AI агенты (использует Chain of Responsibility)
- `spec/04-semantic-layer.md` - Семантика (использует Embedding Pipeline, RAG)
- `spec/05-embeddings-chunking-search.md` - Поиск (использует Hybrid Search, Reranking)

## Заключение

Архитектура SurfSense демонстрирует профессиональное применение как классических, так и современных паттернов проектирования. Использование этих паттернов делает систему:

- **Понятной**: Каждый паттерн имеет четкое назначение
- **Расширяемой**: Легко добавлять новые функции
- **Поддерживаемой**: Изменения локализованы
- **Тестируемой**: Компоненты изолированы и mock'аемы
- **Гибкой**: Компоненты взаимозаменяемы

Следование этим паттернам обеспечивает долгосрочную maintainability и scalability системы.
