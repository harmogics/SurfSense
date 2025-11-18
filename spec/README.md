# SurfSense Document Pipeline Specification

## О спецификации

Данная спецификация описывает полную архитектуру pipeline обработки и индексации документов в системе SurfSense, с фокусом на:
- Работе с узлами графа знаний
- Формировании семантических аспектов
- Выделении концептуального слоя
- Использовании языковых моделей и AI агентов

## Структура документации

### [00-overview.md](./00-overview.md)
**Обзор Pipeline Обработки и Индексации Документов**

Общий архитектурный обзор системы:
- Основные этапы pipeline (от загрузки до AI-powered ответов)
- Ключевые компоненты и их взаимодействие
- Технологический стек
- Граф знаний: три уровня (LangGraph, Document-Chunk, Semantic)
- Метрики производительности

### [01-document-ingestion.md](./01-document-ingestion.md)
**Загрузка и ETL Обработка Документов**

Детальное описание первого этапа pipeline:
- Поддерживаемые источники (файлы, URLs, connectors)
- Три ETL сервиса: Unstructured, LlamaCloud, Docling
- Конвертация в Markdown с сохранением структуры
- Генерация hashes для дедупликации
- Извлечение и форматирование метаданных
- Генерация summary через LLM (LONG_CONTEXT роль)
- Оптимизация контента под context window

**Ключевые функции**:
- `process_file_in_background()` - главный orchestrator
- `add_received_file_document_using_*()` - ETL-специфичная обработка
- `generate_document_summary()` - концептуальная экстракция
- `optimize_content_for_context_window()` - адаптация под LLM

### [02-graph-architecture.md](./02-graph-architecture.md)
**Архитектура Графа Знаний и Узлов**

Трехуровневая архитектура графа:

#### 1. LangGraph State Graph (Workflow)
- **Researcher Agent**: главный orchestrator
  - Узел: `reformulate_user_query` (Strategic LLM)
  - Узел: `handle_qna_workflow` (delegation)
  - Узел: `generate_further_questions` (Strategic LLM)
- **Q&A SubAgent**: специализированный агент
  - Узел: `rerank_documents` (RerankerService)
  - Узел: `answer_question` (Fast LLM + RAG)

#### 2. Document-Chunk Hierarchy (Data Graph)
- Явная структура в PostgreSQL
- Relationships: User → SearchSpace → Document → Chunks
- Foreign keys и back-references
- Метаданные для контекстного обогащения

#### 3. Semantic Graph (Vector Space)
- Неявный граф через embeddings
- Cosine similarity как "ребра"
- Динамическая навигация по семантическому пространству
- Multi-hop traversal для topic exploration

**Интеграция уровней**: LangGraph управляет workflow → запросы к Semantic Graph → traversal по Data Graph

### [03-llm-agents.md](./03-llm-agents.md)
**Языковые Агенты и их Роли**

Детальное описание AI агентов и использования LLM:

#### LLM Configuration
- **Три роли моделей**: FAST, LONG_CONTEXT, STRATEGIC
- **20+ поддерживаемых провайдеров**: OpenAI, Anthropic, Google, Groq, Ollama, и др.
- **Пользовательские конфигурации**: per-user LLM settings с ролевыми моделями

#### Агентные функции

**Researcher Agent**:
1. **reformulate_user_query** (Strategic LLM)
   - Переформулирование с учетом контекста диалога
   - Entity resolution, query expansion
   - Temperature: 0.3 для детерминированности

2. **handle_qna_workflow** (Orchestration)
   - Поиск через ConnectorService
   - Delegation к Q&A SubAgent
   - Интеграция результатов

3. **generate_further_questions** (Strategic LLM)
   - Генерация 3-5 follow-up вопросов
   - Gap identification, topic expansion
   - Temperature: 0.7 для креативности

**Q&A SubAgent**:
4. **rerank_documents** (RerankerService)
   - Cohere/Pinecone reranking
   - Top 20 → Top 10 по релевантности
   - Улучшение precision на 15-30%

5. **answer_question** (Fast LLM)
   - RAG: контекст + генерация
   - Inline цитаты [1], [2], ...
   - Token optimization + streaming
   - Temperature: 0.3 для accuracy

#### Промпты
- `REFORMULATE_QUERY_PROMPT`: контекстное переформулирование
- `QA_PROMPT_TEMPLATE`: генерация ответов с цитатами
- `FURTHER_QUESTIONS_PROMPT`: follow-up вопросы
- `SUMMARY_PROMPT_TEMPLATE`: концептуальная экстракция

### [04-semantic-layer.md](./04-semantic-layer.md)
**Семантическая Обработка и Концептуальный Слой**

Четыре уровня семантической обработки:

#### Level 1: Text Extraction & Normalization
- ETL processing с сохранением структуры
- Markdown как unified format
- Metadata extraction

#### Level 2: Conceptual Extraction (LLM-based)
- **Концептуальные аспекты через LLM**:
  - Main Topics: ключевые темы
  - Key Entities: важные сущности
  - Relationships: связи между концепциями
  - Hierarchical Structure: организация знаний
  - Actionable Insights: практические выводы
- **SUMMARY_PROMPT_TEMPLATE**: детальный промпт для концептуальной экстракции
- **Качество summary**: Coverage, Accuracy, Abstraction, Coherence

#### Level 3: Semantic Embedding
- Vector representation (1536/3072 dimensions)
- Encoding семантического значения
- Multi-lingual support

#### Level 4: Semantic Indexing & Retrieval
- pgvector storage
- Hybrid search (vector + full-text + RRF)
- Semantic navigation

**Концептуальные операции**:
- `compute_semantic_similarity()`: близость концепций
- `cluster_concepts()`: тематические группы
- `navigate_concept_graph()`: multi-hop exploration

**Структурные элементы**:
- Mapping типов (Title, Formula, Table, Code) на семантические структуры
- Сохранение концептуальной целостности при chunking
- Metadata как концептуальный контекст

### [05-embeddings-chunking-search.md](./05-embeddings-chunking-search.md)
**Векторизация, Chunking, Поиск и Интеграция Источников**

Объединенный документ, покрывающий:

#### Часть 1: Embeddings и Vector Search
- **Конфигурация моделей**: text-embedding-3-small/large, ada-002
- **Генерация embeddings**: document-level и chunk-level
- **PostgreSQL + pgvector**: хранение и индексирование
- **Vector search**: cosine distance (<=> operator)
- **Full-text search**: PostgreSQL FTS с ts_rank_cd
- **Hybrid search с RRF**: Reciprocal Rank Fusion
  - Formula: `score = α/(k+rank_semantic) + (1-α)/(k+rank_keyword)`
  - k=60, α=0.5 (configurable)

#### Часть 2: Chunking и Indexing
- **RecursiveChunker**: приоритет разделителей (параграфы → строки → предложения)
- **CodeChunker**: специализация для программного кода
- **Adaptive chunk size**: по max_seq_length embedding модели
- **Процесс**: content → chunks → embeddings → database

#### Часть 3: Connector Integration
- **15+ поддерживаемых источников**: Slack, GitHub, Notion, Jira, и др.
- **Base indexer functions**: дедупликация, date range calculation
- **Пример: Slack Indexer**:
  - Fetch messages по каналам
  - Проверка дубликатов (по message_id)
  - Metadata: channel, author, timestamp, reactions
  - Full pipeline: summary → embeddings → chunks → DB

**ConnectorService**: unified search across all sources

## Архитектурная диаграмма

```
┌─────────────────────────────────────────────────────────────────┐
│                  SURFSENSE FULL ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────┘

┌─ SOURCES ─────────────────┐
│ • Files (PDF, DOCX, etc.) │
│ • URLs (Web Crawling)     │──┐
│ • Connectors (Slack, etc.)│  │
└───────────────────────────┘  │
                               ▼
                      ┌─ ETL PROCESSING ────┐
                      │ • Unstructured      │
                      │ • LlamaCloud        │
                      │ • Docling           │
                      └──────┬──────────────┘
                             ▼
                   ┌─ SEMANTIC LAYER ────────┐
                   │ LLM Summarization        │
                   │ (LONG_CONTEXT)           │
                   │ • Conceptual extraction  │
                   │ • Metadata integration   │
                   └──────┬───────────────────┘
                          ▼
          ┌─ EMBEDDINGS ──────────┬─ CHUNKING ────────┐
          │ • Document embedding  │ • RecursiveChunker│
          │ • Chunk embeddings    │ • CodeChunker     │
          │ (1536/3072 dims)      │ • Adaptive size   │
          └──────┬────────────────┴─────┬─────────────┘
                 │                      │
                 ▼                      ▼
            ┌─────────── STORAGE ───────────┐
            │   PostgreSQL + pgvector       │
            │   • Document table            │
            │   • Chunk table               │
            │   • IVFFlat indexes           │
            └────────────┬──────────────────┘
                         ▼
          ┌────── SEARCH & RETRIEVAL ──────┐
          │  Hybrid Search (RRF)            │
          │  • Vector Search (cosine)       │
          │  • Full-Text Search (FTS)       │
          │  • Reciprocal Rank Fusion       │
          └────────────┬────────────────────┘
                       ▼
             ┌─── RERANKING ─────┐
             │  Cohere / Pinecone │
             │  Top 20 → Top 10   │
             └────────┬───────────┘
                      ▼
          ┌────── AI AGENTS (LangGraph) ──────┐
          │  Researcher Agent:                 │
          │  ├─ reformulate_query (Strategic)  │
          │  ├─ handle_qna_workflow            │
          │  │  └─ Q&A SubAgent:               │
          │  │     ├─ rerank_documents         │
          │  │     └─ answer_question (Fast)   │
          │  └─ generate_further_questions     │
          └────────────┬─────────────────────┘
                       ▼
              ┌─── OUTPUT ────┐
              │ • Answer      │
              │ • Citations   │
              │ • Sources     │
              │ • Follow-ups  │
              └───────────────┘
```

## Ключевые технологии

| Компонент | Технология |
|-----------|------------|
| **ETL** | Unstructured, LlamaCloud, Docling |
| **LLM Orchestration** | LangGraph (StateGraph) |
| **LLM Providers** | OpenAI, Anthropic, Google, Groq, Ollama, 15+ others |
| **Embeddings** | text-embedding-3-small/large, ada-002 |
| **Vector DB** | PostgreSQL + pgvector (IVFFlat index) |
| **Chunking** | Chonkie (RecursiveChunker, CodeChunker) |
| **Search** | Hybrid (Vector + FTS + RRF) |
| **Reranking** | Cohere, Pinecone |
| **Database** | PostgreSQL with pgvector extension |

## Ключевые принципы архитектуры

1. **Многоуровневая семантика**: От текста → концепции → векторы
2. **Граф знаний**: Явный (DB) + Неявный (vectors) + Workflow (LangGraph)
3. **AI-агентная архитектура**: Специализированные агенты с ролевыми LLM
4. **Hybrid approach**: Комбинация semantic и keyword search
5. **Концептуальный слой**: LLM-based извлечение смысла
6. **Гибкость**: 20+ LLM провайдеров, 15+ connectors, 3 ETL сервиса
7. **Производительность**: pgvector IVFFlat, async processing, streaming

## Важные файлы в кодовой базе

| Компонент | Путь |
|-----------|------|
| **Document Processing** | `surfsense_backend/app/tasks/document_processors/file_processors.py` |
| **Summary Generation** | `surfsense_backend/app/utils/document_converters.py` |
| **LLM Service** | `surfsense_backend/app/services/llm_service.py` |
| **Researcher Agent** | `surfsense_backend/app/agents/researcher/graph.py` |
| **Q&A Agent** | `surfsense_backend/app/agents/researcher/qna_agent/graph.py` |
| **Hybrid Search** | `surfsense_backend/app/retriver/chunks_hybrid_search.py` |
| **Reranker** | `surfsense_backend/app/services/reranker_service.py` |
| **Connectors** | `surfsense_backend/app/tasks/connector_indexers/` |
| **Config** | `surfsense_backend/app/config/__init__.py` |
| **Prompts** | `surfsense_backend/app/prompts/__init__.py` |
| **Database Models** | `surfsense_backend/app/db.py` |

## Как читать документацию

**Для общего понимания**: Начните с `00-overview.md`

**Для понимания pipeline**: Читайте последовательно 00 → 01 → 05

**Для понимания AI агентов**: 03 → 02 (agents → graph architecture)

**Для понимания семантики**: 04 → 05 (semantic layer → embeddings)

**Для интеграции источников**: 05 (часть 3 про connectors)

## Концептуальные карты

### Pipeline Flow
```
Source → ETL → Summary (LLM) → Embeddings → Chunking → Database → Search → Reranking → AI Agents → Answer
```

### Knowledge Graph Layers
```
Level 1: LangGraph (Workflow Orchestration)
Level 2: Document-Chunk Hierarchy (Relational Structure)
Level 3: Semantic Graph (Vector Space)
```

### LLM Roles
```
LONG_CONTEXT: Summary generation, document analysis
STRATEGIC: Query reformulation, question generation
FAST: RAG generation, real-time answers
```

## Метрики производительности

- **Дедупликация**: O(1) через hash lookup
- **Embedding**: 1536-3072 dimensions
- **Vector Search**: O(log N) через IVFFlat
- **Hybrid RRF**: k=60 constant
- **Reranking**: +15-30% precision improvement
- **Chunking**: Adaptive size (~512 tokens default)

---

**Дата создания**: 2025-11-18
**Версия**: 1.0
**Автор**: Claude (Anthropic)
