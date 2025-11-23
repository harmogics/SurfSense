# UX Flows: Frontend-Backend Architecture Documentation

## Обзор

Эта директория содержит полную документацию по архитектуре взаимодействия frontend и backend в SurfSense, с фокусом на информационные потоки, обмен данными и циклы обратной связи от источника данных до пользовательского интерфейса.

## Содержание документации

### [00-overview.md](./00-overview.md) - Архитектурный обзор
**Описание**: Высокоуровневый обзор full-stack архитектуры SurfSense

**Ключевые темы**:
- Технологический стек (Next.js + FastAPI + LangGraph)
- Архитектурная диаграмма frontend-backend взаимодействия
- Ключевые архитектурные паттерны (Bidirectional Data Flow, Progressive Enhancement)
- Коммуникационные протоколы (REST API + SSE)
- Информационные потоки (Query, Document, Configuration)
- Масштабируемость и производительность
- Безопасность и observability

**Для кого**: Архитекторы, tech leads, разработчики, желающие понять общую картину

---

### [01-chat-flow.md](./01-chat-flow.md) - Детальный поток чата
**Описание**: Полный E2E flow обработки пользовательского запроса от UI до streaming ответа

**Ключевые темы**:
- **Фаза 1-2**: Frontend инициализация и пользовательский ввод
- **Фаза 3**: Отправка запроса через AI SDK `useChat` hook
- **Фаза 4**: Backend route handler и валидация
- **Фаза 5**: LangGraph workflow execution (reformulate → handle_qna → generate_questions)
- **Фаза 6**: Детали node execution:
  - `reformulate_user_query`: Query enhancement с strategic LLM
  - `handle_qna_workflow`: Multi-connector search + QNA sub-agent (rerank + answer)
  - `generate_further_questions`: Contextual follow-ups с fast LLM
- **Фаза 7**: Streaming response (SSE protocol)
- **Фаза 8**: Frontend rendering (terminal events, sources, text, questions)
- **Фаза 9**: Auto-save в database
- Временная диаграмма (0ms → 4000ms)
- Обработка ошибок и оптимизации

**Для кого**: Backend и frontend разработчики, работающие с chat feature

**Файлы кода**:
- Frontend: `surfsense_web/app/dashboard/[search_space_id]/researcher/[[...chat_id]]/page.tsx`
- Backend: `surfsense_backend/app/routes/chats_routes.py`
- LangGraph: `surfsense_backend/app/agents/researcher/graph.py`

---

### [02-streaming-architecture.md](./02-streaming-architecture.md) - Архитектура streaming
**Описание**: Детальная реализация потоковой передачи данных в реальном времени

**Ключевые темы**:
- **Vercel AI SDK Data Stream Protocol**:
  - Message types: `0:` (text), `8:` (annotation), `3:` (error), `d:` (completion)
  - Примеры stream format
- **Backend Streaming Implementation**:
  - `StreamingService` class: format methods для всех типов сообщений
  - Usage в LangGraph nodes (yield SSE chunks)
  - FastAPI `StreamingResponse` с headers
  - Async generator pattern
- **Frontend Streaming Implementation**:
  - AI SDK `useChat` hook configuration
  - Internal stream processing (SSE parsing)
  - Message type definition (TypeScript interfaces)
- **UI Component Updates**:
  - `ChatTerminalDisplay`: Progress events
  - `ChatSourcesDisplay`: Document references
  - `ChatFurtherQuestions`: Follow-up chips
  - `ChatMessageContent`: Streaming text с markdown
- **Performance Optimizations**:
  - Debounced rendering
  - Virtual scrolling
  - Memoization
- **Error Handling & Recovery**:
  - Backend error streaming
  - Frontend retry mechanism
  - Connection recovery
- **Monitoring & Debugging**:
  - Structured logging
  - Analytics tracking

**Для кого**: Разработчики, реализующие real-time features

**Файлы кода**:
- Backend: `surfsense_backend/app/services/streaming_service.py`
- Frontend: `surfsense_web/components/chat/` (ChatTerminal, ChatSources, etc.)

---

### [03-state-management.md](./03-state-management.md) - Управление состоянием
**Описание**: Трехуровневая архитектура синхронизации состояния

**Ключевые темы**:
- **Level 1: Client State (Frontend)**:
  - Jotai atoms (atomic state): `activeSearchSpaceIdAtom`, `activeChatIdAtom`, `activeChathatUIAtom`
  - TanStack Query (server cache): query client setup, cache keys structure
  - Custom hooks: `useChatState`, `useChats`, `useCreateChat`, `useUpdateChat`
  - LocalStorage persistence: chat configuration storage
- **Level 2: Server State (Backend)**:
  - LangGraph State: workflow state dataclass
  - Configuration: runtime parameters (read-only)
  - Service Layer State: LLMService singleton cache, StreamingService session state
- **Level 3: Persistent State (Database)**:
  - SQLAlchemy models: UserSearchSpacePreference, Chat, Document, Chunk
  - Optimistic locking: state_version для concurrent updates
- **State Synchronization Patterns**:
  - Optimistic updates (immediate UI, rollback on error)
  - Server authoritative (polling for updates)
  - Eventual consistency (debounced auto-save)
  - Cache invalidation (TanStack Query)
- **State Flow Examples**: Create chat, send message, update LLM config
- **Debugging**: DevTools, logging, SQL queries
- **Best Practices**: Single source of truth, immutable updates, normalized state, derived state, state colocation

**Для кого**: Frontend и backend разработчики, работающие с state management

**Файлы кода**:
- Frontend: `surfsense_web/atoms/`, `surfsense_web/hooks/`
- Backend: `surfsense_backend/app/agents/researcher/state.py`

---

### [04-document-flow.md](./04-document-flow.md) - Поток обработки документов
**Описание**: Complete pipeline от загрузки файла до индексации в векторной БД

**Ключевые темы**:
- **Архитектура обработки**: Upload → Create → Process → Index → Search
- **Фаза 1: Frontend Upload**:
  - DropzoneUI с file type validation
  - Upload progress tracking
  - Status polling (processing → completed)
- **Фаза 2: Backend Document Creation**:
  - FormData handling
  - File storage (disk/S3)
  - Content hash для deduplication
  - Celery task trigger
- **Фаза 3: Celery Background Processing**:
  - ETL service selection (Unstructured, LlamaCloud, Docling)
  - Text extraction + metadata
  - Chunking с chonkie library
  - Embedding generation (batch processing)
  - Chunk storage с pgvector
  - Status update (completed/failed)
  - Retry logic (exponential backoff)
- **Фаза 4: ETL Service Implementation**:
  - DoclingService: PDF, DOCX, HTML, images
  - Structured element extraction (tables, figures)
- **Фаза 5: Chunking Strategy**:
  - RecursiveChunker для text/documents
  - CodeChunker для code files
  - Token-aware chunking (512 tokens, 50 overlap)
- **Фаза 6: Embedding Generation**:
  - EmbeddingService: OpenAI text-embedding-3-small/large
  - Batch processing (100 chunks per batch)
  - User preference support
- **Фаза 7: PostgreSQL Storage**:
  - Documents table (metadata, status, chunk_count)
  - Chunks table (text, embedding, metadata)
  - Indices: HNSW (vector), GIN (full-text)
- **Status Tracking & Polling**: Frontend status display, backend status endpoint
- **Error Handling**: Retry strategy, error display, retry UI
- **Performance**: Batch processing, embedding caching, chunking optimization
- **RAG Integration**: Immediate availability в hybrid search

**Для кого**: Backend разработчики, работающие с document processing

**Файлы кода**:
- Frontend: `surfsense_web/app/dashboard/[search_space_id]/documents/page.tsx`
- Backend: `surfsense_backend/app/celery_tasks/document_processing.py`
- Services: `surfsense_backend/app/services/docling_service.py`, `chunking_service.py`, `embedding_service.py`

---

### [05-feedback-loops.md](./05-feedback-loops.md) - Циклы обратной связи
**Описание**: Двунаправленная коммуникация между пользователем и системой

**Ключевые темы**:
- **Типы Feedback Loops**:
  1. **Immediate Feedback** (< 50ms): UI updates, optimistic rendering
  2. **Progressive Feedback** (100ms-30s): Streaming updates, processing status
  3. **Configuration Feedback**: Settings применяются к next request
  4. **Data Feedback**: Uploaded data появляется в results
  5. **Learning Feedback** (future): Implicit preferences → better results
- **1. Immediate Feedback Examples**:
  - Query input validation + character count
  - Document selection toggle + count update
- **2. Progressive Feedback Examples**:
  - Streaming answer: terminal events → sources → text tokens → questions
  - Document processing: upload progress → processing status → completion
- **3. Configuration Feedback Examples**:
  - LLM config change: save → cache clear → next query uses new model
  - Connector selection: toggle → state update → next query excludes/includes connector
- **4. Data Feedback Examples**:
  - Document upload → processing → chunks stored → search results updated
  - Connector indexing: initial → periodic re-index → fresh data в results
- **5. Learning Feedback (Future)**:
  - User interaction tracking
  - Preference calculation
  - Result reranking based on preferences
- **Real-World Examples**:
  - Complete chat query flow (0ms → 6000ms)
  - Settings change flow
  - Document upload flow
- **Best Practices**:
  - Acknowledge immediately
  - Show progress
  - Explain delays
  - Enable cancellation
  - Persist intent

**Для кого**: UX designers, frontend разработчики, product managers

**Файлы кода**:
- Frontend: Various components с user interaction handlers
- Backend: Status tracking, streaming updates

---

## Диаграммы и визуализации

### High-Level Architecture

```
Frontend (Next.js + React)
    ├─ UI Components (LlamaIndex Chat UI)
    ├─ State Management (Jotai + TanStack Query)
    └─ API Communication (AI SDK + fetch)
         │
         │ HTTP/SSE
         │
Backend (FastAPI + LangGraph)
    ├─ API Layer (REST + SSE streaming)
    ├─ Service Layer (LLM, Connector, Reranker)
    ├─ Agent Layer (LangGraph workflows)
    └─ Data Layer (PostgreSQL + pgvector)
```

### Data Flow Patterns

```
Query Flow:
User input → useChat → POST /api/v1/chat → LangGraph → RAG → LLM → SSE stream → UI

Document Flow:
File upload → POST /api/v1/documents → Celery → ETL → Chunk → Embed → Store → Search

Configuration Flow:
User changes → PUT /api/v1/config → Validate → Store → Cache clear → Next request
```

### Feedback Loop Cycle

```
User Action
    ↓
Immediate Feedback (UI update)
    ↓
Backend Processing
    ↓
Progressive Feedback (streaming)
    ↓
Data Persistence
    ↓
Configuration Update
    ↓
Next Request (uses new config/data)
```

## Технологии и инструменты

### Frontend Stack
- **Framework**: Next.js 15.5.6 (App Router)
- **React**: 19.1.0
- **State**: Jotai + TanStack Query
- **Streaming**: Vercel AI SDK (@ai-sdk/react)
- **UI**: LlamaIndex Chat UI + shadcn/ui
- **Types**: TypeScript + Zod

### Backend Stack
- **Framework**: FastAPI
- **Server**: Uvicorn (ASGI)
- **Orchestration**: LangGraph
- **Database**: PostgreSQL + pgvector
- **ORM**: SQLAlchemy 2.0 (async)
- **LLM**: LiteLLM (20+ providers)
- **Tasks**: Celery + Redis
- **Auth**: fastapi-users

### Infrastructure
- **Vector Search**: HNSW indices (pgvector)
- **Full-Text Search**: GIN indices (PostgreSQL)
- **Caching**: TanStack Query (client), LLMService (server)
- **Storage**: Local filesystem / S3
- **Monitoring**: Structured logging

## Как читать эту документацию

### Для новых разработчиков
1. Начните с **00-overview.md** для понимания общей архитектуры
2. Прочитайте **01-chat-flow.md** для понимания основного UX flow
3. Изучите **03-state-management.md** для работы с данными
4. Используйте остальные документы как reference

### Для специализированных задач
- **Работа с chat feature**: 01-chat-flow.md + 02-streaming-architecture.md
- **Работа с document processing**: 04-document-flow.md
- **Работа с user interactions**: 05-feedback-loops.md
- **Debugging state issues**: 03-state-management.md

### Для архитектурных решений
- Все документы содержат "Best Practices" секции
- Обратите внимание на "Performance Optimizations"
- Изучите "Error Handling" patterns

## Связь с другой документацией

### Основная спецификация (`spec/`)
- [00-overview.md](../00-overview.md): Pipeline обработки документов
- [01-document-ingestion.md](../01-document-ingestion.md): ETL стратегии
- [02-graph-architecture.md](../02-graph-architecture.md): LangGraph workflow
- [03-llm-agents.md](../03-llm-agents.md): Роли агентов и LLM
- [04-semantic-layer.md](../04-semantic-layer.md): Семантическая обработка
- [05-embeddings-chunking-search.md](../05-embeddings-chunking-search.md): RAG pipeline

### Архитектурные паттерны (`spec/architecture/`)
- [01-agent-patterns.md](../architecture/01-agent-patterns.md): State Machine, Chain of Responsibility
- [02-semantic-patterns.md](../architecture/02-semantic-patterns.md): RAG, Embedding, Hybrid Search
- [03-architectural-patterns.md](../architecture/03-architectural-patterns.md): Service Layer, Repository, Factory

### Дополнительные ресурсы
- **Codebase**: Все file paths указаны в документации
- **API Docs**: (если есть OpenAPI/Swagger)
- **Design System**: (если есть Figma/Storybook)

## Часто задаваемые вопросы (FAQ)

### Q: Как добавить новый connector type?
A: См. **04-document-flow.md** (ETL service implementation) + backend connector service

### Q: Как изменить LLM модель для пользователя?
A: См. **03-state-management.md** (Configuration Feedback) + **05-feedback-loops.md** (LLM config changes)

### Q: Как работает streaming ответов?
A: См. **02-streaming-architecture.md** (полная детализация SSE protocol)

### Q: Почему мой документ не появляется в search results?
A: См. **04-document-flow.md** (check processing status, chunk count, embeddings)

### Q: Как синхронизируется state между frontend и backend?
A: См. **03-state-management.md** (трехуровневая архитектура + sync patterns)

### Q: Как добавить новый feedback loop?
A: См. **05-feedback-loops.md** (best practices + examples)

## Contributing

При обновлении документации:
1. Сохраняйте consistency с existing structure
2. Добавляйте code examples с file paths
3. Обновляйте диаграммы при изменении architecture
4. Ссылайтесь на related documents
5. Указывайте "Для кого" в начале каждого раздела

## Changelog

- **2025-11-23**: Initial documentation creation
  - 00-overview.md: Full-stack architecture overview
  - 01-chat-flow.md: Complete E2E chat flow (9 phases)
  - 02-streaming-architecture.md: SSE streaming implementation
  - 03-state-management.md: Three-level state architecture
  - 04-document-flow.md: Document processing pipeline (7 phases)
  - 05-feedback-loops.md: User feedback patterns (5 types)

---

**Maintained by**: SurfSense Development Team
**Last Updated**: 2025-11-23
**Version**: 1.0.0
