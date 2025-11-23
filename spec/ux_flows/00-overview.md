# Архитектура взаимодействия Frontend-Backend

## Введение

SurfSense построен как современное full-stack приложение с четким разделением ответственности между frontend и backend компонентами. Архитектура оптимизирована для работы с потоковыми данными в реальном времени и обеспечивает двунаправленный обмен информацией между пользовательским интерфейсом и серверной логикой.

## Технологический стек

### Frontend
- **Framework**: Next.js 15.5.6 (App Router)
- **React**: 19.1.0
- **State Management**: Jotai + TanStack Query
- **Streaming**: Vercel AI SDK (@ai-sdk/react)
- **UI Components**: LlamaIndex Chat UI + shadcn/ui
- **Type Safety**: TypeScript + Zod
- **Build Tool**: Turbopack

### Backend
- **Framework**: FastAPI
- **ASGI Server**: Uvicorn
- **Agent Orchestration**: LangGraph
- **Database**: PostgreSQL + pgvector
- **ORM**: SQLAlchemy 2.0 (async)
- **LLM Integration**: LiteLLM (20+ провайдеров)
- **Task Queue**: Celery
- **Authentication**: fastapi-users

## Архитектурная диаграмма

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (Next.js)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │  React UI    │◄───│    Jotai     │◄───│  TanStack    │             │
│  │  Components  │    │    Atoms     │    │    Query     │             │
│  └──────┬───────┘    └──────────────┘    └──────────────┘             │
│         │                                                               │
│         │  ┌──────────────────────────────────────────┐                │
│         └─►│       Vercel AI SDK (useChat)            │                │
│            │  - SSE Connection Management              │                │
│            │  - Streaming Response Handling            │                │
│            │  - Message State Synchronization          │                │
│            └──────────────┬───────────────────────────┘                │
│                           │                                             │
└───────────────────────────┼─────────────────────────────────────────────┘
                            │
                            │ HTTP/SSE
                            │
┌───────────────────────────▼─────────────────────────────────────────────┐
│                         BACKEND (FastAPI)                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │              API Layer (FastAPI Routes)                  │          │
│  │  /api/v1/chat (SSE) | /api/v1/chats (CRUD)              │          │
│  └────────────────┬─────────────────────────────────────────┘          │
│                   │                                                     │
│  ┌────────────────▼─────────────────────────────────────────┐          │
│  │              Service Layer                                │          │
│  │  - LLMService (model selection & instantiation)          │          │
│  │  - ConnectorService (multi-source search)                │          │
│  │  - RerankerService (relevance optimization)              │          │
│  │  - QueryService (query enhancement)                      │          │
│  │  - StreamingService (SSE protocol)                       │          │
│  └────────────────┬─────────────────────────────────────────┘          │
│                   │                                                     │
│  ┌────────────────▼─────────────────────────────────────────┐          │
│  │           LangGraph Agent Workflow                       │          │
│  │                                                           │          │
│  │  reformulate_query → handle_qna → generate_questions     │          │
│  │         │                 │                 │             │          │
│  │         │                 ├─► QNA Agent:    │             │          │
│  │         │                 │   - rerank      │             │          │
│  │         │                 │   - answer      │             │          │
│  │         │                 │                 │             │          │
│  │         └─────────────────┴─────────────────┘             │          │
│  │                           │                                │          │
│  │                    StreamingService                       │          │
│  │                  (format & yield SSE)                     │          │
│  └────────────────────────────┬──────────────────────────────┘          │
│                               │                                         │
│  ┌────────────────────────────▼──────────────────────────────┐          │
│  │                  Data Layer                                │          │
│  │                                                             │          │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐     │          │
│  │  │ PostgreSQL  │  │ RAG Pipeline │  │  Connectors  │     │          │
│  │  │ + pgvector  │  │ (Hybrid      │  │  (15+ types) │     │          │
│  │  │             │  │  Search+RRF) │  │              │     │          │
│  │  └─────────────┘  └──────────────┘  └──────────────┘     │          │
│  └─────────────────────────────────────────────────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Ключевые архитектурные паттерны

### 1. Bidirectional Data Flow (Двунаправленный поток данных)

**Frontend → Backend**:
- REST API для CRUD операций (создание чата, обновление настроек)
- SSE connection для streaming запросов
- Bearer token authentication на каждом запросе

**Backend → Frontend**:
- JSON responses для синхронных операций
- SSE stream для асинхронных ответов
- Structured data format (Vercel AI SDK protocol)

### 2. Progressive Enhancement Pattern

Система обеспечивает многоуровневую обратную связь:
- **Immediate**: UI обновления (loading states)
- **Progress**: Streaming terminal info ("Searching YouTube...")
- **Partial**: Incremental text rendering (token by token)
- **Complete**: Final state with all metadata (sources, questions)

### 3. State Synchronization Pattern

```
Client State          Server State         Persistent State
(React/Jotai)    ↔    (LangGraph)     ↔     (PostgreSQL)
     │                     │                      │
     │                     │                      │
     ├─ Optimistic UI      ├─ Workflow State      ├─ Chat History
     ├─ Form State         ├─ Agent State         ├─ User Preferences
     ├─ Cache (TanStack)   ├─ Streaming Buffer    ├─ Documents
     └─ Local Storage      └─ Configuration       └─ Embeddings
```

### 4. Event-Driven Architecture

```
User Action → Event → State Change → Side Effect → UI Update
                │                         │
                │                         └─► Backend API Call
                └─► Local State Update
```

## Коммуникационные протоколы

### REST API (CRUD Operations)

**Endpoints**:
- `POST /api/v1/chats` - Create chat
- `GET /api/v1/chats` - List chats
- `GET /api/v1/chats/{id}` - Get chat details
- `PUT /api/v1/chats/{id}` - Update chat
- `DELETE /api/v1/chats/{id}` - Delete chat

**Request/Response Format**:
```typescript
// Request
POST /api/v1/chats
{
  "name": "New Research",
  "search_space_id": 123,
  "research_mode": "QNA"
}

// Response
{
  "id": 456,
  "name": "New Research",
  "created_at": "2025-11-23T...",
  "state_version": 0
}
```

### Server-Sent Events (Streaming)

**Protocol**: Vercel AI SDK Data Stream Protocol

**Message Types**:
- `0:{json}` - Text chunk (streaming answer)
- `8:[{json}]` - Annotation delta (sources, progress, questions)
- `3:{json}` - Error message
- `d:{json}` - Completion marker

**Example Stream**:
```
0:"Based on"
0:" the"
0:" documents"
8:[{"type":"TERMINAL_INFO","data":{"idx":1,"message":"Searching YouTube..."}}]
8:[{"type":"sources","data":{"nodes":[[{...}]]}}]
0:", the answer is"
8:[{"type":"FURTHER_QUESTIONS","data":["What about...?"]}]
d:{"finishReason":"stop","usage":{"promptTokens":1200,"completionTokens":350}}
```

## Информационные потоки

### 1. Query Flow (Поток запроса)
```
User types query → useChat.append() → POST /api/v1/chat → LangGraph → RAG → LLM → Stream response
```

### 2. Document Flow (Поток документа)
```
User uploads file → POST /api/v1/documents → Celery task → Chunk → Embed → Store → Index
```

### 3. Configuration Flow (Поток конфигурации)
```
User changes LLM → PUT /api/v1/llm_configs → Validate → Store → Next query uses new LLM
```

### 4. Feedback Flow (Обратная связь)
```
User selects document → Local state → Included in next query → Backend filters results
```

## Масштабируемость и производительность

### Frontend Optimization
- **Code Splitting**: Automatic route-based splitting (Next.js)
- **Memoization**: React.memo для тяжелых компонентов
- **Virtualization**: Для длинных списков (react-window)
- **Lazy Loading**: Images, components, routes
- **Client-side Caching**: TanStack Query (5 min default)

### Backend Optimization
- **Connection Pooling**: AsyncSession pool для PostgreSQL
- **Index Optimization**: HNSW (vector) + GIN (full-text)
- **Async I/O**: FastAPI + SQLAlchemy async
- **Streaming**: Reduce memory footprint, lower TTFB
- **Background Tasks**: Celery для long-running операций

## Безопасность

### Authentication & Authorization
1. **JWT Tokens**: httpOnly cookies + localStorage backup
2. **Bearer Authentication**: на каждом API запросе
3. **Row-Level Security**: все queries фильтруются по user_id
4. **Search Space Ownership**: проверка доступа на каждом эндпоинте

### Data Protection
- **CORS**: Configured для trusted origins
- **HTTPS**: Enforced в production
- **SQL Injection**: защита через SQLAlchemy ORM
- **XSS**: React auto-escaping + CSP headers
- **API Key Encryption**: для хранения LLM keys (recommended)

## Observability

### Logging
- **Frontend**: Console errors + Sentry (optional)
- **Backend**: Structured logging с timestamps
- **Database**: Slow query log

### Monitoring
- **Health Checks**: `/health` endpoint
- **Metrics**: Request count, latency, error rate
- **Tracing**: LangSmith integration (optional)

## Следующие разделы

- [01-chat-flow.md](./01-chat-flow.md) - Детальный разбор потока чата
- [02-streaming-architecture.md](./02-streaming-architecture.md) - Архитектура streaming
- [03-state-management.md](./03-state-management.md) - Управление состоянием
- [04-document-flow.md](./04-document-flow.md) - Поток обработки документов
- [05-feedback-loops.md](./05-feedback-loops.md) - Циклы обратной связи
