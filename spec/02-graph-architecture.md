# Архитектура Графа Знаний и Узлов

## Введение

В SurfSense граф знаний реализован через три взаимосвязанных уровня:
1. **LangGraph State Graph** - для orchestration AI агентов
2. **Document-Chunk Hierarchy** - для структурирования контента
3. **Semantic Graph** - для семантических связей через embeddings

Это гибридный подход, где явная граф-структура (узлы и ребра) комбинируется с неявным семантическим графом в векторном пространстве.

## 1. LangGraph State Graph (Workflow Graph)

### Концепция

LangGraph - это фреймворк для создания stateful multi-actor приложений с LLM. В SurfSense он используется для orchestration AI агентов в процессе исследования и Q&A.

**Ключевые компоненты**:
- **State** - состояние workflow (данные, передаваемые между узлами)
- **Nodes** - функции обработки (узлы графа)
- **Edges** - связи между узлами (направление потока)
- **Graph** - compiled workflow

### Основной граф: Researcher Agent

**Местоположение**: `surfsense_backend/app/agents/researcher/graph.py`

#### State Definition

**Файл**: `surfsense_backend/app/agents/researcher/state.py`

```python
from dataclasses import dataclass
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.streaming_service import StreamingService

@dataclass
class State:
    """
    Состояние Researcher агента.
    Передается между всеми узлами workflow.
    """
    # Сервисы
    db_session: AsyncSession
    streaming_service: StreamingService

    # Входные данные
    chat_history: list[Any]  # История диалога
    user_query: str | None   # Исходный запрос пользователя

    # Промежуточные результаты
    reformulated_query: str | None          # Переформулированный запрос
    relevant_documents: list[Any] | None    # Найденные документы
    reranked_documents: list[Any] | None    # Переранжированные документы

    # Выходные данные
    final_written_report: str | None        # Итоговый ответ
    further_questions: Any | None           # Follow-up вопросы
    sources: list[dict] | None              # Источники с цитатами
```

**Поля State** представляют собой **узлы данных**, передаваемые по графу.

#### Graph Structure

**Файл**: `surfsense_backend/app/agents/researcher/graph.py`

```python
from langgraph.graph import StateGraph, END
from app.agents.researcher.state import State
from app.agents.researcher import nodes

def create_researcher_graph() -> StateGraph:
    """
    Создает граф Researcher агента.

    Структура:
    START → reformulate_user_query → handle_qna_workflow → generate_further_questions → END
    """
    # Инициализация графа
    graph = StateGraph(State)

    # Добавление узлов (Nodes)
    graph.add_node("reformulate_user_query", nodes.reformulate_user_query)
    graph.add_node("handle_qna_workflow", nodes.handle_qna_workflow)
    graph.add_node("generate_further_questions", nodes.generate_further_questions)

    # Определение ребер (Edges)
    graph.set_entry_point("reformulate_user_query")  # Начальный узел

    graph.add_edge("reformulate_user_query", "handle_qna_workflow")
    graph.add_edge("handle_qna_workflow", "generate_further_questions")
    graph.add_edge("generate_further_questions", END)

    # Компиляция графа
    return graph.compile()
```

**Визуализация графа**:
```
┌─────────────────────────────────────────────────────────────┐
│                    RESEARCHER GRAPH                         │
└─────────────────────────────────────────────────────────────┘

      START
        │
        ▼
  ┌───────────────────────┐
  │ reformulate_user_query│  ← Strategic LLM
  └───────────────────────┘
        │ State.reformulated_query
        ▼
  ┌───────────────────────┐
  │  handle_qna_workflow  │  ← Q&A SubAgent
  └───────────────────────┘
        │ State.final_written_report
        ▼
  ┌───────────────────────┐
  │generate_further_questions│ ← Strategic LLM
  └───────────────────────┘
        │ State.further_questions
        ▼
       END
```

### Узлы Researcher Graph

#### Node 1: reformulate_user_query

**Цель**: Переформулировать запрос пользователя с учетом контекста диалога.

**Функция**: `nodes.reformulate_user_query()`
**Местоположение**: `surfsense_backend/app/agents/researcher/nodes.py`

```python
async def reformulate_user_query(state: State, config: RunnableConfig) -> dict:
    """
    Переформулирует запрос пользователя для улучшения поиска.

    Процесс:
    1. Получает chat_history из state
    2. Извлекает user_query (последнее сообщение)
    3. Использует REFORMULATE_QUERY_PROMPT
    4. Вызывает Strategic LLM
    5. Возвращает обновленный state с reformulated_query

    Input State:
        - chat_history: list[ChatMessage]
        - user_query: str

    Output State:
        - reformulated_query: str
    """
    from app.services.query_service import reformulate_query_with_chat_history

    # Извлечение последнего сообщения пользователя
    user_query = state.chat_history[-1].content if state.chat_history else ""

    # Получение LLM конфигурации пользователя
    user_llm_config = await get_user_llm_config(state.db_session, config.user_id)

    # Strategic LLM для переформулирования
    strategic_llm = llm_service.get_llm(user_llm_config, role=LLMRole.STRATEGIC)

    # Переформулирование запроса
    reformulated = await reformulate_query_with_chat_history(
        user_query=user_query,
        chat_history=state.chat_history[:-1],  # Без последнего сообщения
        llm=strategic_llm
    )

    # Стриминг промежуточного результата
    await state.streaming_service.send_message({
        "type": "reformulated_query",
        "content": reformulated
    })

    return {"reformulated_query": reformulated}
```

**REFORMULATE_QUERY_PROMPT** (из `app/prompts/__init__.py`):
```python
REFORMULATE_QUERY_PROMPT = """
Given the conversation history and the user's latest query, reformulate the query to be more specific and contextual.

## Conversation History
{chat_history}

## User's Latest Query
{user_query}

## Instructions
- If the query is clear and standalone, return it as-is
- If the query references previous context, incorporate that context
- Make the query specific and searchable
- Preserve the user's intent
- Keep it concise (1-2 sentences)

Reformulated query:
"""
```

#### Node 2: handle_qna_workflow

**Цель**: Выполнить Q&A workflow (поиск документов + генерация ответа).

**Функция**: `nodes.handle_qna_workflow()`

```python
async def handle_qna_workflow(state: State, config: RunnableConfig) -> dict:
    """
    Запускает Q&A субагент для поиска и ответа на вопрос.

    Процесс:
    1. Использует reformulated_query для поиска
    2. Вызывает ConnectorService для hybrid search
    3. Запускает Q&A SubAgent (sub-graph)
    4. Возвращает final_written_report + sources

    Input State:
        - reformulated_query: str
        - db_session, streaming_service

    Output State:
        - relevant_documents: list[Document]
        - reranked_documents: list[Document]
        - final_written_report: str
        - sources: list[dict]
    """
    from app.agents.researcher.qna_agent.graph import create_qna_graph
    from app.services.connector_service import ConnectorService

    # 1. Поиск релевантных документов
    connector_service = ConnectorService(state.db_session)

    search_results = await connector_service.search_all_sources(
        user_query=state.reformulated_query,
        user_id=config.user_id,
        search_space_id=config.search_space_id,
        top_k=20,  # Берем top 20 для reranking
        search_mode=SearchMode.CHUNKS
    )

    # 2. Создание Q&A субграфа
    qna_graph = create_qna_graph()

    # 3. Подготовка конфигурации для Q&A
    qna_config = {
        "user_query": state.reformulated_query,
        "relevant_documents": search_results.documents,
        "user_id": config.user_id,
        "search_space_id": config.search_space_id
    }

    # 4. Запуск Q&A субагента
    qna_state = await qna_graph.ainvoke(qna_config, config)

    # 5. Возврат результатов
    return {
        "relevant_documents": search_results.documents,
        "reranked_documents": qna_state.get("reranked_documents"),
        "final_written_report": qna_state.get("answer"),
        "sources": qna_state.get("sources")
    }
```

**Q&A SubAgent** - это отдельный граф, см. секцию ниже.

#### Node 3: generate_further_questions

**Цель**: Сгенерировать follow-up вопросы для углубления исследования.

**Функция**: `nodes.generate_further_questions()`

```python
async def generate_further_questions(state: State, config: RunnableConfig) -> dict:
    """
    Генерирует дополнительные вопросы на основе ответа.

    Процесс:
    1. Использует final_written_report + user_query
    2. Применяет FURTHER_QUESTIONS_PROMPT
    3. Вызывает Strategic LLM
    4. Возвращает список вопросов

    Input State:
        - final_written_report: str
        - reformulated_query: str

    Output State:
        - further_questions: list[str]
    """
    from app.prompts import FURTHER_QUESTIONS_PROMPT

    # Получение Strategic LLM
    user_llm_config = await get_user_llm_config(state.db_session, config.user_id)
    strategic_llm = llm_service.get_llm(user_llm_config, role=LLMRole.STRATEGIC)

    # Формирование промпта
    prompt = FURTHER_QUESTIONS_PROMPT.format(
        user_query=state.reformulated_query,
        answer=state.final_written_report
    )

    # Генерация вопросов
    response = await strategic_llm.ainvoke(prompt)
    questions = parse_questions_from_response(response.content)

    # Стриминг результата
    await state.streaming_service.send_message({
        "type": "further_questions",
        "content": questions
    })

    return {"further_questions": questions}
```

**FURTHER_QUESTIONS_PROMPT**:
```python
FURTHER_QUESTIONS_PROMPT = """
Based on the user's query and the answer provided, generate 3-5 follow-up questions that would help the user explore the topic further.

## User's Query
{user_query}

## Answer Provided
{answer}

## Instructions
- Generate questions that dig deeper into the topic
- Cover different aspects or angles
- Be specific and actionable
- Avoid yes/no questions
- Each question should be standalone

Format your response as a numbered list:
1. [Question 1]
2. [Question 2]
...
"""
```

### Q&A SubAgent Graph

**Местоположение**: `surfsense_backend/app/agents/researcher/qna_agent/graph.py`

#### State Definition

**Файл**: `surfsense_backend/app/agents/researcher/qna_agent/state.py`

```python
@dataclass
class QnAState:
    """
    Состояние Q&A подагента.
    """
    # Входные данные
    user_query: str
    relevant_documents: list[Any]  # Из hybrid search
    user_id: str
    search_space_id: int

    # Сервисы
    db_session: AsyncSession
    streaming_service: StreamingService

    # Промежуточные результаты
    reranked_documents: list[Any] | None

    # Выходные данные
    answer: str | None
    sources: list[dict] | None
```

#### Graph Structure

```python
from langgraph.graph import StateGraph, END
from app.agents.researcher.qna_agent.state import QnAState
from app.agents.researcher.qna_agent import nodes as qna_nodes

def create_qna_graph() -> StateGraph:
    """
    Создает граф Q&A агента.

    Структура:
    START → rerank_documents → answer_question → END
    """
    graph = StateGraph(QnAState)

    # Узлы
    graph.add_node("rerank_documents", qna_nodes.rerank_documents)
    graph.add_node("answer_question", qna_nodes.answer_question)

    # Ребра
    graph.set_entry_point("rerank_documents")
    graph.add_edge("rerank_documents", "answer_question")
    graph.add_edge("answer_question", END)

    return graph.compile()
```

**Визуализация Q&A SubGraph**:
```
      START
        │
        ▼
  ┌───────────────────┐
  │ rerank_documents  │  ← RerankerService (Cohere/Pinecone)
  └───────────────────┘
        │ QnAState.reranked_documents
        ▼
  ┌───────────────────┐
  │  answer_question  │  ← Fast LLM + RAG
  └───────────────────┘
        │ QnAState.answer + sources
        ▼
       END
```

#### Q&A Node 1: rerank_documents

**Файл**: `surfsense_backend/app/agents/researcher/qna_agent/nodes.py`

```python
async def rerank_documents(state: QnAState, config: RunnableConfig) -> dict:
    """
    Переранжирует документы по релевантности к запросу.

    Процесс:
    1. Проверяет, включен ли reranking (RERANKERS_ENABLED)
    2. Если включен, использует RerankerService
    3. Обновляет scores документов
    4. Сортирует по новым scores
    5. Возвращает top_k документов

    Input State:
        - relevant_documents: list (из hybrid search, 20 docs)
        - user_query: str

    Output State:
        - reranked_documents: list (top 10 после reranking)
    """
    from app.services.reranker_service import reranker_service
    from app.config import config

    # Проверка, включен ли reranking
    if not config.RERANKERS_ENABLED:
        # Возвращаем top 10 без reranking
        return {"reranked_documents": state.relevant_documents[:10]}

    # Конвертация документов в формат для reranker
    reranker_docs = [
        {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.document_metadata
        }
        for doc in state.relevant_documents
    ]

    # Reranking
    reranked = await reranker_service.rerank_documents(
        query_text=state.user_query,
        documents=reranker_docs
    )

    # Обновление scores в оригинальных документах
    doc_scores = {doc["id"]: doc["score"] for doc in reranked}

    for doc in state.relevant_documents:
        if doc.id in doc_scores:
            doc.rerank_score = doc_scores[doc.id]

    # Сортировка по rerank_score и выбор top 10
    sorted_docs = sorted(
        state.relevant_documents,
        key=lambda x: getattr(x, 'rerank_score', 0),
        reverse=True
    )[:10]

    return {"reranked_documents": sorted_docs}
```

#### Q&A Node 2: answer_question

```python
async def answer_question(state: QnAState, config: RunnableConfig) -> dict:
    """
    Генерирует ответ на вопрос с использованием переранжированных документов.

    Процесс:
    1. Оптимизирует документы под token limit модели
    2. Форматирует контекст с источниками
    3. Использует QA_PROMPT_TEMPLATE
    4. Вызывает Fast LLM для генерации ответа
    5. Парсит цитаты из ответа
    6. Формирует sources список

    Input State:
        - reranked_documents: list
        - user_query: str

    Output State:
        - answer: str (с inline цитатами)
        - sources: list[dict] (metadata для UI)
    """
    from app.prompts import QA_PROMPT_TEMPLATE
    from app.utils.document_converters import optimize_documents_for_context

    # Получение Fast LLM
    user_llm_config = await get_user_llm_config(state.db_session, state.user_id)
    fast_llm = llm_service.get_llm(user_llm_config, role=LLMRole.FAST)

    # Оптимизация документов под token limit
    optimized_docs = optimize_documents_for_context(
        documents=state.reranked_documents,
        model_name=fast_llm.model_name,
        reserve_tokens=2000  # Для prompt + answer
    )

    # Форматирование контекста
    context = format_documents_with_sources(optimized_docs)

    # Формирование промпта
    prompt = QA_PROMPT_TEMPLATE.format(
        user_query=state.user_query,
        context=context
    )

    # Генерация ответа (со стримингом)
    answer_chunks = []
    async for chunk in fast_llm.astream(prompt):
        answer_chunks.append(chunk.content)
        # Стриминг в реальном времени
        await state.streaming_service.send_message({
            "type": "answer_chunk",
            "content": chunk.content
        })

    answer = "".join(answer_chunks)

    # Парсинг цитат и создание sources
    sources = extract_sources_from_answer(answer, optimized_docs)

    return {
        "answer": answer,
        "sources": sources
    }
```

**QA_PROMPT_TEMPLATE**:
```python
QA_PROMPT_TEMPLATE = """
You are a helpful research assistant. Answer the user's question based on the provided context.

## Context
{context}

## User's Question
{user_query}

## Instructions
- Answer the question accurately based on the provided context
- Cite your sources using [1], [2], etc. format
- If the context doesn't contain enough information, say so
- Be concise but comprehensive
- Use markdown formatting for structure

Answer:
"""
```

**Форматирование контекста**:
```python
def format_documents_with_sources(documents: list) -> str:
    """
    Форматирует документы с номерами источников.

    Output:
    [1] Document Title
    Content of document 1...

    [2] Another Document
    Content of document 2...
    """
    formatted = []
    for idx, doc in enumerate(documents, 1):
        formatted.append(f"[{idx}] {doc.title}")
        formatted.append(doc.content)
        formatted.append("")  # Пустая строка

    return "\n".join(formatted)
```

## 2. Document-Chunk Hierarchy (Data Graph)

### Концепция

Это явная иерархическая структура в базе данных, где Documents содержат Chunks.

### Database Schema

**Файл**: `surfsense_backend/app/db.py`

#### Document Table

```python
from sqlalchemy import Column, Integer, String, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

class Document(BaseModel, TimestampMixin):
    __tablename__ = "documents"

    # Primary Key
    id = Column(Integer, primary_key=True)

    # Content
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)  # Markdown контент

    # Type & Metadata
    document_type = Column(Enum(DocumentType), nullable=False)
    # DocumentType: FILE, SLACK, GITHUB, NOTION, etc.

    document_metadata = Column(JSON, nullable=True)
    # Метаданные источника (url, channel, author, date, etc.)

    # Embeddings
    embedding = Column(Vector(1536), nullable=True)  # Dimension зависит от модели
    # Vector(1536) для text-embedding-3-small
    # Vector(3072) для text-embedding-3-large

    # Hashes (для дедупликации)
    content_hash = Column(String(64), nullable=True)  # SHA-256
    unique_identifier_hash = Column(String(64), nullable=True)  # Источник-специфичный

    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    # One-to-Many: Document → Chunks

    # Ownership
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    search_space_id = Column(Integer, ForeignKey("search_spaces.id"), nullable=False)

    # Foreign Keys
    file_id = Column(Integer, ForeignKey("files.id"), nullable=True)
    connector_id = Column(Integer, ForeignKey("connectors.id"), nullable=True)

    # Timestamps (from TimestampMixin)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
```

#### Chunk Table

```python
class Chunk(BaseModel, TimestampMixin):
    __tablename__ = "chunks"

    # Primary Key
    id = Column(Integer, primary_key=True)

    # Content
    content = Column(Text, nullable=False)  # Сегмент контента

    # Embeddings
    embedding = Column(Vector(1536), nullable=True)  # Тот же dimension, что у Document

    # Relationship
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    document = relationship("Document", back_populates="chunks")
    # Many-to-One: Chunks → Document

    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
```

### Graph Visualization

```
┌─────────────────────────────────────────────────────────────┐
│                   DOCUMENT-CHUNK GRAPH                      │
└─────────────────────────────────────────────────────────────┘

   User
     │
     │ owns
     ▼
 SearchSpace ──────┐
     │             │
     │ contains    │ contains
     ▼             ▼
  Document ────> File/Connector
     │              (source)
     │ embedding: [0.1, 0.2, ...]
     │ content_hash: "abc123..."
     │ metadata: {...}
     │
     │ has_many
     ├─────────┬─────────┬─────────┐
     ▼         ▼         ▼         ▼
   Chunk1   Chunk2   Chunk3   ChunkN
   emb:[..] emb:[..] emb:[..] emb:[..]
```

**Узлы графа**:
- **User** - владелец знаний
- **SearchSpace** - пространство знаний (изоляция)
- **Document** - документ (узел верхнего уровня)
- **Chunk** - сегмент документа (узел контента)
- **File/Connector** - источник данных

**Ребра графа**:
- User → SearchSpace (ownership)
- SearchSpace → Document (containment)
- Document → Chunk (has_many)
- Document → File/Connector (source reference)

### Навигация по графу

#### Получение всех chunks документа:

```python
# ORM подход
document = await session.get(Document, document_id)
chunks = document.chunks  # Lazy loading или eager с joinedload

# SQL подход
chunks = await session.execute(
    select(Chunk).where(Chunk.document_id == document_id)
)
```

#### Получение parent document для chunk:

```python
chunk = await session.get(Chunk, chunk_id)
document = chunk.document  # Back-reference
```

#### Получение всех документов в SearchSpace:

```python
documents = await session.execute(
    select(Document).where(
        Document.search_space_id == search_space_id,
        Document.user_id == user_id
    )
)
```

## 3. Semantic Graph (Vector Space Graph)

### Концепция

Неявный граф, где узлы - это векторные представления (embeddings), а ребра - это семантическая близость (cosine similarity).

### Векторное пространство

**Размерность**: 1536 или 3072 (зависит от embedding модели)

**Каждый Document и Chunk**:
```python
embedding: Vector(1536) = [0.123, -0.456, 0.789, ..., 0.234]
```

### Semantic Similarity как ребра

**Cosine Similarity**:
```
similarity(A, B) = (A · B) / (||A|| * ||B||)

Диапазон: [-1, 1]
- 1: идентичные векторы
- 0: ортогональные (нет связи)
- -1: противоположные
```

**В pgvector** (PostgreSQL):
```sql
-- <=> оператор возвращает cosine distance (1 - similarity)
SELECT id, embedding <=> '[0.1, 0.2, ..., 0.3]' AS distance
FROM chunks
ORDER BY distance
LIMIT 10;
```

### Семантический граф в действии

**Пример запроса**: "How to optimize database performance?"

```
Query Embedding: [0.15, -0.23, 0.67, ...]

Semantic Graph (top 5 по similarity):

  Query
    │
    ├─── 0.92 similarity ──→ Chunk_1042 ("Database indexing strategies...")
    │
    ├─── 0.89 similarity ──→ Chunk_3214 ("Query optimization techniques...")
    │
    ├─── 0.85 similarity ──→ Chunk_5623 ("Performance tuning for PostgreSQL...")
    │
    ├─── 0.82 similarity ──→ Chunk_1893 ("Caching layers for databases...")
    │
    └─── 0.78 similarity ──→ Chunk_4421 ("Indexing best practices...")
```

**Связи** (ребра) определяются динамически при каждом запросе через векторный поиск.

### Multi-hop Semantic Traversal

SurfSense может выполнять многошаговую навигацию по семантическому графу:

**Пример**: Исследование темы "Machine Learning"

```
Step 1: Query "Machine Learning" → Top 10 chunks
Step 2: Для каждого chunk, найти похожие chunks (k=5)
Step 3: Объединить результаты, удалить дубликаты
Step 4: Rerank по релевантности к исходному запросу

Результат: Расширенное покрытие темы через semantic links
```

## Интеграция трех уровней графа

### Unified Graph View

```
┌────────────────────────────────────────────────────────────────┐
│              UNIFIED KNOWLEDGE GRAPH ARCHITECTURE              │
└────────────────────────────────────────────────────────────────┘

LEVEL 1: WORKFLOW GRAPH (LangGraph)
─────────────────────────────────────
  reformulate_query → search → rerank → answer → follow_up
        ↓                ↓         ↓        ↓
     [State propagation through workflow]

LEVEL 2: DATA GRAPH (Document-Chunk Hierarchy)
────────────────────────────────────────────────
  User → SearchSpace → Document → Chunk
                           ↓
                      [Relationships & Foreign Keys]

LEVEL 3: SEMANTIC GRAPH (Vector Space)
────────────────────────────────────────
  Query_embedding ←─[cosine similarity]─→ Chunk_embeddings
                              ↕
                    [Dynamic semantic links]

═══════════════════════════════════════════════════════════════

ORCHESTRATION:
  LangGraph Nodes → Query Semantic Graph → Traverse Data Graph → Return Results
```

### Практический пример: Full Query Flow

**User Query**: "Explain async programming in Python"

#### 1. Workflow Graph (LangGraph)

```
START
  │
  ▼ Node: reformulate_user_query
  reformulated: "Python asynchronous programming: asyncio, async/await, coroutines"
  │
  ▼ Node: handle_qna_workflow
  ├─→ ConnectorService.search_all_sources()
  │   └─→ hybrid_search() [переход к Semantic Graph]
  │
  ▼ Node: answer_question
  generated answer with citations
  │
  ▼ Node: generate_further_questions
  ["How to handle exceptions in async code?", ...]
  │
  ▼
  END
```

#### 2. Semantic Graph (Vector Search)

```python
# В hybrid_search()
query_embedding = embedding_model.embed("Python asynchronous programming...")
# → [0.12, -0.45, 0.78, ..., 0.23]

# Vector search
semantic_results = await session.execute(
    select(Chunk)
    .order_by(Chunk.embedding.op("<=>")(query_embedding))
    .limit(20)
)

# Результат: Top 20 chunks по semantic similarity
```

#### 3. Data Graph (Hierarchy Traversal)

```python
# Для каждого chunk, получить parent document
for chunk in semantic_results:
    document = await session.get(Document, chunk.document_id)
    # Получение метаданных источника
    metadata = document.document_metadata
    # Формирование source reference
    sources.append({
        "chunk_id": chunk.id,
        "document_id": document.id,
        "title": document.title,
        "type": document.document_type,
        "url": metadata.get("url"),
        "author": metadata.get("author")
    })
```

#### 4. Возврат в Workflow Graph

```python
# Обновление State
state.relevant_documents = documents
state.reranked_documents = reranked_documents
state.final_written_report = answer
state.sources = sources
```

## Резюме: Граф знаний в SurfSense

| Уровень | Тип | Узлы | Ребра | Использование |
|---------|-----|------|-------|---------------|
| **Workflow** | LangGraph | Функции обработки | State transitions | AI agent orchestration |
| **Data** | Relational DB | Documents, Chunks, Users | Foreign keys, Relationships | Структурирование контента |
| **Semantic** | Vector Space | Embeddings | Cosine similarity | Поиск по смыслу |

**Ключевые преимущества**:
1. **Явная структура** (Data Graph) для надежного хранения
2. **Семантические связи** (Semantic Graph) для интеллектуального поиска
3. **Workflow orchestration** (LangGraph) для сложных AI процессов

**Следующие документы**:
- `03-llm-agents.md` - детальное описание AI агентов
- `04-semantic-layer.md` - семантическая обработка и концепции
- `05-embeddings-search.md` - векторизация и поиск
