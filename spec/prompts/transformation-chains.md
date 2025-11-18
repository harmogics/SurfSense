# Цепочки преобразований (Transformation Chains)

## Введение

Документ описывает архитектуру семантических и аналитических цепочек преобразования данных в SurfSense. Все цепочки построены на базе LangGraph и представляют собой направленные графы состояний (StateGraph).

## Архитектурные принципы

### 1. State-Driven Architecture
Каждая цепочка управляет своим состоянием через TypedDict

### 2. Composability
Цепочки легко композируются и переиспользуются

### 3. Streaming-First
Все цепочки поддерживают потоковую обработку

### 4. Error Resilience
Обработка ошибок на каждом узле

## Основные цепочки преобразований

## 1. Researcher Agent Chain

### Назначение
Основная цепочка для ответов на вопросы пользователя с использованием персональной базы знаний.

### Архитектура
```
START
  ↓
reformulate_user_query
  ↓
handle_qna_workflow
  ├─→ retrieve_documents
  ├─→ rerank_documents
  └─→ answer_question (with/without docs)
  ↓
generate_further_questions
  ↓
END
```

### Определение состояния
```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class ResearcherState(TypedDict):
    # Input
    query: str
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: int
    language: str | None

    # Intermediate
    reformulated_query: str
    documents: list[Document]
    reranked_documents: list[Document]

    # Output
    answer: str
    citations: list[str]
    further_questions: list[dict]
```

### Реализация узлов
```python
from langgraph.graph import StateGraph

# Node 1: Query Reformulation
async def reformulate_user_query(state: ResearcherState) -> ResearcherState:
    """Улучшает запрос пользователя на основе истории."""
    chat_history = state["messages"][:-1]  # Все кроме последнего
    current_query = state["query"]

    reformulated = await reformulate_query(current_query, chat_history)

    return {"reformulated_query": reformulated}

# Node 2: QnA Workflow
async def handle_qna_workflow(state: ResearcherState) -> ResearcherState:
    """Retrieves docs, reranks, and generates answer."""
    query = state["reformulated_query"]
    user_id = state["user_id"]

    # Retrieve documents
    docs = await retrieve_documents(query, user_id, top_k=30)

    # Rerank for relevance
    if len(docs) > 0:
        reranked = await rerank_documents(query, docs, top_k=10)
    else:
        reranked = []

    # Generate answer
    if len(reranked) > 0:
        answer = await generate_answer_with_citations(
            query=query,
            documents=reranked,
            chat_history=state["messages"],
            language=state.get("language")
        )
    else:
        answer = await generate_answer_without_docs(
            query=query,
            chat_history=state["messages"],
            language=state.get("language")
        )

    return {
        "documents": docs,
        "reranked_documents": reranked,
        "answer": answer
    }

# Node 3: Further Questions
async def generate_further_questions(state: ResearcherState) -> ResearcherState:
    """Generates contextual follow-up questions."""
    chat_history = format_chat_history_as_xml(state["messages"])
    documents = format_documents_as_xml(state["reranked_documents"])

    questions = await generate_further_questions_prompt(
        chat_history=chat_history,
        documents=documents
    )

    return {"further_questions": questions}

# Build the graph
workflow = StateGraph(ResearcherState)

workflow.add_node("reformulate_user_query", reformulate_user_query)
workflow.add_node("handle_qna_workflow", handle_qna_workflow)
workflow.add_node("generate_further_questions", generate_further_questions)

workflow.add_edge("START", "reformulate_user_query")
workflow.add_edge("reformulate_user_query", "handle_qna_workflow")
workflow.add_edge("handle_qna_workflow", "generate_further_questions")
workflow.add_edge("generate_further_questions", "END")

researcher_agent = workflow.compile()
```

### Использование
```python
# Streaming execution
async for chunk in researcher_agent.astream({
    "query": "How does asyncio improve performance?",
    "messages": chat_history,
    "user_id": 123,
    "language": "en"
}):
    if "answer" in chunk:
        print(chunk["answer"], end="", flush=True)
```

## 2. QnA Sub-Agent Chain

### Назначение
Специализированная подцепочка для генерации ответов с цитированием.

### Архитектура
```
START
  ↓
rerank_documents
  ↓
answer_question
  ↓
END
```

### Определение состояния
```python
class QnAState(TypedDict):
    query: str
    documents: list[Document]
    chat_history: list[BaseMessage]
    language: str | None
    answer: str
    citations: list[str]
```

### Реализация
```python
async def rerank_documents_node(state: QnAState) -> QnAState:
    """Rerank documents for relevance."""
    reranker = DocumentReranker()
    reranked = await reranker.rerank_documents(
        query=state["query"],
        documents=state["documents"],
        top_k=10
    )
    return {"documents": reranked}

async def answer_question_node(state: QnAState) -> QnAState:
    """Generate answer with citations."""
    llm = ChatLiteLLM(model="gpt-4o", temperature=0.7, stream=True)

    # Format documents
    docs_xml = format_documents_as_xml(state["documents"])
    history_xml = format_chat_history_as_xml(state["chat_history"])

    # Get system prompt
    system_prompt = get_qna_citation_system_prompt(
        chat_history=history_xml,
        language=state.get("language")
    )

    # Generate answer
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"<documents>{docs_xml}</documents>\n\nQuestion: {state['query']}"}
    ]

    response = await llm.ainvoke(messages)
    answer = response.content

    # Extract citations
    citations = extract_citations_from_answer(answer)

    return {"answer": answer, "citations": citations}

# Build graph
qna_workflow = StateGraph(QnAState)
qna_workflow.add_node("rerank_documents", rerank_documents_node)
qna_workflow.add_node("answer_question", answer_question_node)
qna_workflow.add_edge("START", "rerank_documents")
qna_workflow.add_edge("rerank_documents", "answer_question")
qna_workflow.add_edge("answer_question", "END")

qna_agent = qna_workflow.compile()
```

## 3. Podcaster Agent Chain

### Назначение
Преобразование текстового контента в подкаст-диалог с аудио генерацией.

### Архитектура
```
START
  ↓
create_podcast_transcript
  ↓
create_merged_podcast_audio
  ↓
END
```

### Определение состояния
```python
class PodcasterState(TypedDict):
    # Input
    source_content: str
    user_instructions: str | None
    chat_history: list[BaseMessage]

    # Intermediate
    transcript: dict  # JSON transcript

    # Output
    audio_url: str
    duration_seconds: float
```

### Реализация
```python
async def create_podcast_transcript_node(state: PodcasterState) -> PodcasterState:
    """Generate podcast dialogue from source content."""
    llm = ChatLiteLLM(model="gpt-4o", temperature=0.85)

    prompt = get_podcast_generation_prompt(state.get("user_instructions"))

    response = await llm.ainvoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"<source_content>{state['source_content']}</source_content>"}
    ])

    transcript = json.loads(response.content)

    return {"transcript": transcript}

async def create_merged_podcast_audio_node(state: PodcasterState) -> PodcasterState:
    """Generate audio from transcript."""
    tts = KokoroTTS()
    audio_segments = []

    for dialog in state["transcript"]["podcast_transcripts"]:
        voice = "af_sky" if dialog["speaker_id"] == 0 else "am_adam"
        audio = await tts.synthesize(
            text=dialog["dialog"],
            voice=voice
        )
        audio_segments.append(audio)

    # Merge segments
    merged_audio = merge_with_pauses(audio_segments)

    # Upload to storage
    audio_url = await upload_audio(merged_audio)

    duration = len(merged_audio) / 24000  # 24kHz sample rate

    return {"audio_url": audio_url, "duration_seconds": duration}

# Build graph
podcaster_workflow = StateGraph(PodcasterState)
podcaster_workflow.add_node("create_podcast_transcript", create_podcast_transcript_node)
podcaster_workflow.add_node("create_merged_podcast_audio", create_merged_podcast_audio_node)
podcaster_workflow.add_edge("START", "create_podcast_transcript")
podcaster_workflow.add_edge("create_podcast_transcript", "create_merged_podcast_audio")
podcaster_workflow.add_edge("create_merged_podcast_audio", "END")

podcaster_agent = podcaster_workflow.compile()
```

## 4. Document Indexing Chain

### Назначение
Pipeline для индексации новых документов в систему.

### Архитектура
```
START
  ↓
parse_document
  ↓
generate_summary (parallel)
  ↓
chunk_document (parallel)
  ↓
generate_embeddings
  ↓
store_in_database
  ↓
END
```

### Определение состояния
```python
class DocumentIndexingState(TypedDict):
    # Input
    file_path: str
    user_id: int
    source_type: str

    # Intermediate
    parsed_content: str
    summary: str
    chunks: list[str]
    embeddings: list[list[float]]

    # Output
    document_id: int
    chunk_ids: list[int]
```

### Реализация
```python
async def parse_document_node(state: DocumentIndexingState) -> DocumentIndexingState:
    """Parse document to extract text."""
    parser = DocumentParser()
    content = await parser.parse(state["file_path"])

    return {"parsed_content": content.get_text()}

async def generate_summary_node(state: DocumentIndexingState) -> DocumentIndexingState:
    """Generate document summary."""
    llm = ChatLiteLLM(model="gpt-4o", temperature=0.3)

    prompt = SUMMARY_PROMPT_TEMPLATE.format(document=state["parsed_content"])

    response = await llm.ainvoke([{"role": "user", "content": prompt}])

    return {"summary": response.content}

async def chunk_document_node(state: DocumentIndexingState) -> DocumentIndexingState:
    """Chunk document into smaller pieces."""
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk(state["parsed_content"])

    return {"chunks": chunks}

async def generate_embeddings_node(state: DocumentIndexingState) -> DocumentIndexingState:
    """Generate embeddings for chunks."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(state["chunks"])

    return {"embeddings": embeddings.tolist()}

async def store_in_database_node(state: DocumentIndexingState) -> DocumentIndexingState:
    """Store document and chunks in database."""
    async with async_session() as session:
        # Create document
        doc = Document(
            user_id=state["user_id"],
            source_type=state["source_type"],
            content=state["parsed_content"],
            summary=state["summary"]
        )
        session.add(doc)
        await session.flush()

        # Create chunks
        chunk_ids = []
        for chunk_text, embedding in zip(state["chunks"], state["embeddings"]):
            chunk = DocumentChunk(
                document_id=doc.id,
                content=chunk_text,
                embedding=embedding
            )
            session.add(chunk)
            await session.flush()
            chunk_ids.append(chunk.id)

        await session.commit()

        return {"document_id": doc.id, "chunk_ids": chunk_ids}

# Build graph with parallel execution
indexing_workflow = StateGraph(DocumentIndexingState)

indexing_workflow.add_node("parse_document", parse_document_node)
indexing_workflow.add_node("generate_summary", generate_summary_node)
indexing_workflow.add_node("chunk_document", chunk_document_node)
indexing_workflow.add_node("generate_embeddings", generate_embeddings_node)
indexing_workflow.add_node("store_in_database", store_in_database_node)

indexing_workflow.add_edge("START", "parse_document")

# Parallel execution of summary and chunking
indexing_workflow.add_edge("parse_document", "generate_summary")
indexing_workflow.add_edge("parse_document", "chunk_document")

# Wait for both to complete
indexing_workflow.add_edge("chunk_document", "generate_embeddings")
indexing_workflow.add_edge("generate_embeddings", "store_in_database")
indexing_workflow.add_edge("generate_summary", "store_in_database")

indexing_workflow.add_edge("store_in_database", "END")

document_indexer = indexing_workflow.compile()
```

## Условные переходы (Conditional Edges)

### Пример: Выбор промпта Q&A
```python
def route_qna_prompt(state: ResearcherState) -> str:
    """Choose between citation and no-docs prompts."""
    if not state["reranked_documents"] or len(state["reranked_documents"]) == 0:
        return "answer_without_docs"
    return "answer_with_citations"

workflow.add_conditional_edges(
    "handle_qna_workflow",
    route_qna_prompt,
    {
        "answer_with_citations": "generate_further_questions",
        "answer_without_docs": "generate_further_questions"
    }
)
```

## Обработка ошибок

### Retry механизм
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def node_with_retry(state):
    # Node implementation
    pass
```

### Fallback стратегия
```python
async def safe_node_execution(state):
    try:
        return await primary_node(state)
    except Exception as e:
        logger.error(f"Primary node failed: {e}")
        return await fallback_node(state)
```

## Мониторинг цепочек

### Логирование выполнения
```python
import structlog

logger = structlog.get_logger()

async def logged_node(state):
    start_time = time.time()

    try:
        result = await node_logic(state)

        logger.info(
            "node_completed",
            node_name="reformulate_query",
            duration_ms=(time.time() - start_time) * 1000,
            state_keys=list(result.keys())
        )

        return result
    except Exception as e:
        logger.error(
            "node_failed",
            node_name="reformulate_query",
            error=str(e),
            duration_ms=(time.time() - start_time) * 1000
        )
        raise
```

## Диаграмма взаимодействия цепочек

```
User Query
    ↓
┌───────────────────────────────────┐
│     Researcher Agent Chain        │
│                                   │
│  1. Reformulate Query             │
│  2. Retrieve & Rerank Documents   │
│  3. Generate Answer               │
│  4. Generate Further Questions    │
└───────────────────────────────────┘
    ↓                    ↓
    ↓              ┌──────────────────┐
    ↓              │  QnA Sub-Agent   │
    ↓              │  - Rerank        │
    ↓              │  - Answer        │
    ↓              └──────────────────┘
    ↓
Document Upload
    ↓
┌───────────────────────────────────┐
│   Document Indexing Chain         │
│                                   │
│  1. Parse Document                │
│  2. Generate Summary (parallel)   │
│  3. Chunk Document (parallel)     │
│  4. Generate Embeddings           │
│  5. Store in DB                   │
└───────────────────────────────────┘
    ↓
Podcast Request
    ↓
┌───────────────────────────────────┐
│     Podcaster Agent Chain         │
│                                   │
│  1. Generate Transcript           │
│  2. Synthesize Audio (TTS)        │
│  3. Merge Audio Segments          │
└───────────────────────────────────┘
```

## Связь с промптами

| Цепочка | Использует промпты |
|---------|-------------------|
| Researcher Agent | Query Reformulation, Q&A with Citations, Q&A No Docs, Further Questions |
| QnA Sub-Agent | Q&A with Citations, Document Reranking |
| Podcaster Agent | Podcast Generation |
| Document Indexing | Document Summarization |

## См. также

- [Библиотеки и инструменты](libraries-and-tools.md) - Технологический стек
- [Overview](overview.md) - Обзор всех типов промптов
- [Все промпты](system-prompts/) - Детальное описание каждого промпта
