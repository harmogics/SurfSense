# Библиотеки и инструменты для работы с промптами

## Введение

Документ описывает полный стек технологий, используемых в SurfSense для работы с языковыми моделями, промптами и семантическими цепочками обработки данных.

## LLM Frameworks & Orchestration

### LiteLLM (v1.77.5+)
**Назначение**: Унифицированный интерфейс к 25+ провайдерам LLM

```python
from litellm import acompletion, completion

# Синхронный вызов
response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)

# Асинхронный вызов
response = await acompletion(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)
```

**Поддерживаемые провайдеры**:
- **OpenAI**: gpt-4o, gpt-4o-mini, o1, o3-mini
- **Anthropic**: claude-3-5-sonnet, claude-3-opus, claude-3-haiku
- **Google**: gemini-2.5-pro, gemini-2.5-flash
- **DeepSeek**: deepseek-chat, deepseek-reasoner
- **xAI**: grok-3, grok-4
- **Azure OpenAI**: azure/gpt-4o
- **AWS Bedrock**: bedrock/anthropic.claude-3-5
- **Groq**: groq/llama-3.3-70b
- **Ollama**: ollama/llama3.2
- **20+ других провайдеров**

### LangChain (v0.3+)
**Назначение**: Фреймворк для построения LLM-приложений

```python
from langchain_litellm import ChatLiteLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Создание LLM
llm = ChatLiteLLM(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=2048
)

# Создание промпта
prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question: {question}"
)

# Создание цепочки
chain = prompt | llm

# Выполнение
result = await chain.ainvoke({"question": "What is AI?"})
```

**Ключевые компоненты**:
- `langchain-litellm` (v0.2.3+): LiteLLM интеграция
- `langchain-community` (v0.3.17+): Дополнительные утилиты
- `langchain-core`: Базовые абстракции

### LangGraph (v0.3.29+)
**Назначение**: Оркестрация агентов и управление состоянием

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Определение состояния
class AgentState(TypedDict):
    query: str
    documents: list
    answer: str

# Создание графа
workflow = StateGraph(AgentState)

# Добавление узлов
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("answer", generate_answer)

# Добавление рёбер
workflow.add_edge("START", "retrieve")
workflow.add_edge("retrieve", "answer")
workflow.add_edge("answer", "END")

# Компиляция и выполнение
app = workflow.compile()
result = await app.ainvoke({"query": "What is quantum computing?"})
```

**Функции**:
- StateGraph: Направленный граф состояний
- Conditional edges: Условные переходы
- Checkpointing: Сохранение состояния
- Streaming: Потоковая обработка
- Parallel execution: Параллельное выполнение узлов

## Document Processing

### Docling (v2.15.0+)
**Назначение**: Парсинг и извлечение структуры документов

```python
from docling import DocumentParser

parser = DocumentParser()
document = parser.parse("document.pdf")

# Извлечение структурированного содержимого
text = document.get_text()
tables = document.get_tables()
images = document.get_images()
```

**Поддерживаемые форматы**: PDF, DOCX, PPTX, HTML, Markdown

### Unstructured (v0.16.25+)
**Назначение**: Извлечение контента из неструктурированных документов

```python
from unstructured.partition.auto import partition

# Автоматическое определение типа и парсинг
elements = partition("document.pdf")

# Фильтрация по типу элемента
text_elements = [el for el in elements if el.category == "Text"]
table_elements = [el for el in elements if el.category == "Table"]
```

### Chonkie (v1.4.0+)
**Назначение**: Семантическое разбиение текста на чанки

```python
from chonkie import SemanticChunker

chunker = SemanticChunker(
    chunk_size=512,
    chunk_overlap=50,
    semantic_threshold=0.7
)

chunks = chunker.chunk(document_text)
```

## Embeddings & Vector Search

### Sentence Transformers (v3.4.1+)
**Назначение**: Генерация текстовых эмбеддингов

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([
    "First document",
    "Second document"
])
```

**Популярные модели**:
- `all-MiniLM-L6-v2`: Быстрая, 384 dimensions
- `all-mpnet-base-v2`: Баланс скорости/качества, 768d
- `multi-qa-mpnet-base-dot-v1`: Для Q&A задач

### pgvector (v0.3.6+)
**Назначение**: Векторный поиск в PostgreSQL

```sql
-- Создание таблицы с векторным столбцом
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(768)
);

-- Создание индекса для быстрого поиска
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);

-- Векторный поиск по схожести
SELECT content, 1 - (embedding <=> query_embedding) AS similarity
FROM documents
ORDER BY embedding <=> query_embedding
LIMIT 10;
```

```python
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, Text

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    content = Column(Text)
    embedding = Column(Vector(768))
```

## Search & Retrieval

### Rerankers (v0.7.1+)
**Назначение**: Переранжирование документов для повышения точности

```python
from rerankers import Reranker

reranker = Reranker("flashrank", model_name="ms-marco-MiniLM-L-12-v2")

results = reranker.rank(
    query="Python asyncio performance",
    docs=[
        {"text": "Document 1 content"},
        {"text": "Document 2 content"}
    ],
    top_k=5
)
```

**Поддерживаемые rerankers**:
- FlashRank: Быстрый cross-encoder
- Cohere Rerank: Cloud API
- Jina Reranker: Кастомные модели

### Tavily (v0.3.2+)
**Назначение**: Web search API для актуальной информации

```python
from tavily import TavilyClient

client = TavilyClient(api_key="...")
results = client.search(
    query="latest AI developments",
    search_depth="advanced",
    max_results=5
)
```

### Linkup SDK (v0.2.4+)
**Назначение**: Агрегация поисковых результатов

```python
from linkup import LinkupClient

client = LinkupClient(api_key="...")
results = client.search(query="quantum computing", limit=10)
```

### Elasticsearch (v9.1.1+)
**Назначение**: Full-text search и аналитика

```python
from elasticsearch import AsyncElasticsearch

es = AsyncElasticsearch(["http://localhost:9200"])

# Гибридный поиск (keyword + semantic)
results = await es.search(
    index="documents",
    query={
        "bool": {
            "should": [
                {"match": {"content": query}},
                {"knn": {"field": "embedding", "vector": query_vector}}
            ]
        }
    }
)
```

## Audio Processing

### Kokoro (v0.9.4+)
**Назначение**: Text-to-speech для подкастов

```python
from kokoro import KokoroTTS

tts = KokoroTTS()
audio = tts.synthesize(
    text="This is a podcast about quantum computing",
    voice="af_sky",
    speed=1.0
)
```

**Доступные голоса**:
- `af_sky`: Female voice
- `am_adam`: Male voice
- `af_bella`: Female (expressive)
- `am_michael`: Male (deep)

### Faster Whisper (v1.1.0+)
**Назначение**: Speech-to-text транскрипция

```python
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu")
segments, info = model.transcribe("audio.mp3")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

## Frontend Stack

### Vercel AI SDK (v4.3.19)
**Назначение**: React hooks для стриминга LLM ответов

```typescript
import { useChat } from 'ai/react';

function ChatComponent() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: '/api/chat',
    streamProtocol: 'text'
  });

  return (
    <div>
      {messages.map(m => (
        <div key={m.id}>{m.content}</div>
      ))}
      <form onSubmit={handleSubmit}>
        <input value={input} onChange={handleInputChange} />
      </form>
    </div>
  );
}
```

### Jotai (v2.15.1)
**Назначение**: Атомарное управление состоянием

```typescript
import { atom, useAtom } from 'jotai';

const messagesAtom = atom<Message[]>([]);
const currentQueryAtom = atom<string>('');

function ChatContainer() {
  const [messages, setMessages] = useAtom(messagesAtom);
  const [query, setQuery] = useAtom(currentQueryAtom);
  // ...
}
```

### TanStack Query (v5.90.7)
**Назначение**: Серверное состояние и кэширование

```typescript
import { useQuery, useMutation } from '@tanstack/react-query';

function useSearchDocuments(query: string) {
  return useQuery({
    queryKey: ['documents', query],
    queryFn: () => fetch(`/api/search?q=${query}`).then(r => r.json()),
    staleTime: 5 * 60 * 1000, // 5 minutes
    cacheTime: 10 * 60 * 1000
  });
}
```

## State Management & Databases

### PostgreSQL + SQLAlchemy
**Назначение**: Хранение данных и метаданных

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine = create_async_engine("postgresql+asyncpg://...")
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async with async_session() as session:
    result = await session.execute(select(Document))
    documents = result.scalars().all()
```

### Redis
**Назначение**: Кэширование и rate limiting

```python
from redis import asyncio as aioredis

redis = aioredis.from_url("redis://localhost")

# Кэширование результатов
await redis.setex("search:query123", 3600, json.dumps(results))
cached = await redis.get("search:query123")

# Rate limiting
pipe = redis.pipeline()
pipe.incr(f"ratelimit:{user_id}")
pipe.expire(f"ratelimit:{user_id}", 60)
count, _ = await pipe.execute()
```

## Token Management

### Tiktoken
**Назначение**: Подсчет токенов для OpenAI моделей

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def truncate_to_token_limit(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])
```

## Monitoring & Observability

### Логирование
```python
import structlog

logger = structlog.get_logger()

logger.info(
    "prompt_executed",
    prompt_type="qna_with_citations",
    model="gpt-4o",
    tokens_used=1234,
    latency_ms=567
)
```

### Метрики (опционально)
```python
from prometheus_client import Counter, Histogram

prompt_executions = Counter(
    'prompt_executions_total',
    'Total prompt executions',
    ['prompt_type', 'model']
)

prompt_latency = Histogram(
    'prompt_latency_seconds',
    'Prompt execution latency',
    ['prompt_type', 'model']
)
```

## Матрица совместимости промптов и библиотек

| Промпт | LiteLLM | LangChain | LangGraph | Rerankers | Tiktoken | Kokoro |
|--------|---------|-----------|-----------|-----------|----------|--------|
| Q&A with Citations | ✓ | ✓ | ✓ | - | ✓ | - |
| Q&A No Documents | ✓ | ✓ | ✓ | - | ✓ | - |
| Document Summarization | ✓ | ✓ | - | - | ✓ | - |
| Further Questions | ✓ | ✓ | ✓ | - | - | - |
| Query Reformulation | ✓ | ✓ | ✓ | - | - | - |
| Podcast Generation | ✓ | ✓ | ✓ | - | - | ✓ |
| Document Reranking | - | - | ✓ | ✓ | - | - |

## Версионирование и обновления

Все версии библиотек указаны в:
- `surfsense_backend/pyproject.toml` (Python dependencies)
- `surfsense_frontend/package.json` (TypeScript dependencies)

## См. также

- [Цепочки преобразований](transformation-chains.md) - Архитектура pipeline
- [Overview](overview.md) - Обзор всех типов промптов
