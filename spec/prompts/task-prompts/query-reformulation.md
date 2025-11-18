# Переформулировка запросов (Query Reformulation)

## Описание

Задачно-ориентированный промпт для улучшения пользовательских запросов на основе истории диалога. Промпт преобразует неясные, неполные или контекстно-зависимые запросы в четкие, автономные поисковые запросы, оптимизированные для извлечения релевантных документов.

## Категория
**Task Prompt** - оптимизация поисковых запросов

## Роль в цепочке преобразований

### Позиция в Pipeline
**Первый узел** в цепочке Researcher Agent

```
User Query → [REFORMULATE_QUERY] → retrieve_documents → rerank → answer_question
```

### Входящие данные
1. **Оригинальный запрос** пользователя
2. **История чата** - предыдущие сообщения для контекста
3. **Последние N сообщений** - обычно 5-10 последних обменов

### Исходящие данные
1. **Улучшенный запрос** - оптимизированный для поиска
2. **Расширенные ключевые слова** - для гибридного поиска

## Стратегии переформулировки

### 1. Разрешение кореференций
```python
# Пример
Original: "Tell me more about it"
Context: Previous discussion about "quantum computing"
Reformulated: "Tell me more about quantum computing"
```

### 2. Добавление контекста
```python
Original: "What are the advantages?"
Context: Discussion about "Python asyncio vs multiprocessing"
Reformulated: "What are the advantages of Python asyncio compared to multiprocessing?"
```

### 3. Конкретизация
```python
Original: "How does this work?"
Context: "machine learning text classification"
Reformulated: "How does machine learning text classification work?"
```

### 4. Расширение сокращений
```python
Original: "What's ML best for NLP tasks?"
Reformulated: "What machine learning algorithms are best for natural language processing tasks?"
```

## Используемые библиотеки и инструменты

### LLM Integration
```python
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage, AIMessage

async def reformulate_query(
    query: str,
    chat_history: list[BaseMessage]
) -> str:
    """Reformulates user query based on conversation history."""

    # Берем последние N сообщений
    recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history

    # Формируем промпт
    system_prompt = """
    You are a query reformulation expert. Your task is to transform the user's
    query into a clear, standalone search query that incorporates relevant context
    from the conversation history.

    Rules:
    1. Resolve pronouns and references (it, this, that, they) to specific entities
    2. Add missing context from previous messages
    3. Expand abbreviations and acronyms
    4. Make implicit comparisons explicit
    5. Keep the query concise and focused
    6. Maintain the user's original intent
    7. Return ONLY the reformulated query, nothing else

    If the query is already clear and standalone, return it unchanged.
    """

    # Форматируем историю
    history_str = "\n".join([
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}"
        for msg in recent_history
    ])

    user_prompt = f"""
    Conversation history:
    {history_str}

    Current query: {query}

    Reformulated query:
    """

    llm = ChatLiteLLM(model="gpt-4o-mini", temperature=0.3)
    response = await llm.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    return response.content.strip()
```

### State Management (LangGraph)
```python
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated

class ResearcherState(TypedDict):
    query: str
    reformulated_query: str
    chat_history: list[BaseMessage]
    documents: list[Document]

def reformulate_node(state: ResearcherState) -> ResearcherState:
    """Graph node for query reformulation."""
    reformulated = await reformulate_query(
        state["query"],
        state["chat_history"]
    )
    return {"reformulated_query": reformulated}

# Add to graph
workflow = StateGraph(ResearcherState)
workflow.add_node("reformulate_query", reformulate_node)
workflow.add_edge("START", "reformulate_query")
workflow.add_edge("reformulate_query", "retrieve_documents")
```

## Связи с другими промптами

### Предоставляет вход для
- **Document Retrieval** - улучшенный запрос для поиска
- **[Document Reranking](../retrieval-prompts/document-reranking.md)** - более точное ранжирование
- **[Q&A with Citations](../system-prompts/qna-with-citations.md)** - лучший контекст для ответа

### Зависит от
- **[Chat History Integration](../context-prompts/chat-history-integration.md)** - формат истории

## Пример использования

### Входные данные
```python
chat_history = [
    HumanMessage(content="What is asyncio in Python?"),
    AIMessage(content="Asyncio is a library for writing concurrent code using async/await syntax..."),
    HumanMessage(content="When should I use it?"),
]

current_query = "What are the limitations?"
```

### Выходные данные
```python
reformulated_query = "What are the limitations of Python asyncio?"
```

### Сложный пример
```python
chat_history = [
    HumanMessage(content="I'm comparing SVM and neural networks for text classification"),
    AIMessage(content="SVM works well for smaller datasets, while neural networks excel with large datasets..."),
    HumanMessage(content="Which one is faster?"),
]

current_query = "And more accurate?"

# Reformulated
reformulated_query = "Which is more accurate for text classification: SVM or neural networks?"
```

## Конфигурация LLM

### Рекомендуемые модели
- **Fast LLM**: GPT-4o-mini, Claude 3.5 Haiku
- **Требования**: Быстрый инференс, понимание контекста
- **Контекстное окно**: 8K-16K токенов

### Параметры генерации
```python
llm_config = {
    "model": "gpt-4o-mini",
    "temperature": 0.3,  # Низкая температура для точности
    "max_tokens": 128,   # Короткие запросы
    "top_p": 0.9,
}
```

## Метрики качества

### Критерии оценки
1. **Context Preservation** (95%+) - Сохранение намерения пользователя
2. **Clarity Improvement** (85%+) - Улучшение четкости запроса
3. **Retrieval Quality** (+20% precision) - Улучшение качества поиска
4. **Latency** (<500ms) - Быстрая обработка

## Оптимизации

### Кэширование
```python
from functools import lru_cache
import hashlib

def get_query_hash(query: str, history: list) -> str:
    """Create hash for caching."""
    history_str = "".join([msg.content for msg in history[-5:]])
    return hashlib.md5(f"{query}{history_str}".encode()).hexdigest()

# Redis cache
from redis import asyncio as aioredis

async def reformulate_with_cache(query: str, history: list) -> str:
    cache_key = f"reformulated:{get_query_hash(query, history)}"

    # Try cache first
    cached = await redis.get(cache_key)
    if cached:
        return cached.decode()

    # Reformulate
    result = await reformulate_query(query, history)

    # Cache for 1 hour
    await redis.setex(cache_key, 3600, result)

    return result
```

## См. также

- [Q&A с цитированием](../system-prompts/qna-with-citations.md) - Использует улучшенный запрос
- [Chat History Integration](../context-prompts/chat-history-integration.md) - Источник контекста
- [Document Reranking](../retrieval-prompts/document-reranking.md) - Следующий шаг
- [Цепочки преобразований](../transformation-chains.md) - Полный pipeline
