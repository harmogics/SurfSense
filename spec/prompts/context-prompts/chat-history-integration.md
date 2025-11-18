# Интеграция истории чата (Chat History Integration)

## Описание

Паттерн контекстной обработки для интеграции истории диалога во все промпты системы. Обеспечивает непрерывность беседы, разрешение кореференций и поддержку многошагового рассуждения.

## Категория
**Context Pattern** - архитектурный паттерн для работы с контекстом

## Формат хранения истории

### XML Format (для промптов)
```xml
<chat_history>
<user>What is quantum computing?</user>
<assistant>Quantum computing is a type of computing that uses quantum mechanical
phenomena like superposition and entanglement...</assistant>
<user>How does it compare to classical computing?</user>
<assistant>Quantum computers can solve certain problems exponentially faster...</assistant>
</chat_history>
```

### LangChain Message Format (в коде)
```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

chat_history = [
    HumanMessage(content="What is quantum computing?"),
    AIMessage(content="Quantum computing is a type of computing..."),
    HumanMessage(content="How does it compare to classical computing?"),
    AIMessage(content="Quantum computers can solve certain problems...")
]
```

## Роль в цепочке преобразований

### Используется повсеместно
История чата интегрируется во все промпты, требующие контекста:

```
[CHAT_HISTORY] → Query Reformulation
[CHAT_HISTORY] → Q&A with Citations
[CHAT_HISTORY] → Q&A No Documents
[CHAT_HISTORY] → Further Questions
```

## Используемые библиотеки и инструменты

### LangChain Message History
```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import PostgresChatMessageHistory

class ChatHistoryManager:
    def __init__(self, session_id: str, user_id: int):
        self.session_id = session_id
        self.user_id = user_id

        # PostgreSQL-based history storage
        self.history = PostgresChatMessageHistory(
            session_id=session_id,
            connection_string="postgresql://..."
        )

    async def add_user_message(self, content: str):
        """Add user message to history."""
        await self.history.aadd_messages([HumanMessage(content=content)])

    async def add_ai_message(self, content: str):
        """Add AI response to history."""
        await self.history.aadd_messages([AIMessage(content=content)])

    async def get_recent_history(self, limit: int = 10) -> list[BaseMessage]:
        """Get recent N messages."""
        messages = await self.history.aget_messages()
        return messages[-limit:] if len(messages) > limit else messages

    async def clear_history(self):
        """Clear all history for this session."""
        await self.history.aclear()
```

### LangGraph State Management
```python
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    documents: list[Document]

# LangGraph автоматически управляет историей через messages key
workflow = StateGraph(AgentState)

def process_node(state: AgentState) -> AgentState:
    # История доступна через state["messages"]
    chat_history = state["messages"]
    return {"messages": [AIMessage(content="Response")]}
```

### Форматирование для промптов
```python
def format_chat_history_as_xml(messages: list[BaseMessage]) -> str:
    """Convert LangChain messages to XML format for prompts."""
    if not messages:
        return "<chat_history>\nNO CHAT HISTORY PROVIDED\n</chat_history>"

    xml_lines = ["<chat_history>"]

    for msg in messages:
        if isinstance(msg, HumanMessage):
            xml_lines.append(f"<user>{msg.content}</user>")
        elif isinstance(msg, AIMessage):
            xml_lines.append(f"<assistant>{msg.content}</assistant>")
        elif isinstance(msg, SystemMessage):
            # System messages обычно не включаются в историю
            continue

    xml_lines.append("</chat_history>")
    return "\n".join(xml_lines)

# Использование в промптах
def get_qna_prompt_with_history(
    messages: list[BaseMessage],
    language: str | None = None
):
    history_xml = format_chat_history_as_xml(messages)
    # ... rest of prompt
```

## Связи с другими промптами

### Предоставляет контекст для
- **[Query Reformulation](../task-prompts/query-reformulation.md)** - разрешение кореференций
- **[Q&A with Citations](../system-prompts/qna-with-citations.md)** - поддержка диалога
- **[Q&A No Documents](../system-prompts/qna-no-documents.md)** - контекст без документов
- **[Further Questions](../task-prompts/further-questions.md)** - генерация релевантных вопросов

## Примеры использования

### Пример 1: Разрешение кореференций
```python
chat_history = [
    HumanMessage(content="Tell me about Python asyncio"),
    AIMessage(content="Python asyncio is a library for concurrent programming..."),
    HumanMessage(content="When should I use it?")  # "it" -> "asyncio"
]

# Query Reformulation использует историю:
# "When should I use it?" -> "When should I use Python asyncio?"
```

### Пример 2: Многошаговое рассуждение
```python
chat_history = [
    HumanMessage(content="What's the difference between SVM and neural networks?"),
    AIMessage(content="SVM works well for smaller datasets, neural networks for large..."),
    HumanMessage(content="Which one is faster?"),
    AIMessage(content="SVM is generally faster for training on small datasets..."),
    HumanMessage(content="And more accurate?")  # Контекст: "comparing SVM and neural networks"
]
```

### Пример 3: Персонализация ответов
```python
chat_history = [
    HumanMessage(content="I'm working on a text classification project"),
    AIMessage(content="Great! What type of text data are you working with?"),
    HumanMessage(content="Customer reviews for sentiment analysis"),
    # Следующие ответы будут персонализированы под этот контекст
]
```

## Стратегии управления историей

### Ограничение размера
```python
# Стратегия 1: Фиксированное количество сообщений
def get_limited_history(messages: list[BaseMessage], limit: int = 10):
    return messages[-limit:] if len(messages) > limit else messages

# Стратегия 2: Ограничение по токенам
def get_token_limited_history(
    messages: list[BaseMessage],
    max_tokens: int = 2000,
    model: str = "gpt-4o"
):
    from tiktoken import encoding_for_model
    enc = encoding_for_model(model)

    total_tokens = 0
    limited_messages = []

    # Идем с конца (самые свежие сообщения важнее)
    for msg in reversed(messages):
        msg_tokens = len(enc.encode(msg.content))
        if total_tokens + msg_tokens > max_tokens:
            break
        limited_messages.insert(0, msg)
        total_tokens += msg_tokens

    return limited_messages
```

### Суммаризация истории
```python
async def summarize_old_history(
    messages: list[BaseMessage],
    keep_recent: int = 5
) -> list[BaseMessage]:
    """Summarize old messages to save tokens."""
    if len(messages) <= keep_recent:
        return messages

    # Старые сообщения для суммаризации
    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]

    # Создаем саммари старой истории
    summary_prompt = f"""
    Summarize the following conversation history in 2-3 sentences,
    preserving key context and decisions:

    {format_chat_history_as_xml(old_messages)}
    """

    llm = ChatLiteLLM(model="gpt-4o-mini")
    summary = await llm.ainvoke([{"role": "user", "content": summary_prompt}])

    # Возвращаем саммари + свежие сообщения
    return [
        SystemMessage(content=f"Previous conversation summary: {summary.content}"),
        *recent_messages
    ]
```

## Хранение в базе данных

### Schema
```sql
CREATE TABLE chat_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE chat_messages (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    message_type VARCHAR(50) NOT NULL, -- 'human', 'ai', 'system'
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX idx_messages_session ON chat_messages(session_id);
CREATE INDEX idx_messages_created ON chat_messages(created_at DESC);
```

### SQLAlchemy Models
```python
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), unique=True, nullable=False)
    user_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, server_default="NOW()")
    updated_at = Column(DateTime, server_default="NOW()", onupdate="NOW()")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"))
    message_type = Column(String(50), nullable=False)  # 'human', 'ai', 'system'
    content = Column(Text, nullable=False)
    metadata = Column(JSON)
    created_at = Column(DateTime, server_default="NOW()")
```

## Метрики качества

### Критерии оценки
1. **Context Preservation** (95%+) - Сохранение контекста между ходами
2. **Coreference Resolution** (90%+) - Правильное разрешение "it", "this" и т.д.
3. **Conversation Coherence** (92%+) - Логичность диалога
4. **Memory Efficiency** - Оптимальное использование токенов
5. **Retrieval Speed** (<50ms) - Быстрая загрузка истории

## См. также

- [Query Reformulation](../task-prompts/query-reformulation.md) - Использует историю для улучшения запросов
- [Q&A with Citations](../system-prompts/qna-with-citations.md) - Интегрирует историю в ответы
- [Further Questions](../task-prompts/further-questions.md) - Анализирует историю для вопросов
- [Цепочки преобразований](../transformation-chains.md) - State management в LangGraph
