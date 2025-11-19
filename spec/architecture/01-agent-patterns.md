# Агентские архитектурные паттерны

## Введение

Агентская архитектура SurfSense построена на базе LangGraph и использует несколько ключевых паттернов проектирования для организации AI workflow. Эти паттерны обеспечивают гибкость, масштабируемость и поддерживаемость агентской системы.

## 1. State Machine Pattern (LangGraph StateGraph)

### Описание

State Machine Pattern используется для организации workflow AI агентов как конечного автомата с состояниями (nodes) и переходами (edges).

### Реализация в SurfSense

**Файл**: `surfsense_backend/app/agents/researcher/graph.py`

#### Main Researcher Agent Graph

```python
from langgraph.graph import StateGraph
from app.agents.researcher.state import State, Configuration
from app.agents.researcher import nodes

def build_graph():
    """
    Создает граф Researcher Agent как конечный автомат.

    Состояния (Nodes):
    - reformulate_user_query: трансформация запроса
    - handle_qna_workflow: выполнение Q&A
    - generate_further_questions: генерация follow-up вопросов

    Переходы (Edges):
    Линейный flow: start → reformulate → qna → questions → end
    """
    workflow = StateGraph(State, config_schema=Configuration)

    # Добавление узлов (состояний)
    workflow.add_node("reformulate_user_query", nodes.reformulate_user_query)
    workflow.add_node("handle_qna_workflow", nodes.handle_qna_workflow)
    workflow.add_node("generate_further_questions", nodes.generate_further_questions)

    # Определение рёбер (переходов)
    workflow.add_edge("__start__", "reformulate_user_query")
    workflow.add_edge("reformulate_user_query", "handle_qna_workflow")
    workflow.add_edge("handle_qna_workflow", "generate_further_questions")
    workflow.add_edge("generate_further_questions", "__end__")

    # Компиляция графа в исполняемый workflow
    graph = workflow.compile()
    return graph
```

**Визуализация State Machine**:
```
     ┌───────────┐
     │  __start__ │
     └─────┬─────┘
           │
           ▼
  ┌─────────────────────┐
  │ reformulate_query   │ [State: chat_history → reformulated_query]
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  handle_qna_workflow│ [State: reformulated_query → final_report]
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │ generate_questions  │ [State: final_report → further_questions]
  └─────────┬───────────┘
            │
            ▼
      ┌─────────┐
      │ __end__ │
      └─────────┘
```

#### Q&A SubAgent Graph

**Файл**: `surfsense_backend/app/agents/researcher/qna_agent/graph.py`

```python
from langgraph.graph import StateGraph
from app.agents.researcher.qna_agent.state import State, Configuration
from app.agents.researcher.qna_agent import nodes

def build_graph():
    """
    Q&A SubAgent - специализированный граф для ответов на вопросы.

    Состояния:
    - rerank_documents: переранжирование результатов поиска
    - answer_question: генерация ответа с RAG
    """
    workflow = StateGraph(State, config_schema=Configuration)

    workflow.add_node("rerank_documents", nodes.rerank_documents)
    workflow.add_node("answer_question", nodes.answer_question)

    workflow.add_edge("__start__", "rerank_documents")
    workflow.add_edge("rerank_documents", "answer_question")
    workflow.add_edge("answer_question", "__end__")

    graph = workflow.compile()
    return graph
```

### State Definition

**Файл**: `surfsense_backend/app/agents/researcher/state.py`

```python
from dataclasses import dataclass, field
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession

@dataclass
class State:
    """
    Состояние агента, передаваемое между узлами.

    Каждый узел может читать и модифицировать state,
    передавая результаты следующему узлу.
    """
    # Injected dependencies
    db_session: AsyncSession
    streaming_service: Any

    # State data (mutable across nodes)
    chat_history: list[Any] | None = field(default_factory=list)
    reformulated_query: str | None = field(default=None)
    further_questions: Any | None = field(default=None)
    reranked_documents: list[Any] | None = field(default=None)
    final_written_report: str | None = field(default=None)
```

### Преимущества паттерна

1. **Явная структура**: Граф визуализирует workflow
2. **Композиция**: Подграфы (subagents) можно встраивать друг в друга
3. **Отладка**: Каждое состояние можно инспектировать
4. **Масштабируемость**: Легко добавлять новые узлы и переходы
5. **Тестируемость**: Узлы тестируются независимо

### Пример использования

```python
# Создание и запуск графа
researcher_graph = build_graph()

# Начальное состояние
initial_state = State(
    db_session=session,
    streaming_service=streaming_service,
    chat_history=[{"role": "user", "content": "What is async programming?"}]
)

# Конфигурация
config = {
    "configurable": {
        "user_id": "user123",
        "search_space_id": 1,
        "user_query": "What is async programming?"
    }
}

# Выполнение графа (асинхронно)
async for chunk_type, chunk in researcher_graph.astream(initial_state, config):
    if "reformulated_query" in chunk:
        print(f"Reformulated: {chunk['reformulated_query']}")
    elif "final_written_report" in chunk:
        print(f"Answer: {chunk['final_written_report']}")
    elif "further_questions" in chunk:
        print(f"Questions: {chunk['further_questions']}")
```

---

## 2. Chain of Responsibility Pattern

### Описание

Chain of Responsibility Pattern организует обработку запроса через цепочку обработчиков (handlers), где каждый узел обрабатывает запрос и передает результат следующему.

### Реализация в SurfSense

**Файл**: `surfsense_backend/app/agents/researcher/nodes.py`

#### Node 1: reformulate_user_query

```python
async def reformulate_user_query(
    state: State,
    config: RunnableConfig,
    writer: StreamWriter
) -> dict:
    """
    Handler 1: Переформулирование запроса.

    Входные данные: state.chat_history
    Выходные данные: {"reformulated_query": str}
    Передача: Следующему узлу через обновленный state
    """
    configuration = Configuration.from_runnable_config(config)
    streaming_service = state.streaming_service

    # Получение LLM для стратегических задач
    strategic_llm = await get_user_strategic_llm(
        state.db_session,
        configuration.user_id,
        configuration.search_space_id
    )

    # Переформулирование запроса с учетом истории
    chat_history_str = await langchain_chat_history_to_str(state.chat_history)
    reformulated_query = await reformulate_query_with_chat_history(
        user_query=configuration.user_query,
        session=state.db_session,
        user_id=configuration.user_id,
        search_space_id=configuration.search_space_id,
        chat_history_str=chat_history_str
    )

    # Streaming прогресса
    writer({
        "yield_value": streaming_service.format_terminal_info_delta(
            f"🔄 Reformulated query: {reformulated_query}"
        )
    })

    # Передача результата следующему узлу
    return {"reformulated_query": reformulated_query}
```

#### Node 2: handle_qna_workflow

```python
async def handle_qna_workflow(
    state: State,
    config: RunnableConfig,
    writer: StreamWriter
) -> dict:
    """
    Handler 2: Выполнение Q&A workflow.

    Входные данные: state.reformulated_query (из предыдущего узла)
    Выходные данные: {"final_written_report": str}
    Передача: Следующему узлу через обновленный state
    """
    configuration = Configuration.from_runnable_config(config)
    streaming_service = state.streaming_service

    # Используем reformulated_query из предыдущего узла
    reformulated_query = state.reformulated_query

    # Поиск релевантных документов
    connector_service = ConnectorService(state.db_session, configuration.user_id)
    relevant_documents = await fetch_relevant_documents(
        research_questions=[reformulated_query],
        user_id=configuration.user_id,
        search_space_id=configuration.search_space_id,
        db_session=state.db_session,
        connectors_to_search=configuration.connectors_to_search,
        writer=writer,
        state=state,
        top_k=20
    )

    # Запуск Q&A SubAgent
    qna_agent_graph = build_qna_graph()
    qna_state = {
        "user_query": reformulated_query,
        "relevant_documents": relevant_documents,
        "db_session": state.db_session,
        "streaming_service": streaming_service
    }

    complete_content = ""
    async for chunk_type, chunk in qna_agent_graph.astream(qna_state, config):
        if "final_answer" in chunk:
            complete_content = chunk["final_answer"]
            # Stream answer chunks
            writer({"yield_value": streaming_service.format_text_chunk(chunk)})

    # Передача результата следующему узлу
    return {"final_written_report": complete_content}
```

#### Node 3: generate_further_questions

```python
async def generate_further_questions(
    state: State,
    config: RunnableConfig,
    writer: StreamWriter
) -> dict:
    """
    Handler 3: Генерация follow-up вопросов.

    Входные данные: state.final_written_report (из предыдущего узла)
    Выходные данные: {"further_questions": list}
    Передача: Финальный результат
    """
    configuration = Configuration.from_runnable_config(config)
    streaming_service = state.streaming_service

    # Используем final_written_report из предыдущего узла
    final_report = state.final_written_report
    reformulated_query = state.reformulated_query

    # Получение Strategic LLM
    strategic_llm = await get_user_strategic_llm(
        state.db_session,
        configuration.user_id,
        configuration.search_space_id
    )

    # Генерация вопросов на основе ответа
    from app.prompts import FURTHER_QUESTIONS_PROMPT

    prompt = FURTHER_QUESTIONS_PROMPT.format(
        user_query=reformulated_query,
        answer=final_report
    )

    response = await strategic_llm.ainvoke(prompt, temperature=0.7)
    further_questions = parse_questions_from_response(response.content)

    # Stream результата
    writer({
        "yield_value": streaming_service.format_further_questions_delta(
            further_questions
        )
    })

    # Финальный результат
    return {"further_questions": further_questions}
```

### Цепочка обработки

```
Request → Node 1 (reformulate) → Node 2 (qna) → Node 3 (questions) → Response

State flow:
1. chat_history → reformulated_query
2. reformulated_query → final_written_report
3. final_written_report → further_questions
```

### Преимущества паттерна

1. **Декаплинг**: Узлы независимы друг от друга
2. **Гибкость**: Легко добавлять/удалять узлы из цепочки
3. **Переиспользование**: Узлы можно использовать в разных графах
4. **Тестируемость**: Каждый узел тестируется изолированно
5. **Прозрачность**: State передается явно между узлами

---

## 3. Observer Pattern (Streaming)

### Описание

Observer Pattern реализует механизм подписки, где изменения в одном объекте (Subject) автоматически уведомляют подписчиков (Observers).

### Реализация в SurfSense

**Файл**: `surfsense_backend/app/services/streaming_service.py`

#### StreamingService (Subject)

```python
import json

class StreamingService:
    """
    Subject (Publisher) для streaming events.

    Типы событий:
    - TERMINAL_INFO: информационные сообщения
    - SOURCES: найденные источники
    - ANSWER: chunks ответа
    - FURTHER_QUESTIONS: follow-up вопросы
    """

    def __init__(self):
        self.terminal_idx = 1
        self.message_annotations = [
            {"type": "TERMINAL_INFO", "content": []},
            {"type": "SOURCES", "content": []},
            {"type": "ANSWER", "content": []},
            {"type": "FURTHER_QUESTIONS", "content": []},
        ]

    def format_terminal_info_delta(
        self,
        text: str,
        message_type: str = "info"
    ) -> str:
        """
        Publish terminal info event.

        Event format: Delta annotation для incremental updates
        """
        message = {
            "id": self.terminal_idx,
            "text": text,
            "type": message_type  # info, warning, error, success
        }
        self.terminal_idx += 1

        # Add to history
        self.message_annotations[0]["content"].append(message)

        # Format as delta
        annotation = {"type": "TERMINAL_INFO", "data": message}
        return f"8:[{json.dumps(annotation)}]\n"

    def format_sources_delta(self, sources: list[dict]) -> str:
        """
        Publish sources event.

        Sources structure:
        {
            "id": str,
            "title": str,
            "description": str,
            "url": str,
            "type": str (FILE, SLACK, GITHUB, etc.)
        }
        """
        self.message_annotations[1]["content"] = sources

        annotation = {"type": "SOURCES", "data": sources}
        return f"8:[{json.dumps(annotation)}]\n"

    def format_text_chunk(self, text: str) -> str:
        """
        Publish text chunk event (for streaming answers).

        Used for real-time answer generation.
        """
        return f"0:{json.dumps(text)}\n"

    def format_further_questions_delta(self, questions: list[str]) -> str:
        """
        Publish further questions event.
        """
        questions_data = [{"text": q} for q in questions]
        self.message_annotations[3]["content"] = questions_data

        annotation = {"type": "FURTHER_QUESTIONS", "data": questions_data}
        return f"8:[{json.dumps(annotation)}]\n"

    def format_final_message_annotations(self) -> str:
        """
        Publish final accumulated annotations.
        """
        return f"8:{json.dumps(self.message_annotations)}\n"
```

#### Usage in Agent Nodes (Publisher)

```python
# nodes.py - Publishing events
async def handle_qna_workflow(state: State, config, writer: StreamWriter):
    streaming_service = state.streaming_service  # Subject

    # Event 1: Terminal info
    writer({
        "yield_value": streaming_service.format_terminal_info_delta(
            "🔎 Starting research...",
            message_type="info"
        )
    })

    # Event 2: Sources found
    sources = [...]  # From search
    writer({
        "yield_value": streaming_service.format_sources_delta(sources)
    })

    # Event 3: Streaming answer (multiple chunks)
    async for chunk in llm.astream(prompt):
        writer({
            "yield_value": streaming_service.format_text_chunk(chunk.content)
        })

    # Event 4: Further questions
    questions = [...]
    writer({
        "yield_value": streaming_service.format_further_questions_delta(questions)
    })
```

#### Observer (Client-side)

```python
# Client code (frontend or API consumer)
async for event in agent_stream:
    event_type, data = event

    if event_type == "TERMINAL_INFO":
        # Update terminal UI
        console.log(data["text"])

    elif event_type == "SOURCES":
        # Update sources panel
        sources_panel.update(data)

    elif event_type == "ANSWER":
        # Stream answer to UI (character by character)
        answer_box.append(data)

    elif event_type == "FURTHER_QUESTIONS":
        # Display follow-up questions
        questions_panel.update(data)
```

### Sequence Diagram

```
Agent Node          StreamingService         Writer              Client (Observer)
    │                       │                   │                       │
    │─────format_terminal_info_delta()────────>│                       │
    │                       │                   │                       │
    │                       │<──────delta───────│                       │
    │                       │                   │                       │
    │                       │                   │──────yield_value─────>│
    │                       │                   │                       │
    │                       │                   │                   [Update UI]
    │                       │                   │                       │
    │─────format_sources_delta()──────────────>│                       │
    │                       │                   │──────yield_value─────>│
    │                       │                   │                       │
    │─────format_text_chunk() (multiple)──────>│                       │
    │                       │                   │──────yield_value─────>│
    │                       │                   │──────yield_value─────>│
    │                       │                   │──────yield_value─────>│
```

### Преимущества паттерна

1. **Real-time updates**: Клиент получает события мгновенно
2. **Декаплинг**: Agent nodes не знают о конкретных observers
3. **Масштабируемость**: Можно добавлять новых observers без изменения publisher
4. **Типизированные события**: Разные типы событий для разных UI компонентов
5. **Delta updates**: Эффективная передача только изменений

---

## 4. Command Pattern

### Описание

Command Pattern инкапсулирует запрос как объект, позволяя параметризовать клиентов с разными запросами, ставить запросы в очередь и поддерживать отмену операций.

### Реализация в SurfSense

**Файл**: `surfsense_backend/app/agents/researcher/nodes.py`

#### Connector Commands

```python
async def fetch_relevant_documents(
    research_questions: list[str],
    user_id: str,
    search_space_id: int,
    db_session: AsyncSession,
    connectors_to_search: list[str],  # Command types
    writer: StreamWriter = None,
    state: State = None,
    top_k: int = 10,
    connector_service: ConnectorService = None,
    search_mode: SearchMode = SearchMode.CHUNKS,
) -> list:
    """
    Command Executor: выполняет команды поиска по различным источникам.

    Каждый connector type - это отдельная команда с единым интерфейсом.
    """
    all_documents = []

    for connector in connectors_to_search:
        # Command 1: YouTube search
        if connector == "YOUTUBE_VIDEO":
            source_object, youtube_chunks = await connector_service.search_youtube(
                user_query=user_query,
                user_id=user_id,
                search_space_id=search_space_id,
                top_k=top_k,
                search_mode=search_mode
            )
            all_documents.extend(youtube_chunks)

        # Command 2: Extension search (browser history)
        elif connector == "EXTENSION":
            source_object, extension_chunks = await connector_service.search_extension(
                user_query=user_query,
                user_id=user_id,
                search_space_id=search_space_id,
                top_k=top_k,
                search_mode=search_mode
            )
            all_documents.extend(extension_chunks)

        # Command 3: Slack search
        elif connector == "SLACK_CONNECTOR":
            source_object, slack_chunks = await connector_service.search_slack(
                user_query=user_query,
                user_id=user_id,
                search_space_id=search_space_id,
                top_k=top_k,
                search_mode=search_mode
            )
            all_documents.extend(slack_chunks)

        # Command 4: Notion search
        elif connector == "NOTION_CONNECTOR":
            source_object, notion_chunks = await connector_service.search_notion(...)
            all_documents.extend(notion_chunks)

        # Command 5: GitHub search
        elif connector == "GITHUB_CONNECTOR":
            source_object, github_chunks = await connector_service.search_github(...)
            all_documents.extend(github_chunks)

        # Command 6: Linear search
        elif connector == "LINEAR_CONNECTOR":
            source_object, linear_chunks = await connector_service.search_linear(...)
            all_documents.extend(linear_chunks)

        # Command 7: Jira search
        elif connector == "JIRA_CONNECTOR":
            source_object, jira_chunks = await connector_service.search_jira(...)
            all_documents.extend(jira_chunks)

        # Command 8: Tavily API (external web search)
        elif connector == "TAVILY_API":
            source_object, tavily_results = await connector_service.search_tavily(...)
            all_documents.extend(tavily_results)

        # ... 10+ more commands

    return all_documents
```

#### Command Interface

Все команды имеют единый интерфейс:

```python
# Abstract Command Interface (implicit)
async def search_command(
    user_query: str,
    user_id: str,
    search_space_id: int,
    top_k: int,
    search_mode: SearchMode
) -> tuple[dict, list]:
    """
    Unified command interface for all connectors.

    Returns:
        tuple: (source_object, documents)
        - source_object: metadata о источнике
        - documents: список найденных документов/chunks
    """
    pass
```

#### Command Implementations

**ConnectorService** (Command Receiver):

```python
# services/connector_service.py
class ConnectorService:
    """
    Receiver: выполняет фактические операции для каждой команды.
    """

    async def search_slack(self, user_query, user_id, search_space_id, top_k):
        """Command: Search Slack messages"""
        chunks = await self.chunk_retriever.hybrid_search(
            query_text=user_query,
            top_k=top_k,
            user_id=user_id,
            search_space_id=search_space_id,
            document_type="SLACK_CONNECTOR"
        )

        source_object = {
            "id": "slack_source",
            "title": "Slack Messages",
            "type": "SLACK_CONNECTOR"
        }

        return source_object, chunks

    async def search_notion(self, user_query, user_id, search_space_id, top_k):
        """Command: Search Notion pages"""
        chunks = await self.chunk_retriever.hybrid_search(
            query_text=user_query,
            top_k=top_k,
            user_id=user_id,
            search_space_id=search_space_id,
            document_type="NOTION_CONNECTOR"
        )

        source_object = {
            "id": "notion_source",
            "title": "Notion Pages",
            "type": "NOTION_CONNECTOR"
        }

        return source_object, chunks

    # ... 15+ command implementations
```

### Command Configuration

```python
# Configuration determines which commands to execute
configuration = Configuration.from_runnable_config(config)

# User can select which connectors to search
connectors_to_search = configuration.connectors_to_search
# Example: ["SLACK_CONNECTOR", "NOTION_CONNECTOR", "GITHUB_CONNECTOR"]

# Execute selected commands
documents = await fetch_relevant_documents(
    research_questions=questions,
    connectors_to_search=connectors_to_search,  # Command list
    user_id=user_id,
    search_space_id=search_space_id,
    connector_service=connector_service
)
```

### Преимущества паттерна

1. **Инкапсуляция**: Каждая команда изолирована
2. **Параметризация**: Клиент может выбирать команды динамически
3. **Расширяемость**: Легко добавлять новые команды (connectors)
4. **Композиция**: Можно выполнять несколько команд параллельно
5. **Единый интерфейс**: Все команды возвращают стандартизированный результат

---

## Резюме: Агентские паттерны

| Паттерн | Файл | Назначение | Ключевые компоненты |
|---------|------|------------|---------------------|
| **State Machine** | `agents/researcher/graph.py` | Организация workflow | StateGraph, nodes, edges, State |
| **Chain of Responsibility** | `agents/researcher/nodes.py` | Последовательная обработка | reformulate → qna → questions |
| **Observer** | `services/streaming_service.py` | Real-time updates | StreamingService, events, writer |
| **Command** | `agents/researcher/nodes.py` | Инкапсуляция запросов | fetch_relevant_documents, connectors |

Эти паттерны создают гибкую, масштабируемую и maintainable агентскую архитектуру, которая легко расширяется новыми функциями и интеграциями.
