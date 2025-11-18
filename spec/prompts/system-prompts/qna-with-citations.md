# Q&A с цитированием источников (Q&A with Citations)

## Описание

Системный промпт для генерации детальных ответов на вопросы пользователя с обязательным цитированием источников информации. Промпт обеспечивает прозрачность и проверяемость ответов путем явного указания источника каждого факта в формате `[citation:source_id]`.

## Категория
**System Prompt** - определяет поведение и личность языковой модели

## Расположение в кодовой базе
`surfsense_backend/app/agents/researcher/qna_agent/prompts.py::get_qna_citation_system_prompt()`

## Текст промпта

```python
def get_qna_citation_system_prompt(
    chat_history: str | None = None,
    language: str | None = None
):
    return f"""
Today's date: {datetime.datetime.now().strftime("%Y-%m-%d")}
You are SurfSense, an advanced AI research assistant that provides detailed,
well-researched answers to user questions by synthesizing information from
multiple personal knowledge sources.{language_instruction}

{chat_history_section}

<knowledge_sources>
- EXTENSION: "Web content saved via SurfSense browser extension"
- CRAWLED_URL: "Webpages indexed by SurfSense web crawler"
- FILE: "User-uploaded documents (PDFs, Word, etc.)"
- SLACK_CONNECTOR: "Slack conversations and shared content"
- NOTION_CONNECTOR: "Notion workspace pages and databases"
- YOUTUBE_VIDEO: "YouTube video transcripts and metadata"
- GITHUB_CONNECTOR: "GitHub repository content and issues"
- ELASTICSEARCH_CONNECTOR: "Elasticsearch indexed documents and data"
- LINEAR_CONNECTOR: "Linear project issues and discussions"
- JIRA_CONNECTOR: "Jira project issues, tickets, and comments"
- CONFLUENCE_CONNECTOR: "Confluence pages and comments"
- CLICKUP_CONNECTOR: "ClickUp tasks and project data"
- GOOGLE_CALENDAR_CONNECTOR: "Google Calendar events, meetings"
- GOOGLE_GMAIL_CONNECTOR: "Google Gmail emails and conversations"
- DISCORD_CONNECTOR: "Discord server conversations"
- AIRTABLE_CONNECTOR: "Airtable records, tables, and database content"
- TAVILY_API: "Tavily search API results"
- LINKUP_API: "Linkup search API results"
- LUMA_CONNECTOR: "Luma events"
</knowledge_sources>

<instructions>
1. Review the chat history to understand the conversation context
2. Carefully analyze all provided documents in the <document> sections
3. Extract relevant information that directly addresses the user's question
4. Provide a comprehensive, detailed answer using information from personal knowledge sources
5. For EVERY piece of information, add citation: [citation:knowledge_source_id]
6. Make sure ALL factual statements have proper citations
7. If multiple documents support the same point, include all citations
8. Structure your answer logically and conversationally
9. Use your own words to synthesize ideas, but cite ALL information
10. If documents contain conflicting information, acknowledge this with citations
11. If question cannot be fully answered, clearly state what information is missing
12. Provide actionable insights when relevant
13. Use chat history for conversation continuity
14. CRITICAL: Use the exact source_id value from document metadata
15. CRITICAL: Every citation MUST be in format [citation:knowledge_source_id]
16. CRITICAL: Never modify source_id - use original values exactly
17. CRITICAL: Do not return citations as clickable links
18. CRITICAL: Never format as markdown links "([citation:5](https://...))"
19. CRITICAL: Citations ONLY as [citation:source_id] - no parentheses/hyperlinks
20. CRITICAL: Never make up source IDs
21. CRITICAL: If unsure about source_id, omit citation rather than guessing
22. CRITICAL: All knowledge sources contain personal information
23. CRITICAL: Be conversational and engaging while maintaining accuracy
</instructions>

<format>
- Clear, conversational tone for detailed Q&A discussions
- Comprehensive answers that thoroughly address the question
- Appropriate paragraphs and structure for readability
- Every fact must have citation: [citation:knowledge_source_id]
- Citations at end of sentence containing the information
- Multiple citations separated by commas: [citation:id1], [citation:id2]
- No references section needed - just citations in answer
- NEVER create your own citation format
- NEVER format citations as clickable links
- NEVER make up source IDs if unsure
- ALWAYS provide personalized answers reflecting user's context
- Thorough and detailed while focused on specific question
- Suggest follow-up questions at end if helpful
</format>
"""
```

## Параметры

| Параметр | Тип | Описание |
|----------|-----|----------|
| `chat_history` | `str \| None` | История диалога в XML формате `<chat_history>...</chat_history>` |
| `language` | `str \| None` | Язык ответа (опционально, добавляет языковую инструкцию) |
| `documents` | XML | Документы в формате `<documents><document><metadata>...</metadata><content>...</content></document></documents>` |

## Роль в цепочке преобразований

### Позиция в Pipeline
**Узел генерации ответа** (Answer Generation Node) в цепочке Researcher Agent

```
reformulate_query → retrieve_documents → rerank_documents → [ANSWER_QUESTION] → generate_further_questions
```

### Входящие данные
1. **Переформулированный запрос** от `reformulate_user_query` node
2. **Ранжированные документы** от `rerank_documents` node
3. **История чата** от state manager
4. **Языковые настройки** пользователя

### Исходящие данные
1. **Текст ответа с цитатами** - стриминг в реальном времени
2. **Список использованных source_id** - для отображения источников
3. **Обновленная история** - для контекста следующих вопросов

## Связи с другими промптами

### Прямые зависимости (входящие)
- **[Query Reformulation](../task-prompts/query-reformulation.md)** - улучшенная формулировка запроса
- **[Document Reranking](../retrieval-prompts/document-reranking.md)** - отфильтрованные релевантные документы
- **[Chat History Integration](../context-prompts/chat-history-integration.md)** - контекст диалога

### Прямые зависимости (исходящие)
- **[Further Questions](../task-prompts/further-questions.md)** - генерация follow-up вопросов на основе ответа

### Альтернативные промпты
- **[Q&A No Documents](qna-no-documents.md)** - используется когда документы не найдены

## Используемые библиотеки и инструменты

### LLM Frameworks
```python
# Core LLM integration
from litellm import acompletion  # v1.77.5+
from langchain_litellm import ChatLiteLLM  # v0.2.3+
from langgraph.graph import StateGraph  # v0.3.29+
```

### Semantic Processing
```python
# Token counting and truncation
from tiktoken import encoding_for_model  # Token optimization
from app.utils.token_counter import count_tokens_for_model

# Document formatting
from app.utils.document_formatter import format_documents_as_xml
```

### State Management
```python
# LangGraph state definition
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class QnAState(TypedDict):
    query: str
    documents: list[Document]
    chat_history: list[BaseMessage]
    answer: str
    language: str | None
```

### Streaming
```python
# Real-time response streaming
from langchain_core.runnables import RunnableConfig
stream = graph.astream(
    {"query": user_query},
    config={"configurable": {"thread_id": session_id}}
)
async for chunk in stream:
    # Stream answer to frontend
```

## Пример использования

### Входные данные
```xml
<chat_history>
<user>How does Python asyncio work?</user>
</chat_history>

<documents>
    <document>
        <metadata>
            <source_id>5</source_id>
            <source_type>GITHUB_CONNECTOR</source_type>
        </metadata>
        <content>
            Python's asyncio library provides tools for writing concurrent code
            using the async/await syntax. It's particularly useful for I/O-bound
            and high-level structured network code.
        </content>
    </document>

    <document>
        <metadata>
            <source_id>12</source_id>
            <source_type>YOUTUBE_VIDEO</source_type>
        </metadata>
        <content>
            Asyncio can improve performance by allowing other code to run while
            waiting for I/O operations. However, it's not suitable for CPU-bound
            tasks as it runs on a single thread.
        </content>
    </document>
</documents>
```

### Выходные данные
```markdown
Based on your GitHub repositories and video content, Python's asyncio library
provides tools for writing concurrent code using the async/await syntax [citation:5].
It's particularly useful for I/O-bound and high-level structured network
code [citation:5].

The key advantage of asyncio is that it can improve performance by allowing
other code to run while waiting for I/O operations to complete [citation:12].
This makes it excellent for scenarios like web scraping, API calls, database
operations, or any situation where your program spends time waiting for
external resources.

However, from your video learning, it's important to note that asyncio is not
suitable for CPU-bound tasks as it runs on a single thread [citation:12]. For
computationally intensive work, you'd want to use multiprocessing instead.

Would you like me to explain more about specific asyncio patterns or help you
determine if asyncio is right for a particular project you're working on?
```

## Конфигурация LLM

### Рекомендуемые модели
- **Fast LLM**: GPT-4o, Claude 3.5 Sonnet, Gemini 2.5 Flash
- **Стратегия**: Быстрый инференс для интерактивных ответов
- **Контекстное окно**: 128K+ токенов

### Параметры генерации
```python
llm_config = {
    "model": "gpt-4o",  # или модель из пользовательской конфигурации
    "temperature": 0.7,  # Баланс между креативностью и точностью
    "max_tokens": 2048,  # Максимальная длина ответа
    "stream": True,      # Потоковая генерация
    "top_p": 0.9,
}
```

### Оптимизация токенов
```python
# Динамическое усечение документов для соответствия context window
MAX_CONTEXT_TOKENS = 120000  # Резерв для системного промпта и ответа
truncated_docs = truncate_documents_to_token_limit(
    documents=ranked_documents,
    max_tokens=MAX_CONTEXT_TOKENS,
    model=llm_model_name
)
```

## Метрики качества

### Критерии оценки
1. **Citation Accuracy** (95%+) - Корректность source_id в цитатах
2. **Citation Coverage** (100%) - Все факты должны иметь цитаты
3. **Answer Relevance** (90%+) - Соответствие вопросу пользователя
4. **Factual Accuracy** (98%+) - Точность информации из документов
5. **Response Latency** (<3 sec) - Время до первого токена
6. **Token Efficiency** (70%+) - % токенов, используемых для контента

### Типичные ошибки и их предотвращение
```python
# ❌ Неверный формат цитат
"... asyncio library ([citation:5](https://github.com/repo))"

# ✅ Правильный формат
"... asyncio library [citation:5]"

# ❌ Придуманный source_id
"... asyncio library [citation:123]"  # Если source_id неизвестен

# ✅ Пропуск цитаты при неуверенности
"... asyncio library"  # Лучше пропустить, чем выдумать
```

## Мониторинг и логирование

```python
# Логирование использования промпта
logger.info(
    "QnA Citation Prompt executed",
    extra={
        "session_id": session_id,
        "num_documents": len(documents),
        "has_chat_history": bool(chat_history),
        "language": language,
        "query_length": len(user_query),
        "response_tokens": num_tokens_generated,
        "citations_count": len(extracted_citations),
        "latency_ms": response_time_ms
    }
)
```

## Версионирование

- **Текущая версия**: 2.0
- **Последнее обновление**: 2025-01
- **Изменения**:
  - v2.0: Добавлена поддержка 20+ коннекторов
  - v1.5: Улучшена система цитирования (строгий формат)
  - v1.0: Базовая версия с поддержкой FILES и EXTENSION

## См. также

- [Q&A без документов](qna-no-documents.md) - Альтернативный промпт для общих знаний
- [Генерация дополнительных вопросов](../task-prompts/further-questions.md) - Следующий шаг в pipeline
- [Интеграция истории чата](../context-prompts/chat-history-integration.md) - Контекстный паттерн
- [Цепочки преобразований](../transformation-chains.md) - Полная архитектура pipeline
