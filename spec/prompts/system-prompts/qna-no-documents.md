# Q&A без документов (Q&A No Documents)

## Описание

Системный промпт для генерации полезных ответов на вопросы пользователя в ситуациях, когда в персональной базе знаний не найдено релевантных документов. Промпт использует общие знания модели, сохраняя при этом контекст диалога и личность SurfSense.

## Категория
**System Prompt** - определяет поведение языковой модели при отсутствии персональных данных

## Расположение в кодовой базе
`surfsense_backend/app/agents/researcher/qna_agent/prompts.py::get_qna_no_documents_system_prompt()`

## Текст промпта

```python
def get_qna_no_documents_system_prompt(
    chat_history: str | None = None,
    language: str | None = None
):
    return f"""
Today's date: {datetime.datetime.now().strftime("%Y-%m-%d")}
You are SurfSense, an advanced AI research assistant that provides helpful,
detailed answers to user questions in a conversational manner.{language_instruction}

{chat_history_section}

<context>
The user has asked a question but there are no specific documents from their
personal knowledge base available to answer it. You should provide a helpful
response based on:
1. The conversation history and context
2. Your general knowledge and expertise
3. Understanding of the user's needs and interests based on our conversation
</context>

<instructions>
1. Provide a comprehensive, helpful answer to the user's question
2. Draw upon the conversation history to understand context and the user's needs
3. Use your general knowledge to provide accurate, detailed information
4. Be conversational and engaging, as if having a detailed discussion
5. Acknowledge when you're drawing from general knowledge rather than personal sources
6. Provide actionable insights and practical information when relevant
7. Structure your answer logically and clearly
8. If question would benefit from personalized info, gently suggest adding content to SurfSense
9. Be honest about limitations while still being maximally helpful
10. Maintain the helpful, knowledgeable tone that users expect from SurfSense
</instructions>

<format>
- Clear, conversational tone suitable for detailed Q&A discussions
- Comprehensive answers that thoroughly address the user's question
- Appropriate paragraphs and structure for readability
- No citations needed since you're using general knowledge
- Thorough and detailed while remaining focused on specific question
- Suggest follow-up questions at end if helpful
- When appropriate, mention that adding relevant content to SurfSense could provide
  more personalized answers
</format>
"""
```

## Параметры

| Параметр | Тип | Описание |
|----------|-----|----------|
| `chat_history` | `str \| None` | История диалога для поддержки контекста |
| `language` | `str \| None` | Язык ответа (опционально) |

## Роль в цепочке преобразований

### Позиция в Pipeline
**Альтернативный путь** в цепочке Researcher Agent при отсутствии документов

```
reformulate_query → retrieve_documents → [NO DOCUMENTS FOUND] → [ANSWER_WITH_GENERAL_KNOWLEDGE]
```

### Условие активации
```python
# Промпт активируется когда:
if len(retrieved_documents) == 0 or all_documents_irrelevant:
    system_prompt = get_qna_no_documents_system_prompt(
        chat_history=chat_history,
        language=user_language
    )
else:
    system_prompt = get_qna_citation_system_prompt(...)
```

### Входящие данные
1. **Пользовательский запрос** - оригинальный или переформулированный
2. **История чата** - для поддержки контекста разговора
3. **Языковые настройки** - для мультиязычных ответов

### Исходящие данные
1. **Текст ответа без цитат** - стриминг общих знаний
2. **Suggestion to add content** - рекомендация пополнить базу знаний
3. **Обновленная история** - для контекста следующих вопросов

## Связи с другими промптами

### Альтернатива к
- **[Q&A with Citations](qna-with-citations.md)** - основной промпт при наличии документов

### Общие зависимости
- **[Query Reformulation](../task-prompts/query-reformulation.md)** - улучшение запроса
- **[Chat History Integration](../context-prompts/chat-history-integration.md)** - контекст диалога
- **[Further Questions](../task-prompts/further-questions.md)** - генерация follow-up вопросов

## Используемые библиотеки и инструменты

### LLM Frameworks
```python
from litellm import acompletion
from langchain_litellm import ChatLiteLLM
from langgraph.graph import StateGraph
```

### Conditional Routing
```python
# LangGraph conditional edge для выбора промпта
def route_qna_workflow(state: QnAState) -> str:
    """Выбирает промпт в зависимости от наличия документов."""
    if not state.get("documents") or len(state["documents"]) == 0:
        return "answer_without_documents"
    return "answer_with_citations"

# Добавление условного перехода в граф
workflow.add_conditional_edges(
    "retrieve_documents",
    route_qna_workflow,
    {
        "answer_with_citations": "answer_question_with_docs",
        "answer_without_documents": "answer_question_no_docs"
    }
)
```

## Пример использования

### Входные данные
```xml
<chat_history>
<user>What is quantum computing?</user>
<assistant>Quantum computing is a type of computing that uses quantum mechanical
phenomena like superposition and entanglement to perform operations on data.</assistant>
<user>How does it compare to classical computing in terms of speed?</user>
</chat_history>

Documents: [] (no documents found in personal knowledge base)
```

### Выходные данные
```markdown
Based on our discussion about quantum computing, let me explain the speed comparison
with classical computing.

Quantum computers have the potential to solve certain types of problems exponentially
faster than classical computers. For specific tasks like factoring large numbers,
simulating quantum systems, or searching unsorted databases, quantum computers can
achieve dramatic speedups - potentially reducing problems that would take classical
computers thousands of years to solve down to hours or even minutes.

However, it's important to note that quantum computers aren't universally faster.
For many everyday computing tasks like word processing or browsing the web, classical
computers are actually more efficient. The quantum advantage appears primarily in
problems that involve:

1. Complex optimization challenges
2. Cryptographic calculations
3. Molecular and chemical simulations
4. Machine learning on certain datasets

Since I'm drawing from general knowledge here rather than your personal research or
notes, you might want to save any quantum computing resources you're studying to
your SurfSense knowledge base. That way, I can provide more personalized answers
based on your specific learning materials and research focus.

Would you like to explore any specific applications of quantum computing, or discuss
the challenges in building practical quantum computers?
```

## Конфигурация LLM

### Рекомендуемые модели
- **Fast LLM**: GPT-4o, Claude 3.5 Sonnet, Gemini 2.5 Flash
- **Стратегия**: Быстрый инференс, не требуется длинный контекст
- **Контекстное окно**: 8K-32K токенов (достаточно для истории чата)

### Параметры генерации
```python
llm_config = {
    "model": "gpt-4o-mini",  # Более экономичная модель для общих знаний
    "temperature": 0.7,       # Умеренная креативность
    "max_tokens": 1024,       # Средняя длина ответа
    "stream": True,
    "top_p": 0.9,
}
```

## Метрики качества

### Критерии оценки
1. **Relevance** (85%+) - Соответствие вопросу пользователя
2. **Helpfulness** (90%+) - Полезность общей информации
3. **Conversation Continuity** (95%+) - Использование контекста диалога
4. **Acknowledgment** (100%) - Признание использования общих знаний
5. **Engagement Rate** (70%+) - Пользователь продолжает диалог

### Отличия от Q&A with Citations
```python
# ✅ Правильное поведение
"Based on our discussion... (general knowledge explanation)"
"Since I'm drawing from general knowledge here..."
"You might want to save relevant resources to SurfSense..."

# ❌ Неправильное поведение
"Based on your documents... [citation:5]"  # НЕТ документов!
"According to your GitHub repo..."  # Не должно быть персонализации без данных
```

## Мониторинг и логирование

```python
logger.info(
    "QnA No Documents Prompt executed",
    extra={
        "session_id": session_id,
        "has_chat_history": bool(chat_history),
        "language": language,
        "query_length": len(user_query),
        "response_tokens": num_tokens_generated,
        "latency_ms": response_time_ms,
        "reason": "no_documents_found"  # Причина использования этого промпта
    }
)
```

## UX паттерны

### Визуальные индикаторы
```typescript
// Frontend показывает отличие от ответов с цитатами
interface MessageProps {
  hasDocuments: boolean;  // false для этого промпта
  citations?: Citation[];  // undefined для этого промпта
}

// UI может показывать badge "General Knowledge" или подобное
```

## См. также

- [Q&A с цитированием](qna-with-citations.md) - Основной промпт при наличии документов
- [Интеграция истории чата](../context-prompts/chat-history-integration.md) - Паттерн контекста
- [Цепочки преобразований](../transformation-chains.md) - Условные переходы в pipeline
