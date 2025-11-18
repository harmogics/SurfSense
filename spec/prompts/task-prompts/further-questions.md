# Генерация дополнительных вопросов (Further Questions Generation)

## Описание

Задачно-ориентированный промпт для генерации контекстуально релевантных follow-up вопросов на основе истории диалога и доступных документов. Промпт создает 3-5 вопросов, которые естественно продолжают беседу и предоставляют дополнительную ценность пользователю.

## Категория
**Task Prompt** - генерация контекстуальных вопросов для улучшения UX

## Расположение в кодовой базе
`surfsense_backend/app/agents/researcher/prompts.py::get_further_questions_system_prompt()`

## Текст промпта

```python
def get_further_questions_system_prompt():
    return f"""
Today's date: {datetime.datetime.now().strftime("%Y-%m-%d")}
<further_questions_system>
You are an expert research assistant specializing in generating contextually
relevant follow-up questions. Your task is to analyze the chat history and
available documents to suggest further questions that would naturally extend
the conversation and provide additional value to the user.

<input>
- chat_history: Provided in XML format within <chat_history> tags, containing
  <user> and <assistant> message pairs that show the chronological conversation flow.
- available_documents: Provided in XML format within <documents> tags, containing
  individual <document> elements with <metadata> and <content> sections.
</input>

<output_format>
A JSON object with the following structure:
{{
  "further_questions": [
    {{"id": 0, "question": "further qn 1"}},
    {{"id": 1, "question": "further qn 2"}}
  ]
}}
</output_format>

<instructions>
1. **Analyze Chat History:** Review the entire conversation flow to understand:
   - The main topics and themes discussed
   - The user's interests and areas of focus
   - Questions that have been asked and answered
   - Any gaps or areas that could be explored further
   - The depth level of the current discussion

2. **Evaluate Available Documents:** Consider the documents in context to identify:
   - Additional information that hasn't been explored yet
   - Related topics that could be of interest
   - Specific details or data points that could warrant deeper investigation
   - Cross-references or connections between different documents

3. **Generate Relevant Follow-up Questions:** Create 3-5 further questions that:
   - Are directly related to the ongoing conversation but explore new angles
   - Can be reasonably answered using the available documents or knowledge base
   - Progress the conversation forward rather than repeating previous topics
   - Match the user's apparent level of interest and expertise
   - Are specific and actionable rather than overly broad
   - Consider practical applications, comparisons, deeper analysis

4. **Ensure Question Quality:** Each question should:
   - Be clear and well-formulated
   - Provide genuine value to the user
   - Be distinct from other suggested questions
   - Be answerable within the current context
   - Encourage meaningful exploration of the topic

5. **Prioritize and Order:** Arrange questions by relevance and natural progression

6. **Adhere Strictly to Output Format:** Ensure valid JSON with correct field names
</instructions>
</further_questions_system>
"""
```

## Параметры

| Параметр | Тип | Описание |
|----------|-----|----------|
| `chat_history` | `str` | XML-форматированная история диалога |
| `documents` | `str` | XML-форматированные доступные документы |

## Роль в цепочке преобразований

### Позиция в Pipeline
**Финальный узел** в цепочке Researcher Agent

```
reformulate_query → retrieve_docs → rerank → answer_question → [GENERATE_FURTHER_QUESTIONS]
```

### Входящие данные
1. **Сгенерированный ответ** - для понимания текущего контекста
2. **История чата** - весь диалог с пользователем
3. **Доступные документы** - что еще можно исследовать

### Исходящие данные
1. **JSON массив вопросов** - отправляется на frontend
2. **Вопросы в UI** - отображаются как кликабельные кнопки

## Связи с другими промптами

### Зависит от
- **[Q&A with Citations](../system-prompts/qna-with-citations.md)** - после генерации ответа
- **[Chat History Integration](../context-prompts/chat-history-integration.md)** - использует историю

### Создает вход для
- **[Query Reformulation](query-reformulation.md)** - когда пользователь кликает на вопрос

## Используемые библиотеки и инструменты

### LLM & JSON Parsing
```python
from litellm import acompletion
import json
from pydantic import BaseModel, Field

# Structured output validation
class FurtherQuestion(BaseModel):
    id: int = Field(..., ge=0)
    question: str = Field(..., min_length=10)

class FurtherQuestionsResponse(BaseModel):
    further_questions: list[FurtherQuestion]

# Генерация с validation
async def generate_further_questions(
    chat_history: str,
    documents: str
) -> FurtherQuestionsResponse:
    response = await acompletion(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": get_further_questions_system_prompt()},
            {"role": "user", "content": f"<chat_history>{chat_history}</chat_history><documents>{documents}</documents>"}
        ],
        response_format={"type": "json_object"}
    )
    return FurtherQuestionsResponse.parse_raw(response.choices[0].message.content)
```

### Frontend Integration
```typescript
// React component для отображения вопросов
interface FurtherQuestion {
  id: number;
  question: string;
}

const FurtherQuestions: React.FC<{ questions: FurtherQuestion[] }> = ({ questions }) => {
  const { sendMessage } = useChat();

  return (
    <div className="further-questions">
      {questions.map(q => (
        <button
          key={q.id}
          onClick={() => sendMessage(q.question)}
          className="question-chip"
        >
          {q.question}
        </button>
      ))}
    </div>
  );
};
```

## Пример использования

### Входные данные
```xml
<chat_history>
<user>What are the best machine learning algorithms for text classification?</user>
<assistant>For text classification, several algorithms work well:
- Support Vector Machines (SVM) - Excellent for high-dimensional text data
- Naive Bayes - Simple, fast, works well with small datasets
- Neural Networks - Can capture intricate patterns
- Transformer models - State-of-the-art for most text classification tasks
</assistant>
</chat_history>

<documents>
<document>
<metadata>
<source_id>101</source_id>
<source_type>FILE</source_type>
</metadata>
<content>
Machine Learning for Text Classification: Performance Comparison
Transformer-based models achieve 95%+ accuracy on most benchmarks, while
traditional methods like SVM typically achieve 85-90% accuracy.

Dataset Considerations:
- Small datasets (< 1000 samples): Naive Bayes, SVM
- Large datasets (> 10,000 samples): Neural networks, transformers
- Imbalanced datasets: Require special handling with techniques like SMOTE
</content>
</document>
</documents>
```

### Выходные данные
```json
{
  "further_questions": [
    {
      "id": 0,
      "question": "What are the key differences in performance between traditional algorithms like SVM and modern deep learning approaches for text classification?"
    },
    {
      "id": 1,
      "question": "How do you handle imbalanced datasets when training text classification models?"
    },
    {
      "id": 2,
      "question": "What preprocessing techniques are most effective for improving text classification accuracy?"
    },
    {
      "id": 3,
      "question": "Are there specific domains or use cases where certain classification algorithms perform better than others?"
    }
  ]
}
```

## Конфигурация LLM

### Рекомендуемые модели
- **Fast LLM**: GPT-4o-mini, Claude 3.5 Haiku, Gemini Flash
- **Требования**: JSON mode, быстрый инференс
- **Контекстное окно**: 8K-32K токенов

### Параметры генерации
```python
llm_config = {
    "model": "gpt-4o-mini",
    "temperature": 0.8,  # Высокая креативность для разнообразных вопросов
    "max_tokens": 512,   # Небольшой размер для JSON
    "response_format": {"type": "json_object"},
    "top_p": 0.9,
}
```

## Метрики качества

### Критерии оценки
1. **Relevance** (90%+) - Связь с текущим диалогом
2. **Diversity** (85%+) - Разнообразие тем вопросов
3. **Answerability** (95%+) - Можно ли ответить на основе документов
4. **Click-through Rate** (30%+) - Пользователи кликают на вопросы
5. **Conversation Length** (+2-3 turns) - Продление диалога

## См. также

- [Q&A с цитированием](../system-prompts/qna-with-citations.md) - Предшествующий шаг
- [Query Reformulation](query-reformulation.md) - Обработка кликнутых вопросов
- [Цепочки преобразований](../transformation-chains.md) - Researcher Agent flow
