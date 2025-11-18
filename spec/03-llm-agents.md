# Языковые Агенты и LLM в SurfSense

## Введение

SurfSense использует архитектуру на основе языковых агентов для интеллектуальной обработки запросов, исследования документов и генерации ответов. Система поддерживает 20+ LLM провайдеров и использует специализированные роли для разных задач.

## Архитектура LLM в системе

### LLM Service

**Местоположение**: `surfsense_backend/app/services/llm_service.py`

**Основная задача**: Управление языковыми моделями, валидация конфигураций, выбор модели по роли.

```python
class LLMService:
    """
    Сервис для управления LLM конфигурациями и инстансами.
    """

    def get_llm(
        self,
        llm_config: LLMConfig,
        role: LLMRole = LLMRole.FAST
    ) -> BaseChatModel:
        """
        Получает LLM instance по роли.

        Args:
            llm_config: Конфигурация LLM пользователя
            role: Роль модели (FAST, LONG_CONTEXT, STRATEGIC)

        Returns:
            Инстанс ChatModel (через LiteLLM)
        """
        # Выбор модели по роли
        if role == LLMRole.LONG_CONTEXT:
            model_name = llm_config.long_context_model or llm_config.model_name
        elif role == LLMRole.STRATEGIC:
            model_name = llm_config.strategic_model or llm_config.model_name
        else:  # FAST
            model_name = llm_config.model_name

        # Создание LLM через LiteLLM
        from litellm import ChatCompletion

        return ChatCompletion(
            model=model_name,
            api_key=llm_config.api_key,
            api_base=llm_config.api_base,
            **llm_config.litellm_params
        )

    async def validate_llm_config(self, llm_config: LLMConfig) -> bool:
        """
        Валидирует LLM конфигурацию (проверка подключения).
        """
        try:
            llm = self.get_llm(llm_config, role=LLMRole.FAST)
            response = await llm.ainvoke("Test connection")
            return True
        except Exception as e:
            logger.error(f"LLM config validation failed: {str(e)}")
            return False
```

### LLM Roles (Роли моделей)

SurfSense использует **три специализированные роли** для разных задач:

```python
class LLMRole:
    """
    Роли языковых моделей в системе.
    """
    FAST = "fast"
    # Быстрые ответы, генерация контента
    # Используется: Q&A, генерация ответов
    # Примеры моделей: gpt-4o-mini, claude-3-haiku, gemini-1.5-flash

    LONG_CONTEXT = "long_context"
    # Обработка больших документов
    # Используется: Summarization, анализ длинных текстов
    # Примеры моделей: gpt-4o, claude-3-5-sonnet, gemini-1.5-pro

    STRATEGIC = "strategic"
    # Стратегические задачи, reasoning
    # Используется: Переформулирование запросов, генерация вопросов
    # Примеры моделей: gpt-4o, claude-3-opus, gemini-1.5-pro
```

### LLM Configuration

**Database Model**: `LLMConfig` (`surfsense_backend/app/db.py:308-331`)

```python
class LLMConfig(BaseModel):
    __tablename__ = "llm_configs"

    id = Column(Integer, primary_key=True)

    # Basic Info
    name = Column(String(200), nullable=False)  # User-friendly name
    provider = Column(Enum(LiteLLMProvider), nullable=False)
    # Provider: OpenAI, Anthropic, Google, Ollama, etc.

    # Model Settings
    model_name = Column(String(200), nullable=False)  # e.g., "gpt-4o"
    api_key = Column(String(500), nullable=True)  # Encrypted
    api_base = Column(String(500), nullable=True)  # Custom endpoint

    # Role-specific Models
    long_context_model = Column(String(200), nullable=True)
    strategic_model = Column(String(200), nullable=True)

    # Language & Parameters
    language = Column(String(50), nullable=False, default="English")
    litellm_params = Column(JSON, nullable=True)
    # Дополнительные параметры: temperature, max_tokens, top_p, etc.

    # Ownership
    user_id = Column(String, ForeignKey("users.id"), nullable=False)

    # Status
    is_active = Column(Boolean, default=True)
```

**Пример конфигурации**:
```python
llm_config = LLMConfig(
    name="My GPT-4o Config",
    provider=LiteLLMProvider.OPENAI,
    model_name="gpt-4o-mini",  # FAST role
    long_context_model="gpt-4o",  # LONG_CONTEXT role
    strategic_model="gpt-4o",  # STRATEGIC role
    api_key="sk-...",
    language="English",
    litellm_params={
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_p": 0.9
    }
)
```

## Поддерживаемые LLM провайдеры

**Enum**: `LiteLLMProvider` (`surfsense_backend/app/db.py:82-116`)

### Основные провайдеры

| Провайдер | Префикс модели | Примеры моделей |
|-----------|----------------|-----------------|
| **OpenAI** | `openai/` | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` |
| **Anthropic** | `anthropic/` | `claude-3-5-sonnet`, `claude-3-opus`, `claude-3-haiku` |
| **Google** | `gemini/` | `gemini-1.5-pro`, `gemini-1.5-flash` |
| **Groq** | `groq/` | `llama-3.1-70b`, `mixtral-8x7b` |
| **Ollama** | `ollama/` | `llama3`, `mistral`, `codellama` (локально) |

### Cloud провайдеры

| Провайдер | Описание |
|-----------|----------|
| **Azure OpenAI** | OpenAI модели через Azure |
| **AWS Bedrock** | Claude, Llama через AWS |
| **Vertex AI** | Google модели через GCP |

### Альтернативные провайдеры

| Провайдер | Регион | Модели |
|-----------|--------|--------|
| **DeepSeek** | Китай | `deepseek-chat`, `deepseek-coder` |
| **Moonshot** | Китай | `moonshot-v1-8k` (中文) |
| **Alibaba Qwen** | Китай | `qwen-turbo`, `qwen-plus` |
| **Zhipu** | Китай | `glm-4` (中文) |
| **XAI** | США | `grok-beta` |
| **Cohere** | Глобально | `command-r-plus` |
| **Mistral** | Европа | `mistral-large-latest` |

### Open-source & Self-hosted

| Провайдер | Описание |
|-----------|----------|
| **Ollama** | Локальный запуск open-source моделей |
| **vLLM** | High-performance inference server |
| **Together AI** | Hosted open-source модели |
| **Anyscale** | Managed LLM platform |

**Полный список** (20+ провайдеров):
- OpenAI, Anthropic, Google (Gemini), Azure OpenAI
- Groq, Cohere, Mistral, Ollama
- AWS Bedrock, Vertex AI
- DeepSeek, XAI, OpenRouter
- Moonshot, Alibaba Qwen, Zhipu
- Together AI, Anyscale, Replicate, HuggingFace
- Perplexity, AI21, Palm

## AI Агенты в SurfSense

### Иерархия агентов

```
┌────────────────────────────────────────────────────────┐
│                  RESEARCHER AGENT                      │
│  (Main Agent - Orchestrator)                          │
│                                                        │
│  Nodes:                                               │
│  • reformulate_user_query (Strategic LLM)             │
│  • handle_qna_workflow (delegates to Q&A SubAgent)    │
│  • generate_further_questions (Strategic LLM)         │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │          Q&A SUBAGENT                            │ │
│  │  (Specialized for answering questions)           │ │
│  │                                                   │ │
│  │  Nodes:                                          │ │
│  │  • rerank_documents (RerankerService)            │ │
│  │  • answer_question (Fast LLM + RAG)              │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

## Researcher Agent (Главный агент)

**Местоположение**: `surfsense_backend/app/agents/researcher/`

**Назначение**: Orchestration исследовательского процесса от запроса до финального ответа с follow-up вопросами.

### Агентная функция 1: Переформулирование запроса

**Node**: `reformulate_user_query`
**LLM Role**: `STRATEGIC`

**Задача**: Трансформировать пользовательский запрос с учетом контекста диалога в оптимизированный поисковый запрос.

**Процесс**:
```python
async def reformulate_user_query(state: State, config: RunnableConfig) -> dict:
    """
    Агентная функция: Переформулирование запроса.

    Input:
        - state.chat_history: История диалога
        - state.user_query: Исходный запрос

    Process:
        1. Анализ контекста из chat_history
        2. Извлечение ключевых тем и сущностей
        3. Формирование REFORMULATE_QUERY_PROMPT
        4. Вызов Strategic LLM (reasoning capabilities)
        5. Парсинг переформулированного запроса

    Output:
        - state.reformulated_query: Оптимизированный запрос

    LLM Parameters:
        - temperature: 0.3 (детерминированность)
        - max_tokens: 200 (краткость)

    Example:
        Input: "How about in JavaScript?"
        Context: Previous message was "Explain async programming in Python"
        Output: "Asynchronous programming in JavaScript: promises, async/await, event loop"
    """
```

**Query Service**: `surfsense_backend/app/services/query_service.py`

```python
async def reformulate_query_with_chat_history(
    user_query: str,
    chat_history: list[ChatMessage],
    llm: BaseChatModel
) -> str:
    """
    Переформулирует запрос с учетом истории чата.

    Capabilities:
    - Contextual Understanding: учет предыдущих сообщений
    - Entity Resolution: разрешение местоимений и ссылок
    - Query Expansion: добавление релевантных терминов
    - Specificity Enhancement: уточнение расплывчатых запросов
    """
    # Формирование контекста
    context = format_chat_history(chat_history)

    # Промпт для переформулирования
    prompt = REFORMULATE_QUERY_PROMPT.format(
        chat_history=context,
        user_query=user_query
    )

    # Вызов Strategic LLM
    response = await llm.ainvoke(
        prompt,
        temperature=0.3,  # Low temperature для consistency
        max_tokens=200
    )

    return response.content.strip()
```

**REFORMULATE_QUERY_PROMPT** (из `app/prompts/__init__.py`):
```python
REFORMULATE_QUERY_PROMPT = """
You are an expert at understanding conversational context and reformulating queries.

## Conversation History
{chat_history}

## User's Latest Query
{user_query}

## Task
Reformulate the user's query to be more specific, standalone, and optimized for semantic search.

## Guidelines
1. If the query is already clear and standalone, return it as-is
2. If the query references previous context (pronouns, "it", "that", "also"), incorporate that context explicitly
3. Expand abbreviations and resolve ambiguities
4. Add relevant technical terms if appropriate
5. Keep the reformulated query concise (1-2 sentences max)
6. Preserve the user's original intent and question type

## Output Format
Provide only the reformulated query, without explanations.

Reformulated Query:
"""
```

**Примеры работы**:

| Исходный запрос | Контекст | Переформулированный запрос |
|-----------------|----------|---------------------------|
| "How about in Go?" | Предыдущий: "Explain channels in Rust" | "Channels and concurrency patterns in Go programming language" |
| "What are the alternatives?" | Предыдущий: "Benefits of PostgreSQL" | "Alternative relational database systems to PostgreSQL" |
| "Can you explain more?" | Предыдущий: "What is Docker?" | "Detailed explanation of Docker containerization technology" |

### Агентная функция 2: Q&A Workflow

**Node**: `handle_qna_workflow`
**Delegation**: Q&A SubAgent

**Задача**: Найти релевантные документы и сгенерировать ответ с цитированиями.

**Процесс**:
```python
async def handle_qna_workflow(state: State, config: RunnableConfig) -> dict:
    """
    Агентная функция: Orchestration Q&A процесса.

    Input:
        - state.reformulated_query: Оптимизированный запрос
        - state.db_session: Database session
        - config.user_id, config.search_space_id

    Process:
        1. Поиск документов через ConnectorService
        2. Создание Q&A SubAgent инстанса
        3. Подготовка входных данных для SubAgent
        4. Запуск SubAgent workflow
        5. Получение результатов (answer + sources)

    Output:
        - state.relevant_documents: Найденные документы (top 20)
        - state.reranked_documents: Переранжированные (top 10)
        - state.final_written_report: Ответ с цитатами
        - state.sources: Метаданные источников

    Delegation to:
        Q&A SubAgent (см. секцию ниже)
    """
    from app.services.connector_service import ConnectorService
    from app.agents.researcher.qna_agent.graph import create_qna_graph

    # 1. Поиск документов
    connector_service = ConnectorService(state.db_session)

    # Поиск по всем источникам (files, urls, connectors)
    search_results = await connector_service.search_all_sources(
        user_query=state.reformulated_query,
        user_id=config.user_id,
        search_space_id=config.search_space_id,
        top_k=20,  # Top 20 для reranking
        search_mode=SearchMode.CHUNKS
    )

    # 2. Создание Q&A SubAgent
    qna_graph = create_qna_graph()

    # 3. Подготовка входных данных
    qna_input = {
        "user_query": state.reformulated_query,
        "relevant_documents": search_results.documents,
        "user_id": config.user_id,
        "search_space_id": config.search_space_id,
        "db_session": state.db_session,
        "streaming_service": state.streaming_service
    }

    # 4. Запуск SubAgent
    qna_result = await qna_graph.ainvoke(qna_input, config)

    # 5. Возврат результатов
    return {
        "relevant_documents": search_results.documents,
        "reranked_documents": qna_result["reranked_documents"],
        "final_written_report": qna_result["answer"],
        "sources": qna_result["sources"]
    }
```

### Агентная функция 3: Генерация follow-up вопросов

**Node**: `generate_further_questions`
**LLM Role**: `STRATEGIC`

**Задача**: Сгенерировать дополнительные вопросы для углубления исследования темы.

**Процесс**:
```python
async def generate_further_questions(state: State, config: RunnableConfig) -> dict:
    """
    Агентная функция: Генерация follow-up вопросов.

    Input:
        - state.reformulated_query: Исходный запрос
        - state.final_written_report: Сгенерированный ответ
        - state.sources: Использованные источники

    Process:
        1. Анализ ответа для определения gaps и смежных тем
        2. Формирование FURTHER_QUESTIONS_PROMPT
        3. Вызов Strategic LLM
        4. Парсинг и валидация вопросов
        5. Ранжирование по релевантности

    Output:
        - state.further_questions: list[str] (3-5 вопросов)

    LLM Parameters:
        - temperature: 0.7 (креативность)
        - max_tokens: 500

    Example:
        Input Query: "Explain async/await in Python"
        Answer: [detailed explanation of async/await]
        Output Questions:
        1. "How do you handle exceptions in asynchronous Python code?"
        2. "What are the performance benefits of async/await over threading?"
        3. "How to use asyncio with database connections?"
    """
    from app.prompts import FURTHER_QUESTIONS_PROMPT

    # Получение Strategic LLM
    user_llm_config = await get_user_llm_config(state.db_session, config.user_id)
    strategic_llm = llm_service.get_llm(user_llm_config, role=LLMRole.STRATEGIC)

    # Формирование промпта
    prompt = FURTHER_QUESTIONS_PROMPT.format(
        user_query=state.reformulated_query,
        answer=state.final_written_report,
        sources_summary=summarize_sources(state.sources)
    )

    # Генерация вопросов
    response = await strategic_llm.ainvoke(
        prompt,
        temperature=0.7,  # Higher temperature для разнообразия
        max_tokens=500
    )

    # Парсинг вопросов
    questions = parse_questions_from_response(response.content)

    # Валидация и ограничение (3-5 вопросов)
    questions = validate_and_limit_questions(questions, min_count=3, max_count=5)

    # Стриминг результата в UI
    await state.streaming_service.send_message({
        "type": "further_questions",
        "content": questions
    })

    return {"further_questions": questions}
```

**FURTHER_QUESTIONS_PROMPT**:
```python
FURTHER_QUESTIONS_PROMPT = """
You are a research assistant helping users explore topics in depth.

## User's Original Query
{user_query}

## Answer Provided
{answer}

## Sources Used
{sources_summary}

## Task
Generate 3-5 follow-up questions that would help the user:
1. Dive deeper into specific aspects of the topic
2. Explore related concepts or alternatives
3. Understand practical applications or implementations
4. Address potential challenges or edge cases

## Guidelines
- Each question should be specific and actionable
- Avoid yes/no questions
- Questions should be progressively deeper or explore different angles
- Use the sources to identify gaps or areas for expansion
- Keep questions concise (1-2 sentences each)

## Output Format
Provide a numbered list of questions:

1. [First question]
2. [Second question]
3. [Third question]
...

Follow-up Questions:
"""
```

## Q&A SubAgent (Специализированный агент)

**Местоположение**: `surfsense_backend/app/agents/researcher/qna_agent/`

**Назначение**: Специализированный агент для answering questions с RAG (Retrieval-Augmented Generation).

### Агентная функция 4: Reranking документов

**Node**: `rerank_documents`
**Service**: `RerankerService`

**Задача**: Переранжировать результаты поиска по релевантности к запросу.

**Процесс**:
```python
async def rerank_documents(state: QnAState, config: RunnableConfig) -> dict:
    """
    Агентная функция: Переранжирование документов.

    Input:
        - state.relevant_documents: list[Document] (top 20 из hybrid search)
        - state.user_query: str

    Process:
        1. Проверка, включен ли reranking (RERANKERS_ENABLED)
        2. Конвертация документов в RerankerDocument формат
        3. Вызов RerankerService (Cohere, Pinecone, etc.)
        4. Обновление scores в документах
        5. Сортировка по rerank_score
        6. Выбор top 10 для генерации ответа

    Output:
        - state.reranked_documents: list[Document] (top 10)

    Reranker Models:
        - Cohere rerank-english-v3.0
        - Pinecone rerank
        - Custom reranker models

    Performance:
        Улучшение precision на 15-30% по сравнению с hybrid search alone
    """
    from app.services.reranker_service import reranker_service
    from app.config import config

    # Проверка конфигурации
    if not config.RERANKERS_ENABLED:
        # Reranking отключен, используем top 10 из hybrid search
        logger.info("Reranking disabled, using top 10 from hybrid search")
        return {"reranked_documents": state.relevant_documents[:10]}

    logger.info(f"Reranking {len(state.relevant_documents)} documents")

    # Конвертация в RerankerDocument формат
    reranker_docs = [
        {
            "id": str(doc.id),
            "content": doc.content,
            "title": doc.title,
            "metadata": doc.document_metadata
        }
        for doc in state.relevant_documents
    ]

    # Reranking через RerankerService
    reranked_results = await reranker_service.rerank_documents(
        query_text=state.user_query,
        documents=reranker_docs
    )

    # Создание словаря scores
    doc_scores = {int(doc["id"]): doc["score"] for doc in reranked_results}

    # Обновление scores в оригинальных документах
    for doc in state.relevant_documents:
        if doc.id in doc_scores:
            doc.rerank_score = doc_scores[doc.id]
        else:
            doc.rerank_score = 0.0

    # Сортировка по rerank_score (descending)
    sorted_docs = sorted(
        state.relevant_documents,
        key=lambda x: x.rerank_score,
        reverse=True
    )

    # Выбор top 10
    top_docs = sorted_docs[:10]

    logger.info(f"Reranked to top {len(top_docs)} documents")

    return {"reranked_documents": top_docs}
```

**RerankerService** (`app/services/reranker_service.py`):
```python
class RerankerService:
    def __init__(self):
        self.reranker = self._initialize_reranker()

    def _initialize_reranker(self):
        """
        Инициализирует reranker на основе конфигурации.

        Supported:
        - Cohere (rerank-english-v3.0, rerank-multilingual-v3.0)
        - Pinecone (bge-reranker-v2-m3)
        - Custom models via HuggingFace
        """
        from rerankers import Reranker

        model_name = config.RERANKERS_MODEL_NAME
        model_type = config.RERANKERS_MODEL_TYPE

        if model_type == "cohere":
            return Reranker(model_name, api_key=config.COHERE_API_KEY)
        elif model_type == "pinecone":
            return Reranker(model_name)
        else:
            return Reranker(model_name)

    async def rerank_documents(
        self,
        query_text: str,
        documents: list[dict]
    ) -> list[dict]:
        """
        Переранжирует документы по релевантности.

        Returns:
            list[dict]: Документы с обновленными scores
            [
                {"id": "1", "content": "...", "score": 0.95},
                {"id": "2", "content": "...", "score": 0.87},
                ...
            ]
        """
        # Подготовка документов для reranker
        docs_text = [doc["content"] for doc in documents]

        # Reranking
        results = self.reranker.rank(
            query=query_text,
            docs=docs_text
        )

        # Форматирование результатов
        reranked = []
        for idx, result in enumerate(results):
            reranked.append({
                **documents[result.doc_id],
                "score": result.score
            })

        return reranked
```

### Агентная функция 5: Генерация ответа с RAG

**Node**: `answer_question`
**LLM Role**: `FAST`

**Задача**: Сгенерировать comprehensive ответ на вопрос с inline цитатами и источниками.

**Процесс**:
```python
async def answer_question(state: QnAState, config: RunnableConfig) -> dict:
    """
    Агентная функция: Генерация ответа с RAG.

    Input:
        - state.reranked_documents: list[Document] (top 10 переранжированных)
        - state.user_query: str

    Process:
        1. Оптимизация документов под token limit
        2. Форматирование контекста с источниками
        3. Формирование QA_PROMPT_TEMPLATE
        4. Вызов Fast LLM с streaming
        5. Парсинг цитат из ответа
        6. Создание sources metadata для UI

    Output:
        - state.answer: str (ответ с inline цитатами [1], [2], etc.)
        - state.sources: list[dict] (метаданные для UI)

    LLM Parameters:
        - temperature: 0.3 (accuracy over creativity)
        - max_tokens: 2000
        - streaming: True (real-time UI updates)

    RAG Components:
        - Retrieval: Hybrid search + Reranking
        - Augmentation: Context formatting + token optimization
        - Generation: LLM with citations
    """
    from app.prompts import QA_PROMPT_TEMPLATE
    from app.utils.document_converters import optimize_documents_for_context

    # Получение Fast LLM
    user_llm_config = await get_user_llm_config(state.db_session, state.user_id)
    fast_llm = llm_service.get_llm(user_llm_config, role=LLMRole.FAST)

    # 1. Оптимизация документов под token limit
    optimized_docs = optimize_documents_for_context(
        documents=state.reranked_documents,
        model_name=fast_llm.model_name,
        reserve_tokens=2500  # Для prompt (500) + answer (2000)
    )

    logger.info(f"Optimized {len(optimized_docs)} documents for context")

    # 2. Форматирование контекста с источниками
    context = format_documents_with_citations(optimized_docs)

    # 3. Формирование промпта
    prompt = QA_PROMPT_TEMPLATE.format(
        user_query=state.user_query,
        context=context,
        language=user_llm_config.language  # English, Chinese, etc.
    )

    # 4. Генерация ответа со streaming
    answer_chunks = []

    async for chunk in fast_llm.astream(
        prompt,
        temperature=0.3,
        max_tokens=2000
    ):
        answer_chunks.append(chunk.content)

        # Real-time streaming в UI
        await state.streaming_service.send_message({
            "type": "answer_chunk",
            "content": chunk.content
        })

    answer = "".join(answer_chunks)

    # 5. Парсинг цитат и создание sources
    sources = extract_sources_from_answer(answer, optimized_docs)

    logger.info(f"Generated answer with {len(sources)} sources")

    return {
        "answer": answer,
        "sources": sources
    }
```

**QA_PROMPT_TEMPLATE** (`app/prompts/__init__.py`):
```python
QA_PROMPT_TEMPLATE = """
You are a knowledgeable research assistant. Answer the user's question based on the provided context documents.

## Context Documents
{context}

## User's Question
{user_query}

## Instructions
1. **Answer Accuracy**: Base your answer strictly on the provided context
2. **Citations**: Cite your sources using [1], [2], [3] format after relevant statements
3. **Comprehensiveness**: Provide a detailed and complete answer
4. **Structure**: Use markdown formatting (headers, lists, code blocks) for clarity
5. **Objectivity**: Present information objectively without personal opinions
6. **Language**: Respond in {language}
7. **Limitations**: If the context doesn't contain sufficient information, acknowledge it

## Output Format
Provide a well-structured answer with:
- Clear introduction
- Main content with appropriate citations [N]
- Conclusion or summary if applicable
- Code examples if relevant (with proper syntax highlighting)

Answer:
"""
```

**Форматирование контекста**:
```python
def format_documents_with_citations(documents: list[Document]) -> str:
    """
    Форматирует документы с номерами для цитирования.

    Output:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    [1] Database Optimization Guide
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Source: technical_docs.pdf | Author: John Doe | Date: 2024-01-15

    Content:
    Database indexing is crucial for query performance...

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    [2] PostgreSQL Performance Tuning
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ...
    """
    formatted = []

    for idx, doc in enumerate(documents, 1):
        # Header
        formatted.append("━" * 50)
        formatted.append(f"[{idx}] {doc.title}")
        formatted.append("━" * 50)
        formatted.append("")

        # Metadata
        metadata = doc.document_metadata or {}
        meta_parts = []

        if "FILE_NAME" in metadata:
            meta_parts.append(f"Source: {metadata['FILE_NAME']}")
        if "AUTHOR" in metadata:
            meta_parts.append(f"Author: {metadata['AUTHOR']}")
        if "DATE" in metadata:
            meta_parts.append(f"Date: {metadata['DATE']}")

        if meta_parts:
            formatted.append(" | ".join(meta_parts))
            formatted.append("")

        # Content
        formatted.append("Content:")
        formatted.append(doc.content)
        formatted.append("")

    return "\n".join(formatted)
```

**Извлечение источников**:
```python
def extract_sources_from_answer(answer: str, documents: list[Document]) -> list[dict]:
    """
    Парсит цитаты из ответа и создает sources список.

    Input:
        answer: "Database indexing [1] improves performance [2]..."
        documents: [doc1, doc2, ...]

    Output:
        [
            {
                "citation_number": 1,
                "document_id": 123,
                "chunk_id": 456,
                "title": "Database Optimization Guide",
                "type": "FILE",
                "url": "https://...",
                "metadata": {...}
            },
            ...
        ]
    """
    import re

    # Парсинг цитат [1], [2], etc.
    citation_pattern = r'\[(\d+)\]'
    citations = set(re.findall(citation_pattern, answer))

    sources = []

    for citation in sorted(citations, key=int):
        citation_num = int(citation)

        if 1 <= citation_num <= len(documents):
            doc = documents[citation_num - 1]

            sources.append({
                "citation_number": citation_num,
                "document_id": doc.id,
                "chunk_id": getattr(doc, 'chunk_id', None),
                "title": doc.title,
                "type": doc.document_type.value,
                "url": doc.document_metadata.get("url"),
                "author": doc.document_metadata.get("AUTHOR"),
                "date": doc.document_metadata.get("DATE"),
                "metadata": doc.document_metadata
            })

    return sources
```

**Пример ответа с цитатами**:
```markdown
# Database Performance Optimization

Database performance optimization involves several key strategies [1]:

## Indexing
Creating appropriate indexes is crucial for query performance [1][2]. B-tree indexes are the default and work well for most queries, while hash indexes are optimized for equality comparisons [2].

## Query Optimization
Analyzing query execution plans helps identify bottlenecks [3]. Use EXPLAIN ANALYZE to see:
- Sequential scans vs index scans
- Join strategies
- Estimated vs actual rows

## Caching
Implementing caching layers reduces database load [4]:
1. Application-level caching (Redis, Memcached)
2. Query result caching
3. Database buffer pool tuning

For PostgreSQL specifically, shared_buffers should typically be set to 25% of system RAM [4].
```

## Резюме: Роли языковых агентов

| Агентная функция | LLM Role | Задача | Temperature | Ключевые возможности |
|------------------|----------|--------|-------------|----------------------|
| **reformulate_user_query** | STRATEGIC | Переформулирование запросов | 0.3 | Context understanding, entity resolution |
| **handle_qna_workflow** | N/A | Orchestration | N/A | Delegation, coordination |
| **generate_further_questions** | STRATEGIC | Генерация вопросов | 0.7 | Topic expansion, gap identification |
| **rerank_documents** | N/A | Reranking | N/A | Relevance scoring (Cohere, Pinecone) |
| **answer_question** | FAST | RAG generation | 0.3 | Citation, streaming, multilingual |

**Ключевые принципы**:
1. **Специализация**: Разные LLM роли для разных задач
2. **Модульность**: Агенты можно комбинировать и заменять
3. **Observability**: Streaming и logging для visibility
4. **Flexibility**: Поддержка 20+ LLM провайдеров
5. **Quality**: RAG + Reranking + Citations для accuracy
