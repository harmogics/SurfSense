# Детальный поток обработки чата (Chat Flow)

## Введение

Поток обработки чата является центральным UX-сценарием в SurfSense. Этот документ описывает полный жизненный цикл запроса пользователя от момента ввода текста в UI до получения streaming ответа с источниками и дополнительными вопросами.

## Полный E2E поток

### Фаза 1: Инициализация чата (Frontend)

**Компонент**: `surfsense_web/app/dashboard/[search_space_id]/researcher/[[...chat_id]]/page.tsx`

```typescript
// 1. Пользователь открывает страницу researcher
// URL: /dashboard/123/researcher или /dashboard/123/researcher/456

const ResearcherPage = ({ params }) => {
  const { search_space_id, chat_id } = params;
  const isNewChat = !chat_id;

  // 2. Инициализация useChat hook
  const handler = useChat({
    api: `${BACKEND_URL}/api/v1/chat`,
    streamProtocol: "data",  // Vercel AI SDK SSE protocol

    // 3. Настройка headers с authentication
    headers: {
      Authorization: `Bearer ${token}`,
    },

    // 4. Дополнительные параметры в body.data
    body: {
      data: {
        search_space_id: search_space_id,
        selected_connectors: connectorTypes,      // ["YOUTUBE_VIDEO", "FILE", ...]
        research_mode: researchMode,              // "QNA"
        search_mode: searchMode,                  // "DOCUMENTS" | "CHUNKS"
        document_ids_to_add_in_context: docIds,  // [1, 2, 3]
        top_k: topK,                              // 5
      },
    },
  });

  // 5. Загрузка существующего чата (если chat_id существует)
  useEffect(() => {
    if (!isNewChat && chatIdParam) {
      loadExistingChat(chatIdParam);
    }
  }, [chatIdParam]);
};
```

**Загрузка существующего чата**:
```typescript
const loadExistingChat = async (chatId: string) => {
  // API запрос для загрузки истории
  const response = await fetch(`${BACKEND_URL}/api/v1/chats/${chatId}`, {
    headers: { Authorization: `Bearer ${token}` }
  });

  const chatData = await response.json();

  // Преобразование в формат AI SDK
  const messages = chatData.messages.map(msg => ({
    id: msg.id,
    role: msg.role,
    content: msg.content,
    annotations: msg.annotations,
  }));

  // Установка messages в useChat state
  handler.setMessages(messages);
};
```

### Фаза 2: Пользовательский ввод

**Компонент**: `surfsense_web/components/chat/ChatInput.tsx`

```typescript
const ChatInputUI = ({ onSubmit }) => {
  const [input, setInput] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    // 1. Проверка валидности
    if (!input.trim()) return;

    // 2. Для нового чата - создать chat entity first
    if (isNewChat) {
      const newChatId = await createNewChat(input);

      // 3. Сохранить состояние конфигурации в localStorage
      storeChatState(search_space_id, newChatId, {
        selectedDocuments,
        selectedConnectors,
        searchMode,
        researchMode,
        topK,
      });

      // 4. Навигация на новый chat URL
      router.replace(`/dashboard/${search_space_id}/researcher/${newChatId}`);
    }

    // 5. Отправка сообщения через useChat
    handler.append({
      role: "user",
      content: input,
    });

    // 6. Очистка input
    setInput("");
  };

  return (
    <form onSubmit={handleSubmit}>
      <textarea value={input} onChange={(e) => setInput(e.target.value)} />
      <button type="submit">Send</button>
    </form>
  );
};
```

**Создание нового чата**:
```typescript
const createNewChat = async (firstMessage: string): Promise<string> => {
  const response = await fetch(`${BACKEND_URL}/api/v1/chats`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      name: firstMessage.substring(0, 50),  // First 50 chars as title
      search_space_id: search_space_id,
      research_mode: researchMode,
    }),
  });

  const newChat = await response.json();
  return newChat.id.toString();
};
```

### Фаза 3: Отправка запроса (Frontend → Backend)

**AI SDK Internal Process**:

```typescript
// useChat.append() внутренне выполняет:

// 1. Добавляет сообщение user в локальный state
const newMessages = [...messages, { role: "user", content: input }];

// 2. Устанавливает status = "streaming"
setStatus("streaming");

// 3. Делает POST запрос
const response = await fetch(api, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    ...customHeaders,
  },
  body: JSON.stringify({
    messages: newMessages,
    data: body.data,  // Наши дополнительные параметры
  }),
});

// 4. Открывает SSE connection
const reader = response.body.getReader();
const decoder = new TextDecoder();

// 5. Читает stream chunks
while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  processStreamChunk(chunk);  // Парсинг AI SDK protocol
}
```

### Фаза 4: Обработка на Backend

**Endpoint**: `surfsense_backend/app/routes/chats_routes.py:handle_chat_data`

```python
@router.post("/chat")
async def handle_chat_data(
    request: AISDKChatRequest,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
) -> StreamingResponse:
    """
    Request format:
    {
        "messages": [
            {"role": "user", "content": "What is..."},
            {"role": "assistant", "content": "Based on..."}
        ],
        "data": {
            "search_space_id": 123,
            "selected_connectors": ["YOUTUBE_VIDEO", "FILE"],
            "search_mode": "DOCUMENTS",
            "document_ids_to_add_in_context": [1, 2],
            "top_k": 5
        }
    }
    """

    # 1. Извлечение параметров
    messages = request.messages
    data = request.data
    search_space_id = data.search_space_id
    selected_connectors = data.selected_connectors
    document_ids = data.document_ids_to_add_in_context
    search_mode = data.search_mode
    top_k = data.top_k

    # 2. Валидация и авторизация
    await validate_search_space_access(session, user.id, search_space_id)
    await validate_connectors(selected_connectors)

    # 3. Загрузка user preferences
    user_pref = await session.execute(
        select(UserSearchSpacePreference)
        .filter(
            UserSearchSpacePreference.user_id == user.id,
            UserSearchSpacePreference.search_space_id == search_space_id
        )
    )
    preference = user_pref.scalar_one_or_none()

    # 4. Определение language для LLM
    language = await get_language_from_llm_config(
        session,
        preference.strategic_llm_id
    )

    # 5. Преобразование messages в LangChain format
    langchain_history = convert_to_langchain_format(messages[:-1])
    user_query = messages[-1].content

    # 6. Инициализация streaming generator
    stream_generator = stream_connector_search_results(
        user_query=user_query,
        user_id=str(user.id),
        search_space_id=search_space_id,
        session=session,
        research_mode="QNA",
        selected_connectors=selected_connectors,
        langchain_chat_history=langchain_history,
        search_mode_str=search_mode,
        document_ids_to_add_in_context=document_ids,
        language=language,
        top_k=top_k,
    )

    # 7. Возврат StreamingResponse
    response = StreamingResponse(
        stream_generator,
        media_type="text/event-stream",
    )
    response.headers["x-vercel-ai-data-stream"] = "v1"

    return response
```

### Фаза 5: LangGraph Workflow Execution

**File**: `surfsense_backend/app/tasks/stream_connector_search_results.py`

```python
async def stream_connector_search_results(...) -> AsyncGenerator[str, None]:
    # 1. Инициализация StreamingService
    streaming_service = StreamingService()

    # 2. Создание начального State
    initial_state = State(
        db_session=session,
        streaming_service=streaming_service,
        chat_history=langchain_chat_history,
        reformulated_query=None,
        further_questions=None,
        reranked_documents=[],
        final_written_report="",
    )

    # 3. Создание Configuration
    config = RunnableConfig(
        configurable=Configuration(
            user_query=user_query,
            connectors_to_search=selected_connectors,
            user_id=user_id,
            search_space_id=search_space_id,
            search_mode=SearchMode[search_mode_str],
            document_ids_to_add_in_context=document_ids_to_add_in_context,
            language=language,
            top_k=top_k,
        )
    )

    # 4. Получение compiled graph
    researcher_graph = get_researcher_graph()

    # 5. Stream execution
    async for chunk in researcher_graph.astream(
        initial_state,
        config=config,
        stream_mode="custom",  # Кастомный режим для yield_value
    ):
        if isinstance(chunk, dict) and "yield_value" in chunk:
            yield chunk["yield_value"]

    # 6. Финальный completion marker
    yield streaming_service.format_completion()
```

**LangGraph Researcher Graph**:

```python
# app/agents/researcher/graph.py

def build_graph():
    workflow = StateGraph(State, config_schema=Configuration)

    # Nodes
    workflow.add_node("reformulate_user_query", reformulate_user_query)
    workflow.add_node("handle_qna_workflow", handle_qna_workflow)
    workflow.add_node("generate_further_questions", generate_further_questions)

    # Linear flow
    workflow.add_edge("__start__", "reformulate_user_query")
    workflow.add_edge("reformulate_user_query", "handle_qna_workflow")
    workflow.add_edge("handle_qna_workflow", "generate_further_questions")
    workflow.add_edge("generate_further_questions", "__end__")

    return workflow.compile()
```

### Фаза 6: Node Execution Details

#### Node 1: reformulate_user_query

**File**: `surfsense_backend/app/agents/researcher/nodes.py`

```python
async def reformulate_user_query(state: State, config: RunnableConfig) -> dict:
    configuration: Configuration = config["configurable"]
    streaming_service = state.streaming_service

    # 1. Уведомление UI о начале
    yield streaming_service.format_terminal_info(
        "Reformulating query...",
        "info"
    )

    # 2. Получение strategic LLM для пользователя
    llm = await get_user_llm_instance(
        state.db_session,
        configuration.user_id,
        configuration.search_space_id,
        role="strategic"  # Лучшая модель для reasoning
    )

    # 3. Вызов QueryService
    reformulated = await QueryService.reformulate_query_with_chat_history(
        original_query=configuration.user_query,
        chat_history=state.chat_history,
        llm=llm,
    )

    # 4. Обновление state
    return {"reformulated_query": reformulated}
```

#### Node 2: handle_qna_workflow

**File**: `surfsense_backend/app/agents/researcher/nodes.py`

```python
async def handle_qna_workflow(state: State, config: RunnableConfig) -> dict:
    configuration: Configuration = config["configurable"]
    streaming_service = state.streaming_service
    session = state.db_session

    # === STEP 1: Fetch user-selected documents ===
    yield streaming_service.format_terminal_info(
        "Loading selected documents...", "info"
    )

    user_selected_docs = await fetch_documents_by_ids(
        session=session,
        document_ids=configuration.document_ids_to_add_in_context,
        user_id=configuration.user_id,
        search_space_id=configuration.search_space_id,
    )

    # === STEP 2: Search connectors ===
    yield streaming_service.format_terminal_info(
        "Searching across connectors...", "info"
    )

    connector_service = ConnectorService(
        session=session,
        user_id=configuration.user_id,
        search_space_id=configuration.search_space_id,
    )

    all_documents = []

    for connector_type in configuration.connectors_to_search:
        # Progress update для каждого connector
        yield streaming_service.format_terminal_info(
            f"Searching {connector_type}...", "info"
        )

        # Поиск в specific connector
        if connector_type == "YOUTUBE_VIDEO":
            docs = await connector_service.search_youtube(
                query=state.reformulated_query,
                top_k=configuration.top_k,
                search_mode=configuration.search_mode,
            )
        elif connector_type == "FILE":
            docs = await connector_service.search_files(
                query=state.reformulated_query,
                top_k=configuration.top_k,
                search_mode=configuration.search_mode,
            )
        # ... другие connectors

        all_documents.extend(docs)

        yield streaming_service.format_terminal_info(
            f"Found {len(docs)} chunks from {connector_type}", "success"
        )

    # Объединение с user-selected docs
    all_documents.extend(user_selected_docs)

    # === STEP 3: Stream sources to UI ===
    yield streaming_service.format_sources_delta(all_documents)

    # === STEP 4: Invoke QNA sub-agent ===
    yield streaming_service.format_terminal_info(
        "Generating answer...", "info"
    )

    qna_result = await invoke_qna_agent(
        state=state,
        config=config,
        documents=all_documents,
    )

    # === STEP 5: Stream answer chunks ===
    for chunk in qna_result.answer_chunks:
        yield streaming_service.format_answer_delta(chunk)

    return {
        "reranked_documents": qna_result.reranked_docs,
        "final_written_report": qna_result.full_answer,
    }
```

#### QNA Sub-Agent

**File**: `surfsense_backend/app/agents/qna_agent/graph.py`

```python
def build_qna_graph():
    workflow = StateGraph(QnAState)

    workflow.add_node("rerank_documents", rerank_documents)
    workflow.add_node("answer_question", answer_question)

    workflow.add_edge("__start__", "rerank_documents")
    workflow.add_edge("rerank_documents", "answer_question")
    workflow.add_edge("answer_question", "__end__")

    return workflow.compile()
```

**Rerank Node**:
```python
async def rerank_documents(state: QnAState, config: RunnableConfig) -> dict:
    # 1. Получение reranker instance
    reranker = get_reranker_service()

    # 2. Reranking по relevance
    reranked = await reranker.rerank_documents(
        query=state.query,
        documents=state.documents,
    )

    # 3. Top documents only
    top_docs = reranked[:10]  # Limit for context window

    return {"reranked_documents": top_docs}
```

**Answer Node**:
```python
async def answer_question(state: QnAState, config: RunnableConfig) -> dict:
    streaming_service = state.streaming_service

    # 1. Получение fast LLM
    llm = await get_user_llm_instance(
        state.db_session,
        config.user_id,
        config.search_space_id,
        role="fast",  # Быстрая модель для Q&A
    )

    # 2. Оптимизация documents для context window
    optimized_docs = optimize_content_for_context_window(
        documents=state.reranked_documents,
        max_tokens=llm.max_context_tokens - 1000,  # Reserve for answer
    )

    # 3. Форматирование documents с citations
    formatted_context = format_documents_with_citations(optimized_docs)

    # 4. Построение prompt
    system_message = SystemMessage(content="""
    You are a helpful research assistant. Answer the question based on the provided documents.
    Always cite sources using [1], [2], etc.
    """)

    human_message = HumanMessage(content=f"""
    Question: {state.query}

    Documents:
    {formatted_context}

    Answer:
    """)

    messages = [system_message, human_message]

    # 5. Streaming invocation
    answer_chunks = []
    async for chunk in llm.astream(messages):
        content = chunk.content
        answer_chunks.append(content)

        # Yield для UI
        yield streaming_service.format_answer_delta(content)

    full_answer = "".join(answer_chunks)

    return {"final_answer": full_answer}
```

#### Node 3: generate_further_questions

```python
async def generate_further_questions(state: State, config: RunnableConfig) -> dict:
    streaming_service = state.streaming_service

    yield streaming_service.format_terminal_info(
        "Generating follow-up questions...", "info"
    )

    # 1. Получение fast LLM
    llm = await get_user_llm_instance(
        state.db_session,
        config.user_id,
        config.search_space_id,
        role="fast",
    )

    # 2. Форматирование context
    context = f"""
    Chat History: {state.chat_history}
    User Query: {config.user_query}
    Answer: {state.final_written_report}
    Documents: {summarize_documents(state.reranked_documents)}
    """

    # 3. Генерация вопросов
    prompt = f"""
    Based on the conversation and answer, generate 3-5 follow-up questions
    that the user might ask next.

    {context}

    Return as JSON array: ["Question 1?", "Question 2?", ...]
    """

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    questions = parse_json_array(response.content)

    # 4. Stream to UI
    yield streaming_service.format_further_questions_delta(questions)

    return {"further_questions": questions}
```

### Фаза 7: Streaming Response (Backend → Frontend)

**StreamingService Format Methods**:

```python
# app/services/streaming_service.py

class StreamingService:
    def format_terminal_info_delta(self, text: str, type: str) -> str:
        """Progress updates"""
        self.terminal_idx += 1

        annotation = {
            "type": "TERMINAL_INFO",
            "data": {
                "idx": self.terminal_idx,
                "message": text,
                "type": type,  # "info", "success", "error"
            }
        }

        # Update internal annotations
        self.message_annotations[0]["content"].append(annotation["data"])

        # Return SSE format
        return f"8:[{json.dumps(annotation)}]\n"

    def format_sources_delta(self, sources: list) -> str:
        """Document sources"""
        annotation = {
            "type": "sources",
            "data": {"nodes": [sources]},  # Nested array for multiple turns
        }

        self.message_annotations[1]["content"] = annotation["data"]

        return f"8:[{json.dumps(annotation)}]\n"

    def format_answer_delta(self, chunk: str) -> str:
        """Streaming answer text"""
        return f'0:"{chunk}"\n'

    def format_further_questions_delta(self, questions: list) -> str:
        """Follow-up questions"""
        annotation = {
            "type": "FURTHER_QUESTIONS",
            "data": questions,
        }

        self.message_annotations[3]["content"] = annotation["data"]

        return f"8:[{json.dumps(annotation)}]\n"

    def format_completion(self) -> str:
        """End of stream marker"""
        return json.dumps({
            "finishReason": "stop",
            "usage": {
                "promptTokens": 1200,  # Placeholder
                "completionTokens": 350,
            }
        }) + "\n"
```

### Фаза 8: Frontend Rendering

**AI SDK обрабатывает stream**:

```typescript
// Внутренняя логика useChat

const processStreamChunk = (chunk: string) => {
  // Парсинг AI SDK protocol
  if (chunk.startsWith('0:')) {
    // Text chunk
    const text = JSON.parse(chunk.slice(2));
    appendToLastMessage(text);
  }
  else if (chunk.startsWith('8:')) {
    // Annotation delta
    const annotations = JSON.parse(chunk.slice(2));
    updateMessageAnnotations(annotations);
  }
  else if (chunk.startsWith('d:')) {
    // Completion
    const completion = JSON.parse(chunk.slice(2));
    setStatus('ready');
    setUsage(completion.usage);
  }
  else if (chunk.startsWith('3:')) {
    // Error
    const error = JSON.parse(chunk.slice(2));
    handleError(error);
  }
};
```

**UI Components React to Updates**:

```typescript
// ChatMessages.tsx
const ChatMessagesUI = ({ messages }) => {
  return (
    <div>
      {messages.map((message) => (
        <ChatMessageUI key={message.id} message={message}>
          {/* Terminal Info */}
          <ChatTerminalDisplay events={message.annotations?.TERMINAL_INFO} />

          {/* Main Content */}
          <ChatMessage.Content>{message.content}</ChatMessage.Content>

          {/* Sources */}
          <ChatSourcesDisplay sources={message.annotations?.sources} />

          {/* Further Questions */}
          <ChatFurtherQuestions questions={message.annotations?.FURTHER_QUESTIONS} />
        </ChatMessageUI>
      ))}
    </div>
  );
};
```

### Фаза 9: Auto-save to Database

```typescript
// Автоматическое сохранение после получения ответа
useEffect(() => {
  const lastMessage = handler.messages[handler.messages.length - 1];

  if (
    !isNewChat &&
    handler.status === "ready" &&
    lastMessage?.role === "assistant"
  ) {
    // Сохранить обновленный chat в DB
    updateChatInDatabase(chatIdParam, handler.messages);
  }
}, [handler.messages, handler.status]);

const updateChatInDatabase = async (chatId: string, messages: Message[]) => {
  await fetch(`${BACKEND_URL}/api/v1/chats/${chatId}`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      messages: messages,
      state_version: messages.length,
    }),
  });
};
```

## Временная диаграмма

```
Time →
0ms    User clicks Send
10ms   useChat.append() called
20ms   POST /api/v1/chat sent
100ms  Backend receives request
150ms  LangGraph starts execution
200ms  NODE 1: Reformulating query...
       ├─ SSE: 8:[{"type":"TERMINAL_INFO","data":{"message":"Reformulating query..."}}]
       └─ Frontend updates terminal display
500ms  Query reformulated
       └─ NODE 2: handle_qna_workflow starts
600ms  ├─ SSE: 8:[{"type":"TERMINAL_INFO","data":{"message":"Searching YouTube..."}}]
800ms  ├─ SSE: 8:[{"type":"TERMINAL_INFO","data":{"message":"Found 5 chunks..."}}]
1000ms ├─ SSE: 8:[{"type":"sources","data":{"nodes":[...]}}]
       └─ Frontend renders sources panel
1200ms └─ QNA Agent starts
       ├─ Reranking documents...
1500ms └─ Generating answer...
       ├─ SSE: 0:"Based"
       ├─ SSE: 0:" on"
       ├─ SSE: 0:" the"
       └─ Frontend incrementally renders text
3000ms Answer complete
       └─ NODE 3: generate_further_questions
3500ms ├─ SSE: 8:[{"type":"FURTHER_QUESTIONS","data":["What about...?"]}]
       └─ Frontend renders question chips
4000ms ├─ SSE: d:{"finishReason":"stop"}
       ├─ Frontend sets status="ready"
       └─ Auto-save to database triggered
```

## Обработка ошибок

### Frontend Error Handling

```typescript
const handler = useChat({
  onError: (error) => {
    console.error("Chat error:", error);
    toast.error(`Failed to get response: ${error.message}`);
  },
});

// Retry mechanism
const retryLastMessage = () => {
  handler.reload();  // AI SDK built-in retry
};
```

### Backend Error Handling

```python
try:
    async for chunk in researcher_graph.astream(...):
        yield chunk["yield_value"]
except Exception as e:
    logger.error(f"Chat error: {e}")

    # Send error to frontend
    error_msg = streaming_service.format_error(str(e))
    yield error_msg
```

## Оптимизации производительности

### 1. Parallel Document Fetching
```python
# Fetch from multiple connectors in parallel
tasks = [
    connector_service.search_youtube(...),
    connector_service.search_files(...),
    connector_service.search_slack(...),
]

results = await asyncio.gather(*tasks)
```

### 2. Context Window Optimization
```python
# Truncate documents to fit context window
optimized = optimize_content_for_context_window(
    documents=all_docs,
    max_tokens=llm.max_context_tokens - 1000,
)
```

### 3. Streaming First Token Latency
- Reformulation runs in parallel with UI update
- Progress updates sent immediately
- Answer streaming starts before full retrieval completes

### 4. Client-side Optimizations
- Memoization для message components
- Virtual scrolling для длинных conversations
- Debounced auto-save

## Итоговая архитектура потока

```
User Input
    ↓
useChat Hook
    ↓
POST /api/v1/chat (SSE connection opened)
    ↓
FastAPI Route Handler
    ↓
stream_connector_search_results generator
    ↓
LangGraph Researcher Agent
    ├─ reformulate_user_query → SSE: TERMINAL_INFO
    ├─ handle_qna_workflow
    │   ├─ Search connectors → SSE: TERMINAL_INFO (per connector)
    │   ├─ Aggregate results → SSE: sources
    │   └─ QNA Agent
    │       ├─ rerank_documents
    │       └─ answer_question → SSE: 0:"text chunks"
    └─ generate_further_questions → SSE: FURTHER_QUESTIONS
    ↓
SSE: d:{completion}
    ↓
useChat processes stream
    ↓
UI re-renders incrementally
    ↓
Auto-save to database
```

Этот поток обеспечивает отличный UX через:
- **Immediate feedback**: Terminal updates на каждом шаге
- **Progressive rendering**: Token-by-token streaming
- **Rich metadata**: Sources, questions вместе с ответом
- **Persistent state**: Auto-save в database
