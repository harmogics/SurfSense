# Архитектура потоковой передачи данных (Streaming Architecture)

## Введение

Streaming является ключевой особенностью SurfSense, обеспечивающей мгновенную обратную связь пользователю во время обработки сложных RAG-запросов. Этот документ подробно описывает реализацию streaming от протокола передачи до интеграции с UI.

## Протокол: Vercel AI SDK Data Stream Protocol

### Обзор протокола

SurfSense использует **Vercel AI SDK Data Stream Protocol** - стандартизированный формат для передачи AI-генерируемого контента через Server-Sent Events (SSE).

**Спецификация**: https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol

### Message Types

| Prefix | Type | Description | Example |
|--------|------|-------------|---------|
| `0:` | Text | Incrementальный text chunk | `0:"Hello"` |
| `8:` | Annotation | Metadata (sources, events) | `8:[{"type":"sources"...}]` |
| `3:` | Error | Error message | `3:{"message":"Failed"}` |
| `d:` | Data | Completion marker + usage | `d:{"finishReason":"stop"}` |

### Stream Example

```
0:"Based"
0:" on"
0:" the"
0:" documents,"
8:[{"type":"TERMINAL_INFO","data":{"idx":1,"message":"Searching...","type":"info"}}]
8:[{"type":"sources","data":{"nodes":[[{"id":123,"score":0.95}]]}}]
0:" the"
0:" answer"
0:" is"
8:[{"type":"FURTHER_QUESTIONS","data":["What about X?"]}]
d:{"finishReason":"stop","usage":{"promptTokens":1200,"completionTokens":350}}
```

## Backend Streaming Implementation

### StreamingService Class

**File**: `surfsense_backend/app/services/streaming_service.py`

```python
class StreamingService:
    """
    Сервис для форматирования данных в Vercel AI SDK protocol.
    Используется LangGraph nodes для streaming updates в UI.
    """

    def __init__(self):
        # Terminal events counter
        self.terminal_idx = 0

        # Accumulated annotations (для финального message object)
        self.message_annotations = [
            {"type": "TERMINAL_INFO", "content": []},    # Progress events
            {"type": "SOURCES", "content": []},          # Document references
            {"type": "ANSWER", "content": []},           # Generated answer
            {"type": "FURTHER_QUESTIONS", "content": []},# Follow-up questions
        ]

    # ============================================
    # Text Streaming Methods
    # ============================================

    def format_text_chunk(self, text: str) -> str:
        """
        Format text chunk for streaming.

        Args:
            text: Text content to stream

        Returns:
            SSE-formatted string: '0:"text"\n'
        """
        # Escape quotes and special characters
        escaped = text.replace('"', '\\"').replace('\n', '\\n')
        return f'0:"{escaped}"\n'

    def format_answer_delta(self, answer_chunk: str) -> str:
        """
        Format answer chunk (alias for format_text_chunk).
        Used for semantic clarity in LLM response streaming.
        """
        self.message_annotations[2]["content"].append(answer_chunk)
        return self.format_text_chunk(answer_chunk)

    # ============================================
    # Annotation Methods
    # ============================================

    def format_terminal_info_delta(
        self,
        text: str,
        message_type: str = "info"
    ) -> str:
        """
        Format terminal progress update.

        Args:
            text: Progress message
            message_type: "info" | "success" | "error" | "warning"

        Returns:
            SSE annotation: '8:[{"type":"TERMINAL_INFO","data":{...}}]\n'
        """
        self.terminal_idx += 1

        terminal_event = {
            "idx": self.terminal_idx,
            "message": text,
            "type": message_type,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Append to accumulated annotations
        self.message_annotations[0]["content"].append(terminal_event)

        # Format for SSE
        annotation = {
            "type": "TERMINAL_INFO",
            "data": terminal_event,
        }

        return f"8:[{json.dumps(annotation)}]\n"

    def format_sources_delta(self, sources: list[dict]) -> str:
        """
        Format document sources.

        Args:
            sources: List of source documents with metadata

        Returns:
            SSE annotation with sources
        """
        # Transform to frontend-expected format
        formatted_sources = []

        for source in sources:
            formatted_sources.append({
                "id": source.get("id"),
                "score": source.get("score", 0.0),
                "text": source.get("text", ""),
                "metadata": {
                    "title": source.get("title"),
                    "url": source.get("url"),
                    "source_type": source.get("source_type"),
                    "document_id": source.get("document_id"),
                },
            })

        # Store in annotations
        self.message_annotations[1]["content"] = {"nodes": [formatted_sources]}

        # Format for SSE
        annotation = {
            "type": "sources",
            "data": {"nodes": [formatted_sources]},
        }

        return f"8:[{json.dumps(annotation)}]\n"

    def format_further_questions_delta(self, questions: list[str]) -> str:
        """
        Format follow-up questions.

        Args:
            questions: List of suggested questions

        Returns:
            SSE annotation with questions
        """
        # Store in annotations
        self.message_annotations[3]["content"] = questions

        # Format for SSE
        annotation = {
            "type": "FURTHER_QUESTIONS",
            "data": questions,
        }

        return f"8:[{json.dumps(annotation)}]\n"

    # ============================================
    # Control Messages
    # ============================================

    def format_error(self, error_message: str, error_code: str = None) -> str:
        """
        Format error message.

        Args:
            error_message: Human-readable error
            error_code: Optional error code

        Returns:
            SSE error: '3:{"message":"...","code":"..."}\n'
        """
        error_data = {
            "message": error_message,
            "code": error_code,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return f"3:{json.dumps(error_data)}\n"

    def format_completion(
        self,
        finish_reason: str = "stop",
        usage: dict = None
    ) -> str:
        """
        Format completion marker.

        Args:
            finish_reason: "stop" | "length" | "error"
            usage: Token usage statistics

        Returns:
            SSE completion: 'd:{"finishReason":"stop","usage":{...}}\n'
        """
        completion_data = {
            "finishReason": finish_reason,
            "usage": usage or {
                "promptTokens": 0,
                "completionTokens": 0,
            },
        }

        return f"d:{json.dumps(completion_data)}\n"

    # ============================================
    # Utility Methods
    # ============================================

    def get_accumulated_annotations(self) -> list[dict]:
        """
        Get all accumulated annotations (for database storage).

        Returns:
            List of annotation objects
        """
        return self.message_annotations

    def reset(self):
        """Reset service state for new message."""
        self.terminal_idx = 0
        self.message_annotations = [
            {"type": "TERMINAL_INFO", "content": []},
            {"type": "SOURCES", "content": []},
            {"type": "ANSWER", "content": []},
            {"type": "FURTHER_QUESTIONS", "content": []},
        ]
```

### Usage in LangGraph Nodes

```python
# app/agents/researcher/nodes.py

async def reformulate_user_query(state: State, config: RunnableConfig):
    streaming_service = state.streaming_service

    # Send progress update
    yield streaming_service.format_terminal_info_delta(
        "Reformulating query for better retrieval...",
        "info"
    )

    # ... processing logic ...

    # Send success message
    yield streaming_service.format_terminal_info_delta(
        f"Query reformulated: '{reformulated}'",
        "success"
    )

    return {"reformulated_query": reformulated}
```

```python
async def handle_qna_workflow(state: State, config: RunnableConfig):
    streaming_service = state.streaming_service

    # Search connectors
    for connector_type in config.connectors_to_search:
        yield streaming_service.format_terminal_info_delta(
            f"Searching {connector_type}...",
            "info"
        )

        docs = await search_connector(connector_type, query)

        yield streaming_service.format_terminal_info_delta(
            f"Found {len(docs)} results from {connector_type}",
            "success"
        )

    # Send sources to UI
    yield streaming_service.format_sources_delta(all_documents)

    # Stream LLM answer
    async for chunk in llm.astream(messages):
        yield streaming_service.format_answer_delta(chunk.content)

    # Send follow-up questions
    yield streaming_service.format_further_questions_delta(questions)
```

### FastAPI StreamingResponse

**File**: `surfsense_backend/app/routes/chats_routes.py`

```python
@router.post("/chat")
async def handle_chat_data(...) -> StreamingResponse:
    # Create generator
    stream_generator = stream_connector_search_results(
        user_query=user_query,
        # ... other params
    )

    # Return streaming response
    response = StreamingResponse(
        stream_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )

    # Mark as AI SDK stream
    response.headers["x-vercel-ai-data-stream"] = "v1"

    return response
```

### Async Generator Pattern

```python
# app/tasks/stream_connector_search_results.py

async def stream_connector_search_results(
    user_query: str,
    # ... params
) -> AsyncGenerator[str, None]:
    """
    Main streaming orchestrator.
    Yields SSE-formatted strings.
    """

    # Initialize streaming service
    streaming_service = StreamingService()

    # Create state
    initial_state = State(
        streaming_service=streaming_service,
        # ... other state
    )

    # Get compiled graph
    graph = get_researcher_graph()

    # Stream execution
    try:
        async for chunk in graph.astream(
            initial_state,
            config=config,
            stream_mode="custom",  # Enables yield from nodes
        ):
            # Each chunk is {"yield_value": "SSE string"}
            if isinstance(chunk, dict) and "yield_value" in chunk:
                yield chunk["yield_value"]

        # Send completion
        yield streaming_service.format_completion(
            finish_reason="stop",
            usage={
                "promptTokens": 1200,
                "completionTokens": 350,
            }
        )

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield streaming_service.format_error(str(e))
        yield streaming_service.format_completion(finish_reason="error")
```

## Frontend Streaming Implementation

### AI SDK useChat Hook

**File**: `surfsense_web/app/dashboard/[search_space_id]/researcher/[[...chat_id]]/page.tsx`

```typescript
import { useChat } from "@ai-sdk/react";

const ResearcherPage = () => {
  const handler = useChat({
    // Backend endpoint
    api: `${process.env.NEXT_PUBLIC_FASTAPI_BACKEND_URL}/api/v1/chat`,

    // Stream protocol (SSE)
    streamProtocol: "data",

    // Authentication
    headers: {
      Authorization: `Bearer ${token}`,
    },

    // Initial messages (for existing chat)
    initialMessages: existingMessages,

    // Configuration data (sent in request body)
    body: {
      data: {
        search_space_id: searchSpaceId,
        selected_connectors: connectors,
        search_mode: searchMode,
        document_ids_to_add_in_context: documentIds,
        top_k: topK,
      },
    },

    // Error handling
    onError: (error) => {
      console.error("Chat error:", error);
      toast.error(`Failed: ${error.message}`);
    },

    // Callbacks
    onFinish: (message) => {
      console.log("Message completed:", message);
      // Trigger auto-save
      saveChat(chatId, handler.messages);
    },
  });

  // Handler provides:
  // - messages: Message[]
  // - append(message): send new message
  // - reload(): retry last message
  // - stop(): abort streaming
  // - isLoading: boolean
  // - status: 'ready' | 'streaming'

  return (
    <ChatInterface
      messages={handler.messages}
      onSend={(text) => handler.append({ role: "user", content: text })}
      isLoading={handler.isLoading}
    />
  );
};
```

### Internal Stream Processing

AI SDK internally handles SSE stream parsing:

```typescript
// Simplified internal logic

async function processStream(response: Response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Process complete lines
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) continue;

      // Parse protocol message
      if (line.startsWith("0:")) {
        // Text chunk
        const text = JSON.parse(line.slice(2));
        appendTextToLastMessage(text);
      }
      else if (line.startsWith("8:")) {
        // Annotation
        const annotations = JSON.parse(line.slice(2));
        updateMessageAnnotations(annotations);
      }
      else if (line.startsWith("d:")) {
        // Completion
        const completion = JSON.parse(line.slice(2));
        setStatus("ready");
        setUsage(completion.usage);
      }
      else if (line.startsWith("3:")) {
        // Error
        const error = JSON.parse(line.slice(2));
        throwError(error);
      }
    }
  }
}
```

### Message Type Definition

```typescript
// AI SDK Message type
interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  annotations?: {
    TERMINAL_INFO?: TerminalEvent[];
    sources?: SourceNode[][];
    FURTHER_QUESTIONS?: string[];
  };
  createdAt?: Date;
}

interface TerminalEvent {
  idx: number;
  message: string;
  type: "info" | "success" | "error" | "warning";
  timestamp: string;
}

interface SourceNode {
  id: number;
  score: number;
  text: string;
  metadata: {
    title: string;
    url?: string;
    source_type: string;
    document_id: number;
  };
}
```

## UI Component Updates

### Terminal Display Component

**File**: `surfsense_web/components/chat/ChatTerminal.tsx`

```typescript
import { getAnnotationData } from "@llamaindex/chat-ui";

const ChatTerminalDisplay = ({ message }: { message: Message }) => {
  // Extract terminal events from annotations
  const events = getAnnotationData(message, "TERMINAL_INFO") as TerminalEvent[];

  if (!events || events.length === 0) return null;

  return (
    <div className="terminal-container">
      {events.map((event) => (
        <div key={event.idx} className={`terminal-event ${event.type}`}>
          <span className="terminal-icon">
            {event.type === "info" && "ℹ️"}
            {event.type === "success" && "✅"}
            {event.type === "error" && "❌"}
            {event.type === "warning" && "⚠️"}
          </span>
          <span className="terminal-message">{event.message}</span>
          <span className="terminal-timestamp">
            {new Date(event.timestamp).toLocaleTimeString()}
          </span>
        </div>
      ))}
    </div>
  );
};
```

### Sources Display Component

```typescript
const ChatSourcesDisplay = ({ message }: { message: Message }) => {
  const sources = getAnnotationData(message, "sources") as SourceNode[][];

  if (!sources || sources.length === 0) return null;

  // Flatten nested array (for multi-turn support)
  const allSources = sources.flat();

  return (
    <div className="sources-container">
      <h4>Sources ({allSources.length})</h4>
      <div className="sources-grid">
        {allSources.map((source, idx) => (
          <SourceCard key={source.id} source={source} citation={idx + 1} />
        ))}
      </div>
    </div>
  );
};

const SourceCard = ({ source, citation }) => {
  return (
    <div className="source-card">
      <span className="citation-badge">[{citation}]</span>
      <div className="source-header">
        <h5>{source.metadata.title}</h5>
        <span className="source-score">{(source.score * 100).toFixed(0)}%</span>
      </div>
      <p className="source-snippet">{source.text.slice(0, 150)}...</p>
      {source.metadata.url && (
        <a href={source.metadata.url} target="_blank" rel="noopener">
          View Source →
        </a>
      )}
    </div>
  );
};
```

### Further Questions Component

```typescript
const ChatFurtherQuestions = ({ message }: { message: Message }) => {
  const questions = getAnnotationData(message, "FURTHER_QUESTIONS") as string[];

  if (!questions || questions.length === 0) return null;

  return (
    <div className="further-questions">
      <h4>Follow-up Questions</h4>
      <div className="questions-list">
        {questions.map((question, idx) => (
          <button
            key={idx}
            className="question-chip"
            onClick={() => handler.append({ role: "user", content: question })}
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
};
```

### Streaming Text Component

```typescript
const ChatMessageContent = ({ message }: { message: Message }) => {
  // Markdown rendering with streaming support
  return (
    <div className="message-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Custom renderers
          code: CodeBlock,
          a: LinkComponent,
        }}
      >
        {message.content}
      </ReactMarkdown>

      {/* Blinking cursor during streaming */}
      {message.role === "assistant" && handler.isLoading && (
        <span className="cursor-blink">▋</span>
      )}
    </div>
  );
};
```

## Performance Optimizations

### 1. Debounced Rendering

```typescript
// Avoid re-rendering on every token
const DebouncedMarkdown = ({ content }: { content: string }) => {
  const [debouncedContent, setDebouncedContent] = useState(content);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedContent(content);
    }, 50); // Update every 50ms instead of every token

    return () => clearTimeout(timer);
  }, [content]);

  return <ReactMarkdown>{debouncedContent}</ReactMarkdown>;
};
```

### 2. Virtual Scrolling

```typescript
import { useVirtualizer } from "@tanstack/react-virtual";

const MessageList = ({ messages }: { messages: Message[] }) => {
  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: messages.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 200, // Estimated message height
  });

  return (
    <div ref={parentRef} style={{ height: "600px", overflow: "auto" }}>
      <div style={{ height: `${virtualizer.getTotalSize()}px` }}>
        {virtualizer.getVirtualItems().map((virtualRow) => {
          const message = messages[virtualRow.index];
          return (
            <div
              key={virtualRow.index}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                transform: `translateY(${virtualRow.start}px)`,
              }}
            >
              <ChatMessageUI message={message} />
            </div>
          );
        })}
      </div>
    </div>
  );
};
```

### 3. Memoization

```typescript
const ChatMessageUI = React.memo(
  ({ message }: { message: Message }) => {
    return (
      <div className="message">
        <ChatTerminalDisplay message={message} />
        <ChatMessageContent message={message} />
        <ChatSourcesDisplay message={message} />
        <ChatFurtherQuestions message={message} />
      </div>
    );
  },
  (prevProps, nextProps) => {
    // Only re-render if message content changed
    return (
      prevProps.message.id === nextProps.message.id &&
      prevProps.message.content === nextProps.message.content &&
      JSON.stringify(prevProps.message.annotations) ===
        JSON.stringify(nextProps.message.annotations)
    );
  }
);
```

## Error Handling & Recovery

### Backend Error Streaming

```python
async def stream_with_error_handling(...):
    try:
        async for chunk in graph.astream(...):
            yield chunk["yield_value"]

        yield streaming_service.format_completion()

    except AuthenticationError as e:
        yield streaming_service.format_error(
            "Authentication failed. Please log in again.",
            "AUTH_ERROR"
        )
        yield streaming_service.format_completion(finish_reason="error")

    except RateLimitError as e:
        yield streaming_service.format_error(
            "Rate limit exceeded. Please try again later.",
            "RATE_LIMIT"
        )
        yield streaming_service.format_completion(finish_reason="error")

    except Exception as e:
        logger.exception("Unexpected error in streaming")
        yield streaming_service.format_error(
            "An unexpected error occurred. Please try again.",
            "INTERNAL_ERROR"
        )
        yield streaming_service.format_completion(finish_reason="error")
```

### Frontend Error Handling

```typescript
const handler = useChat({
  onError: (error) => {
    // Log error
    console.error("Streaming error:", error);

    // Show user-friendly message
    if (error.message.includes("AUTH_ERROR")) {
      toast.error("Please log in again");
      router.push("/login");
    } else if (error.message.includes("RATE_LIMIT")) {
      toast.error("Too many requests. Please wait.");
    } else {
      toast.error("Something went wrong. Please try again.");
    }

    // Track error
    analytics.track("chat_error", {
      error_code: error.code,
      message: error.message,
    });
  },
});

// Retry mechanism
const retryLastMessage = () => {
  toast.info("Retrying...");
  handler.reload();
};
```

### Connection Recovery

```typescript
// Auto-reconnect on connection loss
useEffect(() => {
  const handleOnline = () => {
    if (handler.status === "streaming") {
      toast.info("Connection restored. Retrying...");
      handler.reload();
    }
  };

  window.addEventListener("online", handleOnline);
  return () => window.removeEventListener("online", handleOnline);
}, [handler]);
```

## Monitoring & Debugging

### Backend Logging

```python
import logging

logger = logging.getLogger(__name__)

async def stream_connector_search_results(...):
    logger.info(f"Starting stream for user {user_id}, query: {user_query[:50]}")

    start_time = time.time()

    try:
        chunk_count = 0
        async for chunk in graph.astream(...):
            chunk_count += 1
            yield chunk["yield_value"]

        duration = time.time() - start_time
        logger.info(
            f"Stream completed for user {user_id}. "
            f"Duration: {duration:.2f}s, Chunks: {chunk_count}"
        )

    except Exception as e:
        logger.error(
            f"Stream failed for user {user_id}: {e}",
            exc_info=True
        )
        raise
```

### Frontend Analytics

```typescript
useEffect(() => {
  if (handler.status === "streaming") {
    analytics.track("chat_stream_started", {
      chat_id: chatId,
      search_space_id: searchSpaceId,
      connectors: selectedConnectors,
    });
  }

  if (handler.status === "ready" && handler.messages.length > 0) {
    const lastMessage = handler.messages[handler.messages.length - 1];
    if (lastMessage.role === "assistant") {
      analytics.track("chat_stream_completed", {
        chat_id: chatId,
        message_length: lastMessage.content.length,
        sources_count: lastMessage.annotations?.sources?.flat().length || 0,
        duration: Date.now() - streamStartTime,
      });
    }
  }
}, [handler.status]);
```

## Architecture Benefits

### 1. Real-time Feedback
- Пользователь видит прогресс на каждом этапе
- Снижает perceived latency
- Улучшает UX для long-running операций

### 2. Progressive Enhancement
- Частичные результаты полезны даже без completion
- Graceful degradation при ошибках
- Пользователь может остановить stream при необходимости

### 3. Efficient Resource Usage
- Backend не буферизует весь ответ
- Frontend рендерит incrementally
- Меньше memory footprint на обеих сторонах

### 4. Structured Data Flow
- Стандартизированный protocol (AI SDK)
- Type-safe на frontend (TypeScript)
- Легко расширяемый (новые annotation types)

### 5. Observability
- Detailed progress updates видны пользователю
- Logging на каждом этапе
- Easy debugging через terminal events

## Итоговая диаграмма

```
┌────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                       │
│                                                            │
│  LangGraph Node                                            │
│       │                                                     │
│       │ yield streaming_service.format_terminal_info(...)  │
│       │ yield streaming_service.format_sources_delta(...)  │
│       │ yield streaming_service.format_answer_delta(...)   │
│       ▼                                                     │
│  AsyncGenerator[str, None]                                 │
│       │                                                     │
│       │ '8:[{"type":"TERMINAL_INFO"...}]\n'                │
│       │ '8:[{"type":"sources"...}]\n'                      │
│       │ '0:"text chunk"\n'                                 │
│       ▼                                                     │
│  StreamingResponse                                         │
│       │                                                     │
└───────┼────────────────────────────────────────────────────┘
        │
        │ SSE over HTTP
        │
┌───────▼────────────────────────────────────────────────────┐
│                 Frontend (Next.js + AI SDK)                │
│                                                            │
│  useChat Hook                                              │
│       │                                                     │
│       │ Parse SSE stream                                   │
│       │ Update messages state                              │
│       ▼                                                     │
│  messages: Message[]                                       │
│       │                                                     │
│       │ message.content (incremental text)                 │
│       │ message.annotations.TERMINAL_INFO (events)         │
│       │ message.annotations.sources (documents)            │
│       │ message.annotations.FURTHER_QUESTIONS (questions)  │
│       ▼                                                     │
│  UI Components                                             │
│       ├─► ChatTerminalDisplay (events)                     │
│       ├─► ChatMessageContent (markdown + cursor)           │
│       ├─► ChatSourcesDisplay (source cards)                │
│       └─► ChatFurtherQuestions (question chips)            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

Эта архитектура обеспечивает seamless streaming experience с rich metadata и excellent error handling.
