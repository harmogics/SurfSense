# Циклы обратной связи (Feedback Loops)

## Введение

Циклы обратной связи в SurfSense обеспечивают двунаправленную коммуникацию между пользователем и системой. Этот документ описывает, как действия пользователя влияют на поведение системы, и как система адаптируется к изменениям конфигурации и данных.

## Типы Feedback Loops

```
┌──────────────────────────────────────────────────────────────┐
│                   ТИПЫ ОБРАТНОЙ СВЯЗИ                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Immediate Feedback (мгновенная)                          │
│     User action → UI update (optimistic)                     │
│     Latency: < 50ms                                          │
│                                                              │
│  2. Progressive Feedback (прогрессивная)                     │
│     User action → Processing → Streaming updates             │
│     Latency: 100ms - 30s                                     │
│                                                              │
│  3. Configuration Feedback (конфигурационная)                │
│     User changes settings → Next request uses new config     │
│     Latency: Immediate (next action)                         │
│                                                              │
│  4. Data Feedback (данные)                                   │
│     User adds/modifies data → Search results updated         │
│     Latency: Seconds to minutes (processing)                 │
│                                                              │
│  5. Learning Feedback (обучающая)                            │
│     User selections → Implicit preferences → Future results  │
│     Latency: Hours to days (aggregation)                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## 1. Immediate Feedback Loops

### Query Input Feedback

**Scenario**: Пользователь вводит текст в chat input

```typescript
const ChatInputUI = () => {
  const [input, setInput] = useState("");
  const [isValid, setIsValid] = useState(true);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;

    // Immediate UI update
    setInput(value);

    // Validation feedback
    if (value.length > 5000) {
      setIsValid(false);
      toast.error("Query too long (max 5000 characters)");
    } else {
      setIsValid(true);
    }

    // Character count feedback
    setCharCount(value.length);
  };

  return (
    <div>
      <textarea
        value={input}
        onChange={handleInputChange}
        className={isValid ? "" : "border-red-500"}
      />
      <div className="text-sm text-gray-500">
        {charCount} / 5000 characters
      </div>
    </div>
  );
};
```

**Timeline**:
```
User types → 0ms → Input value updates
          → 10ms → Validation runs
          → 20ms → UI feedback (red border, error message)
```

### Document Selection Feedback

**Scenario**: Пользователь выбирает документы для контекста

```typescript
const DocumentSelector = ({ onSelectionChange, selectedDocuments }) => {
  const [localSelection, setLocalSelection] = useState(selectedDocuments);

  const handleToggle = (document: Document) => {
    // Optimistic update
    setLocalSelection((prev) => {
      if (prev.find((d) => d.id === document.id)) {
        // Remove
        return prev.filter((d) => d.id !== document.id);
      } else {
        // Add
        return [...prev, document];
      }
    });

    // Propagate to parent
    onSelectionChange(localSelection);

    // Visual feedback
    toast.success(
      localSelection.find((d) => d.id === document.id)
        ? "Document removed from context"
        : "Document added to context"
    );
  };

  return (
    <div>
      {documents.map((doc) => (
        <Checkbox
          key={doc.id}
          checked={localSelection.find((d) => d.id === doc.id)}
          onChange={() => handleToggle(doc)}
        />
      ))}

      <div className="text-sm">
        {localSelection.length} documents selected
      </div>
    </div>
  );
};
```

**Timeline**:
```
User clicks checkbox → 0ms → Checkbox toggles
                    → 10ms → Selection count updates
                    → 50ms → Toast notification appears
```

## 2. Progressive Feedback Loops

### Streaming Answer Feedback

**Scenario**: Пользователь отправляет вопрос, получает streaming ответ

```
User sends query
    ↓
    0ms: UI shows "Sending..."
    ↓
    100ms: Backend receives request
    ↓
    200ms: SSE: "Reformulating query..."
    ↓
    500ms: SSE: "Query reformulated"
    ↓
    600ms: SSE: "Searching YouTube..."
    ↓
    1000ms: SSE: "Found 5 results from YouTube"
    ↓
    1200ms: SSE: "Searching Files..."
    ↓
    1500ms: SSE: "Found 3 results from Files"
    ↓
    1600ms: SSE: Sources panel updates
    ↓
    2000ms: SSE: "Generating answer..."
    ↓
    2100ms: SSE: "Based" (first token)
    ↓
    2110ms: SSE: " on" (second token)
    ↓
    ... tokens stream ...
    ↓
    5000ms: SSE: Answer complete
    ↓
    5500ms: SSE: "Generating follow-up questions..."
    ↓
    6000ms: SSE: Questions appear as chips
    ↓
    6100ms: SSE: Completion marker
    ↓
    6200ms: Auto-save to database
```

**Implementation**:

```typescript
// Terminal updates
useEffect(() => {
  const events = message.annotations?.TERMINAL_INFO || [];
  const latestEvent = events[events.length - 1];

  if (latestEvent) {
    // Show toast для important events
    if (latestEvent.type === "success") {
      toast.success(latestEvent.message, { duration: 2000 });
    } else if (latestEvent.type === "error") {
      toast.error(latestEvent.message);
    }
  }
}, [message.annotations?.TERMINAL_INFO]);

// Progressive text rendering
useEffect(() => {
  if (message.role === "assistant") {
    // Auto-scroll to bottom as text streams
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }
}, [message.content]);
```

### Document Processing Feedback

**Scenario**: Пользователь загружает документ, видит прогресс обработки

```
User uploads file.pdf
    ↓
    0ms: Upload starts, progress bar appears
    ↓
    2000ms: Upload complete (100%)
    ↓
    2100ms: "Document uploaded, processing..."
    ↓
    2200ms: Poll status every 2 seconds
    ↓
    4200ms: Status: "processing" (50% estimated)
    ↓
    6200ms: Status: "processing" (75% estimated)
    ↓
    8200ms: Status: "completed"
    ↓
    8300ms: "Document ready for search!"
    ↓
    8400ms: Document list refreshes
```

**Implementation**:

```typescript
const DocumentUpload = () => {
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingStatus, setProcessingStatus] = useState<string | null>(null);

  const handleUpload = async (file: File) => {
    // Upload with progress
    await uploadWithProgress(file, (progress) => {
      setUploadProgress(progress);
    });

    // Start polling for processing status
    setProcessingStatus("processing");

    const documentId = response.data.id;

    const interval = setInterval(async () => {
      const doc = await fetchDocument(documentId);

      if (doc.status === "completed") {
        clearInterval(interval);
        setProcessingStatus("completed");
        toast.success("Document processed successfully!");
        refetchDocuments();
      } else if (doc.status === "failed") {
        clearInterval(interval);
        setProcessingStatus("failed");
        toast.error(`Processing failed: ${doc.error_message}`);
      }
    }, 2000);
  };

  return (
    <div>
      {uploadProgress > 0 && uploadProgress < 100 && (
        <ProgressBar value={uploadProgress} />
      )}

      {processingStatus === "processing" && (
        <div className="flex items-center">
          <Spinner />
          <span>Processing document...</span>
        </div>
      )}

      {processingStatus === "completed" && (
        <div className="text-green-600">
          <CheckIcon />
          Ready for search
        </div>
      )}
    </div>
  );
};
```

## 3. Configuration Feedback Loops

### LLM Configuration Changes

**Scenario**: Пользователь меняет LLM модель, следующий запрос использует новую модель

```
User opens LLM settings
    ↓
User changes "Fast LLM" from gpt-3.5-turbo to gpt-4
    ↓
User clicks "Save"
    ↓
    100ms: Frontend sends PUT /api/v1/llm_configs
    ↓
    200ms: Backend validates config
    ↓
    300ms: Backend updates UserSearchSpacePreference
    ↓
    350ms: Backend clears LLMService cache for user
    ↓
    400ms: Backend returns success
    ↓
    450ms: Frontend invalidates TanStack Query cache
    ↓
    500ms: UI shows "Settings saved ✓"
    ↓
User sends next chat query
    ↓
    Backend loads new LLM config (gpt-4)
    ↓
    Query answered with gpt-4
```

**Implementation**:

```typescript
// Frontend
const { mutate: updateLLMConfig, isLoading } = useMutation({
  mutationFn: async (data: UpdateLLMConfigRequest) => {
    return llmConfigApiService.updateConfig(data);
  },
  onSuccess: () => {
    // Invalidate cache
    queryClient.invalidateQueries({
      queryKey: cacheKeys.activeSearchSpace.llmConfigs(searchSpaceId),
    });

    // User feedback
    toast.success("LLM settings saved successfully");

    // Close modal
    setIsOpen(false);
  },
  onError: (error) => {
    toast.error(`Failed to save: ${error.message}`);
  },
});
```

```python
# Backend
@router.put("/llm_configs/{config_id}")
async def update_llm_config(
    config_id: int,
    data: UpdateLLMConfigRequest,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    # Update config
    config = await update_config_in_db(session, config_id, data)

    # Clear cache so next request uses new config
    LLMService.clear_cache(user_id=str(user.id))

    logger.info(f"LLM config updated: {config_id} by user {user.id}")

    return config
```

### Connector Selection Feedback

**Scenario**: Пользователь включает/выключает connectors, результаты поиска меняются

```
User в настройках чата
    ↓
User toggles YouTube connector (ON → OFF)
    ↓
    Immediate: Checkbox updates
    ↓
    10ms: selectedConnectors state updates
    ↓
    50ms: State persisted to localStorage
    ↓
User sends next query
    ↓
    Query body includes: selected_connectors: ["FILE", "SLACK"] (no YouTube)
    ↓
    Backend searches only FILE and SLACK
    ↓
    Results exclude YouTube videos
```

**Implementation**:

```typescript
const ConnectorSelector = ({ selected, onChange }) => {
  const handleToggle = (connectorType: string) => {
    const newSelection = selected.includes(connectorType)
      ? selected.filter((c) => c !== connectorType)
      : [...selected, connectorType];

    // Immediate update
    onChange(newSelection);

    // Persist to localStorage
    localStorage.setItem(
      `connectors_${searchSpaceId}_${chatId}`,
      JSON.stringify(newSelection)
    );

    // Feedback
    toast.info(
      selected.includes(connectorType)
        ? `${connectorType} disabled`
        : `${connectorType} enabled`
    );
  };

  return (
    <div>
      {CONNECTOR_TYPES.map((type) => (
        <Switch
          key={type}
          checked={selected.includes(type)}
          onCheckedChange={() => handleToggle(type)}
        />
      ))}
    </div>
  );
};
```

## 4. Data Feedback Loops

### Document Upload → Search Results

**Scenario**: Пользователь загружает новый документ, он появляется в результатах поиска

```
User uploads "ML_Paper.pdf"
    ↓
    Document created (status=processing)
    ↓
    Celery processes document
    ├─ Extract text
    ├─ Chunk into 50 chunks
    └─ Generate embeddings
    ↓
    Document status → completed
    ↓
User searches "machine learning"
    ↓
    Backend hybrid search includes new chunks
    ↓
    Results include snippets from ML_Paper.pdf
```

**Timeline**:
```
T+0s:    Upload complete
T+5s:    Processing complete (50 chunks stored)
T+10s:   User searches
T+10.5s: Search results include new document
```

### Connector Indexing → Fresh Data

**Scenario**: Пользователь добавляет Notion connector, система периодически синхронизирует данные

```
User adds Notion connector
    ↓
    Connector created with config: {"workspace_id": "..."}
    ↓
    Trigger initial indexing (Celery task)
    ↓
    Notion API fetches pages
    ↓
    Pages converted to documents & chunks
    ↓
    Embeddings generated & stored
    ↓
    Connector.last_indexed_at updated
    ↓
    Enable periodic indexing (every 24 hours)
    ↓
    [24 hours later]
    ↓
    Celery scheduler triggers re-indexing
    ↓
    Fetch updated/new pages from Notion
    ↓
    Update existing chunks, add new ones
    ↓
    Search results now include latest Notion content
```

**Implementation**:

```python
# Periodic indexing with Celery Beat

@celery_app.task
def periodic_connector_indexing():
    """
    Scheduled task to re-index connectors.
    Runs every hour, checks which connectors need indexing.
    """

    with get_db_session() as session:
        # Find connectors due for indexing
        now = datetime.utcnow()

        connectors = session.execute(
            select(SearchSourceConnector).filter(
                SearchSourceConnector.periodic_indexing_enabled == True,
                SearchSourceConnector.next_scheduled_at <= now,
            )
        ).scalars().all()

        for connector in connectors:
            # Trigger indexing task
            index_connector.delay(connector.id)

            # Update next scheduled time
            connector.next_scheduled_at = now + timedelta(
                minutes=connector.indexing_frequency_minutes
            )

        session.commit()


@celery_app.task
def index_connector(connector_id: int):
    """Index specific connector"""

    with get_db_session() as session:
        connector = session.get(SearchSourceConnector, connector_id)

        if connector.connector_type == "NOTION_CONNECTOR":
            # Fetch from Notion API
            pages = fetch_notion_pages(connector.config)

            # Process each page
            for page in pages:
                # Check if already exists
                existing = session.execute(
                    select(Document).filter(
                        Document.source_id == page.id,
                        Document.source_type == "NOTION",
                    )
                ).scalar_one_or_none()

                if existing:
                    # Update if content changed
                    if existing.content_hash != page.content_hash:
                        update_document(existing, page)
                else:
                    # Create new document
                    create_document_from_notion_page(page, connector)

        connector.last_indexed_at = datetime.utcnow()
        session.commit()
```

## 5. Learning Feedback Loops (Future)

### Implicit Preference Learning

**Scenario**: Система учится на действиях пользователя для улучшения результатов

```
User frequently clicks on YouTube sources
    ↓
    System tracks: source_type="YOUTUBE_VIDEO", click_count++
    ↓
    [After N interactions]
    ↓
    System infers: User prefers video content
    ↓
    Future searches boost YouTube results
```

**Potential Implementation**:

```python
# Track user interactions
class UserInteraction(Base):
    __tablename__ = "user_interactions"

    id = Column(Integer, primary_key=True)
    user_id = Column(UUID, ForeignKey("user.id"))
    interaction_type = Column(String)  # "click", "like", "copy", "share"
    source_type = Column(String)  # "YOUTUBE_VIDEO", "FILE", etc.
    document_id = Column(Integer, ForeignKey("documents.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)


# Compute preferences
async def get_user_source_preferences(user_id: str) -> dict[str, float]:
    """
    Calculate user's source type preferences based on interactions.

    Returns:
        {"YOUTUBE_VIDEO": 0.8, "FILE": 0.5, ...}
    """

    interactions = await session.execute(
        select(
            UserInteraction.source_type,
            func.count().label("count")
        )
        .filter(UserInteraction.user_id == user_id)
        .group_by(UserInteraction.source_type)
    )

    total = sum(row.count for row in interactions)

    preferences = {
        row.source_type: row.count / total
        for row in interactions
    }

    return preferences


# Apply preferences to search results
async def rerank_by_preferences(
    documents: list[Document],
    user_id: str,
) -> list[Document]:
    """Boost documents from preferred sources"""

    preferences = await get_user_source_preferences(user_id)

    for doc in documents:
        # Boost score based on preference
        preference_boost = preferences.get(doc.source_type, 0.5)
        doc.score *= (1 + preference_boost)

    return sorted(documents, key=lambda d: d.score, reverse=True)
```

## Real-World Feedback Flow Examples

### Example 1: Complete Chat Query Flow

```
┌─────────────────────────────────────────────────────────────┐
│ USER ACTION: Sends chat query "What is RAG?"                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ├─ [0ms] Immediate Feedback
                 │  └─ Input clears, "Sending..." spinner
                 │
                 ├─ [100ms] Progressive Feedback
                 │  └─ "Reformulating query..."
                 │
                 ├─ [500ms] Progressive Feedback
                 │  └─ "Searching connectors..."
                 │
                 ├─ [1500ms] Data Feedback
                 │  └─ Sources panel populates
                 │
                 ├─ [2000ms] Progressive Feedback
                 │  └─ Answer starts streaming
                 │
                 ├─ [5000ms] Progressive Feedback
                 │  └─ Answer complete
                 │
                 └─ [6000ms] Configuration Feedback
                    └─ Follow-up questions generated (using user's LLM config)
```

### Example 2: Settings Change Flow

```
┌─────────────────────────────────────────────────────────────┐
│ USER ACTION: Changes top_k from 5 to 10                     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ├─ [0ms] Immediate Feedback
                 │  └─ Slider updates visually
                 │
                 ├─ [10ms] Configuration Feedback
                 │  └─ State updates (setTopK(10))
                 │
                 ├─ [50ms] Immediate Feedback
                 │  └─ "Settings saved ✓" toast
                 │
                 └─ [Next query] Configuration Feedback
                    └─ Backend receives top_k=10, returns 10 results per connector
```

### Example 3: Document Upload Flow

```
┌─────────────────────────────────────────────────────────────┐
│ USER ACTION: Uploads research_paper.pdf                     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ├─ [0-2000ms] Progressive Feedback
                 │  └─ Upload progress bar 0% → 100%
                 │
                 ├─ [2100ms] Immediate Feedback
                 │  └─ "Document uploaded!" toast
                 │
                 ├─ [2200-8000ms] Progressive Feedback
                 │  └─ "Processing..." spinner + status polls
                 │
                 ├─ [8100ms] Data Feedback
                 │  └─ Document status → "Completed"
                 │
                 ├─ [8200ms] Immediate Feedback
                 │  └─ Document list refreshes, shows 50 chunks
                 │
                 └─ [Next search] Data Feedback
                    └─ Search results include chunks from new document
```

## Best Practices for Feedback Design

### 1. Acknowledge Immediately
```typescript
// ✅ Good: Instant acknowledgment
const handleSubmit = async () => {
  setIsSubmitting(true); // Immediate UI update
  await submitForm();
  setIsSubmitting(false);
};

// ❌ Bad: No feedback until completion
const handleSubmit = async () => {
  await submitForm(); // User waits wondering if click registered
};
```

### 2. Show Progress
```typescript
// ✅ Good: Granular progress
const steps = ["Uploading", "Processing", "Indexing", "Complete"];
const [currentStep, setCurrentStep] = useState(0);

// ❌ Bad: Single loading state
const [isLoading, setIsLoading] = useState(false);
```

### 3. Explain Delays
```typescript
// ✅ Good: Contextual message
<Spinner message="Generating embeddings for 100 chunks..." />

// ❌ Bad: Generic loading
<Spinner />
```

### 4. Enable Cancellation
```typescript
// ✅ Good: User can abort
<Button onClick={handler.stop}>Cancel</Button>

// ❌ Bad: User is stuck waiting
// (no cancel option)
```

### 5. Persist Intent
```typescript
// ✅ Good: Survive navigation
localStorage.setItem("draft_message", input);

// ❌ Bad: Lost on refresh
// (only in memory)
```

## Итоговая диаграмма всех feedback loops

```
┌────────────────────────────────────────────────────────────┐
│                      USER ACTIONS                          │
│  Type query | Select docs | Change LLM | Upload file       │
└──────┬─────────────┬────────────┬─────────────┬────────────┘
       │             │            │             │
       │             │            │             │
   Immediate    Configuration   Data        Progressive
   Feedback      Feedback      Feedback     Feedback
       │             │            │             │
       ▼             ▼            ▼             ▼
┌────────────────────────────────────────────────────────────┐
│                    FEEDBACK TYPES                          │
│                                                            │
│  UI Updates   Next Request   Search        Streaming      │
│  (< 50ms)     Uses Config    Results       Updates        │
│                              Updated       (100ms-30s)     │
│                              (processing)                  │
└────────────────────────────────────────────────────────────┘
```

Все эти feedback loops работают вместе для создания responsive, transparent и adaptive user experience.
