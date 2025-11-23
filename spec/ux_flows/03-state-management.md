# Управление состоянием (State Management)

## Введение

SurfSense использует многоуровневую архитектуру управления состоянием, синхронизируя данные между frontend, backend и persistent storage. Этот документ описывает паттерны state management и механизмы синхронизации.

## Трехуровневая архитектура состояния

```
┌──────────────────────────────────────────────────────────────────┐
│  Level 1: Client State (React + Jotai)                          │
│  - UI state (form inputs, modals, loading states)               │
│  - Transient state (не сохраняется)                             │
│  - Optimistic updates                                           │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           │ TanStack Query (cache layer)
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│  Level 2: Server State (LangGraph + Services)                   │
│  - Workflow state (agent execution)                             │
│  - Streaming buffers                                            │
│  - Configuration (runtime)                                      │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           │ SQLAlchemy ORM
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│  Level 3: Persistent State (PostgreSQL)                         │
│  - User data (chats, documents, preferences)                    │
│  - Embeddings & vectors                                         │
│  - Source of truth                                              │
└──────────────────────────────────────────────────────────────────┘
```

## Level 1: Client State (Frontend)

### Jotai Atoms

**Atomic State Management** - минимальные единицы состояния

#### Active Search Space

**File**: `surfsense_web/atoms/seach-spaces/seach-space-queries.atom.ts`

```typescript
import { atom } from "jotai";

// Simple atom для текущего search space ID
export const activeSearchSpaceIdAtom = atom<string | null>(null);

// Derived atom с TanStack Query integration
export const activeSearchSpaceAtom = atomWithQuery((get) => {
  const activeId = get(activeSearchSpaceIdAtom);

  return {
    queryKey: cacheKeys.activeSearchSpace.details(activeId ?? ""),
    queryFn: async ({ queryKey }) => {
      const [, id] = queryKey;
      if (!id) return null;

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_FASTAPI_BACKEND_URL}/api/v1/search_spaces/${id}`,
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("surfsense_bearer_token")}`,
          },
        }
      );

      if (!response.ok) throw new Error("Failed to fetch search space");

      return response.json();
    },
    enabled: !!activeId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  };
});
```

#### Active Chat

**File**: `surfsense_web/atoms/chats/chat-querie.atoms.ts`

```typescript
export const activeChatIdAtom = atom<string | null>(null);

export const activeChatAtom = atomWithQuery<ActiveChatState>((get) => {
  const activeChatId = get(activeChatIdAtom);

  return {
    queryKey: cacheKeys.activeSearchSpace.activeChat(activeChatId ?? ""),
    queryFn: async () => {
      if (!activeChatId) return null;

      // Parallel fetch для chat details и podcast
      const [chatDetails, podcast] = await Promise.all([
        chatApiService.getChatDetails({ id: Number(activeChatId) }),
        getPodcastByChatId(activeChatId, authToken),
      ]);

      return {
        chatId: activeChatId,
        chatDetails,
        podcast,
      };
    },
    enabled: !!activeChatId,
  };
});
```

#### UI State

**File**: `surfsense_web/atoms/chats/ui.atoms.ts`

```typescript
interface ActiveChathatUIState {
  isChatPannelOpen: boolean;
  selectedTab: "sources" | "artifacts" | "settings";
  isTerminalExpanded: boolean;
}

export const activeChathatUIAtom = atom<ActiveChathatUIState>({
  isChatPannelOpen: false,
  selectedTab: "sources",
  isTerminalExpanded: true,
});

// Derived atoms для specific properties
export const isChatPannelOpenAtom = atom(
  (get) => get(activeChathatUIAtom).isChatPannelOpen,
  (get, set, newValue: boolean) => {
    set(activeChathatUIAtom, {
      ...get(activeChathatUIAtom),
      isChatPannelOpen: newValue,
    });
  }
);
```

### TanStack Query (Server State Cache)

**File**: `surfsense_web/lib/query-client/query-client.provider.tsx`

```typescript
import { QueryClient } from "@tanstack/react-query";
import { QueryClientAtomProvider } from "jotai-tanstack-query";

// Global query client
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes (formerly cacheTime)
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
    },
    mutations: {
      retry: 1,
    },
  },
});

export function ReactQueryClientProvider({ children }) {
  return (
    <QueryClientAtomProvider client={queryClient}>
      {children}
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientAtomProvider>
  );
}
```

**Cache Keys Structure**:

```typescript
// lib/query-client/cache-keys.ts

export const cacheKeys = {
  auth: {
    user: ["auth", "user"],
  },

  searchSpaces: {
    all: ["search-spaces"],
    details: (id: string) => ["search-spaces", id],
  },

  activeSearchSpace: {
    details: (id: string) => ["active-search-space", id],
    chats: (searchSpaceId: string) => ["active-search-space", "chats", searchSpaceId],
    activeChat: (chatId: string) => ["active-search-space", "active-chat", chatId],
    documents: (searchSpaceId: string) => [
      "active-search-space",
      "documents",
      searchSpaceId,
    ],
    connectors: (searchSpaceId: string) => [
      "active-search-space",
      "connectors",
      searchSpaceId,
    ],
    llmConfigs: (searchSpaceId: string) => [
      "active-search-space",
      "llm-configs",
      searchSpaceId,
    ],
  },

  chats: {
    list: (searchSpaceId: string) => ["chats", searchSpaceId],
    details: (chatId: string) => ["chats", chatId],
  },

  documents: {
    list: (searchSpaceId: string) => ["documents", searchSpaceId],
    details: (documentId: string) => ["documents", documentId],
  },
};
```

### Custom Hooks для State Encapsulation

#### useChatState

**File**: `surfsense_web/hooks/use-chat.ts`

```typescript
export function useChatState({ chat_id }: UseChatStateProps) {
  const [token, setToken] = useState<string | null>(null);

  // Chat configuration state
  const [searchMode, setSearchMode] = useState<"DOCUMENTS" | "CHUNKS">("DOCUMENTS");
  const [researchMode, setResearchMode] = useState<ResearchMode>("QNA");
  const [selectedConnectors, setSelectedConnectors] = useState<string[]>([]);
  const [selectedDocuments, setSelectedDocuments] = useState<Document[]>([]);
  const [topK, setTopK] = useState<number>(5);

  // Load from localStorage on mount
  useEffect(() => {
    const storedToken = localStorage.getItem("surfsense_bearer_token");
    setToken(storedToken);

    if (chat_id) {
      const restoredState = restoreChatState(search_space_id, chat_id);
      if (restoredState) {
        setSearchMode(restoredState.searchMode);
        setResearchMode(restoredState.researchMode);
        setSelectedConnectors(restoredState.selectedConnectors);
        setSelectedDocuments(restoredState.selectedDocuments);
        setTopK(restoredState.topK);
      }
    }
  }, [chat_id]);

  return {
    token,
    searchMode,
    setSearchMode,
    researchMode,
    setResearchMode,
    selectedConnectors,
    setSelectedConnectors,
    selectedDocuments,
    setSelectedDocuments,
    topK,
    setTopK,
  };
}
```

#### useChats (with TanStack Query)

**File**: `surfsense_web/hooks/use-chats.ts`

```typescript
export function useChats(searchSpaceId: string) {
  return useQuery({
    queryKey: cacheKeys.chats.list(searchSpaceId),
    queryFn: async () => {
      const token = localStorage.getItem("surfsense_bearer_token");
      const response = await fetch(
        `${BACKEND_URL}/api/v1/chats?search_space_id=${searchSpaceId}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) throw new Error("Failed to fetch chats");

      return response.json();
    },
    enabled: !!searchSpaceId,
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
}

// Mutations
export function useCreateChat() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: CreateChatRequest) => {
      return chatApiService.createChat(data);
    },
    onSuccess: (newChat, variables) => {
      // Invalidate chats list
      queryClient.invalidateQueries({
        queryKey: cacheKeys.chats.list(variables.search_space_id.toString()),
      });

      // Optimistically update cache
      queryClient.setQueryData(
        cacheKeys.chats.details(newChat.id.toString()),
        newChat
      );
    },
  });
}

export function useUpdateChat() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: UpdateChatRequest) => {
      return chatApiService.updateChat(data);
    },
    onMutate: async (variables) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({
        queryKey: cacheKeys.chats.details(variables.id.toString()),
      });

      // Snapshot previous value
      const previousChat = queryClient.getQueryData(
        cacheKeys.chats.details(variables.id.toString())
      );

      // Optimistically update
      queryClient.setQueryData(
        cacheKeys.chats.details(variables.id.toString()),
        (old: any) => ({
          ...old,
          ...variables,
        })
      );

      return { previousChat };
    },
    onError: (err, variables, context) => {
      // Rollback on error
      if (context?.previousChat) {
        queryClient.setQueryData(
          cacheKeys.chats.details(variables.id.toString()),
          context.previousChat
        );
      }
    },
    onSettled: (data, error, variables) => {
      // Refetch to ensure consistency
      queryClient.invalidateQueries({
        queryKey: cacheKeys.chats.details(variables.id.toString()),
      });
    },
  });
}
```

### LocalStorage Persistence

```typescript
// lib/utils/chat-state-storage.ts

interface ChatState {
  selectedDocuments: Document[];
  selectedConnectors: string[];
  searchMode: "DOCUMENTS" | "CHUNKS";
  researchMode: "QNA";
  topK: number;
}

export const storeChatState = (
  searchSpaceId: string,
  chatId: string,
  state: ChatState
) => {
  const key = `surfsense_chat_state_${searchSpaceId}_${chatId}`;
  localStorage.setItem(key, JSON.stringify(state));
};

export const restoreChatState = (
  searchSpaceId: string,
  chatId: string
): ChatState | null => {
  const key = `surfsense_chat_state_${searchSpaceId}_${chatId}`;
  const stored = localStorage.getItem(key);

  if (!stored) return null;

  try {
    return JSON.parse(stored);
  } catch {
    return null;
  }
};

export const clearChatState = (searchSpaceId: string, chatId: string) => {
  const key = `surfsense_chat_state_${searchSpaceId}_${chatId}`;
  localStorage.removeItem(key);
};
```

## Level 2: Server State (Backend)

### LangGraph State

**File**: `surfsense_backend/app/agents/researcher/state.py`

```python
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.streaming_service import StreamingService


@dataclass
class State:
    """
    Researcher agent workflow state.
    Shared across all nodes in the graph.
    """

    # Dependencies
    db_session: AsyncSession
    streaming_service: StreamingService

    # Conversation context
    chat_history: list[Any] | None = field(default_factory=list)

    # Intermediate results (updated by nodes)
    reformulated_query: str | None = field(default=None)
    reranked_documents: list = field(default_factory=list)
    final_written_report: str = field(default="")
    further_questions: Any = field(default=None)

    # Metadata
    start_time: float = field(default=0.0)
    end_time: float = field(default=0.0)
```

**State Updates**:

```python
# Nodes return dict with updated fields
async def reformulate_user_query(state: State, config: RunnableConfig) -> dict:
    # ... processing logic ...

    # Return updates (merged into state)
    return {
        "reformulated_query": reformulated_text,
    }

# LangGraph automatically merges returned dict into State
```

### Configuration (Runtime Parameters)

**File**: `surfsense_backend/app/agents/researcher/configuration.py`

```python
from dataclasses import dataclass
from enum import Enum


class SearchMode(str, Enum):
    CHUNKS = "CHUNKS"
    DOCUMENTS = "DOCUMENTS"


@dataclass
class Configuration:
    """
    Runtime configuration passed to graph execution.
    Read-only during workflow (immutable).
    """

    # User query
    user_query: str

    # Search configuration
    connectors_to_search: list[str]
    search_mode: SearchMode
    top_k: int

    # Context
    user_id: str
    search_space_id: int
    document_ids_to_add_in_context: list[int]

    # LLM settings
    language: str | None = None

    # Metadata
    request_id: str = None
```

**Usage in Nodes**:

```python
async def handle_qna_workflow(state: State, config: RunnableConfig):
    # Access configuration
    configuration: Configuration = config["configurable"]

    connectors = configuration.connectors_to_search
    top_k = configuration.top_k
    user_id = configuration.user_id

    # Configuration is immutable - cannot be updated
    # Only State can be modified
```

### Service Layer State

#### LLM Service (Singleton Pattern)

**File**: `surfsense_backend/app/services/llm_service.py`

```python
class LLMService:
    """
    Manages LLM instance creation and caching.
    Maintains in-memory cache of instantiated models.
    """

    _instance_cache: dict[str, ChatLiteLLM] = {}

    @classmethod
    async def get_user_llm_instance(
        cls,
        session: AsyncSession,
        user_id: str,
        search_space_id: int,
        role: str,  # "long_context", "fast", "strategic"
    ) -> ChatLiteLLM:
        # Generate cache key
        cache_key = f"{user_id}_{search_space_id}_{role}"

        # Return cached instance if available
        if cache_key in cls._instance_cache:
            return cls._instance_cache[cache_key]

        # Fetch user preference from database
        user_pref = await session.execute(
            select(UserSearchSpacePreference).filter(
                UserSearchSpacePreference.user_id == user_id,
                UserSearchSpacePreference.search_space_id == search_space_id,
            )
        )
        preference = user_pref.scalar_one_or_none()

        # Get LLM config ID based on role
        if role == "long_context":
            llm_config_id = preference.long_context_llm_id
        elif role == "fast":
            llm_config_id = preference.fast_llm_id
        elif role == "strategic":
            llm_config_id = preference.strategic_llm_id

        # Load config (global or custom)
        if llm_config_id < 0:
            # Global config from YAML
            llm_instance = await cls._load_global_llm(llm_config_id)
        else:
            # Custom config from database
            llm_instance = await cls._load_custom_llm(session, llm_config_id)

        # Cache instance
        cls._instance_cache[cache_key] = llm_instance

        return llm_instance

    @classmethod
    def clear_cache(cls, user_id: str = None):
        """Clear cache (e.g., when user updates LLM config)"""
        if user_id:
            # Clear specific user's cache
            keys_to_remove = [k for k in cls._instance_cache.keys() if k.startswith(user_id)]
            for key in keys_to_remove:
                del cls._instance_cache[key]
        else:
            # Clear all
            cls._instance_cache.clear()
```

#### StreamingService State

```python
class StreamingService:
    """
    Maintains state for current streaming session.
    Reset for each new message.
    """

    def __init__(self):
        self.terminal_idx = 0
        self.message_annotations = [
            {"type": "TERMINAL_INFO", "content": []},
            {"type": "SOURCES", "content": []},
            {"type": "ANSWER", "content": []},
            {"type": "FURTHER_QUESTIONS", "content": []},
        ]

    def reset(self):
        """Reset for new message"""
        self.terminal_idx = 0
        self.message_annotations = [
            {"type": "TERMINAL_INFO", "content": []},
            {"type": "SOURCES", "content": []},
            {"type": "ANSWER", "content": []},
            {"type": "FURTHER_QUESTIONS", "content": []},
        ]
```

## Level 3: Persistent State (Database)

### SQLAlchemy Models

#### User Preferences

```python
# app/models/user_search_space_preference.py

class UserSearchSpacePreference(Base):
    __tablename__ = "user_search_space_preferences"

    user_id: Mapped[UUID] = mapped_column(ForeignKey("user.id"), primary_key=True)
    search_space_id: Mapped[int] = mapped_column(
        ForeignKey("search_spaces.id"), primary_key=True
    )

    # LLM configuration
    long_context_llm_id: Mapped[int] = mapped_column(default=-1)
    fast_llm_id: Mapped[int] = mapped_column(default=-2)
    strategic_llm_id: Mapped[int] = mapped_column(default=-3)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="search_space_preferences")
    search_space: Mapped["SearchSpace"] = relationship(
        back_populates="user_preferences"
    )
```

#### Chat Model

```python
# app/models/chat.py

class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))

    # Messages stored as JSONB
    messages: Mapped[list[dict]] = mapped_column(
        JSONB,
        default=[],
        server_default="[]",
    )

    # Optimistic locking
    state_version: Mapped[int] = mapped_column(default=0, server_default="0")

    # Metadata
    research_mode: Mapped[str] = mapped_column(String(50), default="QNA")
    selected_connectors: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        default=[],
        server_default="{}",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user.id"))
    search_space_id: Mapped[int] = mapped_column(ForeignKey("search_spaces.id"))

    user: Mapped["User"] = relationship(back_populates="chats")
    search_space: Mapped["SearchSpace"] = relationship(back_populates="chats")
```

### Database Transactions

```python
# Optimistic locking для concurrent updates

async def update_chat_with_optimistic_lock(
    session: AsyncSession,
    chat_id: int,
    new_messages: list[dict],
    expected_version: int,
):
    # Fetch current chat
    result = await session.execute(
        select(Chat).filter(Chat.id == chat_id)
    )
    chat = result.scalar_one_or_none()

    if not chat:
        raise NotFoundError("Chat not found")

    # Check version
    if chat.state_version != expected_version:
        raise ConflictError(
            f"Chat has been modified. Expected version {expected_version}, "
            f"but current is {chat.state_version}"
        )

    # Update
    chat.messages = new_messages
    chat.state_version += 1
    chat.updated_at = datetime.utcnow()

    await session.commit()

    return chat
```

## State Synchronization Patterns

### Pattern 1: Optimistic Updates

```typescript
// Frontend immediately updates UI, then syncs to backend

const handleLikeSource = async (sourceId: number) => {
  // 1. Optimistic UI update
  setSources((prev) =>
    prev.map((s) => (s.id === sourceId ? { ...s, liked: true } : s))
  );

  try {
    // 2. Sync to backend
    await fetch(`${BACKEND_URL}/api/v1/sources/${sourceId}/like`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    });
  } catch (error) {
    // 3. Rollback on error
    setSources((prev) =>
      prev.map((s) => (s.id === sourceId ? { ...s, liked: false } : s))
    );
    toast.error("Failed to like source");
  }
};
```

### Pattern 2: Server Authoritative

```typescript
// Backend is source of truth, frontend subscribes to updates

const { data: chat, refetch } = useQuery({
  queryKey: cacheKeys.chats.details(chatId),
  queryFn: () => chatApiService.getChatDetails({ id: chatId }),
});

// Poll for updates (or use WebSocket in future)
useEffect(() => {
  const interval = setInterval(() => {
    refetch();
  }, 5000); // Refresh every 5 seconds

  return () => clearInterval(interval);
}, [refetch]);
```

### Pattern 3: Eventual Consistency

```typescript
// Frontend auto-saves periodically, tolerates temporary inconsistency

const autoSave = useMemo(
  () =>
    debounce(async (chatId: string, messages: Message[]) => {
      try {
        await chatApiService.updateChat({
          id: Number(chatId),
          messages: messages,
          state_version: messages.length,
        });
        toast.success("Auto-saved", { duration: 1000 });
      } catch (error) {
        console.error("Auto-save failed:", error);
        // Will retry on next edit
      }
    }, 2000), // Save 2 seconds after last edit
  []
);

useEffect(() => {
  if (handler.messages.length > 0 && handler.status === "ready") {
    autoSave(chatId, handler.messages);
  }
}, [handler.messages, handler.status]);
```

### Pattern 4: Cache Invalidation

```typescript
// Invalidate cache when data changes

const { mutate: deleteChat } = useMutation({
  mutationFn: async (chatId: string) => {
    return chatApiService.deleteChat({ id: Number(chatId) });
  },
  onSuccess: (_, chatId) => {
    // Invalidate related caches
    queryClient.invalidateQueries({
      queryKey: cacheKeys.chats.list(searchSpaceId),
    });
    queryClient.removeQueries({
      queryKey: cacheKeys.chats.details(chatId),
    });

    // Redirect to chat list
    router.push(`/dashboard/${searchSpaceId}/chats`);

    toast.success("Chat deleted");
  },
});
```

## State Flow Examples

### Example 1: Create New Chat

```
User clicks "New Chat"
    ↓
Frontend:
    1. Set isCreating = true (UI state)
    2. Call createChat mutation
    ↓
Backend:
    3. Insert into database
    4. Return new chat object
    ↓
Frontend:
    5. Update TanStack Query cache
    6. Navigate to /researcher/{new_chat_id}
    7. Set isCreating = false
    ↓
URL changes → React renders new chat page
    ↓
useChat initializes with empty messages
```

### Example 2: Send Message in Existing Chat

```
User types message and clicks Send
    ↓
Frontend:
    1. handler.append() (AI SDK)
    2. Optimistically add user message to UI
    3. Set status = "streaming"
    ↓
Backend:
    4. Receive POST /api/v1/chat
    5. Initialize LangGraph state
    6. Execute workflow
    7. Stream SSE responses
    ↓
Frontend:
    8. Process SSE chunks
    9. Update message.content (incremental)
    10. Update message.annotations (sources, events)
    11. Re-render UI
    ↓
Backend:
    12. Send completion marker
    ↓
Frontend:
    13. Set status = "ready"
    14. Trigger auto-save (debounced)
    ↓
Backend:
    15. Update chat.messages in database
    16. Increment chat.state_version
```

### Example 3: Update LLM Config

```
User changes LLM in settings
    ↓
Frontend:
    1. Set isSaving = true
    2. Call updateLLMConfig mutation
    ↓
Backend:
    3. Validate config
    4. Update UserSearchSpacePreference
    5. Clear LLMService cache for user
    6. Return success
    ↓
Frontend:
    7. Invalidate llmConfigs cache
    8. Refetch configs
    9. Set isSaving = false
    10. Toast "Settings saved"
    ↓
Next chat query uses new LLM automatically
```

## Debugging State Issues

### Frontend DevTools

```typescript
// React Query DevTools (already included)
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";

<ReactQueryClientProvider>
  {children}
  <ReactQueryDevtools initialIsOpen={false} />
</ReactQueryClientProvider>;
```

```typescript
// Jotai DevTools
import { DevTools } from "jotai-devtools";

<DevTools>
  <App />
</DevTools>;
```

### Backend Logging

```python
# Log state transitions in LangGraph

async def reformulate_user_query(state: State, config: RunnableConfig):
    logger.info(
        f"Node: reformulate_user_query | "
        f"User: {config['configurable'].user_id} | "
        f"Query: {config['configurable'].user_query[:50]}"
    )

    # ... processing ...

    logger.info(
        f"Node: reformulate_user_query COMPLETE | "
        f"Reformulated: {reformulated[:50]}"
    )

    return {"reformulated_query": reformulated}
```

### Database Queries

```sql
-- Check chat state
SELECT id, name, state_version, updated_at, array_length(messages, 1) as message_count
FROM chats
WHERE user_id = 'user-uuid'
ORDER BY updated_at DESC;

-- Check user preferences
SELECT * FROM user_search_space_preferences
WHERE user_id = 'user-uuid' AND search_space_id = 123;

-- Check LLM configs
SELECT id, name, provider, model_name, language
FROM llm_configs
WHERE search_space_id = 123;
```

## Best Practices

### 1. Single Source of Truth
- Database является source of truth
- Frontend cache должен быть invalidated при изменениях
- Optimistic updates должны rollback при ошибках

### 2. Immutable State Updates
```typescript
// ❌ Bad: Mutating state
messages.push(newMessage);

// ✅ Good: Creating new array
setMessages([...messages, newMessage]);
```

### 3. Normalized State
```typescript
// ❌ Bad: Nested duplicates
{
  chats: [
    { id: 1, search_space: { id: 123, name: "My Space" } },
    { id: 2, search_space: { id: 123, name: "My Space" } }, // Duplicate!
  ];
}

// ✅ Good: Normalized
{
  chats: { 1: { id: 1, search_space_id: 123 }, 2: { id: 2, search_space_id: 123 } },
  searchSpaces: { 123: { id: 123, name: "My Space" } }
}
```

### 4. Derived State
```typescript
// ❌ Bad: Storing computed values
const [messageCount, setMessageCount] = useState(0);

// ✅ Good: Deriving from source
const messageCount = messages.length;
```

### 5. State Colocation
```typescript
// ❌ Bad: Global state для local UI
const [isModalOpen, setIsModalOpen] = useAtom(globalModalAtom);

// ✅ Good: Local state для local UI
const [isModalOpen, setIsModalOpen] = useState(false);
```

## Итоговая диаграмма

```
┌─────────────────────────────────────────────────────────────┐
│                     Client State (React)                    │
│                                                             │
│  Jotai Atoms ◄──┐                                           │
│      ▲          │                                           │
│      │          │                                           │
│      │          ├── TanStack Query (Server Cache)           │
│      │          │        ▲                                  │
│      │          │        │ Fetch/Invalidate                 │
│  React         │        │                                  │
│  Components ───┘        │                                  │
│      │                  │                                  │
└──────┼──────────────────┼───────────────────────────────────┘
       │                  │
       │ User Actions     │ API Calls
       │                  │
┌──────▼──────────────────▼───────────────────────────────────┐
│                    Backend (FastAPI)                        │
│                                                             │
│  Route Handlers                                             │
│      │                                                      │
│      ├──► Service Layer (stateless)                        │
│      │                                                      │
│      └──► LangGraph (stateful workflow)                    │
│             │                                               │
│             ├── State (per-request)                        │
│             └── Configuration (immutable)                  │
│                                                             │
│      │                                                      │
└──────┼──────────────────────────────────────────────────────┘
       │
       │ ORM Queries
       │
┌──────▼──────────────────────────────────────────────────────┐
│                   Database (PostgreSQL)                     │
│                                                             │
│  Tables:                                                    │
│    - chats (messages JSONB, state_version INT)              │
│    - user_search_space_preferences (LLM configs)            │
│    - search_spaces, documents, chunks, embeddings           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Эта многоуровневая архитектура обеспечивает:
- **Responsive UI** через optimistic updates
- **Data consistency** через cache invalidation
- **Offline support** через localStorage
- **Scalability** через stateless backend services
