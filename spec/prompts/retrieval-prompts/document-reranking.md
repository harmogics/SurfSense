# Переранжирование документов (Document Reranking)

## Описание

Промпт для оптимизации релевантности извлеченных документов путем их переранжирования относительно запроса пользователя. Использует специализированные reranker модели (FlashRank) для повышения точности поиска после гибридного retrieval.

## Категория
**Retrieval Prompt** - оптимизация поиска и ранжирования

## Роль в цепочке преобразований

### Позиция в Pipeline
**Промежуточный узел** между извлечением документов и генерацией ответа

```
retrieve_documents → [RERANK_DOCUMENTS] → answer_question
```

### Входящие данные
1. **Кандидаты документов** (top 20-50) от hybrid search
2. **Запрос пользователя** (оригинальный или переформулированный)
3. **Метрики первичного поиска** (vector similarity, BM25 score)

### Исходящие данные
1. **Ранжированный список** (top 5-10 наиболее релевантных документов)
2. **Relevance scores** - новые оценки релевантности
3. **Filtered documents** - документы выше порога релевантности

## Алгоритм переранжирования

### Гибридный подход
```python
# Комбинация методов:
# 1. Vector similarity (embedding-based)
# 2. BM25 (keyword-based)
# 3. Reranker model (cross-encoder)

final_score = (
    0.3 * vector_score +
    0.2 * bm25_score +
    0.5 * reranker_score
)
```

## Используемые библиотеки и инструменты

### Reranking Models
```python
from rerankers import Reranker
from rerankers.models.flashrank import FlashRankReranker

class DocumentReranker:
    def __init__(self):
        # Инициализация FlashRank reranker
        self.reranker = Reranker(
            "flashrank",
            model_name="ms-marco-MiniLM-L-12-v2"
        )

    async def rerank_documents(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 10
    ) -> list[Document]:
        """Rerank documents using FlashRank."""

        # Подготовка данных для reranker
        passages = [
            {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]

        # Переранжирование
        results = self.reranker.rank(
            query=query,
            docs=passages,
            top_k=top_k
        )

        # Сортировка документов по новым scores
        reranked_docs = [
            documents[result.doc_id]
            for result in results
        ]

        # Добавление reranker scores в metadata
        for doc, result in zip(reranked_docs, results):
            doc.metadata["reranker_score"] = result.score
            doc.metadata["reranker_rank"] = result.rank

        return reranked_docs
```

### Hybrid Search Integration
```python
from app.search.hybrid_search import hybrid_search

async def search_and_rerank(
    query: str,
    user_id: int,
    top_k: int = 10,
    use_reranker: bool = True
) -> list[Document]:
    """Combined search and reranking pipeline."""

    # Шаг 1: Гибридный поиск (vector + keyword)
    # Получаем больше кандидатов для переранжирования
    candidate_docs = await hybrid_search(
        query=query,
        user_id=user_id,
        top_k=top_k * 3,  # 30 кандидатов для reranking
        alpha=0.7  # Вес vector search vs keyword search
    )

    # Шаг 2: Переранжирование (опционально)
    if use_reranker and len(candidate_docs) > 0:
        reranker = DocumentReranker()
        reranked_docs = await reranker.rerank_documents(
            query=query,
            documents=candidate_docs,
            top_k=top_k
        )
        return reranked_docs

    # Без reranker - просто берем top_k
    return candidate_docs[:top_k]
```

### Evaluation Metrics
```python
from sklearn.metrics import ndcg_score
import numpy as np

def evaluate_reranking_quality(
    original_docs: list[Document],
    reranked_docs: list[Document],
    relevance_labels: list[int]
) -> dict:
    """Evaluate reranking quality metrics."""

    # NDCG (Normalized Discounted Cumulative Gain)
    original_scores = [doc.metadata.get("initial_score", 0) for doc in original_docs]
    reranked_scores = [doc.metadata.get("reranker_score", 0) for doc in reranked_docs]

    ndcg_original = ndcg_score([relevance_labels], [original_scores])
    ndcg_reranked = ndcg_score([relevance_labels], [reranked_scores])

    # Precision@K
    def precision_at_k(docs, labels, k=5):
        return sum(labels[:k]) / k

    return {
        "ndcg_improvement": (ndcg_reranked - ndcg_original) / ndcg_original * 100,
        "ndcg_original": ndcg_original,
        "ndcg_reranked": ndcg_reranked,
        "precision@5_original": precision_at_k(original_docs, relevance_labels, 5),
        "precision@5_reranked": precision_at_k(reranked_docs, relevance_labels, 5)
    }
```

## Связи с другими промптами

### Зависит от
- **[Query Reformulation](../task-prompts/query-reformulation.md)** - улучшенный запрос
- **Hybrid Search** - кандидаты документов

### Предоставляет вход для
- **[Q&A with Citations](../system-prompts/qna-with-citations.md)** - релевантные документы
- **Token Optimization** - усечение по релевантности

## Пример использования

### Входные данные
```python
query = "How does Python asyncio improve performance?"

documents = [
    Document(
        page_content="Python asyncio library for concurrent programming...",
        metadata={"initial_score": 0.78, "source_id": 5}
    ),
    Document(
        page_content="Asyncio performance improvements in I/O operations...",
        metadata={"initial_score": 0.82, "source_id": 12}
    ),
    Document(
        page_content="Python threading vs asyncio comparison...",
        metadata={"initial_score": 0.75, "source_id": 8}
    )
]
```

### Выходные данные
```python
reranked_documents = [
    Document(  # Переранжировано на первое место
        page_content="Asyncio performance improvements in I/O operations...",
        metadata={
            "initial_score": 0.82,
            "reranker_score": 0.94,
            "reranker_rank": 1,
            "source_id": 12
        }
    ),
    Document(
        page_content="Python asyncio library for concurrent programming...",
        metadata={
            "initial_score": 0.78,
            "reranker_score": 0.89,
            "reranker_rank": 2,
            "source_id": 5
        }
    ),
    Document(
        page_content="Python threading vs asyncio comparison...",
        metadata={
            "initial_score": 0.75,
            "reranker_score": 0.71,
            "reranker_rank": 3,
            "source_id": 8
        }
    )
]
```

## Конфигурация

### Модели Reranker
```python
RERANKER_MODELS = {
    "flashrank-mini": "ms-marco-MiniLM-L-6-v2",   # Быстрая, меньше точность
    "flashrank-base": "ms-marco-MiniLM-L-12-v2",  # Баланс
    "flashrank-large": "ms-marco-TinyBERT-L-6"     # Более точная
}

# Выбор модели в зависимости от требований
reranker_config = {
    "model": RERANKER_MODELS["flashrank-base"],
    "batch_size": 32,
    "max_length": 512
}
```

### Пороги релевантности
```python
RELEVANCE_THRESHOLDS = {
    "high": 0.8,    # Очень релевантные
    "medium": 0.6,  # Средне релевантные
    "low": 0.4      # Минимально релевантные
}

# Фильтрация по порогу
def filter_by_relevance(docs: list[Document], threshold: float = 0.6):
    return [
        doc for doc in docs
        if doc.metadata.get("reranker_score", 0) >= threshold
    ]
```

## Оптимизации

### Кэширование результатов
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def get_reranking_cache_key(query: str, doc_ids: tuple) -> str:
    """Generate cache key for reranking results."""
    return hashlib.md5(f"{query}:{','.join(map(str, doc_ids))}".encode()).hexdigest()

async def rerank_with_cache(query: str, documents: list[Document]) -> list[Document]:
    doc_ids = tuple(doc.metadata["id"] for doc in documents)
    cache_key = get_reranking_cache_key(query, doc_ids)

    # Check Redis cache
    cached = await redis.get(f"rerank:{cache_key}")
    if cached:
        return pickle.loads(cached)

    # Rerank
    result = await reranker.rerank_documents(query, documents)

    # Cache for 1 hour
    await redis.setex(f"rerank:{cache_key}", 3600, pickle.dumps(result))

    return result
```

## Метрики качества

### Критерии оценки
1. **NDCG@10** (0.85+) - Normalized Discounted Cumulative Gain
2. **Precision@5** (0.90+) - Точность top-5 результатов
3. **Recall Improvement** (+15%+) - Улучшение полноты
4. **Latency** (<200ms) - Время переранжирования
5. **User Satisfaction** (85%+) - Пользователь доволен результатами

## См. также

- [Query Reformulation](../task-prompts/query-reformulation.md) - Предшествующий шаг
- [Q&A with Citations](../system-prompts/qna-with-citations.md) - Использует результаты
- [Библиотеки и инструменты](../libraries-and-tools.md) - Reranker stack
- [Цепочки преобразований](../transformation-chains.md) - Retrieval pipeline
