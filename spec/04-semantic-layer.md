# Семантическая Обработка и Концептуальный Слой

## Введение

Семантический слой в SurfSense отвечает за извлечение смысла из документов, формирование концептуальных представлений и обеспечение семантического поиска. Это ключевой компонент, который превращает сырой текст в структурированные знания.

## Уровни семантической обработки

```
┌────────────────────────────────────────────────────────────┐
│              SEMANTIC PROCESSING LAYERS                    │
└────────────────────────────────────────────────────────────┘

LEVEL 1: TEXT EXTRACTION & NORMALIZATION
─────────────────────────────────────────
  Raw Content → Markdown → Normalized Text
  • ETL processing (Unstructured, Docling, LlamaCloud)
  • Structure preservation (headers, lists, tables, code)
  • Metadata extraction (title, author, date, source)

        ↓

LEVEL 2: CONCEPTUAL EXTRACTION (LLM-based)
───────────────────────────────────────────
  Normalized Text → LLM Summarization → Conceptual Summary
  • Key topics identification
  • Main themes extraction
  • Entity recognition (implicit via LLM)
  • Hierarchical structuring

        ↓

LEVEL 3: SEMANTIC EMBEDDING (Vector Representation)
────────────────────────────────────────────────────
  Conceptual Summary → Embedding Model → Vector[1536/3072]
  • Dense vector representation
  • Semantic similarity encoding
  • Multi-lingual support
  • Domain adaptation (via fine-tuning options)

        ↓

LEVEL 4: SEMANTIC INDEXING & RETRIEVAL
───────────────────────────────────────
  Vectors → pgvector Storage → Hybrid Search
  • Vector similarity search (cosine distance)
  • Full-text search (PostgreSQL FTS)
  • Hybrid RRF fusion
  • Semantic reranking
```

## 1. Концептуальная экстракция через LLM

### Генерация Summary (Концептуальный слой)

**Функция**: `generate_document_summary()`
**Местоположение**: `surfsense_backend/app/utils/document_converters.py:97-145`

**Назначение**: Извлечение концептуальной сути документа через LLM summarization.

#### Процесс концептуальной экстракции

```python
async def generate_document_summary(
    content: str,
    user_llm: BaseChatModel,
    document_metadata: dict | None = None
) -> tuple[str, list[float]]:
    """
    Генерирует концептуальный summary документа.

    Концептуальные аспекты:
    1. Main Topics: ключевые темы и концепции
    2. Key Entities: важные сущности (люди, места, технологии)
    3. Relationships: связи между концепциями
    4. Hierarchical Structure: организация знаний по уровням
    5. Actionable Insights: практические выводы

    Input:
        content: Полный текст документа (Markdown)
        user_llm: LLM instance (LONG_CONTEXT role)
        document_metadata: Контекстная информация

    Process:
        1. Content Optimization: адаптация под context window
        2. Metadata Integration: обогащение контекстом
        3. LLM Summarization: концептуальная экстракция
        4. Embedding Generation: векторизация summary
        5. Quality Validation: проверка полноты

    Output:
        (summary_content: str, summary_embedding: list[float])

    Conceptual Quality Metrics:
        - Coverage: охват основных концепций
        - Accuracy: точность представления
        - Abstraction: уровень обобщения
        - Coherence: связность концепций
    """
```

### SUMMARY_PROMPT_TEMPLATE (Концептуальная экстракция)

**Местоположение**: `surfsense_backend/app/prompts/__init__.py`

```python
SUMMARY_PROMPT_TEMPLATE = """
You are an expert knowledge analyst specializing in conceptual extraction and summarization.

## Document Metadata
{metadata}

## Document Content
{content}

## Task
Create a comprehensive conceptual summary that extracts and organizes the key knowledge from this document.

## Conceptual Extraction Guidelines

### 1. Main Concepts & Topics
- Identify 3-7 core concepts or topics
- Explain each concept clearly and concisely
- Show relationships between concepts

### 2. Key Entities & Definitions
- Extract important entities (people, places, technologies, organizations)
- Provide clear definitions for domain-specific terms
- Contextualize entities within the document's scope

### 3. Hierarchical Knowledge Structure
- Organize information from general to specific
- Create logical groupings of related concepts
- Show parent-child relationships where applicable

### 4. Key Facts & Data Points
- Preserve critical facts, figures, and dates
- Maintain quantitative information
- Keep specific technical details

### 5. Actionable Insights
- Extract practical implications or applications
- Identify recommendations or best practices
- Highlight important conclusions

## Output Format
Provide the summary in well-structured Markdown:

# Conceptual Summary

## Overview
[High-level conceptual overview in 2-3 sentences]

## Core Concepts
### [Concept 1 Name]
[Explanation and significance]

### [Concept 2 Name]
[Explanation and significance]

[Continue for all core concepts...]

## Key Entities & Definitions
- **[Entity/Term]**: [Definition and context]
- **[Entity/Term]**: [Definition and context]

## Detailed Analysis
[Deeper exploration organized by topics/themes]

### [Topic 1]
[Content related to topic 1]

### [Topic 2]
[Content related to topic 2]

## Key Takeaways
- [Important conclusion 1]
- [Important conclusion 2]
- [Important conclusion 3]

## Facts & Figures
- [Important fact/stat 1]
- [Important fact/stat 2]

Begin your conceptual summary now:
"""
```

### Пример концептуального summary

**Исходный документ** (фрагмент технической документации):
```
PostgreSQL Performance Tuning Guide

This document covers advanced techniques for optimizing PostgreSQL database performance...

[5000 words of technical content about indexes, query optimization, caching, etc.]
```

**Концептуальный summary**:
```markdown
# Conceptual Summary

## Overview
This document presents a comprehensive framework for PostgreSQL performance optimization, covering three primary dimensions: indexing strategies, query optimization techniques, and caching mechanisms. The approach emphasizes data-driven tuning based on EXPLAIN ANALYZE and system metrics.

## Core Concepts

### 1. Index Optimization
Indexes are the primary mechanism for improving query performance in PostgreSQL. The document distinguishes between:
- **B-tree indexes**: Default, optimal for range queries and sorting
- **Hash indexes**: Specialized for equality comparisons
- **GIN/GiST indexes**: Full-text search and complex data types

**Key Principle**: Index selectivity (percentage of rows matched) determines effectiveness. High selectivity = better performance.

### 2. Query Execution Plans
Understanding query execution plans is fundamental to optimization. PostgreSQL's planner chooses between:
- Sequential scans (full table reads)
- Index scans (using indexes)
- Bitmap scans (combining multiple indexes)

**Analysis Tool**: EXPLAIN ANALYZE reveals actual vs estimated costs.

### 3. Caching Layers
Multi-level caching strategy:
- **shared_buffers**: PostgreSQL's internal buffer pool (25% of RAM recommended)
- **OS page cache**: Filesystem caching layer
- **Application cache**: Redis/Memcached for query results

## Key Entities & Definitions

- **EXPLAIN ANALYZE**: PostgreSQL command that shows query execution plan with actual runtime statistics
- **shared_buffers**: PostgreSQL configuration parameter controlling internal memory allocation for caching
- **VACUUM**: Maintenance process that reclaims storage and updates statistics
- **Planner**: PostgreSQL's query optimizer component
- **Cardinality**: Number of unique values in a column, affects index usefulness

## Detailed Analysis

### Index Strategy Selection

**When to use B-tree indexes**:
- Range queries (>, <, BETWEEN)
- Sorting operations (ORDER BY)
- Pattern matching (LIKE 'prefix%')

**When to use Hash indexes**:
- Exact equality matches only (=)
- Memory-efficient for large datasets
- Note: Not WAL-logged until PostgreSQL 10+

### Query Optimization Workflow

1. **Identify slow queries** using pg_stat_statements
2. **Analyze execution plan** with EXPLAIN ANALYZE
3. **Identify bottlenecks**:
   - Sequential scans on large tables → add indexes
   - High loop counts in nested loops → adjust join strategy
   - Large difference between estimated and actual rows → run ANALYZE
4. **Implement optimizations**
5. **Measure improvements**

### Caching Configuration

**Shared Buffers Tuning**:
```sql
-- For systems with 16GB RAM
shared_buffers = '4GB'  -- 25% of total RAM
effective_cache_size = '12GB'  -- 75% of total RAM
```

**Trade-offs**:
- More shared_buffers = less memory for OS page cache
- Optimal balance depends on workload characteristics

## Key Takeaways

- **Index optimization is data-dependent**: What works for one dataset may not work for another
- **Measure, don't assume**: Always use EXPLAIN ANALYZE to verify optimizations
- **Caching is multi-layered**: Optimize at database, OS, and application levels
- **Statistics are critical**: Outdated statistics lead to poor query plans
- **Regular maintenance is essential**: VACUUM and ANALYZE should run frequently

## Facts & Figures

- B-tree indexes provide O(log n) lookup time
- Hash indexes are 30-50% faster for equality comparisons but less versatile
- Recommended shared_buffers: 25% of system RAM (but not more than 8GB for most workloads)
- Sequential scans become faster than index scans when accessing >5-10% of table rows
- EXPLAIN ANALYZE overhead: 10-20% additional runtime for measurement
```

**Концептуальная глубина**:
- **Абстрактные концепции**: Indexing strategies, query optimization, caching layers
- **Конкретные техники**: Specific index types, EXPLAIN ANALYZE usage
- **Взаимосвязи**: How indexes affect query plans, cache interdependencies
- **Практические инсайты**: When to use which technique, recommended configurations

## 2. Извлечение структурных элементов

### Конвертация элементов в Markdown

**Функция**: `convert_element_to_markdown()`
**Местоположение**: `surfsense_backend/app/utils/document_converters.py:167-202`

**Назначение**: Сохранение семантической структуры документа при конвертации.

#### Mapping типов элементов на семантические структуры

```python
def convert_element_to_markdown(element: dict) -> str:
    """
    Конвертирует структурный элемент в Markdown с сохранением семантики.

    Semantic Mappings:
    ==================

    STRUCTURAL ELEMENTS:
    - Title → # Heading (semantic: document/section title)
    - Header → ## Subheading (semantic: section header)
    - NarrativeText → Paragraph (semantic: main content)
    - ListItem → - Bullet / 1. Number (semantic: enumeration)

    MATHEMATICAL & SCIENTIFIC:
    - Formula → ```math (semantic: mathematical expression)
    - FigureCaption → *Figure: ...* (semantic: visual description)
    - Table → HTML table (semantic: structured data)

    CODE & TECHNICAL:
    - CodeSnippet → ``` language (semantic: executable code)
    - Address → `code` inline (semantic: reference/path)

    METADATA:
    - Footer → <!-- metadata --> (semantic: supplementary info)
    - PageBreak → --- (semantic: section separator)
    """

    element_type = element.get('type')
    text = element.get('text', '')
    metadata = element.get('metadata', {})

    # Semantic processing based on type
    if element_type == 'Title':
        # Semantic: Main document title or section heading
        return f"# {text}\n\n"

    elif element_type == 'NarrativeText':
        # Semantic: Primary content carrying main information
        return f"{text}\n\n"

    elif element_type == 'ListItem':
        # Semantic: Enumerated items, structured information
        # Preserve numbering if present
        if metadata.get('is_numbered'):
            return f"{metadata.get('number', '1')}. {text}\n"
        else:
            return f"- {text}\n"

    elif element_type == 'Table':
        # Semantic: Structured data with relationships
        # Convert to HTML for rich formatting
        return convert_table_to_html(element)

    elif element_type == 'CodeSnippet':
        # Semantic: Executable or reference code
        language = metadata.get('language', '')
        return f"```{language}\n{text}\n```\n\n"

    elif element_type == 'Formula':
        # Semantic: Mathematical expression or equation
        # LaTeX format for proper rendering
        return f"```math\n{text}\n```\n\n"

    elif element_type == 'FigureCaption':
        # Semantic: Description of visual content
        # Preserve as italic for distinction
        return f"*Figure: {text}*\n\n"

    elif element_type == 'Image':
        # Semantic: Visual content reference
        image_path = metadata.get('image_path', '')
        if image_path:
            return f"![{text}]({image_path})\n\n"
        else:
            return f"*[Image: {text}]*\n\n"

    else:
        # Default: treat as narrative text
        return f"{text}\n\n"
```

### Семантическое значение структурных элементов

| Элемент | Семантика | Концептуальная роль |
|---------|-----------|---------------------|
| **Title** | Идентификатор темы | Определяет основную концепцию документа/секции |
| **NarrativeText** | Контент знания | Содержит основные факты и объяснения |
| **ListItem** | Перечисление | Структурирует связанные концепции или этапы |
| **Table** | Структурированные данные | Представляет отношения между сущностями |
| **Formula** | Математическое знание | Формализует количественные отношения |
| **CodeSnippet** | Процедурное знание | Определяет алгоритмы и реализации |
| **FigureCaption** | Визуальное знание | Описывает графические концепции |

## 3. Metadata как концептуальный контекст

### Формирование метаданных для семантического обогащения

**Функция**: `build_document_metadata_markdown()`
**Местоположение**: `surfsense_backend/app/tasks/connector_indexers/base.py:202-232`

#### Структура метаданных

```python
# Пример метаданных для обогащения контекста
document_metadata = {
    # Идентификация источника (концептуальный контекст)
    "source_type": "technical_documentation",
    "domain": "database_systems",
    "subdomain": "performance_optimization",

    # Temporal context (временной контекст)
    "created_at": "2024-01-15",
    "last_modified": "2024-03-20",
    "version": "2.1",

    # Authorship (экспертный контекст)
    "author": "Database Team",
    "author_expertise": "senior_dba",
    "organization": "Tech Corp",

    # Content characteristics (характеристики контента)
    "language": "en",
    "content_type": "how-to_guide",
    "technical_level": "advanced",
    "estimated_reading_time": "15 minutes",

    # Relationships (концептуальные связи)
    "related_topics": ["indexing", "query_optimization", "postgresql"],
    "prerequisites": ["basic_sql", "database_fundamentals"],
    "references": ["postgresql_docs", "performance_tuning_book"],

    # Source-specific metadata
    "FILE_NAME": "postgres_performance_guide.pdf",
    "URL": "https://docs.example.com/postgres-perf",
    "PAGE_COUNT": 45
}
```

### Интеграция метаданных в концептуальный слой

**Обогащение summary метаданными**:
```python
async def generate_document_summary(content, user_llm, document_metadata):
    # Форматирование метаданных для LLM
    metadata_context = format_metadata_for_llm(document_metadata)

    # Метаданные влияют на концептуальную экстракцию:
    # 1. Domain context → определяет терминологию
    # 2. Technical level → влияет на глубину объяснений
    # 3. Content type → определяет структуру summary
    # 4. Related topics → обеспечивает контекстные связи

    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        metadata=metadata_context,
        content=optimized_content
    )

    # Enhanced summary включает концептуальный контекст
    enhanced_summary = f"""
    {metadata_context}

    {llm_generated_summary}
    """

    # Embedding включает метаданные для семантической близости
    embedding = embedding_model.embed(enhanced_summary)

    return enhanced_summary, embedding
```

## 4. Оптимизация контента для context window

### Binary Search для семантической полноты

**Функция**: `optimize_content_for_context_window()`
**Местоположение**: `surfsense_backend/app/utils/document_converters.py:23-94`

**Концептуальная задача**: Максимизировать семантическую полноту при ограничениях на токены.

```python
def optimize_content_for_context_window(
    content: str,
    document_metadata: dict | None,
    model_name: str
) -> str:
    """
    Оптимизирует контент для максимальной семантической полноты.

    Semantic Optimization Strategy:
    1. Preserve structural elements (headings, lists)
    2. Maintain conceptual coherence (don't cut mid-concept)
    3. Prioritize informational density
    4. Keep metadata intact

    Algorithm:
    - Binary search for maximum content length
    - Token counting via litellm
    - Reserve tokens for:
      * Prompt template (500 tokens)
      * Output generation (1500 tokens)
      * Metadata context (variable)
      * Safety buffer (200 tokens)

    Conceptual Trade-offs:
    - Longer content = more concepts but risk truncation
    - Shorter content = guaranteed processing but less coverage
    - Optimization targets sweet spot for completeness
    """

    # Model capacity
    max_input_tokens = get_model_max_tokens(model_name)

    # Reservations
    PROMPT_TOKENS = 500
    OUTPUT_TOKENS = 1500
    SAFETY_BUFFER = 200

    # Metadata overhead
    metadata_tokens = count_tokens(format_metadata(document_metadata), model_name)

    # Available tokens for content
    available_tokens = max_input_tokens - PROMPT_TOKENS - OUTPUT_TOKENS - SAFETY_BUFFER - metadata_tokens

    # Binary search for optimal content length
    left, right = 0, len(content)
    best_length = 0

    # Optimization: Try to preserve sentence/paragraph boundaries
    while left <= right:
        mid = (left + right) // 2

        # Find nearest sentence boundary
        candidate_length = find_nearest_sentence_boundary(content, mid)
        prefix = content[:candidate_length]

        token_count = count_tokens(prefix, model_name)

        if token_count <= available_tokens:
            best_length = candidate_length
            left = mid + 1
        else:
            right = mid - 1

    optimized_content = content[:best_length]

    # Add truncation indicator if needed
    if best_length < len(content):
        # Semantic indicator of incompleteness
        optimized_content += "\n\n[Content truncated to fit context window. Summary covers first {} of {} total characters.]".format(
            best_length, len(content)
        )

    return optimized_content


def find_nearest_sentence_boundary(text: str, position: int, lookback: int = 200) -> int:
    """
    Находит ближайшую границу предложения для сохранения концептуальной целостности.

    Semantic Importance:
    Cutting mid-sentence or mid-paragraph breaks conceptual flow.
    This function ensures concepts remain complete.
    """
    # Search backwards for sentence terminators
    search_start = max(0, position - lookback)
    search_text = text[search_start:position]

    # Sentence terminators in order of preference
    terminators = ['. ', '.\n', '! ', '!\n', '? ', '?\n', '\n\n']

    for terminator in terminators:
        last_pos = search_text.rfind(terminator)
        if last_pos != -1:
            # Return position after terminator
            return search_start + last_pos + len(terminator)

    # Fallback: use original position
    return position
```

## 5. Концептуальный слой в векторном пространстве

### Embedding как семантическая репрезентация

**Векторные embeddings** превращают концептуальный контент в плотные векторные представления, которые кодируют семантическое значение.

#### Характеристики semantic embeddings

```python
# Document-level embedding (summary)
summary_embedding = embedding_model.embed(conceptual_summary)
# Dimension: 1536 или 3072
# Encodes: Main concepts, themes, entities, relationships

# Концептуальные свойства embeddings:
# 1. Semantic Similarity: похожие концепции → близкие векторы
# 2. Compositional: vector arithmetic отражает концептуальные отношения
# 3. Multi-lingual: cross-lingual semantic equivalence
# 4. Domain-aware: fine-tuned models capture domain-specific semantics
```

#### Семантические операции в векторном пространстве

```python
# 1. Concept Similarity
def compute_semantic_similarity(concept_a: str, concept_b: str) -> float:
    """
    Вычисляет семантическую близость концепций.
    """
    emb_a = embedding_model.embed(concept_a)
    emb_b = embedding_model.embed(concept_b)

    # Cosine similarity
    similarity = cosine_similarity(emb_a, emb_b)
    return similarity

# Example:
similarity = compute_semantic_similarity(
    "database indexing",
    "query optimization"
)
# Result: ~0.75 (high semantic relatedness)


# 2. Concept Clustering (тематические группы)
def cluster_concepts(documents: list[Document], n_clusters: int = 5):
    """
    Группирует документы по концептуальной близости.
    """
    embeddings = [doc.embedding for doc in documents]

    # K-means clustering в semantic space
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings)

    # Каждый cluster = концептуальная тема
    return group_documents_by_cluster(documents, clusters)


# 3. Concept Navigation (многошаговое исследование)
def navigate_concept_graph(start_concept: str, hops: int = 2, k: int = 5):
    """
    Навигация по семантическому графу концепций.
    """
    visited = set()
    current_concepts = [start_concept]

    for hop in range(hops):
        next_concepts = []

        for concept in current_concepts:
            # Найти k ближайших концепций
            similar_docs = semantic_search(concept, top_k=k)

            for doc in similar_docs:
                if doc.id not in visited:
                    next_concepts.append(extract_main_concept(doc))
                    visited.add(doc.id)

        current_concepts = next_concepts

    return list(visited)

# Example: "async programming" → ["event loop", "promises", "callbacks", ...]
```

## Резюме: Концептуальный слой

| Уровень | Техника | Концептуальная роль | Технология |
|---------|---------|---------------------|------------|
| **Извлечение** | LLM Summarization | Идентификация ключевых концепций | GPT-4, Claude, Gemini |
| **Структурирование** | Markdown Conversion | Сохранение семантической структуры | ETL parsers |
| **Контекст** | Metadata Integration | Обогащение концептуального контекста | JSON metadata |
| **Векторизация** | Embeddings | Семантическая репрезентация | text-embedding-3 |
| **Навигация** | Vector Search | Исследование концептуального пространства | pgvector |

**Ключевые принципы**:
1. **Многоуровневая абстракция**: От текста к концепциям к векторам
2. **Контекстное обогащение**: Метаданные усиливают семантику
3. **Структурная осведомленность**: Сохранение иерархии знаний
4. **Семантическая навигация**: Граф концепций через embeddings
