# Загрузка и ETL Обработка Документов

## Введение

Модуль загрузки и ETL обработки документов отвечает за прием документов из различных источников, их парсинг, конвертацию в унифицированный формат (Markdown) и подготовку к дальнейшей индексации.

## Архитектура модуля

**Основной модуль**: `surfsense_backend/app/tasks/document_processors/`

**Ключевые файлы**:
- `file_processors.py` - основная логика обработки файлов
- `audio_processors.py` - обработка аудио через STT
- `surfsense_backend/app/utils/document_converters.py` - утилиты конвертации

## Поддерживаемые источники документов

### 1. Файлы (File Upload)

**Поддерживаемые форматы**:
- Документы: PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS
- Текст: TXT, MD, RTF
- Код: PY, JS, TS, JAVA, CPP, C, GO, RS, etc.
- Изображения: PNG, JPG, JPEG (OCR через ETL сервисы)
- Аудио: MP3, WAV, M4A, OGG (транскрипция через STT)

**Точка входа**: `process_file_in_background()`
- Местоположение: `file_processors.py:404-1022`

### 2. Web URLs (Crawling)

**Процесс**:
1. Crawling веб-страниц через встроенный crawler
2. Извлечение контента (HTML → Markdown)
3. Извлечение метаданных (title, url, date)

### 3. Connectors (внешние сервисы)

**Поддерживаемые источники** (15+):
- Slack, GitHub, Notion, Jira, Confluence
- Discord, Google Calendar, Gmail
- Linear, ClickUp, Airtable, Luma
- Elasticsearch, и другие

**Детали**: См. документ `07-connectors.md`

## ETL Сервисы

SurfSense поддерживает три ETL сервиса для обработки документов:

### 1. Unstructured (API)

**Конфигурация**:
```python
ETL_SERVICE = "UNSTRUCTURED"
UNSTRUCTURED_API_URL = "https://api.unstructured.io/general/v0/general"
UNSTRUCTURED_API_KEY = "your_api_key"
```

**Возможности**:
- Парсинг сложных документов (PDF с таблицами, изображениями)
- Извлечение структурных элементов (Title, Table, Figure, Formula, etc.)
- OCR для изображений и сканированных документов
- Сохранение иерархии документа

**Функция**: `add_received_file_document_using_unstructured()`
- Местоположение: `file_processors.py:34-147`

**Процесс**:
```python
async def add_received_file_document_using_unstructured(
    session: AsyncSession,
    user_id: str,
    search_space_id: int,
    file_elements: list,  # Результат Unstructured parser
    original_file_name: str,
    file_type: str,
    original_file_content: bytes,
    document_metadata: dict | None = None
) -> int:
    """
    1. Конвертация elements в Markdown
    2. Генерация hashes (content_hash, unique_identifier_hash)
    3. Проверка дубликатов
    4. Генерация summary + embedding
    5. Создание chunks с embeddings
    6. Сохранение в БД
    """
```

**Element types** (Unstructured):
```python
- Title → # heading
- NarrativeText → paragraph
- ListItem → - bullet
- Table → HTML table
- Image → ![alt](path)
- Formula → ```math ... ```
- FigureCaption → *Figure: ...*
- Header, Footer → metadata
```

### 2. LlamaCloud (API)

**Конфигурация**:
```python
ETL_SERVICE = "LLAMACLOUD"
LLAMACLOUD_API_KEY = "your_api_key"
LLAMACLOUD_BASE_URL = "https://api.cloud.llamaindex.ai"
```

**Возможности**:
- Managed ETL pipeline от LlamaIndex
- Высокоуровневая обработка документов
- Автоматическая конвертация в Markdown
- Встроенная генерация metadata

**Функция**: `add_received_file_document_using_llamacloud()`
- Местоположение: `file_processors.py:149-263`

**Процесс**:
```python
async def add_received_file_document_using_llamacloud(
    session: AsyncSession,
    user_id: str,
    search_space_id: int,
    markdown_document: str,  # Готовый Markdown от LlamaCloud
    original_file_name: str,
    file_type: str,
    original_file_content: bytes,
    document_metadata: dict | None = None
) -> int:
    """
    1. Получение готового Markdown
    2. Генерация hashes
    3. Проверка дубликатов
    4. Генерация summary + embedding
    5. Создание chunks
    6. Сохранение в БД
    """
```

### 3. Docling (Local/API)

**Конфигурация**:
```python
ETL_SERVICE = "DOCLING"
DOCLING_SERVICE_URL = "http://localhost:5000" or API endpoint
```

**Возможности**:
- Локальная или cloud-based обработка
- Специализация на больших документах
- Поддержка batch processing
- Оптимизация для технических документов

**Функция**: `add_received_file_document_using_docling()`
- Местоположение: `file_processors.py:265-401`

**Сервис**: `surfsense_backend/app/services/docling_service.py`

```python
class DoclingService:
    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        mime_type: str
    ) -> str:
        """
        Отправляет документ в Docling сервис для обработки.
        Возвращает Markdown контент.
        """
```

## Основной Pipeline обработки файлов

### Функция: `process_file_in_background()`

**Местоположение**: `file_processors.py:404-1022`

**Сигнатура**:
```python
async def process_file_in_background(
    session: AsyncSession,
    file_id: int,
    user_id: str,
    search_space_id: int,
    etl_service: str = "UNSTRUCTURED",
    task_logging_service: TaskLoggingService | None = None
) -> tuple[int, str | None]:
    """
    Обрабатывает файл в фоновом режиме.

    Returns:
        (document_id, error_message)
    """
```

### Этапы обработки:

#### 1. Получение файла из БД
```python
# Загрузка file_obj из таблицы File
file_obj = await session.get(File, file_id)
file_content = file_obj.file_content
file_name = file_obj.file_name
file_type = file_obj.file_type
```

#### 2. Проверка лимитов
```python
# Проверка PAGE_LIMIT для больших документов
if has_page_limit:
    if estimated_pages > PAGE_LIMIT:
        raise PageLimitExceeded(f"Document has {estimated_pages} pages, limit is {PAGE_LIMIT}")
```

#### 3. Выбор ETL сервиса

**Для аудио файлов**:
```python
if file_type in ['audio/mpeg', 'audio/wav', 'audio/mp3']:
    # Использование STT сервиса
    from app.services.stt_service import stt_service
    transcript = await stt_service.transcribe(file_content, file_name)
    markdown_content = transcript
```

**Для текстовых файлов** (TXT, MD):
```python
if file_type in ['text/plain', 'text/markdown']:
    # Прямая конвертация без ETL
    markdown_content = file_content.decode('utf-8')
```

**Для остальных документов**:
```python
if etl_service == "UNSTRUCTURED":
    # Парсинг через Unstructured API
    elements = await unstructured_parser.parse(file_content, file_name)
    return await add_received_file_document_using_unstructured(...)

elif etl_service == "LLAMACLOUD":
    # Обработка через LlamaCloud
    markdown_doc = await llamacloud_parser.parse(file_content, file_name)
    return await add_received_file_document_using_llamacloud(...)

elif etl_service == "DOCLING":
    # Обработка через Docling
    markdown_doc = await docling_service.process_document(file_content, file_name, file_type)
    return await add_received_file_document_using_docling(...)
```

#### 4. Конвертация в Markdown

**Функция**: `convert_element_to_markdown()`
**Местоположение**: `document_converters.py:167-202`

```python
def convert_element_to_markdown(element: dict) -> str:
    """
    Конвертирует элемент документа в Markdown.

    Mapping:
    - Formula → ```math ... ```
    - FigureCaption → *Figure: description*
    - Title → # Heading
    - Table → HTML table
    - CodeSnippet → ``` language ... ```
    - ListItem → - bullet / 1. numbered
    - NarrativeText → paragraph
    - Image → ![alt](path) or base64 embedded
    """
    element_type = element.get('type')
    text = element.get('text', '')
    metadata = element.get('metadata', {})

    if element_type == 'Formula':
        return f"```math\n{text}\n```\n\n"
    elif element_type == 'FigureCaption':
        return f"*Figure: {text}*\n\n"
    elif element_type == 'Title':
        return f"# {text}\n\n"
    elif element_type == 'Table':
        return format_table_as_html(element)
    # ... и т.д.
```

#### 5. Генерация Hashes

**Content Hash** (дедупликация по содержимому):
```python
import hashlib

content_hash = hashlib.sha256(markdown_content.encode()).hexdigest()
```

**Unique Identifier Hash** (дедупликация по источнику):
```python
# Для файлов: filename + user_id + search_space_id
unique_identifier = f"{file_name}:{user_id}:{search_space_id}"
unique_identifier_hash = hashlib.sha256(unique_identifier.encode()).hexdigest()

# Для connectors: используется специфичный ID (message_id, repo_url, etc.)
```

#### 6. Проверка дубликатов

**Функция**: `check_duplicate_document_by_hash()`, `check_document_by_unique_identifier()`
**Местоположение**: `connector_indexers/base.py:11-60`

```python
# Проверка по content_hash
existing_doc = await session.execute(
    select(Document).where(
        Document.content_hash == content_hash,
        Document.user_id == user_id,
        Document.search_space_id == search_space_id
    )
)

if existing_doc:
    # Документ уже существует, пропускаем
    return existing_doc.id, None

# Проверка по unique_identifier_hash
existing_doc = await session.execute(
    select(Document).where(
        Document.unique_identifier_hash == unique_identifier_hash,
        Document.user_id == user_id,
        Document.search_space_id == search_space_id
    )
)

if existing_doc:
    # Обновляем существующий документ (если содержимое изменилось)
    if existing_doc.content_hash != content_hash:
        await update_document_content(existing_doc, markdown_content)
    return existing_doc.id, None
```

#### 7. Извлечение и форматирование метаданных

**Метаданные документа**:
```python
document_metadata = {
    "FILE_NAME": original_file_name,
    "FILE_TYPE": file_type,
    "ETL_SERVICE": etl_service,
    "FILE_SIZE": len(file_content),
    "UPLOADED_AT": str(datetime.now()),
    "USER_ID": user_id,
    "SEARCH_SPACE_ID": search_space_id,
    # + custom metadata из connector или upload
}
```

**Форматирование метаданных в Markdown**:
**Функция**: `build_document_metadata_markdown()`
**Местоположение**: `connector_indexers/base.py:202-232`

```python
def build_document_metadata_markdown(metadata_sections: list[dict]) -> str:
    """
    Создает Markdown блок с метаданными.

    Input:
        metadata_sections = [
            {"title": "File Info", "fields": {"name": "doc.pdf", "type": "PDF"}},
            {"title": "Source", "fields": {"url": "https://..."}}
        ]

    Output:
        ---
        ## File Info
        - **name**: doc.pdf
        - **type**: PDF

        ## Source
        - **url**: https://...
        ---
    """
    markdown = "---\n"
    for section in metadata_sections:
        markdown += f"## {section['title']}\n"
        for key, value in section['fields'].items():
            markdown += f"- **{key}**: {value}\n"
        markdown += "\n"
    markdown += "---\n\n"
    return markdown
```

## Генерация Summary и первичных Embeddings

**Функция**: `generate_document_summary()`
**Местоположение**: `document_converters.py:97-145`

**Процесс**:
```python
async def generate_document_summary(
    content: str,
    user_llm: Any,  # LLM instance для пользователя
    document_metadata: dict | None = None
) -> tuple[str, list[float]]:
    """
    Генерирует summary документа с метаданными.

    Этапы:
    1. Оптимизация контента под context window модели
    2. Форматирование метаданных
    3. Генерация summary через LLM (SUMMARY_PROMPT_TEMPLATE)
    4. Генерация embedding для summary
    5. Возврат (summary_content, summary_embedding)
    """

    # 1. Оптимизация контента
    optimized_content = optimize_content_for_context_window(
        content,
        document_metadata,
        user_llm.model_name
    )

    # 2. Форматирование метаданных
    metadata_markdown = ""
    if document_metadata:
        metadata_markdown = format_metadata_as_markdown(document_metadata)

    # 3. Генерация summary
    from app.prompts import SUMMARY_PROMPT_TEMPLATE

    summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
        metadata=metadata_markdown,
        content=optimized_content
    )

    # Вызов LLM (long_context роль для больших документов)
    summary_response = await user_llm.ainvoke(summary_prompt)
    summary_content = summary_response.content

    # 4. Генерация embedding
    enhanced_summary = f"{metadata_markdown}\n{summary_content}"
    embedding = config.embedding_model_instance.embed(enhanced_summary)

    return summary_content, embedding
```

### SUMMARY_PROMPT_TEMPLATE

**Местоположение**: `surfsense_backend/app/prompts/__init__.py`

```python
SUMMARY_PROMPT_TEMPLATE = """
You are an expert document analyst tasked with creating a comprehensive summary.

## Document Metadata
{metadata}

## Document Content
{content}

## Instructions
Create a detailed summary that:
1. Accurately captures the main topics and themes
2. Preserves key facts, figures, and dates
3. Maintains objectivity and neutrality
4. Uses clear, concise language
5. Organizes information hierarchically (using Markdown headers)

## Output Format
Provide the summary in Markdown format with the following structure:

# Summary

## Overview
[High-level overview of the document]

## Key Points
- [Key point 1]
- [Key point 2]
- ...

## Details
[More detailed information organized by topics]

## Conclusions
[Main conclusions or takeaways]

Begin your summary now:
"""
```

**Характеристики summary**:
- **Accuracy**: точное отражение содержимого
- **Objectivity**: без субъективных оценок
- **Comprehensiveness**: охват всех важных аспектов
- **Structure**: иерархическая организация (Markdown)
- **Brevity**: краткость при сохранении полноты

## Оптимизация контента для Context Window

**Функция**: `optimize_content_for_context_window()`
**Местоположение**: `document_converters.py:23-94`

**Проблема**: LLM имеют ограничение на максимальное количество tokens в контексте.

**Решение**: Binary search для оптимального размера контента.

```python
def optimize_content_for_context_window(
    content: str,
    document_metadata: dict | None,
    model_name: str
) -> str:
    """
    Оптимизирует контент под context window модели.

    Процесс:
    1. Получение max_input_tokens для модели (через litellm)
    2. Резервирование tokens для prompt + metadata + output
    3. Binary search для определения максимального префикса контента
    4. Возврат обрезанного контента
    """
    import litellm

    # 1. Получение лимита модели
    model_info = litellm.get_model_info(model_name)
    max_input_tokens = model_info.get('max_input_tokens', 4096)

    # 2. Резервирование tokens
    PROMPT_TOKENS = 500  # SUMMARY_PROMPT_TEMPLATE
    OUTPUT_TOKENS = 1500  # Ожидаемый размер summary

    metadata_text = format_metadata_as_markdown(document_metadata) if document_metadata else ""
    metadata_tokens = litellm.token_counter(model=model_name, text=metadata_text)

    available_tokens = max_input_tokens - PROMPT_TOKENS - OUTPUT_TOKENS - metadata_tokens

    # 3. Binary search для оптимального размера
    left, right = 0, len(content)
    best_length = 0

    while left <= right:
        mid = (left + right) // 2
        prefix = content[:mid]

        token_count = litellm.token_counter(model=model_name, text=prefix)

        if token_count <= available_tokens:
            best_length = mid
            left = mid + 1
        else:
            right = mid - 1

    # 4. Возврат оптимизированного контента
    optimized_content = content[:best_length]

    if best_length < len(content):
        # Добавляем индикатор обрезки
        optimized_content += "\n\n[Content truncated to fit context window]"

    return optimized_content
```

**Пример резервирования** (модель с 8192 tokens):
```
max_input_tokens: 8192
- PROMPT_TOKENS: 500
- OUTPUT_TOKENS: 1500
- metadata_tokens: 200
─────────────────────────
available_tokens: 5992

→ Максимальный контент: ~5992 tokens (~24000 chars)
```

## Создание объекта Document

После всех этапов обработки создается объект Document:

```python
from app.db import Document, DocumentType

# Создание Document
document = Document(
    title=extract_title_from_content(markdown_content) or original_file_name,
    content=markdown_content,
    document_type=DocumentType.FILE,  # FILE, SLACK, GITHUB, etc.
    document_metadata=document_metadata,
    content_hash=content_hash,
    unique_identifier_hash=unique_identifier_hash,
    embedding=summary_embedding,  # Из generate_document_summary()
    user_id=user_id,
    search_space_id=search_space_id,
    file_id=file_id  # Связь с исходным файлом
)

session.add(document)
await session.flush()  # Получаем document.id

# Создание chunks (см. 06-chunking-indexing.md)
chunks = await create_document_chunks(markdown_content)
for chunk in chunks:
    chunk.document_id = document.id
    session.add(chunk)

await session.commit()
```

## Обработка ошибок и логирование

### Task Logging Service

**Местоположение**: `surfsense_backend/app/services/task_logging_service.py`

```python
# В начале обработки
log_entry = await task_logging_service.log_task_start(
    task_name="process_file",
    source="file_upload",
    message=f"Processing file: {file_name}",
    metadata={"file_id": file_id, "etl_service": etl_service}
)

# Прогресс
await task_logging_service.log_task_progress(
    log_entry,
    message="Parsing document with Unstructured",
    metadata={"elements_count": len(elements)}
)

# Успех
await task_logging_service.log_task_success(
    log_entry,
    message=f"Document processed successfully",
    metadata={"document_id": document.id, "chunks_count": len(chunks)}
)

# Ошибка
await task_logging_service.log_task_failure(
    log_entry,
    message="Failed to process document",
    error=str(exception),
    metadata={"file_id": file_id}
)
```

### Обработка специфичных ошибок

```python
try:
    # ETL processing
    elements = await unstructured_parser.parse(file_content, file_name)
except UnstructuredAPIError as e:
    return None, f"Unstructured API error: {str(e)}"
except PageLimitExceeded as e:
    return None, f"Page limit exceeded: {str(e)}"
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    return None, f"Failed to process file: {str(e)}"
```

## Резюме: Ключевые компоненты

| Компонент | Файл | Функция |
|-----------|------|---------|
| **Main Pipeline** | `file_processors.py` | `process_file_in_background()` |
| **Unstructured ETL** | `file_processors.py` | `add_received_file_document_using_unstructured()` |
| **LlamaCloud ETL** | `file_processors.py` | `add_received_file_document_using_llamacloud()` |
| **Docling ETL** | `file_processors.py` | `add_received_file_document_using_docling()` |
| **Summary Generation** | `document_converters.py` | `generate_document_summary()` |
| **Content Optimization** | `document_converters.py` | `optimize_content_for_context_window()` |
| **Markdown Conversion** | `document_converters.py` | `convert_element_to_markdown()` |
| **Duplicate Check** | `connector_indexers/base.py` | `check_duplicate_document_by_hash()` |
| **Metadata Formatting** | `connector_indexers/base.py` | `build_document_metadata_markdown()` |

## Следующие шаги

После успешной обработки и создания Document:
1. **Chunking**: разбиение контента на chunks (см. `06-chunking-indexing.md`)
2. **Embeddings**: генерация embeddings для chunks (см. `05-embeddings-search.md`)
3. **Indexing**: сохранение в БД с pgvector индексами
4. **Search**: доступность для hybrid search и AI агентов
