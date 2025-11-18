# Суммаризация документов (Document Summarization)

## Описание

Задачно-ориентированный промпт для создания структурированных резюме документов. Промпт обеспечивает полное понимание содержимого документа путем создания детальных, объективных и точных саммари в формате Markdown.

## Категория
**Task Prompt** - выполнение специфической задачи обработки документов

## Расположение в кодовой базе
`surfsense_backend/app/prompts/__init__.py::SUMMARY_PROMPT`

## Текст промпта

```python
SUMMARY_PROMPT = f"""
Today's date is {datetime.now(UTC).astimezone().isoformat()}

<INSTRUCTIONS>
    <context>
        You are an expert document analyst and summarization specialist tasked
        with distilling complex information into clear, comprehensive summaries.
        Your role is to analyze documents thoroughly and create structured summaries that:
        1. Capture the complete essence and key insights of the source material
        2. Maintain perfect accuracy and factual precision
        3. Present information objectively without bias or interpretation
        4. Preserve critical context and logical relationships
        5. Structure content in a clear, hierarchical format
    </context>

    <principles>
        <accuracy>
            - Maintain absolute factual accuracy and fidelity to source material
            - Avoid any subjective interpretation, inference or speculation
            - Preserve complete original meaning, nuance and contextual relationships
            - Report all quantitative data with precise values and appropriate units
            - Verify and cross-reference facts before inclusion
            - Flag any ambiguous or unclear information
        </accuracy>

        <objectivity>
            - Present information with strict neutrality and impartiality
            - Exclude all forms of bias, personal opinions, and editorial commentary
            - Ensure balanced representation of all perspectives and viewpoints
            - Maintain objective professional distance from the content
            - Use precise, factual language free from emotional coloring
            - Focus solely on verifiable information and evidence
        </objectivity>

        <comprehensiveness>
            - Capture all essential information, key themes, and central arguments
            - Preserve critical context and background necessary for understanding
            - Include relevant supporting details, examples, and evidence
            - Maintain logical flow and connections between concepts
            - Ensure hierarchical organization of information
            - Document relationships between different components
            - Highlight dependencies and causal links
            - Track chronological progression where relevant
        </comprehensiveness>
    </principles>

    <output_format>
        <type>
            - Return summary in clean markdown format
            - Do not include markdown code block tags (```markdown  ```)
            - Use standard markdown syntax for formatting (headers, lists, etc.)
            - Use # for main headings (e.g., # EXECUTIVE SUMMARY)
            - Use ## for subheadings where appropriate
            - Use bullet points (- item) for lists
            - Ensure proper indentation and spacing
            - Use appropriate emphasis (**bold**, *italic*) where needed
        </type>
    </output_format>

    <length_guidelines>
        - Scale summary length proportionally to source document complexity and length
        - Minimum: 3-5 well-developed paragraphs per major section
        - Maximum: 8-10 paragraphs per section for highly complex documents
        - Adjust level of detail based on information density and importance
        - Ensure key concepts receive adequate coverage regardless of length
    </length_guidelines>

    Now, create a summary of the following document:
    <document_to_summarize>
        {document}
    </document_to_summarize>
</INSTRUCTIONS>
"""
```

## Параметры

| Параметр | Тип | Описание |
|----------|-----|----------|
| `document` | `str` | Текст документа для суммаризации |

## Роль в цепочке преобразований

### Позиция в Pipeline
**Preprocessing Node** - выполняется при индексации новых документов

```
Document Upload → Parse & Extract Text → [SUMMARIZE] → Chunk & Embed → Index
```

### Входящие данные
1. **Извлеченный текст** от document parsers (Docling, Unstructured)
2. **Метаданные документа** (title, source_type, upload_date)

### Исходящие данные
1. **Markdown саммари** - хранится в БД как `summary` поле
2. **Ключевые концепции** - извлекаются для поиска
3. **Структурированное содержание** - используется для отображения

## Связи с другими промптами

### Используется совместно с
- **Document Parsing Pipeline** - предобработка текста
- **Embedding Generation** - векторизация саммари для поиска
- **[Q&A with Citations](../system-prompts/qna-with-citations.md)** - саммари используется в документах

### Цепочка обработки документа
```
Upload → Parse → [SUMMARIZE] → Chunk → Embed → Store → Retrieve → Answer
```

## Используемые библиотеки и инструменты

### Document Processing
```python
# Парсинг документов
from docling import DocumentParser  # v2.15.0+
from unstructured import partition  # v0.16.25+

# Обработка различных форматов
parsers = {
    "pdf": PDFParser(),
    "docx": DocxParser(),
    "html": HTMLParser(),
    "markdown": MarkdownParser()
}
```

### LLM Integration
```python
from langchain_core.prompts import PromptTemplate
from langchain_litellm import ChatLiteLLM

# Создание промпта
SUMMARY_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["document"],
    template=SUMMARY_PROMPT
)

# Использование с LLM
async def summarize_document(document_text: str) -> str:
    llm = ChatLiteLLM(model="gpt-4o")  # Long context LLM
    chain = SUMMARY_PROMPT_TEMPLATE | llm
    summary = await chain.ainvoke({"document": document_text})
    return summary.content
```

### Storage
```python
# Хранение в PostgreSQL
from sqlalchemy import Column, Text
from pgvector.sqlalchemy import Vector

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    content = Column(Text)  # Оригинальный текст
    summary = Column(Text)  # Сгенерированное саммари
    summary_embedding = Column(Vector(1024))  # Векторное представление
```

## Пример использования

### Входные данные
```python
document_text = """
Quantum Computing: A Revolutionary Paradigm

Quantum computing represents a fundamental shift in computational capability,
leveraging quantum mechanical phenomena such as superposition and entanglement
to process information in ways impossible for classical computers.

Unlike classical bits that exist in states of 0 or 1, quantum bits (qubits)
can exist in superposition - simultaneously representing both states. This
property, combined with entanglement, enables quantum computers to explore
vast solution spaces exponentially faster than classical approaches for
certain problem classes.

Current applications focus on:
- Cryptographic systems and security
- Drug discovery and molecular simulation
- Optimization problems in logistics
- Machine learning and AI enhancement

Challenges remain in qubit stability, error correction, and scaling to
commercially viable systems.
"""
```

### Выходные данные
```markdown
# Quantum Computing Overview

## Core Concept
Quantum computing represents a fundamental shift in computational capability,
leveraging quantum mechanical phenomena such as superposition and entanglement
to process information in ways impossible for classical computers.

## Technical Foundation
Unlike classical bits that exist in binary states (0 or 1), quantum bits (qubits)
utilize superposition to simultaneously represent both states. When combined with
quantum entanglement, this property enables quantum computers to explore vast
solution spaces exponentially faster than classical approaches for specific
problem classes.

## Current Applications
The technology currently focuses on four primary domains:

- **Cryptographic Systems**: Enhancing security protocols and encryption methods
- **Drug Discovery**: Molecular simulation for pharmaceutical development
- **Optimization**: Solving complex logistics and routing problems
- **Machine Learning**: AI capability enhancement and acceleration

## Implementation Challenges
Several technical obstacles remain before achieving commercially viable quantum
computing systems:
- Maintaining qubit stability and coherence
- Developing effective quantum error correction mechanisms
- Scaling systems to sufficient qubit counts for practical applications
```

## Конфигурация LLM

### Рекомендуемые модели
- **Long Context LLM**: GPT-4o, Claude 3.5 Sonnet, Gemini 2.5 Pro
- **Требования**: Длинное контекстное окно (128K+ токенов)
- **Reason**: Документы могут быть очень большими

### Параметры генерации
```python
llm_config = {
    "model": "gpt-4o",
    "temperature": 0.3,  # Низкая температура для точности
    "max_tokens": 4096,  # Достаточно для детального саммари
    "top_p": 0.95,
}
```

## Метрики качества

### Критерии оценки
1. **Factual Accuracy** (98%+) - Соответствие оригиналу
2. **Comprehensiveness** (90%+) - Покрытие ключевых тем
3. **Objectivity** (95%+) - Отсутствие интерпретации
4. **Readability** (85%+) - Четкость структуры
5. **Compression Ratio** (3:1 to 10:1) - Оптимальное сжатие

## См. также

- [Q&A с цитированием](../system-prompts/qna-with-citations.md) - Использует саммари
- [Библиотеки и инструменты](../libraries-and-tools.md) - Document processing stack
- [Цепочки преобразований](../transformation-chains.md) - Document indexing pipeline
