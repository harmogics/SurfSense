# Генерация подкастов (Podcast Generation)

## Описание

Креативный промпт для преобразования текстового контента в живой, естественный диалог подкаста между двумя ведущими. Промпт создает ~6-минутную беседу (~1000 слов) с динамичным взаимодействием, естественными репликами и аутентичной химией между спикерами.

## Категория
**Creative Prompt** - генерация креативного диалогового контента

## Расположение в кодовой базе
`surfsense_backend/app/agents/podcaster/prompts.py::get_podcast_generation_prompt()`

## Текст промпта (ключевые элементы)

```python
def get_podcast_generation_prompt(user_prompt: str | None = None):
    return f"""
Today's date: {datetime.datetime.now().strftime("%Y-%m-%d")}
<podcast_generation_system>
You are a master podcast scriptwriter, adept at transforming diverse input
content into a lively, engaging, and natural-sounding conversation between
two distinct podcast hosts.

{f'<user_instruction>{user_prompt}</user_instruction>' if user_prompt else ""}

<output_format>
{{
  "podcast_transcripts": [
    {{"speaker_id": 0, "dialog": "Speaker 0 dialog here"}},
    {{"speaker_id": 1, "dialog": "Speaker 1 dialog here"}}
  ]
}}
</output_format>

<guidelines>
1. **Establish Distinct & Consistent Host Personas:**
   - Speaker 0 (Lead Host): Drives conversation, introduces segments, poses key questions
   - Speaker 1 (Co-Host/Expert): Offers deeper insights, alternative viewpoints, asks clarifying questions
   - Consistency is Key: Each speaker maintains distinct voice, vocabulary, sentence structure

2. **Craft Natural & Dynamic Dialogue:**
   - Emulate Real Conversation: Use contractions, interjections, discourse markers
   - Foster Interaction & Chemistry: Genuine reactions, build on points, ask follow-ups
   - Vary Rhythm & Pace: Mix short punchy lines with longer explanatory ones
   - Inject Personality & Relatability: Humor, surprise, curiosity, brief personal reflections

3. **Structure for Flow and Listener Engagement:**
   - Natural Beginning: Flow naturally after introduction (added manually)
   - Logical Progression & Signposting: Clear transitions between ideas
   - Meaningful Conclusion: Summarize takeaways, final thought, teaser

4. **Integrate Source Content Seamlessly & Accurately:**
   - Translate, Don't Recite: Rephrase into conversational language
   - Explain & Contextualize: Use analogies, examples, storytelling
   - Weave Information Naturally: Facts within dialogue, not standalone blocks
   - Balance Depth & Accessibility: Informative but clear for general audience

5. **Length & Pacing:**
   - Six-Minute Duration: ~1000 words total (150 words per minute)
   - Concise Speaking Turns: Brief and focused, natural back-and-forth
   - Essential Content Only: Quality over quantity
</guidelines>
</podcast_generation_system>
"""
```

## Параметры

| Параметр | Тип | Описание |
|----------|-----|----------|
| `user_prompt` | `str \| None` | Дополнительные инструкции пользователя (стиль, тон, акценты) |
| `source_content` | `str` | Текстовый контент для преобразования в подкаст |

## Роль в цепочке преобразований

### Позиция в Pipeline
**Первый узел** в цепочке Podcaster Agent

```
Source Content → [GENERATE_TRANSCRIPT] → create_audio_segments → merge_audio → Output
```

### Входящие данные
1. **Source Content** - документы, чат-история, статьи
2. **User Instructions** - кастомизация стиля (опционально)
3. **Chat History** - контекст беседы с пользователем

### Исходящие данные
1. **JSON транскрипт** - структурированный диалог
2. **Speaker-separated segments** - для TTS обработки
3. **Metadata** - длительность, количество слов

## Связи с другими промптами

### Зависит от
- **[Document Summarization](../task-prompts/document-summarization.md)** - может использовать саммари как input
- **[Q&A with Citations](../system-prompts/qna-with-citations.md)** - может преобразовывать Q&A в подкаст

### Предоставляет вход для
- **TTS Pipeline** (Kokoro) - синтез речи
- **Audio Merging** - объединение аудио сегментов

## Используемые библиотеки и инструменты

### LLM Integration
```python
from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel
import json

class PodcastDialog(BaseModel):
    speaker_id: int
    dialog: str

class PodcastTranscript(BaseModel):
    podcast_transcripts: list[PodcastDialog]

async def generate_podcast_transcript(
    source_content: str,
    user_instructions: str | None = None
) -> PodcastTranscript:
    """Generates podcast transcript from source content."""

    llm = ChatLiteLLM(
        model="gpt-4o",  # Strategic LLM for creative tasks
        temperature=0.8   # Higher temperature for creativity
    )

    prompt = get_podcast_generation_prompt(user_instructions)
    response = await llm.ainvoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"<source_content>{source_content}</source_content>"}
    ])

    # Parse and validate JSON
    transcript_data = json.loads(response.content)
    return PodcastTranscript(**transcript_data)
```

### Text-to-Speech (Kokoro)
```python
from kokoro import KokoroTTS
import soundfile as sf
import asyncio

class PodcastAudioGenerator:
    def __init__(self):
        self.tts = KokoroTTS()
        # Разные голоса для разных спикеров
        self.voices = {
            0: "af_sky",    # Speaker 0: Female voice
            1: "am_adam"    # Speaker 1: Male voice
        }

    async def generate_audio_segment(
        self,
        dialog: PodcastDialog
    ) -> bytes:
        """Generate audio for single dialog segment."""
        voice = self.voices[dialog.speaker_id]

        audio_data = await self.tts.synthesize(
            text=dialog.dialog,
            voice=voice,
            speed=1.0,
            emotion="conversational"
        )

        return audio_data

    async def generate_full_podcast(
        self,
        transcript: PodcastTranscript,
        output_path: str
    ):
        """Generate complete podcast audio."""
        audio_segments = []

        for dialog in transcript.podcast_transcripts:
            segment = await self.generate_audio_segment(dialog)
            audio_segments.append(segment)

        # Merge segments with pauses
        merged_audio = self.merge_with_pauses(audio_segments)

        # Save to file
        sf.write(output_path, merged_audio, samplerate=24000)
```

### Audio Processing
```python
from pydub import AudioSegment
import numpy as np

def merge_with_pauses(
    segments: list[bytes],
    pause_duration_ms: int = 300
) -> np.ndarray:
    """Merge audio segments with natural pauses."""
    merged = AudioSegment.empty()
    pause = AudioSegment.silent(duration=pause_duration_ms)

    for i, segment_bytes in enumerate(segments):
        segment = AudioSegment.from_wav(io.BytesIO(segment_bytes))
        merged += segment

        # Add pause between speakers (not after last segment)
        if i < len(segments) - 1:
            merged += pause

    return np.array(merged.get_array_of_samples())
```

## Пример использования

### Входные данные
```python
source_content = """
Quantum computing uses quantum bits or qubits which can exist in multiple
states simultaneously due to superposition. This enables quantum computers
to solve certain problems exponentially faster than classical computers.
"""

user_instructions = "Make it very engaging and accessible to non-technical audience"
```

### Выходные данные
```json
{
  "podcast_transcripts": [
    {
      "speaker_id": 0,
      "dialog": "Today we're diving into the mind-bending world of quantum computing. You know, this is a topic I've been excited to cover for weeks."
    },
    {
      "speaker_id": 1,
      "dialog": "Same here! And I know our listeners have been asking for it. But I have to admit, the concept of quantum computing makes my head spin a little. Can we start with the basics?"
    },
    {
      "speaker_id": 0,
      "dialog": "Absolutely. So regular computers use bits, right? Little on-off switches that are either 1 or 0. But quantum computers use something called qubits, and this is where it gets fascinating."
    },
    {
      "speaker_id": 1,
      "dialog": "Wait, what makes qubits so special compared to regular bits?"
    },
    {
      "speaker_id": 0,
      "dialog": "The magic is in something called superposition. These qubits can exist in multiple states at the same time, not just 1 or 0."
    },
    {
      "speaker_id": 1,
      "dialog": "That sounds impossible! How would you even picture that?"
    },
    {
      "speaker_id": 0,
      "dialog": "Think of it like a coin spinning in the air. Before it lands, is it heads or tails?"
    },
    {
      "speaker_id": 1,
      "dialog": "Well, it's... neither? Or I guess both, until it lands? Oh, I think I see where you're going with this."
    }
  ]
}
```

## Конфигурация LLM

### Рекомендуемые модели
- **Strategic LLM**: GPT-4o, Claude 3.5 Sonnet, Gemini 2.5 Pro
- **Требования**: Креативность, понимание диалога
- **Контекстное окно**: 32K-128K токенов

### Параметры генерации
```python
llm_config = {
    "model": "gpt-4o",
    "temperature": 0.85,  # Высокая креативность
    "max_tokens": 4096,
    "response_format": {"type": "json_object"},
    "top_p": 0.95,
}
```

## Метрики качества

### Критерии оценки
1. **Naturalness** (90%+) - Естественность диалога
2. **Engagement** (85%+) - Увлекательность контента
3. **Information Accuracy** (95%+) - Точность информации из source
4. **Speaker Distinction** (90%+) - Различимость персонажей
5. **Audio Quality** (88%+) - Качество финального аудио (после TTS)
6. **Listen-through Rate** (70%+) - % слушателей до конца

## См. также

- [Document Summarization](../task-prompts/document-summarization.md) - Источник контента
- [Библиотеки и инструменты](../libraries-and-tools.md) - TTS stack (Kokoro)
- [Цепочки преобразований](../transformation-chains.md) - Podcaster Agent pipeline
