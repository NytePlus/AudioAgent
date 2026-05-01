# Qwen3-ASR-Flash API Tool

API-based speech recognition using DashScope `qwen3-asr-flash`. This tool does not load a local model or require a GPU.

## Tool

### `transcribe_qwen3_asr_flash`

Transcribes speech from an audio file or public audio URL.

Inputs:

- `audio_path`: local audio path, `audio_id` such as `audio_0`, or a public HTTP(S) audio URL.
- `context`: optional recognition-biasing context, such as OCR text, proper nouns, domain terms, slide text, or reference vocabulary.
- `language`: optional language hint, such as `zh`, `en`, `ja`, `ko`, or `auto`.
- `enable_itn`: optional inverse text normalization for Chinese/English.
- `stream`: optional streaming response collection.

Output is JSON text containing:

- `text`: transcript
- `model`: API model name
- `audio_path`: input audio source
- `language`, `enable_itn`, `context_used`
- `annotations` and `usage`, when returned by the API

## Setup

```bash
cd audio_agent/tools/catalog/qwen3_asr_flash
./setup.sh
export DASHSCOPE_API_KEY="sk-xxx"
./test_env.sh
```

## Example

```python
from audio_agent.core.schemas import ToolCallRequest

request = ToolCallRequest(
    tool_name="transcribe_qwen3_asr_flash",
    args={
        "audio_path": "/path/to/audio.wav",
        "context": "Katharina Morlang, DSJ, 4th International Conference, I COACH KIDS",
        "language": "en",
        "enable_itn": False,
    },
)
```

For local files, the tool sends a Base64 data URL to the API and enforces the default `10 MB` local file limit. Use a public HTTP(S) URL for larger audio.
