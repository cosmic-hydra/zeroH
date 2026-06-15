# zeroH for VS Code — Zero Hallucination Memory

Bring [**zeroH**](https://github.com/cosmic-hydra/zeroH)'s grounded, durable
memory layer into your editor. Remember facts, ingest documents, **fact-check
text against memory**, and get **cited, abstaining** answers — so nothing
reaches you unless it can be grounded in something you actually stored.

This extension is a thin UI over the pure-Python `zeroh` package. zeroH never
invents text: it surrounds retrieval, grounding, and verification around the
memory you build (and, optionally, around your own LLM).

## Requirements

The extension shells out to a Python interpreter that has `zeroh` installed:

```bash
pip install zeroh
```

By default it calls `python` on your `PATH`. Point `zeroh.pythonPath` at a
specific interpreter (e.g. a virtualenv) if needed.

## Commands

Open the Command Palette (`Ctrl/Cmd+Shift+P`) and type **zeroH**:

| Command | What it does |
| --- | --- |
| **Remember a Fact** | Store a single fact durably (prefilled with your selection). |
| **Ingest Active File** | Chunk the current file and store it as memory. |
| **Verify Selection Against Memory** | Fact-check selected text; unsupported claims are stripped/abstained (no LLM needed). |
| **Recall Memories** | Retrieve the most relevant stored memories for a query. |
| **Answer From Memory (no LLM)** | Compose an extractive, cited answer purely from memory. |
| **Ask (grounded completion via your LLM)** | Retrieve → augment → call your configured LLM → verify. |

**Verify**, **Ingest**, **Recall**, and **Answer From Memory** are fully
offline and require no LLM or API key. Only **Ask** needs an LLM.

## Settings

| Setting | Default | Description |
| --- | --- | --- |
| `zeroh.pythonPath` | `python` | Interpreter that has `zeroh` installed. |
| `zeroh.databasePath` | _(empty)_ | Durable SQLite store path. Empty → extension global storage. |
| `zeroh.topK` | `5` | Memories retrieved as context per query. |
| `zeroh.llm.provider` | `openai` | `openai` (any OpenAI-compatible endpoint) or `ollama`. |
| `zeroh.llm.model` | `gpt-4o-mini` | Model name for the provider. |
| `zeroh.llm.baseUrl` | _(empty)_ | Endpoint base URL (e.g. `http://localhost:11434/v1`). |
| `zeroh.llm.apiKey` | _(empty)_ | API key for cloud OpenAI-compatible endpoints. |

> Prefer a local model (Ollama/LM Studio/vLLM via `baseUrl`) if you would rather
> not store an API key in settings.

## How memory persists

Memories are stored in a durable SQLite database (WAL journaling). Corrections
*supersede* facts rather than deleting them, so history is retained for audit.
The store lives in the extension's global storage by default, or wherever you
point `zeroh.databasePath`.

## Building from source

```bash
cd vscode-extension
npm install
npm run compile      # tsc -> out/
npm run package      # produces a .vsix you can install or publish
```

## License

MIT — see [LICENSE](./LICENSE).
