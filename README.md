# ModelIO Lab

Stress-testing LangChain's Model I/O layer (prompts, chat models, output parsers) against a plain SDK baseline.

Developers praise LangChain's unified interface but keep raising the same five complaints: too much ceremony, painful debugging, provider swaps that don't really work, fragile output parsing, and dependency bloat. This project turns each complaint into something you can run and measure, and shows a practical fix inside LangChain for each one.

The task is simple on purpose: turn a free-text product review into a validated `ReviewAnalysis` object (sentiment, 1–5 rating, pros, cons, summary).

## What it tests

| Complaint | Where | What you see |
|---|---|---|
| Over-abstraction | `benchmark.loc_comparison` | Lines of code for the LangChain pipeline vs. `native_baseline.py` |
| Debugging | `debug.py` | A callback trace of every request and raw response, and errors reduced to one line with the root cause |
| Provider swap | `prompts.py`, `benchmark.model_swap` | Same code across providers; small models get a stricter few-shot prompt |
| Fragile parsing | `parsers.py`, `benchmark.parser_stress_test` | Stock `PydanticOutputParser` vs. a robust parser on 12 realistic messy outputs, plus LLM repair retries |
| Dependency bloat | `benchmark.dependency_footprint` | Transitive dependency count for `langchain` vs. provider SDKs |

Sample result from the offline stress test: the stock parser accepts 4 of 12 outputs, the robust parser 7 of 12 (the other 5 are genuinely invalid and should fail).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # optional: add API keys
pytest -q
streamlit run app.py
```

No API keys? Everything runs with the `fake` provider, which returns deliberately messy output (markdown fences, chatty prose, trailing commas) so the parsers have something to fail on.

## CLI

```bash
python cli.py analyze "Great sound, weak battery" --provider fake --mode strict
python cli.py analyze --provider openai --mode structured --trace
python cli.py analyze --native --provider anthropic
python cli.py stress
python cli.py swap --providers fake openai anthropic groq
python cli.py footprint
```

## Parser modes

- `strict`: LangChain's stock `PydanticOutputParser`, no recovery.
- `robust`: strips fences and prose, removes trailing commas, validates with Pydantic, and on failure asks the model to repair its own output (configurable retries).
- `structured`: `model.with_structured_output(ReviewAnalysis)` using native tool calling. Needs OpenAI, Anthropic, Groq or Ollama. It uses its own prompt with no format instructions and no plain-JSON few-shot examples, because those pushed Groq's model to answer in text instead of calling the forced tool.

## Project structure

```
modelio-lab/
├── app.py                     Streamlit UI (5 tabs)
├── cli.py                     Command-line interface
├── modelio_lab/
│   ├── config.py              Providers, default models, key detection
│   ├── schemas.py             ReviewAnalysis model and RunResult record
│   ├── prompts.py             Provider-aware ChatPromptTemplates and the repair prompt
│   ├── models.py              init_chat_model factory and offline fake model
│   ├── parsers.py             Strict and robust output parsers
│   ├── debug.py               Trace callback and readable errors
│   ├── chains.py              prompt | model | parser pipeline with retries
│   ├── native_baseline.py     Same task with the OpenAI / Anthropic SDKs
│   └── benchmark.py           Stress test, model swap, footprint, LOC
├── data/
│   ├── sample_reviews.json
│   └── malformed_outputs.json
└── tests/
```

## Takeaways

LangChain's Model I/O layer earns its keep when you really do run several providers, want tracing callbacks, or need `with_structured_output` across vendors. For a single provider and a simple prompt, the native SDK is shorter and easier to debug. Most parsing pain disappears once you either use tool-calling structured output or add a cleanup-and-repair step, which is a few dozen lines either way.

## License

MIT
