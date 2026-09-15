You are finishing a Python project called **ModelIO Lab** that I have already placed in the current directory. It compares LangChain's Model I/O layer (prompt templates, chat models, output parsers) against a native SDK baseline on one task: turning a product review into a validated `ReviewAnalysis` Pydantic object. Read `README.md` first, then every file under `modelio_lab/`, `app.py`, `cli.py`, `tests/` and `data/` before changing anything.

## Goal
Get the project working end to end, verified, committed, and ready to push to GitHub as `modelio-lab`. Keep the existing design and scope. Do not add new features, frameworks or files beyond what is listed below.

## Steps

1. **Check the layout.** Confirm this tree exists. If a file is missing or was saved in the wrong folder during download, move it into place (do not rewrite it from scratch unless it is truly missing):
   ```
   app.py  cli.py  requirements.txt  .env.example  .gitignore  LICENSE  README.md  GITHUB_META.md
   modelio_lab/__init__.py config.py schemas.py prompts.py models.py parsers.py debug.py chains.py native_baseline.py benchmark.py
   data/sample_reviews.json  data/malformed_outputs.json
   tests/__init__.py test_parsers.py test_prompts.py test_pipelines_offline.py
   ```
   Make sure `.gitignore` and `.env.example` kept their leading dot.

2. **Environment.** Create `.venv` with Python 3.10+, activate it, and run `pip install -r requirements.txt`. If a provider package fails to install, report which one and continue; the core must still work.

3. **Offline verification (no API keys).** Run and fix until all pass:
   - `pytest -q` (all tests green)
   - `python cli.py stress` (strict parser should score lower than robust)
   - `python cli.py analyze --mode strict` and `python cli.py analyze --mode robust`
   - `python cli.py analyze --native`
   - `python cli.py footprint`
   If an import or API changed in the installed LangChain version (for example `init_chat_model`, `FakeListChatModel`, `FewShotChatMessagePromptTemplate`, `with_structured_output`, callback signatures), fix it using the installed package's actual API and check the source in `.venv` rather than guessing. Keep fixes minimal.

4. **Streamlit check.** Write a temporary script using `streamlit.testing.v1.AppTest` that loads `app.py`, runs it, and asserts no exception is raised on first render and after clicking the "Run both pipelines", "Run stress test" and "Measure" buttons with the `fake` provider. Fix any errors it surfaces (such as deprecated arguments like `use_container_width`, widget key clashes, or nested column limits). Delete the temporary script afterwards, or move it into `tests/test_app.py` if it runs in under 10 seconds.

5. **Live providers (only if keys exist).** If `.env` contains any of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, run `python cli.py swap --providers fake <available providers>` and `python cli.py analyze --provider <each> --mode structured`. If a default model name in `config.py` is rejected by the provider, update it to a current cheap model for that provider. Never print or commit key values. If no keys exist, skip this step and say so.

6. **Code quality.** Run `python -m compileall -q .` and, if available, `ruff check .` (install ruff in the venv if needed). Fix real problems only; do not reformat the whole codebase.

7. **README.** Update the "Sample result" line with the actual numbers from `python cli.py stress`, and update the LOC and dependency numbers if you mention them. Keep the rest of the README as is.

8. **Git.** Initialise a repo if none exists, confirm `.env` is ignored, and make one commit: `feat: ModelIO Lab - LangChain Model I/O vs native SDK stress test`. Print the exact commands I need to create the GitHub repo and push, using the name, description and topics from `GITHUB_META.md` (with `gh repo create` if the GitHub CLI is installed, otherwise manual steps).

## Done means
- `pytest -q` passes, and the CLI commands in step 3 exit without tracebacks.
- `streamlit run app.py` starts and all five tabs render with the `fake` provider.
- A single clean commit exists and `.env` is not in it.

Finish with a short report: what you fixed and why, test results, the stress-test numbers, any provider that could not be tested, and the push commands.
