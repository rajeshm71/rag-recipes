## What changed

<!-- One or two sentences: what does this PR do? -->

## Why

<!-- What problem does this solve, or what value does it add? Link an issue if one exists. -->

## How this was tested

<!-- Which commands did you run locally? Paste output if useful. -->

- [ ] `ruff check .`
- [ ] `python scripts/lint_notebooks.py`
- [ ] `pytest tests/`
- [ ] `RAG_RECIPES_LLM=mock papermill notebooks/<affected>.ipynb notebooks/<affected>.ipynb --cwd notebooks`

## Checklist

- [ ] No new dependency added without prior discussion in an issue (≤12 top-level runtime deps)
- [ ] Every LLM/embedder call uses a pinned dated snapshot, not a bare alias
- [ ] If this adds/changes a pattern notebook: all 8 mandatory sections present, in order,
      including a real "Where this pattern FAILS" section with at least 2 analyzed failures
- [ ] The held-constant generation prompt (`prompts/generation_prompt.txt`) is unchanged, or this
      PR was pre-discussed in an issue if it needs to change
- [ ] New/changed metrics report a 95% bootstrap CI where applicable
- [ ] No `Co-Authored-By` or AI-tool attribution trailer on commits
- [ ] This PR does one logical thing (a new pattern and a bug fix are two PRs, not one)

## One logical change per PR

A new pattern and a bug fix are two PRs, not one. If this PR grew to cover more than one thing,
consider splitting it before requesting review.
