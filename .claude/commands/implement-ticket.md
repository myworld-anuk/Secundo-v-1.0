---
description: Implement a Jira ticket end to end - reuse check, implementation, self-review with tests, and a commit that documents everything
argument-hint: <KAN-ticket-key> [--comment]
---

Implement Jira ticket **$ARGUMENTS** in this repo (Musical Notation OMR). Follow the steps in order and do not skip any. Do not push, and do not touch `.venv/`.

Jira site: https://anaghakrishna.atlassian.net (cloudId `9ca37fed-3ce6-4bb5-bffe-afb2ee9e1c20`, project `KAN`).

## Step 0: Understand the ticket

1. Read the ticket with the Atlassian MCP (`getJiraIssue`, include issue links). Note its summary, description, "Done when" criteria, parent epic, and **blocked-by** links.
2. For each blocker, check whether the work exists in the codebase (code, files, or merged commits). If a blocker is clearly missing, stop and tell the user which one and why it matters. Do not stub it out silently.
3. Confirm you are not on `main`. If you are, create a branch named `<ticket-key>-<short-slug>`.
4. Read `PLAN.md` and `README.md` if present, to match design principles (for example: the CNN classifies *what* a symbol is, pitch is computed from geometry; everything is measured relative to the staff space).

## Step 1: Reuse and duplication check (before writing any code)

Search the codebase for logic the ticket overlaps with:

- Grep and read for the key nouns and verbs in the ticket (function names, file types, algorithms). Check `omr_server.py`, `single_note_cnn.py`, `generate_half_notes.py`, `Deskewing/`, and `src/` as relevant.
- Look for: existing functions to reuse, near-duplicates that should be extended instead of copied, code that will need refactoring to be shared, and conventions (naming, CLI style, paths, constants like image size).
- Check whether a teammate's open or recent work already covers part of it (`git log --oneline -20`, `git branch -a`, `git diff main --stat`).

Write a short **Reuse report** (keep it in your working notes, it goes into the final description):

- Reusable as is: file:function, what it gives us
- Extend or refactor: file:function, why
- Duplicate risk: what NOT to rewrite
- Genuinely new: what has to be built

## Step 2: Implement

- Build only what the ticket asks for. Base the design on the Reuse report: call or extend existing code, do not copy it.
- Match surrounding style (naming, comment density, argparse/CLI conventions, `pathlib` use).
- Keep changes scoped. Do not reformat unrelated code or change pinned dependencies unless the ticket requires it. If it does, say so in the description.
- Make functions importable (no work at import time) so they can be tested.
- Satisfy every "Done when" line in the ticket. Where a criterion can't be met (missing data, needs real sample pages, needs the user's decision), do not fake it. Record it as **Not done** with the reason.

## Step 3: Self-review and tests

1. **Syntax**: run `python -m py_compile` on every Python file you changed. Run `ruff check` or `flake8` if installed. For TypeScript changes, run `npx tsc --noEmit -p tsconfig.app.json` if feasible.
2. **Logic review**: re-read your diff (`git diff`) as a reviewer. Check off-by-one errors, coordinate order (x/y vs row/col), dtype and value range (0-1 vs 0-255, inverted binarization), empty or degenerate input, division by zero, and paths that assume the working directory.
3. **Make up test cases** under `tests/` (pytest, one `test_<module>.py` per module). Include:
   - a normal case, built from a small synthetic image or XML snippet created in the test (no large fixtures);
   - at least two edge cases (empty image, one staff line missing, rotated input, wrong dtype, unknown label...);
   - at least one regression check that existing behaviour still works (for example the single-note path).
4. Run the tests (`.venv/bin/python -m pytest tests -q`, or `python -m pytest` if no venv). If TensorFlow can't load the saved models in this environment, say so and mark model-dependent tests skipped with a reason, rather than deleting them.
5. Fix what the review and tests find, then re-run. Repeat until clean. Record anything you could not resolve.

## Step 4: Description and commit

Write the description in this format. It is the commit message body, so keep it factual and complete:

```
<TICKET-KEY> (OMR-n): <ticket summary>

Summary
<2-4 sentences: what was built and why it matters to the pipeline>

Reuse
- Reused: <file:function ...>
- Extended: <...>
- Avoided duplicating: <...>

Changes
- <file>: <what changed>

Tests
- <test file>: <what it covers>; <passed/failed/skipped counts>
- Checks run: <py_compile / ruff / tsc, results>

Done when
- [x] <criterion met, with how it was verified>
- [ ] <criterion not met, with reason>

Not done / deviations
- <anything skipped, simplified, assumed, or left for another ticket; "None" if none>

Risks and follow-ups
- <known limits, affected downstream tickets (blocks: KAN-..), suggested next steps>

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
```

Then:

1. Stage only the files you changed or created (`git add <paths>`). Never `git add -A`. Never stage `.venv/`, `__pycache__/`, model binaries, or large generated data.
2. Show the user `git status --short` and the full commit message, then commit with that message (use `git commit -F <file>` with the message written to the scratchpad directory).
3. Do **not** push, open a PR, or change the Jira ticket status unless the user asked. If `--comment` was passed in the arguments, post the same description to the Jira ticket as a comment, and say that you did.
4. Finish with a short report: what was done, what was not done, test results, and the commit hash.

## Rules

- Honesty over polish: if a test fails or a step was skipped, say so plainly in both the report and the commit message.
- If the ticket is ambiguous in a way that changes the design (for example OMR-16's detector choice), ask the user one precise question before implementing.
- If a ticket needs inputs only the user has (sample pages, ground-truth XML, Jira decisions), build the scaffolding and list the missing inputs under **Not done**.
