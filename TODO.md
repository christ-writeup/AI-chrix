# TODO — Chrix Tech Persona AI

## Completed
- [x] Reduce LLM verbosity by lowering `max_tokens`.
- [x] Harden `clean_reply()` to remove `<think>` blocks (including truncated cases).
- [x] Enforce a strict 1–5 sentence output limit in `clean_reply()`.
- [x] Update `SYSTEM_INSTRUCTIONS` to explicitly forbid emitting reasoning/meta blocks.
- [x] Validate syntax via `python -m py_compile app.py`.
- [x] Add deterministic portfolio + GitHub links to the knowledge base (BM25) in `app.py`.
- [x] Make greeting-only messages short and prevent long re-intros.
- [x] Update the initial UI greeting to be short (update `templates/index.html` + `static/app.js`).
- [x] Anti-repetition for greeting/general replies (avoid repeating last AI reply; basic similarity checks).

## Next
- [x] Expand greeting variation pool (more than 3 templates).
- [x] Suggestion anti-repetition (don’t keep showing the same chips after small changes).

- [ ] Quick chat tests after edits:
  - ask: "hey" repeatedly
  - ask: "Are you available for freelance?"
  - ask: "What projects"

