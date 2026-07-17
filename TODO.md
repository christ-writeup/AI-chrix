# TODO — Chrix Tech Persona AI

## Completed
- [x] Reduce LLM verbosity by lowering `max_tokens`.
- [x] Harden `clean_reply()` to remove `<think>` blocks (including truncated cases).
- [x] Enforce a strict 1–5 sentence output limit in `clean_reply()`.
- [x] Update `SYSTEM_INSTRUCTIONS` to explicitly forbid emitting reasoning/meta blocks.
- [x] Validate syntax via `python -m py_compile app.py`.
- [x] Add deterministic portfolio + GitHub links to the knowledge base (BM25) in `app.py`.
- [x] Make greeting-only messages short and prevent long re-intros.
- [x] Make the initial UI greeting short (update `templates/index.html` + `static/app.js`).

## Next
- [ ] Quick chat tests:
  - ask: "hey" repeatedly
  - ask: "Are you available for freelance?"
  - ask: "What projects"
- [ ] Ensure suggestions don’t steer back into repetitive prompts.
- [ ] If needed: add “anti-repetition” using last AI message similarity (string-level) in `clean_reply()`.

