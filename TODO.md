# TODO — Chrix Tech Persona AI

## Completed
- [x] Reduce LLM verbosity by lowering `max_tokens`.
- [x] Harden `clean_reply()` to remove `<think>` blocks (including truncated cases).
- [x] Enforce a strict 1–5 sentence output limit in `clean_reply()`.
- [x] Update `SYSTEM_INSTRUCTIONS` to explicitly forbid emitting reasoning/meta blocks.
- [x] Validate syntax via `python -m py_compile app.py`.

## Next
- [x] Add deterministic portfolio + GitHub links to the knowledge base (BM25) in `app.py`.
- [ ] Ensure the assistant returns those exact URLs when asked about portfolio/GitHub.
- [ ] Quick chat tests (ask: "portfolio" and "GitHub link").
- [ ] Update KB with SHS school name + expected completion/graduation date (and university expected completion date if you want it in KB). (Missing currently)



