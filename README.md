---
title: Chrix Persona
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# Chrix Tech — Persona AI v4

A RAG-powered personal AI chatbot that speaks as Christian Agyapong (Chrix Tech).
Built with LangChain, Groq (Qwen3.6 27B), ChromaDB, and Gradio.

---

## Project Structure

```
chrix-persona/
├── app.py               ← Main entry point (run this to launch)
├── requirements.txt     ← Python dependencies
├── .env.example         ← Environment variable template
├── Dockerfile           ← For Fly.io / Docker deployment
├── fly.toml             ← Fly.io config
├── .dockerignore        ← Docker build exclusions
├── .gitignore           ← Git exclusions
├── notebooks/
│   └── persona.ipynb    ← Original development notebook
└── chrix_db_v4/         ← ChromaDB vector store (auto-generated on first run)
```

---

## Quickstart (local)

```bash
# 1. Clone / download this folder
# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq API key
export GROQ_API_KEY=your_key_here   # Windows: set GROQ_API_KEY=your_key_here

# 5. Run
python app.py
# → Open http://localhost:7860
```

---

## Deploy to Hugging Face Spaces (easiest)

1. Create a new Space at https://huggingface.co/new-space
2. Choose **Gradio** as the SDK
3. Upload all files in this folder (except `notebooks/` if you want)
4. Go to **Settings → Repository secrets** and add:
   - `GROQ_API_KEY` = your Groq key
5. The Space will auto-build and launch.

> The `chrix_db_v4/` folder will be regenerated automatically on first startup — no need to upload it.

---

## Deploy to Fly.io

```bash
# Install flyctl if not already installed
# https://fly.io/docs/hands-on/install-flyctl/

fly auth login
fly launch          # follow prompts; use fly.toml config provided
fly secrets set GROQ_API_KEY=your_key_here
fly deploy
```

---

## Environment Variables

| Variable       | Required | Description              |
|----------------|----------|--------------------------|
| `GROQ_API_KEY` | Yes      | Your Groq API key        |
| `PORT`         | No       | Server port (default 7860) |

---

## Notes

- The ChromaDB vector store (`chrix_db_v4/`) is built from the bio on first startup.
  It is excluded from Git via `.gitignore` and will regenerate automatically.
- The model used is `qwen/qwen3.6-27b` via Groq — fast and free-tier friendly.
- All Groq API calls are made server-side; the key is never exposed to the browser.
