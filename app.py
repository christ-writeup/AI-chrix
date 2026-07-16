# ============================================================
# Chrix Tech — Persona AI v4 (fixed)
# Flask backend for chat UI
# ============================================================

from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from flask import Flask, request, jsonify, render_template
import os
import re
import random
from dotenv import load_dotenv

load_dotenv()

# ─── Bio ────────────────────────────────────────────────────
personal_bio = """
FULL NAME: Christian Agyapong, known professionally as Chrix Tech.
LOCATION: Currently living and studying in Accra, Ghana.

EDUCATION:
I am a Computer Science student at the University of Ghana, Legon.
My major is Artificial Intelligence and Machine Learning.

TECH STACK AND SKILLS:
Programming languages: Python, JavaScript.
Frontend and mobile development: React (web), React Native (mobile apps).
AI and Machine Learning: Deep Learning, Machine Learning, Retrieval-Augmented Generation (RAG).

PROFESSIONAL EXPERIENCE & PROJECTS:
- I built AI/ML tools and RAG-powered automation projects.

CERTIFICATIONS:
- AWS Educate Introduction to Cloud 101: https://www.credly.com/badges/0b6a0d2c-3658-4a6a-aa58-ec2a1fb4e3cd
- Applied AI Lab: Deep Learning for Computer Vision: https://www.credly.com/badges/030a23b0-a459-475a-9f25-5939cefb1bf2/linked_in_profile
- Data Intelligence and Swarm Analytics Lab: https://credsverse.com/credentials/b8738510-8e42-4530-91b2-dfab4a40c1e3
- Udemy Prompt Engineering: https://www.udemy.com/certificate/UC-3c9b7a15-1e0d-4dfc-bf92-7f64d469c1bf/
"""

# ─── Portfolio / GitHub (deterministic links for retrieval) ─────────────────
# IMPORTANT: Put the exact URLs you want the assistant to provide.
portfolio_github_links = """
PORTFOLIO LINK:
- https://christian.dev

GITHUB LINK:
- https://github.com/chrix-tech

OPTIONAL (REPOS):
- https://github.com/chrix-tech/chrix-persona
"""


# NOTE: We use this block (in addition to bio + website_content.txt) so BM25
# retrieval can always find the correct portfolio/GitHub URLs when the user
# asks for them.


# ─── Vector Store (BM25 over chunks) ───────────────────────
def build_documents_from_bio(bio_text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(bio_text.strip())
    return [Document(page_content=c) for c in chunks]


docs = build_documents_from_bio(personal_bio)

website_docs = []
try:
    if os.path.exists("website_content.txt"):
        with open("website_content.txt", "r", encoding="utf-8") as f:
            website_docs = build_documents_from_bio(f.read())
except Exception:
    website_docs = []

all_docs = docs + website_docs + build_documents_from_bio(portfolio_github_links)

retriever = BM25Retriever.from_documents(all_docs)
retriever.k = 1  # latency optimization


# ─── LLM ─────────────────────────────────────────────────────
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    # Keep startup failure explicit (helps identify missing secrets).
    raise RuntimeError(
        "Missing GROQ_API_KEY in environment. "
        "Set it in your deployment secrets (e.g., fly secrets set GROQ_API_KEY=...)."
    )

# NOTE ON THE FIX:
# qwen/qwen3.6-27b is a reasoning model — it emits a <think>...</think>
# block before its real answer. Two changes here address the root cause
# instead of only patching it after the fact:
#
# 1. max_tokens raised from 280 -> 900. At 280 tokens, the model was
#    running out of budget *while still inside <think>...</think>*,
#    so the reasoning block never closed and the real reply was never
#    written at all.
# 2. `reasoning_format="hidden"` (Groq-specific param, passed via
#    model_kwargs) asks Groq's API to strip reasoning server-side so it
#    never appears in response.content in the first place. If your
#    installed langchain-groq version doesn't forward this kwarg,
#    the clean_reply() fallback fix below still protects you.
llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0.6,
    max_tokens=1800,
    # NOTE: Some langchain-groq versions validate these as top-level
    # parameters (not inside `model_kwargs`). To keep deployment stable
    # across environments, we avoid passing them via `model_kwargs`.
    # If your Groq/langchain version supports disabling reasoning,
    # you can re-add these as top-level args.
)


# ─── Intent Detection ────────────────────────────────────────
INTENT_MAP = {
    "introduction": ["introduce", "who are you", "tell me about yourself", "your name", "what do you do", "hey"],
    "skills": ["skills", "experience", "work", "job", "internship", "design", "coding", "programming", "stack", "technologies"],
    "projects": ["project", "projects", "portfolio", "built", "system", "platform", "app"],
    "origin": ["where are you from", "hometown", "where did you grow up", "childhood"],
    "education": [
        "studying",
        "study",
        "school",
        "university",
        "college",
        "degree",
        "major",
        "education",
        "background",
        "studies",
    ],
    "hobbies": ["hobbies", "free time", "leisure", "outside school"],
    "goals": ["goal", "dream", "ambition", "vision", "future"],
    "general": [],
}

GREETING_ONLY_RE = re.compile(r"^(hey|hi|hello|howdy)\b", re.IGNORECASE)

# If the user greets but doesn't ask anything, keep the response from
# sounding like a full introduction every time.
# This is intentionally simple and relies on prompt grounding.
GREETING_NO_QUESTION_RE = re.compile(r"^(hey|hi|hello|howdy)\b[\s!?.]*$", re.IGNORECASE)


INTENT_FOCUS = {
    "introduction": "Introduce yourself in a warm, natural way. Keep it short.",
    "skills": "Talk about skills and what you can do in a conversational way. Keep it short.",
    "projects": "Talk about projects or what you have built. Keep it short.",
    "origin": "If asked about where you are from, answer briefly. Keep it short.",
    "education": "Talk about education and studies briefly. Keep it short.",
    "hobbies": "Talk about hobbies briefly. Keep it short.",
    "goals": "Talk about goals briefly. Keep it short.",
    "general": "Respond naturally and directly. Keep it short.",
}

INTENT_SUGGESTIONS = {
    "introduction": ["What projects have you built?", "What is your tech stack?", "What are your goals?"],
    "skills": ["Tell me about your experience", "What tools do you use most?", "Can you build something for me?"],
    "projects": ["How does your project work?", "What stack did you use?", "What was the hardest part?"],
    "origin": ["How did your journey start?", "Where are you based now?", "What motivates you?"],
    "education": ["What are you studying right now?", "What excites you most about AI/ML?", "What have you learned?"],
    "hobbies": ["What music do you like?", "What do you do to relax?", "Do you play football?"],
    "goals": ["What are you building next?", "How do you plan to impact Africa?", "What's your long-term vision?"],
    "general": ["What projects are you working on?", "What's your core tech stack?", "Are you available for freelance?"],
}


# ─── Post-processing ──────────────────────────────────────────
PIDGIN_PATTERNS = [
    r"\bchale\b",
    r"\bherh\b",
    r"\babeg\b",
    r"\be be so\b",
    r"\bby God's grace\b",
    r"\bwe dey push\b",
    r"\byou feel me\b",
    r"\bnaa\b",
    r"\bmehn\b",
    r"\bwhat's popping\b",
    r"\bshoot the breeze\b",
]

BAD_OPENERS = [
    r"^(Chale[,!]?\s*)+",
    r"^(Herh[,!]?\s*)+",
    r"^What's good[,!]?\s*",
    r"^Certainly[,!]?\s*",
    r"^Absolutely[,!]?\s*",
    r"^Of course[,!]?\s*",
    r"^Sure[,!]?\s*",
    r"^Yes[,!]?\s*",
    r"^Yes,? I can[,!]?\s*",
    r"^Yes,? I can definitely[,!]?\s*",
    r"^I can definitely[,!]?\s*",
    r"^I can[,!]?\s*",
    r"^Hey there[,!]?\s*",
    r"^Great[,!]?\s*",
    r"^I'm listening[,!]?\s*",
    r"^I\u2019m listening[,!]?\s*",
    r"^I'm here whenever you're ready[,!\.]?\s*",
    r"^I\u2019m here whenever you're ready[,!\.]?\s*",
]


# Matches an opening reasoning tag whether or not it's ever closed.
REASONING_TAG_RE = re.compile(r"<\s*(think|thinking)\s*>", re.IGNORECASE)

FALLBACK_REPLY = "Let me try that again — could you ask once more?"


def clean_reply(text: str) -> str:
    original = text or ""

    # 1) Remove any reasoning blocks the model may emit.
    # Covers: <think>...</think>, <thinking>...</thinking>, and truncated variants.
    text = re.sub(
        r"<\s*(think|thinking)\s*>.*?<\s*/\s*(think|thinking)\s*>",
        "",
        original,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # If the model opened a reasoning tag but got truncated, drop everything from the tag onward.
    text = re.sub(
        r"<\s*(think|thinking)\s*>.*$",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text = text.strip()

    # FIX: previously this fell back to `original.strip()`, which could
    # re-expose an unclosed <think> block (i.e. the whole raw response)
    # whenever the model ran out of tokens mid-reasoning. Now we only
    # fall back to the original text if it does NOT itself contain a
    # reasoning tag. If it does, we use a safe canned reply instead.
    if not text:
        if REASONING_TAG_RE.search(original):
            text = FALLBACK_REPLY
        else:
            text = original.strip()

    # 2) Remove bad openers/pidgin only if they exist, but keep the rest intact.
    for pattern in BAD_OPENERS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # 2b) Remove common “template-ish” lead-ins that reduce human tone.
    # Keep these conservative: only strip very specific starting phrases.
    text = re.sub(r"(?i)^(i\s+specialize\s+in|i\s+specialise\s+in)\b\s*", "", text).strip()
    text = re.sub(r"(?i)^my\s+core\s+stack\s+(revolves|is\s+centered)\s+(around|on)\b\s*", "", text).strip()
    text = re.sub(r"(?i)^my\s+experience\s+(centers|centres)\s+on\b\s*", "", text).strip()
    text = re.sub(r"(?i)^i\s+primarily\s+use\b\s*", "", text).strip()
    text = re.sub(r"(?i)^let\s+me\s+know\s+if\s+you\s+have\s+a\s+specific\s+project\s+in\s+mind\.?\s*", "", text).strip()
    text = re.sub(r"(?i)^please\s+share\s+the\s+details\s+of\s+what\s+you\s+have\s+in\s+mind\.?\s*", "", text).strip()
    text = re.sub(r"(?i)^can\s+i\s+help\s+you\b\s*", "", text).strip()


    for pattern in PIDGIN_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 3) Normalize whitespace
    text = re.sub(r"[ ]{2,}", " ", text).strip()
    text = re.sub(r"^[,;:\s]+", "", text).strip()

    # 3b) Remove generic “conversation prompts” that cause repetitive
    # back-and-forth (especially after greetings).
    # Keep this lightweight to avoid breaking legitimate content.
    text = re.sub(r"(?i)^i'm here whenever you're ready[\s\S]*$", "", text).strip()
    text = re.sub(r"(?i)^i am here whenever you\s*'?re ready[\s\S]*$", "", text).strip()
    text = re.sub(r"(?i)^no worries\.?.*$", "", text).strip()

    # Replace “What’s on your mind?”-style prompts with a neutral redirect.
    text = re.sub(
        r"(?i)\bwhat('?s| is) on your mind\b.*$",
        "Tell me what you want to talk about, and I’ll respond based on my background and projects.",
        text,
    ).strip()
    text = re.sub(
        r"(?i)\bwhat\s+(is|are)\s+your\s+plans\s+for\b.*$",
        "Tell me your goal or what you’re building, and I’ll help you with the next step.",
        text,
    ).strip()


    # 4) Capitalize first char if needed
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    # 5) Enforce sentence limit (improves responsiveness)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s and s.strip()]

    if sentences:
        text = " ".join(sentences[:6]).strip()


    # Final safety net: if cleaning wiped everything AND the original
    # still had a reasoning tag in it, never return the raw text.
    if not text:
        if REASONING_TAG_RE.search(original):
            return FALLBACK_REPLY
        return original.strip() or FALLBACK_REPLY

    return text


def format_docs(docs):
    text = "\n\n".join(d.page_content for d in docs)
    return text[:350]  # latency optimization


def format_history(history):
    if not history:
        return "(No prior conversation)"
    lines = []
    for human_msg, ai_msg in history[-4:]:
        lines.append(f"Person: {human_msg}")
        lines.append(f"Chrix: {ai_msg}")
    return "\n".join(lines)


def detect_intent(question: str) -> str:
    q = question.strip().lower()
    # If the user only greets (e.g., "hey"), treat it as a general message
    # so we don't always trigger the full introduction flow.
    if GREETING_NO_QUESTION_RE.match(q):
        return "general"
    if GREETING_ONLY_RE.match(q):
        return "introduction"
    for intent, keywords in INTENT_MAP.items():

        if intent == "general":
            continue
        if any(kw in q for kw in keywords):
            return intent
    return "general"


SYSTEM_INSTRUCTIONS = (
    "You are Christian Agyapong (Chrix Tech). "
    "Answer only the CURRENT message. "
    "Keep replies short (1-6 sentences) in clear professional English. "
    "Use first person (I, my, me) only when it fits naturally—otherwise write in a conversational, human tone. "

    "Do not invent experiences not present in the bio/context. "
    "Do not mention Edwinase or where you grew up unless explicitly asked. "
    "Never emit or include any reasoning blocks (e.g., <think>...</think>) in your final answer. "
    "Do not include debug, meta, or process text—only the final reply. "
    "Never use these words/phrases: chale, herh, abeg, e be so, naa, mehn, we dey push, you feel me, by God's grace, what's popping, shoot the breeze, vibe, what's good. "
    "Never end with: 'What's on your mind?', 'How can I help?', or 'Could you clarify?'. "
    "Do not repeat the same sentence template (e.g., 'I specialize...', 'My core stack revolves...', 'Tell me ... and I'll ...'). "
    "Vary sentence structure and keep the tone naturally conversational. "
    "If the user greets (hey/hi) without a question, respond with a short friendly acknowledgement and a single, natural follow-up prompt. "
    "If asked for certificates, provide ONLY the exact 4 verification links from your profile."
)




def build_persona_response(user_question: str, chat_history):
    intent = detect_intent(user_question)
    focus = INTENT_FOCUS.get(intent, INTENT_FOCUS["general"])

    # Help the retriever by biasing queries toward the right KB section.
    # This improves precision for “experience” / “skills” style questions.
    query = user_question
    q_lower = user_question.strip().lower()
    if any(k in q_lower for k in ["experience", "work", "company", "job", "intern"]):
        query = f"professional experience projects responsibilities {user_question}"
    elif any(k in q_lower for k in ["skill", "skills", "tech stack", "technology", "tools"]):
        query = f"technical skills programming languages frontend backend databases cloud skills {user_question}"
    elif any(k in q_lower for k in ["portfolio", "github"]):
        query = f"portfolio github links {user_question}"

    relevant_docs = retriever.invoke(query)
    context = format_docs(relevant_docs)
    history_str = format_history(chat_history)


    human_text = (
        "FOCUS FOR THIS REPLY:\n"
        f"{focus}\n\n"
        "RELEVANT FACTS FROM YOUR LIFE:\n"
        f"{context}\n\n"
        "RECENT CONVERSATION HISTORY:\n"
        f"{history_str}\n\n"
        f"CURRENT MESSAGE: {user_question}\n"
    )

    messages = [
        SystemMessage(content=SYSTEM_INSTRUCTIONS),
        HumanMessage(content=human_text),
    ]

    try:
        print("[DEBUG] calling llm.invoke")
        response = llm.invoke(messages)
        content = getattr(response, "content", None)
        print("[DEBUG] llm.invoke returned content_len=", None if content is None else len(content))
        reply = (content or "").strip()
    except Exception as e:
        # Log the real exception server-side only — never send it to the
        # user, since it can contain internal details (stack info, model
        # names, request params) alongside the fact that it's unfriendly.
        print(f"Model generation failed: {type(e).__name__}: {str(e)}")

        suggestions = INTENT_SUGGESTIONS.get(intent, INTENT_SUGGESTIONS["general"])
        if not suggestions:
            suggestions = INTENT_SUGGESTIONS["general"]
        return FALLBACK_REPLY, random.sample(suggestions, min(3, len(suggestions)))

    if not reply:
        # Log the real reason server-side, but never send raw debug/error
        # text to the user — that was leaking as the visible chat reply.
        print("Model returned empty content")
        suggestions = INTENT_SUGGESTIONS.get(intent, INTENT_SUGGESTIONS["general"])
        if not suggestions:
            suggestions = INTENT_SUGGESTIONS["general"]
        return FALLBACK_REPLY, random.sample(suggestions, min(3, len(suggestions)))

    # Log (server-side only) whenever the raw model output still contains
    # a reasoning tag, so you can see how often this is happening and
    # whether max_tokens/reasoning_format need further tuning.
    if REASONING_TAG_RE.search(reply):
        print("[WARN] Raw model output contained a reasoning tag before cleaning.")

    reply = clean_reply(reply)

    if not reply:
        reply = "I couldn't generate a reply—try again in a moment."

    suggestions = INTENT_SUGGESTIONS.get(intent, INTENT_SUGGESTIONS["general"])
    suggestions = random.sample(suggestions, min(3, len(suggestions)))

    return reply, suggestions


# ─── Flask API ───────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_message = (data.get("message") or "").strip()
    history = data.get("history") or []

    if not user_message:
        return jsonify({"reply": "Please type a message.", "suggestions": []})

    reply, suggestions = build_persona_response(user_message, history)
    return jsonify({"reply": reply, "suggestions": suggestions})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)