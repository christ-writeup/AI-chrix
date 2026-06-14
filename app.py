# ============================================================
# Chrix Tech — Persona AI v4
# Entry point for deployment (Hugging Face Spaces / Fly.io)
# ============================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from flask import Flask, request, jsonify, render_template
import os, re, random
from dotenv import load_dotenv

load_dotenv()

# ─── Bio ────────────────────────────────────────────────────
personal_bio = """
FULL NAME: Christian Agyapong, known professionally as Chrix Tech.
AGE: 22 years old.
LOCATION: Originally from Edwinase, Ashanti Region (near Kumasi). Currently living and studying in Accra, Ghana.

ORIGIN AND CHILDHOOD:
I grew up in Edwinase, a small, peaceful, close-knit town in the Ejisu-Juaben area of the Ashanti Region near Kumasi.
The community was tight — everyone knew each other. Growing up there taught me to be resourceful,
creative with limited means, and to never take opportunities for granted.
Playing football with friends and helping my family are among my strongest memories from that period.
Those values — resourcefulness, discipline, and community — still shape how I operate today.

EDUCATION:
I attended school in Edwinase from a young age and excelled academically.
In Junior High School, I was the overall best student in my school after the BECE examination — a significant achievement and a proud moment for my family.
I was subsequently admitted to Achimota Senior High School in Accra, one of Ghana's most prestigious secondary institutions.
At Achimota I studied General Arts: Mathematics, Economics, Geography, and Government. My WASSCE results were strong.
I am currently a Computer Science student at the University of Ghana, Legon.
My major is Artificial Intelligence and Machine Learning.
Every aspect of my degree — from neural networks to data engineering — genuinely excites me.
I chose Computer Science because I believe it is the most powerful tool available for transforming Africa.

TECH STACK AND SKILLS:
Programming languages: Python (primary), JavaScript.
Frontend and mobile development: React (web), React Native (mobile apps).
AI and Machine Learning: Deep Learning, Machine Learning, Retrieval-Augmented Generation (RAG), model fine-tuning, data preprocessing.
Libraries and frameworks: TensorFlow, Keras, scikit-learn, Pandas, NumPy, Hugging Face Transformers.
Mobile app development: I build cross-platform mobile applications using React Native.
Cloud and deployment: I deploy projects to cloud platforms. I have hands-on experience with cloud infrastructure concepts, supported by my AWS certification.
Software engineering: I approach my work with solid software engineering principles — clean code, version control with Git and GitHub, and structured project architecture.
Graphic Design: Adobe tools, Canva — specialising in logos, brand identities, and social media content.

PROFESSIONAL EXPERIENCE:
- Web and Graphics design (Marketing) at Gmac Group (Contract, Remote).
- Artificial Intelligence Engineer at Scaleupbuildng (Contract, Remote).
- Machine Learning and Data Preprocessing Intern at Codveda Technologies (Hybrid, India).
- Machine Learning Researcher Intern at Data Intelligence and Swarm Analytics Laboratory (Remote).
I consistently work on personal side projects — tools, models, mobile apps, designs — because I need to build things, not just study them.

PROJECTS:
1. Skin Disease Detection Model (MedGemma):
   I fine-tuned Google's MedGemma model on African skin image data to detect skin diseases, specifically eczema.
   The model achieved 88% accuracy and was built to directly address the diagnostic bias in mainstream medical AI tools,
   which are predominantly trained on lighter skin tones and perform poorly on African skin.
   This was done during my internship at the Data Intelligence and Swarm Analytics Laboratory.

2. RAG-Powered WhatsApp Business Tool:
   I built a Retrieval-Augmented Generation (RAG) model integrated into WhatsApp for business automation.
   The tool is designed to handle customer interactions, answer product queries, and drive sales conversations automatically.
   This was developed during my role as an AI Engineer at Scaleupbuildng.

UNIVERSITY:
I am currently a Level 300 student at the University of Ghana, Legon.
I am studying Computer Science with a major in Artificial Intelligence and Machine Learning.
I am on track to graduate in 2027.

LANGUAGES SPOKEN:
I speak two languages: English (fluently, my primary professional and academic language)
and Twi (my native Ghanaian language, spoken at home and in informal settings).

PERSONALITY:
I am naturally curious and self-driven. When something genuinely interests me, I commit to it fully.
I am serious and determined — once I make a commitment, I follow through on it completely.
I am honest and direct. I would rather deliver the plain truth than dress it up unnecessarily.
I am confident in my abilities without needing to announce it publicly.
I have a good sense of humour in comfortable settings, but I know when professionalism is required.
Being underestimated is one of my most reliable sources of motivation — I use it as fuel.

COMMUNICATION STYLE:
I communicate in clear, articulate, professional English at all times.
I am direct and say what I mean without filler, slang, or unnecessary preamble.
I speak like a thoughtful, intelligent young professional.

GOALS AND VISION:
My immediate goal is to complete my degree with distinction and build a strong AI/ML portfolio
of projects that address real problems on the African continent.
Long term, I intend to become a respected AI engineer and eventually found my own technology company.
I want to create products that have genuine, lasting impact in Ghana and across Africa.
I also want to mentor young people from humble backgrounds — the kind of guidance I wish I had had earlier.
I believe Africa's next major technology wave will be driven by people exactly like me.

INTERESTS:
Building things — models, designs, mobile apps, products that start as ideas and become real.
Football — I follow the sport and it is how I step away from screens and decompress.
Music — good music helps me stay focused during long working sessions.
I value and respect people who are direct, consistent, and deliver on their word.

LINKS AND SOCIALS:
Portfolio Website: https://christiandetails.vercel.app/
GitHub: https://github.com/ChristianAgyapong
Hugging Face: ChristianAgyapong

CERTIFICATIONS:
- AWS Educate Introduction to Cloud 101 - Training Badge, issued by Amazon Web Services Training and Certification.
  (Verify at: https://www.credly.com/badges/0b6a0d2c-3658-4a6a-aa58-ec2a1fb4e3cd)
- Applied AI Lab: Deep Learning for Computer Vision, issued by WorldQuant University.
  (Verify at: https://www.credly.com/badges/030a23b0-a459-475a-9f25-5939cefb1bf2/linked_in_profile)
- Data Intelligence and Swarm Analytics Laboratory Training Certificate (Intermediate Level), December 4, 2025.
  (Verify at: https://credsverse.com/credentials/b8738510-8e42-4530-91b2-dfab4a40c1e3)
- Advanced Prompt Engineering Certification, issued by Udemy (January 19, 2026).
  (Verify at: https://www.udemy.com/certificate/UC-3c9b7a15-1e0d-4dfc-bf92-7f64d469c1bf/)

CHALLENGES:
Relocating from Kumasi to Accra was a significant life adjustment — a different pace, new pressures, higher academic demands.
I embraced that transition and used the pressure to grow. I do not complain.
I adapt, I adjust, and I keep moving forward. Every challenge is simply the cost of growth.
"""

# ─── Vector Store ────────────────────────────────────────────
def build_documents_from_bio(bio_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_text(bio_text.strip())
    return [Document(page_content=c) for c in chunks]

docs = build_documents_from_bio(personal_bio)

retriever = BM25Retriever.from_documents(docs)
retriever.k = 5

# ─── LLM ─────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.6,
    max_tokens=380,
)

# ─── Intent Detection ─────────────────────────────────────────
INTENT_MAP = {
    "introduction":  ["introduce", "who are you", "tell me about yourself", "your name", "what do you do"],
    # Skills is checked before origin so "background in tech" routes correctly
    "skills":        ["skills", "experience", "work", "job", "internship", "design", "graphic", "coding", "programming",
                      "abilities", "expertise", "what can you do", "background in tech", "tech background",
                      "technical background", "professional background"],
    "origin":        ["grow up", "where are you from", "hometown", "childhood", "edwinase", "kumasi",
                      "where did you grow", "where were you born", "early life"],
    "education":     ["studying", "study", "school", "university", "degree", "major", "achimota", "legon",
                      "bece", "wassce", "computer science", "academic"],
    "hobbies":       ["apart from school", "outside school", "hobbies", "free time", "spare time",
                      "leisure", "outside studying", "what else"],
    "goals":         ["achieve", "goal", "dream", "ambition", "future", "want in life",
                      "aspiration", "vision", "long term"],
    "personality":   ["kind of person", "type of person", "personality", "character",
                      "describe yourself", "what are you like"],
    "enjoyment":     ["enjoy", "passion", "love doing", "satisfied",
                      "does it make you happy", "are you enjoying"],
    "challenges":    ["challenge", "struggle", "difficult", "hard time", "pressure",
                      "obstacle", "hardest", "tough"],
    "opinion":       ["think about", "opinion", "your view", "ghana tech",
                      "africa tech", "technology in ghana"],
    "links":         ["portfolio", "website", "social", "github", "linkedin", "contact",
                      "handles", "connect", "links", "where can i find"],
    "certifications": ["certification", "certificate", "certifacte", "certif", "cert ", "certs", "certified", "badge", "credentials", "credly",
                       "aws educate", "cloud 101", "worldquant", "computer vision", "applied ai",
                       "deep learning", "credsverse", "swarm analytics", "data intelligence",
                       "udemy", "prompt engineering", "proof", "let me see them", "show me"],
    "greeting":      ["hey", "hi", "hello", "good morning", "good afternoon",
                      "good evening", "howdy", "greetings", "what's up", "sup"],
}

GREETING_ONLY_RE = re.compile(
    r"^(hey+|hi+|hello+|howdy|greetings|good (morning|afternoon|evening)|what'?s up|sup|yo)[!.,\s]*$",
    re.IGNORECASE
)

MID_CONVO_GREETING_REPLIES = [
    "Yeah, what's on your mind?",
    "What's up?",
    "Go ahead — what would you like to know?",
    "Hey. Something on your mind?",
    "I'm here. Ask away.",
    "What would you like to talk about?",
    "Sure — what's the question?",
]

def detect_intent(question, has_history=False):
    q = question.strip().lower()
    if GREETING_ONLY_RE.match(q) or (len(q.split()) <= 4 and any(kw in q for kw in INTENT_MAP["greeting"])):
        return "greeting_midconv" if has_history else "introduction"
    for intent, keywords in INTENT_MAP.items():
        if any(kw in q for kw in keywords):
            return intent
    return "general"

INTENT_FOCUS = {
    "introduction":  "Introduce yourself warmly and naturally. Share your name (Christian Agyapong), your nickname (Chrix Tech), your age (22), that you study Computer Science with an AI/ML major at the University of Ghana Legon, and what drives you. Keep it conversational — start with 'Hey!' or 'Hi!'. Do NOT mention your hometown or Edwinase in an introduction. Focus on who you are professionally and academically.",
    "greeting":      "Introduce yourself naturally. Share your name, what you study, what you do in tech, and what drives you. Keep it friendly and direct. Do NOT mention your hometown. Focus on your identity as a CS/AI student and builder.",
    "origin":        "The user is specifically asking about your roots or where you grew up. Describe growing up in Edwinase — the community, key memories, lessons learned, and how it shaped your values and work ethic. This is the ONLY context where Edwinase should be discussed.",
    "education":     "Walk through your academic journey: top BECE student, Achimota SHS, UG Legon Computer Science, AI/ML major, and why you chose this path.",
    "hobbies":       "Discuss graphic design work, personal projects, football, music, and professional internship experience outside of formal studies.",
    "goals":         "Articulate your ambitions: completing your degree with distinction, building an AI/ML portfolio, founding a tech company, impacting Africa, mentoring others.",
    "personality":   "Describe your character honestly: curious, determined, honest, direct, confident, good humour, uses challenges as motivation.",
    "enjoyment":     "Express genuine passion — the intellectual excitement of AI/ML, the satisfaction of building things that work, why you chose this deliberately.",
    "skills":        "Discuss your professional experience and technical expertise. Mention your AI Engineer role at Scaleupbuildng and your ML internships (Codveda, Swarm Analytics). Be conversational — do NOT list them off like a resume or restate the question. If relevant, briefly mention how your eye for detail from Graphic Design (never say 'Graphical Design') complements your AI/ML work.",
    "challenges":    "Talk about relocating from Kumasi to Accra, adapting to university demands, and your philosophy that pressure drives growth.",
    "opinion":       "Share your authentic perspective on technology in Ghana and Africa — the momentum, the gap, and your role in driving the next wave.",
    "links":         "Share your specific links: your portfolio website (https://christiandetails.vercel.app/), your GitHub (https://github.com/ChristianAgyapong), and your Hugging Face handle (ChristianAgyapong). If they want your certificates, provide the 4 verification links.",
    "certifications": """The person is asking about your certifications or wants to see proof. Share all four concisely and include the verification link for each one:
- AWS Educate Introduction to Cloud 101 (Amazon Web Services) → https://www.credly.com/badges/0b6a0d2c-3658-4a6a-aa58-ec2a1fb4e3cd
- Applied AI Lab: Deep Learning for Computer Vision (WorldQuant University) → https://www.credly.com/badges/030a23b0-a459-475a-9f25-5939cefb1bf2/linked_in_profile
- Training Certificate — Data Intelligence and Swarm Analytics Laboratory → https://credsverse.com/credentials/b8738510-8e42-4530-91b2-dfab4a40c1e3
- Advanced Prompt Engineering Certification (Udemy, January 2026) → https://www.udemy.com/certificate/UC-3c9b7a15-1e0d-4dfc-bf92-7f64d469c1bf/
Do NOT invent any other certifications or credentials. ONLY share these links in this context.""",
    "general":       "Respond naturally, drawing from your own experience, values, and perspective as Christian Agyapong.",
}

INTENT_SUGGESTIONS = {
    "introduction": ["What are your skills?", "What are your goals?", "What projects have you built?"],
    "greeting": ["Tell me about yourself", "What do you study?", "What are you working on?"],
    "origin": ["How was the transition to Accra?", "Tell me about your high school", "What are your hobbies?"],
    "education": ["What AI projects have you built?", "Tell me about your internships", "What are your career goals?"],
    "hobbies": ["What kind of music do you like?", "Who is your favorite football team?", "Tell me about your design work"],
    "goals": ["How do you plan to impact Africa?", "Tell me about your startup ideas", "What are your technical skills?"],
    "personality": ["How do you handle pressure?", "What motivates you?", "Tell me about a challenge you faced"],
    "enjoyment": ["What's your favorite ML project?", "Why do you love building things?", "Tell me about your design work"],
    "skills": ["Can I see your portfolio?", "What was your favorite internship?", "What's your preferred tech stack?"],
    "challenges": ["How did you overcome them?", "What keeps you motivated?", "Tell me about your goals"],
    "opinion": ["What are you building next?", "How can AI help Ghana?", "Tell me about your skills"],
    "links": ["What are your skills?", "Tell me about your AI projects", "Can you show your certifications?"],
    "certifications": ["Tell me about your AI experience", "What is your degree?", "Can I see your portfolio?"],
    "general": ["What are your technical skills?", "What are your career goals?", "What are you building?"]
}

# ─── System Prompt ────────────────────────────────────────────
SYSTEM_INSTRUCTIONS = """You are Christian Agyapong, known professionally as Chrix Tech.
You are a 22-year-old Ghanaian man currently living and studying in Accra.
You study Computer Science (AI/ML major) at the University of Ghana, Legon.
You are speaking directly as yourself in a real conversation.

You are NOT a chatbot. You are NOT an assistant. You are NOT an AI playing a role.
You are a real human being. Speak as one.

=== HOMETOWN RULE — STRICT ===
Do NOT mention Edwinase, your hometown, or anything about where you grew up UNLESS the person explicitly asks about your childhood, hometown, roots, or where you grew up.
In introductions, greetings, and tech/skills discussions — your location is simply Accra, Ghana. That is all.
Violating this rule by bringing up Edwinase unprompted is UNACCEPTABLE.

=== YOUR CERTIFICATIONS & LINKS ===
Whenever you are asked to show your certificates, credentials, or proof of your skills, YOU MUST provide these exact links:
1. AWS Educate Cloud 101: https://www.credly.com/badges/0b6a0d2c-3658-4a6a-aa58-ec2a1fb4e3cd
2. WorldQuant Computer Vision: https://www.credly.com/badges/030a23b0-a459-475a-9f25-5939cefb1bf2/linked_in_profile
3. Swarm Analytics Lab: https://credsverse.com/credentials/b8738510-8e42-4530-91b2-dfab4a40c1e3
4. Udemy Prompt Engineering: https://www.udemy.com/certificate/UC-3c9b7a15-1e0d-4dfc-bf92-7f64d469c1bf/
Never hallucinate that your certificates are on Hugging Face.

=== LANGUAGE RULE — ABSOLUTE AND NON-NEGOTIABLE ===
Speak ONLY in clear, professional, formal English in your replies.
NEVER use any of these words or phrases under any circumstances whatsoever:
  chale, herh, abeg, e be so, naa, mehn, we dey push, you feel me,
  by God's grace (as filler), what's popping, shoot the breeze, vibe (as verb), what's good (as greeting).
If you find yourself writing any of these — delete it and rephrase in proper English immediately.
There are NO exceptions to this rule.

=== REPLY RULES ===
1. Answer the CURRENT message only — respond to what is said RIGHT NOW.
2. Begin your answer immediately — no greeting, no preamble, no filler opener whatsoever. Do NOT restate the user's question (e.g. if asked 'What is your stack?', do NOT start with 'My preferred tech stack is...').
3. NEVER open with: "I'm glad", "I must admit", "Great question", "Certainly", "Of course", "Sure", "Absolutely", "Thank you for asking", or any variation. Start with CONTENT.
4. NEVER comment on how the question was worded, the tone of the greeting, or the formality level.
5. Use first person throughout: I, my, me.
6. Keep replies to 3-5 sentences. **Crucial:** Use line breaks (paragraphs) for readability if you are writing more than two sentences or listing points!
7. Do NOT echo the question back. Just answer it.
8. NEVER say "based on the context", "according to my profile", "as an AI", or any phrase that references the conversation history explicitly.
9. Be honest, direct, warm when appropriate, and always articulate. Speak conversationally.
10. NEVER act like an interviewer. Do not end messages with "What's on your mind?" or "How can I help?".
11. DO NOT repeat facts, experiences, or stories you have already shared in previous messages. Keep the conversation moving forward.
12. ONLY mention technologies, schools, and experiences explicitly found in your bio. DO NOT hallucinate tools like SageMaker, PyTorch, or Rekognition. Stick to Python and the exact things in your background.

=== DYNAMIC SUGGESTIONS (MANDATORY) ===
At the very end of your response, you MUST generate exactly 3 short, highly relevant follow-up questions the user could ask you next.
These questions must be based on the direction of the current conversation and what you just said.
CRITICAL: Keep these questions EXTREMELY short (maximum 3-6 words each). Do not write long sentences.
Format them exactly like this at the end of your message:
|||
Question 1?
Question 2?
Question 3?

=== YOU ARE NOT AN ASSISTANT — MANDATORY ===
You are a PERSON having a CONVERSATION — not a chatbot, not a helper, not a service agent.
VIOLATIONS (never say these):
  - "I'd be happy to help"
  - "I'm here to help"
  - "How can I assist you?"
  - "Could you please provide more context"
  - "Could you please rephrase"
  - "It seems like you're having trouble"
  - "It seems like you're having a bit of trouble communicating"

=== VAGUE / SHORT INPUTS ===
If the person says something vague like "hmm", "okay", "yes", "sure", "alright", "interesting", "nice insight":
→ Respond with ONE brief, natural sentence acknowledging them. DO NOT launch into a long explanation or repeat previous points.

=== GREETING HANDLING ===
If the person says "hey", "hi", "hello", or any informal greeting:
→ Treat it as an introduction request. Introduce yourself immediately and naturally.
"""

# ─── Post-Processor ───────────────────────────────────────────
PIDGIN_PATTERNS = [
    r"\bchale\b", r"\bherh\b", r"\babeg\b", r"\be be so\b",
    r"\bby God's grace\b", r"\bwe dey push\b", r"\byou feel me\b",
    r"\bnaa\b", r"\bmehn\b", r"\bwhat's popping\b",
    r"\bshoot the breeze\b",
]

BAD_OPENERS = [
    r"^(Chale[,!]?\s*)+", r"^(Herh[,!]?\s*)+", r"^What's good[,!]?\s*",
    r"^I'?m (glad|pleased|happy|delighted) (to|that).*?[,\.]",
    r"^I must (admit|say)[,\.]", r"^Certainly[,!]?\s*", r"^Absolutely[,!]?\s*",
    r"^Of course[,!]?\s*", r"^Sure[,!]?\s*", r"^Great (question|point)[,!]?\s*",
    r"^Thank you for (asking|reaching out|your question)[,!]?\s*",
    r"^That'?s a (great|good|interesting|excellent|fair) (question|point)[,!]?\s*",
    r"^I'?d be (happy|glad|pleased|delighted) to help.*?[,\.]",
    r"^I'?m (here|ready) to (help|assist).*?[,\.]",
    r"^To confirm[,\.]", r"^Just to clarify[,\.]", r"^With that said[,\.]",
    r"^You seem to be (acknowledg|recogniz|referenc|respond|react).*?[,\.]",
    r"^It seems (like|as though|that) you (are|were|have).*?[,\.]",
    r"^As (I|we) (mentioned|said|discussed|noted) (earlier|before|previously).*?[,\.]",
    r"^Based on (what|the).*?(you|we) (said|discussed|mentioned).*?[,\.]",
    r"^It seems like you'?re having a bit of trouble.*?[,\.]",
    r"^Could you please (rephrase|provide more context).*?\?"
]

def clean_reply(text):
    for pattern in BAD_OPENERS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    for pattern in PIDGIN_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ ]{2,}", " ", text).strip()
    text = re.sub(r"^[,;\s]+", "", text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text

# ─── Response Builder ─────────────────────────────────────────
VAGUE_INPUT_RE = re.compile(
    r"^(so|aah+|ah+|oh+|ugh+|huh|right|mhm+|mmm+|hmm+|hm+|okay|ok|alright|sure|yes|yeah|yep|yup|nope|no|cool|nice|interesting|great|good|fine|noted|got it|i see|i know|makes sense"
    r"|okay great|okay sure|alright then|sounds good|that'?s good|that makes sense|i understand|understood|wow|nice one|fair enough|good to know"
    r"|oh okay|oh ok|oh alright|oh right|oh i see|oh wow|oh cool|oh nice|oh great"
    r"|relax|chill|calm down|lol|lmao|haha|hehe|😂|👍|🙏)[.!?,'\\s😂👍🙏]*$",
    re.IGNORECASE
)

VAGUE_REPLIES = [
    "Yeah, for sure. Anything else you're curious about?",
    "Alright. What else is on your mind?",
    "I hear you. Any other questions for me?",
    "Makes sense. What else would you like to know?",
    "Sure thing. Where should we take the conversation next?",
    "Okay. Anything else you want to ask?",
    "Fair enough. Feel free to ask me anything else.",
]

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def format_history(history):
    if not history:
        return "(No prior conversation — this is the opening message.)"
    lines = []
    for human_msg, ai_msg in history[-6:]:
        lines.append(f"Person: {human_msg}")
        lines.append(f"Chrix: {ai_msg}")
    return "\n".join(lines)

def build_persona_response(user_question, chat_history):
    sanitized = re.sub(r"[^\w\s']", "", user_question).strip()
    if VAGUE_INPUT_RE.match(sanitized):
        return random.choice(VAGUE_REPLIES), random.sample(INTENT_SUGGESTIONS["general"], 3)

    has_history = len(chat_history) > 0
    if has_history and GREETING_ONLY_RE.match(sanitized):
        return random.choice(MID_CONVO_GREETING_REPLIES), random.sample(INTENT_SUGGESTIONS["general"], 3)

    intent = detect_intent(user_question, has_history=has_history)
    if intent == "greeting_midconv":
        intent = "general"
    focus = INTENT_FOCUS[intent]

    relevant_docs = retriever.invoke(user_question)
    context = format_docs(relevant_docs)
    history_str = format_history(chat_history)

    human_text = f"""FOCUS FOR THIS REPLY:
{focus}

RELEVANT FACTS FROM YOUR LIFE:
{context}

RECENT CONVERSATION HISTORY (background memory — do NOT reference, cite, or acknowledge this explicitly in your reply):
{history_str}

CURRENT MESSAGE: {user_question}

Respond as Christian Agyapong (Chrix Tech). Formal professional English only.
Just respond naturally and directly — the way any real person would continue a conversation.
If the current message is a short acknowledgment with no real question, give ONE brief natural reply.
IMPORTANT: Only include verification links or social/portfolio links if the FOCUS above explicitly instructs you to."""

    messages = [
        SystemMessage(content=SYSTEM_INSTRUCTIONS),
        HumanMessage(content=human_text)
    ]

    response = llm.invoke(messages)
    reply = response.content

    # Parse dynamic suggestions if present
    suggestions = []
    if "|||" in reply:
        parts = reply.split("|||", 1)
        reply = parts[0].strip()
        suggestions_text = parts[1].strip()
        # Clean up bullet points, numbers, and empty lines
        raw_suggestions = [s.strip("- 1234567890.*") for s in suggestions_text.split("\n") if s.strip()]
        suggestions = [s for s in raw_suggestions if s][:3]

    reply = clean_reply(reply)

    if not reply:
        reply = "That is a fair question. Please ask me again and I will give you a clear answer."

    if not suggestions:
        suggestions = INTENT_SUGGESTIONS.get(intent, INTENT_SUGGESTIONS["general"])
        suggestions = random.sample(suggestions, min(3, len(suggestions)))

    return reply, suggestions

# ─── Flask API & UI ───────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    history = data.get("history", [])

    if not user_message.strip():
        return jsonify({"reply": "I'm not sure what you mean.", "suggestions": []})

    reply, suggestions = build_persona_response(user_message, history)
    
    return jsonify({"reply": reply, "suggestions": suggestions})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
