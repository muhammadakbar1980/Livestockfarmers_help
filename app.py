"""
Punjab Livestock Rehnuma - Bilingual (English / Urdu) RAG chatbot
Livestock & Dairy Development Department, Punjab

How it works
1. Loads the FAQ PDF(s) placed in the GitHub repo (same folder as app.py, or a data/ folder)
2. Extracts text page by page, skips unreadable pages
3. Splits PDF text into token chunks (tiktoken); each FAQ in Urdu_FAQ.txt is one chunk
4. Creates embeddings (FastEmbed; TF-IDF backup) and stores them in FAISS (NumPy backup)
   Only streamlit + pypdf are strictly needed; every other package has a built-in backup.
5. Translates the farmer's question (English / Urdu / Roman Urdu) into an English search query (Groq)
6. Finds the most relevant FAQ chunks and Groq writes the answer in English and Urdu

Run locally:   streamlit run app.py
Secret needed: GROQ_API_KEY
"""

import glob
import hashlib
import io
import json
import os
import re
from datetime import datetime

import numpy as np
import streamlit as st

st.set_page_config(page_title="Punjab Livestock Rehnuma | پنجاب لائیو سٹاک رہنما",
                   page_icon="🐄", layout="centered", initial_sidebar_state="collapsed")

# =====================================================================
# SETTINGS
# =====================================================================
ANSWER_MODELS = {
    "GPT-OSS 120B (best quality)": "openai/gpt-oss-120b",
    "GPT-OSS 20B (faster)": "openai/gpt-oss-20b",
}
QUERY_MODEL = "openai/gpt-oss-20b"          # fast model for translating the question
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # free, understands English + Urdu
CHUNK_TOKENS = 150                          # the embedding model reads ~128 tokens
CHUNK_OVERLAP = 30
TOP_K = 5
MIN_RELEVANCE = 0.45                        # below this the documents probably lack the answer
HELPLINE = "0800-09211"
WEBSITE = "https://livestock.punjab.gov.pk"

TECH_LEVELS = {
    "👨‍🌾 Farmer (simple)": (
        "The reader is a livestock farmer in rural Punjab. Use very simple everyday words, "
        "short sentences and practical steps. Avoid scientific terms; if one is needed, explain it "
        "in brackets. Use local examples (bhains/buffalo, gaye/cow, bakri/goat, chara/fodder, wanda/concentrate)."),
    "🧑‍🏫 Extension worker (moderate)": (
        "The reader is a livestock assistant or extension worker. Use clear language with correct "
        "technical terms, short explanations of why, and practical field advice for farmers."),
    "🩺 Veterinary expert (technical)": (
        "The reader is a veterinary professional. Use correct scientific and veterinary terminology, "
        "mention causes, signs, diagnosis, prevention and control measures, and departmental procedures."),
}

RESPONSE_SIZES = {
    "Short": "Keep each language section to about 60-80 words: the direct answer and 2-3 key points.",
    "Medium": "Keep each language section to about 120-180 words with clear bullet points.",
    "Detailed": ("Give a complete answer of about 250-350 words per language section, with headings "
                 "or bullet points, steps, precautions and where to get help."),
}

ANSWER_LANGUAGES = ["English + اردو", "English only", "اردو only"]

SAMPLE_QUESTIONS = [
    "How can I apply for the Livestock Card?",
    "What are the signs of heat (estrus) in buffaloes?",
    "مویشیوں کو منہ کھر کی بیماری سے کیسے بچائیں؟",
    "Artificial insemination ki fees kitni hai?",
    "What should I do if the veterinary staff does not come to my farm?",
    "بکریوں کی ویکسینیشن کب کروانی چاہیے؟",
]

# =====================================================================
# STYLE (Urdu right-to-left)
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&display=swap');

/* hide Streamlit menu, footer and sidebar so it looks like a chat app */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {display: none !important;}
header[data-testid="stHeader"] {background: transparent; height: 0;}
.block-container {padding-top: 0.8rem; padding-bottom: 6rem; max-width: 780px;}

/* chat header */
.chat-header {display: flex; align-items: center; gap: 0.8rem; padding: 0.7rem 1rem; margin-bottom: 0.6rem;
  border-radius: 16px; background: linear-gradient(135deg, #0b6e4f, #13945f); color: #fff;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);}
.chat-header .logo {font-size: 2rem; background: #fff; border-radius: 50%; width: 52px; height: 52px;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;}
.chat-header .title {font-weight: 700; font-size: 1.15rem; line-height: 1.2;}
.chat-header .sub {font-size: 0.8rem; opacity: 0.9;}
.chat-header .dot {display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: #7CFC9A; margin-right: 4px;}
.chat-header a.call {margin-left: auto; color: #fff !important; text-decoration: none; background: rgba(255,255,255,0.2);
  padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.85rem; white-space: nowrap;}

/* chat bubbles */
[data-testid="stChatMessage"] {border-radius: 18px; padding: 0.7rem 0.9rem 1rem 0.9rem;
  margin-bottom: 0.5rem; overflow: visible;}
[data-testid="stChatMessage"]:has(.u-msg) {flex-direction: row-reverse; background: rgba(37,211,102,0.16);
  width: fit-content; max-width: 82%; min-width: 35%; margin-left: auto; border-bottom-right-radius: 4px;}
[data-testid="stChatMessage"]:has(.u-msg) [data-testid="stChatMessageContent"] {text-align: right;}
/* hide the invisible marker used to tell user and bot bubbles apart */
[data-testid="stChatMessageContent"] [data-testid="stElementContainer"]:has(.u-msg),
[data-testid="stChatMessageContent"] [data-testid="stElementContainer"]:has(.b-msg) {display: none;}
[data-testid="stChatMessage"]:has(.b-msg) {background: rgba(128,128,128,0.08);
  border: 1px solid rgba(128,128,128,0.15); margin-right: 6%; border-bottom-left-radius: 4px;}
.msg-time {font-size: 0.7rem; opacity: 0.55; margin-top: 0.2rem;}

/* Urdu text */
.urdu {direction: rtl; text-align: right; font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif;
       line-height: 2.3; font-size: 1.05rem;}
.urdu ul, .urdu ol {padding-right: 1.5rem; padding-left: 0;}
.lang-label {font-size: 0.75rem; font-weight: 600; opacity: 0.6; margin: 0.3rem 0 0.1rem 0;}
.lang-sep {border-top: 1px dashed rgba(128,128,128,0.35); margin: 0.6rem 0;}

/* quick-reply chips */
[data-testid="stChatMessage"] div[data-testid="stButton"] button {border-radius: 20px; font-size: 0.85rem;
  padding: 0.2rem 0.9rem; border-color: #13945f; text-align: left;}
[data-testid="stChatMessage"] div[data-testid="stButton"] button p {white-space: normal; overflow: visible;
  text-overflow: clip; line-height: 1.35;}

/* typing animation */
.typing span {display: inline-block; width: 8px; height: 8px; margin: 0 2px; border-radius: 50%;
  background: #13945f; animation: blink 1.4s infinite both;}
.typing span:nth-child(2) {animation-delay: 0.2s;} .typing span:nth-child(3) {animation-delay: 0.4s;}
@keyframes blink {0%, 80%, 100% {opacity: 0.2;} 40% {opacity: 1;}}
.disclaimer {font-size: 0.72rem; opacity: 0.6; text-align: center; margin-top: 0.4rem;}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# HELPERS
# =====================================================================
def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or os.environ.get("GROQ_API_KEY", "")


def find_repo_files():
    """FAQ files in the repo: PDFs and UTF-8 text files (e.g. Urdu_FAQ.txt). requirements.txt is ignored."""
    folder = os.path.dirname(os.path.abspath(__file__))
    files = []
    for sub in ("", "data"):
        for pattern in ("*.pdf", "*.txt"):
            files += glob.glob(os.path.join(folder, sub, pattern))
    return sorted(f for f in set(files) if not os.path.basename(f).lower().startswith("requirements"))


def is_garbled(text):
    """True if text is broken (e.g. Urdu fonts that do not extract correctly)."""
    if not text:
        return True
    broken = len(re.findall(r"[က-῿-]", text)) + text.count("(cid:") * 5
    return broken / max(len(text), 1) > 0.03


def clean_text(text):
    text = (text or "").replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource(show_spinner=False)
def load_tokenizer():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None                                    # word-based fallback


def light_mode():
    """LIGHT_MODE = true in Streamlit secrets switches off the neural model (uses ~200 MB less memory)."""
    try:
        value = st.secrets.get("LIGHT_MODE", False)
    except Exception:
        value = os.environ.get("LIGHT_MODE", False)
    return str(value).strip().lower() in ("true", "1", "yes")


class FastEmbedder:
    """Neural multilingual embeddings via FastEmbed - best quality. Uses little memory:
    2 CPU threads and small batches, so it stays inside Streamlit Cloud's free limits."""
    name = "FastEmbed multilingual MiniLM (English + Urdu)"
    min_relevance = MIN_RELEVANCE

    def __init__(self):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=EMBED_MODEL, threads=2)

    def embed(self, texts):
        return np.array(list(self.model.embed(list(texts), batch_size=16)), dtype="float32")


class TfidfEmbedder:
    """Backup embeddings made with NumPy only (TF-IDF of words + word pairs).
    Used automatically if fastembed is not installed or its model cannot download."""
    name = "TF-IDF (NumPy backup)"
    min_relevance = 0.09                      # TF-IDF similarity scores are naturally lower
    STOP = set("a an the of to in on for and or is are was be by with as at it this that from what "
               "how can do does i my me we our you your which who when where will should".split())

    def _terms(self, text):
        words = [w for w in re.findall(r"[a-z0-9]+|[\u0600-\u06FF]+", text.lower()) if w not in self.STOP and len(w) > 1]
        return words + [a + "_" + b for a, b in zip(words, words[1:])]

    def fit(self, texts):
        doc_freq = {}
        for text in texts:
            for term in set(self._terms(text)):
                doc_freq[term] = doc_freq.get(term, 0) + 1
        terms = sorted(doc_freq, key=lambda t: -doc_freq[t])[:30000]
        self.vocab = {t: i for i, t in enumerate(terms)}
        n = len(texts)
        self.idf = np.array([np.log((1 + n) / (1 + doc_freq[t])) + 1 for t in terms], dtype="float32")
        return self

    def embed(self, texts):
        vectors = np.zeros((len(texts), len(self.vocab)), dtype="float32")
        for row, text in enumerate(texts):
            for term in self._terms(text):
                col = self.vocab.get(term)
                if col is not None:
                    vectors[row, col] += 1
        return np.log1p(vectors) * self.idf


@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def load_fastembed():
    if light_mode():
        return None                                   # light mode: TF-IDF only, very low memory
    try:
        return FastEmbedder()
    except Exception:
        return None


def normalise(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


# =====================================================================
# STEP 1-4: BUILD THE KNOWLEDGE BASE (cached, built once per server)
# =====================================================================
def extract_pages(name, data):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    good, skipped = [], 0
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text())
        if not text:
            continue
        if is_garbled(text):
            skipped += 1
            continue
        good.append({"source": name, "page": number, "text": text})
    return good, skipped, len(reader.pages)


def make_chunks(pages):
    tokenizer = load_tokenizer()
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    chunks = []
    for page in pages:
        if tokenizer is not None:
            tokens = tokenizer.encode(page["text"])
            for start in range(0, len(tokens), step):
                piece = tokens[start:start + CHUNK_TOKENS]
                chunks.append({**page, "text": tokenizer.decode(piece), "tokens": len(piece)})
                if start + CHUNK_TOKENS >= len(tokens):
                    break
        else:
            words = page["text"].split()
            size, wstep = int(CHUNK_TOKENS / 1.3), int(step / 1.3)
            for start in range(0, len(words), wstep):
                piece = words[start:start + size]
                chunks.append({**page, "text": " ".join(piece), "tokens": int(len(piece) * 1.3)})
                if start + size >= len(words):
                    break
    return chunks


class NumpyIndex:
    """Backup vector store with the same search() as FAISS IndexFlatIP (used if faiss is not installed)."""

    def __init__(self, dim):
        self.vectors = np.zeros((0, dim), dtype="float32")

    def add(self, vectors):
        self.vectors = np.vstack([self.vectors, vectors.astype("float32")])

    def search(self, queries, k):
        scores = queries @ self.vectors.T
        ids = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, ids, axis=1), ids


def make_index(dim):
    try:
        import faiss
        return faiss.IndexFlatIP(dim), "FAISS"
    except ImportError:
        return NumpyIndex(dim), "NumPy (backup)"


def is_urdu(text):
    urdu = len(re.findall(r"[\u0600-\u06FF]", text))
    return urdu > len(re.findall(r"[A-Za-z]", text))


def read_text_faq(name, data):
    """Each block separated by a blank line becomes one chunk. Lines starting with # are notes."""
    text = data.decode("utf-8", errors="ignore")
    chunks = []
    for block in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in block.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
        if not lines:
            continue
        page_match = re.search(r"\(PDF page (\d+)\)", block)
        body = " ".join(l for l in lines if not re.fullmatch(r"\(PDF page \d+\)", l))
        chunks.append({"source": name, "page": int(page_match.group(1)) if page_match else "-",
                       "text": body, "tokens": int(len(body.split()) * 1.3)})
    return chunks


@st.cache_resource(show_spinner="📚 Building the knowledge base (first visit only)...")
def build_knowledge_base(files_key, file_items):
    """file_items: tuple of (name, bytes). files_key makes the cache rebuild when files change."""
    pages, text_chunks, skipped, total = [], [], 0, 0
    for name, data in file_items:
        if name.lower().endswith(".txt"):
            text_chunks.extend(read_text_faq(name, data))
            continue
        good, bad, count = extract_pages(name, data)
        pages.extend(good)
        skipped += bad
        total += count
    chunks = make_chunks(pages) + text_chunks
    if not chunks:
        return None
    for c in chunks:
        c["lang"] = "ur" if is_urdu(c["text"]) else "en"
    texts = [c["text"] for c in chunks]
    embedder = load_fastembed() or TfidfEmbedder().fit(texts)
    vectors = normalise(embedder.embed(texts))
    dimension = int(vectors.shape[1])
    index, store = make_index(dimension)
    index.add(vectors)
    del vectors                                       # free memory: the index keeps its own copy
    import gc
    gc.collect()
    stats = {"files": len(file_items), "pages_total": total, "pages_used": len(pages), "store": store,
             "pages_skipped": skipped, "chunks": len(chunks),
             "tokens": sum(c["tokens"] for c in chunks), "dimension": dimension,
             "embedder": embedder.name, "urdu_chunks": sum(c["lang"] == "ur" for c in chunks)}
    return {"index": index, "chunks": chunks, "stats": stats, "embedder": embedder}


def retrieve(queries, kb, top_k=TOP_K):
    """Search with every query (original, English, Urdu) and keep the best score for each chunk."""
    queries = [q for q in dict.fromkeys(queries) if q and q.strip()]
    vectors = normalise(kb["embedder"].embed(queries))
    k = min(top_k, len(kb["chunks"]))
    best = {}
    for row_scores, row_ids in zip(*kb["index"].search(vectors, k)):
        for score, i in zip(row_scores, row_ids):
            if i != -1 and float(score) > best.get(int(i), -1):
                best[int(i)] = float(score)
    ranked = sorted(best.items(), key=lambda x: -x[1])
    english = [i for i, _ in ranked if kb["chunks"][i]["lang"] == "en"][:4]
    urdu = [i for i, _ in ranked if kb["chunks"][i]["lang"] == "ur"][:4]
    chosen = sorted(set(english + urdu), key=lambda i: -best[i])
    return [{**kb["chunks"][i], "score": best[i]} for i in chosen]


# =====================================================================
# STEP 5-6: GROQ (question understanding + answer)
# =====================================================================
def groq_chat(api_key, model, messages, temperature=0.2, json_mode=False):
    """Call Groq. Uses the groq package; if it is not installed, calls the Groq REST API directly."""
    params = {"model": model, "messages": messages, "temperature": temperature,
              "max_completion_tokens": 4096, "reasoning_effort": "low"}
    if json_mode:
        params["response_format"] = {"type": "json_object"}
    try:
        from groq import Groq
    except ImportError:
        Groq = None

    if Groq is not None:
        client = Groq(api_key=api_key)
        try:
            reply = client.chat.completions.create(**params)
        except Exception:
            params.pop("reasoning_effort")             # model without reasoning option
            reply = client.chat.completions.create(**params)
        return reply.choices[0].message.content or ""

    import requests                                   # installed together with streamlit
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=params, timeout=90)
    if response.status_code == 400:
        params.pop("reasoning_effort")
        response = requests.post(url, headers=headers, json=params, timeout=90)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"] or ""


QUERY_PROMPT = """You help a livestock chatbot for Punjab, Pakistan.
The farmer's question may be in English, Urdu script, Roman Urdu or Punjabi.
Return ONLY JSON: {"language": "english" | "urdu" | "roman_urdu" | "mixed",
"search_query": "the question rewritten as a clear English search query, using livestock terms
(e.g. 'mooh khur' -> foot and mouth disease, 'gal ghotu' -> haemorrhagic septicaemia,
'bhains' -> buffalo, 'wanda' -> concentrate feed), including context from the chat history if the
question is a follow-up",
"urdu_query": "the same question written in proper Urdu script (اردو)"}"""


def understand_question(question, history, api_key):
    recent = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
    try:
        raw = groq_chat(api_key, QUERY_MODEL, [
            {"role": "system", "content": QUERY_PROMPT},
            {"role": "user", "content": f"CHAT HISTORY:\n{recent}\n\nQUESTION: {question}"},
        ], temperature=0, json_mode=True)
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return data.get("search_query") or question, data.get("urdu_query") or ""
    except Exception:
        return question, ""


def build_system_prompt(level, size, language):
    if language == "English + اردو":
        lang_rule = ("Write the answer TWICE: first in English, then the same answer in Urdu (Urdu script, "
                     "simple everyday Urdu that Punjab farmers understand). Put a line containing only "
                     "===URDU=== between the English and the Urdu parts. No other headings for languages.")
    elif language == "English only":
        lang_rule = "Write the answer in English only."
    else:
        lang_rule = "Write the answer in simple Urdu script only."

    return f"""You are "Punjab Livestock Rehnuma", the official-style assistant of the Livestock & Dairy
Development Department, Punjab, Pakistan. You help livestock farmers with animal health, diseases,
vaccination, breeding and artificial insemination, feeding, dairy, poultry and department services
(Livestock Card, helpline, veterinary hospitals, programmes).

KNOWLEDGE RULES
- Use the CONTEXT from the department's FAQ documents as your main source. Prefer its facts, fees,
  procedures, eligibility and contact details.
- The CONTEXT has [ENGLISH FAQ] and [URDU FAQ] parts. Write the English answer from the English FAQ.
  Write the Urdu answer DIRECTLY from the Urdu FAQ: keep its wording, terms and sentence style as far
  as possible instead of translating from English. The Urdu FAQ text was read by OCR, so silently fix
  obvious spelling mistakes (e.g. "کرذ" -> "کرنے", "کے لے" -> "کے لیے", "سکت ہیں" -> "سکتے ہیں").
  Only translate into Urdu when no Urdu FAQ covers that point.
- If the context does not contain the answer, you may give general good-practice advice suited to
  Punjab conditions (hot summers, floods, buffalo and cattle based small farms, fodder shortage in
  May-June and Nov-Dec, common diseases such as FMD, HS, LSD, PPR, mastitis, ticks), but clearly say
  that this is general advice and the farmer should confirm with the nearest veterinary hospital.
- Never invent fees, dates, phone numbers, programme rules or eligibility.
- Do not prescribe exact medicine doses unless they are written in the context; advise consulting a
  veterinary doctor. For emergencies (animal collapse, bleeding, severe bloat, poisoning, difficult
  calving, sudden deaths) tell the farmer to call the Livestock Helpline {HELPLINE} immediately.
- Politely decline questions not related to livestock, poultry, dairy or the department.

STYLE
- Technical level: {TECH_LEVELS[level]}
- Length: {RESPONSE_SIZES[size]}
- {lang_rule}
- Use bullet points or numbered steps where useful. Be respectful and encouraging.
- Do not add a sources list; the app shows sources separately.
- At the very end, add a line containing only ===NEXT=== followed by 3 short follow-up questions the
  farmer might ask next (one per line, no numbering, max 10 words each, in the same language and
  script as the farmer's question)."""


def answer_question(question, search_query, chunks, history, settings, api_key):
    context = "\n\n".join(
        f"[{'URDU FAQ' if c.get('lang') == 'ur' else 'ENGLISH FAQ'} - {c['source']}, page {c['page']}]\n{c['text']}"
        for c in chunks) or "No relevant document context found."
    messages = [{"role": "system",
                 "content": build_system_prompt(settings["level"], settings["size"], settings["language"])}]
    for m in history[-4:]:
        messages.append({"role": m["role"], "content": m["content"][:1500]})
    messages.append({"role": "user", "content":
                     f"CONTEXT:\n{context}\n\nENGLISH SEARCH MEANING: {search_query}\n\nFARMER'S QUESTION: {question}"})
    return groq_chat(api_key, settings["model"], messages, temperature=0.3)


# =====================================================================
# DISPLAY HELPERS
# =====================================================================
def md_to_html(text):
    """Very small markdown -> HTML for the Urdu (RTL) block."""
    import html
    lines, out, in_list = text.strip().split("\n"), [], False
    for line in lines:
        s = line.strip()
        if re.match(r"^([-*•]|\d+[.)])\s+", s):
            if not in_list:
                out.append("<ul>")
                in_list = True
            item = re.sub(r"^([-*•]|\d+[.)])\s+", "", s)
            out.append(f"<li>{html.escape(item)}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if s:
            s = html.escape(s.lstrip("#").strip())
            out.append(f"<p>{s}</p>")
    if in_list:
        out.append("</ul>")
    html_text = "\n".join(out)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html_text)


def is_mostly_urdu(text):
    urdu = len(re.findall(r"[؀-ۿ]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return urdu > 0 and urdu > latin


def split_answer(raw):
    """Separate the answer from the suggested follow-up questions."""
    answer, next_questions = raw, []
    if "===NEXT===" in raw:
        answer, tail = raw.split("===NEXT===", 1)
        for line in tail.strip().split("\n"):
            q = re.sub(r"^([-*•]|\d+[.)])\s*", "", line.strip()).strip()
            if 3 < len(q) < 120:
                next_questions.append(q)
    return answer.strip(), next_questions[:3]


def show_answer(text):
    if "===URDU===" in text:
        english, urdu = text.split("===URDU===", 1)
        st.markdown('<div class="lang-label">ENGLISH</div>', unsafe_allow_html=True)
        st.markdown(english.strip())
        st.markdown('<div class="lang-sep"></div><div class="lang-label" style="text-align:right">اردو</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="urdu">{md_to_html(urdu)}</div>', unsafe_allow_html=True)
    elif is_mostly_urdu(text):
        st.markdown(f'<div class="urdu">{md_to_html(text)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(text)


def show_sources(sources, low_confidence):
    if not sources:
        return
    label = "📎 Sources" + (" · general advice (low match)" if low_confidence else "")
    with st.expander(label):
        for s in sources:
            st.markdown(f"**{s['source']} - page {s['page']}** · relevance {s['score']:.2f}")
            st.caption(s["text"][:400] + "...")


def user_bubble(text, time_text):
    with st.chat_message("user", avatar="👨‍🌾"):
        st.markdown('<span class="u-msg"></span>', unsafe_allow_html=True)
        if is_mostly_urdu(text):
            st.markdown(f'<div class="urdu">{md_to_html(text)}</div>', unsafe_allow_html=True)
        else:
            st.markdown(text)
        st.markdown(f'<div class="msg-time">{time_text} ✓✓</div>', unsafe_allow_html=True)


def now_text():
    return datetime.now().strftime("%I:%M %p")


# =====================================================================
# SETTINGS (stored in session so the ⚙️ popover can change them)
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

api_key = get_api_key()

# Which packages are installed? (every one except pypdf has a built-in backup)
import importlib.util
PACKAGES = {"pypdf": "PDF reading (required)", "groq": "Groq SDK (backup: direct API call)",
            "fastembed": "Neural embeddings (backup: TF-IDF)", "faiss": "FAISS vector store (backup: NumPy)",
            "tiktoken": "Token counting (backup: word count)"}


def is_installed(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


installed = {name: is_installed(name) for name in PACKAGES}

# =====================================================================
# CHAT HEADER
# =====================================================================
st.markdown(f"""
<div class="chat-header">
  <div class="logo">🐄</div>
  <div>
    <div class="title">Punjab Livestock Rehnuma</div>
    <div class="sub">پنجاب لائیو سٹاک رہنما · <span class="dot"></span>Online · L&amp;DD Punjab</div>
  </div>
  <a class="call" href="tel:{HELPLINE.replace('-', '')}">📞 {HELPLINE}</a>
</div>
""", unsafe_allow_html=True)

col_settings, col_new, col_dl = st.columns([1.3, 1, 1])

with col_settings.popover("⚙️ Settings", use_container_width=True):
    level = st.radio("Technical level · تکنیکی سطح", list(TECH_LEVELS.keys()), index=0)
    size = st.radio("Response size · جواب کی لمبائی", list(RESPONSE_SIZES.keys()), index=1, horizontal=True)
    language = st.radio("Answer language · جواب کی زبان", ANSWER_LANGUAGES, index=0, horizontal=True)
    model_label = st.selectbox("AI model (open-source, Groq)", list(ANSWER_MODELS.keys()))
    show_src = st.toggle("Show sources", value=True)
    if not api_key:
        api_key = st.text_input("Groq API key", type="password", help="Free key: console.groq.com/keys")
    with st.expander("🛠️ Admin"):
        extra = st.file_uploader("Add extra PDF (this session only)", type=["pdf"],
                                 accept_multiple_files=True)
        status_box = st.container()

if col_new.button("🗑️ New chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

settings = {"level": level, "size": size, "language": language, "model": ANSWER_MODELS[model_label]}

# =====================================================================
# LOAD KNOWLEDGE BASE
# =====================================================================
file_items = []
for path in find_repo_files():
    with open(path, "rb") as f:
        file_items.append((os.path.basename(path), f.read()))
for up in extra or []:
    file_items.append((up.name, up.getvalue()))

kb, setup_problem = None, ""
if not installed["pypdf"]:
    setup_problem = ("⚙️ Setup problem: the Python packages were not installed. In GitHub the file must be "
                     "named exactly **requirements.txt** (with an **s**) next to app.py. Rename it, then on "
                     "Streamlit Cloud click **Manage app → Reboot app**.")
elif not file_items:
    setup_problem = ("No FAQ file found. Add your PDF (and Urdu_FAQ.txt) to the GitHub repo next to app.py, or upload one in "
                     "**⚙️ Settings → Admin**.")
else:
    key = hashlib.md5(b"".join(hashlib.md5(d).digest() for _, d in file_items)).hexdigest()
    try:
        kb = build_knowledge_base(key, tuple(file_items))
        if kb is None:
            setup_problem = "The PDF has no readable text."
    except ModuleNotFoundError as error:
        setup_problem = (f"Missing package: {error.name}. Name the file exactly 'requirements.txt' (with an s), "
                         "next to app.py, then reboot the app.")
    except Exception as error:
        setup_problem = f"Could not build the knowledge base: {error}"

with status_box:
    if kb:
        s = kb["stats"]
        st.caption(f"📚 {s['files']} file(s) · {s['pages_used']}/{s['pages_total']} pages used · "
                   f"{s['chunks']} chunks · {s['tokens']:,} tokens · vector size {s['dimension']} · "
                   f"store: {s['store']} · embeddings: {s['embedder']}")
        st.caption(f"🇵🇰 Urdu FAQ chunks: {s['urdu_chunks']}" +
                   ("" if s["urdu_chunks"] else " - add Urdu_FAQ.txt to the repo for direct Urdu answers"))
        if s["pages_skipped"]:
            st.caption(f"ℹ️ {s['pages_skipped']} PDF page(s) skipped (unreadable Urdu font); "
                       "their content comes from Urdu_FAQ.txt instead.")
    st.markdown("**🩺 System status**")
    for name, purpose in PACKAGES.items():
        st.markdown(f"{'✅' if installed[name] else '⚠️'} **{name}** - {purpose}")

if st.session_state.messages:
    lines = [f"Punjab Livestock Rehnuma chat - {datetime.now():%d %b %Y %H:%M}", f"Helpline: {HELPLINE}", ""]
    for i, m in enumerate(st.session_state.messages):
        who = "Farmer" if m["role"] == "user" else "Punjab Livestock Rehnuma"
        rating = st.session_state.get(f"fb{i}")
        mark = {1: " [👍]", 0: " [👎]"}.get(rating, "")
        lines += [f"{who} ({m.get('time', '')}){mark}:", m["content"].replace("===URDU===", "\n--- اردو ---\n"), ""]
    col_dl.download_button("📥 Save chat", "\n".join(lines), file_name="punjab_livestock_rehnuma_chat.txt",
                           mime="text/plain", use_container_width=True)

if setup_problem:
    st.error(setup_problem)

# =====================================================================
# CHAT
# =====================================================================
pending = None

# Welcome message (always shown first)
with st.chat_message("assistant", avatar="🐄"):
    st.markdown('<span class="b-msg"></span>', unsafe_allow_html=True)
    st.markdown("**Assalam-o-Alaikum! 👋** I am **Punjab Livestock Rehnuma**, the assistant of the "
                "Livestock & Dairy Development Department, Punjab. Ask me about animal diseases, vaccination, "
                "breeding, feeding, the Livestock Card and department services, in English, اردو or Roman Urdu.")
    st.markdown('<div class="urdu">السلام علیکم! میں پنجاب لائیو سٹاک رہنما ہوں۔ جانوروں کی بیماریوں، ویکسینیشن، '
                'نسل کشی، خوراک، لائیو سٹاک کارڈ اور محکمے کی خدمات کے بارے میں سوال پوچھیں۔</div>',
                unsafe_allow_html=True)
    if not st.session_state.messages:
        st.markdown('<div class="lang-label">Tap a question · سوال منتخب کریں</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate(SAMPLE_QUESTIONS):
            if cols[i % 2].button(q, key=f"sample{i}", use_container_width=True):
                pending = q

# Chat history
last = len(st.session_state.messages) - 1
for i, m in enumerate(st.session_state.messages):
    if m["role"] == "user":
        user_bubble(m["content"], m.get("time", ""))
        continue
    with st.chat_message("assistant", avatar="🐄"):
        st.markdown('<span class="b-msg"></span>', unsafe_allow_html=True)
        show_answer(m["content"])
        if show_src:
            show_sources(m.get("sources", []), m.get("low", False))
        c_time, c_fb = st.columns([3, 1])
        c_time.markdown(f'<div class="msg-time">{m.get("time", "")}</div>', unsafe_allow_html=True)
        with c_fb:
            st.feedback("thumbs", key=f"fb{i}")
        if i == last and m.get("next"):
            st.markdown('<div class="lang-label">You can also ask · مزید پوچھیں</div>', unsafe_allow_html=True)
            for j, q in enumerate(m["next"]):
                if st.button(q, key=f"next{i}_{j}"):
                    pending = q

st.markdown(f'<div class="disclaimer">⚠️ General guidance only. For sick animals consult a veterinary doctor '
            f'or call {HELPLINE} · بیمار جانور کے لیے ڈاکٹر سے رجوع کریں</div>', unsafe_allow_html=True)

typed = st.chat_input("Type your question · اپنا سوال لکھیں")
question = typed or pending

if question:
    if not api_key:
        st.error("Groq API key is missing. Add it in Streamlit secrets or ⚙️ Settings.")
        st.stop()

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    asked_at = now_text()
    user_bubble(question, asked_at)

    with st.chat_message("assistant", avatar="🐄"):
        st.markdown('<span class="b-msg"></span>', unsafe_allow_html=True)
        typing = st.empty()
        typing.markdown('<div class="typing"><span></span><span></span><span></span></div>'
                        '<div class="msg-time">Rehnuma is typing… · رہنما لکھ رہا ہے</div>',
                        unsafe_allow_html=True)
        next_questions = []
        try:
            search_query, urdu_query = understand_question(question, history, api_key)
            sources = retrieve([question, search_query, urdu_query], kb) if kb else []
            cutoff = kb["embedder"].min_relevance if kb else MIN_RELEVANCE
            low = (not sources) or sources[0]["score"] < cutoff
            # always keep the best English and best Urdu match, plus any other good match
            top_each = [next((s for s in sources if s["lang"] == lg), None) for lg in ("en", "ur")]
            relevant = [s for s in sources if s in top_each or s["score"] >= cutoff * 0.8]
            raw = answer_question(question, search_query, relevant, history, settings, api_key)
            answer, next_questions = split_answer(raw)
            if not answer:
                answer = f"Sorry, no answer was generated. Please try again or call {HELPLINE}."
        except Exception as error:
            sources, low = [], True
            answer = (f"⚠️ The service is busy or unavailable ({error}). "
                      f"Please try again, or call the Livestock Helpline {HELPLINE}.")
        typing.empty()

    st.session_state.messages.append({"role": "user", "content": question, "time": asked_at})
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources,
                                      "low": low, "time": now_text(), "next": next_questions})
    st.rerun()
