"""
Livestock Helper - Bilingual (English / Urdu) RAG chatbot
Livestock & Dairy Development Department, Punjab

How it works
1. Loads the FAQ PDF(s) placed in the GitHub repo (same folder as app.py, or a data/ folder)
2. Extracts text page by page, skips unreadable pages
3. Splits text into token chunks (tiktoken)
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

st.set_page_config(page_title="Livestock Helper | لائیو سٹاک مددگار",
                   page_icon="🐄", layout="wide")

# =====================================================================
# SETTINGS
# =====================================================================
ANSWER_MODELS = {
    "GPT-OSS 120B (best quality)": "openai/gpt-oss-120b",
    "GPT-OSS 20B (faster)": "openai/gpt-oss-20b",
}
QUERY_MODEL = "openai/gpt-oss-20b"          # fast model for translating the question
EMBED_MODEL = "BAAI/bge-small-en-v1.5"      # free open-source embedding model
CHUNK_TOKENS = 350
CHUNK_OVERLAP = 60
TOP_K = 5
MIN_RELEVANCE = 0.55                        # below this the documents probably lack the answer
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
.urdu {direction: rtl; text-align: right; font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif;
       line-height: 2.3; font-size: 1.05rem;}
.urdu ul, .urdu ol {padding-right: 1.5rem; padding-left: 0;}
.lang-label {font-weight: 600; opacity: 0.75; margin: 0.4rem 0 0.2rem 0;}
.helpline {padding: 0.6rem; border-radius: 0.5rem; background: rgba(34,139,34,0.12); text-align: center;}
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


def find_repo_pdfs():
    folder = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(folder, "*.pdf")) + glob.glob(os.path.join(folder, "data", "*.pdf"))
    return sorted(set(files))


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


class FastEmbedder:
    """Neural embeddings (BAAI/bge-small-en-v1.5) via FastEmbed - best quality."""
    name = "FastEmbed bge-small-en-v1.5"
    min_relevance = MIN_RELEVANCE

    def __init__(self):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=EMBED_MODEL)

    def embed(self, texts):
        return np.array(list(self.model.embed(list(texts))), dtype="float32")


class TfidfEmbedder:
    """Backup embeddings made with NumPy only (TF-IDF of words + word pairs).
    Used automatically if fastembed is not installed or its model cannot download."""
    name = "TF-IDF (NumPy backup)"
    min_relevance = 0.09                      # TF-IDF similarity scores are naturally lower
    STOP = set("a an the of to in on for and or is are was be by with as at it this that from what "
               "how can do does i my me we our you your which who when where will should".split())

    def _terms(self, text):
        words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in self.STOP and len(w) > 1]
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


@st.cache_resource(show_spinner="📚 Building the knowledge base (first visit only)...")
def build_knowledge_base(files_key, file_items):
    """file_items: tuple of (name, bytes). files_key makes the cache rebuild when files change."""
    pages, skipped, total = [], 0, 0
    for name, data in file_items:
        good, bad, count = extract_pages(name, data)
        pages.extend(good)
        skipped += bad
        total += count
    if not pages:
        return None
    chunks = make_chunks(pages)
    texts = [c["text"] for c in chunks]
    embedder = load_fastembed() or TfidfEmbedder().fit(texts)
    vectors = normalise(embedder.embed(texts))
    index, store = make_index(vectors.shape[1])
    index.add(vectors)
    stats = {"files": len(file_items), "pages_total": total, "pages_used": len(pages), "store": store,
             "pages_skipped": skipped, "chunks": len(chunks),
             "tokens": sum(c["tokens"] for c in chunks), "dimension": int(vectors.shape[1]),
             "embedder": embedder.name}
    return {"index": index, "chunks": chunks, "stats": stats, "embedder": embedder}


def retrieve(query, kb, top_k=TOP_K):
    query_vector = normalise(kb["embedder"].embed([query]))
    scores, ids = kb["index"].search(query_vector, min(top_k, len(kb["chunks"])))
    return [{**kb["chunks"][i], "score": float(s)} for s, i in zip(scores[0], ids[0]) if i != -1]


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
question is a follow-up"}"""


def understand_question(question, history, api_key):
    recent = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
    try:
        raw = groq_chat(api_key, QUERY_MODEL, [
            {"role": "system", "content": QUERY_PROMPT},
            {"role": "user", "content": f"CHAT HISTORY:\n{recent}\n\nQUESTION: {question}"},
        ], temperature=0, json_mode=True)
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return data.get("search_query") or question, data.get("language", "unknown")
    except Exception:
        return question, "unknown"


def build_system_prompt(level, size, language):
    if language == "English + اردو":
        lang_rule = ("Write the answer TWICE: first in English, then the same answer in Urdu (Urdu script, "
                     "simple everyday Urdu that Punjab farmers understand). Put a line containing only "
                     "===URDU=== between the English and the Urdu parts. No other headings for languages.")
    elif language == "English only":
        lang_rule = "Write the answer in English only."
    else:
        lang_rule = "Write the answer in simple Urdu script only."

    return f"""You are "Livestock Helper", the official-style assistant of the Livestock & Dairy
Development Department, Punjab, Pakistan. You help livestock farmers with animal health, diseases,
vaccination, breeding and artificial insemination, feeding, dairy, poultry and department services
(Livestock Card, helpline, veterinary hospitals, programmes).

KNOWLEDGE RULES
- Use the CONTEXT from the department's FAQ documents as your main source. Prefer its facts, fees,
  procedures, eligibility and contact details.
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
- Do not add a sources list; the app shows sources separately."""


def answer_question(question, search_query, chunks, history, settings, api_key):
    context = "\n\n".join(f"[{c['source']} - page {c['page']}]\n{c['text']}" for c in chunks) \
        or "No relevant document context found."
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


def show_answer(text):
    if "===URDU===" in text:
        english, urdu = text.split("===URDU===", 1)
        st.markdown('<div class="lang-label">🇬🇧 English</div>', unsafe_allow_html=True)
        st.markdown(english.strip())
        st.markdown('<div class="lang-label" style="text-align:right">اردو 🇵🇰</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="urdu">{md_to_html(urdu)}</div>', unsafe_allow_html=True)
    elif re.search(r"[؀-ۿ]", text) and len(re.findall(r"[A-Za-z]", text)) < len(text) * 0.2:
        st.markdown(f'<div class="urdu">{md_to_html(text)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(text)


def show_sources(sources, low_confidence):
    if not sources:
        return
    label = "📎 Sources from department FAQs" + (" (low match - general advice)" if low_confidence else "")
    with st.expander(label):
        for s in sources:
            st.markdown(f"**{s['source']} - page {s['page']}** · relevance {s['score']:.2f}")
            st.caption(s["text"][:450] + "...")


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("## 🐄 Livestock Helper")
    st.markdown(f'<div class="helpline">📞 <b>Livestock Helpline</b><br>'
                f'<span style="font-size:1.3rem">{HELPLINE}</span><br>24/7 · One Punjab, One Helpline</div>',
                unsafe_allow_html=True)
    st.divider()

    st.subheader("⚙️ Answer settings")
    level = st.radio("Technical level", list(TECH_LEVELS.keys()), index=0)
    size = st.select_slider("Response size", list(RESPONSE_SIZES.keys()), value="Medium")
    language = st.radio("Answer language", ANSWER_LANGUAGES, index=0)
    model_label = st.selectbox("AI model (open-source, Groq)", list(ANSWER_MODELS.keys()))
    show_src = st.toggle("Show sources", value=True)

    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input("Groq API key", type="password", help="Free key: console.groq.com/keys")

    st.divider()
    if st.button("🗑️ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("🛠️ Admin: knowledge base"):
        extra = st.file_uploader("Add extra PDF (this session only)", type=["pdf"],
                                 accept_multiple_files=True)

settings = {"level": level, "size": size, "language": language, "model": ANSWER_MODELS[model_label]}

# =====================================================================
# LOAD KNOWLEDGE BASE
# =====================================================================
file_items = []
for path in find_repo_pdfs():
    with open(path, "rb") as f:
        file_items.append((os.path.basename(path), f.read()))
for up in extra or []:
    file_items.append((up.name, up.getvalue()))

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

kb = None
if not installed["pypdf"]:
    st.error("⚙️ Setup problem: the Python packages were not installed. In GitHub, the file must be "
             "named exactly **requirements.txt** (with an **s**) and be in the main folder next to app.py. "
             "Rename it, then on Streamlit Cloud click **Manage app → Reboot app**.")
elif file_items:
    key = hashlib.md5(b"".join(hashlib.md5(d).digest() for _, d in file_items)).hexdigest()
    try:
        kb = build_knowledge_base(key, tuple(file_items))
    except ModuleNotFoundError as error:
        st.error(f"Missing package: {error.name}. Make sure the file is named exactly "
                 "'requirements.txt' (with an s), is in the main folder next to app.py, "
                 "then reboot the app (Manage app → Reboot).")
    except Exception as error:
        st.error(f"Could not build the knowledge base: {error}")

with st.sidebar:
    if kb:
        s = kb["stats"]
        st.caption(f"📚 {s['files']} file(s) · {s['pages_used']}/{s['pages_total']} pages used · "
                   f"{s['chunks']} chunks · {s['tokens']:,} tokens · vector size {s['dimension']} · "
                   f"store: {s['store']} · embeddings: {s['embedder']}")
        if s["pages_skipped"]:
            st.caption(f"⚠️ {s['pages_skipped']} page(s) skipped: text could not be read correctly "
                       "(usually Urdu fonts). Answers in Urdu are still provided by translation.")
    with st.expander("🩺 System status"):
        for name, purpose in PACKAGES.items():
            st.markdown(f"{'✅' if installed[name] else '⚠️'} **{name}** - {purpose}")
        if not all(installed.values()):
            st.caption("⚠️ = not installed, backup in use. Check that the file is named requirements.txt.")
    st.caption("⚠️ This assistant gives general guidance. For sick animals, always consult a veterinary doctor.")

# =====================================================================
# MAIN PAGE
# =====================================================================
st.title("🐄 Livestock Helper · لائیو سٹاک مددگار")
st.caption("Livestock & Dairy Development Department, Punjab · Ask in English, اردو or Roman Urdu")

if not file_items:
    st.warning("No FAQ PDF found. Add your PDF to the GitHub repo (next to app.py) or upload it in "
               "the sidebar under **Admin: knowledge base**.")
elif kb is None:
    st.warning("The PDF has no readable text.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sample questions on an empty chat
pending = None
if not st.session_state.messages:
    st.markdown("#### 💬 Try a question · سوال پوچھیں")
    cols = st.columns(3)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if cols[i % 3].button(q, key=f"sample{i}", use_container_width=True):
            pending = q

# Chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="👨‍🌾" if m["role"] == "user" else "🐄"):
        if m["role"] == "assistant":
            show_answer(m["content"])
            if show_src:
                show_sources(m.get("sources", []), m.get("low", False))
        else:
            st.markdown(m["content"])

typed = st.chat_input("Type your question · اپنا سوال لکھیں")
question = typed or pending

if question:
    if not api_key:
        st.error("Groq API key is missing. Add it in Streamlit secrets or the sidebar.")
        st.stop()

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👨‍🌾"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🐄"):
        with st.spinner("Searching department FAQs... · معلومات تلاش کی جا رہی ہیں"):
            try:
                search_query, _lang = understand_question(question, history, api_key)
                sources = retrieve(search_query, kb) if kb else []
                cutoff = kb["embedder"].min_relevance if kb else MIN_RELEVANCE
                low = (not sources) or sources[0]["score"] < cutoff
                relevant = [s for s in sources if s["score"] >= cutoff * 0.8]
                answer = answer_question(question, search_query, relevant, history, settings, api_key)
                if not answer.strip():
                    answer = f"Sorry, no answer was generated. Please try again or call {HELPLINE}."
            except Exception as error:
                sources, low = [], True
                answer = (f"⚠️ The service is busy or unavailable ({error}). "
                          f"Please try again, or call the Livestock Helpline {HELPLINE}.")
        show_answer(answer)
        if show_src:
            show_sources(sources, low)

    st.session_state.messages.append({"role": "assistant", "content": answer,
                                      "sources": sources, "low": low})

# Download conversation
if st.session_state.messages:
    lines = [f"Livestock Helper chat - {datetime.now():%d %b %Y %H:%M}", f"Helpline: {HELPLINE}", ""]
    for m in st.session_state.messages:
        who = "Farmer" if m["role"] == "user" else "Livestock Helper"
        lines += [f"{who}:", m["content"].replace("===URDU===", "\n--- اردو ---\n"), ""]
    st.download_button("📥 Download this conversation", "\n".join(lines),
                       file_name="livestock_helper_chat.txt", mime="text/plain")
