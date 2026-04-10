import streamlit as st
import os
import re
import time
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="METU IE Summer Practice Assistant",
    page_icon="🎓",
    layout="centered",
)

# ── Constants ────────────────────────────────────────────────
APP_DIR = Path(__file__).parent

# Look for knowledge base files: first try knowledge_base/ subfolder, then root
KB_DIR = APP_DIR / "knowledge_base"
if not KB_DIR.exists() or not list(KB_DIR.glob("*.txt")):
    KB_DIR = APP_DIR

SYSTEM_INSTRUCTION = """You are the official METU Industrial Engineering Summer Practice Assistant.
Your sole purpose is to help students with questions about METU IE Summer Practice (IE 300 and IE 400).

RULES:
1. Answer ONLY based on the provided context. Do not make up information.
2. If the context contains the answer, provide a clear, helpful, and complete response.
3. If the question is outside the scope of METU IE Summer Practice (e.g., weather, sports, unrelated courses), 
   politely decline by saying: "This question is outside the scope of METU IE Summer Practice. 
   For more information, please visit https://sp-ie.metu.edu.tr/en or contact the SP Committee at ie-staj@metu.edu.tr."
4. When referencing documents or forms, guide students to the Documents/Forms page: https://sp-ie.metu.edu.tr/en/forms
5. Be concise but thorough. Use bullet points when listing multiple items.
6. You can respond in both English and Turkish — match the language of the student's question.
7. If you're not sure about something, say so honestly and direct them to the SP Committee."""


# ── Text chunking ────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# ── Get knowledge base txt files ─────────────────────────────
def get_kb_files() -> list[Path]:
    pattern = re.compile(r"^\d{2}_.*\.txt$")
    return sorted([f for f in KB_DIR.iterdir() if f.is_file() and pattern.match(f.name)])


# ── Build knowledge base with TF-IDF (ZERO API calls) ───────
@st.cache_resource(show_spinner="📚 Bilgi tabanı yükleniyor / Loading knowledge base...")
def build_knowledge_base():
    """Load text files, chunk them, build TF-IDF index. No API needed."""
    all_chunks = []
    all_sources = []

    kb_files = get_kb_files()
    if not kb_files:
        raise ValueError(f"No knowledge base files found in {KB_DIR}")

    for txt_file in kb_files:
        text = txt_file.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for c in chunks:
            all_chunks.append(c)
            all_sources.append(txt_file.name)

    if not all_chunks:
        raise ValueError("Knowledge base is empty.")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(all_chunks)

    return all_chunks, all_sources, vectorizer, tfidf_matrix


def retrieve(query, chunks, sources, vectorizer, tfidf_matrix, top_k=5):
    """Find most similar chunks using TF-IDF cosine similarity. No API needed."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {"text": chunks[i], "source": sources[i], "score": float(scores[i])}
        for i in top_indices if scores[i] > 0
    ]


def ask_gemini(question, context, chat_history, api_key):
    """Generate answer using Gemini API. Only 1 API call per question."""
    client = genai.Client(api_key=api_key)

    history_text = ""
    for msg in chat_history[-6:]:
        role = "Student" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""{SYSTEM_INSTRUCTION}

CONTEXT FROM KNOWLEDGE BASE:
---
{context}
---

PREVIOUS CONVERSATION:
{history_text}

STUDENT'S QUESTION: {question}

Provide a helpful answer based on the context above."""

    # Retry logic for rate limits
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 30 * (attempt + 1)
                time.sleep(wait)
            else:
                raise e
    raise Exception("API rate limit exceeded. Please try again in a minute.")


# ── UI ───────────────────────────────────────────────────────
st.title("🎓 METU IE Summer Practice Assistant")
st.caption(
    "Ask me anything about IE 300 & IE 400 Summer Practice — procedures, documents, "
    "deadlines, report guidelines, and more.  \n"
    "Kaynak: [sp-ie.metu.edu.tr](https://sp-ie.metu.edu.tr/en)"
)

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = ""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Google Gemini API Key", type="password", placeholder="AIza...")
        st.caption("🔒 API anahtarınız sunucuda saklanmaz.")
        st.markdown("[Ücretsiz API Key al →](https://aistudio.google.com/apikey)")

    st.divider()
    st.markdown("""
**Useful Links**
- [SP Website](https://sp-ie.metu.edu.tr/en)
- [General Information](https://sp-ie.metu.edu.tr/en/general-information)
- [Steps to Follow](https://sp-ie.metu.edu.tr/en/steps-follow)
- [Documents / Forms](https://sp-ie.metu.edu.tr/en/forms)
- [FAQ](https://sp-ie.metu.edu.tr/en/faq)
- [SP Opportunities](https://sp-ie.metu.edu.tr/en/sp-opportunities)
    """)
    st.divider()
    st.markdown("""
**SP Committee Contact**  
📧 ie-staj@metu.edu.tr  
📧 sp-belge@metu.edu.tr *(for evaluation forms)*
    """)
    if st.button("🗑️ Sohbeti Temizle / Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if not api_key:
    st.info("👈 Lütfen sol panelden Google Gemini API anahtarınızı girin.\n\n"
            "Please enter your Google Gemini API key in the sidebar.\n\n"
            "[Ücretsiz key al / Get free key →](https://aistudio.google.com/apikey)")
    st.stop()

# ── Build knowledge base (NO API calls here!) ────────────────
try:
    chunks, sources, vectorizer, tfidf_matrix = build_knowledge_base()
except Exception as e:
    st.error(f"Bilgi tabanı yüklenirken hata oluştu: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.messages:
    st.markdown("##### 💡 Örnek Sorular / Sample Questions")
    cols = st.columns(2)
    sample_questions = [
        "What are the prerequisites for IE 300?",
        "SGK sigortası için ne zaman başvurmalıyım?",
        "What companies are available for summer practice?",
        "IE 400 raporunda hangi bölümler var?",
    ]
    for i, q in enumerate(sample_questions):
        if cols[i % 2].button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input("Sorunuzu yazın / Type your question...")
question = pending or user_input

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Düşünüyorum... / Thinking..."):
            try:
                # 1. Retrieve with TF-IDF (no API call)
                retrieved = retrieve(question, chunks, sources, vectorizer, tfidf_matrix, top_k=5)
                context = "\n\n---\n\n".join(
                    [f"[Source: {r['source']}]\n{r['text']}" for r in retrieved]
                )

                # 2. Generate answer with Gemini (1 API call)
                answer = ask_gemini(question, context, st.session_state.messages, api_key)
                st.markdown(answer)

                source_files = sorted(set(r["source"] for r in retrieved))
                with st.expander("📄 Kaynaklar / Sources"):
                    for s in source_files:
                        st.caption(f"• {s}")

            except Exception as e:
                answer = f"Bir hata oluştu / An error occurred: {str(e)}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
