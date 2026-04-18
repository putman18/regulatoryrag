"""
app.py - RegulatoryRAG: AI Document Q&A for Life Sciences

Upload any regulatory PDF and ask plain-English questions.
Claude answers only from the document — no hallucination from training data.
Every answer includes page number citations.

Run:
    streamlit run app.py
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Streamlit Cloud injects secrets via st.secrets — set env var if present
import streamlit as _st_early
if "ANTHROPIC_API_KEY" in _st_early.secrets:
    os.environ["ANTHROPIC_API_KEY"] = _st_early.secrets["ANTHROPIC_API_KEY"]

try:
    import streamlit as st
except ImportError:
    print("Missing streamlit. Run: pip install streamlit")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("Missing anthropic. Run: pip install anthropic")
    sys.exit(1)

try:
    import fitz
except ImportError:
    print("Missing pymupdf. Run: pip install pymupdf")
    sys.exit(1)

from ingest import build_index, retrieve, load_index, DEFAULT_PDF, INDEX_PATH, MIN_CHARS

MIN_CHARS = 500

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

SAMPLE_QUESTIONS = [
    "What is a Warning Letter and when is it issued?",
    "What are the key components required in a Warning Letter?",
    "How long does a company have to respond to a Warning Letter?",
    "What is the difference between a Warning Letter and an Untitled Letter?",
]

st.set_page_config(
    page_title="RegulatoryRAG",
    page_icon="📋",
    layout="centered",
)

st.markdown("""
<style>
    .main { max-width: 800px; }
    .citation { background: #1a1a2e; border-left: 3px solid #4a90e2;
                padding: 8px 14px; border-radius: 4px; font-size: 0.85em;
                color: #a0b4d0; margin-top: 8px; }
    .answer-box { background: #0f1117; border: 1px solid #2a2a3e;
                  border-radius: 8px; padding: 16px; margin: 12px 0; }
    .warning { background: #2a1a1a; border-left: 3px solid #e24a4a;
               padding: 8px 14px; border-radius: 4px; color: #d08080; }
</style>
""", unsafe_allow_html=True)

st.title("📋 RegulatoryRAG")
st.caption("AI Document Q&A for Life Sciences — answers grounded in your document only")


def get_index(pdf_bytes=None, pdf_name=None):
    """Load existing index or build from uploaded PDF."""
    if pdf_bytes is not None:
        import tempfile
        tmp_path = Path(tempfile.gettempdir()) / pdf_name
        tmp_path.write_bytes(pdf_bytes)
        return build_index(tmp_path)
    existing = load_index()
    if existing:
        return existing
    return build_index(DEFAULT_PDF)


def ask_claude(question: str, chunks: list[dict], pdf_name: str) -> tuple[str, list[dict]]:
    """Send question + retrieved chunks to Claude. Returns answer + source chunks."""
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Source {i+1} — Page {chunk['page']}]\n{chunk['text']}\n"

    # Log chunks for every query so we can see what Claude actually receives
    st.session_state.setdefault("last_chunks_debug", [])
    st.session_state["last_chunks_debug"] = [
        {"page": c["page"], "preview": c["text"][:200]} for c in chunks
    ]

    system_prompt = (
        "You are a regulatory document assistant. Your job is to find and explain "
        "information from the document excerpts provided to you.\n\n"
        "RULES — follow these exactly:\n"
        "1. Read ALL excerpts carefully before answering.\n"
        "2. The user's question may be informal, vague, or use different words than "
        "the document — that is fine. Look for the underlying intent and match it to "
        "relevant content in the excerpts.\n"
        "3. If an excerpt contains relevant information — even partial — use it. "
        "Quote or paraphrase the relevant text directly in your answer.\n"
        "4. Interpret regulatory language: if a document says something is 'prohibited', "
        "'not permitted', 'shall not', or 'must not' — that means it is NOT allowed. "
        "Say so clearly in plain English.\n"
        "5. Only say you cannot find the answer if NONE of the excerpts contain "
        "anything remotely related to the question after careful reading.\n"
        "6. End every answer with the page numbers you used: (Source: Page X, Page Y)"
    )

    user_prompt = (
        f"Document: {pdf_name}\n\n"
        f"Excerpts from the document:\n{context}\n\n"
        f"Question: {question}"
    )

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    answer = resp.content[0].text

    # Log a warning in session state if Claude still said cannot find
    if "could not find" in answer.lower() or "cannot find" in answer.lower():
        st.session_state["retrieval_miss"] = {
            "question": question,
            "chunks": st.session_state["last_chunks_debug"],
        }

    return answer, chunks


# --- Sidebar ---
with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload a regulatory PDF", type=["pdf"])
    st.divider()
    st.markdown("**Loaded document:**")

    if "index" not in st.session_state:
        with st.spinner("Loading FDA guidance document..."):
            try:
                st.session_state.index = get_index()
                st.session_state.pdf_name = st.session_state.index["pdf_name"]
            except ValueError as e:
                st.markdown(f'<div class="warning">⚠️ {e}</div>', unsafe_allow_html=True)
                st.stop()

    if uploaded:
        pdf_bytes = uploaded.read()
        # Guard: check if PDF is extractable
        import fitz as _fitz
        doc = _fitz.open(stream=pdf_bytes, filetype="pdf")
        total_text = " ".join(p.get_text() for p in doc)
        if len(total_text.strip()) < MIN_CHARS:
            st.markdown(
                '<div class="warning">⚠️ Could not extract text from this PDF. '
                'It may be scanned or image-based. Please use a text-based PDF.</div>',
                unsafe_allow_html=True
            )
        else:
            with st.spinner("Ingesting document..."):
                st.session_state.index = get_index(pdf_bytes, uploaded.name)
                st.session_state.pdf_name = uploaded.name
                st.session_state.messages = []

    idx = st.session_state.get("index", {})
    st.markdown(f"**{idx.get('pdf_name', 'None')}**")
    st.caption(f"{idx.get('total_pages', '?')} pages · {idx.get('chunk_count', '?')} chunks")
    st.divider()
    st.markdown("**Try a sample question:**")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.sample_q = q

# --- Main chat area ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            pages = sorted(set(c["page"] for c in msg["citations"]))
            page_str = ", ".join(f"Page {p}" for p in pages)
            st.markdown(f'<div class="citation">📄 Sources: {page_str} — {idx.get("pdf_name","")}</div>',
                        unsafe_allow_html=True)

# Handle sample question button
prefill = st.session_state.pop("sample_q", None)

question = st.chat_input("Ask a question about the document...") or prefill

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            try:
                index = st.session_state.index
                chunks = retrieve(question, index, top_k=6)
                answer, sources = ask_claude(question, chunks, idx.get("pdf_name", ""))
                st.markdown(answer)
                pages = sorted(set(c["page"] for c in sources))
                page_str = ", ".join(f"Page {p}" for p in pages)
                st.markdown(
                    f'<div class="citation">📄 Sources: {page_str} — {idx.get("pdf_name","")}</div>',
                    unsafe_allow_html=True
                )
                # Debug expander — shows raw chunks Claude received
                with st.expander("🔍 Debug: chunks retrieved", expanded=False):
                    for d in st.session_state.get("last_chunks_debug", []):
                        st.markdown(f"**Page {d['page']}:** {d['preview']}...")
                    if st.session_state.get("retrieval_miss"):
                        st.warning("Claude said 'cannot find' but chunks were retrieved — likely a retrieval or prompt mismatch.")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": sources,
                })
            except anthropic.APIConnectionError:
                st.markdown(
                    '<div class="warning">⚠️ Could not reach the Anthropic API. '
                    'Check your internet connection and try again.</div>',
                    unsafe_allow_html=True
                )
            except anthropic.RateLimitError:
                st.markdown(
                    '<div class="warning">⚠️ API rate limit hit. Wait a moment and try again.</div>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.markdown(
                    f'<div class="warning">⚠️ Error: {str(e)}</div>',
                    unsafe_allow_html=True
                )
