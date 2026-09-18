import os
import pickle
from html import escape

import faiss
import numpy as np
import streamlit as st

from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

INDEX_FILE = "faiss_index/index.faiss"
METADATA_FILE = "faiss_index/metadata.pkl"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 6

MIN_RELEVANCE = 0.25

# Number of knowledge categories shown in the navigation bar
# before the rest are collapsed into a "+N" pill.
MAX_NAV_CATEGORIES = 4


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Knowledge Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# HTML HELPERS
# ============================================================
# Streamlit runs st.markdown() through a Markdown parser first.
# In Markdown, a blank line followed by a line indented with
# 4+ spaces becomes a CODE BLOCK. That is exactly why the hero
# HTML was being printed as code. This helper removes all
# indentation and blank lines so HTML is always rendered as HTML.

def clean_html(markup):

    return "\n".join(
        line.strip()
        for line in markup.splitlines()
        if line.strip()
    )


def render_html(markup):

    st.markdown(
        clean_html(markup),
        unsafe_allow_html=True
    )


# ============================================================
# DESIGN SYSTEM
# ============================================================

# Message selectors (used inside the CSS below).
ASSISTANT_MESSAGE = (
    '[data-testid="stChatMessage"]:has('
    '[data-testid="stChatMessageAvatarAssistant"], '
    '[data-testid="chatAvatarIcon-assistant"])'
)

USER_MESSAGE = (
    '[data-testid="stChatMessage"]:has('
    '[data-testid="stChatMessageAvatarUser"], '
    '[data-testid="chatAvatarIcon-user"])'
)

APP_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');


/* ==========================================================
   DESIGN TOKENS
   ========================================================== */

:root {
    --ink: #0C1F2C;
    --petrol: #0F4C52;
    --petrol-deep: #0A3A3F;
    --petrol-soft: #17777C;
    --mist: #F2F5F5;
    --paper: #FFFFFF;
    --line: #DCE4E5;
    --text: #293B46;
    --muted: #62757F;
    --faint: #93A3AB;
    --saffron: #E3A72F;
    --ok: #2FB584;

    --font-ui: 'Plus Jakarta Sans', -apple-system, 'Segoe UI', sans-serif;
    --font-display: 'Newsreader', Georgia, serif;

    --content-width: 1080px;
}


/* ==========================================================
   STREAMLIT CHROME: sidebar, header, toolbar, footer
   ========================================================== */

[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu,
footer {
    display: none !important;
}


/* ==========================================================
   APP CANVAS
   ========================================================== */

.stApp {
    font-family: var(--font-ui);
    color: var(--ink);
    background:
        linear-gradient(180deg, #E8F0F0 0px, var(--mist) 460px),
        var(--mist);
    background-attachment: fixed;
}

[data-testid="stMainBlockContainer"],
.block-container {
    max-width: var(--content-width);
    padding: 6.6rem 20px 9rem;
}

[data-testid="stMarkdownContainer"],
[data-testid="stChatInput"] textarea {
    font-family: var(--font-ui);
}


/* ==========================================================
   NAVIGATION BAR
   ========================================================== */

.nav {
    position: fixed;
    top: 14px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 1000;

    width: min(1040px, calc(100vw - 32px));

    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;

    padding: 9px 12px 9px 10px;

    background: rgba(255, 255, 255, 0.82);
    -webkit-backdrop-filter: blur(18px) saturate(150%);
    backdrop-filter: blur(18px) saturate(150%);

    border: 1px solid rgba(220, 228, 229, 0.95);
    border-radius: 999px;

    box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.9) inset,
        0 14px 40px rgba(12, 31, 44, 0.09);
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 0;
}

.nav-mark {
    width: 40px;
    height: 40px;
    min-width: 40px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 50%;
    background: var(--ink);
    color: var(--saffron);

    font-size: 18px;
    line-height: 1;
}

.nav-text {
    display: flex;
    flex-direction: column;
    line-height: 1.2;
    min-width: 0;
}

.nav-name {
    color: var(--ink);
    font-weight: 700;
    font-size: 0.98rem;
    letter-spacing: -0.015em;
    white-space: nowrap;
}

.nav-sub {
    color: var(--muted);
    font-size: 0.74rem;
    font-weight: 500;
    white-space: nowrap;
}

.nav-categories {
    flex: 1;
    min-width: 0;

    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;

    overflow: hidden;
}

.nav-chip {
    padding: 6px 13px;

    border-radius: 999px;
    background: #EEF3F3;
    border: 1px solid transparent;

    color: var(--petrol);
    font-size: 0.78rem;
    font-weight: 600;
    white-space: nowrap;
}

.nav-chip-more {
    background: transparent;
    border-color: var(--line);
    color: var(--muted);
}

.nav-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;

    padding: 8px 15px;

    border-radius: 999px;
    background: #E6F5EF;

    color: #17694F;
    font-size: 0.78rem;
    font-weight: 700;
    white-space: nowrap;
}

.nav-dot {
    width: 8px;
    height: 8px;

    border-radius: 50%;
    background: var(--ok);

    box-shadow: 0 0 0 3px rgba(47, 181, 132, 0.2);
}


/* ==========================================================
   HERO
   ========================================================== */

.hero {
    position: relative;
    overflow: hidden;

    margin: 6px 0 30px;
    padding: 46px 48px 38px;

    border-radius: 30px;

    background:
        repeating-radial-gradient(
            circle at 100% 0%,
            transparent 0px,
            transparent 46px,
            rgba(255, 255, 255, 0.055) 46px,
            rgba(255, 255, 255, 0.055) 47px
        ),
        linear-gradient(135deg, var(--ink) 0%, var(--petrol-deep) 62%, var(--petrol) 100%);

    box-shadow:
        0 30px 70px rgba(12, 31, 44, 0.20),
        0 2px 6px rgba(12, 31, 44, 0.10);
}

.hero-title {
    max-width: 640px;

    color: #FFFFFF;

    font-family: var(--font-display);
    font-size: clamp(2.1rem, 4.4vw, 3.3rem);
    font-weight: 500;
    line-height: 1.06;
    letter-spacing: -0.025em;
}

.hero-text {
    max-width: 560px;

    margin-top: 16px;

    color: rgba(226, 236, 236, 0.84);

    font-size: 1.02rem;
    line-height: 1.7;
}

.hero-status {
    display: inline-flex;
    align-items: center;
    gap: 9px;

    margin-top: 24px;

    padding: 8px 15px;

    border-radius: 999px;
    background: rgba(47, 181, 132, 0.14);
    border: 1px solid rgba(47, 181, 132, 0.38);

    color: #8FE3C3;
    font-size: 0.8rem;
    font-weight: 600;
}

.hero-status .nav-dot {
    box-shadow: 0 0 0 3px rgba(47, 181, 132, 0.22);
}

.hero-points {
    display: flex;
    flex-wrap: wrap;
    gap: 12px 30px;

    margin-top: 32px;
    padding-top: 22px;

    border-top: 1px solid rgba(255, 255, 255, 0.14);
}

.hero-point {
    display: flex;
    align-items: center;
    gap: 10px;

    color: rgba(226, 236, 236, 0.9);

    font-size: 0.86rem;
    font-weight: 500;
}

.hero-point svg {
    flex-shrink: 0;
    color: var(--saffron);
}


/* ==========================================================
   CHAT MESSAGES
   ========================================================== */

[data-testid="stChatMessage"] {
    gap: 14px;
    align-items: flex-start;

    margin-bottom: 18px;

    color: var(--text);
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
    color: var(--text);

    font-size: 0.96rem;
    line-height: 1.72;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    margin-bottom: 0.7rem;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {
    color: var(--ink);
    font-weight: 700;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h4 {
    margin: 0;
    padding: 0.5rem 0 0.25rem;

    color: var(--ink);

    font-family: var(--font-display);
    font-weight: 600;
    letter-spacing: -0.01em;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1 { font-size: 1.45rem; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2 { font-size: 1.28rem; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3 { font-size: 1.12rem; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h4 { font-size: 1rem; }

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a {
    color: var(--petrol-soft);
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
    padding: 2px 6px;

    border-radius: 6px;
    background: #EAF1F1;

    color: var(--petrol);
    font-size: 0.86em;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] table {
    width: 100%;

    border-collapse: separate;
    border-spacing: 0;

    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;

    font-size: 0.88rem;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th {
    background: #EEF3F3;
    color: var(--ink);
    text-align: left;
    font-weight: 700;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] td {
    padding: 9px 13px;
    border-bottom: 1px solid var(--line);
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] tr:last-child td {
    border-bottom: 0;
}


/* Assistant message */

@ASSISTANT {
    width: 100%;

    padding: 20px 26px !important;

    background: var(--paper) !important;

    border: 1px solid var(--line);
    border-radius: 24px 24px 24px 8px;

    box-shadow:
        0 1px 2px rgba(12, 31, 44, 0.04),
        0 12px 32px rgba(12, 31, 44, 0.06);
}

[data-testid="stChatMessageAvatarAssistant"],
[data-testid="chatAvatarIcon-assistant"] {
    background: var(--ink) !important;
    color: var(--saffron) !important;

    border-radius: 12px !important;
}


/* User message */

@USER {
    width: fit-content;
    max-width: 80%;

    margin-left: auto;

    padding: 14px 22px !important;

    background: var(--petrol) !important;

    border: 0;
    border-radius: 24px 24px 8px 24px;

    box-shadow: 0 10px 26px rgba(15, 76, 82, 0.24);
}

@USER [data-testid="stMarkdownContainer"] p,
@USER [data-testid="stMarkdownContainer"] li {
    margin: 0;

    color: #FFFFFF !important;
}

[data-testid="stChatMessageAvatarUser"],
[data-testid="chatAvatarIcon-user"] {
    display: none !important;
}


/* ==========================================================
   CHAT INPUT
   ========================================================== */

[data-testid="stBottom"] > div,
[data-testid="stBottom"] {
    background: transparent !important;
}

[data-testid="stBottom"]::before {
    content: "";

    position: absolute;
    inset: -48px 0 0 0;

    background: linear-gradient(
        180deg,
        rgba(242, 245, 245, 0) 0%,
        rgba(242, 245, 245, 0.94) 42%,
        var(--mist) 100%
    );

    pointer-events: none;
}

[data-testid="stBottomBlockContainer"] {
    position: relative;

    max-width: var(--content-width);

    padding: 0 20px 1.7rem;
}

[data-testid="stChatInput"] {
    padding: 0;
    background: transparent !important;
}

[data-testid="stChatInput"] > div {
    padding: 6px 8px 6px 14px;

    background: var(--paper) !important;

    border: 1px solid var(--line) !important;
    border-radius: 24px !important;

    box-shadow:
        0 1px 2px rgba(12, 31, 44, 0.05),
        0 16px 44px rgba(12, 31, 44, 0.11) !important;

    transition:
        border-color 0.2s ease,
        box-shadow 0.2s ease;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--petrol) !important;

    box-shadow:
        0 0 0 4px rgba(15, 76, 82, 0.13),
        0 16px 44px rgba(12, 31, 44, 0.11) !important;
}

[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"] {
    background: transparent !important;
    border: 0 !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;

    color: var(--ink) !important;

    font-size: 0.98rem;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--faint);
    opacity: 1;
}

[data-testid="stChatInput"] button {
    width: 42px;
    height: 42px;

    background: var(--petrol) !important;
    color: #FFFFFF !important;

    border: 0 !important;
    border-radius: 16px !important;

    box-shadow: 0 8px 20px rgba(15, 76, 82, 0.32);

    transition:
        background 0.15s ease,
        transform 0.15s ease;
}

[data-testid="stChatInput"] button:hover:not(:disabled) {
    background: var(--petrol-deep) !important;
    transform: translateY(-1px);
}

[data-testid="stChatInput"] button:focus-visible {
    outline: 3px solid rgba(227, 167, 47, 0.7);
    outline-offset: 2px;
}

[data-testid="stChatInput"] button:disabled {
    background: #E4EBEC !important;
    color: var(--faint) !important;

    box-shadow: none;
}


/* ==========================================================
   SOURCES
   ========================================================== */

[data-testid="stExpander"] {
    border: 0 !important;

    background: transparent !important;
}

[data-testid="stExpander"] details {
    margin-top: 6px;

    background: #F7FAFA !important;

    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
}

[data-testid="stExpander"] summary {
    color: var(--ink);

    font-size: 0.86rem;
    font-weight: 600;
}

.source-card {
    margin: 10px 0;
    padding: 16px 18px;

    background: var(--paper);

    border: 1px solid var(--line);
    border-radius: 16px;

    box-shadow: 0 4px 14px rgba(12, 31, 44, 0.04);
}

.source-top {
    display: flex;
    align-items: center;
    gap: 12px;
}

.source-index {
    width: 26px;
    height: 26px;
    min-width: 26px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 50%;
    background: var(--ink);
    color: var(--saffron);

    font-size: 0.74rem;
    font-weight: 700;
}

.source-file {
    flex: 1;
    min-width: 0;

    color: var(--ink);

    font-size: 0.92rem;
    font-weight: 700;

    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.source-score {
    color: var(--petrol);

    font-size: 0.78rem;
    font-weight: 700;
    white-space: nowrap;
}

.source-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;

    margin-top: 12px;
}

.source-chip {
    padding: 4px 11px;

    border-radius: 999px;
    background: #EEF3F3;

    color: var(--muted);
    font-size: 0.75rem;
    font-weight: 600;
}

.score-track {
    height: 4px;

    margin-top: 14px;

    border-radius: 999px;
    background: #E4EBEC;

    overflow: hidden;
}

.score-fill {
    height: 100%;

    border-radius: 999px;
    background: var(--saffron);
}

.source-excerpt {
    margin-top: 13px;

    color: var(--muted);

    font-size: 0.84rem;
    line-height: 1.65;
}


/* ==========================================================
   FOOTNOTE, SPINNER, SCROLLBAR
   ========================================================== */

.footnote {
    margin-top: 26px;

    color: var(--faint);

    font-size: 0.76rem;
    text-align: center;
}

[data-testid="stSpinner"] {
    color: var(--muted);
}

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: #C5D2D4;

    border-radius: 999px;
}

::-webkit-scrollbar-thumb:hover {
    background: #9FB1B5;
}


/* ==========================================================
   RESPONSIVE
   ========================================================== */

@media (max-width: 980px) {

    .nav-categories {
        display: none;
    }

    .nav {
        justify-content: space-between;
    }
}

@media (max-width: 640px) {

    [data-testid="stMainBlockContainer"],
    .block-container {
        padding: 5.6rem 14px 9rem;
    }

    .nav-sub {
        display: none;
    }

    .hero {
        padding: 30px 24px 26px;

        border-radius: 24px;
    }

    .hero-points {
        flex-direction: column;
        gap: 12px;
    }

    @USER {
        max-width: 94%;
    }

    @ASSISTANT {
        padding: 16px 18px !important;
    }
}

@media (prefers-reduced-motion: reduce) {

    * {
        transition: none !important;
    }
}

</style>
"""

render_html(
    APP_CSS
    .replace("@ASSISTANT", ASSISTANT_MESSAGE)
    .replace("@USER", USER_MESSAGE)
)


# ============================================================
# LOAD RESOURCES
# ============================================================

@st.cache_resource
def load_resources():

    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(
            "FAISS index not found."
        )

    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(
            "Metadata file not found."
        )

    if "GROQ_API_KEY" not in st.secrets:
        raise ValueError(
            "GROQ_API_KEY is missing "
            "from Streamlit Secrets."
        )

    index = faiss.read_index(
        INDEX_FILE
    )

    with open(
        METADATA_FILE,
        "rb"
    ) as file:

        metadata = pickle.load(
            file
        )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    groq_client = Groq(
        api_key=st.secrets[
            "GROQ_API_KEY"
        ]
    )

    return (
        index,
        metadata,
        embedding_model,
        groq_client
    )


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_documents(
    question,
    index,
    metadata,
    embedding_model,
    top_k=TOP_K
):

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position < 0:
            continue

        item = metadata[
            index_position
        ].copy()

        item["score"] = float(score)

        if item["score"] >= MIN_RELEVANCE:

            results.append(item)

    return results


# ============================================================
# BUILD LLM CONTEXT
# ============================================================

def build_context(results):

    context = []

    for number, item in enumerate(
        results,
        start=1
    ):

        context.append(
            f"""
SOURCE {number}

Document:
{item['source_file']}

Department:
{item['department']}

Page:
{item['page']}

Version:
{item.get('version', 'Unknown')}

Effective Date:
{item.get('effective_date', 'Unknown')}

Content:
{item['text']}
"""
        )

    return "\n".join(context)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    results,
    groq_client
):

    context = build_context(
        results
    )

    system_prompt = """
You are an enterprise knowledge and
policy assistant.

You must answer ONLY using the
provided knowledge-base sources.

Rules:

1. Never invent company policies.

2. Never use outside knowledge as though
   it came from the company's documents.

3. If the retrieved sources do not contain
   enough information, clearly say that the
   knowledge base does not contain enough
   information to answer.

4. Cite relevant source documents using
   their exact filename and page number.

5. Pay close attention to:
   - dates
   - versions
   - limits
   - requirements
   - exceptions
   - approval conditions

6. If documents conflict, explicitly identify
   the conflict.

7. Never silently choose one conflicting
   policy over another.

8. Distinguish mandatory policy requirements
   from recommendations or procedures.

9. Do not invent missing information.

10. Keep the answer professional and concise.
"""

    user_prompt = f"""
USER QUESTION:

{question}


KNOWLEDGE BASE:

{context}


Answer the user's question using ONLY
the knowledge base above.

If there is a conflict between documents,
explain the conflict and identify each
relevant document.
"""

    response = (
        groq_client
        .chat
        .completions
        .create(
            model=GROQ_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0.1,

            max_tokens=1400
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# RENDER SOURCE CARDS
# ============================================================

def render_sources(sources):

    cards = []

    for number, source in enumerate(
        sources,
        start=1
    ):

        excerpt = (
            source["text"]
            .replace("\n", " ")
        )

        if len(excerpt) > 300:

            excerpt = (
                excerpt[:300]
                + "..."
            )

        score = source["score"]

        percent = max(
            0,
            min(100, round(score * 100))
        )

        version = source.get(
            "version",
            "Unknown"
        )

        cards.append(
            f"""
<div class="source-card">
<div class="source-top">
<div class="source-index">{number}</div>
<div class="source-file">📄 {escape(str(source['source_file']))}</div>
<div class="source-score">Relevance {score:.3f}</div>
</div>
<div class="source-meta">
<span class="source-chip">{escape(str(source['department']))}</span>
<span class="source-chip">Page {escape(str(source['page']))}</span>
<span class="source-chip">Version {escape(str(version))}</span>
</div>
<div class="score-track">
<div class="score-fill" style="width: {percent}%"></div>
</div>
<div class="source-excerpt">{escape(excerpt)}</div>
</div>
"""
        )

    render_html(
        "\n".join(cards)
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    (
        index,
        metadata,
        embedding_model,
        groq_client
    ) = load_resources()

except Exception as error:

    st.error(
        f"Knowledge base error: {error}"
    )

    st.stop()


# ============================================================
# NAVIGATION BAR
# ============================================================

departments = sorted(
    {
        str(
            item.get(
                "department",
                "General"
            )
        )
        for item in metadata
    }
)

category_chips = "".join(
    f'<span class="nav-chip">{escape(department)}</span>'
    for department in departments[:MAX_NAV_CATEGORIES]
)

if len(departments) > MAX_NAV_CATEGORIES:

    category_chips += (
        '<span class="nav-chip nav-chip-more">'
        f"+{len(departments) - MAX_NAV_CATEGORIES}"
        "</span>"
    )

render_html(
    f"""
<div class="nav">
<div class="nav-brand">
<div class="nav-mark">✦</div>
<div class="nav-text">
<span class="nav-name">Knowledge Intelligence</span>
<span class="nav-sub">Enterprise knowledge assistant</span>
</div>
</div>
<div class="nav-categories">{category_chips}</div>
<div class="nav-status">
<span class="nav-dot"></span>
Connected
</div>
</div>
"""
)


# ============================================================
# HERO
# ============================================================

CHECK_ICON = (
    '<svg width="18" height="18" viewBox="0 0 20 20" fill="none">'
    '<circle cx="10" cy="10" r="9" stroke="currentColor" stroke-width="1.5"/>'
    '<path d="M6 10.5l2.5 2.5L14 7.5" stroke="currentColor" '
    'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>'
    "</svg>"
)

render_html(
    f"""
<div class="hero">
<div class="hero-title">Enterprise Knowledge Assistant</div>
<div class="hero-text">
Search company policies, procedures, compliance documentation and internal knowledge with grounded AI.
</div>
<div class="hero-status">
<span class="nav-dot"></span>
Knowledge base online
</div>
<div class="hero-points">
<div class="hero-point">{CHECK_ICON}Answers come only from your indexed documents</div>
<div class="hero-point">{CHECK_ICON}Every answer cites the file and page</div>
<div class="hero-point">{CHECK_ICON}Conflicting policies are flagged, not silently resolved</div>
</div>
</div>
"""
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                f"View retrieved sources ({len(message['sources'])})"
            ):

                render_sources(
                    message["sources"]
                )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask about your company's policies, procedures or knowledge..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(
            question
        )

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching knowledge base..."
        ):

            results = retrieve_documents(
                question,
                index,
                metadata,
                embedding_model
            )

        if not results:

            answer = (
                "I couldn't find sufficiently "
                "relevant information in the "
                "knowledge base to answer "
                "that question."
            )

        else:

            with st.spinner(
                "Generating grounded answer..."
            ):

                answer = generate_answer(
                    question,
                    results,
                    groq_client
                )

        st.markdown(
            answer
        )

        if results:

            with st.expander(
                f"View retrieved sources ({len(results)})"
            ):

                render_sources(
                    results
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": results
        }
    )


# ============================================================
# FOOTNOTE
# ============================================================

render_html(
    """
<div class="footnote">
Answers are grounded in the indexed enterprise knowledge base.
</div>
"""
)
