import os
import pickle

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


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Knowledge Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# ENTERPRISE UI
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
   GLOBAL APP
   ========================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 0% 0%,
            rgba(59, 130, 246, 0.055),
            transparent 25%
        ),
        radial-gradient(
            circle at 100% 0%,
            rgba(99, 102, 241, 0.045),
            transparent 25%
        ),
        linear-gradient(
            180deg,
            #f8fafc 0%,
            #f5f7fb 45%,
            #f8fafc 100%
        );

    color: #0f172a;
}


/* ==========================================================
   MAIN CONTAINER
   ========================================================== */

.block-container {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 5rem;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    border-right: 1px solid #e2e8f0;
}

section[data-testid="stSidebar"] > div {
    background:
        linear-gradient(
            180deg,
            #ffffff 0%,
            #f8fafc 100%
        );
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.7rem;
}


/* ==========================================================
   BRAND
   ========================================================== */

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 7px;
}

.brand-mark {
    width: 42px;
    height: 42px;
    min-width: 42px;

    border-radius: 12px;

    display: flex;
    align-items: center;
    justify-content: center;

    background:
        linear-gradient(
            135deg,
            #0f172a 0%,
            #1e293b 55%,
            #334155 100%
        );

    color: #ffffff;

    font-size: 20px;
    font-weight: 600;

    box-shadow:
        0 8px 22px rgba(15, 23, 42, 0.16);
}

.brand-name {
    font-size: 1.05rem;
    font-weight: 750;

    color: #0f172a;

    letter-spacing: -0.025em;
}


/* ==========================================================
   SIDEBAR LABELS
   ========================================================== */

.small-label {
    color: #64748b;

    font-size: 0.72rem;

    text-transform: uppercase;

    letter-spacing: 0.11em;

    font-weight: 750;

    margin-bottom: 9px;
}


/* ==========================================================
   SIDEBAR CONNECTED CARD
   ========================================================== */

section[data-testid="stSidebar"] .stAlert {
    border-radius: 12px;

    border: 1px solid #bbf7d0;

    background:
        linear-gradient(
            135deg,
            #f0fdf4,
            #ecfdf5
        );

    color: #166534;
}


/* ==========================================================
   SIDEBAR METRIC
   ========================================================== */

section[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: #ffffff;

    border: 1px solid #e2e8f0;

    border-radius: 14px;

    padding: 12px 14px;

    margin-top: 12px;

    box-shadow:
        0 4px 14px rgba(15, 23, 42, 0.035);
}

section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #64748b;
}

section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #0f172a;
}


/* ==========================================================
   HERO
   ========================================================== */

.hero {
    position: relative;

    overflow: hidden;

    padding: 38px 40px;

    margin: 12px 0 28px;

    border: 1px solid #e2e8f0;

    border-radius: 22px;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.97) 0%,
            rgba(248,250,252,0.94) 55%,
            rgba(239,246,255,0.92) 100%
        );

    box-shadow:
        0 20px 55px rgba(15, 23, 42, 0.065),
        0 2px 8px rgba(15, 23, 42, 0.025);
}


/* subtle enterprise accent */

.hero::before {
    content: "";

    position: absolute;

    width: 260px;
    height: 260px;

    right: -90px;
    top: -120px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(59,130,246,0.10),
            transparent 68%
        );

    pointer-events: none;
}

.hero::after {
    content: "";

    position: absolute;

    width: 180px;
    height: 180px;

    left: -80px;
    bottom: -110px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(99,102,241,0.07),
            transparent 70%
        );

    pointer-events: none;
}


.hero h1 {
    position: relative;

    margin: 0;

    color: #0f172a;

    font-size: 2.45rem;

    line-height: 1.12;

    font-weight: 750;

    letter-spacing: -0.045em;
}


.hero p {
    position: relative;

    margin-top: 11px;
    margin-bottom: 0;

    color: #64748b;

    font-size: 1rem;

    line-height: 1.65;

    max-width: 720px;
}


/* ==========================================================
   STATUS PILL
   ========================================================== */

.status {
    position: relative;

    display: inline-flex;

    align-items: center;

    gap: 8px;

    margin-top: 19px;

    padding: 7px 12px;

    border-radius: 999px;

    background:
        rgba(236, 253, 245, 0.9);

    border:
        1px solid #bbf7d0;

    color: #047857;

    font-size: 0.78rem;

    font-weight: 650;
}

.dot {
    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #10b981;

    box-shadow:
        0 0 0 3px rgba(16,185,129,0.12);
}


/* ==========================================================
   CHAT AREA
   ========================================================== */

[data-testid="stChatMessage"] {
    border-radius: 16px;

    margin-bottom: 10px;
}


/* Assistant message */

[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-assistant"]
) {
    background: rgba(255,255,255,0.68);

    border: 1px solid rgba(226,232,240,0.75);

    box-shadow:
        0 5px 20px rgba(15,23,42,0.025);
}


/* ==========================================================
   CHAT INPUT
   ========================================================== */

[data-testid="stChatInput"] {
    padding-top: 8px;
}

[data-testid="stChatInput"] > div {
    border-radius: 16px !important;

    border: 1px solid #dbe3ee !important;

    background: rgba(255,255,255,0.94) !important;

    box-shadow:
        0 10px 30px rgba(15,23,42,0.055) !important;

    transition:
        border-color 0.2s ease,
        box-shadow 0.2s ease;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: #94a3b8 !important;

    box-shadow:
        0 0 0 3px rgba(59,130,246,0.07),
        0 10px 30px rgba(15,23,42,0.055) !important;
}


/* ==========================================================
   SOURCE CARDS
   ========================================================== */

.source-card {
    padding: 16px 18px;

    border-radius: 14px;

    border: 1px solid #e2e8f0;

    background:
        linear-gradient(
            135deg,
            #ffffff,
            #f8fafc
        );

    margin: 9px 0;

    box-shadow:
        0 5px 18px rgba(15,23,42,0.035);

    transition:
        transform 0.15s ease,
        box-shadow 0.15s ease;
}

.source-card:hover {
    transform: translateY(-1px);

    box-shadow:
        0 8px 24px rgba(15,23,42,0.065);
}

.source-file {
    font-weight: 700;

    color: #0f172a;

    font-size: 0.91rem;
}

.source-meta {
    color: #64748b;

    font-size: 0.78rem;

    margin-top: 6px;

    line-height: 1.5;
}

.source-excerpt {
    color: #475569;

    font-size: 0.82rem;

    line-height: 1.55;

    margin-top: 9px;
}


/* ==========================================================
   EXPANDER
   ========================================================== */

[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;

    border-radius: 13px !important;

    background: rgba(255,255,255,0.72) !important;
}


/* ==========================================================
   DIVIDERS
   ========================================================== */

hr {
    border-color: #e2e8f0 !important;
}


/* ==========================================================
   CAPTIONS
   ========================================================== */

.stCaption {
    color: #64748b !important;
}


/* ==========================================================
   SCROLLBAR
   ========================================================== */

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: #cbd5e1;

    border-radius: 999px;
}

::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}


/* ==========================================================
   MOBILE
   ========================================================== */

@media (max-width: 768px) {

    .block-container {
        padding-top: 1rem;
    }

    .hero {
        padding: 27px 24px;

        border-radius: 18px;
    }

    .hero h1 {
        font-size: 1.85rem;
    }

    .hero p {
        font-size: 0.92rem;
    }

}

</style>
""",
    unsafe_allow_html=True
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
# SIDEBAR
# ============================================================

with st.sidebar:

    # IMPORTANT:
    # HTML is intentionally NOT indented.
    # Indented HTML can be rendered as a
    # Markdown code block by Streamlit.

    st.markdown(
        """
<div class="brand">
    <div class="brand-mark">✦</div>
    <div class="brand-name">
        Knowledge Intelligence
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="small-label">Knowledge Base</div>',
        unsafe_allow_html=True
    )

    st.success(
        "Connected"
    )

    st.metric(
        "Indexed chunks",
        f"{index.ntotal:,}"
    )

    departments = sorted(
        {
            item.get(
                "department",
                "General"
            )
            for item in metadata
        }
    )

    st.markdown(
        '<div class="small-label">Categories</div>',
        unsafe_allow_html=True
    )

    for department in departments:

        st.caption(
            f"• {department}"
        )

    st.divider()

    st.caption(
        "Answers are grounded in the "
        "indexed enterprise knowledge base."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">

    <h1>Enterprise Knowledge Assistant</h1>

    <p>
        Search company policies, procedures,
        compliance documentation and internal
        knowledge with grounded AI.
    </p>

    <div class="status">
        <span class="dot"></span>
        Knowledge base online
    </div>

</div>
""",
    unsafe_allow_html=True
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
                "View retrieved sources"
            ):

                for source in message["sources"]:

                    excerpt = (
                        source["text"]
                        .replace(
                            "\n",
                            " "
                        )
                    )

                    if len(excerpt) > 300:

                        excerpt = (
                            excerpt[:300]
                            + "..."
                        )

                    st.markdown(
                        f"""
<div class="source-card">

    <div class="source-file">
        📄 {source['source_file']}
    </div>

    <div class="source-meta">
        {source['department']}
        · Page {source['page']}
        · Version {source.get('version', 'Unknown')}
        · Relevance {source['score']:.3f}
    </div>

    <div class="source-excerpt">
        {excerpt}
    </div>

</div>
""",
                        unsafe_allow_html=True
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
                "View retrieved sources"
            ):

                for source in results:

                    excerpt = (
                        source["text"]
                        .replace(
                            "\n",
                            " "
                        )
                    )

                    if len(excerpt) > 300:

                        excerpt = (
                            excerpt[:300]
                            + "..."
                        )

                    st.markdown(
                        f"""
<div class="source-card">

    <div class="source-file">
        📄 {source['source_file']}
    </div>

    <div class="source-meta">
        {source['department']}
        · Page {source['page']}
        · Version {source.get('version', 'Unknown')}
        · Relevance {source['score']:.3f}
    </div>

    <div class="source-excerpt">
        {excerpt}
    </div>

</div>
""",
                        unsafe_allow_html=True
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": results
        }
    )
