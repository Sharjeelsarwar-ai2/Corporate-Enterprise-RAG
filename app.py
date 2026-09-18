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

EMBEDDING_MODEL = (
    "BAAI/bge-small-en-v1.5"
)

GROQ_MODEL = (
    "openai/gpt-oss-120b"
)

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
# PREMIUM UI
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
        radial-gradient(
            circle at 10% 0%,
            rgba(99,102,241,.09),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(14,165,233,.08),
            transparent 25%
        ),
        #f7f9fc;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
        margin-bottom: 8px;
    }

    .brand-mark {
        width: 42px;
        height: 42px;
        border-radius: 13px;

        display: flex;
        align-items: center;
        justify-content: center;

        background:
        linear-gradient(
            135deg,
            #111827,
            #334155
        );

        color: white;
        font-size: 20px;

        box-shadow:
        0 10px 30px
        rgba(15,23,42,.16);
    }

    .brand-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #111827;
        letter-spacing: -.02em;
    }

    .hero {
        padding: 32px;

        margin:
        18px 0 22px;

        border:
        1px solid
        rgba(148,163,184,.22);

        border-radius: 24px;

        background:
        rgba(255,255,255,.78);

        backdrop-filter:
        blur(14px);

        box-shadow:
        0 18px 50px
        rgba(15,23,42,.07);
    }

    .hero h1 {
        margin: 0;

        font-size: 2.35rem;

        letter-spacing: -.045em;

        color: #0f172a;
    }

    .hero p {
        margin-top: 9px;

        color: #64748b;

        font-size: 1rem;

        max-width: 720px;
    }

    .status {
        display: inline-flex;

        align-items: center;

        gap: 7px;

        margin-top: 17px;

        padding:
        7px 12px;

        border-radius: 999px;

        background: #ecfdf5;

        color: #047857;

        font-size: .82rem;

        font-weight: 600;
    }

    .dot {
        width: 7px;
        height: 7px;

        border-radius: 50%;

        background: #10b981;
    }

    .source-card {
        padding: 15px 17px;

        border-radius: 16px;

        border:
        1px solid #e5e7eb;

        background: white;

        margin: 9px 0;

        box-shadow:
        0 5px 18px
        rgba(15,23,42,.035);
    }

    .source-file {
        font-weight: 700;

        color: #111827;
    }

    .source-meta {
        color: #64748b;

        font-size: .82rem;

        margin-top: 5px;
    }

    .source-excerpt {
        color: #475569;

        font-size: .84rem;

        line-height: 1.5;

        margin-top: 8px;
    }

    .small-label {
        color: #64748b;

        font-size: .76rem;

        text-transform: uppercase;

        letter-spacing: .08em;

        font-weight: 700;
    }

    section[data-testid="stSidebar"] {
        border-right:
        1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] > div {
        background:
        rgba(255,255,255,.88);
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

    if not os.path.exists(
        INDEX_FILE
    ):
        raise FileNotFoundError(
            "FAISS index not found."
        )

    if not os.path.exists(
        METADATA_FILE
    ):
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

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL
        )
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

    query_embedding = (
        embedding_model.encode(
            [question],
            normalize_embeddings=True
        )
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = (
        index.search(
            query_embedding,
            top_k
        )
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

        item["score"] = float(
            score
        )

        if item["score"] >= MIN_RELEVANCE:

            results.append(
                item
            )

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

    return "\n".join(
        context
    )


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
                    "content":
                        system_prompt
                },
                {
                    "role": "user",
                    "content":
                        user_prompt
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

    st.markdown(
        """
        <div class="brand">

            <div class="brand-mark">
                ✦
            </div>

            <div class="brand-name">
                Knowledge Intelligence
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="small-label">'
        'Knowledge Base'
        '</div>',
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
        '<div class="small-label">'
        'Categories'
        '</div>',
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

        <h1>
            Enterprise Knowledge Assistant
        </h1>

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


for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] ==
            "assistant"
            and
            message.get("sources")
        ):

            with st.expander(
                "View retrieved sources"
            ):

                for source in (
                    message["sources"]
                ):

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
                                📄
                                {source['source_file']}
                            </div>

                            <div class="source-meta">
                                {source['department']}
                                · Page {source['page']}
                                · Version
                                {source.get('version', 'Unknown')}
                                · Relevance
                                {source['score']:.3f}
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

            results = (
                retrieve_documents(
                    question,
                    index,
                    metadata,
                    embedding_model
                )
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

                answer = (
                    generate_answer(
                        question,
                        results,
                        groq_client
                    )
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
                                📄
                                {source['source_file']}
                            </div>

                            <div class="source-meta">
                                {source['department']}
                                · Page {source['page']}
                                · Version
                                {source.get('version', 'Unknown')}
                                · Relevance
                                {source['score']:.3f}
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
