# Enterprise Policy & Compliance RAG Copilot

A generic enterprise RAG knowledge assistant using:

- Google Drive public/shared folder as the knowledge source
- Python + Streamlit
- Sentence Transformers for embeddings
- FAISS for vector search
- Groq `openai/gpt-oss-120b` for grounded answer generation
- Metadata for department, source file, page, version and effective date

## Architecture

Google Drive -> ingestion -> chunks + metadata -> embeddings -> FAISS -> Streamlit -> retrieval -> Groq -> answer + sources

## Important

Embeddings are generated once by `ingest.py`. The deployed Streamlit app loads the existing FAISS index and metadata; it does not regenerate the entire knowledge base on every startup.

## Local/Colab ingestion

1. Install packages from requirements.
2. Put your public Google Drive folder URL into `GOOGLE_DRIVE_FOLDER_URL` in `ingest.py`.
3. Run `python ingest.py`.
4. Commit `faiss_index/index.faiss` and `faiss_index/metadata.pkl` to your private GitHub repository.
5. Deploy the repository to Streamlit Cloud.
6. Add `GROQ_API_KEY` under Streamlit Settings -> Secrets.

## Security note

Do not put real confidential company documents in a public Google Drive folder or public GitHub repository. For real client deployments, use authenticated/private storage and appropriate access controls.
