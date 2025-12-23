# Capstone Project

Scripts for ingesting PDF documents into Azure AI Search with Azure OpenAI embeddings, plus a small CLI helper to chat with an orchestrator agent through Azure AI Foundry.

## Prerequisites
- Python 3.10+ and `pip`
- Access to Azure AI Search, Azure Blob Storage, and Azure OpenAI resources
- An orchestrator agent set up in Azure AI Foundry (for `agent_trigger.py`)

## Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your values:
```powershell
copy .env.example .env
```

Required environment variables:
- `AZURE_SEARCH_ENDPOINT` / `AZURE_SEARCH_KEY` / `AZURE_SEARCH_INDEX` / `AZURE_EMBED_DIM`
- `AZURE_BLOB_CONNECTION_STRING` / `AZURE_BLOB_CONTAINER`
- `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_KEY` / `AZURE_OPENAI_EMBEDDING_MODEL`
- `PROJECT_ENDPOINT` / `ORCHESTRATOR_AGENT_ID` (for the agent helper)
- Optional: `PROJECT_CONNECTION_STRING`, `AZURE_SEARCH_API_VERSION`

## Ingest PDFs to Azure AI Search
`azure_ingest.py` downloads PDFs from the configured blob container, chunks text (~1500 chars with 200-char overlap), embeds with Azure OpenAI, and uploads to the search index (HNSW vector search + semantic config).
```powershell
.venv\Scripts\activate
python azure_ingest.py
```
Notes:
- Only `.pdf` blobs are processed; empty PDFs are skipped.
- The script auto-detects the first working Search API version unless `AZURE_SEARCH_API_VERSION` is set.

## Chat with the orchestrator agent
`agent_trigger.py` opens an interactive loop that posts user messages to a single thread and prints the latest agent reply.
```powershell
.venv\Scripts\activate
python agent_trigger.py
```
If `PROJECT_CONNECTION_STRING` is set it is preferred; otherwise `PROJECT_ENDPOINT` + `DefaultAzureCredential` are used.

## Troubleshooting
- Authentication errors from Search: verify `AZURE_SEARCH_KEY`.
- API version errors: set `AZURE_SEARCH_API_VERSION` explicitly.
- Embedding size mismatches: ensure `AZURE_EMBED_DIM` matches the embedding model’s output dimension.
