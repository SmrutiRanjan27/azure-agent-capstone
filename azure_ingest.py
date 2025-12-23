import logging
import os
import uuid
from io import BytesIO
from typing import Iterable, List, Sequence

import requests
from azure.storage.blob import BlobClient, BlobServiceClient
from dotenv import load_dotenv
from openai import AzureOpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv(override=True)


# Use a widely available API version for index management. If your service supports newer versions,
# you can bump this, but 2023-11-01-Preview is broadly available across regions.
DEFAULT_API_VERSIONS = [
    "2024-10-01-Preview",
    "2024-05-01-Preview",
    "2023-11-01",
    "2023-11-01-Preview",
]
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX", "team2-legal-doc-inde-2")
EMBEDDING_DIM = int(os.getenv("AZURE_EMBED_DIM", "1536"))

# Matches the schema provided by the user.
INDEX_DEFINITION = {
    "name": INDEX_NAME,
    "fields": [
        {
            "name": "id",
            "type": "Edm.String",
            "key": True,
            "filterable": False,
            "searchable": False,
        },
        {
            "name": "document_id",
            "type": "Edm.String",
            "filterable": True,
            "searchable": False,
        },
        {
            "name": "chunk_id",
            "type": "Edm.String",
            "filterable": True,
            "searchable": False,
        },
        {
            "name": "content",
            "type": "Edm.String",
            "searchable": True,
            "analyzer": "en.microsoft",
        },
        {
            "name": "embedding",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "dimensions": EMBEDDING_DIM,
            "vectorSearchProfile": "vector-profile",
        },
        {
            "name": "source",
            "type": "Edm.String",
            "filterable": True,
            "searchable": False,
        },
    ],
    "vectorSearch": {
        "algorithms": [
            {
                "name": "hnsw-algorithm",
                "kind": "hnsw",
                "hnswParameters": {
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine",
                },
            }
        ],
        "profiles": [
            {
                "name": "vector-profile",
                "algorithm": "hnsw-algorithm",
            }
        ],
    },
    "semantic": {
        "configurations": [
            {
                "name": "semantic-config",
                "prioritizedFields": {
                    "prioritizedContentFields": [{"fieldName": "content"}],
                },
            }
        ]
    },
}


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    clean_text = text.replace("\r", "\n").replace("\t", " ").strip()
    chunks: List[str] = []
    start = 0
    while start < len(clean_text):
        end = start + chunk_size
        chunks.append(clean_text[start:end])
        start = end - overlap
    return chunks


def extract_pdf_text(blob: BlobClient) -> str:
    stream = blob.download_blob()
    reader = PdfReader(BytesIO(stream.readall()))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return "\n".join(pages)


def ensure_index(search_endpoint: str, search_key: str, api_version: str) -> None:
    url = f"{search_endpoint}/indexes('{INDEX_NAME}')?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": search_key,
    }
    response = requests.put(url, headers=headers, json=INDEX_DEFINITION, timeout=30)
    if not response.ok:
        raise RuntimeError(f"Index create/update failed: {response.status_code} {response.text}")
    logging.info("Index %s created or updated", INDEX_NAME)


def embed_chunks(client: AzureOpenAI, model: str, chunks: Sequence[str]) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for chunk in chunks:
        result = client.embeddings.create(model=model, input=chunk)
        embeddings.append(result.data[0].embedding)
    return embeddings


def upload_documents(
    search_endpoint: str, search_key: str, docs: Sequence[dict], api_version: str
) -> None:
    if not docs:
        logging.info("No documents to upload.")
        return
    url = f"{search_endpoint}/indexes('{INDEX_NAME}')/docs/index?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": search_key}
    batch_size = 32
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        payload = {"value": [{"@search.action": "upload", **doc} for doc in batch]}
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if not response.ok:
            raise RuntimeError(f"Document upload failed: {response.status_code} {response.text}")
        logging.info("Uploaded batch %s-%s", i + 1, i + len(batch))


def collect_documents(
    container: str,
    blob_service: BlobServiceClient,
    embed_client: AzureOpenAI,
    embedding_model: str,
) -> List[dict]:
    documents: List[dict] = []
    container_client = blob_service.get_container_client(container)

    for blob_props in container_client.list_blobs():
        if not blob_props.name.lower().endswith(".pdf"):
            continue
        blob_client = container_client.get_blob_client(blob_props)
        text = extract_pdf_text(blob_client)
        if not text.strip():
            logging.warning("Skipped empty PDF: %s", blob_props.name)
            continue

        doc_id = os.path.splitext(os.path.basename(blob_props.name))[0] or str(uuid.uuid4())
        chunks = chunk_text(text)
        embeddings = embed_chunks(embed_client, embedding_model, chunks)

        for idx, chunk in enumerate(chunks):
            documents.append(
                {
                    "id": f"{doc_id}-{idx}",
                    "document_id": doc_id,
                    "chunk_id": str(idx),
                    "content": chunk,
                    "embedding": embeddings[idx],
                    "source": blob_props.name,
                }
            )

    return documents


def resolve_api_version(search_endpoint: str, search_key: str) -> str:
    """Pick a working API version by probing the service, or use AZURE_SEARCH_API_VERSION if set."""
    override = os.getenv("AZURE_SEARCH_API_VERSION")
    if override:
        logging.info("Using API version from AZURE_SEARCH_API_VERSION: %s", override)
        return override

    headers = {"api-key": search_key}
    for version in DEFAULT_API_VERSIONS:
        url = f"{search_endpoint}/indexes?api-version={version}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
        except requests.RequestException as exc:
            logging.warning("API version probe failed for %s: %s", version, exc)
            continue
        if resp.status_code == 200:
            logging.info("Detected supported API version: %s", version)
            return version
        if resp.status_code == 401:
            raise RuntimeError("Unauthorized: check AZURE_SEARCH_KEY.")
        if resp.status_code == 403:
            raise RuntimeError("Forbidden: key may lack permissions.")
        logging.info("API version %s not accepted (status %s).", version, resp.status_code)

    raise RuntimeError(
        "No supported API version found. Set AZURE_SEARCH_API_VERSION explicitly "
        f"or verify the service endpoint. Tried: {', '.join(DEFAULT_API_VERSIONS)}."
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load environment variables from .env if present (development convenience).
    load_dotenv()

    search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/")
    search_key = os.environ["AZURE_SEARCH_KEY"]
    blob_connection = os.environ["AZURE_BLOB_CONNECTION_STRING"]
    blob_container = os.environ["AZURE_BLOB_CONTAINER"]

    openai_client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version="2024-02-15-preview",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    embedding_model = os.environ["AZURE_OPENAI_EMBEDDING_MODEL"]

    api_version = resolve_api_version(search_endpoint, search_key)

    ensure_index(search_endpoint, search_key, api_version)

    blob_service = BlobServiceClient.from_connection_string(blob_connection)
    docs = collect_documents(blob_container, blob_service, openai_client, embedding_model)
    upload_documents(search_endpoint, search_key, docs, api_version)
    logging.info("Ingestion complete: %s documents uploaded", len(docs))


if __name__ == "__main__":
    main()
