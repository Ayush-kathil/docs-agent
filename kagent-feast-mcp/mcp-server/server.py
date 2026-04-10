from fastmcp import FastMCP
import logging
import threading
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

MILVUS_URI = "http://milvus.<YOUR_NAMESPACE>.svc.cluster.local:19530"
MILVUS_USER = "root"
MILVUS_PASSWORD = "Milvus"
COLLECTION_NAME = "kubeflow_docs_docs_rag"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
PORT = 8000

mcp = FastMCP("Kubeflow Docs MCP Server")
logger = logging.getLogger(__name__)

model: SentenceTransformer = None
client: MilvusClient = None
_initialized = False
_init_lock = threading.Lock()


def _init():
    """Initialize shared model/client exactly once.

    Synchronization strategy:
    - Fast path: return immediately after successful initialization.
    - Slow path: take a process-local lock and re-check state (double-checked locking).

    This guarantees that concurrent callers block until initialization completes,
    and all callers observe the same initialized instances.
    """
    global model, client, _initialized

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        logger.info("Initializing shared MCP resources")

        # Build local instances first, then publish atomically under the lock.
        local_model = SentenceTransformer(EMBEDDING_MODEL)
        local_client = MilvusClient(uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD)

        model = local_model
        client = local_client
        _initialized = True

        logger.info("Shared MCP resources initialized")


@mcp.tool()
def search_kubeflow_docs(query: str, top_k: int = 5) -> str:
    """Search Kubeflow documentation using semantic similarity.

    Args:
        query: The search query about Kubeflow.
        top_k: Number of results to return (default 5).

    Returns:
        Formatted search results with content and citation URLs.
    """
    _init()

    embedding = model.encode(query).tolist()

    hits = client.search(
        collection_name=COLLECTION_NAME,
        data=[embedding],
        limit=top_k,
        output_fields=["content_text", "citation_url", "file_path"],
    )[0]

    if not hits:
        return "No results found for your query."

    results = []
    for i, hit in enumerate(hits, 1):
        entity = hit["entity"]
        entry = f"### Result {i} (score: {hit['distance']:.4f})"
        entry += f"\n**Source:** {entity.get('citation_url', '')}"
        entry += f"\n**File:** {entity.get('file_path', '')}"
        entry += f"\n\n{entity.get('content_text', '')}\n"
        results.append(entry)

    return "\n---\n".join(results)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORT)
