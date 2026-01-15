from __future__ import annotations

import argparse
import pathway as pw

from pathway.stdlib.indexing.nearest_neighbors import UsearchKnnFactory
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
from pathway.xpacks.llm.servers import DocumentStoreServer
from pathway.xpacks.llm.splitters import RecursiveSplitter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--books_dir", type=str, default="data/books")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)

    # Chunking
    parser.add_argument("--chunk_tokens", type=int, default=350)
    parser.add_argument("--chunk_overlap", type=int, default=60)

    # Embedding
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embed_device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--embed_batch_size", type=int, default=64)

    args = parser.parse_args()

    # ✅ Correct format is "plaintext" (not "text")
    docs = pw.io.fs.read(args.books_dir, format="plaintext_by_file", with_metadata=True)

    # Token-based fixed chunks + overlap
    splitter = RecursiveSplitter(
        chunk_size=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        encoding_name="cl100k_base",
    )

    # Embedder (lightweight; GPU optional)
    embedder = SentenceTransformerEmbedder(
        model=args.embed_model,
        device=args.embed_device,
        batch_size=args.embed_batch_size,
    )

    retriever_factory = UsearchKnnFactory(embedder=embedder)

    store = DocumentStore(
        docs=docs,
        splitter=splitter,
        retriever_factory=retriever_factory,
    )

    server = DocumentStoreServer(
        host=args.host,
        port=args.port,
        document_store=store,
    )

    # Keep server running in foreground (CLI-friendly)
    server.run(threaded=False, with_cache=False)


if __name__ == "__main__":
    main()
